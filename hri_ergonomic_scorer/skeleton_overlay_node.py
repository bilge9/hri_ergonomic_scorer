#!/usr/bin/env python3
"""
skeleton_overlay_node.py

Subscribes to Skeleton3DList (Vulcanexus COCO-18 format) and optionally a
camera image topic. Projects/draws the 2D skeleton with bone connections and
publishes the result as sensor_msgs/Image so it can be served by
web_video_server and embedded in Grafana via an <img> HTML panel.

Two modes:
  - OVERLAY mode: if a camera image + camera_info topic is available, joints
    are projected into pixel space using the pinhole camera model and drawn
    on top of the real video frame.
  - CANVAS mode (fallback): if no camera image is available, draws the
    skeleton on a blank canvas using only the relative X/Y of the 3D points
    (useful for a quick demo without needing camera intrinsics).
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import numpy as np
import cv2
from cv_bridge import CvBridge

from sensor_msgs.msg import Image, CameraInfo
from hri_msgs.msg import Skeleton3DList
from hri_ergonomic_msgs.msg import RebaAssessmentList

# --- COCO-18 joint order (same as ergonomic_scorer_node.py) ---
JOINT_NAMES = [
    "Nose", "Neck", "R_Shoulder", "R_Elbow", "R_Wrist",
    "L_Shoulder", "L_Elbow", "L_Wrist", "R_Hip", "R_Knee",
    "R_Ankle", "L_Hip", "L_Knee", "L_Ankle", "R_Eye", "L_Eye",
    "R_Ear", "L_Ear"
]

# Bone connections as (joint_index_a, joint_index_b) pairs
BONES = [
    (0, 1), (1, 2), (1, 5),           # nose-neck, neck-shoulders
    (2, 3), (3, 4),                   # right arm
    (5, 6), (6, 7),                   # left arm
    (1, 8), (1, 11), (8, 11),         # torso
    (8, 9), (9, 10),                  # right leg
    (11, 12), (12, 13),               # left leg
    (0, 14), (0, 15),                 # nose-eyes
    (14, 16), (15, 17),               # eyes-ears
]

# Per-REBA-region bone grouping, mirroring reba.py's own breakdown:
#   compute_score_a() -> partial_a = [neck_score, trunk_score, leg_score]
#   compute_score_b() -> partial_b = [upper_arm_score, lower_arm_score, wrist_score]
# Each bone is colored using its own region's score instead of one shared
# Group A / Group B color, so e.g. neck can be red while trunk stays green.
NECK_BONES = {(0, 1)}
TRUNK_BONES = {(1, 2), (1, 5), (1, 8), (1, 11), (8, 11)}
LEG_BONES = {(8, 9), (9, 10), (11, 12), (12, 13)}
UPPER_ARM_BONES = {(2, 3), (5, 6)}
LOWER_ARM_BONES = {(3, 4), (6, 7)}
# Wrist has no dedicated bone in the 18-joint set (wrist IS the endpoint,
# joints 4/7) - wrist_score is instead applied to the wrist joint dot color.
WRIST_JOINTS = {4, 7}

NEUTRAL_COLOR = (200, 200, 200)   # BGR - light gray, for face bones / no data
JOINT_COLOR = (0, 0, 255)         # BGR - red, default joint dot color
CANVAS_SIZE = (720, 1280)         # (height, width) fallback canvas

# REBA partial score -> color bands (approximate, for quick visual read).
# NOTE: neck/trunk/leg/upper_arm/lower_arm/wrist have different real maxima
# (e.g. neck maxes around 3, trunk around 5, upper_arm around 6) so a single
# green<=3/yellow<=6/red>6 banding is a rough approximation, not the official
# REBA per-region action level. Good enough for a quick visual read; if exact
# per-region thresholds matter later, we should derive them from table_a /
# table_b row/column ranges in reba.py instead.
def color_for_score(score):
    if score is None:
        return NEUTRAL_COLOR
    if score <= 2:
        return (0, 200, 0)      # green - low risk
    elif score <= 4:
        return (0, 220, 255)    # yellow/amber - medium risk
    else:
        return (0, 0, 255)      # red - high risk


def is_valid(pt):
    if any(np.isnan(v) for v in pt):
        return False
    if all(abs(v) < 1e-6 for v in pt):
        return False
    return True


class SkeletonOverlayNode(Node):
    def __init__(self):
        super().__init__('skeleton_overlay_node')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.bridge = CvBridge()
        self.camera_matrix = None
        self.latest_frame = None
        self.latest_assessments = {}  # key -> RebaAssessment, updated as they arrive

        self.declare_parameter('camera_image_topic', '')  # e.g. '/zed/left/image_rect_color'
        self.declare_parameter('camera_info_topic', '')    # e.g. '/zed/left/camera_info'
        self.declare_parameter('output_topic', '/humans/bodies/skel3D/overlay')

        image_topic = self.get_parameter('camera_image_topic').get_parameter_value().string_value
        info_topic = self.get_parameter('camera_info_topic').get_parameter_value().string_value
        output_topic = self.get_parameter('output_topic').get_parameter_value().string_value

        if image_topic:
            self.create_subscription(Image, image_topic, self.image_callback, qos)
            self.get_logger().info(f"OVERLAY mode: subscribing to {image_topic}")
        else:
            self.get_logger().info("CANVAS mode: no camera_image_topic given, drawing on blank canvas")

        if info_topic:
            self.create_subscription(CameraInfo, info_topic, self.camera_info_callback, qos)

        self.create_subscription(Skeleton3DList, '/humans/bodies/skel3D', self.skeleton_callback, qos)
        self.create_subscription(
            RebaAssessmentList, '/humans/bodies/ergonomics/reba', self.reba_callback, qos
        )
        self.publisher = self.create_publisher(Image, output_topic, qos)

        self.get_logger().info(f"Publishing overlay to {output_topic}")

    def reba_callback(self, msg):
        """Cache the latest REBA assessment per tracked body key."""
        for assessment in msg.assessments:
            if assessment.key:
                self.latest_assessments[assessment.key] = assessment

    def camera_info_callback(self, msg):
        self.camera_matrix = np.array(msg.k).reshape(3, 3)

    def image_callback(self, msg):
        self.latest_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def project_point(self, pt):
        """Project a 3D camera-frame point into 2D pixel coords using intrinsics."""
        fx, fy = self.camera_matrix[0, 0], self.camera_matrix[1, 1]
        cx, cy = self.camera_matrix[0, 2], self.camera_matrix[1, 2]
        x, y, z = pt
        if z <= 0:
            return None
        u = int((fx * x / z) + cx)
        v = int((fy * y / z) + cy)
        return (u, v)

    def canvas_point(self, pt, scale=150):
        """Fallback: map relative X/Y directly onto a blank canvas (no real camera).

        NOTE: ZED camera Y-axis already points downward (same convention as
        image/pixel coordinates), same issue solved in reba.py/pose_remap.py
        via FLIP_Y_SIGN. So we do NOT flip here - flipping would double-invert
        and render the skeleton upside down.
        """
        h, w = CANVAS_SIZE
        u = int(w / 2 + pt[0] * scale)
        v = int(h / 2 + pt[1] * scale)  # no flip: ZED Y is already down-positive
        return (u, v)

    def skeleton_callback(self, msg):
        if not msg.skeletons:
            return

        # Pick the first valid tracked body for the demo overlay
        body = msg.skeletons[0]

        if self.latest_frame is not None and self.camera_matrix is not None:
            frame = self.latest_frame.copy()
            project_fn = self.project_point
        else:
            frame = np.zeros((*CANVAS_SIZE, 3), dtype=np.uint8)
            project_fn = self.canvas_point

        points_2d = [None] * len(JOINT_NAMES)
        for i, pt_obj in enumerate(body.skeleton[:len(JOINT_NAMES)]):
            pt = [pt_obj.x, pt_obj.y, pt_obj.z]
            if is_valid(pt):
                points_2d[i] = project_fn(pt)

        # Look up this body's latest REBA scores (may be None if not yet computed)
        assessment = self.latest_assessments.get(body.key)
        neck_s = assessment.neck_score if assessment else None
        trunk_s = assessment.trunk_score if assessment else None
        leg_s = assessment.leg_score if assessment else None
        upper_arm_s = assessment.upper_arm_score if assessment else None
        lower_arm_s = assessment.lower_arm_score if assessment else None
        wrist_s = assessment.wrist_score if assessment else None

        color_neck = color_for_score(neck_s)
        color_trunk = color_for_score(trunk_s)
        color_leg = color_for_score(leg_s)
        color_upper_arm = color_for_score(upper_arm_s)
        color_lower_arm = color_for_score(lower_arm_s)
        color_wrist = color_for_score(wrist_s)

        # Draw bones, each colored by its own REBA region (not one shared
        # body/arm color) so e.g. neck can be red while trunk stays green.
        for a, b in BONES:
            if points_2d[a] is None or points_2d[b] is None:
                continue
            bone = (a, b)
            if bone in NECK_BONES:
                bone_color = color_neck
            elif bone in TRUNK_BONES:
                bone_color = color_trunk
            elif bone in LEG_BONES:
                bone_color = color_leg
            elif bone in UPPER_ARM_BONES:
                bone_color = color_upper_arm
            elif bone in LOWER_ARM_BONES:
                bone_color = color_lower_arm
            else:
                bone_color = NEUTRAL_COLOR
            cv2.line(frame, points_2d[a], points_2d[b], bone_color, 3)

        # Draw joints. Wrist joints (4=R_Wrist, 7=L_Wrist) get the wrist
        # region's color since wrist_score has no dedicated bone to color.
        for i, pt in enumerate(points_2d):
            if pt is None:
                continue
            joint_color = color_wrist if i in WRIST_JOINTS else JOINT_COLOR
            cv2.circle(frame, pt, 5, joint_color, -1)

        # Risk summary text overlay, top-left corner
        if assessment is not None:
            label = f"Score C: {assessment.score_c}  |  Risk Level: {assessment.risk_level}"
        else:
            label = "REBA: waiting for assessment..."
        cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (255, 255, 255), 2, cv2.LINE_AA)

        region_lines = [
            ("Neck", neck_s, color_neck),
            ("Trunk", trunk_s, color_trunk),
            ("Legs", leg_s, color_leg),
            ("Upper Arm", upper_arm_s, color_upper_arm),
            ("Lower Arm", lower_arm_s, color_lower_arm),
            ("Wrist", wrist_s, color_wrist),
        ]
        y = 70
        for name, score, color in region_lines:
            cv2.putText(frame, f"{name}: {score}", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
            y += 25

        out_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        out_msg.header = msg.header
        self.publisher.publish(out_msg)


def main(args=None):
    rclpy.init(args=args)
    node = SkeletonOverlayNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()