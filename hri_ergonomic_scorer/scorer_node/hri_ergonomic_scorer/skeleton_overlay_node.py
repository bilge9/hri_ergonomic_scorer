#!/usr/bin/env python3
"""
skeleton_overlay_node.py

Subscribes to Skeleton3DList (Vulcanexus COCO-18 format) and optionally a
camera image topic. Projects/draws the 2D skeleton with bone connections and
publishes the result as sensor_msgs/Image so it can be served by
web_video_server and embedded in Grafana via an <img> HTML panel.
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

# --- COCO-18 joint order ---
JOINT_NAMES = [
    "Nose", "Neck", "R_Shoulder", "R_Elbow", "R_Wrist",
    "L_Shoulder", "L_Elbow", "L_Wrist", "R_Hip", "R_Knee",
    "R_Ankle", "L_Hip", "L_Knee", "L_Ankle", "L_Eye", "R_Eye",
    "L_Ear", "R_Ear"
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

# Per-REBA-region bone grouping
NECK_BONES = {(0, 1)}
TRUNK_BONES = {(1, 2), (1, 5), (1, 8), (1, 11), (8, 11)}
LEG_BONES = {(8, 9), (9, 10), (11, 12), (12, 13)}
UPPER_ARM_BONES = {(2, 3), (5, 6)}
LOWER_ARM_BONES = {(3, 4), (6, 7)}
WRIST_JOINTS = {4, 7}

NEUTRAL_COLOR = (200, 200, 200)
JOINT_COLOR = (0, 0, 255)
CANVAS_SIZE = (720, 1280)

# REBA Table C Matrix (Used to reverse-engineer Activity Score)
TABLE_C = np.array([
    [1, 1, 1, 2, 3, 3, 4, 5, 6, 7, 7, 7],
    [1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 7, 8],
    [2, 3, 3, 3, 4, 5, 6, 7, 7, 8, 8, 8],
    [3, 4, 4, 4, 5, 6, 7, 8, 8, 9, 9, 9],
    [4, 4, 4, 5, 6, 7, 8, 8, 9, 9, 9, 9],
    [6, 6, 6, 7, 8, 8, 9, 9, 10, 10, 10, 10],
    [7, 7, 7, 8, 9, 9, 9, 10, 10, 11, 11, 11],
    [8, 8, 8, 9, 10, 10, 10, 10, 10, 11, 11, 11],
    [9, 9, 9, 10, 10, 10, 11, 11, 11, 12, 12, 12],
    [10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 12],
    [11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12],
    [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12],
])

REGION_THRESHOLDS = {
    "neck": (1, 2),        
    "trunk": (2, 3),       
    "leg": (1, 2),         
    "upper_arm": (2, 4),   
    "lower_arm": (1, 1),   
    "wrist": (1, 2)        
}

def color_for_region(score, region):
    if score is None or score == 0:
        return NEUTRAL_COLOR
        
    max_green, max_yellow = REGION_THRESHOLDS.get(region, (2, 4))
    
    if score <= max_green:
        return (0, 200, 0)      
    elif score <= max_yellow:
        return (0, 220, 255)    
    else:
        return (0, 0, 255)      

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
        self.latest_assessments = {}

        self.declare_parameter('camera_image_topic', '')
        self.declare_parameter('camera_info_topic', '')
        self.declare_parameter('output_topic', '/humans/bodies/skel3D/overlay')
        self.declare_parameter('projection_mode', 'optical')

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

    def reba_callback(self, msg):
        for assessment in msg.assessments:
            if assessment.key:
                self.latest_assessments[assessment.key] = assessment

    def camera_info_callback(self, msg):
        self.camera_matrix = np.array(msg.k).reshape(3, 3)

    def image_callback(self, msg):
        self.latest_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def project_point(self, pt):
        fx, fy = self.camera_matrix[0, 0], self.camera_matrix[1, 1]
        cx, cy = self.camera_matrix[0, 2], self.camera_matrix[1, 2]
        x, y, z = pt
        if z <= 0:
            return None
        u = int((fx * x / z) + cx)
        v = int((fy * y / z) + cy)
        return (u, v)

    def canvas_point(self, pt, scale=150):
        h, w = CANVAS_SIZE
        u = int(w / 2 - pt[1] * scale)
        v = int(h / 2 - pt[2] * scale)  
        return (u, v)

    def skeleton_callback(self, msg):
        if not msg.skeletons:
            return

        body = msg.skeletons[0]

        if self.latest_frame is not None and self.camera_matrix is not None:
            frame = self.latest_frame.copy()
            project_fn = self.project_point
        else:
            frame = np.zeros((*CANVAS_SIZE, 3), dtype=np.uint8)
            project_fn = self.canvas_point

        neck = body.skeleton[1]
        center_x = neck.x if not np.isnan(neck.x) else 0.0
        center_y = neck.y if not np.isnan(neck.y) else 0.0
        center_z = neck.z if not np.isnan(neck.z) else 0.0

        points_2d = [None] * len(JOINT_NAMES)
        for i, pt_obj in enumerate(body.skeleton[:len(JOINT_NAMES)]):
            pt = [pt_obj.x, pt_obj.y, pt_obj.z]
            if is_valid(pt):
                if self.latest_frame is None or self.camera_matrix is None:
                    dx = pt[0] - center_x
                    dy = pt[1] - center_y
                    dz = pt[2] - center_z
                    
                    h, w = CANVAS_SIZE
                    scale = 150
                    
                    # YENİ EKLENEN: Dinamik Projeksiyon Seçimi
                    mode = self.get_parameter('projection_mode').get_parameter_value().string_value
                    
                    if mode == 'isometric':
                        # Eski sistem: Z ekseninin yukarıda olduğu açılı görünüm
                        u = int(w / 2 + (dx - dy) * 0.707 * scale)
                        v = int(h / 2 - dz * scale + (dx + dy) * 0.35 * scale)
                    else:
                        # Optik sistem: Standart kamera düzlemi (X yatay, Y dikey)
                        u = int(w / 2 + dx * scale)
                        v = int(h / 2 + dy * scale)
                        
                    points_2d[i] = (u, v)
                else:
                    points_2d[i] = project_fn(pt)

        assessment = self.latest_assessments.get(body.key)
        
        neck_s = assessment.neck_score if assessment else None
        trunk_s = assessment.trunk_score if assessment else None
        leg_s = assessment.leg_score if assessment else None
        upper_arm_s = assessment.upper_arm_score if assessment else None
        lower_arm_s = assessment.lower_arm_score if assessment else None
        wrist_s = assessment.wrist_score if assessment else None

        color_neck = color_for_region(neck_s, "neck")
        color_trunk = color_for_region(trunk_s, "trunk")
        color_leg = color_for_region(leg_s, "leg")
        color_upper_arm = color_for_region(upper_arm_s, "upper_arm")
        color_lower_arm = color_for_region(lower_arm_s, "lower_arm")
        color_wrist = color_for_region(wrist_s, "wrist")

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

        for i, pt in enumerate(points_2d):
            if pt is None:
                continue
            joint_color = color_wrist if i in WRIST_JOINTS else JOINT_COLOR
            cv2.circle(frame, pt, 5, joint_color, -1)

        # ---------------------------------------------------------
        # NEW DASHBOARD TEXT OVERLAY WITH SCORE A, B, AND ACTIVITY
        # ---------------------------------------------------------
        if assessment is not None:
            # Safely clamp indices to avoid out-of-bounds error
            safe_a = min(12, max(1, assessment.score_a))
            safe_b = min(12, max(1, assessment.score_b))
            
            # Reverse engineer the activity score
            raw_table_c_score = TABLE_C[safe_a - 1][safe_b - 1]
            activity_score = assessment.score_c - raw_table_c_score

            # Main Header
            cv2.putText(frame, f"FINAL REBA: {assessment.score_c}/15 | Risk: {assessment.risk_level}", 
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Breakdown Sub-Headers
            cv2.putText(frame, f"Score A (Body + Load) : {assessment.score_a}", 
                        (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Score B (Arm + Coupl) : {assessment.score_b}", 
                        (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
            
            # Activity Penalty with dynamic color (Green if 0, Orange if > 0)
            act_color = (0, 200, 0) if activity_score == 0 else (0, 165, 255)
            cv2.putText(frame, f"Activity Penalty      : +{activity_score}", 
                        (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.55, act_color, 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "REBA: waiting for assessment...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # Shifted Y coordinate down to make room for the new sub-headers
        region_lines = [
            ("Neck", neck_s, color_neck),
            ("Trunk", trunk_s, color_trunk),
            ("Legs", leg_s, color_leg),
            ("Upper Arm", upper_arm_s, color_upper_arm),
            ("Lower Arm", lower_arm_s, color_lower_arm),
            ("Wrist", wrist_s, color_wrist),
        ]
        
        y = 175 # Lowered starting position
        for name, score, color in region_lines:
            cv2.putText(frame, f"{name:<10}: {score}", (20, y),
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