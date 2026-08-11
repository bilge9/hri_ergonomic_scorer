#!/usr/bin/env python3
"""
annotation_recorder.py

Step 1 of the accuracy-validation protocol (see validation/README.md).

Replays a bag through the scorer and records, for every assessed frame:
  - the system's REBA region scores and final score,
  - how complete the skeleton was and which items were actually observed,
  - the matching RAW camera image as a PNG.

The image is deliberately the raw camera frame, not the skeleton overlay:
annotators must judge the posture from what a human would see. Showing them
the tracker's own skeleton would let the system's errors leak into the
reference and quietly inflate the agreement figure.

Usage:
    ros2 run hri_ergonomic_scorer annotation_recorder \\
        --ros-args -p image_topic:=/zed/zed_node/left/image_rect_color \\
                   -p output_dir:=~/reba_validation/run1

    # in another terminal
    ros2 bag play <bag>

Then feed output_dir to validation/select_frames.py.
"""

import csv
import os
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image
from hri_ergonomic_msgs.msg import RebaAssessmentList

SCORE_NOT_ASSESSED = 255
RISK_UNKNOWN = 255

CSV_FIELDS = [
    "frame_id", "stamp_sec", "stamp_nanosec", "body_key", "image",
    "completeness", "score_a", "score_b", "score_c", "risk_level",
    "neck_score", "trunk_score", "leg_score",
    "upper_arm_score", "lower_arm_score", "wrist_score",
    "neck_assessed", "trunk_assessed", "legs_assessed",
    "upper_arm_assessed", "lower_arm_assessed",
    "score_is_lower_bound", "deba_score", "deba_valid",
]


def _stamp_to_float(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


class AnnotationRecorder(Node):
    def __init__(self):
        super().__init__('annotation_recorder')

        self.declare_parameter('image_topic', '')
        self.declare_parameter('output_dir', './reba_validation')
        self.declare_parameter('use_sensor_qos', False)
        # Images and assessments arrive on separate topics with independent
        # timing, so each assessment takes the closest image within this many
        # seconds. Anything further apart is a different posture.
        self.declare_parameter('max_stamp_skew', 0.10)
        self.declare_parameter('image_buffer', 60)

        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.output_dir = os.path.expanduser(
            self.get_parameter('output_dir').get_parameter_value().string_value)
        self.max_skew = self.get_parameter('max_stamp_skew').get_parameter_value().double_value
        buf_len = self.get_parameter('image_buffer').get_parameter_value().integer_value
        use_sensor_qos = self.get_parameter('use_sensor_qos').get_parameter_value().bool_value

        self.images_dir = os.path.join(self.output_dir, 'images')
        os.makedirs(self.images_dir, exist_ok=True)
        self.csv_path = os.path.join(self.output_dir, 'frames.csv')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT if use_sensor_qos else ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.bridge = None
        self.image_buffer = deque(maxlen=buf_len)
        self.have_images = bool(image_topic)

        if self.have_images:
            try:
                from cv_bridge import CvBridge
                self.bridge = CvBridge()
            except ImportError as exc:
                self.get_logger().error(
                    f"cv_bridge unavailable, recording scores WITHOUT images: {exc}")
                self.have_images = False

        if self.have_images:
            self.create_subscription(Image, image_topic, self.image_callback, qos)
            self.get_logger().info(f"Recording images from {image_topic}")
        else:
            self.get_logger().warn(
                "No image_topic given - scores will be recorded but there will "
                "be nothing for an annotator to look at.")

        self.create_subscription(
            RebaAssessmentList, '/humans/bodies/ergonomics/reba', self.reba_callback, qos)

        self.csv_file = open(self.csv_path, 'w', newline='', encoding='utf-8')
        self.writer = csv.DictWriter(self.csv_file, fieldnames=CSV_FIELDS)
        self.writer.writeheader()

        self.frame_id = 0
        self.written = 0
        self.no_image = 0
        self.get_logger().info(f"Recording to {self.output_dir}")

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().error(f"Image conversion failed: {exc}")
            return
        self.image_buffer.append((_stamp_to_float(msg.header.stamp), frame))

    def _closest_image(self, target):
        """Nearest buffered frame within max_stamp_skew, or None."""
        best, best_dt = None, None
        for stamp, frame in self.image_buffer:
            dt = abs(stamp - target)
            if best_dt is None or dt < best_dt:
                best, best_dt = frame, dt
        if best_dt is not None and best_dt <= self.max_skew:
            return best
        return None

    def reba_callback(self, msg):
        target = _stamp_to_float(msg.header.stamp)

        for assessment in msg.assessments:
            if not assessment.key:
                continue

            image_name = ''
            if self.have_images:
                frame = self._closest_image(target)
                if frame is None:
                    self.no_image += 1
                else:
                    import cv2
                    image_name = f"frame_{self.frame_id:06d}.png"
                    cv2.imwrite(os.path.join(self.images_dir, image_name), frame)

            self.writer.writerow({
                "frame_id": self.frame_id,
                "stamp_sec": msg.header.stamp.sec,
                "stamp_nanosec": msg.header.stamp.nanosec,
                "body_key": assessment.key,
                "image": image_name,
                "completeness": round(float(assessment.completeness), 4),
                "score_a": int(assessment.score_a),
                "score_b": int(assessment.score_b),
                "score_c": int(assessment.score_c),
                "risk_level": int(assessment.risk_level),
                "neck_score": int(assessment.neck_score),
                "trunk_score": int(assessment.trunk_score),
                "leg_score": int(assessment.leg_score),
                "upper_arm_score": int(assessment.upper_arm_score),
                "lower_arm_score": int(assessment.lower_arm_score),
                "wrist_score": int(assessment.wrist_score),
                "neck_assessed": int(assessment.neck_assessed),
                "trunk_assessed": int(assessment.trunk_assessed),
                "legs_assessed": int(assessment.legs_assessed),
                "upper_arm_assessed": int(assessment.upper_arm_assessed),
                "lower_arm_assessed": int(assessment.lower_arm_assessed),
                "score_is_lower_bound": int(assessment.score_is_lower_bound),
                "deba_score": round(float(assessment.deba_score), 4) if assessment.deba_valid else '',
                "deba_valid": int(assessment.deba_valid),
            })
            self.frame_id += 1
            self.written += 1

        if self.written and self.written % 100 == 0:
            self.csv_file.flush()
            self.get_logger().info(
                f"Recorded {self.written} assessments "
                f"({self.no_image} without a matching image)")

    def destroy_node(self):
        try:
            self.csv_file.flush()
            self.csv_file.close()
            self.get_logger().info(
                f"Wrote {self.written} rows to {self.csv_path} "
                f"({self.no_image} without a matching image)")
        except Exception:
            pass
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = AnnotationRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n[INFO] Recording stopped by user.")
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
