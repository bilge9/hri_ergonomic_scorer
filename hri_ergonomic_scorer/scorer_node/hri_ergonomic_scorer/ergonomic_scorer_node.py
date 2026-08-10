import rclpy
from rclpy.node import Node
import numpy as np
import math
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from ament_index_python.packages import get_package_share_directory
import time

import threading
import queue

from hri_msgs.msg import Skeleton3DList
from hri_ergonomic_msgs.msg import RebaAssessment, RebaAssessmentList

from .reba import RebaScore, SCORE_NOT_ASSESSED
from .pose_remap import remap_pose_to_reba, reba_inputs_are_sufficient, remap_scalar_to_reba, reba_region_confidence
from .deba_model_def import load_checkpoint

import os
import torch

TOTAL_JOINTS = 18

# Order the DEBA network expects, matching ALL_FEATURE_FIELDS in
# generate_deba_dataset.py. The checkpoint carries its own copy and the node
# verifies the two agree at load time, so a reordering on the training side
# can no longer silently feed the wrong column into the wrong input.
DEBA_FEATURE_ORDER = [
    "neck_angle", "trunk_angle", "legs_angle", "load_kg",
    "upper_arm_angle", "lower_arm_angle", "wrist_angle",
    "neck_side", "trunk_side", "legs_unstable",
    "shoulder_raised", "arm_abducted", "leaning", "wrist_twisted",
    "activity_static", "activity_repeated", "activity_rapid_change",
    "coupling_score",
]

# Published in risk_level when no final REBA score could be produced. Must
# match RebaAssessment_Constants::RISK_UNKNOWN in the IDL.
RISK_UNKNOWN = 255

JOINT_NAMES = [
    "Nose", "Neck", "R_Shoulder", "R_Elbow", "R_Wrist",
    "L_Shoulder", "L_Elbow", "L_Wrist", "R_Hip", "R_Knee",
    "R_Ankle", "L_Hip", "L_Knee", "L_Ankle", "L_Eye", "R_Eye",
    "L_Ear", "R_Ear"
]

BODY_ANGLE_NAMES = [
    "Neck Angle",
    "Neck Side Bend",
    "Trunk Angle",
    "Trunk Side Bend",
    "Legs Unstable",
    "Leg Angle",
    "Load"
]

ARM_ANGLE_NAMES = [
    "Upper Arm Angle",
    "Shoulder Raised",
    "Arm Abducted",
    "Arm Supported",
    "Lower Arm Angle",
    "Wrist Angle",
    "Wrist Twisted"
]

RISK_NAMES = {
    0: "Negligible Risk",
    1: "Low Risk. Change may be needed",
    2: "Medium Risk. Further Investigate. Change Soon",
    3: "High Risk. Investigate and Implement Change",
    4: "Very High Risk. Implement Change",
}

def is_valid(pt_obj, threshold=0.4):
    # 1. Confidence check
    if hasattr(pt_obj, 'confidence') and pt_obj.confidence < threshold:
        return False

    pt = [pt_obj.x, pt_obj.y, pt_obj.z]

    # 2. NaN (Not a Number) check
    if any(math.isnan(v) for v in pt):
        return False

    # 3. Z-outlier or zero point check
    if all(abs(v) < 1e-6 for v in pt):
        return False

    return True

class ErgonomicScorerNode(Node):
    def __init__(self):
        super().__init__('ergonomic_scorer_node')

        # QoS. RELIABLE matches the recorded bags, but a live ZED /
        # hri_pose_detect pipeline publishes BEST_EFFORT, and a RELIABLE
        # subscriber will simply never connect to it. Exposed as a parameter
        # so the same node works against both without a code change.
        self.declare_parameter('use_sensor_qos', False)
        use_sensor_qos = self.get_parameter('use_sensor_qos').get_parameter_value().bool_value

        skel_qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT if use_sensor_qos else ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        pub_qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.subscription = self.create_subscription(
            Skeleton3DList,
            '/humans/bodies/skel3D',
            self.skeleton_callback,
            skel_qos_profile
        )

        self.reba_pub = self.create_publisher(
            RebaAssessmentList,
            '/humans/bodies/ergonomics/reba',
            pub_qos_profile
        )

        self.declare_parameter('high_risk_threshold', 8)
        self.threshold = self.get_parameter('high_risk_threshold').get_parameter_value().integer_value

        # Set to False to prevent printing verbose coordinate logs to screen
        self.declare_parameter('verbose_logging', False)
        self.verbose = self.get_parameter('verbose_logging').get_parameter_value().bool_value

        self.declare_parameter('confidence_threshold', 0.4)
        self.confidence_threshold = self.get_parameter('confidence_threshold').get_parameter_value().double_value

        # --- REBA terms not derivable from skeleton geometry alone ---
        # These cannot be computed from pose data (no hand/object tracking,
        # no load-cell/scale input), so they are exposed as ROS parameters
        # until a proper sensor/estimator feeds them. Each has a companion
        # '*_known' flag that is published in the assessment, so a consumer can
        # distinguish "0 because the posture is good" from "0 because nothing
        # ever measured it" - the two used to be indistinguishable.
        self.declare_parameter('load_kg', 0.0)
        self.load_kg = self.get_parameter('load_kg').get_parameter_value().double_value

        self.declare_parameter('load_known', False)
        self.load_known = self.get_parameter('load_known').get_parameter_value().bool_value

        self.declare_parameter('coupling_score', 0)  # 0=good,1=fair,2=poor,3=unacceptable
        self.coupling_score = self.get_parameter('coupling_score').get_parameter_value().integer_value

        self.declare_parameter('coupling_known', False)
        self.coupling_known = self.get_parameter('coupling_known').get_parameter_value().bool_value

        # REBA's upper-arm "-1" modifier: the arm is supported or the person is
        # leaning. Not observable from a COCO-18 skeleton. It used to be
        # inferred from trunk flexion > 30 deg, which removed a point from
        # exactly the forward-bent postures that should score highest.
        self.declare_parameter('arm_supported', False)
        self.arm_supported = self.get_parameter('arm_supported').get_parameter_value().bool_value

        # --- REBA Activity modifier (0-3, see reba.py compute_activity_score) ---
        # TODO: replace these manual overrides with a real per-body pose
        # history tracker (static/repeated/rapid-change detection over time).
        self.declare_parameter('activity_static', False)
        self.activity_static = self.get_parameter('activity_static').get_parameter_value().bool_value

        self.declare_parameter('activity_repeated', False)
        self.activity_repeated = self.get_parameter('activity_repeated').get_parameter_value().bool_value

        self.declare_parameter('activity_rapid_change', False)
        self.activity_rapid_change = self.get_parameter('activity_rapid_change').get_parameter_value().bool_value

        self.declare_parameter('activity_known', False)
        self.activity_known = self.get_parameter('activity_known').get_parameter_value().bool_value

        # Shoulder depth difference above which the body counts as side-on and
        # the far-side joints are treated as hallucinated. Shoulder width is
        # ~0.40 m, so the previous 0.15 m fired at roughly 22 deg of torso
        # rotation and stripped half the skeleton on most frames.
        self.declare_parameter('profile_depth_threshold', 0.25)
        self.profile_depth_threshold = self.get_parameter(
            'profile_depth_threshold').get_parameter_value().double_value

        self.declare_parameter('swap_y_z', False)
        self.declare_parameter('flip_y_sign', True)

        # Dictionary to control output print rate (throttling)
        self.last_print_times = {}

        # Real-time factor bookkeeping: a dropped frame is the honest signal
        # that the pipeline cannot keep up with the camera.
        self.dropped_frames = 0
        self.received_frames = 0

        self.get_logger().info(f"Parameter changed. Threshold: {self.threshold}")
        self.get_logger().info("Ergonomic Scorer Node started. Granular partial REBA enabled.")
        self.device = torch.device("cpu")  # inference only, CPU is plenty
        self.deba_model = None
        self.deba_ready = False

        package_share_directory = get_package_share_directory('hri_ergonomic_scorer')
        model_path = os.path.join(package_share_directory, 'models', 'deba_model.pth')

        try:
            model, feature_names = load_checkpoint(model_path, map_location=self.device)
            if feature_names and list(feature_names) != DEBA_FEATURE_ORDER:
                raise ValueError(
                    f"DEBA checkpoint feature order does not match the node.\n"
                    f"  checkpoint: {list(feature_names)}\n"
                    f"  node      : {DEBA_FEATURE_ORDER}"
                )
            self.deba_model = model.to(self.device)
            self.deba_ready = True
            self.get_logger().info(f"DEBA PyTorch model loaded: {model_path}")
        except Exception as e:
            # DEBA stays off rather than publishing numbers from a randomly
            # initialised or mismatched network. REBA is unaffected.
            self.get_logger().error(
                f"DEBA model could not be loaded - DEBA output DISABLED, REBA "
                f"continues normally. Regenerate the dataset and retrain with "
                f"train_deba_model.py. Error: {e}")

        self.skel_queue = queue.Queue(maxsize=5)

        self.inference_thread = threading.Thread(target=self.inference_worker)
        self.inference_thread.daemon = True
        self.inference_thread.start()
        self.get_logger().info("Background ML Inference thread started.")

    def skeleton_callback(self, msg):
        """PRODUCER: hands the message to the worker without blocking the executor."""
        self.received_frames += 1
        try:
            self.skel_queue.put_nowait(msg)
        except queue.Full:
            # Worker is behind; drop this frame but count it so the real-time
            # factor can actually be reported instead of silently degrading.
            self.dropped_frames += 1

    def inference_worker(self):
        """CONSUMER: REBA + DEBA loop running off the executor thread."""
        while rclpy.ok():
            try:
                msg = self.skel_queue.get(timeout=0.1)
                self._process_message(msg)
            except queue.Empty:
                continue
            except Exception as e:
                self.get_logger().error(f"Inference worker error: {e}")

    def _process_message(self, msg):
        start_time = time.perf_counter()

        active_bodies = []
        for i, s in enumerate(getattr(msg, 'skeletons', [])):
            valid_joint_count = 0
            for pt_obj in s.skeleton:
                if is_valid(pt_obj, self.confidence_threshold):
                    valid_joint_count += 1

            if valid_joint_count >= 5:
                if not s.key or s.key.strip() == "":
                    s.key = f"human_untracked_{i}"
                active_bodies.append(s)

        if not active_bodies:
            return

        out_list = RebaAssessmentList()
        out_list.header = msg.header

        out_index = 0
        for body in active_bodies:
            if out_index >= 10:
                break
            assessment = self._assess_body(body, msg.header)
            if assessment is not None:
                out_list.assessments[out_index] = assessment
                out_index += 1

        if out_index > 0:
            self.reba_pub.publish(out_list)

        end_time = time.perf_counter()
        processing_time_ms = (end_time - start_time) * 1000.0

        current_sec = msg.header.stamp.sec
        if not hasattr(self, 'last_log_sec') or current_sec != self.last_log_sec:
            drop_pct = (100.0 * self.dropped_frames / self.received_frames) if self.received_frames else 0.0
            self.get_logger().info(
                f"[Real-Time Profiler] Processed {out_index} human(s) in {processing_time_ms:.2f} ms "
                f"| Max possible FPS: {1000.0 / processing_time_ms if processing_time_ms > 0 else 0:.1f} "
                f"| Dropped {self.dropped_frames}/{self.received_frames} frames ({drop_pct:.1f}%)"
            )
            self.last_log_sec = current_sec

    def _assess_body(self, body, header):
        pose_matrix = np.zeros((TOTAL_JOINTS, 3))
        confidence_matrix = np.zeros(TOTAL_JOINTS)
        valid_mask = np.zeros(TOTAL_JOINTS, dtype=bool)
        valid_joints_count = 0

        # Get body overall confidence score and normalize to 0.0 - 1.0 range
        body_overall_conf = getattr(body, 'confidence', 0.0)
        if body_overall_conf > 1.0:
            body_overall_conf /= 100.0

        num_joints = min(len(body.skeleton), TOTAL_JOINTS)
        for i in range(num_joints):
            pt_obj = body.skeleton[i]

            # Use body overall confidence if joint-specific confidence is missing
            confidence_matrix[i] = getattr(pt_obj, 'confidence', body_overall_conf)

            if is_valid(pt_obj, self.confidence_threshold):
                pose_matrix[i] = [pt_obj.x, pt_obj.y, pt_obj.z]
                valid_mask[i] = True
                valid_joints_count += 1

        # --- DEPTH OCCLUSION FILTER ---
        if valid_mask[2] and valid_mask[5]:
            z_diff = pose_matrix[2, 2] - pose_matrix[5, 2]
            if abs(z_diff) > self.profile_depth_threshold:
                if z_diff > 0:
                    hidden_joints = [2, 3, 4, 8, 9, 10]
                    side_name = "Right"
                else:
                    hidden_joints = [5, 6, 7, 11, 12, 13]
                    side_name = "Left"

                for hj in hidden_joints:
                    if valid_mask[hj]:
                        valid_mask[hj] = False
                        valid_joints_count -= 1

                if self.verbose:
                    self.get_logger().info(f"[{body.key}] Profile pose detected! Filtered hallucinated {side_name} side joints.")
        # ---------------------------------------------------

        completeness = valid_joints_count / float(TOTAL_JOINTS)
        if valid_joints_count == 0:
            return None

        p_swap = self.get_parameter('swap_y_z').get_parameter_value().bool_value
        p_flip = self.get_parameter('flip_y_sign').get_parameter_value().bool_value

        reba_pose, reba_valid = remap_pose_to_reba(pose_matrix, valid_mask, swap_y_z=p_swap, flip_y_sign=p_flip)
        readiness = reba_inputs_are_sufficient(reba_valid)

        reba_confidence = remap_scalar_to_reba(confidence_matrix)
        region_conf = reba_region_confidence(reba_confidence)

        try:
            assessment = RebaAssessment()
            assessment.key = body.key
            assessment.header = header
            assessment.completeness = float(completeness)

            reba = RebaScore()

            # ==========================================================
            # GROUP A (BODY)
            # ==========================================================
            # A region that could not be measured keeps its neutral 0 here.
            # That is unavoidable if we want to score partial skeletons at
            # all, but it is no longer silent: every region publishes an
            # *_assessed flag and score_is_lower_bound marks the whole
            # assessment as an under-estimate.
            body_angles = np.zeros(7)
            leg_side = "Unknown"

            calc_angles_r = reba.get_body_angles_from_pose_right(reba_pose)
            calc_angles_l = reba.get_body_angles_from_pose_left(reba_pose)

            neck_assessed = bool(readiness["neck_ok"])
            trunk_assessed = bool(readiness["trunk_ok"])

            # Neck and trunk are built from joints shared by both sides, so
            # the left/right variants return identical values for them.
            if neck_assessed:
                body_angles[0] = calc_angles_r[0]
                body_angles[1] = calc_angles_r[1]

            if trunk_assessed:
                body_angles[2] = calc_angles_r[2]
                body_angles[3] = calc_angles_r[3]

            # Legs: use whichever side is valid; if both, take the higher risk.
            if readiness["right_leg_ok"] and readiness["left_leg_ok"]:
                if abs(calc_angles_r[5]) >= abs(calc_angles_l[5]):
                    body_angles[4] = calc_angles_r[4]
                    body_angles[5] = calc_angles_r[5]
                    leg_side = "Right"
                else:
                    body_angles[4] = calc_angles_l[4]
                    body_angles[5] = calc_angles_l[5]
                    leg_side = "Left"
            elif readiness["right_leg_ok"]:
                body_angles[4] = calc_angles_r[4]
                body_angles[5] = calc_angles_r[5]
                leg_side = "Right"
            elif readiness["left_leg_ok"]:
                body_angles[4] = calc_angles_l[4]
                body_angles[5] = calc_angles_l[5]
                leg_side = "Left"

            legs_assessed = leg_side != "Unknown"
            group_a_valid = neck_assessed or trunk_assessed or legs_assessed

            body_conf = np.zeros(7)
            if neck_assessed:
                body_conf[0] = body_conf[1] = region_conf["neck_conf"]
            if trunk_assessed:
                body_conf[2] = body_conf[3] = region_conf["trunk_conf"]
            if leg_side == "Right":
                body_conf[4] = body_conf[5] = region_conf["right_leg_conf"]
            elif leg_side == "Left":
                body_conf[4] = body_conf[5] = region_conf["left_leg_conf"]

            body_angles[6] = self.load_kg if self.load_known else 0.0

            if group_a_valid:
                reba.set_body(body_angles)
                score_a, partial_a = reba.compute_score_a()
                assessment.score_a = int(score_a)
                assessment.neck_score = int(partial_a[0]) if neck_assessed else SCORE_NOT_ASSESSED
                assessment.trunk_score = int(partial_a[1]) if trunk_assessed else SCORE_NOT_ASSESSED
                assessment.leg_score = int(partial_a[2]) if legs_assessed else SCORE_NOT_ASSESSED
                assessment.load_score = int(partial_a[3]) if self.load_known else SCORE_NOT_ASSESSED
            else:
                score_a = None
                assessment.score_a = SCORE_NOT_ASSESSED
                assessment.neck_score = SCORE_NOT_ASSESSED
                assessment.trunk_score = SCORE_NOT_ASSESSED
                assessment.leg_score = SCORE_NOT_ASSESSED
                assessment.load_score = SCORE_NOT_ASSESSED

            assessment.group_a_valid = group_a_valid
            assessment.neck_assessed = neck_assessed
            assessment.trunk_assessed = trunk_assessed
            assessment.legs_assessed = legs_assessed
            assessment.load_known = bool(self.load_known)
            assessment.body_confidence = body_conf.tolist()

            # ==========================================================
            # GROUP B (ARMS) - bilateral, highest risk side wins
            # ==========================================================
            calc_arm_r = reba.get_arms_angles_from_pose_right(reba_pose)
            calc_arm_l = reba.get_arms_angles_from_pose_left(reba_pose)

            score_b_r, partial_b_r = 0, [0, 0, 0, 0]
            score_b_l, partial_b_l = 0, [0, 0, 0, 0]

            if readiness["right_arm_ok"]:
                arm_angles_r = np.zeros(8)
                arm_angles_r[:7] = calc_arm_r
                arm_angles_r[3] = 1.0 if self.arm_supported else 0.0
                arm_angles_r[7] = self.coupling_score if self.coupling_known else 0
                reba.set_arms(arm_angles_r)
                score_b_r, partial_b_r = reba.compute_score_b()

            if readiness["left_arm_ok"]:
                arm_angles_l = np.zeros(8)
                arm_angles_l[:7] = calc_arm_l
                arm_angles_l[3] = 1.0 if self.arm_supported else 0.0
                arm_angles_l[7] = self.coupling_score if self.coupling_known else 0
                reba.set_arms(arm_angles_l)
                score_b_l, partial_b_l = reba.compute_score_b()

            arm_angles = np.zeros(7)
            if score_b_r >= score_b_l and score_b_r > 0:
                score_b = score_b_r
                partial_b = partial_b_r
                arm_side = "Right"
                arm_angles = calc_arm_r
            elif score_b_l > score_b_r:
                score_b = score_b_l
                partial_b = partial_b_l
                arm_side = "Left"
                arm_angles = calc_arm_l
            else:
                score_b = None
                partial_b = [0, 0, 0, 0]
                arm_side = "Unknown"

            group_b_valid = arm_side != "Unknown"
            if arm_side == "Right":
                upper_arm_assessed = bool(readiness["right_upper_arm_ok"])
                lower_arm_assessed = bool(readiness["right_lower_arm_ok"])
            elif arm_side == "Left":
                upper_arm_assessed = bool(readiness["left_upper_arm_ok"])
                lower_arm_assessed = bool(readiness["left_lower_arm_ok"])
            else:
                upper_arm_assessed = lower_arm_assessed = False

            arm_conf = np.zeros(7)
            if arm_side == "Right":
                arm_conf[0] = arm_conf[1] = arm_conf[2] = arm_conf[3] = region_conf["right_upper_arm_conf"]
                arm_conf[4] = region_conf["right_lower_arm_conf"]
            elif arm_side == "Left":
                arm_conf[0] = arm_conf[1] = arm_conf[2] = arm_conf[3] = region_conf["left_upper_arm_conf"]
                arm_conf[4] = region_conf["left_lower_arm_conf"]

            assessment.group_b_valid = group_b_valid
            assessment.score_b = int(score_b) if group_b_valid else SCORE_NOT_ASSESSED
            assessment.upper_arm_score = int(partial_b[0]) if upper_arm_assessed else SCORE_NOT_ASSESSED
            assessment.lower_arm_score = int(partial_b[1]) if lower_arm_assessed else SCORE_NOT_ASSESSED
            # COCO-18 carries no wrist orientation, so the wrist item is never
            # measured. Publishing the neutral 1 as if it were an observation
            # made Table B look better than the data supports.
            assessment.wrist_score = SCORE_NOT_ASSESSED
            assessment.coupling_score = int(partial_b[3]) if self.coupling_known else SCORE_NOT_ASSESSED
            assessment.upper_arm_assessed = upper_arm_assessed
            assessment.lower_arm_assessed = lower_arm_assessed
            assessment.wrist_assessed = False
            assessment.coupling_known = bool(self.coupling_known)
            assessment.arm_confidence = arm_conf.tolist()

            # ==========================================================
            # FINAL SCORE (C)
            # ==========================================================
            activity_score = reba.compute_activity_score(
                self.activity_static,
                self.activity_repeated,
                self.activity_rapid_change
            ) if self.activity_known else 0
            assessment.activity_known = bool(self.activity_known)

            # Table C needs BOTH group scores. Substituting a neutral 1 for a
            # group we never measured is what used to turn a fully occluded
            # body into "REBA 1 / Negligible Risk".
            if group_a_valid and group_b_valid:
                score_c_raw, final_score, caption = reba.compute_score_c(score_a, score_b, activity_score)
                assessment.score_c = int(final_score)
                assessment.risk_level = RebaScore.score_c_to_5_classes(final_score)
                risk_lvl = caption
            else:
                score_c_raw, final_score, caption = None, None, "Not assessed (incomplete skeleton)"
                assessment.score_c = SCORE_NOT_ASSESSED
                assessment.risk_level = RISK_UNKNOWN
                risk_lvl = caption

            # Any unmeasured contributor means the published score can only be
            # a floor on the real risk.
            assessment.score_is_lower_bound = not all([
                neck_assessed, trunk_assessed, legs_assessed,
                upper_arm_assessed, lower_arm_assessed,
                self.load_known, self.coupling_known, self.activity_known,
            ])

            # ==========================================================
            # DEBA
            # ==========================================================
            # Only run when both groups are real. Feeding zero-filled angles to
            # the network produced a confident number for a posture nobody
            # observed.
            predicted_deba = float('nan')
            deba_valid = False
            if self.deba_ready and group_a_valid and group_b_valid:
                with torch.no_grad():
                    deba_input = [
                        float(body_angles[0]), float(body_angles[2]), float(body_angles[5]), float(body_angles[6]),
                        float(arm_angles[0]), float(arm_angles[4]), float(arm_angles[5]),
                        float(body_angles[1]), float(body_angles[3]), float(body_angles[4]),
                        float(arm_angles[1]), float(arm_angles[2]),
                        1.0 if self.arm_supported else 0.0, float(arm_angles[6]),
                        float(self.activity_static and self.activity_known),
                        float(self.activity_repeated and self.activity_known),
                        float(self.activity_rapid_change and self.activity_known),
                        float(self.coupling_score if self.coupling_known else 0)
                    ]
                    tensor_input = torch.tensor([deba_input], dtype=torch.float32).to(self.device)
                    predicted_deba = self.deba_model(tensor_input).item()
                    deba_valid = True

            assessment.deba_score = float(predicted_deba)
            assessment.deba_valid = deba_valid

            if final_score is not None and final_score >= self.threshold:
                self.get_logger().warn(
                    f"[{body.key}] REBA {final_score} >= high_risk_threshold "
                    f"({self.threshold}): {risk_lvl}"
                )

            # --- THROTTLE SCREEN OUTPUT ---
            current_time = self.get_clock().now().nanoseconds / 1e9
            last_time = self.last_print_times.get(body.key, 0.0)

            if current_time - last_time > 0.5:
                self._print_reba_breakdown(
                    key=body.key,
                    score_a=score_a,
                    score_b=score_b,
                    score_c=final_score,
                    risk_lvl=risk_lvl,
                    completeness=completeness,
                    valid_joints=valid_joints_count,
                    leg_side=leg_side,
                    arm_side=arm_side,
                    body_angles=body_angles,
                    arm_angles=arm_angles,
                    body_conf=body_conf,
                    arm_conf=arm_conf,
                    deba_score=predicted_deba,
                    deba_valid=deba_valid,
                    lower_bound=assessment.score_is_lower_bound,
                )

                if self.verbose:
                    self._print_raw_dump(body.key, pose_matrix, completeness)

                self.last_print_times[body.key] = current_time

        except Exception as e:
            # Returning the half-filled assessment here published score_c = 0
            # and risk_level = 0, which every consumer read as "negligible".
            self.get_logger().error(f"[{body.key}] REBA calculation error: {e}")
            return None

        return assessment

    @staticmethod
    def _fmt_score(value):
        return "n/a" if value is None else f"{int(value)}"

    def _print_reba_breakdown(
        self,
        key,
        score_a,
        score_b,
        score_c,
        risk_lvl,
        completeness,
        valid_joints,
        leg_side,
        arm_side,
        body_angles,
        arm_angles,
        body_conf,
        arm_conf,
        deba_score,
        deba_valid,
        lower_bound,
    ):
        deba_txt = f"{deba_score:.2f}" if deba_valid else "n/a (incomplete)"
        bound_txt = "  [LOWER BOUND - some terms never measured]" if lower_bound else ""

        lines = [
            "",
            "REBA ASSESSMENT ---------------------------------------------",
            f"  Target ID     : {key}",
            f"  Completeness  : {completeness*100:3.0f}%  ({valid_joints}/{TOTAL_JOINTS} joints)",
            f"  Score A / B   : {self._fmt_score(score_a)} / {self._fmt_score(score_b)}",
            f"  Final REBA    : {self._fmt_score(score_c)}  =>  {risk_lvl}{bound_txt}",
            f"  DEBA          : {deba_txt}",
            "  ---------------------------------------------------------",
            f"  [GROUP A: BODY]  leg side used: {leg_side}",
            "    (neck and trunk are side-independent, not tied to leg side)",
        ]
        for name, value, conf in zip(BODY_ANGLE_NAMES, body_angles, body_conf):
            conf_str = f"{conf*100:3.0f}%" if conf > 0 else "  - "
            lines.append(f"    {name:<18}: {float(value):8.2f}   (conf {conf_str})")

        lines.append(f"  [GROUP B: ARM ]  side evaluated: {arm_side}")
        for name, value, conf in zip(ARM_ANGLE_NAMES, arm_angles, arm_conf):
            conf_str = f"{conf*100:3.0f}%" if conf > 0 else "  - "
            lines.append(f"    {name:<18}: {float(value):8.2f}   (conf {conf_str})")
        lines.append("-------------------------------------------------------------")

        self.get_logger().info("\n".join(lines))

    def _print_raw_dump(self, key, pose_matrix, completeness):
        raw_dump = ""
        for i in range(TOTAL_JOINTS):
            pt = pose_matrix[i]
            if any(abs(v) > 1e-6 for v in pt):
                raw_dump += f"  {JOINT_NAMES[i]:<12}: [X: {pt[0]:5.2f}, Y: {pt[1]:5.2f}, Z: {pt[2]:5.2f}]\n"
            else:
                raw_dump += f"  {JOINT_NAMES[i]:<12}: MISSING\n"

        self.get_logger().info(
            f"\n--- RAW JOINT COORDINATES (Vulcanexus COCO-18) ---\n"
            f"Target ID : {key}\n"
            f"Data Ratio: {completeness*100:.0f}%\n"
            f"--------------------------------------------------\n"
            f"{raw_dump}"
            f"--------------------------------------------------"
        )

def main(args=None):
    rclpy.init(args=args)
    node = ErgonomicScorerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n[INFO] Node stopped by user.")
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass

if __name__ == '__main__':
    main()
