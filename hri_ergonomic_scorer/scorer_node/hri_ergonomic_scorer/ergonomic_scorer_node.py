import rclpy
from rclpy.node import Node
import numpy as np
import math
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from hri_msgs.msg import Skeleton3DList
from hri_ergonomic_msgs.msg import RebaAssessment, RebaAssessmentList

from .reba import RebaScore
from .pose_remap import remap_pose_to_reba, reba_inputs_are_sufficient, remap_scalar_to_reba, reba_region_confidence

import os
import torch
import torch.nn as nn

class DebaMLP(nn.Module):
    def __init__(self, input_dim=18):
        super(DebaMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.network(x)

TOTAL_JOINTS = 18

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
    "Walking",
    "Leg Angle",
    "Load"
]

ARM_ANGLE_NAMES = [
    "Upper Arm Angle",
    "Shoulder Raised",
    "Arm Abducted",
    "Leaning",
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

        bag_qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.subscription = self.create_subscription(
            Skeleton3DList,
            '/humans/bodies/skel3D',
            self.skeleton_callback,
            bag_qos_profile
        )

        self.reba_pub = self.create_publisher(
            RebaAssessmentList,
            '/humans/bodies/ergonomics/reba',
            bag_qos_profile
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
        # until a proper sensor/estimator feeds them. Defaults are the most
        # conservative REBA values (no load, good coupling).
        self.declare_parameter('load_kg', 0.0)
        self.load_kg = self.get_parameter('load_kg').get_parameter_value().double_value

        self.declare_parameter('coupling_score', 0)  # 0=good,1=fair,2=poor,3=unacceptable
        self.coupling_score = self.get_parameter('coupling_score').get_parameter_value().integer_value

        # --- REBA Activity modifier (0-3, see reba.py compute_activity_score) ---
        # TODO: replace these manual overrides with a real per-body pose
        # history tracker (static/repeated/rapid-change detection over time)
        # once the DEBA work needs it too.
        self.declare_parameter('activity_static', False)
        self.activity_static = self.get_parameter('activity_static').get_parameter_value().bool_value

        self.declare_parameter('activity_repeated', False)
        self.activity_repeated = self.get_parameter('activity_repeated').get_parameter_value().bool_value

        self.declare_parameter('activity_rapid_change', False)
        self.activity_rapid_change = self.get_parameter('activity_rapid_change').get_parameter_value().bool_value

        self.declare_parameter('swap_y_z', False)
        self.declare_parameter('flip_y_sign', True)

        # Dictionary to control output print rate (throttling)
        self.last_print_times = {}

        self.get_logger().info(f"Parameter changed. Threshold: {self.threshold}")
        self.get_logger().info("Ergonomic Scorer Node started. Granular partial REBA enabled.")
        self.device = torch.device("cpu") # Sadece çıkarım (inference) yapacağımız için CPU fazlasıyla yeterli
        self.deba_model = DebaMLP(input_dim=18).to(self.device)
        
        model_path = os.path.expanduser('~/hri_ws/src/hri_ergonomic_scorer/hri_ergonomic_scorer/scorer_node/hri_ergonomic_scorer/deba_model.pth')
        
        try:
            self.deba_model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.deba_model.eval() # Eğitimi kapat, sadece tahmin yap
            self.get_logger().info(f"DEBA PyTorch modeli basariyla yuklendi: {model_path}")
        except Exception as e:
            self.get_logger().error(f"DEBA Modeli yuklenemedi! Hata: {e}")

    def skeleton_callback(self, msg):
        active_bodies = []
        for i, s in enumerate(getattr(msg, 'skeletons', [])):
            valid_joint_count = 0
            for pt_obj in s.skeleton:
                # Get threshold from node parameter
                if is_valid(pt_obj, self.confidence_threshold):
                    valid_joint_count += 1
            
            # Ignore ghost skeletons (e.g. only a single knee or nose detected).
            # A skeleton needs at least 5 valid joints to be evaluated.
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
            assessment = self._assess_body(body)
            if assessment is not None:
                out_list.assessments[out_index] = assessment
                out_index += 1
                
        if out_index > 0:
            self.reba_pub.publish(out_list)

    def _assess_body(self, body):
        pose_matrix = np.zeros((TOTAL_JOINTS, 3))
        confidence_matrix = np.zeros(TOTAL_JOINTS)   # <-- new
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
            PROFILE_THRESHOLD = 0.15 
            if abs(z_diff) > PROFILE_THRESHOLD:
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

        # Parametreleri dinamik olarak çek
        p_swap = self.get_parameter('swap_y_z').get_parameter_value().bool_value
        p_flip = self.get_parameter('flip_y_sign').get_parameter_value().bool_value

        # Fonksiyona gönder
        reba_pose, reba_valid = remap_pose_to_reba(pose_matrix, valid_mask, swap_y_z=p_swap, flip_y_sign=p_flip)  
        readiness = reba_inputs_are_sufficient(reba_valid)

        reba_confidence = remap_scalar_to_reba(confidence_matrix)
        region_conf = reba_region_confidence(reba_confidence)

        assessment = RebaAssessment()
        assessment.key = body.key
        assessment.completeness = float(completeness)

        try:
            reba = RebaScore()
            
            # 1. GROUP A (BODY) CALCULATION - Independent Parts
            body_angles = np.zeros(7)
            body_side = "Unknown"  # overall Group A validity flag (any part computed)
            # NOTE: neck/trunk angles are computed from joints shared by both
            # sides (mid-hip average, both shoulders) - get_body_angles_from_
            # pose_left/right give IDENTICAL neck/trunk results, only the leg
            # calculation actually differs by side. So "side" only has real
            # meaning for the legs; leg_side is tracked separately below and
            # is what gets shown on the dashboard, instead of overloading a
            # single body_side as if it described the whole Group A.
            leg_side = "Unknown"

            calc_angles_r = reba.get_body_angles_from_pose_right(reba_pose)
            calc_angles_l = reba.get_body_angles_from_pose_left(reba_pose)

            # Neck (side-independent, see NOTE above - calc_angles_r/_l give the same value)
            if readiness["neck_ok"]:
                body_angles[0] = calc_angles_r[0]
                body_angles[1] = calc_angles_r[1]
                body_side = "Computed"

            # Trunk (side-independent, see NOTE above)
            if readiness["trunk_ok"]:
                body_angles[2] = calc_angles_r[2]
                body_angles[3] = calc_angles_r[3]
                body_side = "Computed"

            # Legs (Choose the leg that is valid, if both are valid choose highest risk angle)
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

            if leg_side != "Unknown":
                body_side = "Computed"

            body_conf = np.zeros(7)
            if readiness["neck_ok"]:
                body_conf[0] = body_conf[1] = region_conf["neck_conf"]
            if readiness["trunk_ok"]:
                body_conf[2] = body_conf[3] = region_conf["trunk_conf"]
            if leg_side == "Right":
                body_conf[4] = body_conf[5] = region_conf["right_leg_conf"]
            elif leg_side == "Left":
                body_conf[4] = body_conf[5] = region_conf["left_leg_conf"]

            body_angles[6] = self.load_kg

            reba.set_body(body_angles)
            score_a, partial_a = reba.compute_score_a()
            
            assessment.group_a_valid = True if body_side != "Unknown" else False
            assessment.score_a = int(score_a)
            assessment.neck_score = int(partial_a[0])
            assessment.trunk_score = int(partial_a[1])
            assessment.leg_score = int(partial_a[2])
            assessment.load_score = int(partial_a[3])
            assessment.body_confidence = body_conf.tolist()

            # 2. GROUP B (ARM) CALCULATION - BILATERAL AGGREGATION
            calc_arm_r = reba.get_arms_angles_from_pose_right(reba_pose)
            calc_arm_l = reba.get_arms_angles_from_pose_left(reba_pose)
            
            score_b_r, partial_b_r = 0, [0, 0, 0, 0]
            score_b_l, partial_b_l = 0, [0, 0, 0, 0]
            
            # Compute Score B for right arm
            if readiness["right_arm_ok"]:
                arm_angles_r = np.zeros(8)
                arm_angles_r[:7] = calc_arm_r
                arm_angles_r[7] = self.coupling_score
                reba.set_arms(arm_angles_r)
                score_b_r, partial_b_r = reba.compute_score_b()

            # Compute Score B for left arm
            if readiness["left_arm_ok"]:
                arm_angles_l = np.zeros(8)
                arm_angles_l[:7] = calc_arm_l
                arm_angles_l[7] = self.coupling_score
                reba.set_arms(arm_angles_l)
                score_b_l, partial_b_l = reba.compute_score_b()

            # Compare both arms and select the highest risk score
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
                score_b = 0
                partial_b = [0, 0, 0, 0]
                arm_side = "Unknown"

            arm_conf = np.zeros(7)
            if arm_side == "Right":
                arm_conf[0] = arm_conf[1] = arm_conf[2] = arm_conf[3] = region_conf["right_upper_arm_conf"]
                arm_conf[4] = region_conf["right_lower_arm_conf"]
            elif arm_side == "Left":
                arm_conf[0] = arm_conf[1] = arm_conf[2] = arm_conf[3] = region_conf["left_upper_arm_conf"]
                arm_conf[4] = region_conf["left_lower_arm_conf"]

            assessment.group_b_valid = True if arm_side != "Unknown" else False
            assessment.score_b = int(score_b)
            assessment.upper_arm_score = int(partial_b[0])
            assessment.lower_arm_score = int(partial_b[1])
            assessment.wrist_score = int(partial_b[2])
            assessment.coupling_score = int(partial_b[3])
            assessment.arm_confidence = arm_conf.tolist()

            # 3. FINAL SCORE (C) CALCULATION
            # Compute activity score
            activity_score = reba.compute_activity_score(
                self.activity_static, 
                self.activity_repeated, 
                self.activity_rapid_change
            )
            
            # Get final REBA score (up to 15)
            score_c_raw, final_score, caption = reba.compute_score_c(score_a, score_b, activity_score)
            
            assessment.score_c = int(final_score)

            # YENI EKLENDI: DEBA Skoru Çıkarımı (Inference)
            # CSV veri setindeki sırayla 18 adet feature'ı diziyoruz.
            # body_angles: 0:Neck, 1:NeckSide, 2:Trunk, 3:TrunkSide, 4:Walking, 5:LegAngle, 6:Load 
            # arm_angles: 0:UpperArm, 1:ShoulderRaised, 2:ArmAbducted, 3:Leaning, 4:LowerArm, 5:WristAngle, 6:WristTwisted
            with torch.no_grad():
                deba_input = [
                    float(body_angles[0]), float(body_angles[2]), float(body_angles[5]), float(body_angles[6]), # Neck, Trunk, Leg, Load
                    float(arm_angles[0]), float(arm_angles[4]), float(arm_angles[5]),                           # Upper, Lower, Wrist
                    float(body_angles[1]), float(body_angles[3]), float(body_angles[4]),                        # NeckSide, TrunkSide, Walking
                    float(arm_angles[1]), float(arm_angles[2]), float(arm_angles[3]), float(arm_angles[6]),     # Sh.Raised, Abd, Leaning, Tw.
                    float(self.activity_static), float(self.activity_repeated), float(self.activity_rapid_change),
                    float(self.coupling_score)
                ]
                tensor_input = torch.tensor([deba_input], dtype=torch.float32).to(self.device)
                predicted_deba = self.deba_model(tensor_input).item()
                
            assessment.deba_score = float(predicted_deba)
            
            # Map risk level based on final score
            if final_score <= 1: risk_num = 0
            elif final_score <= 3: risk_num = 1
            elif final_score <= 7: risk_num = 2
            elif final_score <= 10: risk_num = 3
            else: risk_num = 4
            
            assessment.risk_level = risk_num
            risk_lvl = caption

            # --- THROTTLE SCREEN OUTPUT ---
            current_time = self.get_clock().now().nanoseconds / 1e9
            last_time = self.last_print_times.get(body.key, 0.0)
            
            # Allow printing to screen twice a second (every 0.5 seconds)
            if current_time - last_time > 0.5:
                # Dashboard Print
                self._print_reba_breakdown(
                    key=body.key,
                    score_a=score_a,
                    score_b=score_b,
                    score_c=score_c_raw,
                    risk_lvl=risk_lvl,
                    completeness=completeness,
                    valid_joints=valid_joints_count,
                    body_side=body_side,
                    leg_side=leg_side,
                    arm_side=arm_side,
                    body_angles=body_angles,
                    arm_angles=arm_angles,
                    body_conf=body_conf,
                    arm_conf=arm_conf,
                    deba_score=predicted_deba
                )
                
                # Print raw coordinates if verbose logging is enabled
                if self.verbose:
                    self._print_raw_dump(body.key, pose_matrix, completeness)
                    
                self.last_print_times[body.key] = current_time

        except Exception as e:
            self.get_logger().error(f"[{body.key}] REBA calculation error: {e}")

        return assessment

    def _print_reba_breakdown(
        self,
        key,
        score_a,
        score_b,
        score_c,
        risk_lvl,
        completeness,
        valid_joints,
        body_side,
        leg_side,
        arm_side,
        body_angles,
        arm_angles,
        body_conf, 
        arm_conf,
        deba_score
    ):
        text = f"\n"
        text += f"┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n"
        text += f"┃ REBA ASSESSMENT DASHBOARD                                      ┃\n"
        text += f"┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫\n"
        text += f"┃ Target ID : {key:<50}                                          ┃\n"
        text += f"┃ Completeness: {completeness*100:3.0f}% ({valid_joints}/18 joints)┃\n"
        text += f"┃ Final Score : {score_c:<2}  =>  {risk_lvl:<36}                 ┃\n"
        text += f"┃ DEBA Score  : {deba_score:<5.2f}                               ┃\n"
        text += f"┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫\n"
        text += f"┃ [GROUP A: BODY] Leg Side Used: {leg_side:<28} ┃\n"
        text += f"┃   (Neck/Trunk are side-independent - not tied to leg side)     ┃\n"
        text += f"┠────────────────────────────────────────────────────────────────┨\n"

        for name, value, conf in zip(BODY_ANGLE_NAMES, body_angles, body_conf):
            val = float(value)
            conf_str = f"{conf*100:3.0f}%" if conf > 0 else "  - "
            text += f"┃   {name:<24}: {val:7.2f}°  (conf: {conf_str})           ┃\n"

        text += f"┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫\n"
        text += f"┃ [GROUP B: ARM ] Evaluated Side: {arm_side:<30} ┃\n"
        text += f"┠────────────────────────────────────────────────────────────────┨\n"

        for name, value, conf in zip(ARM_ANGLE_NAMES, arm_angles, arm_conf):
            val = float(value)
            conf_str = f"{conf*100:3.0f}%" if conf > 0 else "  - "
            text += f"┃   {name:<24}: {val:7.2f}°  (conf: {conf_str})           ┃\n"

        text += f"┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛\n"
        self.get_logger().info(text)

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