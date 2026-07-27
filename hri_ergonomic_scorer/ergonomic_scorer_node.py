import rclpy
from rclpy.node import Node
import numpy as np
import math
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from hri_msgs.msg import Skeleton3DList
from hri_ergonomic_msgs.msg import RebaAssessment, RebaAssessmentList

from .reba import RebaScore
from .pose_remap import remap_pose_to_reba, reba_inputs_are_sufficient

TOTAL_JOINTS = 18

JOINT_NAMES = [
    "Nose", "Neck", "R_Shoulder", "R_Elbow", "R_Wrist",
    "L_Shoulder", "L_Elbow", "L_Wrist", "R_Hip", "R_Knee",
    "R_Ankle", "L_Hip", "L_Knee", "L_Ankle", "R_Eye", "L_Eye",
    "R_Ear", "L_Ear"
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

def is_valid(pt):
    """
    Check if 3D point is valid.
    Returns False if NaN or exactly zero.
    """
    if any(math.isnan(v) for v in pt):
        return False
    if all(abs(v) < 1e-6 for v in pt):
        return False
    return True

class ErgonomicScorerNode(Node):
    def __init__(self):
        super().__init__('ergonomic_scorer_node')

        # QoS profile for reliable data transmission
        bag_qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Subscriber for 3D skeleton data
        self.subscription = self.create_subscription(
            Skeleton3DList,
            '/humans/bodies/skel3D',
            self.skeleton_callback,
            bag_qos_profile
        )

        # Publisher for REBA assessments
        self.reba_pub = self.create_publisher(
            RebaAssessmentList,
            '/humans/bodies/ergonomics/reba',
            bag_qos_profile
        )

        # Node parameters
        self.declare_parameter('high_risk_threshold', 8)
        self.threshold = self.get_parameter('high_risk_threshold').get_parameter_value().integer_value
        
        self.declare_parameter('verbose_logging', True)
        self.verbose = self.get_parameter('verbose_logging').get_parameter_value().bool_value

        self.get_logger().info(f"Parameter changed. Threshold: {self.threshold}")
        self.get_logger().info("Ergonomic Scorer Node started. Waiting for 3D Skeleton data...")

    def skeleton_callback(self, msg):
        """
        Process incoming skeleton data and trigger REBA assessment.
        """
        active_bodies = []
        
        # Filter skeletons with valid joints
        for i, s in enumerate(getattr(msg, 'skeletons', [])):
            has_valid_data = False
            for pt_obj in s.skeleton:
                pt = [pt_obj.x, pt_obj.y, pt_obj.z]
                if is_valid(pt):
                    has_valid_data = True
                    break
            
            if has_valid_data:
                # Assign temporary ID if missing
                if not s.key or s.key.strip() == "":
                    s.key = f"human_untracked_{i}"
                active_bodies.append(s)
        
        if not active_bodies:
            return

        out_list = RebaAssessmentList()
        out_list.header = msg.header

        out_index = 0
        for body in active_bodies:
            # Max 10 tracked bodies
            if out_index >= 10:
                break
            assessment = self._assess_body(body)
            if assessment is not None:
                # Overwrite pre-allocated slots
                out_list.assessments[out_index] = assessment
                out_index += 1
                
        # Publish if assessment generated
        if out_index > 0:
            self.reba_pub.publish(out_list)

    def _assess_body(self, body):
        """
        Convert 3D coordinates into a REBA assessment.
        """
        pose_matrix = np.zeros((TOTAL_JOINTS, 3))
        valid_mask = np.zeros(TOTAL_JOINTS, dtype=bool)
        valid_joints_count = 0
        
        # Extract joints to NumPy array
        num_joints = min(len(body.skeleton), TOTAL_JOINTS)
        for i in range(num_joints):
            pt = [body.skeleton[i].x, body.skeleton[i].y, body.skeleton[i].z]
            if is_valid(pt):
                pose_matrix[i] = pt
                valid_mask[i] = True
                valid_joints_count += 1

        # --- DEPTH OCCLUSION FILTER ---
        # COCO-18 Indices: R_Shoulder=2, L_Shoulder=5
        # ZED camera optical frame: +Z is forward (away from camera).
        if valid_mask[2] and valid_mask[5]:
            # Calculate depth (Z) difference between shoulders
            z_diff = pose_matrix[2, 2] - pose_matrix[5, 2]
            
            # If depth difference > 15cm, assume profile stance
            PROFILE_THRESHOLD = 0.15 
            
            if abs(z_diff) > PROFILE_THRESHOLD:
                # Hide the side that is further away
                if z_diff > 0:
                    # Right side is occluded
                    hidden_joints = [2, 3, 4, 8, 9, 10]
                    side_name = "Right"
                else:
                    # Left side is occluded
                    hidden_joints = [5, 6, 7, 11, 12, 13]
                    side_name = "Left"
                    
                # Remove hallucinated occluded joints
                for hj in hidden_joints:
                    if valid_mask[hj]:
                        valid_mask[hj] = False
                        valid_joints_count -= 1
                
                if self.verbose:
                    self.get_logger().info(f"[{body.key}] Profile pose detected! Filtered hallucinated {side_name} side joints. Running partial REBA.")
        # ---------------------------------------------------

        completeness = valid_joints_count / float(TOTAL_JOINTS)
        if valid_joints_count == 0:
            return None

        reba_pose, reba_valid = remap_pose_to_reba(pose_matrix, valid_mask)
        readiness = reba_inputs_are_sufficient(reba_valid)

        group_a_valid = readiness["body_group_ok"]
        group_b_valid = readiness["left_arm_ok"] or readiness["right_arm_ok"]

        # Initialize ROS 2 message
        assessment = RebaAssessment()
        assessment.key = body.key
        assessment.completeness = float(completeness)
        assessment.group_a_valid = bool(group_a_valid)
        assessment.group_b_valid = bool(group_b_valid)

        try:
            reba = RebaScore()
            score_a = score_b = 0
            
            final_body_angles = {}
            final_arm_angles = {}

            if group_a_valid:
                # Calculate body scores
                angles_r = reba.get_body_angles_from_pose_right(reba_pose)
                reba.set_body(np.abs(angles_r))
                score_a_r, partial_a_r = reba.compute_score_a()

                angles_l = reba.get_body_angles_from_pose_left(reba_pose)
                reba.set_body(np.abs(angles_l))
                score_a_l, partial_a_l = reba.compute_score_a()
                
                # Prevent NumPy type mismatch errors
                s_r_val = float(np.max(np.array(score_a_r)))
                s_l_val = float(np.max(np.array(score_a_l)))

                # Select highest risk side
                if s_r_val >= s_l_val:
                    score_a = int(s_r_val)
                    final_body_angles = angles_r
                    final_body_partial = partial_a_r
                else:
                    score_a = int(s_l_val)
                    final_body_angles = angles_l
                    final_body_partial = partial_a_l

                assessment.score_a = score_a
                assessment.neck_score = int(final_body_partial[0])
                assessment.trunk_score = int(final_body_partial[1])
                assessment.leg_score = int(final_body_partial[2])

            if group_b_valid:
                candidates = []
                arm_angles_r = arm_angles_l = {}
                
                if readiness["right_arm_ok"]:
                    arm_angles_r = reba.get_arms_angles_from_pose_right(reba_pose)
                    reba.set_arms(arm_angles_r)
                    s_r, partial_b_r = reba.compute_score_b()

                    candidates.append((float(np.max(np.array(s_r))), abs(arm_angles_r[0]), arm_angles_r, "Right", s_r, partial_b_r))
                    
                if readiness["left_arm_ok"]:
                    arm_angles_l = reba.get_arms_angles_from_pose_left(reba_pose)
                    reba.set_arms(np.abs(arm_angles_l))
                    s_l, partial_b_l = reba.compute_score_b()
                    candidates.append((float(np.max(np.array(s_l))), abs(arm_angles_l[0]), arm_angles_l, "Left", s_l, partial_b_l))
                
                if len(candidates) > 0:
                    best_candidate = max(candidates, key=lambda item: (item[0], item[1]))
                    score_b = int(best_candidate[0])
                    assessment.score_b = score_b
                    final_arm_angles = best_candidate[2]
                    
                    calculated_arm_side = best_candidate[3]
                    winning_s_r = best_candidate[4]
                    final_arm_partial = best_candidate[5]

                    assessment.upper_arm_score = int(final_arm_partial[0])
                    assessment.lower_arm_score = int(final_arm_partial[1])
                    assessment.wrist_score = int(final_arm_partial[2])

            # Kısmi değerlendirmeye izin ver (VEYA mantığı)
            if group_a_valid or group_b_valid:
                
                # Grup A (Gövde) eksikse, minimum REBA risk puanı olan 1'i ata
                if not group_a_valid:
                    score_a = 1
                    body_side = "Unknown (Occluded)"
                else:
                    body_side = "Right" if score_a_r >= score_a_l else "Left"

                # Grup B (Kol) eksikse, minimum REBA risk puanı olan 1'i ata
                if not group_b_valid:
                    score_b = 1
                    arm_side = "Unknown (Occluded)"
                else:
                    arm_side = calculated_arm_side

                # Score C artık her halükarda hesaplanabilecek
                score_c, risk_lvl = reba.compute_score_c(score_a, score_b)
                assessment.score_c = int(score_c)
                
                # Risk seviyesi eşlemeleri (0-4)
                if score_c <= 1: risk_num = 0
                elif score_c <= 3: risk_num = 1
                elif score_c <= 7: risk_num = 2
                elif score_c <= 10: risk_num = 3
                else: risk_num = 4
                
                assessment.risk_level = risk_num

                self._print_reba_breakdown(
                    key=body.key,
                    score_a=score_a,
                    score_b=score_b,
                    score_c=score_c,
                    risk_lvl=risk_lvl,
                    completeness=completeness,
                    valid_joints=valid_joints_count,
                    body_side=body_side,
                    arm_side=arm_side,
                    body_angles=final_body_angles, # Eksik grupta boş dict {} gidecek, yazdırmada sorun çıkarmaz
                    arm_angles=final_arm_angles
                )
                    
            else:
                self.get_logger().info(f"[{body.key}] No valid groups! Assessment skipped.")

        except Exception as e:
            self.get_logger().error(f"[{body.key}] REBA calculation error: {e}")

        # Print raw coordinates
        if self.verbose:
            self._print_raw_dump(body.key, pose_matrix, completeness)

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
        arm_side,
        body_angles,
        arm_angles
    ):
        """
        Print REBA assessment dashboard.
        """
        text = f"\n"
        text += f"┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n"
        text += f"┃ REBA ASSESSMENT DASHBOARD                                      ┃\n"
        text += f"┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫\n"
        text += f"┃ Target ID : {key:<50} ┃\n"
        text += f"┃ Completeness: {completeness*100:3.0f}% ({valid_joints}/18 joints)                          ┃\n"
        text += f"┃ Final Score : {score_c:<2}  =>  {risk_lvl:<36} ┃\n"
        text += f"┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫\n"
        text += f"┃ [GROUP A: BODY] Evaluated Side: {body_side:<30} ┃\n"
        text += f"┠────────────────────────────────────────────────────────────────┨\n"

        if isinstance(body_angles, np.ndarray):
            for name, value in zip(BODY_ANGLE_NAMES, body_angles):
                val = float(value)
                text += f"┃   {name:<24}: {val:7.2f}°                           ┃\n"
        elif isinstance(body_angles, dict):
            for k, v in body_angles.items():
                if "angle" in k.lower():
                    val = float(v)
                    text += f"┃   {k:<24}: {val:7.2f}°                           ┃\n"
                else:
                    text += f"┃   {k:<24}: {v:<33} ┃\n"

        text += f"┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫\n"
        text += f"┃ [GROUP B: ARM ] Evaluated Side: {arm_side:<30} ┃\n"
        text += f"┠────────────────────────────────────────────────────────────────┨\n"

        if isinstance(arm_angles, np.ndarray):
            for name, value in zip(ARM_ANGLE_NAMES, arm_angles):
                val = float(value)
                text += f"┃   {name:<24}: {val:7.2f}°                           ┃\n"
        elif isinstance(arm_angles, dict):
            for k, v in arm_angles.items():
                if "angle" in k.lower():
                    val = float(v)
                    text += f"┃   {k:<24}: {val:7.2f}°                           ┃\n"
                else:
                    text += f"┃   {k:<24}: {v:<33} ┃\n"

        text += f"┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛\n"
        self.get_logger().info(text)

    def _print_raw_dump(self, key, pose_matrix, completeness):
        """
        Print raw X, Y, Z joint coordinates.
        """
        raw_dump = ""
        for i in range(TOTAL_JOINTS):
            pt = pose_matrix[i]
            if is_valid(pt):
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