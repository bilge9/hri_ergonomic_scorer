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
    Check if the given 3D point is valid.
    Returns False if any coordinate is NaN or if all coordinates are exactly zero.
    """
    if any(math.isnan(v) for v in pt):
        return False
    if all(abs(v) < 1e-6 for v in pt):
        return False
    return True

class ErgonomicScorerNode(Node):
    def __init__(self):
        super().__init__('ergonomic_scorer_node')

        # Define QoS profile suitable for reliable sensor data transmission
        bag_qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Subscriber for incoming 3D skeleton data
        self.subscription = self.create_subscription(
            Skeleton3DList,
            '/humans/bodies/skel3D',
            self.skeleton_callback,
            bag_qos_profile
        )

        # Publisher for the calculated REBA ergonomic assessments
        self.reba_pub = self.create_publisher(
            RebaAssessmentList,
            '/humans/bodies/ergonomics/reba',
            bag_qos_profile
        )

        # Node parameters for risk thresholds and logging verbosity 
        self.declare_parameter('high_risk_threshold', 8)
        self.threshold = self.get_parameter('high_risk_threshold').get_parameter_value().integer_value
        
        self.declare_parameter('verbose_logging', True)
        self.verbose = self.get_parameter('verbose_logging').get_parameter_value().bool_value

        self.get_logger().info(f"Parameter changed. Threshold: {self.threshold}")
        self.get_logger().info("Ergonomic Scorer Node started. Waiting for 3D Skeleton data...")

    def skeleton_callback(self, msg):
        """
        Callback triggered whenever new skeleton data is received.
        Filters valid bodies and triggers the REBA assessment process.
        """
        active_bodies = []
        
        # Filter skeletons that have at least one valid joint
        for i, s in enumerate(getattr(msg, 'skeletons', [])):
            has_valid_data = False
            for pt_obj in s.skeleton:
                pt = [pt_obj.x, pt_obj.y, pt_obj.z]
                if is_valid(pt):
                    has_valid_data = True
                    break
            
            if has_valid_data:
                # Assign a temporary ID if the tracking system did not provide one
                if not s.key or s.key.strip() == "":
                    s.key = f"human_untracked_{i}"
                active_bodies.append(s)
        
        if not active_bodies:
            return

        out_list = RebaAssessmentList()
        out_list.header = msg.header

        out_index = 0
        for body in active_bodies:
            # Enforce the IDL limit of a maximum of 10 tracked bodies
            if out_index >= 10:
                break
            assessment = self._assess_body(body)
            if assessment is not None:
                # Overwrite the pre-allocated slots instead of using append()
                out_list.assessments[out_index] = assessment
                out_index += 1
        # Publish the array if at least one assessment was successfully generated
        if out_index > 0:
            self.reba_pub.publish(out_list)

    def _assess_body(self, body):
        """
        Core logic to convert raw 3D coordinates into a structured REBA assessment.
        Handles coordinate normalization, side-selection, and message population.
        """
        pose_matrix = np.zeros((TOTAL_JOINTS, 3))
        valid_mask = np.zeros(TOTAL_JOINTS, dtype=bool)
        valid_joints_count = 0
        # Extract joints into a NumPy array for easier mathematical operations
        num_joints = min(len(body.skeleton), TOTAL_JOINTS)
        for i in range(num_joints):
            pt = [body.skeleton[i].x, body.skeleton[i].y, body.skeleton[i].z]
            if is_valid(pt):
                pose_matrix[i] = pt
                valid_mask[i] = True
                valid_joints_count += 1

        completeness = valid_joints_count / float(TOTAL_JOINTS)
        if valid_joints_count == 0:
            return None

        reba_pose, reba_valid = remap_pose_to_reba(pose_matrix, valid_mask)
        
        # Compensate for camera tilt: Ensure the skeleton is vertically aligned
        # by treating the shoulder line as the horizontal reference.
        shoulder_center_y = (pose_matrix[2, 1] + pose_matrix[5, 1]) / 2
        pose_matrix[:, 1] -= shoulder_center_y
        
        reba_pose, reba_valid = remap_pose_to_reba(pose_matrix, valid_mask)
        readiness = reba_inputs_are_sufficient(reba_valid)

        group_a_valid = readiness["body_group_ok"]
        group_b_valid = readiness["left_arm_ok"] or readiness["right_arm_ok"]

        # Initialize the custom ROS 2 message
        assessment = RebaAssessment()
        assessment.key = body.key
        assessment.completeness = float(completeness)
        assessment.group_a_valid = bool(group_a_valid)
        assessment.group_b_valid = bool(group_b_valid)

        try:
            reba = RebaScore()
            score_a = score_b = 0
            
            # Dictionaries to store calculated angles for terminal output
            final_body_angles = {}
            final_arm_angles = {}

            if group_a_valid:
                # Calculate body scores for both right and left sides
                angles_r = reba.get_body_angles_from_pose_right(reba_pose)
                reba.set_body(angles_r)
                score_a_r, _ = reba.compute_score_a()
                
                angles_l = reba.get_body_angles_from_pose_left(reba_pose)
                reba.set_body(angles_l)
                score_a_l, _ = reba.compute_score_a()
                
                # Digitize values to prevent NumPy type mismatch errors in ROS 2
                s_r_val = float(np.max(np.array(score_a_r)))
                s_l_val = float(np.max(np.array(score_a_l)))

                # Select the side representing the highest risk
                if s_r_val >= s_l_val:
                    score_a = int(s_r_val)
                    final_body_angles = angles_r
                else:
                    score_a = int(s_l_val)
                    final_body_angles = angles_l

                assessment.score_a = score_a

            if group_b_valid:
                candidates = []
                arm_angles_r = arm_angles_l = {}
                
                if readiness["right_arm_ok"]:
                    arm_angles_r = reba.get_arms_angles_from_pose_right(reba_pose)
                    reba.set_arms(arm_angles_r)
                    s_r, _ = reba.compute_score_b()

                    candidates.append((float(np.max(np.array(s_r))), abs(arm_angles_r[0]), arm_angles_r, "Right", s_r))
                    
                if readiness["left_arm_ok"]:
                    arm_angles_l = reba.get_arms_angles_from_pose_left(reba_pose)
                    reba.set_arms(arm_angles_l)
                    s_l, _ = reba.compute_score_b()
                    candidates.append((float(np.max(np.array(s_l))), abs(arm_angles_l[0]), arm_angles_l, "Left", s_l))
                
                if len(candidates) > 0:
                    best_candidate = max(candidates, key=lambda item: (item[0], item[1]))
                    score_b = int(best_candidate[0])
                    assessment.score_b = score_b
                    final_arm_angles = best_candidate[2]
                    
                    calculated_arm_side = best_candidate[3]
                    winning_s_r = best_candidate[4]

            if group_a_valid and group_b_valid:
                score_c, risk_lvl = reba.compute_score_c(score_a, score_b)
                # Both body and arms are valid; calculate the grand final Score C
                assessment.score_c = int(score_c)
                
                # Map Score C to IDL 'uint8' risk levels (0-4)
                if score_c <= 1: risk_num = 0
                elif score_c <= 3: risk_num = 1
                elif score_c <= 7: risk_num = 2
                elif score_c <= 10: risk_num = 3
                else: risk_num = 4
                
                assessment.risk_level = risk_num

                body_side = "Right" if score_a_r >= score_a_l else "Left"
                
                arm_side = calculated_arm_side

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
                    body_angles=final_body_angles,
                    arm_angles=final_arm_angles
                )
                    
            else:
                self.get_logger().info(f"[{body.key}] Incomplete skeleton! Assessment skipped. Group A Valid: {group_a_valid}, Group B Valid: {group_b_valid}")

        except Exception as e:
            self.get_logger().error(f"[{body.key}] REBA calculation error: {e}")

        # Dump raw coordinates to terminal if verbose logging is enabled
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
        Prints a highly formatted, readable dashboard for REBA assessments.
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
        Prints the raw X, Y, Z joint coordinates for debugging.
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