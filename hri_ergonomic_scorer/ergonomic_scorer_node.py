import rclpy
from rclpy.node import Node
import numpy as np
import math
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from .reba import RebaScore

# will confirm the exact import path after the ROSbag files
from hri_msgs.msg import Skeleton3D, Skeleton3DList

# Import the original rs9000 REBA algorithm we migrated into our package.
from .reba import RebaScore

def calculate_angle(p1, p2, p3):
    """
    Calculates the angle in degrees between three points. p2 is the vertex. Format: [x, y, z]
    """
    ux, uy, uz = p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]
    vx, vy, vz = p3[0] - p2[0], p3[1] - p2[1], p3[2] - p2[2]
    
    dot_product = ux*vx + uy*vy + uz*vz
    mag_u = math.sqrt(ux**2 + uy**2 + uz**2)
    mag_v = math.sqrt(vx**2 + vy**2 + vz**2)
    
    if mag_u == 0 or mag_v == 0:
        return None
        
    cos_theta = dot_product / (mag_u * mag_v)
    cos_theta = max(-1.0, min(1.0, cos_theta))
    
    angle_rad = math.acos(cos_theta)
    return math.degrees(angle_rad)

def is_valid(pt):
    if math.isnan(pt[0]) or math.isnan(pt[1]) or math.isnan(pt[2]):
        return False
    
    if pt[0] == 0.0 and pt[1] == 0.0 and pt[2] == 0.0:
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

        # REBA risk threshold parameter.
        self.declare_parameter('high_risk_threshold', 8)

        threshold = self.get_parameter('high_risk_threshold').get_parameter_value().integer_value
        self.get_logger().info(f"Parameter changed. Threshold: {threshold}")

        self.get_logger().info("Ergonomic Scorer Node started. Waiting for 3D Skeleton data...")

    def skeleton_callback(self, msg):
        """
        Parses the incoming Skeleton3DList message and constructs the 
        REBA pose matrix for the first detected human with valid data.
        """
        # Extract the list of skeletons from the message
        human_list = getattr(msg, 'skeletons', None)
        
        # Fallback in case the message structure directly provides a list
        if human_list is None:
            human_list = msg if isinstance(msg, list) else [msg]

        # Ignore empty frames
        if not human_list or len(human_list) == 0:
            return

        # Target the first detected human in the frame
        # TODO: Implement ID tracking (msg.key) for multi-human scenarios
        target_human = human_list[0]

        # Initialize the 18x3 matrix
        pose_matrix = np.zeros((18, 3))
        valid_joints_count = 0

        # Populate the matrix, handling missing (NaN) data
        num_joints = min(len(target_human.skeleton),18)
        for i in range(num_joints):
            pt = [target_human.skeleton[i].x, target_human.skeleton[i].y, target_human.skeleton[i].z]
            if is_valid(pt):
                pose_matrix[i] = pt
                valid_joints_count += 1
            else:
                pose_matrix[i] = [0.0, 0.0, 0.0]

        completeness_score = (valid_joints_count / 18.0) * 100
        
        if valid_joints_count == 0:
            return

        nose = pose_matrix[0]
        neck = pose_matrix[1]
        
        r_sho, r_elb, r_wri = pose_matrix[2], pose_matrix[3], pose_matrix[4]
        l_sho, l_elb, l_wri = pose_matrix[5], pose_matrix[6], pose_matrix[7]
        
        r_hip, r_knee, r_ank = pose_matrix[8], pose_matrix[9], pose_matrix[10]
        l_hip, l_knee, l_ank = pose_matrix[11], pose_matrix[12], pose_matrix[13]

        trunk_angle, neck_angle = None, None
        r_knee_angle, l_knee_angle = None, None
        r_upper_arm_angle, l_upper_arm_angle = None, None
        r_lower_arm_angle, l_lower_arm_angle = None, None

        mid_hip = [
            (r_hip[0] + l_hip[0]) / 2.0,
            (r_hip[1] + l_hip[1]) / 2.0,
            (r_hip[2] + l_hip[2]) / 2.0
        ]
        
        # 1. Trunk Flexion
        if is_valid(neck) and is_valid(r_hip) and is_valid(l_hip):
            mid_hip = [(r_hip[0] + l_hip[0]) / 2.0, (r_hip[1] + l_hip[1]) / 2.0, (r_hip[2] + l_hip[2]) / 2.0]
            vertical_ref_pt = [mid_hip[0], mid_hip[1], mid_hip[2] + 1.0]
            trunk_angle = calculate_angle(neck, mid_hip, vertical_ref_pt)
        
        # 2. Neck Flexion
        if is_valid(nose) and is_valid(neck):
            neck_vertical_ref = [neck[0], neck[1], neck[2] + 1.0]
            neck_angle = calculate_angle(nose, neck, neck_vertical_ref)

        # 3. Knees
        if is_valid(r_hip) and is_valid(r_knee) and is_valid(r_ank):
            r_knee_angle = calculate_angle(r_hip, r_knee, r_ank)
        if is_valid(l_hip) and is_valid(l_knee) and is_valid(l_ank):
            l_knee_angle = calculate_angle(l_hip, l_knee, l_ank)

        # 4. Upper Arms
        if is_valid(r_sho) and is_valid(r_elb):
            r_hip_ref = [r_sho[0], r_sho[1], r_sho[2] - 1.0] 
            r_upper_arm_angle = calculate_angle(r_elb, r_sho, r_hip_ref)
        if is_valid(l_sho) and is_valid(l_elb):
            l_hip_ref = [l_sho[0], l_sho[1], l_sho[2] - 1.0]
            l_upper_arm_angle = calculate_angle(l_elb, l_sho, l_hip_ref)

        # 5. Lower Arms
        if is_valid(r_sho) and is_valid(r_elb) and is_valid(r_wri):
            r_lower_arm_angle = calculate_angle(r_sho, r_elb, r_wri)
        if is_valid(l_sho) and is_valid(l_elb) and is_valid(l_wri):
            l_lower_arm_angle = calculate_angle(l_sho, l_elb, l_wri)

        def fmt(angle): return f"{angle:.1f}°" if angle is not None else "Unknown"

        def fmt(angle): return f"{angle:.1f}°" if angle is not None else "Unknown"

        # Vulcanexus (COCO-18) Uzuv İsimleri
        joint_names = [
            "Nose", "Neck", "R_Shoulder", "R_Elbow", "R_Wrist",
            "L_Shoulder", "L_Elbow", "L_Wrist", "R_Hip", "R_Knee",
            "R_Ankle", "L_Hip", "L_Knee", "L_Ankle", "R_Eye", "L_Eye",
            "R_Ear", "L_Ear"
        ]

        # 18 Uzvun ham X, Y, Z verilerini string olarak birleştir
        raw_dump = ""
        for i in range(18):
            pt = pose_matrix[i]
            if is_valid(pt):
                raw_dump += f"  {joint_names[i]:<12}: [X: {pt[0]:.2f}, Y: {pt[1]:.2f}, Z: {pt[2]:.2f}]\n"
            else:
                raw_dump += f"  {joint_names[i]:<12}: MISSING\n"

        # Hem Ham Verileri hem de REBA açılarını tek bir dev log ekranında fırlat
        self.get_logger().info(
            f"\n==========================================================\n"
            f"--- RAW JOINT COORDINATES (Vulcanexus COCO-18) ---\n"
            f"{raw_dump}"
            f"--- FULL BODY ERGONOMIC REPORT (Completeness: {completeness_score:.0f}%) ---\n"
            f"[Group A] Trunk: {fmt(trunk_angle)} | Neck: {fmt(neck_angle)}\n"
            f"          R. Knee: {fmt(r_knee_angle)} | L. Knee: {fmt(l_knee_angle)}\n"
            f"[Group B] R. Upper Arm: {fmt(r_upper_arm_angle)} | L. Upper Arm: {fmt(l_upper_arm_angle)}\n"
            f"          R. Lower Arm: {fmt(r_lower_arm_angle)} | L. Lower Arm: {fmt(l_lower_arm_angle)}\n"
            f"=========================================================="
        )

        try:
            reba = RebaScore()
            body_params = reba.get_body_angles_from_pose_right(pose_matrix) 
            arms_params = reba.get_arms_angles_from_pose_left(pose_matrix) 

            reba.set_body(body_params)
            score_a, _ = reba.compute_score_a()

            reba.set_arms(arms_params)
            score_b, _ = reba.compute_score_b()

            final_score, risk_level = reba.compute_score_c(score_a, score_b)

            self.get_logger().info(f"REBA FINAL RISK: {final_score} - {risk_level}")
        except Exception as e:
            self.get_logger().error(f"REBA calculation error: {e}")

def main(args=None):

    rclpy.init(args=args)

    # Create the ergnomic scorer node
    node = ErgonomicScorerNode()

    try:
        # Keep the node running.
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