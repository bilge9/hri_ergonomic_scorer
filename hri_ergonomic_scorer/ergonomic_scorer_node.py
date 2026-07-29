import rclpy
from rclpy.node import Node
import numpy as np
import math
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from hri_msgs.msg import Skeleton3DList
from hri_ergonomic_msgs.msg import RebaAssessment, RebaAssessmentList

from .reba import RebaScore
from .pose_remap import remap_pose_to_reba, reba_inputs_are_sufficient, remap_scalar_to_reba, reba_region_confidence

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
    # 1. Confidence kontrolü
    if hasattr(pt_obj, 'confidence') and pt_obj.confidence < threshold:
        return False
        
    pt = [pt_obj.x, pt_obj.y, pt_obj.z]
    
    # 2. NaN (Not a Number) kontrolü
    if any(math.isnan(v) for v in pt):
        return False
        
    # 3. Z-outlier veya sıfır noktası kontrolü
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
        
        # True olan değeri False yapıyoruz ki kalabalık koordinatlar ekrana basılmasın
        self.declare_parameter('verbose_logging', False)
        self.verbose = self.get_parameter('verbose_logging').get_parameter_value().bool_value

        self.declare_parameter('confidence_threshold', 0.4)
        self.confidence_threshold = self.get_parameter('confidence_threshold').get_parameter_value().double_value

        # Çıktı hızını kontrol etmek için yeni bir sözlük ekliyoruz
        self.last_print_times = {}

        self.get_logger().info(f"Parameter changed. Threshold: {self.threshold}")
        self.get_logger().info("Ergonomic Scorer Node started. Granular partial REBA enabled.")

    def skeleton_callback(self, msg):
        active_bodies = []
        for i, s in enumerate(getattr(msg, 'skeletons', [])):
            valid_joint_count = 0
            for pt_obj in s.skeleton:
                # Eşik değerini node'un parametresinden alıyoruz
                if is_valid(pt_obj, self.confidence_threshold):
                    valid_joint_count += 1
            
            # Hayalet iskeletleri (örneğin sadece tek bir diz veya burun algılaması) yoksay.
            # Bir iskeletin değerlendirmeye alınması için en az 5 geçerli eklemi olmalı.
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
        confidence_matrix = np.zeros(TOTAL_JOINTS)   # <-- yeni
        valid_mask = np.zeros(TOTAL_JOINTS, dtype=bool)
        valid_joints_count = 0

        # Vücudun genel güvenilirlik skorunu al ve 0.0 - 1.0 aralığına normalize et
        body_overall_conf = getattr(body, 'confidence', 0.0)
        if body_overall_conf > 1.0:
            body_overall_conf /= 100.0

        num_joints = min(len(body.skeleton), TOTAL_JOINTS)
        for i in range(num_joints):
            pt_obj = body.skeleton[i]
            
            # Eklemin kendi puanı yoksa vücudun genel puanını (body_overall_conf) kullan
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

        reba_pose, reba_valid = remap_pose_to_reba(pose_matrix, valid_mask)
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
            body_side = "Unknown"
            
            calc_angles_r = reba.get_body_angles_from_pose_right(reba_pose)
            calc_angles_l = reba.get_body_angles_from_pose_left(reba_pose)

            # Neck
            if readiness["neck_ok"]:
                body_angles[0] = calc_angles_r[0]
                body_angles[1] = calc_angles_r[1]
                body_side = "Right" 
                
            # Trunk
            if readiness["trunk_ok"]:
                body_angles[2] = calc_angles_r[2]
                body_angles[3] = calc_angles_r[3]
                body_side = "Right"

            # Legs (Choose the leg that is valid, if both are valid choose highest risk angle)
            if readiness["right_leg_ok"] and readiness["left_leg_ok"]:
                if abs(calc_angles_r[5]) >= abs(calc_angles_l[5]):
                    body_angles[4] = calc_angles_r[4]
                    body_angles[5] = calc_angles_r[5]
                    body_side = "Right"
                else:
                    body_angles[4] = calc_angles_l[4] 
                    body_angles[5] = calc_angles_l[5] 
                    body_side = "Left"
            elif readiness["right_leg_ok"]:
                body_angles[4] = calc_angles_r[4] 
                body_angles[5] = calc_angles_r[5] 
                body_side = "Right"
            elif readiness["left_leg_ok"]:
                body_angles[4] = calc_angles_l[4] 
                body_angles[5] = calc_angles_l[5] 
                body_side = "Left"

            body_conf = np.zeros(7)
            if readiness["neck_ok"]:
                body_conf[0] = body_conf[1] = region_conf["neck_conf"]
            if readiness["trunk_ok"]:
                body_conf[2] = body_conf[3] = region_conf["trunk_conf"]
            if body_side == "Right":
                body_conf[4] = body_conf[5] = region_conf["right_leg_conf"]
            elif body_side == "Left":
                body_conf[4] = body_conf[5] = region_conf["left_leg_conf"]
            reba.set_body(body_angles)
            score_a, partial_a = reba.compute_score_a()
            
            assessment.group_a_valid = True if body_side != "Unknown" else False
            assessment.score_a = int(score_a)
            assessment.neck_score = int(partial_a[0])
            assessment.trunk_score = int(partial_a[1])
            assessment.leg_score = int(partial_a[2])
            assessment.body_confidence = body_conf.tolist()

            # 2. GROUP B (ARM) CALCULATION - Independent Parts
            arm_angles = np.zeros(7)
            arm_side = "Unknown"
            
            calc_arm_r = reba.get_arms_angles_from_pose_right(reba_pose)
            calc_arm_l = reba.get_arms_angles_from_pose_left(reba_pose)
            
            eval_r_upper = calc_arm_r[0] if readiness["right_upper_arm_ok"] else 0
            eval_l_upper = calc_arm_l[0] if readiness["left_upper_arm_ok"] else 0
            
            if readiness["right_upper_arm_ok"] or readiness["right_lower_arm_ok"]:
                # We will process Right Arm
                if readiness["right_upper_arm_ok"]:
                    arm_angles[0] = calc_arm_r[0]
                    arm_angles[1] = calc_arm_r[1]
                    arm_angles[2] = calc_arm_r[2]
                    arm_angles[3] = calc_arm_r[3]
                if readiness["right_lower_arm_ok"]:
                    arm_angles[4] = calc_arm_r[4]
                arm_side = "Right"

            elif readiness["left_upper_arm_ok"] or readiness["left_lower_arm_ok"]:
                # We will process Left Arm
                if readiness["left_upper_arm_ok"]:
                    arm_angles[0] = calc_arm_l[0]
                    arm_angles[1] = calc_arm_l[1]
                    arm_angles[2] = calc_arm_l[2]
                    arm_angles[3] = calc_arm_l[3]
                if readiness["left_lower_arm_ok"]:
                    arm_angles[4] = calc_arm_l[4]
                arm_side = "Left"

            arm_conf = np.zeros(7)
            if arm_side == "Right":
                arm_conf[0] = arm_conf[1] = arm_conf[2] = arm_conf[3] = region_conf["right_upper_arm_conf"]
                arm_conf[4] = region_conf["right_lower_arm_conf"]
            elif arm_side == "Left":
                arm_conf[0] = arm_conf[1] = arm_conf[2] = arm_conf[3] = region_conf["left_upper_arm_conf"]
                arm_conf[4] = region_conf["left_lower_arm_conf"]
            reba.set_arms(arm_angles)
            score_b, partial_b = reba.compute_score_b()

            assessment.group_b_valid = True if arm_side != "Unknown" else False
            assessment.score_b = int(score_b)
            assessment.upper_arm_score = int(partial_b[0])
            assessment.lower_arm_score = int(partial_b[1])
            assessment.wrist_score = int(partial_b[2])
            assessment.arm_confidence = arm_conf.tolist()

            # 3. FINAL SCORE (C) CALCULATION
            # Even if a group is completely occluded, it defaults to a minimum risk score of 1.
            # 3. FINAL SCORE (C) CALCULATION
            # Even if a group is completely occluded, it defaults to a minimum risk score of 1.
            score_c, risk_lvl = reba.compute_score_c(score_a, score_b)
            assessment.score_c = int(score_c)
            
            if score_c <= 1: risk_num = 0
            elif score_c <= 3: risk_num = 1
            elif score_c <= 7: risk_num = 2
            elif score_c <= 10: risk_num = 3
            else: risk_num = 4
            
            assessment.risk_level = risk_num

            # --- EKRAN ÇIKTISINI YAVAŞLATMA (THROTTLE) ---
            # Zamanı saniye cinsinden alıyoruz
            current_time = self.get_clock().now().nanoseconds / 1e9
            last_time = self.last_print_times.get(body.key, 0.0)
            
            # Sadece 0.5 saniyede bir (saniyede 2 kez) ekrana yazdırmaya izin ver
            if current_time - last_time > 0.5:
                # Dashboard Print
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
                    body_angles=body_angles,
                    arm_angles=arm_angles,
                    body_conf=body_conf,
                    arm_conf=arm_conf
                )
                
                # Eğer parametrelerden verbose_logging tekrar True yapılırsa ham koordinatları da yazdırır
                if self.verbose:
                    self._print_raw_dump(body.key, pose_matrix, completeness)
                    
                # Son yazdırma zamanını kaydet
                self.last_print_times[body.key] = current_time

        except Exception as e:
            self.get_logger().error(f"[{body.key}] REBA calculation error: {e}")

        # Not: Eski koddaki if self.verbose bloğu buradan silinip yukarıdaki zamanlayıcının içine taşındı.

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
        arm_angles,
        body_conf, 
        arm_conf
    ):
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
            # NumPy dizisi olduğu için is_valid yerine doğrudan sıfır noktası kontrolü yapıyoruz
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