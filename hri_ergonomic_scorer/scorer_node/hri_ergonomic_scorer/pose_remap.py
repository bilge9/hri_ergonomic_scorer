import numpy as np

# Vulcanexus COCO-18 source indices
V_NOSE = 0; V_NECK = 1
V_R_SHO = 2; V_R_ELB = 3; V_R_WRI = 4
V_L_SHO = 5; V_L_ELB = 6; V_L_WRI = 7
V_R_HIP = 8; V_R_KNE = 9; V_R_ANK = 10
V_L_HIP = 11; V_L_KNE = 12; V_L_ANK = 13

# rs9000 target index -> Vulcanexus source index
V2R = {
    # NOTE: Since there is no separate "Head" joint in the Vulcanexus COCO-18 format, the "Nose" joint is used as a proxy reference for the head.
    0: V_NOSE,   # rs9000 Head        <- Vulcanexus Nose 
    1: V_NECK,   # rs9000 Neck        <- Vulcanexus Neck 
    2: V_L_SHO,  # rs9000 L_Shoulder  <- Vulcanexus L_Shoulder
    3: V_L_ELB,  # rs9000 L_Elbow     <- Vulcanexus L_Elbow
    4: V_L_WRI,  # rs9000 L_Wrist     <- Vulcanexus L_Wrist
    5: V_R_SHO,  # rs9000 R_Shoulder  <- Vulcanexus R_Shoulder
    6: V_R_ELB,  # rs9000 R_Elbow     <- Vulcanexus R_Elbow
    7: V_R_WRI,  # rs9000 R_Wrist     <- Vulcanexus R_Wrist
    8: V_L_HIP,  # rs9000 L_Hip       <- Vulcanexus L_Hip
    9: V_L_KNE,  # rs9000 L_Knee      <- Vulcanexus L_Knee
    10: V_L_ANK, # rs9000 L_Ankle     <- Vulcanexus L_Ankle
    11: V_R_HIP, # rs9000 R_Hip       <- Vulcanexus R_Hip
    12: V_R_KNE, # rs9000 R_Knee      <- Vulcanexus R_Knee
    13: V_R_ANK, # rs9000 R_Ankle     <- Vulcanexus R_Ankle
}

REBA_JOINT_COUNT = 14  

def remap_pose_to_reba(pose_matrix: np.ndarray, valid_mask: np.ndarray, swap_y_z: bool = False, flip_y_sign: bool = False):
    """
    pose_matrix: (18,3) array, Vulcanexus COCO-18 order.
    valid_mask:  (18,) bool array.
    """
    src = pose_matrix
    if swap_y_z or flip_y_sign:
        src = pose_matrix.copy()
        if swap_y_z:
            src[:, [1, 2]] = src[:, [2, 1]]
        if flip_y_sign:
            src[:, 1] = -src[:, 1]

    reba_pose = np.zeros((REBA_JOINT_COUNT, 3))
    reba_valid = np.zeros(REBA_JOINT_COUNT, dtype=bool)
    for target_idx, source_idx in V2R.items():
        reba_pose[target_idx] = src[source_idx]
        reba_valid[target_idx] = valid_mask[source_idx]
    return reba_pose, reba_valid

def reba_inputs_are_sufficient(reba_valid: np.ndarray) -> dict:
    """
    Checks independent readiness of each body part.
    Indices are in REMAPPED rs9000 space:
    0: Head, 1: Neck, 2: L_Shoulder, 3: L_Elbow, 4: L_Wrist
    5: R_Shoulder, 6: R_Elbow, 7: R_Wrist, 8: L_Hip, 9: L_Knee, 10: L_Ankle
    11: R_Hip, 12: R_Knee, 13: R_Ankle
    """
    mid_hip_ok = reba_valid[8] and reba_valid[11]
    shoulder_axis_ok = reba_valid[2] and reba_valid[5]

    trunk_ok = reba_valid[1] and mid_hip_ok
    neck_ok = reba_valid[0] and reba_valid[1] and mid_hip_ok and shoulder_axis_ok

    left_leg_ok = reba_valid[8] and reba_valid[9] and reba_valid[10]
    right_leg_ok = reba_valid[11] and reba_valid[12] and reba_valid[13]

    left_upper_arm_ok = reba_valid[2] and reba_valid[3] and shoulder_axis_ok
    left_lower_arm_ok = reba_valid[3] and reba_valid[4] and reba_valid[2]
    
    right_upper_arm_ok = reba_valid[5] and reba_valid[6] and shoulder_axis_ok
    right_lower_arm_ok = reba_valid[6] and reba_valid[7] and reba_valid[5]

    return {
        "trunk_ok": trunk_ok,
        "neck_ok": neck_ok,
        "left_leg_ok": left_leg_ok,
        "right_leg_ok": right_leg_ok,
        "left_upper_arm_ok": left_upper_arm_ok,
        "left_lower_arm_ok": left_lower_arm_ok,
        "right_upper_arm_ok": right_upper_arm_ok,
        "right_lower_arm_ok": right_lower_arm_ok,
        "body_group_ok": trunk_ok or neck_ok or left_leg_ok or right_leg_ok,
        "left_arm_ok": left_upper_arm_ok or left_lower_arm_ok,
        "right_arm_ok": right_upper_arm_ok or right_lower_arm_ok
    }

def remap_scalar_to_reba(scalar_array: np.ndarray) -> np.ndarray:
    """
    scalar_array: (18,) per-joint scalar (e.g. confidence), Vulcanexus order.
    Applies the same V2R index mapping as remap_pose_to_reba, but for a
    single scalar per joint instead of an (x,y,z) triplet.
    """
    reba_scalar = np.zeros(REBA_JOINT_COUNT)
    for target_idx, source_idx in V2R.items():
        reba_scalar[target_idx] = scalar_array[source_idx]
    return reba_scalar


def reba_region_confidence(reba_confidence: np.ndarray) -> dict:
    """
    Average confidence backing each REBA region's angle computation.
    NOT a validity check (see reba_inputs_are_sufficient for that) - this is
    a reliability readout so the dashboard can show "this angle was computed
    from joints with X% average confidence" next to the angle itself.

    Index legend (REMAPPED rs9000 space):
    0: Head, 1: Neck, 2: L_Shoulder, 3: L_Elbow, 4: L_Wrist
    5: R_Shoulder, 6: R_Elbow, 7: R_Wrist, 8: L_Hip, 9: L_Knee, 10: L_Ankle
    11: R_Hip, 12: R_Knee, 13: R_Ankle
    """
    def avg(indices):
        vals = [reba_confidence[i] for i in indices]
        return float(np.mean(vals)) if vals else 0.0

    return {
        "neck_conf": avg([0, 1, 2, 5]),
        "trunk_conf": avg([1, 8, 11]),
        "left_leg_conf": avg([8, 9, 10]),
        "right_leg_conf": avg([11, 12, 13]),
        "left_upper_arm_conf": avg([2, 3, 5]),
        "left_lower_arm_conf": avg([2, 3, 4]),
        "right_upper_arm_conf": avg([5, 6, 2]),
        "right_lower_arm_conf": avg([5, 6, 7]),
    }