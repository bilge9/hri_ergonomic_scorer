import numpy as np

# Vulcanexus COCO-18 source indices
V_NOSE = 0; V_NECK = 1
V_R_SHO = 2; V_R_ELB = 3; V_R_WRI = 4
V_L_SHO = 5; V_L_ELB = 6; V_L_WRI = 7
V_R_HIP = 8; V_R_KNE = 9; V_R_ANK = 10
V_L_HIP = 11; V_L_KNE = 12; V_L_ANK = 13

# rs9000 target index -> Vulcanexus source index
# (verified against reba.py's actual class docstring + rotate_pose usage,
# NOT just the README -- see get_body_angles_from_pose_right: uses pose[:,8]/
# pose[:,11] as hips, pose[:,1] as Neck, pose[:,0] as Head)
V2R = {
    0: V_NOSE,   # rs9000 Head        <- Vulcanexus Nose (closest available proxy, no exact "head" joint)
    1: V_NECK,   # rs9000 Neck        <- Vulcanexus Neck (exact match)
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

REBA_JOINT_COUNT = 14  # no hand data from Vulcanexus -> exactly 14, not 16
                        # (reba.py's own `if pose.shape[1] > 14:` guard then
                        # safely defaults wrist_angle/wrist_twisted to 0
                        # instead of us feeding it a fake (0,0,0) hand point)

# Set this to True ONLY as a controlled experiment, one variable at a time,
# after confirming the joint-index remap alone doesn't fix trunk_angle.
# Evidence from real bag data (Y already encodes correct top-to-bottom
# anatomical order: eyes/ears smallest Y, ankles largest Y) suggests this
# should stay False.
SWAP_Y_Z = False

# Evidence from real test logs: neck_angle/upper_arm_angle (RELATIVE angles,
# which internally subtract/add trunk_angle) come out physically sane, while
# trunk_angle/legs_angle (ABSOLUTE angles measured against reba.py's assumed
# vertical) sit stuck around +-175-180 deg for an upright person. That's the
# signature of reba.py assuming +Y = up, while our data has Y increasing
# downward (optical camera frame). Flipping Y's SIGN (not swapping with Z)
# should fix this while keeping Y as the correct vertical axis.
FLIP_Y_SIGN = True


def remap_pose_to_reba(pose_matrix: np.ndarray, valid_mask: np.ndarray):
    """
    pose_matrix: (18,3) array, Vulcanexus COCO-18 order.
    valid_mask:  (18,) bool array.

    Returns:
      reba_pose:  (14,3) array, rs9000 order.
      reba_valid: (14,) bool array, same remapping applied.
    """
    src = pose_matrix
    if SWAP_Y_Z or FLIP_Y_SIGN:
        src = pose_matrix.copy()
        if SWAP_Y_Z:
            src[:, [1, 2]] = src[:, [2, 1]]
        if FLIP_Y_SIGN:
            src[:, 1] = -src[:, 1]

    reba_pose = np.zeros((REBA_JOINT_COUNT, 3))
    reba_valid = np.zeros(REBA_JOINT_COUNT, dtype=bool)
    for target_idx, source_idx in V2R.items():
        reba_pose[target_idx] = src[source_idx]
        reba_valid[target_idx] = valid_mask[source_idx]
    return reba_pose, reba_valid


def reba_inputs_are_sufficient(reba_valid: np.ndarray) -> dict:
    """
    NOTE: indices here are in REMAPPED rs9000 space (post remap_pose_to_reba),
    not raw Vulcanexus indices -- 8/11 are hips in rs9000 space.
    """
    hips_ok = reba_valid[8] and reba_valid[11]
    trunk_neck_ok = hips_ok and reba_valid[0] and reba_valid[1]

    # Tolerant per your latest change: legs aren't required for body_group_ok,
    # missing legs just means reba.py's own defaults (0) apply to legs_angle.
    body_group_ok = trunk_neck_ok

    left_arm_ok = reba_valid[2] and (reba_valid[3] or reba_valid[4])
    right_arm_ok = reba_valid[5] and (reba_valid[6] or reba_valid[7])

    return {
        "body_group_ok": body_group_ok,
        "left_arm_ok": left_arm_ok,
        "right_arm_ok": right_arm_ok,
    }