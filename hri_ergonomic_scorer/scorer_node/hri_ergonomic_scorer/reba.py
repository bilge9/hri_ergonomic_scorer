# -*- coding: utf-8 -*-
# ---------------------

import numpy as np

# Sentinel published when a REBA term could not be measured at all. Kept here
# so reba.py, the ROS node and the message consumers agree on a single value.
# 255 is chosen because every score field in RebaAssessment.idl is a uint8 and
# no real REBA score can reach it.
SCORE_NOT_ASSESSED = 255

# rs9000 joint indices used throughout this module:
#   0 Head, 1 Neck, 2 L_Shoulder, 3 L_Elbow, 4 L_Wrist,
#   5 R_Shoulder, 6 R_Elbow, 7 R_Wrist,
#   8 L_Hip, 9 L_Knee, 10 L_Ankle, 11 R_Hip, 12 R_Knee, 13 R_Ankle
WORLD_UP = np.array([0.0, 1.0, 0.0])

# Vertical ankle separation above which one foot counts as raised. See
# _body_angles() for why this replaced the old horizontal-distance test.
FOOT_RAISED_M = 0.15

# Angular tolerances for the "side bending / twisted" REBA modifiers.
SIDE_BEND_DEG = 10.0

# Dead band around 0 deg of neck flexion. The neck angle comes from atan2, so a
# physically neutral neck lands on a tiny negative float; without this an
# anatomically upright neck was scored as extension (2) roughly half the time.
# Mirrors the 1 deg dead band the trunk item already had.
NECK_NEUTRAL_DEG = 1.0
ABDUCTION_DEG = 45.0
SHOULDER_RAISED_M = 0.02


def _unit(v):
    return v / (np.linalg.norm(v) + 1e-9)


def _trunk_frame(pose):
    """
    Body-fixed orthonormal frame built by Gram-Schmidt:
      e_up   along the trunk (mid-hip -> neck)
      e_lat  across the shoulders, orthogonalised against e_up
      e_fwd  = e_lat x e_up

    Decomposing a limb vector in this frame separates flexion from lateral
    bend and, unlike a world-aligned frame, does not depend on how the camera
    is mounted.

    Returns (trunk_vec, e_up, e_lat, e_fwd, frame_ok). frame_ok is False when
    the trunk/shoulder joints are missing (zero-filled); callers then get a
    world-aligned fallback frame so the maths still produces finite numbers,
    but they can tell the frame is not body-fixed.
    """
    mid_hip = (pose[8] + pose[11]) / 2.0
    trunk_vec = pose[1] - mid_hip
    lateral_axis = pose[2] - pose[5]

    if np.linalg.norm(trunk_vec) < 1e-6 or np.linalg.norm(lateral_axis) < 1e-6:
        e_up = WORLD_UP
        e_lat = np.array([1.0, 0.0, 0.0])
        e_fwd = np.cross(e_lat, e_up)
        return trunk_vec, e_up, e_lat, e_fwd, False

    e_up = _unit(trunk_vec)
    lat_perp = lateral_axis - np.dot(lateral_axis, e_up) * e_up
    e_lat = _unit(lat_perp)
    e_fwd = _unit(np.cross(e_lat, e_up))
    return trunk_vec, e_up, e_lat, e_fwd, True


class RebaScore:
    """
    Class to compute REBA metrics.
    """
    def __init__(self):
        self.table_a = np.zeros((3, 5, 4))
        self.table_b = np.zeros((6, 2, 3))
        self.table_c = np.zeros((12, 12))

        self.body = {'neck_angle': 0, 'neck_side': False,
                     'trunk_angle': 0, 'trunk_side': False,
                     'legs_unstable': False, 'legs_angle': 0,
                     'load': 0}

        self.arms = {'upper_arm_angle': 0, 'shoulder_raised': False, 'arm_abducted': False, 'leaning': False,
                     'lower_arm_angle': 0,
                     'wrist_angle': 0, 'wrist_twisted': False,
                     'coupling_score': 0}

        self.init_table_a()
        self.init_table_b()
        self.init_table_c()

    def init_table_a(self):
        self.table_a = np.array([
                                [[1, 2, 3, 4], [2, 3, 4, 5], [2, 4, 5, 6], [3, 5, 6, 7], [4, 6, 7, 8]],
                                [[1, 2, 3, 4], [3, 4, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8], [6, 7, 8, 9]],
                                [[3, 3, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8], [6, 7, 8, 9], [7, 8, 9, 9]]
                                ])

    def init_table_b(self):
        self.table_b = np.array([
                                [[1, 2, 2], [1, 2, 3]],
                                [[1, 2, 3], [2, 3, 4]],
                                [[3, 4, 5], [4, 5, 5]],
                                [[4, 5, 5], [5, 6, 7]],
                                [[6, 7, 8], [7, 8, 8]],
                                [[7, 8, 8], [8, 9, 9]],
                                ])

    def init_table_c(self):
        self.table_c = np.array([
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

    def set_body(self, values):
        assert len(values) == len(self.body)
        for i, (key, _) in enumerate(self.body.items()):
            self.body[key] = values[i]

    def set_arms(self, values):
        assert len(values) == len(self.arms)
        for i, (key, _) in enumerate(self.arms.items()):
            self.arms[key] = values[i]

    def compute_score_a(self):
        neck_score, trunk_score, leg_score, load_score = 0, 0, 0, 0

        # Neck position score calculation
        if -NECK_NEUTRAL_DEG <= self.body['neck_angle'] <= 20:
            neck_score += 1
        else:
            neck_score += 2
        neck_score += 1 if self.body['neck_side'] else 0

        # Trunk position score calculation
        if 0 <= self.body['trunk_angle'] <= 1:
            trunk_score += 1
        elif self.body['trunk_angle'] <= 20:
            trunk_score += 2
        elif 20 <= self.body['trunk_angle'] <= 60:
            trunk_score += 3
        elif self.body['trunk_angle'] > 60:
            trunk_score += 4
        trunk_score += 1 if self.body['trunk_side'] else 0

        # Legs position score calculation. REBA scores 2 when weight is not
        # borne bilaterally (one leg raised / unstable posture), 1 otherwise.
        leg_score += 2 if self.body['legs_unstable'] else 1
        if 30 <= self.body['legs_angle'] <= 60:
            leg_score += 1
        elif self.body['legs_angle'] > 60:
            leg_score += 2

        # Load score calculation
        if 5 <= self.body['load'] <= 10:
            load_score += 1
        elif self.body['load'] > 10:
            load_score += 2

        assert neck_score > 0 and trunk_score > 0 and leg_score > 0
        score_a = self.table_a[neck_score-1][trunk_score-1][leg_score-1]

        # Add load score to Table A result
        score_a += load_score

        neck_score = max(1, neck_score)
        trunk_score = max(1, trunk_score)
        leg_score = max(1, leg_score)

        return score_a, np.array([neck_score, trunk_score, leg_score, load_score])

    def compute_score_b(self):
        upper_arm_score, lower_arm_score, wrist_score = 0, 0, 0

        # Upper arm position score calculation
        angle = self.arms['upper_arm_angle']
        if -20 <= angle <= 20:
            upper_arm_score += 1
        elif (20 < angle <= 45) or (-45 <= angle < -20):
            upper_arm_score += 2
        elif (45 < angle <= 90) or (-90 <= angle < -45):
            upper_arm_score += 3
        else: # angle > 90 or angle < -90
            upper_arm_score += 4

        upper_arm_score += 1 if self.arms['shoulder_raised'] else 0
        upper_arm_score += 1 if self.arms['arm_abducted'] else 0
        # NOTE: 'leaning' is REBA's "arm is supported or the person is leaning"
        # -1 modifier. It cannot be derived from joint positions, so the caller
        # must supply it (see the arm_supported ROS parameter). It used to be
        # inferred from trunk flexion > 30 deg, which is wrong twice over:
        # bending forward does not unload the shoulder, and it fired on exactly
        # the construction postures that should score highest.
        upper_arm_score -= 1 if self.arms['leaning'] else 0

        upper_arm_score = max(1, upper_arm_score)

        # Lower arm position score calculation
        if 60 <= self.arms['lower_arm_angle'] <= 100:
            lower_arm_score += 1
        else:
            lower_arm_score += 2

        # Wrist position score calculation
        if -15 <= self.arms['wrist_angle'] <= 15:
            wrist_score += 1
        else:
            wrist_score += 2

        wrist_score += 1 if self.arms['wrist_twisted'] else 0

        assert lower_arm_score > 0 and wrist_score > 0
        score_b = self.table_b[upper_arm_score-1][lower_arm_score-1][wrist_score-1]

        # Add coupling score to Table B result
        coupling_score = self.arms['coupling_score']
        score_b += coupling_score

        return score_b, np.array([upper_arm_score, lower_arm_score, wrist_score, coupling_score])

    def compute_activity_score(self, static_posture=False, repeated_action=False, rapid_large_change=False):
        """
        Compute REBA activity score (0-3).
        """
        activity_score = 0
        activity_score += 1 if static_posture else 0
        activity_score += 1 if repeated_action else 0
        activity_score += 1 if rapid_large_change else 0
        return activity_score

    def compute_score_c(self, score_a, score_b, activity_score=0):
        """
        Compute final REBA score using Table C and activity score.

        score_a and score_b must be real computed scores (>= 1). Callers that
        could not measure a whole group must NOT substitute a neutral 1 here -
        that fabricates "negligible risk" out of missing data. Publish
        SCORE_NOT_ASSESSED instead.
        """
        reba_scoring = [
            'Negligible Risk',
            'Low Risk. Change may be needed',
            'Medium Risk. Further Investigate. Change Soon',
            'High Risk. Investigate and Implement Change',
            'Very High Risk. Implement Change'
        ]

        # Ensure matrix indices are integers
        score_a = int(score_a)
        score_b = int(score_b)

        if score_a < 1 or score_b < 1:
            raise ValueError(
                f"compute_score_c requires measured Table A/B scores, got "
                f"score_a={score_a}, score_b={score_b}"
            )

        score_c = self.table_c[min(score_a, 12)-1][min(score_b, 12)-1]
        final_score = score_c + activity_score
        ix = self.score_c_to_5_classes(final_score)
        caption = reba_scoring[ix]
        return score_c, final_score, caption

    @staticmethod
    def score_c_to_5_classes(score_c):
        if score_c < 1:
            raise ValueError(f"REBA score must be >= 1, got {score_c}")
        if score_c == 1: ret = 0
        elif 2 <= score_c <= 3: ret = 1
        elif 4 <= score_c <= 7: ret = 2
        elif 8 <= score_c <= 10: ret = 3
        else: ret = 4
        return ret

    # ------------------------------------------------------------------
    # Geometry -> REBA inputs
    # ------------------------------------------------------------------

    @staticmethod
    def _body_angles(pose, hip_idx, knee_idx, ankle_idx):
        """
        Group A angles. Only the knee angle depends on which side is passed -
        neck and trunk are built from joints shared by both sides.
        """
        pose = np.asarray(pose, dtype=float)

        trunk_vec, e_up, e_lat, e_fwd, _ = _trunk_frame(pose)
        lateral_axis = pose[2] - pose[5]

        # Trunk flexion is measured against gravity, which is correct: REBA's
        # trunk item is about the trunk's deviation from upright.
        trunk_angle = np.degrees(np.arccos(np.clip(np.dot(_unit(trunk_vec), WORLD_UP), -1.0, 1.0)))

        # Neck decomposed in the body-fixed frame: atan2 is continuous, so a
        # purely lateral bend no longer flips the flexion sign by ~130 deg.
        neck_vec = pose[0] - pose[1]
        c_up = np.dot(neck_vec, e_up)
        c_fwd = np.dot(neck_vec, e_fwd)
        c_lat = np.dot(neck_vec, e_lat)
        neck_angle = np.degrees(np.arctan2(c_fwd, c_up))
        neck_side = 1 if abs(np.degrees(np.arctan2(c_lat, c_up))) > SIDE_BEND_DEG else 0

        cos_trunk_side = np.dot(_unit(trunk_vec), _unit(lateral_axis))
        trunk_side_angle = abs(90.0 - np.degrees(np.arccos(np.clip(cos_trunk_side, -1.0, 1.0))))
        trunk_side = 1 if trunk_side_angle > SIDE_BEND_DEG else 0

        # REBA's legs item is about weight bearing, not travel: score 2 means
        # one leg raised or an unstable posture. The previous test compared the
        # HORIZONTAL ankle-to-ankle distance against 0.1 m, so an ordinary
        # shoulder-width stance (0.20-0.35 m) was flagged as walking on nearly
        # every frame and inflated Score A by one point. Vertical separation is
        # what actually distinguishes a raised foot from a normal stance.
        legs_unstable = 1 if abs(pose[10][1] - pose[13][1]) > FOOT_RAISED_M else 0

        v_thigh = pose[knee_idx] - pose[hip_idx]
        v_shin = pose[ankle_idx] - pose[knee_idx]
        cos_leg = np.clip(np.dot(_unit(v_thigh), _unit(v_shin)), -1.0, 1.0)
        legs_angle = np.degrees(np.arccos(cos_leg))

        # Load is not derivable from geometry; the caller supplies it.
        load = 0

        return np.array([neck_angle, neck_side, trunk_angle, trunk_side,
                         legs_unstable, legs_angle, load])

    @staticmethod
    def get_body_angles_from_pose_left(pose, verbose=False):
        return RebaScore._body_angles(pose, hip_idx=8, knee_idx=9, ankle_idx=10)

    @staticmethod
    def get_body_angles_from_pose_right(pose, verbose=False):
        return RebaScore._body_angles(pose, hip_idx=11, knee_idx=12, ankle_idx=13)

    @staticmethod
    def _arm_angles(pose, shoulder_idx, elbow_idx, wrist_idx):
        """
        Group B angles for one arm. Side-independent: the flexion sign comes
        from the body-fixed frame rather than from camera-specific tweaks.
        """
        pose = np.asarray(pose, dtype=float)

        v_upper = pose[shoulder_idx] - pose[elbow_idx]
        v_lower = pose[wrist_idx] - pose[elbow_idx]
        cos_elbow = np.clip(np.dot(_unit(v_upper), _unit(v_lower)), -1.0, 1.0)
        lower_arm_angle = 180.0 - np.degrees(np.arccos(cos_elbow))

        # REBA measures upper arm flexion RELATIVE TO THE TRUNK, not to the
        # world vertical. Measuring against gravity scored a worker bent 60 deg
        # forward with arms hanging as 0 deg flexion (upper arm score 1) where
        # REBA gives 60 deg (score 3) - the largest single source of
        # under-scoring on construction postures. When the trunk joints are
        # missing, _trunk_frame falls back to a world-aligned frame, which
        # degrades to the previous behaviour rather than producing garbage.
        trunk_vec, e_up, e_lat, e_fwd, _ = _trunk_frame(pose)

        arm_vec = pose[elbow_idx] - pose[shoulder_idx]
        cos_upper = np.clip(np.dot(_unit(arm_vec), e_up), -1.0, 1.0)
        upper_arm_angle = 180.0 - np.degrees(np.arccos(cos_upper))

        # Forward of the trunk plane -> flexion (positive), behind -> extension
        # (negative). e_fwd follows the same convention as the neck angle
        # above, so both agree on which way "forward" points. This replaces the
        # two hand-tuned per-side sign flips that were tied to one camera's
        # depth orientation.
        upper_arm_angle *= 1.0 if np.dot(arm_vec, e_fwd) >= 0 else -1.0

        mid_shoulder_y = (pose[2][1] + pose[5][1]) / 2.0
        shoulder_raised = 1 if (pose[shoulder_idx][1] - mid_shoulder_y) > SHOULDER_RAISED_M else 0

        # Abduction: how far the arm swings out along the trunk-orthogonal
        # lateral axis. A hanging arm is perpendicular to e_lat -> 0 deg.
        cos_abduct = np.clip(np.dot(_unit(arm_vec), e_lat), -1.0, 1.0)
        abduct_angle = abs(90.0 - np.degrees(np.arccos(cos_abduct)))
        arm_abducted = 1 if abduct_angle > ABDUCTION_DEG else 0

        # 'leaning' (arm supported / person leaning) and the wrist terms are
        # not observable from a COCO-18 skeleton. They are returned as 0 and
        # must be overridden by the caller if a real source exists; the node
        # publishes *_assessed flags so consumers can tell "0 because good"
        # from "0 because never measured".
        leaning = 0
        wrist_angle = 0
        wrist_twisted = 0

        return np.array([upper_arm_angle, shoulder_raised, arm_abducted, leaning,
                         lower_arm_angle, wrist_angle, wrist_twisted])

    @staticmethod
    def get_arms_angles_from_pose_left(pose, verbose=False):
        return RebaScore._arm_angles(pose, shoulder_idx=2, elbow_idx=3, wrist_idx=4)

    @staticmethod
    def get_arms_angles_from_pose_right(pose, verbose=False):
        return RebaScore._arm_angles(pose, shoulder_idx=5, elbow_idx=6, wrist_idx=7)


def normalize_angle(angle):
    """
    Wrap an angle into (-180, 180].

    No longer applied to the pose->angle results: arccos already returns
    [0, 180] and atan2 returns (-180, 180], so wrapping was a no-op except at
    exactly 180 deg, where it flipped a fully inverted trunk to -180 and made
    compute_score_a() read it as a 2 instead of a 4.
    """
    return ((angle + 180) % 360) - 180
