# -*- coding: utf-8 -*-
# ---------------------

import numpy as np

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
                     'legs_walking': False, 'legs_angle': 0,
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
        if 0 <= self.body['neck_angle'] <= 20:
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

        # Legs position score calculation
        leg_score += 2 if self.body['legs_walking'] else 1
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

        return score_a, np.array([neck_score, trunk_score, leg_score, load_score])

    def compute_score_b(self):
        upper_arm_score, lower_arm_score, wrist_score = 0, 0, 0

        # Upper arm position score calculation
        if -20 <= self.arms['upper_arm_angle'] <= 20:
            upper_arm_score += 1
        elif self.arms['upper_arm_angle'] <= 45:
            upper_arm_score += 2
        elif 45 <= self.arms['upper_arm_angle'] <= 90:
            upper_arm_score += 3
        elif self.arms['upper_arm_angle'] > 90:
            upper_arm_score += 4

        upper_arm_score += 1 if self.arms['shoulder_raised'] else 0
        upper_arm_score += 1 if self.arms['arm_abducted'] else 0
        upper_arm_score -= 1 if self.arms['leaning'] else 0

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
        
        score_c = self.table_c[score_a-1][score_b-1]
        final_score = score_c + activity_score
        ix = self.score_c_to_5_classes(final_score)
        caption = reba_scoring[ix]
        return score_c, final_score, caption

    @staticmethod
    def score_c_to_5_classes(score_c):
        if score_c == 1: ret = 0
        elif 2 <= score_c <= 3: ret = 1
        elif 4 <= score_c <= 7: ret = 2
        elif 8 <= score_c <= 10: ret = 3
        else: ret = 4
        return ret

    @staticmethod
    def get_body_angles_from_pose_left(pose, verbose=False):
        pose = np.expand_dims(np.copy(pose), 0)

        mid_hip_3d = (pose[0, 8] + pose[0, 11]) / 2.0
        trunk_vec_3d = pose[0, 1] - mid_hip_3d
        vertical_up = np.array([0.0, 1.0, 0.0])
        
        cos_trunk = np.dot(trunk_vec_3d, vertical_up) / (np.linalg.norm(trunk_vec_3d) + 1e-9)
        cos_trunk = np.clip(cos_trunk, -1.0, 1.0)
        trunk_angle = np.degrees(np.arccos(cos_trunk))

        neck_vec_3d = pose[0, 0] - pose[0, 1]
        cos_neck = np.dot(trunk_vec_3d, neck_vec_3d) / (np.linalg.norm(trunk_vec_3d) * np.linalg.norm(neck_vec_3d) + 1e-9)
        cos_neck = np.clip(cos_neck, -1.0, 1.0)
        neck_angle = np.degrees(np.arccos(cos_neck))

        lateral_axis = pose[0, 2] - pose[0, 5]
        neck_sign_vec = np.cross(trunk_vec_3d, neck_vec_3d)
        
        # FIX: Flipped < to >= for correct ZED depth orientation
        neck_sign = 1.0 if np.dot(neck_sign_vec, lateral_axis) >= 0 else -1.0
        neck_angle = neck_sign * neck_angle

        cos_trunk_side = np.dot(trunk_vec_3d, lateral_axis) / (np.linalg.norm(trunk_vec_3d) * np.linalg.norm(lateral_axis) + 1e-9)
        trunk_side_angle = abs(90.0 - np.degrees(np.arccos(np.clip(cos_trunk_side, -1.0, 1.0))))
        trunk_side = 1 if trunk_side_angle > 10.0 else 0

        cos_neck_side = np.dot(neck_vec_3d, lateral_axis) / (np.linalg.norm(neck_vec_3d) * np.linalg.norm(lateral_axis) + 1e-9)
        neck_side_angle = abs(90.0 - np.degrees(np.arccos(np.clip(cos_neck_side, -1.0, 1.0))))
        neck_side = 1 if neck_side_angle > 10.0 else 0

        step_size = np.linalg.norm(pose[0, 10] - pose[0, 13])
        legs_walking = 1 if step_size > 0.1 else 0

        v_thigh = pose[0, 9] - pose[0, 8]
        v_shin = pose[0, 10] - pose[0, 9]
        cos_leg = np.dot(v_thigh, v_shin) / (np.linalg.norm(v_thigh) * np.linalg.norm(v_shin) + 1e-9)
        cos_leg = np.clip(cos_leg, -1.0, 1.0)
        legs_angle = np.degrees(np.arccos(cos_leg))

        load = 0

        neck_angle = normalize_angle(neck_angle)
        trunk_angle = normalize_angle(trunk_angle)
        legs_angle = normalize_angle(legs_angle)
        
        return np.array([neck_angle, neck_side, trunk_angle, trunk_side, legs_walking, legs_angle, load])

    @staticmethod
    def get_body_angles_from_pose_right(pose, verbose=False):
        pose = np.expand_dims(np.copy(pose), 0)

        mid_hip_3d = (pose[0, 8] + pose[0, 11]) / 2.0
        trunk_vec_3d = pose[0, 1] - mid_hip_3d
        vertical_up = np.array([0.0, 1.0, 0.0])
        
        cos_trunk = np.dot(trunk_vec_3d, vertical_up) / (np.linalg.norm(trunk_vec_3d) + 1e-9)
        cos_trunk = np.clip(cos_trunk, -1.0, 1.0)
        trunk_angle = np.degrees(np.arccos(cos_trunk))

        neck_vec_3d = pose[0, 0] - pose[0, 1]
        cos_neck = np.dot(trunk_vec_3d, neck_vec_3d) / (np.linalg.norm(trunk_vec_3d) * np.linalg.norm(neck_vec_3d) + 1e-9)
        cos_neck = np.clip(cos_neck, -1.0, 1.0)
        neck_angle = np.degrees(np.arccos(cos_neck))

        lateral_axis = pose[0, 2] - pose[0, 5]
        neck_sign_vec = np.cross(trunk_vec_3d, neck_vec_3d)
        
        # FIX: Flipped < to >= for correct ZED depth orientation
        neck_sign = 1.0 if np.dot(neck_sign_vec, lateral_axis) >= 0 else -1.0
        neck_angle = neck_sign * neck_angle

        cos_trunk_side = np.dot(trunk_vec_3d, lateral_axis) / (np.linalg.norm(trunk_vec_3d) * np.linalg.norm(lateral_axis) + 1e-9)
        trunk_side_angle = abs(90.0 - np.degrees(np.arccos(np.clip(cos_trunk_side, -1.0, 1.0))))
        trunk_side = 1 if trunk_side_angle > 10.0 else 0

        cos_neck_side = np.dot(neck_vec_3d, lateral_axis) / (np.linalg.norm(neck_vec_3d) * np.linalg.norm(lateral_axis) + 1e-9)
        neck_side_angle = abs(90.0 - np.degrees(np.arccos(np.clip(cos_neck_side, -1.0, 1.0))))
        neck_side = 1 if neck_side_angle > 10.0 else 0

        step_size = np.linalg.norm(pose[0, 10] - pose[0, 13])
        legs_walking = 1 if step_size > 0.1 else 0

        v_thigh = pose[0, 12] - pose[0, 11]
        v_shin = pose[0, 13] - pose[0, 12]
        cos_leg = np.dot(v_thigh, v_shin) / (np.linalg.norm(v_thigh) * np.linalg.norm(v_shin) + 1e-9)
        cos_leg = np.clip(cos_leg, -1.0, 1.0)
        legs_angle = np.degrees(np.arccos(cos_leg))

        load = 0

        neck_angle = normalize_angle(neck_angle)
        trunk_angle = normalize_angle(trunk_angle)
        legs_angle = normalize_angle(legs_angle)
        
        return np.array([neck_angle, neck_side, trunk_angle, trunk_side, legs_walking, legs_angle, load])

    @staticmethod
    def get_arms_angles_from_pose_left(pose, verbose=False):
        pose = np.expand_dims(np.copy(pose), 0)

        v_upper_3d = pose[0, 2] - pose[0, 3]
        v_lower_3d = pose[0, 4] - pose[0, 3]
        cos_elbow = np.dot(v_upper_3d, v_lower_3d) / (np.linalg.norm(v_upper_3d) * np.linalg.norm(v_lower_3d) + 1e-9)
        cos_elbow = np.clip(cos_elbow, -1.0, 1.0)
        true_lower_arm_angle = 180.0 - np.degrees(np.arccos(cos_elbow))

        arm_vec_3d = pose[0, 3] - pose[0, 2]
        vertical_down = np.array([0.0, 1.0, 0.0])
        
        cos_upper = np.dot(arm_vec_3d, vertical_down) / (np.linalg.norm(arm_vec_3d) + 1e-9)
        cos_upper = np.clip(cos_upper, -1.0, 1.0)
        true_upper_arm_angle = 180.0 - np.degrees(np.arccos(cos_upper))

        lateral_axis = pose[0, 2] - pose[0, 5]
        upper_arm_sign_vec = np.cross(vertical_down, arm_vec_3d)
        
        # FIX: Flipped < to >= for correct ZED depth orientation
        upper_arm_sign = 1.0 if np.dot(upper_arm_sign_vec, lateral_axis) >= 0 else -1.0
        true_upper_arm_angle = upper_arm_sign * true_upper_arm_angle

        mid_shoulder_y = (pose[0, 2, 1] + pose[0, 5, 1]) / 2.0
        shoulder_raised = 1 if (pose[0, 2, 1] - mid_shoulder_y) > 0.02 else 0

        cos_abduct = np.dot(arm_vec_3d, lateral_axis) / (np.linalg.norm(arm_vec_3d) * np.linalg.norm(lateral_axis) + 1e-9)
        abduct_angle = abs(90.0 - np.degrees(np.arccos(np.clip(cos_abduct, -1.0, 1.0))))
        arm_abducted = 1 if abduct_angle > 45.0 else 0

        mid_hip_3d = (pose[0, 8] + pose[0, 11]) / 2.0
        trunk_vec_3d = pose[0, 1] - mid_hip_3d
        vertical_up = np.array([0.0, 1.0, 0.0])
        cos_trunk = np.dot(trunk_vec_3d, vertical_up) / (np.linalg.norm(trunk_vec_3d) + 1e-9)
        trunk_angle = np.degrees(np.arccos(np.clip(cos_trunk, -1.0, 1.0)))
        leaning = 1 if trunk_angle > 30.0 else 0

        lower_arm_angle = true_lower_arm_angle
        wrist_angle = 0
        wrist_twisted = 0
        upper_arm_angle = true_upper_arm_angle

        upper_arm_angle = normalize_angle(upper_arm_angle)
        lower_arm_angle = normalize_angle(lower_arm_angle)
        wrist_angle = normalize_angle(wrist_angle)

        return np.array([upper_arm_angle, shoulder_raised, arm_abducted, leaning, lower_arm_angle, wrist_angle, wrist_twisted])

    @staticmethod
    def get_arms_angles_from_pose_right(pose, verbose=False):
        pose = np.expand_dims(np.copy(pose), 0)

        v_upper_3d = pose[0, 5] - pose[0, 6]
        v_lower_3d = pose[0, 7] - pose[0, 6]
        cos_elbow = np.dot(v_upper_3d, v_lower_3d) / (np.linalg.norm(v_upper_3d) * np.linalg.norm(v_lower_3d) + 1e-9)
        cos_elbow = np.clip(cos_elbow, -1.0, 1.0)
        true_lower_arm_angle = 180.0 - np.degrees(np.arccos(cos_elbow))

        arm_vec_3d = pose[0, 6] - pose[0, 5]
        vertical_down = np.array([0.0, 1.0, 0.0])
        
        cos_upper = np.dot(arm_vec_3d, vertical_down) / (np.linalg.norm(arm_vec_3d) + 1e-9)
        cos_upper = np.clip(cos_upper, -1.0, 1.0)
        true_upper_arm_angle = 180.0 - np.degrees(np.arccos(cos_upper))

        lateral_axis = pose[0, 2] - pose[0, 5]
        upper_arm_sign_vec = np.cross(vertical_down, arm_vec_3d)
        
        # FIX: Flipped >= to < for correct ZED depth orientation on the right arm
        upper_arm_sign = 1.0 if np.dot(upper_arm_sign_vec, lateral_axis) < 0 else -1.0
        true_upper_arm_angle = upper_arm_sign * true_upper_arm_angle

        mid_shoulder_y = (pose[0, 2, 1] + pose[0, 5, 1]) / 2.0
        shoulder_raised = 1 if (pose[0, 5, 1] - mid_shoulder_y) > 0.02 else 0

        cos_abduct = np.dot(arm_vec_3d, lateral_axis) / (np.linalg.norm(arm_vec_3d) * np.linalg.norm(lateral_axis) + 1e-9)
        abduct_angle = abs(90.0 - np.degrees(np.arccos(np.clip(cos_abduct, -1.0, 1.0))))
        arm_abducted = 1 if abduct_angle > 45.0 else 0

        mid_hip_3d = (pose[0, 8] + pose[0, 11]) / 2.0
        trunk_vec_3d = pose[0, 1] - mid_hip_3d
        vertical_up = np.array([0.0, 1.0, 0.0])
        cos_trunk = np.dot(trunk_vec_3d, vertical_up) / (np.linalg.norm(trunk_vec_3d) + 1e-9)
        trunk_angle = np.degrees(np.arccos(np.clip(cos_trunk, -1.0, 1.0)))
        leaning = 1 if trunk_angle > 60.0 else 0

        lower_arm_angle = true_lower_arm_angle
        wrist_angle = 0
        wrist_twisted = 0
        upper_arm_angle = true_upper_arm_angle

        upper_arm_angle = normalize_angle(upper_arm_angle)
        lower_arm_angle = normalize_angle(lower_arm_angle)
        wrist_angle = normalize_angle(wrist_angle)
        return np.array([upper_arm_angle, shoulder_raised, arm_abducted, leaning, lower_arm_angle, wrist_angle, wrist_twisted])

def normalize_angle(angle):
    return ((angle + 180) % 360) - 180