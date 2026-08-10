"""
Regression tests for the REBA scoring maths.

Each test pins down a defect that was found by auditing the scorer against
hand-computed REBA, so a future refactor cannot quietly reintroduce it. They
depend only on numpy + reba.py, so they run without a ROS 2 environment:

    python3 -m pytest hri_ergonomic_scorer/scorer_node/test/test_reba.py
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'hri_ergonomic_scorer'))

from reba import RebaScore, SCORE_NOT_ASSESSED  # noqa: E402


# rs9000 joint order:
#   0 Head, 1 Neck, 2 L_Sho, 3 L_Elb, 4 L_Wri, 5 R_Sho, 6 R_Elb, 7 R_Wri,
#   8 L_Hip, 9 L_Knee, 10 L_Ankle, 11 R_Hip, 12 R_Knee, 13 R_Ankle
def make_pose(stance=0.05, trunk_flex_deg=0.0, arm_flex_deg=0.0,
              arms_along_gravity=False, raise_left_foot=0.0):
    """
    Build a synthetic skeleton with Y up and Z forward.

    arm_flex_deg is measured RELATIVE TO THE TRUNK, the way REBA defines it.
    arms_along_gravity overrides it with a limp arm hanging straight down,
    which is the same thing as arm_flex_deg == trunk_flex_deg.
    """
    p = np.zeros((14, 3))
    sh_half, trunk_len = 0.20, 0.50

    p[8] = [stance / 2, 0.90, 0.0]
    p[11] = [-stance / 2, 0.90, 0.0]
    p[9] = [stance / 2, 0.50, 0.0]
    p[12] = [-stance / 2, 0.50, 0.0]
    p[10] = [stance / 2, 0.05 + raise_left_foot, 0.0]
    p[13] = [-stance / 2, 0.05, 0.0]

    mid_hip = (p[8] + p[11]) / 2.0
    t = np.radians(trunk_flex_deg)
    neck = mid_hip + np.array([0.0, trunk_len * np.cos(t), trunk_len * np.sin(t)])
    p[1] = neck
    # Head stays aligned with the trunk -> zero neck flexion.
    p[0] = neck + np.array([0.0, 0.20 * np.cos(t), 0.20 * np.sin(t)])
    p[2] = neck + np.array([sh_half, 0.0, 0.0])
    p[5] = neck + np.array([-sh_half, 0.0, 0.0])

    # Direction at angle theta from +Y in the YZ plane. The trunk points along
    # theta = trunk_flex; an arm hanging along the trunk is theta + 180, and
    # forward flexion of the shoulder subtracts from there.
    flex = trunk_flex_deg if arms_along_gravity else arm_flex_deg
    theta = np.radians(trunk_flex_deg + 180.0 - flex)
    upper = 0.30 * np.array([0.0, np.cos(theta), np.sin(theta)])

    p[3] = p[2] + upper
    p[6] = p[5] + upper
    p[4] = p[3] + upper * 0.9
    p[7] = p[6] + upper * 0.9
    return p


def full_score(pose, load=0, coupling=0, activity=0, arm_supported=False):
    reba = RebaScore()
    body = reba.get_body_angles_from_pose_right(pose)
    body[6] = load
    reba.set_body(body)
    score_a, partial_a = reba.compute_score_a()

    arm = reba.get_arms_angles_from_pose_right(pose)
    arm_values = np.zeros(8)
    arm_values[:7] = arm
    arm_values[3] = 1.0 if arm_supported else 0.0
    arm_values[7] = coupling
    reba.set_arms(arm_values)
    score_b, partial_b = reba.compute_score_b()

    _, final, caption = reba.compute_score_c(score_a, score_b, activity)
    return dict(body=body, arm=arm, score_a=score_a, partial_a=partial_a,
                score_b=score_b, partial_b=partial_b, final=final, caption=caption)


# ---------------------------------------------------------------- baseline

def test_neutral_upright_scores_one():
    r = full_score(make_pose())
    assert r['final'] == 1
    assert r['caption'] == 'Negligible Risk'


def test_neutral_neck_is_not_scored_as_extension():
    """
    The neck angle comes from atan2, so an anatomically neutral neck lands on
    a tiny negative float. Without a dead band it scored as extension (2).
    """
    r = full_score(make_pose(trunk_flex_deg=60.0, arms_along_gravity=True))
    assert r['partial_a'][0] == 1, "neutral neck must score 1, not extension"


# ------------------------------------------------- B3: walking detection

@pytest.mark.parametrize('stance', [0.20, 0.25, 0.30])
def test_normal_stance_is_not_walking(stance):
    """
    An ordinary shoulder-width stance used to trip the ankle-distance test and
    add a point to the leg score on nearly every frame.
    """
    r = full_score(make_pose(stance=stance))
    assert r['body'][4] == 0, f"stance {stance} m must not count as unstable"
    assert r['partial_a'][2] == 1


def test_raised_foot_is_unstable():
    """A genuinely raised foot must still reach leg base score 2."""
    r = full_score(make_pose(stance=0.20, raise_left_foot=0.30))
    assert r['body'][4] == 1
    assert r['partial_a'][2] == 2


# ------------------------------------- B4: upper arm relative to the trunk

def test_upper_arm_is_measured_against_the_trunk_not_gravity():
    """
    Bent 60 deg forward with the arms hanging along gravity, the arm is at
    60 deg of flexion relative to the trunk -> REBA upper arm score 3.
    Measuring against the world vertical reported 0 deg and score 1.
    """
    r = full_score(make_pose(trunk_flex_deg=60.0, arms_along_gravity=True))
    assert r['arm'][0] == pytest.approx(60.0, abs=1.0)
    assert r['partial_b'][0] == 3


def test_arm_aligned_with_bent_trunk_is_neutral():
    """The mirror case: an arm hanging along a tilted trunk is 0 deg flexion."""
    pose = make_pose(trunk_flex_deg=60.0, arm_flex_deg=0.0)
    r = full_score(pose)
    assert abs(r['arm'][0]) < 1.0
    assert r['partial_b'][0] == 1


# ------------------------------------------------- B5: the leaning modifier

def test_forward_bend_does_not_grant_the_supported_arm_discount():
    """
    REBA's -1 means the arm is supported or the person is leaning on
    something. Trunk flexion used to trigger it, removing a point from exactly
    the postures that should score highest.
    """
    bent = full_score(make_pose(trunk_flex_deg=60.0, arms_along_gravity=True))
    supported = full_score(make_pose(trunk_flex_deg=60.0, arms_along_gravity=True),
                           arm_supported=True)
    assert supported['partial_b'][0] == bent['partial_b'][0] - 1


# -------------------------------------------- B1/B2: no fabricated scores

def test_score_c_refuses_unmeasured_groups():
    """
    Substituting a neutral 1 for a group that was never observed is what
    turned a fully occluded body into "REBA 1 / Negligible Risk".
    """
    reba = RebaScore()
    with pytest.raises(ValueError):
        reba.compute_score_c(0, 4)
    with pytest.raises(ValueError):
        reba.compute_score_c(4, 0)


def test_risk_class_rejects_impossible_scores():
    with pytest.raises(ValueError):
        RebaScore.score_c_to_5_classes(0)


def test_sentinel_is_out_of_band():
    assert SCORE_NOT_ASSESSED == 255
    assert SCORE_NOT_ASSESSED > 15


# ----------------------------------- profile poses / degraded trunk frame

def _valid_all():
    return np.ones(14, dtype=bool)


def test_single_hip_still_measures_trunk_flexion():
    """
    Side-on workers lose one hip. Requiring both declared the trunk
    unmeasurable, and the scorer then substituted an upright trunk for
    someone who could be bent double.
    """
    pose = make_pose(trunk_flex_deg=60.0, arms_along_gravity=True)
    valid = _valid_all()
    valid[11] = False   # right hip occluded

    both = RebaScore.get_body_angles_from_pose_left(pose)
    one = RebaScore.get_body_angles_from_pose_left(pose, valid=valid)

    assert one[2] == pytest.approx(both[2], abs=8.0), \
        "one-hip trunk flexion must stay close to the two-hip value"
    assert one[2] > 45.0, "a 60 deg bend must not read as upright"


def test_lateral_terms_suppressed_when_the_frame_is_one_sided():
    """
    A single hip displaces the trunk base by half a pelvis width, which fakes
    ~10 deg of lean - right at the side-bend threshold. The lateral items must
    not be reported from such a frame.
    """
    pose = make_pose(trunk_flex_deg=20.0)
    valid = _valid_all()
    valid[11] = False

    angles = RebaScore.get_body_angles_from_pose_left(pose, valid=valid)
    assert angles[1] == 0, "neck side bend must not be claimed from one hip"
    assert angles[3] == 0, "trunk side bend must not be claimed from one hip"


def test_single_shoulder_still_measures_upper_arm():
    pose = make_pose(trunk_flex_deg=60.0, arms_along_gravity=True)
    valid = _valid_all()
    valid[5] = False    # right shoulder occluded; left arm still visible

    arm = RebaScore.get_arms_angles_from_pose_left(pose, valid=valid)
    assert arm[0] == pytest.approx(60.0, abs=5.0)


def test_profile_pose_reaches_a_trunk_score():
    """The whole point: a side-on worker must still be assessable."""
    from pose_remap import reba_inputs_are_sufficient

    reba_valid = _valid_all()
    for idx in (5, 6, 7, 11, 12, 13):   # right shoulder/arm and right leg gone
        reba_valid[idx] = False

    readiness = reba_inputs_are_sufficient(reba_valid)
    assert readiness['trunk_ok'], "profile pose must still yield a trunk score"
    assert readiness['neck_ok']
    assert readiness['left_leg_ok']
    assert readiness['left_arm_ok']
    assert not readiness['mid_hip_exact']
    assert not readiness['shoulder_axis_exact']


def test_no_hip_at_all_is_still_refused():
    from pose_remap import reba_inputs_are_sufficient

    reba_valid = _valid_all()
    reba_valid[8] = reba_valid[11] = False
    readiness = reba_inputs_are_sufficient(reba_valid)
    assert not readiness['trunk_ok']
    assert not readiness['neck_ok']


# ------------------------------------------------------- table integrity

def test_table_c_indices_cover_the_full_modifier_range():
    """
    Score A reaches 9 + 2 load and Score B 9 + 3 coupling, so Table C must be
    addressable up to 12 on both axes.
    """
    reba = RebaScore()
    assert reba.table_c.shape == (12, 12)
    _, final, _ = reba.compute_score_c(11, 12, 3)
    assert final == 15


def test_risk_class_boundaries():
    cases = {1: 0, 2: 1, 3: 1, 4: 2, 7: 2, 8: 3, 10: 3, 11: 4, 15: 4}
    for score, expected in cases.items():
        assert RebaScore.score_c_to_5_classes(score) == expected


# ------------------------------------------------------ monotonic behaviour

def test_deeper_trunk_flexion_never_lowers_the_score():
    previous = 0
    for angle in (0, 10, 25, 45, 70, 90):
        r = full_score(make_pose(trunk_flex_deg=angle, arm_flex_deg=0.0))
        assert r['score_a'] >= previous
        previous = r['score_a']


def test_load_and_coupling_raise_the_score():
    base = full_score(make_pose(trunk_flex_deg=45.0, arm_flex_deg=45.0))
    loaded = full_score(make_pose(trunk_flex_deg=45.0, arm_flex_deg=45.0),
                        load=15, coupling=2)
    assert loaded['score_a'] > base['score_a']
    assert loaded['score_b'] > base['score_b']
    assert loaded['final'] >= base['final']
