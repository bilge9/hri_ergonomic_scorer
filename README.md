# hri_ergonomic_scorer

REBA- and DEBA-inspired ergonomic risk assessment for Vulcanexus HRI.

The node subscribes to 3D skeleton tracking (`hri_msgs/Skeleton3DList`), assesses
how complete the skeleton is, scores the posture even when joints are missing,
and publishes the result as `hri_ergonomic_msgs/RebaAssessmentList`.

Derived from [rs9000/ergonomics](https://github.com/rs9000/ergonomics), ported to
ROS 2 and extended with the load, coupling and activity modifiers, an explicit
bilateral arm policy, partial-skeleton handling and a continuous DEBA
companion score.

## Packages

| Package | Contents |
| --- | --- |
| `hri_ergonomic_msgs` | `RebaAssessment` / `RebaAssessmentList` IDL definitions |
| `hri_ergonomic_scorer/scorer_node` | scorer node, FIWARE bridge, skeleton overlay, DEBA training scripts |

## Build

```bash
cd <your_ws>
rosdep install --from-paths src --ignore-src -r -y
colcon build --packages-select hri_ergonomic_msgs hri_ergonomic_scorer
source install/setup.bash
```

## Run

```bash
ros2 launch hri_ergonomic_scorer reba_dashboard.launch.py
```

Or the scorer alone:

```bash
ros2 run hri_ergonomic_scorer scorer_node
```

| Topic | Type | Direction |
| --- | --- | --- |
| `/humans/bodies/skel3D` | `hri_msgs/Skeleton3DList` | subscribed |
| `/humans/bodies/ergonomics/reba` | `hri_ergonomic_msgs/RebaAssessmentList` | published |
| `/humans/bodies/skel3D/overlay` | `sensor_msgs/Image` | published by the overlay node |

## Parameters

| Parameter | Default | Purpose |
| --- | --- | --- |
| `use_sensor_qos` | `false` | `false` = RELIABLE (rosbag replay), `true` = BEST_EFFORT (live ZED / `hri_pose_detect`). A RELIABLE subscriber will not connect to a BEST_EFFORT publisher. |
| `confidence_threshold` | `0.4` | Minimum keypoint confidence to accept a joint |
| `profile_depth_threshold` | `0.25` | Shoulder depth difference (m) above which the far side is treated as hallucinated |
| `swap_y_z`, `flip_y_sign` | `false`, `true` | Camera frame alignment |
| `high_risk_threshold` | `8` | Logs a warning at or above this final score |
| `arm_supported` | `false` | REBA's "arm supported / person leaning" -1 modifier |
| `load_kg`, `load_known` | `0.0`, `false` | Carried load |
| `coupling_score`, `coupling_known` | `0`, `false` | Hand-object coupling (0 good … 3 unacceptable) |
| `activity_static`, `activity_repeated`, `activity_rapid_change`, `activity_known` | `false` | REBA activity modifier |
| `verbose_logging` | `false` | Dump raw joint coordinates |

## Reading the output: measured vs. not measured

A 3D skeleton cannot supply every REBA term. Wrist orientation, carried load,
hand coupling and activity-over-time have no source in the pose data, and
occlusion routinely removes whole limbs.

Scoring those as their neutral best case and publishing the result as a
measurement is the failure mode this package is built to avoid: it makes a
fully occluded worker read as "REBA 1 / Negligible Risk". So:

* every REBA item carries an `*_assessed` / `*_known` flag;
* any score whose item was not observed is published as `SCORE_NOT_ASSESSED`
  (255), never as a plausible number;
* `score_c == 255` and `risk_level == 255` (`RISK_UNKNOWN`) mean **no score
  could be produced for this frame**, not low risk;
* `score_is_lower_bound` is true when at least one contributor was unobserved,
  so the published score is a floor on the real risk.

**Consumers must check the flags before rendering a score.** The `*_known`
parameters default to `false`, so out of the box the final score is always a
lower bound.

## DEBA

DEBA is a continuous, differentiable surrogate for REBA. REBA stays the
authoritative output; DEBA is secondary. Its value is not accuracy — it is
that the score is differentiable (usable as a gradient target for robot motion
optimisation in pHRI) and that it moves continuously inside a discrete REBA
category, which supports early-warning thresholds.

The checkpoint is a build artefact and is **not** committed, because `reba.py`
is the teacher that labels its training data. Regenerate it whenever the
scoring rules change:

```bash
cd hri_ergonomic_scorer/scorer_node/hri_ergonomic_scorer
python3 generate_deba_dataset.py          # synthetic postures, labelled by reba.py
python3 train_deba_model.py               # writes deba_model.pth
```

Without a checkpoint the node logs an error, disables DEBA and continues
publishing REBA normally.

Two properties of the pipeline are load-bearing and easy to break:

* the raw sample pool is split into train/val **before** class balancing.
  Balancing first duplicates rows across the split and the validation score
  stops measuring generalisation;
* oversampled rows are **re-labelled** after noise injection. REBA is a step
  function, so perturbed features frequently belong to a different class than
  the row they were derived from.

The validation set is deliberately left unbalanced — it is the held-out
distribution. Report the per-class agreement the training script prints, not
only the overall figure.

## Tests

```bash
python3 -m pytest hri_ergonomic_scorer/scorer_node/test/test_reba.py
```

These pin down the scoring defects found during the audit (upper arm measured
against the trunk rather than gravity, stance mistaken for walking, the
supported-arm discount, fabricated scores from missing groups).

## Known limitations

* Wrist flexion and twist are never assessed — COCO-18 has no wrist
  orientation. `wrist_assessed` is always `false`.
* Load, coupling and activity require sources the vision pipeline does not
  have; they are parameters, and default to unknown.
* With every `*_known` parameter at its default, the reachable final score
  range is 1–11, not 1–15.
* `shoulder_raised` is an asymmetry proxy: it detects one shoulder raised
  relative to the other, not both raised together.
* **No accuracy figure has been measured against expert-annotated ground
  truth yet.** Any accuracy number quoted for this package must state the
  protocol it came from.
