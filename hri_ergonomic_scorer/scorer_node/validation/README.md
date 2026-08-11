# Accuracy validation

Tooling for measuring how well the scorer agrees with human ergonomic
judgement — the measurement requested in review note #13.

This answers a different question from the DEBA agreement figure in the main
README. That one says how faithfully DEBA reproduces REBA; this one says
whether the system's REBA matches what a trained person would score. A model
that perfectly imitated a flawed teacher would score 100% on the first and
poorly on this one.

## Why it needs people

There is no instrumented ground truth for ergonomic risk in these recordings.
The reference has to be human judgement, and two independent judgements, so
the reference's own reliability can be measured and reported. Nothing here can
be automated away without turning the result into two algorithms agreeing with
each other, which is not evidence.

---

## Step 1 — Record

Replay the bag through the scorer and capture every assessment with its camera
frame.

```bash
# terminal 1
ros2 run hri_ergonomic_scorer scorer_node

# terminal 2
ros2 run hri_ergonomic_scorer annotation_recorder --ros-args \
  -p image_topic:=/zed/zed_node/left/image_rect_color \
  -p output_dir:=~/reba_validation/run1

# terminal 3
ros2 bag play <your bag>
```

Stop the recorder with Ctrl-C when the bag ends. It writes
`frames.csv` plus one PNG per frame under `images/`.

Set `image_topic` to whatever RGB topic the bag actually contains — check with
`ros2 bag info <bag>`. Without it you get scores but no pictures, and there is
nothing to annotate.

## Step 2 — Build the annotation task

```bash
python3 select_frames.py \
  --input ~/reba_validation/run1 \
  --output ~/reba_validation/task \
  --n 120 --annotators A B
```

Frames are sampled **stratified** by risk class and skeleton completeness, not
uniformly. Ordinary standing dominates real footage, so a uniform sample would
be almost entirely low-risk frames and would say nothing about the demanding
postures that matter. Rare strata are taken in full.

The output contains one `annotation_<name>.csv` per annotator, an `images/`
folder, and `system_reference.csv`.

**Give each annotator only their own CSV and the images folder.** Do not show
them `system_reference.csv`, and do not let them compare notes while scoring —
both destroy the independence the kappa in step 3 depends on.

## Step 3 — Annotate (the human part)

Two people, independently, roughly 45 seconds per frame — about **1.5 hours
each** for 120 frames.

Fill in five columns per frame. Do **not** compute a final REBA score; the
tables are applied automatically, and looking them up by hand is slow and
error-prone.

| Column | Range | REBA rule |
| --- | --- | --- |
| `neck_score` | 1–3 | 1 = 0–20° flexion · 2 = >20° or any extension · +1 if twisted or side-bent |
| `trunk_score` | 1–5 | 1 = upright · 2 = 0–20° flex or any extension · 3 = 20–60° · 4 = >60° · +1 if twisted or side-bent |
| `leg_score` | 1–4 | 1 = weight on both legs · 2 = one leg raised or unstable · +1 if a knee is bent 30–60° · +2 if >60° |
| `upper_arm_score` | 1–6 | 1 = ±20° · 2 = 20–45° or >20° extension · 3 = 45–90° · 4 = >90° · +1 shoulder raised · +1 arm abducted · −1 arm supported / leaning |
| `lower_arm_score` | 1–2 | 1 = elbow at 60–100° · 2 = anything else |

Mark `unusable = 1` if the posture genuinely cannot be judged from the image
(person out of frame, heavily occluded, motion blur). Those frames are dropped
rather than guessed.

Two rules that protect the result:

- **Judge the arm relative to the trunk**, not to the room. This is what REBA
  specifies, and getting it wrong here was one of the defects found in the
  code.
- **Score what you see in the photo**, not what you think the person was about
  to do.

Angles are estimated by eye. That is expected — the kappa in step 3 measures
how much that costs, and reporting it is part of the method.

## Step 4 — Evaluate

```bash
python3 evaluate.py \
  --task ~/reba_validation/task \
  --annotations annotation_A.csv annotation_B.csv \
  --out validation_report.md
```

`--consensus strict` (default) drops frames where the two annotators disagree.
`--consensus max` keeps them and takes the higher score. Strict is cleaner;
max keeps more data and errs toward caution. Report which you used.

The report contains:

1. **Inter-rater agreement first.** It measures the reference, not the system.
   Below κ ≈ 0.60 on the risk level, the reference is unreliable and no system
   number computed against it is interpretable — tighten the guidance and
   re-annotate rather than concluding anything about the code.
2. Exact and ±1 agreement, MAE, and directional bias. A negative bias means
   the system under-reports risk, which is the dangerous direction.
3. The 5-class risk confusion matrix — what the dashboard actually shows.
4. **High-risk recall, separately.** Missing a dangerous posture matters more
   than a false alarm, and rare classes barely move an overall percentage.
   Lead with this number.
5. A breakdown by skeleton completeness, because scoring partial skeletons is
   a stated requirement of the task.

## What this measures, and what it does not

Wrist is fixed at 1 and load / coupling / activity at 0 on **both** sides,
because the vision pipeline cannot observe them. The comparison is therefore
about the posture geometry the system actually measures, not the complete REBA
instrument. State that alongside any number quoted from it.

If the sample contains no genuinely high-risk postures, the report says so
instead of reporting a recall figure — a validation run over gentle footage
cannot demonstrate that the system finds dangerous postures.
