#!/usr/bin/env python3
"""
select_frames.py

Step 2 of the accuracy-validation protocol (see README.md).

Turns a full recording into a manageable, representative annotation task.

Two things make this more than "take every Nth frame":

  * Stratified sampling. Real footage is dominated by ordinary standing, so a
    uniform sample would be almost entirely low-risk frames and the resulting
    accuracy figure would say nothing about the high-risk postures that matter.
    Frames are bucketed by system risk class and by how complete the skeleton
    was, then sampled evenly across buckets.

  * The annotator packet contains NO system scores. If annotators can see what
    the system decided, they anchor to it, and the reference stops being
    independent. The scores are written to a separate reference file that
    evaluate.py reads later.

Usage:
    python3 select_frames.py --input ~/reba_validation/run1 \\
                             --output ~/reba_validation/task \\
                             --n 120 --annotators A B
"""

import argparse
import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path

SCORE_NOT_ASSESSED = 255
RISK_UNKNOWN = 255

# Columns the annotator fills in. Deliberately region scores, not a final REBA
# number: looking up Table A/B/C by hand is slow and error-prone, and those
# tables are not what we are testing. evaluate.py applies them.
ANNOTATION_COLUMNS = [
    "neck_score",       # 1-3
    "trunk_score",      # 1-5
    "leg_score",        # 1-4
    "upper_arm_score",  # 1-6
    "lower_arm_score",  # 1-2
    "unusable",         # 1 if the posture cannot be judged from this image
    "notes",
]

REFERENCE_COLUMNS = [
    "frame_id", "image", "stamp_sec", "stamp_nanosec", "body_key",
    "completeness", "score_a", "score_b", "score_c", "risk_level",
    "neck_score", "trunk_score", "leg_score",
    "upper_arm_score", "lower_arm_score", "wrist_score",
    "neck_assessed", "trunk_assessed", "legs_assessed",
    "upper_arm_assessed", "lower_arm_assessed",
    "score_is_lower_bound", "deba_score", "deba_valid",
]


def completeness_band(value):
    """Coarse buckets so the report can separate full from partial skeletons."""
    try:
        c = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if c >= 0.90:
        return "full"
    if c >= 0.75:
        return "high"
    if c >= 0.50:
        return "partial"
    return "sparse"


def stratum_of(row):
    risk = row.get("risk_level", "")
    risk = "unknown" if str(risk) in ("", str(RISK_UNKNOWN)) else f"risk{risk}"
    return f"{risk}/{completeness_band(row.get('completeness'))}"


def stratified_sample(rows, n, seed=42):
    """
    Even coverage across strata, spending any shortfall on the strata that
    still have frames left. Small strata are not dropped - a risk class seen
    only twice in the whole bag is exactly the case worth annotating.
    """
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for row in rows:
        buckets[stratum_of(row)].append(row)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    selected, cursor = [], defaultdict(int)
    while len(selected) < n:
        progressed = False
        for key in sorted(buckets):
            if len(selected) >= n:
                break
            idx = cursor[key]
            if idx < len(buckets[key]):
                selected.append(buckets[key][idx])
                cursor[key] += 1
                progressed = True
        if not progressed:
            break   # every stratum exhausted

    selected.sort(key=lambda r: int(r["frame_id"]))
    return selected, {k: len(v) for k, v in buckets.items()}


def main():
    parser = argparse.ArgumentParser(description="Build a REBA annotation task from a recording")
    parser.add_argument("--input", required=True, help="annotation_recorder output dir")
    parser.add_argument("--output", required=True, help="where to write the annotation packet")
    parser.add_argument("--n", type=int, default=120, help="frames to annotate")
    parser.add_argument("--annotators", nargs="+", default=["A", "B"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--require-image", action="store_true", default=True,
                        help="skip frames with no matching camera image")
    args = parser.parse_args()

    in_dir = Path(args.input).expanduser()
    out_dir = Path(args.output).expanduser()
    frames_csv = in_dir / "frames.csv"
    if not frames_csv.exists():
        raise FileNotFoundError(f"{frames_csv} not found - run annotation_recorder first.")

    rows = list(csv.DictReader(frames_csv.open(encoding="utf-8")))
    total = len(rows)
    if args.require_image:
        rows = [r for r in rows if r.get("image")]

    if not rows:
        raise SystemExit("No frames with images. Was image_topic set on the recorder?")

    selected, bucket_sizes = stratified_sample(rows, args.n, args.seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    img_out = out_dir / "images"
    img_out.mkdir(exist_ok=True)

    missing = 0
    for row in selected:
        src = in_dir / "images" / row["image"]
        if src.exists():
            shutil.copy2(src, img_out / row["image"])
        else:
            missing += 1

    # --- annotator packets: image + empty score columns, nothing else ---
    for name in args.annotators:
        path = out_dir / f"annotation_{name}.csv"
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["frame_id", "image"] + ANNOTATION_COLUMNS)
            for row in selected:
                writer.writerow([row["frame_id"], row["image"]] + [""] * len(ANNOTATION_COLUMNS))
        print(f"  annotator packet : {path}")

    # --- system reference, kept apart from the annotators ---
    ref_path = out_dir / "system_reference.csv"
    with ref_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=REFERENCE_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in selected:
            writer.writerow(row)

    print(f"\nSelected {len(selected)} of {len(rows)} usable frames ({total} recorded).")
    if missing:
        print(f"  WARNING: {missing} image files were missing from the recording.")
    print(f"  system reference : {ref_path}")
    print(f"  images           : {img_out}")
    print("\nStrata found in the recording (risk_class/completeness -> frames available):")
    for key in sorted(bucket_sizes):
        chosen = sum(1 for r in selected if stratum_of(r) == key)
        print(f"    {key:<22} available {bucket_sizes[key]:>5}   selected {chosen:>3}")
    print("\nGive each annotator ONLY their annotation_*.csv and the images folder.")
    print("Do not show them system_reference.csv - it would anchor their judgement.")


if __name__ == "__main__":
    main()
