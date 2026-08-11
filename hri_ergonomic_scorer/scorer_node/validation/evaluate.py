#!/usr/bin/env python3
"""
evaluate.py

Step 3 of the accuracy-validation protocol (see README.md).

Compares the system against a human reference and writes the report that
review note #13 asked for.

What it reports, and why each number is there:

  * Inter-rater agreement (Cohen's kappa) FIRST. It measures the reference,
    not the system. If two trained people cannot agree on a posture, no
    system score computed against their consensus means anything, and the
    honest conclusion is that the annotation guide needs tightening - not
    that the system is wrong.

  * Exact and +/-1 agreement. REBA is ordinal, so being one point out is a
    materially different error from being five points out, and a single
    "accuracy %" hides that.

  * The 5-class risk confusion matrix. That is what the dashboard shows and
    what a safety decision is actually made on.

  * High-risk recall, called out separately. Missing a genuinely dangerous
    posture is the failure that matters in ergonomics; overall accuracy can
    look fine while this is poor, because dangerous frames are rare.

  * A breakdown by skeleton completeness, because scoring partial skeletons
    is a stated requirement of the task and deserves its own number.

Both sides are scored on the same footing: the wrist item is fixed at 1 and
load / coupling / activity at 0 for annotators too, because the system cannot
observe them. This keeps the comparison about the posture geometry the system
actually measures, rather than penalising it for inputs it never had.

Usage:
    python3 evaluate.py --task ~/reba_validation/task \\
                        --annotations annotation_A.csv annotation_B.csv \\
                        --out report.md
"""

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "hri_ergonomic_scorer"))
from reba import RebaScore  # noqa: E402

SCORE_NOT_ASSESSED = 255

REGIONS = ["neck_score", "trunk_score", "leg_score", "upper_arm_score", "lower_arm_score"]
REGION_RANGE = {
    "neck_score": (1, 3), "trunk_score": (1, 5), "leg_score": (1, 4),
    "upper_arm_score": (1, 6), "lower_arm_score": (1, 2),
}
RISK_NAMES = ["Negligible", "Low", "Medium", "High", "Very High"]


def reba_from_regions(neck, trunk, leg, upper_arm, lower_arm, wrist=1):
    """Apply the REBA tables to region scores. Mirrors reba.py exactly."""
    r = RebaScore()
    score_a = int(r.table_a[neck - 1][trunk - 1][leg - 1])
    score_b = int(r.table_b[upper_arm - 1][lower_arm - 1][wrist - 1])
    score_c = int(r.table_c[score_a - 1][score_b - 1])
    return score_a, score_b, score_c


def risk_class(score):
    return RebaScore.score_c_to_5_classes(score)


def cohens_kappa(a, b):
    """Unweighted Cohen's kappa over paired categorical labels."""
    if not a:
        return float("nan")
    n = len(a)
    observed = sum(1 for x, y in zip(a, b) if x == y) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum((ca[k] / n) * (cb[k] / n) for k in set(ca) | set(cb))
    if abs(1.0 - expected) < 1e-12:
        return float("nan")
    return (observed - expected) / (1.0 - expected)


def kappa_verdict(k):
    if k != k:
        return "not computable"
    if k < 0.20: return "poor"
    if k < 0.40: return "fair"
    if k < 0.60: return "moderate"
    if k < 0.80: return "substantial"
    return "almost perfect"


def load_annotation(path):
    rows = {}
    skipped = 0
    for row in csv.DictReader(Path(path).expanduser().open(encoding="utf-8")):
        fid = row["frame_id"].strip()
        if not fid:
            continue
        if str(row.get("unusable", "")).strip() in ("1", "y", "yes", "true"):
            skipped += 1
            continue
        try:
            scores = {}
            for region in REGIONS:
                value = int(float(row[region]))
                lo, hi = REGION_RANGE[region]
                if not lo <= value <= hi:
                    raise ValueError(
                        f"{region}={value} out of range {lo}-{hi} on frame {fid}")
                scores[region] = value
        except (KeyError, ValueError, TypeError) as exc:
            if any(str(row.get(r, "")).strip() for r in REGIONS):
                raise ValueError(f"{path}: {exc}") from exc
            skipped += 1      # entirely blank row: not yet annotated
            continue
        rows[fid] = scores
    return rows, skipped


def main():
    parser = argparse.ArgumentParser(description="Score the system against a human reference")
    parser.add_argument("--task", required=True, help="select_frames.py output dir")
    parser.add_argument("--annotations", nargs="+", required=True)
    parser.add_argument("--out", default="validation_report.md")
    parser.add_argument("--consensus", choices=["strict", "max"], default="strict",
                        help="strict = drop frames the annotators disagree on; "
                             "max = take the higher (more conservative) score")
    args = parser.parse_args()

    task = Path(args.task).expanduser()
    system = {r["frame_id"]: r for r in csv.DictReader((task / "system_reference.csv").open(encoding="utf-8"))}

    annos, names = [], []
    for path in args.annotations:
        p = Path(path)
        if not p.is_absolute() and not p.exists():
            p = task / p
        rows, skipped = load_annotation(p)
        annos.append(rows)
        names.append(p.stem.replace("annotation_", ""))
        print(f"Loaded {len(rows)} annotations from {p.name} ({skipped} blank/unusable)")

    if len(annos) < 2:
        print("WARNING: only one annotator - inter-rater agreement cannot be computed, "
              "so the reference has no measured quality.", file=sys.stderr)

    common = set(annos[0])
    for rows in annos[1:]:
        common &= set(rows)
    common &= set(system)
    common = sorted(common, key=int)
    if not common:
        raise SystemExit("No frames annotated by everyone. Nothing to evaluate.")

    lines = []
    W = lines.append
    W("# REBA accuracy validation\n")
    W(f"Frames annotated by all raters and scored by the system: **{len(common)}**\n")

    # ---------------- inter-rater ----------------
    W("\n## 1. Reference quality (inter-rater agreement)\n")
    if len(annos) >= 2:
        W("| Item | Cohen's kappa | Strength | Exact agreement |")
        W("| --- | ---: | --- | ---: |")
        for region in REGIONS:
            a = [annos[0][f][region] for f in common]
            b = [annos[1][f][region] for f in common]
            k = cohens_kappa(a, b)
            exact = 100.0 * sum(1 for x, y in zip(a, b) if x == y) / len(a)
            W(f"| {region.replace('_score','').replace('_',' ')} | {k:.3f} | {kappa_verdict(k)} | {exact:.1f}% |")

        ra = [risk_class(reba_from_regions(*[annos[0][f][r] for r in REGIONS])[2]) for f in common]
        rb = [risk_class(reba_from_regions(*[annos[1][f][r] for r in REGIONS])[2]) for f in common]
        k_risk = cohens_kappa(ra, rb)
        W(f"| **5-class risk level** | **{k_risk:.3f}** | **{kappa_verdict(k_risk)}** | "
          f"**{100.0*sum(1 for x,y in zip(ra,rb) if x==y)/len(ra):.1f}%** |")
        W("\nRead this table before the next one. A kappa below 0.60 on the risk level means "
          "the human reference is itself unreliable, and any system score measured against it "
          "is not interpretable. Tighten the annotation guide and re-annotate before drawing "
          "conclusions about the system.\n")
    else:
        W("_Only one annotator supplied - reference quality is unmeasured._\n")

    # ---------------- consensus ----------------
    consensus, dropped = {}, 0
    for f in common:
        merged = {}
        for region in REGIONS:
            values = [rows[f][region] for rows in annos]
            if len(set(values)) == 1:
                merged[region] = values[0]
            elif args.consensus == "max":
                merged[region] = max(values)
            else:
                merged = None
                break
        if merged is None:
            dropped += 1
        else:
            consensus[f] = merged

    W(f"\n## 2. Consensus reference\n")
    W(f"Mode: `{args.consensus}`. Frames in consensus: **{len(consensus)}**"
      + (f" ({dropped} dropped for disagreement)\n" if dropped else "\n"))
    if not consensus:
        raise SystemExit("Consensus is empty. Try --consensus max, or re-annotate.")

    # ---------------- system vs consensus ----------------
    diffs, sys_risks, ref_risks, per_completeness = [], [], [], defaultdict(list)
    region_hits = {r: [0, 0] for r in REGIONS}

    for f, ref in consensus.items():
        srow = system[f]
        if int(srow["score_c"]) == SCORE_NOT_ASSESSED:
            continue
        _, _, ref_c = reba_from_regions(*[ref[r] for r in REGIONS])
        sys_c = int(srow["score_c"])

        diffs.append(sys_c - ref_c)
        sys_risks.append(risk_class(sys_c))
        ref_risks.append(risk_class(ref_c))

        try:
            comp = float(srow["completeness"])
        except (TypeError, ValueError):
            comp = -1.0
        band = "full (>=90%)" if comp >= 0.90 else "high (75-90%)" if comp >= 0.75 \
            else "partial (50-75%)" if comp >= 0.50 else "sparse (<50%)"
        per_completeness[band].append(sys_c - ref_c)

        for region in REGIONS:
            sv = int(srow[region])
            if sv != SCORE_NOT_ASSESSED:
                region_hits[region][1] += 1
                if sv == ref[region]:
                    region_hits[region][0] += 1

    n = len(diffs)
    if not n:
        raise SystemExit("No frames where the system produced a final score.")

    exact = 100.0 * sum(1 for d in diffs if d == 0) / n
    within1 = 100.0 * sum(1 for d in diffs if abs(d) <= 1) / n
    mae = sum(abs(d) for d in diffs) / n
    bias = sum(diffs) / n
    risk_exact = 100.0 * sum(1 for a, b in zip(sys_risks, ref_risks) if a == b) / n

    W("\n## 3. System vs. human consensus\n")
    W("| Metric | Value |")
    W("| --- | ---: |")
    W(f"| Frames scored | {n} |")
    W(f"| Exact REBA agreement | {exact:.1f}% |")
    W(f"| Within ±1 | {within1:.1f}% |")
    W(f"| **5-class risk agreement** | **{risk_exact:.1f}%** |")
    W(f"| MAE | {mae:.2f} |")
    W(f"| Directional bias | {bias:+.2f} |")
    W("\nA negative bias means the system scores lower than a human would, i.e. it "
      "under-reports risk. That is the dangerous direction for an ergonomics tool.\n")

    # ---------------- confusion matrix ----------------
    W("\n### 5-class risk confusion matrix\n")
    W("Rows: human consensus. Columns: system.\n")
    W("| Human \\\\ System | " + " | ".join(RISK_NAMES) + " | total |")
    W("| --- | " + " | ".join(["---:"] * (len(RISK_NAMES) + 1)) + " |")
    matrix = [[0] * 5 for _ in range(5)]
    for s, r in zip(sys_risks, ref_risks):
        matrix[r][s] += 1
    for i, name in enumerate(RISK_NAMES):
        total = sum(matrix[i])
        W(f"| {name} | " + " | ".join(str(v) for v in matrix[i]) + f" | {total} |")

    # ---------------- high-risk recall ----------------
    hi_total = sum(1 for r in ref_risks if r >= 3)
    hi_caught = sum(1 for s, r in zip(sys_risks, ref_risks) if r >= 3 and s >= 3)
    W("\n### High-risk detection\n")
    if hi_total:
        W(f"- Frames a human rated High or Very High: **{hi_total}**")
        W(f"- Of those, flagged as High or Very High by the system: **{hi_caught}** "
          f"(**{100.0*hi_caught/hi_total:.1f}%** recall)")
        W(f"- **Missed high-risk frames: {hi_total - hi_caught}**")
        W("\nThis is the number to lead with. In ergonomics a missed dangerous posture "
          "costs more than a false alarm, and rare classes barely move overall accuracy.\n")
    else:
        W("_No frame was rated High or Very High by the consensus. This sample cannot "
          "say anything about high-risk detection - annotate footage containing genuinely "
          "demanding postures before claiming the system finds them._\n")

    # ---------------- per-region ----------------
    W("\n### Per-region agreement (system vs consensus)\n")
    W("| Region | Agreement | Frames assessed |")
    W("| --- | ---: | ---: |")
    for region in REGIONS:
        hit, tot = region_hits[region]
        label = region.replace("_score", "").replace("_", " ")
        W(f"| {label} | {100.0*hit/tot:.1f}% | {tot} |" if tot else f"| {label} | n/a | 0 |")
    W("\nRegions the system never observed are excluded rather than counted as wrong.\n")

    # ---------------- completeness ----------------
    W("\n### Accuracy by skeleton completeness\n")
    W("| Completeness | Frames | Exact | Within ±1 | MAE | Bias |")
    W("| --- | ---: | ---: | ---: | ---: | ---: |")
    for band in ["full (>=90%)", "high (75-90%)", "partial (50-75%)", "sparse (<50%)"]:
        d = per_completeness.get(band)
        if not d:
            continue
        W(f"| {band} | {len(d)} | {100.0*sum(1 for x in d if x==0)/len(d):.1f}% | "
          f"{100.0*sum(1 for x in d if abs(x)<=1)/len(d):.1f}% | "
          f"{sum(abs(x) for x in d)/len(d):.2f} | {sum(d)/len(d):+.2f} |")
    W("\nHandling incomplete skeletons is a stated requirement, so this breakdown belongs "
      "in the report rather than being averaged away.\n")

    W("\n## 4. Scope of this measurement\n")
    W("- Wrist is fixed at 1 and load / coupling / activity at 0 on **both** sides, "
      "because the vision pipeline cannot observe them. The comparison is therefore about "
      "posture geometry, not the complete REBA instrument.")
    W("- The reference is human judgement from RGB footage, not instrumented ground truth. "
      "Its quality is the kappa in section 1.")
    W("- Frames where the system produced no final score are excluded from section 3 and "
      "should be reported separately as coverage, not as errors.\n")

    out = Path(args.out).expanduser()
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {out}")
    print(f"  exact {exact:.1f}%   within±1 {within1:.1f}%   risk {risk_exact:.1f}%   "
          f"MAE {mae:.2f}   bias {bias:+.2f}")
    if hi_total:
        print(f"  high-risk recall {100.0*hi_caught/hi_total:.1f}% ({hi_total-hi_caught} missed)")


if __name__ == "__main__":
    main()
