import argparse
import csv
import random
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from scipy.stats import qmc

try:
    from reba import RebaScore
except ImportError as e:
    raise ImportError(
        "reba.py bulunamadı. Bu script'i reba.py ile aynı dizine koyun.\n"
        f"Orijinal hata: {e}"
    )

CONTINUOUS_RANGES = {
    "neck_angle": (-45.0, 75.0),
    "trunk_angle": (0.0, 130.0),
    "legs_angle": (0.0, 170.0),
    "load_kg": (0.0, 25.0),
    "upper_arm_angle": (-60.0, 170.0),
    "lower_arm_angle": (0.0, 180.0),
    "wrist_angle": (-45.0, 45.0),
}

# Narrow bands around a neutral standing posture. Plain Latin Hypercube over
# CONTINUOUS_RANGES practically never lands on the low-risk corner (all seven
# variables neutral at once), so the teacher produced no REBA 1 labels at all
# and the student could not represent "negligible risk". A fraction of every
# batch is drawn from here instead.
NEUTRAL_RANGES = {
    "neck_angle": (-5.0, 20.0),
    "trunk_angle": (0.0, 20.0),
    "legs_angle": (0.0, 30.0),
    "load_kg": (0.0, 5.0),
    "upper_arm_angle": (-20.0, 20.0),
    "lower_arm_angle": (60.0, 100.0),
    "wrist_angle": (-15.0, 15.0),
}
NEUTRAL_FLAG_PROB = 0.1

BINARY_FIELDS = [
    "neck_side", "trunk_side", "legs_unstable", "shoulder_raised",
    "arm_abducted", "leaning", "wrist_twisted",
    "activity_static", "activity_repeated", "activity_rapid_change"
]

CATEGORICAL_FIELDS = {
    "coupling_score": (0, 1, 2, 3),
}

ALL_FEATURE_FIELDS = list(CONTINUOUS_RANGES.keys()) + BINARY_FIELDS + list(CATEGORICAL_FIELDS.keys())
LABEL_FIELDS = ["score_a", "score_b", "activity_score", "score_c_raw", "final_reba_score"]


@dataclass
class RebaFeatureSample:
    neck_angle: float
    trunk_angle: float
    legs_angle: float
    load_kg: float
    upper_arm_angle: float
    lower_arm_angle: float
    wrist_angle: float
    neck_side: int
    trunk_side: int
    legs_unstable: int
    shoulder_raised: int
    arm_abducted: int
    leaning: int
    wrist_twisted: int
    activity_static: int
    activity_repeated: int
    activity_rapid_change: int
    coupling_score: int


def sample_dataset_lhs(n_samples: int, seed: int = 42, include_wrist: bool = True) -> list:
    """Latin Hypercube Sampling kullanarak uzayı çok daha homojen tarar."""
    d_cont = len(CONTINUOUS_RANGES)
    d_bin = len(BINARY_FIELDS)
    d_cat = len(CATEGORICAL_FIELDS)
    
    sampler = qmc.LatinHypercube(d=d_cont + d_bin + d_cat, seed=seed)
    lhs_points = sampler.random(n=n_samples)
    
    samples = []
    for point in lhs_points:
        values = {}
        
        # Sürekli değişkenler
        for i, (name, (lo, hi)) in enumerate(CONTINUOUS_RANGES.items()):
            if name == "wrist_angle" and not include_wrist:
                values[name] = 0.0
            else:
                values[name] = lo + (hi - lo) * point[i]
                
        # İkili değişkenler
        for i, name in enumerate(BINARY_FIELDS):
            idx = d_cont + i
            if name == "wrist_twisted" and not include_wrist:
                values[name] = 0
            else:
                values[name] = 1 if point[idx] > 0.5 else 0
                
        # Kategorik değişkenler
        for i, (name, opts) in enumerate(CATEGORICAL_FIELDS.items()):
            idx = d_cont + d_bin + i
            opt_idx = int(point[idx] * len(opts))
            opt_idx = min(opt_idx, len(opts) - 1)
            values[name] = opts[opt_idx]
            
        samples.append(RebaFeatureSample(**values))

    return samples


def sample_neutral(n_samples: int, seed: int = 42, include_wrist: bool = True) -> list:
    """
    Draw postures from the low-risk corner of the feature space, which uniform
    LHS over the full ranges misses almost entirely.
    """
    rng = random.Random(seed + 1)
    samples = []
    for _ in range(n_samples):
        values = {}
        for name, (lo, hi) in NEUTRAL_RANGES.items():
            if name == "wrist_angle" and not include_wrist:
                values[name] = 0.0
            else:
                values[name] = rng.uniform(lo, hi)
        for name in BINARY_FIELDS:
            if name == "wrist_twisted" and not include_wrist:
                values[name] = 0
            else:
                values[name] = 1 if rng.random() < NEUTRAL_FLAG_PROB else 0
        for name, opts in CATEGORICAL_FIELDS.items():
            values[name] = opts[0] if rng.random() > NEUTRAL_FLAG_PROB else rng.choice(opts)
        samples.append(RebaFeatureSample(**values))
    return samples


def compute_reba_label(sample: RebaFeatureSample) -> dict:
    reba = RebaScore()
    body_values = [
        sample.neck_angle, sample.neck_side, sample.trunk_angle,
        sample.trunk_side, sample.legs_unstable, sample.legs_angle, sample.load_kg
    ]
    # Argümanların liste olarak mı yoksa tek tek mi beklendiğine dair güvenli yapı:
    try:
        reba.set_body(body_values)
    except TypeError:
        reba.set_body(*body_values)

    score_a, _ = reba.compute_score_a()

    arm_values = [
        sample.upper_arm_angle, sample.shoulder_raised, sample.arm_abducted,
        sample.leaning, sample.lower_arm_angle, sample.wrist_angle,
        sample.wrist_twisted, sample.coupling_score
    ]
    try:
        reba.set_arms(arm_values)
    except TypeError:
        reba.set_arms(*arm_values)

    score_b, _ = reba.compute_score_b()
    activity_score = reba.compute_activity_score(
        static_posture=bool(sample.activity_static),
        repeated_action=bool(sample.activity_repeated),
        rapid_large_change=bool(sample.activity_rapid_change),
    )

    score_c_raw, final_score, caption = reba.compute_score_c(score_a, score_b, activity_score)

    return {
        "score_a": int(score_a), "score_b": int(score_b),
        "activity_score": int(activity_score), "score_c_raw": int(score_c_raw),
        "final_reba_score": int(final_score), "risk_caption": caption
    }


def inject_noise(sample: RebaFeatureSample, include_wrist: bool) -> RebaFeatureSample:
    """Oversampled verilerdeki sürekli değerlere %2 oranında Gaussian gürültü ekler."""
    noisy_values = {}
    for name in CONTINUOUS_RANGES.keys():
        val = getattr(sample, name)
        if name == "wrist_angle" and not include_wrist:
            noisy_values[name] = 0.0
        else:
            lo, hi = CONTINUOUS_RANGES[name]
            noise = random.gauss(0, (hi - lo) * 0.02)
            noisy_values[name] = np.clip(val + noise, lo, hi)
            
    return replace(sample, **noisy_values)


def balance_by_label(samples: list, labels: list, target_per_bin: int, seed: int = 42, include_wrist: bool = True):
    """
    Even out the label histogram by oversampling rare score bins.

    Every oversampled row is RE-LABELLED after noise injection. Copying the
    base sample's label instead corrupted 14.5% of the previous dataset: REBA
    is a step function, so a 2% perturbation routinely crosses a category
    boundary and the stored label no longer describes the stored features.
    That put a hard ceiling of ~85% on any student model's rounded agreement.

    Because relabelling can move a row into a different bin, the resulting
    histogram is approximately, not exactly, flat. That is the correct
    trade-off: a truthful label matters more than an exact bin count.
    """
    rng = random.Random(seed)
    bins: dict = {}
    for idx, label in enumerate(labels):
        key = label["final_reba_score"]
        bins.setdefault(key, []).append(idx)

    balanced_samples, balanced_labels = [], []
    for score_bin, idx_list in sorted(bins.items()):
        if len(idx_list) >= target_per_bin:
            chosen = rng.sample(idx_list, target_per_bin)
            for i in chosen:
                balanced_samples.append(samples[i])
                balanced_labels.append(labels[i])
        else:
            for _ in range(target_per_bin):
                i = rng.choice(idx_list)
                noisy_sample = inject_noise(samples[i], include_wrist)
                balanced_samples.append(noisy_sample)
                balanced_labels.append(compute_reba_label(noisy_sample))

    return balanced_samples, balanced_labels


def train_val_split(samples: list, labels: list, val_ratio: float, seed: int = 42):
    rng = random.Random(seed)
    # Sample ve labelları birbirine bağlayıp komple karıştırıyoruz
    combined = list(zip(samples, labels))
    rng.shuffle(combined)
    
    n_val = int(len(combined) * val_ratio)
    val_set = combined[:n_val]
    train_set = combined[n_val:]
    
    train_s = [item[0] for item in train_set]
    train_l = [item[1] for item in train_set]
    val_s = [item[0] for item in val_set]
    val_l = [item[1] for item in val_set]
    
    return train_s, train_l, val_s, val_l


def write_csv(path: Path, samples: list, labels: list) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(ALL_FEATURE_FIELDS + LABEL_FIELDS)
        for sample, label in zip(samples, labels):
            writer.writerow([getattr(sample, name) for name in ALL_FEATURE_FIELDS] + 
                            [label[name] for name in LABEL_FIELDS])


def write_npz(path: Path, samples: list, labels: list) -> None:
    feature_matrix = np.array([[getattr(s, n) for n in ALL_FEATURE_FIELDS] for s in samples], dtype=np.float32)
    label_matrix = np.array([[l[n] for n in LABEL_FIELDS] for l in labels], dtype=np.float32)
    np.savez(path, features=feature_matrix, labels=label_matrix, 
             feature_names=np.array(ALL_FEATURE_FIELDS), label_names=np.array(LABEL_FIELDS))


def main():
    parser = argparse.ArgumentParser(description="DEBA için sentetik REBA veri seti üretici")
    parser.add_argument("--n-raw-samples", type=int, default=300_000)
    parser.add_argument("--target-per-bin", type=int, default=8_000)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--neutral-fraction", type=float, default=0.1,
                        help="Share of raw samples drawn from the neutral/low-risk band")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="./deba_dataset")
    parser.add_argument("--exclude-wrist", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    include_wrist = not args.exclude_wrist

    n_neutral = int(args.n_raw_samples * args.neutral_fraction)
    n_lhs = args.n_raw_samples - n_neutral

    print(f"[1/5] {n_lhs} LHS + {n_neutral} neutral-band ham örnek üretiliyor...")
    raw_samples = sample_dataset_lhs(n_lhs, args.seed, include_wrist)
    raw_samples += sample_neutral(n_neutral, args.seed, include_wrist)

    print("[2/5] Örnekler etiketleniyor...")
    raw_labels = [compute_reba_label(s) for s in raw_samples]

    # ORDER MATTERS. Splitting AFTER balancing leaked the validation set: an
    # oversampled row and its 2%-noise siblings landed on both sides of the
    # split, so ~48% of validation rows had a near-duplicate in training with
    # an identical label. The model memorised those copies and the reported
    # validation MSE dropped below the dataset's own label-noise floor.
    # Splitting the RAW pool first keeps the held-out set genuinely unseen.
    print(f"[3/5] Ham havuz train/val olarak ayrılıyor (val oranı: {args.val_ratio})...")
    train_raw_s, train_raw_l, val_s, val_l = train_val_split(
        raw_samples, raw_labels, args.val_ratio, args.seed
    )

    # Balancing is a TRAINING-side technique only. The validation set keeps the
    # teacher's natural label distribution so the reported score describes
    # generalisation rather than a re-weighted artefact.
    print(f"[4/5] Sadece train tarafı dengeleniyor (bin başına hedef: {args.target_per_bin})...")
    train_s, train_l = balance_by_label(
        train_raw_s, train_raw_l, args.target_per_bin, args.seed, include_wrist
    )
    rng = random.Random(args.seed)
    combined = list(zip(train_s, train_l))
    rng.shuffle(combined)
    train_s = [c[0] for c in combined]
    train_l = [c[1] for c in combined]

    print(f"[5/5] Diske yazılıyor -> {out_dir}")
    write_csv(out_dir / "deba_train.csv", train_s, train_l)
    write_csv(out_dir / "deba_val.csv", val_s, val_l)
    write_npz(out_dir / "deba_train.npz", train_s, train_l)
    write_npz(out_dir / "deba_val.npz", val_s, val_l)

    train_hist = Counter(l["final_reba_score"] for l in train_l)
    val_hist = Counter(l["final_reba_score"] for l in val_l)
    print(f"\nTamamlandı. Train: {len(train_s)}, Val: {len(val_s)}")
    print(f"  train label histogram: {dict(sorted(train_hist.items()))}")
    print(f"  val   label histogram: {dict(sorted(val_hist.items()))}")
    print("  NOTE: the val set is deliberately UNBALANCED - it is the held-out "
          "distribution, not a re-weighted one. Report per-class metrics on it, "
          "not just overall accuracy.")

if __name__ == "__main__":
    main()