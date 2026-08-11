"""
train_deba_model.py

DEBA (Differentiable Ergonomic Body Assessment) model training script.

  - Consumes the npz dataset produced by generate_deba_dataset.py.
  - Standardises inputs; the constants are stored inside the checkpoint so the
    ROS 2 node cannot drift out of sync with them.
  - Regresses the continuous REBA score with MSE loss.
  - Reports metrics honestly: the validation set is the held-out, UNBALANCED
    distribution, so overall accuracy alone is misleading and per-class
    agreement is printed alongside it. A mean-predictor baseline is printed
    too, because "MSE 0.5" means nothing without knowing the label variance.
"""

import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from deba_model_def import DebaMLP, save_checkpoint


class DebaDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.features = torch.tensor(data["features"], dtype=torch.float32)
        # LABEL_FIELDS: ["score_a","score_b","activity_score","score_c_raw","final_reba_score"]
        self.labels = torch.tensor(data["labels"][:, 4], dtype=torch.float32).unsqueeze(1)
        self.feature_names = [str(n) for n in data["feature_names"]]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def evaluate(model, loader, device):
    """Return (mse, overall rounded agreement, per-class agreement dict)."""
    model.eval()
    total_sq_err, n = 0.0, 0
    correct = 0
    per_class = defaultdict(lambda: [0, 0])   # label -> [correct, total]

    with torch.no_grad():
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)

            total_sq_err += ((outputs - labels) ** 2).sum().item()
            n += labels.numel()

            hit = (torch.round(outputs) == labels)
            correct += hit.sum().item()
            for lbl, ok in zip(labels.view(-1).tolist(), hit.view(-1).tolist()):
                bucket = per_class[int(lbl)]
                bucket[1] += 1
                bucket[0] += int(ok)

    return total_sq_err / n, 100.0 * correct / n, per_class


def main():
    parser = argparse.ArgumentParser(description="DEBA MLP training script")
    parser.add_argument("--data-dir", type=str, default="./deba_dataset")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--out-model", type=str, default="deba_model.pth")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train_path = data_dir / "deba_train.npz"
    val_path = data_dir / "deba_val.npz"

    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(f"Dataset files not found under {data_dir}.")

    print("[1/4] Loading datasets...")
    train_dataset = DebaDataset(train_path)
    val_dataset = DebaDataset(val_path)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"      Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")

    print("[2/4] Building model, optimiser and loss...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = train_dataset.features.shape[1]
    model = DebaMLP(input_dim=input_dim, hidden=args.hidden).to(device)

    # Normalisation constants come from the TRAINING split only - deriving them
    # from the full dataset would leak validation statistics into the model.
    model.set_normalisation(
        train_dataset.features.mean(dim=0),
        train_dataset.features.std(dim=0),
    )

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    val_labels = val_dataset.labels.view(-1)
    baseline_mse = ((val_labels - train_dataset.labels.mean()) ** 2).mean().item()
    print(f"      Baseline (predict train mean) val MSE: {baseline_mse:.4f}")
    print(f"      Training on {device.type.upper()}\n")

    best_val_loss = float('inf')
    best_report = None

    print("[3/4] Training...")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(features), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * features.size(0)
        train_loss /= len(train_dataset)

        val_mse, val_acc, per_class = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1:02d}/{args.epochs} | Train MSE: {train_loss:.4f} | "
              f"Val MSE: {val_mse:.4f} | Rounded agreement: {val_acc:.2f}%")

        if val_mse < best_val_loss:
            best_val_loss = val_mse
            best_report = (val_mse, val_acc, dict(per_class))
            save_checkpoint(model, args.out_model, train_dataset.feature_names)

    print(f"\n[4/4] Done. Best checkpoint written to '{args.out_model}'.")
    val_mse, val_acc, per_class = best_report
    print(f"      Val MSE            : {val_mse:.4f}  (baseline {baseline_mse:.4f})")
    print(f"      Rounded agreement  : {val_acc:.2f}%")
    print("      Per-class agreement on the held-out (unbalanced) val set:")
    for label in sorted(per_class):
        ok, total = per_class[label]
        print(f"        REBA {label:>2}: {100.0*ok/total:5.1f}%  (n={total})")
    print("\n      Report the per-class numbers, not just the overall figure: the "
          "val set is unbalanced on purpose, so a good overall score can hide "
          "poor accuracy exactly where risk is highest.")


if __name__ == "__main__":
    main()
