"""
train_deba_model.py

DEBA (Differentiable Ergonomic Body Assessment) Modeli Eğitim Scripti
Özellikler:
- npz formatındaki dengelenmiş sentetik veriyi kullanır.
- CPU üzerinde çok hızlı eğitilebilen hafif bir MLP mimarisi (Linear -> ReLU) içerir.
- Sürekli REBA skorunu (Continuous Score) tahmin etmek için MSE Loss kullanır.
- Eğitim sonrası en iyi ağırlıkları ROS 2 node'unda kullanılmak üzere .pth olarak kaydeder.
"""

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# --------------------------------------------------------------------------
# 1. PyTorch Veri Yükleyici (Dataset)
# --------------------------------------------------------------------------
class DebaDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        # 18 adet feature (sürekli açılar, binary bayraklar, coupling_score)
        self.features = torch.tensor(data["features"], dtype=torch.float32)
        
        # LABEL_FIELDS: ["score_a", "score_b", "activity_score", "score_c_raw", "final_reba_score"]
        # Hedefimiz sadece 4. indeksteki final_reba_score
        self.labels = torch.tensor(data["labels"][:, 4], dtype=torch.float32).unsqueeze(1)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# --------------------------------------------------------------------------
# 2. MLP Model Mimarisi
# --------------------------------------------------------------------------
class DebaMLP(nn.Module):
    def __init__(self, input_dim=18):
        super(DebaMLP, self).__init__()
        # Hafif ve hızlı yapı: 18 girdi -> 64 -> 64 -> 1 sürekli çıktı
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.network(x)

# --------------------------------------------------------------------------
# 3. Eğitim ve Validasyon Döngüsü
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="DEBA MLP Eğitim Scripti")
    parser.add_argument("--data-dir", type=str, default="./deba_dataset", help="npz dosyalarının bulunduğu klasör")
    parser.add_argument("--epochs", type=int, default=30, help="Eğitim turu sayısı")
    parser.add_argument("--batch-size", type=int, default=256, help="Veri yığın boyutu")
    parser.add_argument("--lr", type=float, default=0.001, help="Öğrenme hızı (Learning Rate)")
    parser.add_argument("--out-model", type=str, default="deba_model.pth", help="Kaydedilecek model adı")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train_path = data_dir / "deba_train.npz"
    val_path = data_dir / "deba_val.npz"

    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(f"Veri dosyaları bulunamadı! Lütfen {data_dir} yolunu kontrol et.")

    print("[1/4] Veri setleri belleğe yükleniyor (DataLoader)...")
    train_dataset = DebaDataset(train_path)
    val_dataset = DebaDataset(val_path)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"      Train: {len(train_dataset)} örnek, Val: {len(val_dataset)} örnek")

    print("[2/4] Model, Optimize Edici ve Kayıp Fonksiyonu (MSE) hazırlanıyor...")
    # RTX 4050 var ama bu model o kadar küçük ki CPU'da bile saniyeler sürer.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DebaMLP(input_dim=18).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print(f"      Eğitim {device.type.upper()} üzerinde başlıyor...\n")

    best_val_loss = float('inf')

    # Eğitim Döngüsü (Epochs)
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * features.size(0)
            
        train_loss /= len(train_dataset)

        # Validasyon Döngüsü
        model.eval()
        val_loss = 0.0
        correct_rounded = 0
        
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                
                loss = criterion(outputs, labels)
                val_loss += loss.item() * features.size(0)
                
                # Rounded Agreement (Yuvarlanmış Uyum): Tahmin yuvarlandığında gerçek skoru tutuyor mu?
                rounded_preds = torch.round(outputs)
                correct_rounded += (rounded_preds == labels).sum().item()

        val_loss /= len(val_dataset)
        accuracy = (correct_rounded / len(val_dataset)) * 100

        print(f"Epoch {epoch+1:02d}/{args.epochs} | "
              f"Train MSE: {train_loss:.4f} | "
              f"Val MSE: {val_loss:.4f} | "
              f"Rounded Agreement: %{accuracy:.2f}")

        # En iyi modeli kaydet
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.out_model)

    print(f"\n[4/4] Eğitim tamamlandı! En iyi model '{args.out_model}' olarak kaydedildi.")
    print(f"      En düşük Validation MSE: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()