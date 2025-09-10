"""
train_cnn.py: Train a simple 2D CNN on log-mel spectrogram features for jingle detection.
"""
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix


class MelDataset(Dataset):
    def __init__(self, npz_path, meta_path, oversample_pos=False, augment_pos=False):
        data = np.load(npz_path)
        X = data['X_mel']
        y = data['y']
        # fallback for shape (N, 1, H, W) or (N, H, W)
        if X.ndim == 3:
            X = X[:, None, :, :]
        # Oversample positives if requested
        if oversample_pos:
            pos_idx = np.where(y == 1)[0]
            neg_idx = np.where(y == 0)[0]
            n_pos, n_neg = len(pos_idx), len(neg_idx)
            if n_pos > 0 and n_neg > n_pos:
                reps = int(np.ceil(n_neg / n_pos))
                pos_idx_oversampled = np.tile(pos_idx, reps)[:n_neg]
                idx = np.concatenate([neg_idx, pos_idx_oversampled])
                np.random.shuffle(idx)
                X = X[idx]
                y = y[idx]
        self.X = X
        self.y = y
        self.augment_pos = augment_pos
        self.meta = None
        if os.path.exists(meta_path):
            import pandas as pd
            self.meta = pd.read_csv(meta_path)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = int(self.y[idx])
        # Stronger augmentation for positives (train only)
        if self.augment_pos and y == 1:
            x = self.strong_augment(x)
        return torch.tensor(x, dtype=torch.float32), y

    def strong_augment(self, x):
        # x: (1, H, W) numpy array
        # Random time masking
        x = x.copy()
        _, h, w = x.shape
        # Time mask
        if np.random.rand() < 0.7:
            t0 = np.random.randint(0, w // 2)
            t1 = t0 + np.random.randint(1, w // 4)
            x[:, :, t0:t1] = 0
        # Frequency mask
        if np.random.rand() < 0.7:
            f0 = np.random.randint(0, h // 2)
            f1 = f0 + np.random.randint(1, h // 4)
            x[:, f0:f1, :] = 0
        # Additive Gaussian noise
        if np.random.rand() < 0.7:
            x += np.random.normal(0, 0.1 * np.std(x), size=x.shape)
        return x


class SimpleCNN(nn.Module):
    def __init__(self, in_channels=1, n_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3,
                      padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 64), nn.ReLU(),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        return self.net(x)


def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    threshold = 0.3  # Lowered threshold for positive class
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            logits = model(X)
            # Probability for class 1
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            pred = (probs >= threshold).astype(int)
            y_pred.extend(pred)
            y_true.extend(y.numpy())
    return np.array(y_true), np.array(y_pred)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features_dir', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--model_out', type=str,
                        default='./models/cnn_model.pt')
    args = parser.parse_args()

    train_set = MelDataset(
        os.path.join(args.features_dir, 'train_features.npz'),
        os.path.join(args.features_dir, 'train_meta.csv'),
        oversample_pos=True, augment_pos=True)
    val_set = MelDataset(
        os.path.join(args.features_dir, 'val_features.npz'),
        os.path.join(args.features_dir, 'val_meta.csv'))
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)

    # Compute class weights (inverse frequency)
    y_train = train_set.y
    n_pos = np.sum(y_train == 1)
    n_neg = np.sum(y_train == 0)
    weight = torch.tensor([1.0 / n_neg, 1.0 / n_pos], dtype=torch.float32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(weight=weight.to(device))

    best_f1 = 0
    for epoch in range(args.epochs):
        loss = train(model, train_loader, optimizer, criterion, device)
        y_true, y_pred = evaluate(model, val_loader, device)
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0)
        print(
            f"Epoch {epoch+1}: loss={loss:.4f} val_f1={f1:.3f} val_p={p:.3f} val_r={r:.3f}")
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), args.model_out)
            print(f"[INFO] Saved best model to {args.model_out}")
    # Final confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix:\n", cm)


if __name__ == '__main__':
    main()
