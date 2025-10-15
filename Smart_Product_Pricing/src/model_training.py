from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class OptimizedMLP(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.3):
        super().__init__()
        self.fc_in = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.res1 = ResidualBlock(256, dropout)
        self.fc_mid = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.res2 = ResidualBlock(128, dropout)
        self.fc_out = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc_in(x)
        x = self.res1(x)
        x = self.fc_mid(x)
        x = self.res2(x)
        return self.fc_out(x)


def prepare_features(
    embeddings: np.ndarray,
    pca_components: int = 512,
) -> Tuple[np.ndarray, StandardScaler, PCA]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(embeddings)
    pca = PCA(n_components=pca_components, random_state=42)
    X_reduced = pca.fit_transform(X_scaled)
    return X_reduced, scaler, pca


def train_single_model(
    X_reduced: np.ndarray,
    y: np.ndarray,
    seed: int,
    dropout: float = 0.3,
    lr: float = 3e-4,
    weight_decay: float = 1e-3,
    epochs: int = 400,
    batch_size: int = 128,
    early_stop_patience: int = 40,
) -> nn.Module:
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train, X_val, y_train, y_val = train_test_split(
        X_reduced, y, test_size=0.2, random_state=42
    )

    y_train = np.asarray(y_train)
    y_val = np.asarray(y_val)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

    model = OptimizedMLP(X_train_t.shape[1], dropout=dropout).to(device)
    criterion = nn.HuberLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    best_val = float("inf")
    patience = 0
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(X_train_t.size(0))
        total_loss = 0.0
        for i in range(0, X_train_t.size(0), batch_size):
            idx = perm[i:i + batch_size]
            bx = X_train_t[idx].to(device)
            by = y_train_t[idx].to(device)
            bx = bx + 0.01 * torch.randn_like(bx)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t.to(device)), y_val_t.to(device)).item()
        scheduler.step()

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = model.state_dict()
            patience = 0
        else:
            patience += 1
            if patience >= early_stop_patience:
                break

    model.load_state_dict(best_state)
    return model


def ensemble_predict(models: List[nn.Module], X_reduced: np.ndarray) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_t = torch.tensor(X_reduced, dtype=torch.float32).to(device)
    preds: List[np.ndarray] = []
    for m in models:
        m.eval()
        with torch.no_grad():
            p = m(X_t).cpu().numpy()
            preds.append(np.expm1(p))
    y = np.mean(preds, axis=0)
    return np.maximum(y, 0)


def save_artifacts(
    models: List[nn.Module],
    scaler: StandardScaler,
    pca: PCA,
    out_dir: str,
    model_prefix: str = "final_model",
) -> None:
    """Save ensemble weights, scaler, and PCA to disk."""
    import os

    os.makedirs(out_dir, exist_ok=True)
    # Save each model
    for i, m in enumerate(models):
        torch.save(m.state_dict(), os.path.join(out_dir, f"{model_prefix}_{i+1}.pth"))
    # Save scaler and PCA
    joblib.dump(scaler, os.path.join(out_dir, "scaler.pkl"))
    joblib.dump(pca, os.path.join(out_dir, "pca.pkl"))


def load_artifacts(
    input_dim: int,
    in_dir: str,
    model_prefix: str = "final_model",
    num_models: int = 5,
    dropout: float = 0.3,
) -> tuple[List[nn.Module], StandardScaler, PCA]:
    """Load ensemble weights, scaler, and PCA from disk."""
    import os

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models: List[nn.Module] = []
    for i in range(num_models):
        m = OptimizedMLP(input_dim, dropout=dropout).to(device)
        weights_path = os.path.join(in_dir, f"{model_prefix}_{i+1}.pth")
        m.load_state_dict(torch.load(weights_path, map_location=device))
        m.eval()
        models.append(m)
    scaler: StandardScaler = joblib.load(os.path.join(in_dir, "scaler.pkl"))
    pca: PCA = joblib.load(os.path.join(in_dir, "pca.pkl"))
    return models, scaler, pca


def predict_from_embeddings(
    embeddings: np.ndarray,
    models: List[nn.Module],
    scaler: StandardScaler,
    pca: PCA,
) -> np.ndarray:
    """Full inference: scale → PCA → ensemble → non-negative prices."""
    X_scaled = scaler.transform(embeddings)
    X_reduced = pca.transform(X_scaled)
    return ensemble_predict(models, X_reduced)


