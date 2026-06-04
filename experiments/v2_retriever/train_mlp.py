"""Part 2.1 — train a 2-layer MLP candidate-generation model.

Split by seeds (no episode leakage):
  Train: seeds 2, 3, 4, 5, 6, 7   (6 seeds)
  Val:   seeds 8, 9                (2 seeds)
  Test:  seeds 10, 11               (2 seeds)
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(THIS_DIR, "dataset_v2.npz")


def main():
    d = np.load(DATA, allow_pickle=True)
    X = d["X"]
    y = d["y"]
    seed = d["seed"]
    feat_names = d["feature_names"]
    print(f"Loaded dataset X={X.shape}, y={y.shape}")
    print(f"Features ({len(feat_names)}): {list(feat_names)}")

    train_seeds = {2, 3, 4, 5, 6, 7}
    val_seeds = {8, 9}
    test_seeds = {10, 11}

    train_mask = np.array([s in train_seeds for s in seed])
    val_mask = np.array([s in val_seeds for s in seed])
    test_mask = np.array([s in test_seeds for s in seed])

    X_tr, y_tr = X[train_mask], y[train_mask]
    X_va, y_va = X[val_mask], y[val_mask]
    X_te, y_te = X[test_mask], y[test_mask]
    print(f"\nTrain: {len(y_tr)} (pos={y_tr.sum()}/{len(y_tr)} = {y_tr.mean()*100:.1f}%)")
    print(f"Val:   {len(y_va)} (pos={y_va.sum()}/{len(y_va)} = {y_va.mean()*100:.1f}%)")
    print(f"Test:  {len(y_te)} (pos={y_te.sum()}/{len(y_te)} = {y_te.mean()*100:.1f}%)")

    # Z-score normalisation from train
    mean = X_tr.mean(axis=0)
    std = X_tr.std(axis=0) + 1e-6
    X_tr_n = (X_tr - mean) / std
    X_va_n = (X_va - mean) / std
    X_te_n = (X_te - mean) / std

    d_in = X.shape[1]
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(d_in, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    )
    pos_weight = torch.tensor(
        max(1.0, (len(y_tr) - y_tr.sum()) / max(1, y_tr.sum())), dtype=torch.float32
    )
    print(f"\nclass-imbalance pos_weight: {pos_weight.item():.2f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)

    xtr = torch.tensor(X_tr_n, dtype=torch.float32)
    ytr = torch.tensor(y_tr, dtype=torch.float32)
    xva = torch.tensor(X_va_n, dtype=torch.float32)
    yva = torch.tensor(y_va, dtype=torch.float32)
    xte = torch.tensor(X_te_n, dtype=torch.float32)
    yte = torch.tensor(y_te, dtype=torch.float32)

    def _acc(model, x, y):
        model.eval()
        with torch.no_grad():
            pred = (torch.sigmoid(model(x)).squeeze() > 0.5).float()
        return (pred == y).float().mean().item()

    def _ap_at_recall(model, x, y, recall_target=0.7):
        """At fixed recall, what is the precision?"""
        model.eval()
        with torch.no_grad():
            scores = torch.sigmoid(model(x)).squeeze().numpy()
        order = np.argsort(-scores)
        sorted_y = y.numpy()[order]
        tp_cum = np.cumsum(sorted_y)
        n_pos = sorted_y.sum()
        if n_pos == 0:
            return float("nan")
        recall = tp_cum / n_pos
        precision = tp_cum / np.arange(1, len(sorted_y) + 1)
        ok = recall >= recall_target
        if not ok.any():
            return float("nan")
        idx = np.argmax(ok)
        return float(precision[idx])

    best_val_loss = float("inf")
    best_state = None
    n_epochs = 200
    batch = 64
    print("\nTraining...")
    for epoch in range(n_epochs):
        model.train()
        idx = torch.randperm(len(xtr))
        for i in range(0, len(xtr), batch):
            j = idx[i:i+batch]
            opt.zero_grad()
            out = model(xtr[j]).squeeze()
            loss = criterion(out, ytr[j])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(xva).squeeze(), yva).item()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 20 == 0:
            print(f"  epoch {epoch:3d}  val_loss={val_loss:.4f}  "
                  f"train_acc={_acc(model, xtr, ytr):.3f}  "
                  f"val_acc={_acc(model, xva, yva):.3f}")

    model.load_state_dict(best_state)
    print(f"\nBest val_loss={best_val_loss:.4f}")
    print(f"Train acc: {_acc(model, xtr, ytr):.3f}")
    print(f"Val   acc: {_acc(model, xva, yva):.3f}")
    print(f"Test  acc: {_acc(model, xte, yte):.3f}")
    print()
    p_at_r07_train = _ap_at_recall(model, xtr, ytr, 0.7)
    p_at_r07_val = _ap_at_recall(model, xva, yva, 0.7)
    p_at_r07_test = _ap_at_recall(model, xte, yte, 0.7)
    print(f"Train precision@recall=0.70: {p_at_r07_train:.3f}")
    print(f"Val   precision@recall=0.70: {p_at_r07_val:.3f}")
    print(f"Test  precision@recall=0.70: {p_at_r07_test:.3f}")

    out_path = os.path.join(THIS_DIR, "candidate_model.pt")
    torch.save({
        "state_dict": model.state_dict(),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "feature_names": list(feat_names),
    }, out_path)
    print(f"\nSaved model → {out_path}")


if __name__ == "__main__":
    main()
