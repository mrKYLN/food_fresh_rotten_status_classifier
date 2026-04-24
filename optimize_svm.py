"""
optimize_svm.py — GridSearchCV on cached PCA features
=====================================================
Finds optimal SVM hyperparameters without re-extracting CNN features.
Updates the saved model with the best SVM.

Usage:
    python optimize_svm.py
"""
import matplotlib
matplotlib.use("Agg")

import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from pathlib import Path

from src.config import MODELS_DIR, RANDOM_SEED, CLASSES
from src.hybrid_model import HybridCNNSOM
from src.evaluate import compute_metrics

TRAIN_CACHE = MODELS_DIR / "feature_cache" / "train_features.npz"
TEST_CACHE  = MODELS_DIR / "feature_cache" / "test_features.npz"


def main():
    print("\n" + "=" * 58)
    print("  SVM GridSearchCV — Tomato Freshness Classifier")
    print("=" * 58)

    # Load cached features
    print("\n[1] Loading cached CNN features...")
    tr = np.load(TRAIN_CACHE)
    te = np.load(TEST_CACHE)
    X_train_raw, y_train = tr["features"], tr["labels"]
    X_test_raw,  y_test  = te["features"], te["labels"]

    # Load existing model for preprocessor (scaler + pca already fitted)
    print("[2] Loading existing model for preprocessing pipeline...")
    model = HybridCNNSOM.load(MODELS_DIR / "hybrid_cnn_som.pkl")

    X_train_pca = model._transform_preprocess(X_train_raw)
    X_test_pca  = model._transform_preprocess(X_test_raw)
    print(f"  PCA shape: {X_train_pca.shape}")

    # ── Grid search ────────────────────────────────────────────────────
    param_grid = {
        "C":      [0.1, 1, 5, 10, 50, 100],
        "gamma":  ["scale", "auto", 0.001, 0.005, 0.01],
        "kernel": ["rbf"],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    print(f"\n[3] Running GridSearchCV ({len(param_grid['C']) * len(param_grid['gamma'])} combos × 5-fold)...")
    gs = GridSearchCV(
        SVC(probability=True, random_state=RANDOM_SEED),
        param_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    gs.fit(X_train_pca, y_train)

    print(f"\n  Best params : {gs.best_params_}")
    print(f"  Best CV F1  : {gs.best_score_:.4f}")

    # ── Evaluate best estimator on hold-out test set ───────────────────
    print("\n[4] Test-set evaluation with best estimator...")
    y_test_pred = gs.best_estimator_.predict(X_test_pca)
    metrics = compute_metrics(y_test, y_test_pred, CLASSES, split="Test (best SVM)")

    # ── Compare vs current model ────────────────────────────────────────
    print("\n[5] Comparison:")
    cur_pred = model.predict(X_test_raw)
    cur_metrics = compute_metrics(y_test, cur_pred, CLASSES, split="Current model")
    delta = metrics["f1_macro"] - cur_metrics["f1_macro"]
    print(f"\n  F1 delta (best vs current): {delta:+.4f}")

    # ── Optionally update saved model ─────────────────────────────────
    if delta > 0.0001:
        print(f"\n  Updating model SVM with best params: {gs.best_params_}")
        model.svm = gs.best_estimator_
        model.svm_C     = gs.best_params_["C"]
        model.svm_gamma = gs.best_params_["gamma"]
        model.save()
        print("  Model saved with improved SVM.")
    else:
        print("\n  Current SVM is already optimal. No update needed.")

    print("\n" + "=" * 58)
    return gs.best_params_, metrics


if __name__ == "__main__":
    main()
