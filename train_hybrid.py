"""Hybrid CNN-SOM eğitim pipeline'ı. CNN özellikleri ilk çalıştırmada cache'lenir."""
import matplotlib
matplotlib.use("Agg")  # display gerektirmeden plot kaydet

import sys
import numpy as np
from pathlib import Path

from src.config import MODELS_DIR, RANDOM_SEED, CLASSES
from src.data_loader import DataLoader
from src.feature_extractor import MobileNetV2Extractor
from src.hybrid_model import HybridCNNSOM
from src.evaluate import (
    compute_metrics,
    plot_confusion_matrix,
    plot_som_map,
    plot_pca_variance,
    plot_pca_scatter,
)

TRAIN_CACHE = MODELS_DIR / "feature_cache" / "train_features.npz"
TEST_CACHE  = MODELS_DIR / "feature_cache" / "test_features.npz"


def load_or_extract_features(extractor: MobileNetV2Extractor,
                              loader: DataLoader,
                              force_reextract: bool = False):
    if not force_reextract and TRAIN_CACHE.exists() and TEST_CACHE.exists():
        print("  Loading cached CNN features...")
        tr = np.load(TRAIN_CACHE, allow_pickle=True)
        te = np.load(TEST_CACHE,  allow_pickle=True)
        return tr["features"], tr["labels"], te["features"], te["labels"]

    print("  Loading dataset from disk...")
    images, labels = loader.load_dataset()
    print(f"  Total: {len(images)} images across {len(set(labels))} classes")

    X_train, X_test, y_train, y_test = loader.split(images, labels)
    print(f"  Train: {len(X_train)}  |  Test: {len(X_test)}")

    print("  Extracting CNN features (train)...")
    X_train_feat = extractor.extract(X_train)

    print("  Extracting CNN features (test)...")
    X_test_feat = extractor.extract(X_test)

    np.savez(TRAIN_CACHE, features=X_train_feat, labels=y_train)
    np.savez(TEST_CACHE,  features=X_test_feat,  labels=y_test)
    print(f"  Features cached to {MODELS_DIR / 'feature_cache'}")

    return X_train_feat, y_train, X_test_feat, y_test


def main(force_reextract: bool = False):
    banner = "=" * 58
    print(f"\n{banner}")
    print("  Hybrid CNN-SOM  ·  Tomato Freshness Classifier")
    print(banner)

    np.random.seed(RANDOM_SEED)
    loader    = DataLoader()
    extractor = MobileNetV2Extractor()
    print(f"\n[1] Device: {extractor.device}")

    print("\n[2] Preparing features...")
    X_train_feat, y_train, X_test_feat, y_test = load_or_extract_features(
        extractor, loader, force_reextract
    )
    print(f"  Feature dim: {X_train_feat.shape[1]}")

    print("\n[3] Training Hybrid CNN-SOM...")
    model = HybridCNNSOM()
    model.fit(X_train_feat, y_train)

    print("\n  Neuron stats:")
    for pos, info in model.neuron_stats().items():
        print(f"    Neuron {pos} → {info['dominant']:12s}  "
              f"purity={info['distribution'][info['dominant']]:.1%}  "
              f"n={info['total']}")

    print("\n[4] Evaluating...")
    y_train_pred = model.predict(X_train_feat)
    y_test_pred  = model.predict(X_test_feat)

    train_metrics = compute_metrics(y_train, y_train_pred, CLASSES, split="Train")
    test_metrics  = compute_metrics(y_test,  y_test_pred,  CLASSES, split="Test")

    # Overfitting kontrolü
    gap = train_metrics["accuracy"] - test_metrics["accuracy"]
    if gap > 0.10:
        print(f"  [WARN] Accuracy gap train-test = {gap:.2%} — possible overfitting.")

    print("\n[5] Saving model and plots...")
    model.save()
    plot_confusion_matrix(y_test, y_test_pred, CLASSES)
    plot_som_map(model)
    plot_pca_variance(model.pca)

    # PCA scatter için train features'ı dönüştür
    X_train_pca = model.pca.transform(
        model.scaler.transform(X_train_feat.astype("float64"))
    )
    plot_pca_scatter(X_train_pca, y_train)

    print(f"\n{banner}")
    print(f"  Test Accuracy : {test_metrics['accuracy']*100:.2f}%")
    print(f"  Test F1 (w)   : {test_metrics['f1_weighted']:.4f}")
    print(f"  Test F1 (mac) : {test_metrics['f1_macro']:.4f}")
    print(banner)
    print(f"  Model   → {MODELS_DIR / 'hybrid_cnn_som.pkl'}")
    print(f"  Outputs → outputs/")
    print(banner)

    return test_metrics


if __name__ == "__main__":
    force = "--force" in sys.argv
    main(force_reextract=force)
