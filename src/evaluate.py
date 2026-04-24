"""Metrik hesaplama ve görselleştirme fonksiyonları."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
from src.config import OUTPUTS_DIR

_CLASS_COLORS = {
    "Taze": "#2ecc71",
    "Yenebilir": "#f39c12",
    "Çürük": "#e74c3c",
}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_pred, classes, split="Test") -> dict:
    acc = accuracy_score(y_true, y_pred)
    f1_w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_m = f1_score(y_true, y_pred, average="macro", zero_division=0)

    sep = "=" * 52
    print(f"\n{sep}")
    print(f"  {split} Results")
    print(sep)
    print(f"  Accuracy       : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  F1 (weighted)  : {f1_w:.4f}")
    print(f"  F1 (macro)     : {f1_m:.4f}")
    print(f"\n{classification_report(y_true, y_pred, target_names=classes, zero_division=0)}")

    return {"accuracy": acc, "f1_weighted": f1_w, "f1_macro": f1_m}


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(y_true, y_pred, classes, save=True, tag=""):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, data, title, fmt in zip(
        axes,
        [cm, cm_norm],
        ["Confusion Matrix  (counts)", "Confusion Matrix  (row-normalised)"],
        ["d", ".2%"],
    ):
        sns.heatmap(
            data,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=classes,
            yticklabels=classes,
            ax=ax,
            linewidths=0.5,
            annot_kws={"size": 12},
        )
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.set_ylabel("True Label", fontsize=11)
        ax.set_xlabel("Predicted Label", fontsize=11)

    fig.suptitle(
        "Hybrid CNN-SOM  ·  Tomato Freshness Classifier",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()

    if save:
        out = OUTPUTS_DIR / f"confusion_matrix{tag}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Saved → {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# SOM map
# ---------------------------------------------------------------------------

def plot_som_map(model, save=True, tag=""):
    """SOM grid'ini nöron etiketleri ve purity ile görselleştirir."""
    stats = model.neuron_stats()
    n_neurons = model.som_x * model.som_y

    fig, ax = plt.subplots(figsize=(max(8, model.som_x * 3), 3))

    for (col, row), info in stats.items():
        label = info["dominant"]
        purity = info["distribution"].get(label, 0)
        total = info["total"]
        color = _CLASS_COLORS.get(label, "#95a5a6")

        rect = mpatches.FancyBboxPatch(
            (col + 0.05, row + 0.05),
            0.9,
            0.9,
            boxstyle="round,pad=0.05",
            facecolor=color,
            alpha=0.35,
            edgecolor=color,
            linewidth=2,
        )
        ax.add_patch(rect)

        ax.text(
            col + 0.5,
            row + 0.62,
            label,
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
            color=color,
        )
        ax.text(
            col + 0.5,
            row + 0.35,
            f"purity {purity:.0%}  |  n={total}",
            ha="center",
            va="center",
            fontsize=9,
            color="#555",
        )

    ax.set_xlim(0, model.som_x)
    ax.set_ylim(0, model.som_y)
    ax.set_xticks([i + 0.5 for i in range(model.som_x)])
    ax.set_xticklabels([f"Neuron {i+1}" for i in range(model.som_x)], fontsize=11)
    ax.set_yticks([])
    ax.set_title(
        "SOM Map  ·  Neuron Label Distribution",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    plt.tight_layout()

    if save:
        out = OUTPUTS_DIR / f"som_map{tag}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Saved → {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# PCA scatter (2-D projection)
# ---------------------------------------------------------------------------

def plot_pca_scatter(features_pca: np.ndarray, labels: np.ndarray, save=True, tag=""):
    """PC1 vs PC2 scatter plot — sınıf ayrılabilirliğini gösterir."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for cls, color in _CLASS_COLORS.items():
        mask = labels == cls
        ax.scatter(
            features_pca[mask, 0],
            features_pca[mask, 1],
            c=color,
            label=cls,
            alpha=0.35,
            s=14,
            edgecolors="none",
        )
    ax.set_xlabel("PC 1", fontsize=11)
    ax.set_ylabel("PC 2", fontsize=11)
    ax.set_title("PCA Feature Space  ·  PC1 vs PC2", fontweight="bold", fontsize=13)
    ax.legend(title="Class", fontsize=10)
    ax.grid(alpha=0.2)
    plt.tight_layout()

    if save:
        out = OUTPUTS_DIR / f"pca_scatter{tag}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Saved → {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# PCA variance explained
# ---------------------------------------------------------------------------

def plot_pca_variance(pca, save=True, tag=""):
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(cum_var) + 1), cum_var, marker="o", markersize=3, color="#3498db")
    ax.axhline(0.90, ls="--", color="#e74c3c", alpha=0.7, label="90% threshold")
    ax.axhline(0.95, ls="--", color="#f39c12", alpha=0.7, label="95% threshold")
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Cumulative Explained Variance")
    ax.set_title("PCA — Cumulative Explained Variance", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save:
        out = OUTPUTS_DIR / f"pca_variance{tag}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Saved → {out}")
    plt.close(fig)
