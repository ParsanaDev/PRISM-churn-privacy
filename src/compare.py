"""Compare logistic regression against the k-NN baseline on the same hold-out test set."""
import matplotlib

matplotlib.use("Agg")  # write files only, never open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cmn.config import OUTPUT_DIR, RESULTS_DIR
from cmn.tiers import TIERS

MODELS = {"k-NN (baseline)": OUTPUT_DIR / "knn",
          "Logistic regression": OUTPUT_DIR / "logistic_regression"}
CV_FILES = {"k-NN (baseline)": "best_k.csv", "Logistic regression": "best_params.csv"}
SHOWN = ["precision", "recall", "f_measure", "roc_auc", "pr_auc"]
LABELS = {"precision": "Precision", "recall": "Recall", "f_measure": "F-measure",
          "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC"}

COMPARISON_OUTPUT_DIR = OUTPUT_DIR / "comparison"
COMPARISON_RESULTS_DIR = RESULTS_DIR / "comparison"


def load_results():
    """One row per (model, tier): CV F-measure of the chosen setting plus every test metric."""
    rows = []
    for model, folder in MODELS.items():
        test = pd.read_csv(folder / "test_results.csv").set_index("tier")
        cv = pd.read_csv(folder / CV_FILES[model]).set_index("tier")
        for tier in TIERS:
            rows.append({"model": model, "tier": tier, "n_features": len(TIERS[tier]),
                         "cv_mean_f_measure": cv.loc[tier, "cv_mean_f_measure"],
                         **{f"test_{m}": test.loc[tier, m] for m in SHOWN}})
    return pd.DataFrame(rows)


def plot(results, path):
    """Grouped bars per tier for the three summary metrics, one bar per model."""
    metrics = ["f_measure", "roc_auc", "pr_auc"]
    tiers = list(TIERS)
    x = np.arange(len(tiers))
    width = 0.38
    fig, axes = plt.subplots(1, len(metrics), figsize=(14, 4.2))
    for ax, metric in zip(axes, metrics):
        for i, model in enumerate(MODELS):
            vals = results[results["model"] == model].set_index("tier") \
                .loc[tiers, f"test_{metric}"]
            bars = ax.bar(x + (i - 0.5) * width, vals, width, label=model)
            ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
        ax.set_xticks(x, [f"{t}\n({len(TIERS[t])} feat.)" for t in tiers])
        ax.set_title(f"Hold-out {LABELS[metric]}")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Logistic regression vs k-NN baseline across privacy tiers")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    """Write the comparison table and chart."""
    COMPARISON_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    COMPARISON_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = load_results()
    results.to_csv(COMPARISON_OUTPUT_DIR / "test_comparison.csv", index=False)
    plot(results, COMPARISON_RESULTS_DIR / "model_comparison.png")

    print(f"\n{'#' * 60}\nComparison on the hold-out test set\n{'#' * 60}")
    print(f"{'Model':<22}{'Tier':<8}{'CV F':>7}" + "".join(f"{LABELS[m]:>11}" for m in SHOWN))
    for _, r in results.iterrows():
        print(f"{r['model']:<22}{r['tier']:<8}{r['cv_mean_f_measure']:>7.3f}"
              + "".join(f"{r[f'test_{m}']:>11.3f}" for m in SHOWN))
    print(f"\nComparison saved to {COMPARISON_OUTPUT_DIR} and {COMPARISON_RESULTS_DIR}")
    return results


if __name__ == "__main__":
    run()
