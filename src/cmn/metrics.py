"""Evaluation metrics shared by every model (churn = positive class)."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

METRICS = ["precision", "recall", "f_measure", "roc_auc", "pr_auc"]


def fold_metrics(y_true, y_pred, y_score):
    """Confusion counts, precision, recall, F-measure, ROC-AUC and PR-AUC."""
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred).astype(bool)

    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    tn = int(np.sum(~y_true & ~y_pred))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f_measure = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f_measure": f_measure,
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
    }


def format_test_line(tier, row, label):
    """One-line summary of a hold-out test result."""
    return (f"{tier} TEST ({label}): [[TP, FP], [FN, TN]] = "
            f"[[{row['tp']}, {row['fp']}], [{row['fn']}, {row['tn']}]]  "
            f"Precision = {row['precision']:.3f}  Recall = {row['recall']:.3f}  "
            f"F-measure = {row['f_measure']:.3f}  ROC-AUC = {row['roc_auc']:.3f}  "
            f"PR-AUC = {row['pr_auc']:.3f}")
