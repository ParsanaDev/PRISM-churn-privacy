import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

# make src/ importable when this file is run as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from knn.config import K_CAP, K_EXTEND_STEP, K_MAX, K_MIN, N_FOLDS, SEED
from knn.data_prep import build_pipeline

METRICS = ["precision", "recall", "f_measure", "roc_auc", "pr_auc"]


def fold_metrics(y_true, y_pred, y_score):
    """Confusion counts, precision, recall, F-measure, ROC-AUC and PR-AUC (churn = positive)."""
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


def make_folds(X, y):
    """Stratified folds; the same folds are reused for every k and every tier."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    return list(skf.split(X, y))


def cv_for_k(tier, X, y, features, k, folds, verbose=False):
    """Run the k-NN model on every fold and return one row of metrics per fold."""
    rows = []
    for fold, (train_idx, test_idx) in enumerate(folds, start=1):
        X_train, X_test = X.iloc[train_idx][features], X.iloc[test_idx][features]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = build_pipeline(features, k)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_score = model.predict_proba(X_test)[:, 1]

        row = {"tier": tier, "k": k, "fold": fold,
               "n_train": len(train_idx), "n_test": len(test_idx),
               **fold_metrics(y_test, y_pred, y_score)}
        rows.append(row)

        if verbose:
            print(f"\n{'=' * 45}\n{tier} | k = {k} | FOLD {fold}\n{'=' * 45}")
            print(f"Confusion Matrix [[TP, FP], [FN, TN]]: "
                  f"[[{row['tp']}, {row['fp']}], [{row['fn']}, {row['tn']}]]")
            print(f"Precision = {row['precision']:.3f}  Recall = {row['recall']:.3f}  "
                  f"F-measure = {row['f_measure']:.3f}  "
                  f"ROC-AUC = {row['roc_auc']:.3f}  PR-AUC = {row['pr_auc']:.3f}")
    return rows


def sweep_k(tier, X, y, features, folds, k_values, verbose=False):
    """Fold-level results for every k in k_values."""
    rows = []
    for k in k_values:
        rows += cv_for_k(tier, X, y, features, k, folds, verbose)
    return pd.DataFrame(rows)


def summarize(fold_df):
    """Mean and std of each metric across folds, for every (tier, k)."""
    agg = fold_df.groupby(["tier", "k"])[METRICS].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    return agg.reset_index()


def pick_best_k(summary):
    """k with the highest mean F-measure (smallest k wins ties)."""
    summary = summary.sort_values("k")
    return int(summary.loc[summary["f_measure_mean"].idxmax(), "k"])


def search_best_k(tier, X, y, features, folds, verbose=False):
    """Sweep k = K_MIN..K_MAX; if the best k is the largest tested, keep extending the range."""
    k_max = K_MAX
    fold_df = sweep_k(tier, X, y, features, folds, range(K_MIN, k_max + 1), verbose)
    best_k = pick_best_k(summarize(fold_df))

    # best k is at the upper edge, so performance never turned down: test larger k
    while best_k == k_max and k_max < K_CAP:
        new_ks = range(k_max + 1, min(k_max + K_EXTEND_STEP, K_CAP) + 1)
        k_max = new_ks[-1]
        print(f"{tier}: best k at upper edge, extending sweep to k = {k_max}")
        fold_df = pd.concat([fold_df, sweep_k(tier, X, y, features, folds, new_ks, verbose)],
                            ignore_index=True)
        best_k = pick_best_k(summarize(fold_df))

    return fold_df, summarize(fold_df), best_k, k_max


def print_summary(summary, tier):
    """Mean metrics across folds for each k, like the notes notebook."""
    s = summary[summary["tier"] == tier]
    print(f"\n{tier}: mean metrics across {N_FOLDS} folds")
    print("-" * 78)
    print(f"{'k':>3} {'Precision':>10} {'Recall':>8} {'F-measure':>10} "
          f"{'ROC-AUC':>8} {'PR-AUC':>8}")
    print("-" * 78)
    for _, r in s.iterrows():
        print(f"{int(r['k']):>3} {r['precision_mean']:>10.3f} {r['recall_mean']:>8.3f} "
              f"{r['f_measure_mean']:>10.3f} {r['roc_auc_mean']:>8.3f} {r['pr_auc_mean']:>8.3f}")
    best = pick_best_k(s)
    best_f = s.loc[s["k"] == best, "f_measure_mean"].iloc[0]
    print("-" * 78)
    print(f"Best k by mean F-measure: {best} (F-measure = {best_f:.3f})")
