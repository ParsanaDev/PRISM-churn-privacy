"""Cross-validated k sweep for the k-NN baseline."""
import pandas as pd

from cmn.config import N_FOLDS
from cmn.metrics import METRICS, fold_metrics
from knn.config import K_CAP, K_EXTEND_STEP, K_MAX, K_MIN
from knn.model import build_pipeline


def cv_for_k(tier, X, y, features, k, folds):
    """Run the k-NN model on every fold and return one row of metrics per fold."""
    rows = []
    for fold, (train_idx, test_idx) in enumerate(folds, start=1):
        X_train, X_test = X.iloc[train_idx][features], X.iloc[test_idx][features]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = build_pipeline(features, k).fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_score = model.predict_proba(X_test)[:, 1]

        rows.append({"tier": tier, "k": k, "fold": fold,
                     "n_train": len(train_idx), "n_test": len(test_idx),
                     **fold_metrics(y_test, y_pred, y_score)})
    return rows


def sweep_k(tier, X, y, features, folds, k_values):
    """Fold-level results for every k in k_values."""
    rows = []
    for k in k_values:
        rows += cv_for_k(tier, X, y, features, k, folds)
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


def search_best_k(tier, X, y, features, folds):
    """Sweep k = K_MIN..K_MAX; if the best k is the largest tested, keep extending the range."""
    k_max = K_MAX
    fold_df = sweep_k(tier, X, y, features, folds, range(K_MIN, k_max + 1))
    best_k = pick_best_k(summarize(fold_df))

    # best k is at the upper edge, so performance never turned down: test larger k
    while best_k == k_max and k_max < K_CAP:
        new_ks = range(k_max + 1, min(k_max + K_EXTEND_STEP, K_CAP) + 1)
        k_max = new_ks[-1]
        print(f"{tier}: best k at upper edge, extending sweep to k = {k_max}")
        fold_df = pd.concat([fold_df, sweep_k(tier, X, y, features, folds, new_ks)],
                            ignore_index=True)
        best_k = pick_best_k(summarize(fold_df))

    return fold_df, summarize(fold_df), best_k, k_max


def print_summary(summary, tier):
    """Mean metrics across folds for each k."""
    s = summary[summary["tier"] == tier]
    print(f"\n{tier}: mean metrics across {N_FOLDS} folds")
    print("-" * 60)
    print(f"{'k':>3} {'Precision':>10} {'Recall':>8} {'F-measure':>10} "
          f"{'ROC-AUC':>8} {'PR-AUC':>8}")
    print("-" * 60)
    for _, r in s.iterrows():
        print(f"{int(r['k']):>3} {r['precision_mean']:>10.3f} {r['recall_mean']:>8.3f} "
              f"{r['f_measure_mean']:>10.3f} {r['roc_auc_mean']:>8.3f} {r['pr_auc_mean']:>8.3f}")
    best = pick_best_k(s)
    best_f = s.loc[s["k"] == best, "f_measure_mean"].iloc[0]
    print("-" * 60)
    print(f"Best k by mean F-measure: {best} (F-measure = {best_f:.3f})")
