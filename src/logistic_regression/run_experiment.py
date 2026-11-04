import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

# make src/ importable when this file is run as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from logistic_regression.config import (BATCH_SIZES, EPOCHS, LEARNING_RATES, LOSSES, N_FOLDS,
                                        OUTPUT_DIR, SEED, TEST_SIZE, THRESHOLD)
from logistic_regression.data_prep import build_preprocessor, load_and_clean
from logistic_regression.model import LOSS_FUNCTIONS, predict_proba, train_lr
from tier_subset import TARGET, TIERS

PARAMS = ["loss", "learning_rate", "epochs", "batch_size"]
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
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f_measure": f_measure,
            "roc_auc": roc_auc_score(y_true, y_score),
            "pr_auc": average_precision_score(y_true, y_score)}


def fit_and_score(X_train, y_train, X_test, y_test, features, params):
    """Fit preprocessing + LR on the training rows, return (metrics, scores, preds, weights, names)."""
    prep = build_preprocessor(features).fit(X_train[features])
    w = train_lr(prep.transform(X_train[features]), y_train.to_numpy(float),
                 params["learning_rate"], params["epochs"], params["batch_size"],
                 LOSS_FUNCTIONS[params["loss"]])
    y_score = predict_proba(prep.transform(X_test[features]), w)
    y_pred = (y_score >= THRESHOLD).astype(int)
    return (fold_metrics(y_test, y_pred, y_score), y_score, y_pred, w,
            list(prep.get_feature_names_out()))


def grid_search(tier, X, y, features, folds):
    """5-fold CV for every hyperparameter combination; one row per (combination, fold)."""
    rows = []
    for loss, lr, epochs, bs in itertools.product(LOSSES, LEARNING_RATES, EPOCHS, BATCH_SIZES):
        params = {"loss": loss, "learning_rate": lr, "epochs": epochs, "batch_size": bs}
        for fold, (tr, te) in enumerate(folds, start=1):
            metrics = fit_and_score(X.iloc[tr], y.iloc[tr], X.iloc[te], y.iloc[te],
                                    features, params)[0]
            rows.append({"tier": tier, **params, "fold": fold, "n_train": len(tr),
                         "n_test": len(te), **metrics})
    return pd.DataFrame(rows)


def summarize(fold_df):
    """Mean and std of every metric per tier and hyperparameter combination."""
    agg = fold_df.groupby(["tier"] + PARAMS)[METRICS].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    return agg.reset_index()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. load and clean the data (same cleaning as k-NN)
    df, report = load_and_clean()
    (OUTPUT_DIR / "cleaning_report.json").write_text(json.dumps(report, indent=2))
    print("Cleaning report:", report)

    # 2. stratified 80/20 hold-out split, same split for every tier (and as k-NN)
    train, test = train_test_split(df, test_size=TEST_SIZE, stratify=df[TARGET],
                                   random_state=SEED)
    train, test = train.reset_index(drop=True), test.reset_index(drop=True)
    y_train = train[TARGET]
    print(f"\nTrain = {len(train)}, Test = {len(test)}")

    # 3. same 5 folds on the training set for every tier and every combination
    folds = list(StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
                 .split(train, y_train))

    fold_parts, best_rows, test_rows, pred_parts, coef_parts = [], [], [], [], []
    for tier, features in TIERS.items():
        print(f"\n{'#' * 60}\n{tier}: {len(features)} features\n{'#' * 60}")

        # 4. grid search with 5-fold CV, best combination by mean F-measure
        fold_df = grid_search(tier, train, y_train, features, folds)
        fold_parts.append(fold_df)
        summary = summarize(fold_df)
        best = summary.loc[summary["f_measure_mean"].idxmax()]
        params = {p: best[p] for p in PARAMS}
        params["epochs"], params["batch_size"] = int(params["epochs"]), int(params["batch_size"])
        best_rows.append({"tier": tier, "n_features": len(features), **params,
                          "cv_mean_f_measure": best["f_measure_mean"],
                          "cv_std_f_measure": best["f_measure_std"]})
        print(f"best (by mean F-measure): {params}  CV F = {best['f_measure_mean']:.3f}")

        # 5. refit on the full training set and score once on the hold-out test set
        metrics, y_score, y_pred, w, names = fit_and_score(train, y_train, test, test[TARGET],
                                                           features, params)
        test_rows.append({"tier": tier, **params, "n_test": len(test), **metrics})
        pred_parts.append(pd.DataFrame({"tier": tier, "CustomerID": test["CustomerID"],
                                        "y_true": test[TARGET], "y_score": y_score,
                                        "y_pred": y_pred}))
        coef_parts.append(pd.DataFrame({"tier": tier, "feature": names + ["bias"],
                                        "coefficient": w}))
        t = test_rows[-1]
        print(f"{tier} TEST: [[TP, FP], [FN, TN]] = [[{t['tp']}, {t['fp']}], "
              f"[{t['fn']}, {t['tn']}]]  Precision = {t['precision']:.3f}  "
              f"Recall = {t['recall']:.3f}  F-measure = {t['f_measure']:.3f}  "
              f"ROC-AUC = {t['roc_auc']:.3f}  PR-AUC = {t['pr_auc']:.3f}")

    # 6. save everything to output/logistic_regression/
    fold_results = pd.concat(fold_parts, ignore_index=True)
    fold_results.to_csv(OUTPUT_DIR / "fold_results.csv", index=False)
    summarize(fold_results).to_csv(OUTPUT_DIR / "grid_summary.csv", index=False)
    pd.DataFrame(best_rows).to_csv(OUTPUT_DIR / "best_params.csv", index=False)
    pd.DataFrame(test_rows).to_csv(OUTPUT_DIR / "test_results.csv", index=False)
    pd.concat(pred_parts, ignore_index=True).to_csv(OUTPUT_DIR / "test_predictions.csv",
                                                    index=False)
    pd.concat(coef_parts, ignore_index=True).to_csv(OUTPUT_DIR / "coefficients.csv", index=False)
    config = {"seed": SEED, "n_folds": N_FOLDS, "test_size": TEST_SIZE, "threshold": THRESHOLD,
              "learning_rates": LEARNING_RATES, "epochs": EPOCHS, "batch_sizes": BATCH_SIZES,
              "losses": LOSSES, "selection": "max mean CV F-measure",
              "n_train": len(train), "n_test": len(test),
              "sklearn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__}
    (OUTPUT_DIR / "run_config.json").write_text(json.dumps(config, indent=2))
    print(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
