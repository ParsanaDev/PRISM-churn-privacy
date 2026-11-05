"""Logistic regression experiment: grid search with 5-fold CV, then score once on the hold-out set."""
import itertools
import json

import pandas as pd

from cmn.data import build_preprocessor
from cmn.metrics import METRICS, fold_metrics, format_test_line
from cmn.tiers import TARGET, TIERS
from logistic_regression.config import (BATCH_SIZES, EPOCHS, LEARNING_RATES, LOSSES,
                                        LR_OUTPUT_DIR, THRESHOLD)
from logistic_regression.model import LOSS_FUNCTIONS, predict_proba, train_lr

PARAMS = ["loss", "learning_rate", "epochs", "batch_size"]


def fit_and_score(X_train, y_train, X_test, y_test, features, params):
    """Fit preprocessing + LR on the training rows; return (metrics, scores, preds, weights, names)."""
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


def run(train, test, folds):
    """Run logistic regression for every tier and save results to output/logistic_regression/."""
    LR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    y_train = train[TARGET]

    fold_parts, best_rows, test_rows, pred_parts, coef_parts = [], [], [], [], []
    for tier, features in TIERS.items():
        print(f"\n{'#' * 60}\nLogistic regression | {tier}: {len(features)} features\n{'#' * 60}")

        # grid search with 5-fold CV, best combination by mean F-measure
        fold_df = grid_search(tier, train, y_train, features, folds)
        fold_parts.append(fold_df)
        summary = summarize(fold_df)
        best = summary.loc[summary["f_measure_mean"].idxmax()]
        params = {p: best[p] for p in PARAMS}
        params["learning_rate"] = float(params["learning_rate"])
        params["epochs"], params["batch_size"] = int(params["epochs"]), int(params["batch_size"])
        best_rows.append({"tier": tier, "n_features": len(features), **params,
                          "cv_mean_f_measure": best["f_measure_mean"],
                          "cv_std_f_measure": best["f_measure_std"]})
        print(f"best (by mean F-measure): {params}  CV F = {best['f_measure_mean']:.3f}")

        # refit on the full training set and score once on the hold-out test set
        metrics, y_score, y_pred, w, names = fit_and_score(train, y_train, test, test[TARGET],
                                                           features, params)
        test_rows.append({"tier": tier, **params, "n_test": len(test), **metrics})
        pred_parts.append(pd.DataFrame({"tier": tier, "CustomerID": test["CustomerID"],
                                        "y_true": test[TARGET], "y_score": y_score,
                                        "y_pred": y_pred}))
        coef_parts.append(pd.DataFrame({"tier": tier, "feature": names + ["bias"],
                                        "coefficient": w}))
        print(format_test_line(tier, test_rows[-1], "logistic regression"))

    fold_results = pd.concat(fold_parts, ignore_index=True)
    fold_results.to_csv(LR_OUTPUT_DIR / "fold_results.csv", index=False)
    summarize(fold_results).to_csv(LR_OUTPUT_DIR / "grid_summary.csv", index=False)
    pd.DataFrame(best_rows).to_csv(LR_OUTPUT_DIR / "best_params.csv", index=False)
    pd.DataFrame(test_rows).to_csv(LR_OUTPUT_DIR / "test_results.csv", index=False)
    pd.concat(pred_parts, ignore_index=True).to_csv(LR_OUTPUT_DIR / "test_predictions.csv",
                                                    index=False)
    pd.concat(coef_parts, ignore_index=True).to_csv(LR_OUTPUT_DIR / "coefficients.csv",
                                                    index=False)
    config = {"threshold": THRESHOLD, "learning_rates": LEARNING_RATES, "epochs": EPOCHS,
              "batch_sizes": BATCH_SIZES, "losses": LOSSES,
              "selection": "max mean CV F-measure"}
    (LR_OUTPUT_DIR / "run_config.json").write_text(json.dumps(config, indent=2))
    print(f"\nLogistic regression results saved to {LR_OUTPUT_DIR}")
    return pd.DataFrame(test_rows)
