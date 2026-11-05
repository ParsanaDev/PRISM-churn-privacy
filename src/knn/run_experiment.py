"""k-NN baseline experiment: pick k by 5-fold CV, then score once on the hold-out test set."""
import json

import pandas as pd

from cmn.metrics import fold_metrics, format_test_line
from cmn.tiers import TARGET, TIERS
from knn.config import K_CAP, K_EXTEND_STEP, K_MAX, K_MIN, KNN_OUTPUT_DIR
from knn.cv import print_summary, search_best_k, summarize
from knn.model import build_pipeline


def run(train, test, folds):
    """Run the k-NN baseline for every tier and save results to output/knn/."""
    KNN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    y_train = train[TARGET]

    fold_parts, best_rows, test_rows, pred_parts, k_tested = [], [], [], [], {}
    for tier, features in TIERS.items():
        print(f"\n{'#' * 60}\nk-NN | {tier}: {len(features)} features\n{'#' * 60}")

        # sweep k with 5-fold CV and pick the best k by mean F-measure
        fold_df, summary, best_k, k_max = search_best_k(tier, train, y_train, features, folds)
        print_summary(summary, tier)
        fold_parts.append(fold_df)
        k_tested[tier] = k_max
        best_f = summary.loc[summary["k"] == best_k, "f_measure_mean"].iloc[0]
        best_rows.append({"tier": tier, "n_features": len(features), "best_k": best_k,
                          "cv_mean_f_measure": best_f, "k_min_tested": K_MIN,
                          "k_max_tested": k_max})

        # refit on the full training set and score once on the hold-out test set
        model = build_pipeline(features, best_k).fit(train[features], y_train)
        y_pred = model.predict(test[features])
        y_score = model.predict_proba(test[features])[:, 1]
        test_rows.append({"tier": tier, "k": best_k, "n_test": len(test),
                          **fold_metrics(test[TARGET], y_pred, y_score)})
        pred_parts.append(pd.DataFrame({"tier": tier, "CustomerID": test["CustomerID"],
                                        "y_true": test[TARGET], "y_score": y_score,
                                        "y_pred": y_pred.astype(int)}))
        print(format_test_line(tier, test_rows[-1], f"k = {best_k}"))

    fold_results = pd.concat(fold_parts, ignore_index=True)
    fold_results.to_csv(KNN_OUTPUT_DIR / "fold_results.csv", index=False)
    summarize(fold_results).to_csv(KNN_OUTPUT_DIR / "k_sweep_summary.csv", index=False)
    pd.DataFrame(best_rows).to_csv(KNN_OUTPUT_DIR / "best_k.csv", index=False)
    pd.DataFrame(test_rows).to_csv(KNN_OUTPUT_DIR / "test_results.csv", index=False)
    pd.concat(pred_parts, ignore_index=True).to_csv(KNN_OUTPUT_DIR / "test_predictions.csv",
                                                    index=False)
    config = {"k_min": K_MIN, "k_max_initial": K_MAX, "k_extend_step": K_EXTEND_STEP,
              "k_cap": K_CAP, "k_max_tested": k_tested,
              "selection": "max mean CV F-measure (smallest k on ties)"}
    (KNN_OUTPUT_DIR / "run_config.json").write_text(json.dumps(config, indent=2))
    print(f"\nk-NN results saved to {KNN_OUTPUT_DIR}")
    return pd.DataFrame(test_rows)
