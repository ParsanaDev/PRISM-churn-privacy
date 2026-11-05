import json
import sys
from pathlib import Path

import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

# make src/ importable when this file is run as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from knn.config import K_CAP, K_EXTEND_STEP, K_MAX, K_MIN, N_FOLDS, OUTPUT_DIR, SEED, TEST_SIZE
from knn.cv_metrics import (fold_metrics, make_folds, pick_best_k, print_summary,
                            search_best_k, summarize)
from knn.data_prep import build_pipeline, load_and_clean
from tier_subset import TARGET, TIERS


def print_best_k_folds(fold_df, tier, best_k):
    """Per-fold results for the chosen k, plus the mean row."""
    rows = fold_df[(fold_df["tier"] == tier) & (fold_df["k"] == best_k)]
    print(f"\n{tier}: per-fold results at k = {best_k}")
    print(f"{'Fold':>4} {'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5} {'Precision':>10} "
          f"{'Recall':>8} {'F-measure':>10} {'ROC-AUC':>8} {'PR-AUC':>8}")
    for _, r in rows.iterrows():
        print(f"{int(r['fold']):>4} {int(r['tp']):>5} {int(r['fp']):>5} {int(r['fn']):>5} "
              f"{int(r['tn']):>5} {r['precision']:>10.3f} {r['recall']:>8.3f} "
              f"{r['f_measure']:>10.3f} {r['roc_auc']:>8.3f} {r['pr_auc']:>8.3f}")
    print(f"{'Mean':>4} {'':>23} {rows['precision'].mean():>10.3f} {rows['recall'].mean():>8.3f} "
          f"{rows['f_measure'].mean():>10.3f} {rows['roc_auc'].mean():>8.3f} "
          f"{rows['pr_auc'].mean():>8.3f}")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # 1. load and clean the data
    df, report = load_and_clean()
    df.to_csv(OUTPUT_DIR / "cleaned_data.csv", index=False)
    (OUTPUT_DIR / "cleaning_report.json").write_text(json.dumps(report, indent=2))
    print("Cleaning report:", report)

    # 2. stratified 80/20 hold-out split, same split for every tier
    train, test = train_test_split(df, test_size=TEST_SIZE, stratify=df[TARGET],
                                   random_state=SEED)
    train, test = train.reset_index(drop=True), test.reset_index(drop=True)
    X_train, y_train = train, train[TARGET]
    print(f"\nTrain = {len(train)}, Test = {len(test)}")

    # 3. same 5 folds on the training set for every tier and every k
    folds = make_folds(X_train, y_train)

    fold_parts, best_rows, test_rows, pred_parts, k_tested = [], [], [], [], {}
    for tier, features in TIERS.items():
        print(f"\n{'#' * 60}\n{tier}: {len(features)} features\n{'#' * 60}")

        # 4. sweep k with 5-fold CV and pick the best k by mean F-measure
        fold_df, summary, best_k, k_max = search_best_k(tier, X_train, y_train, features, folds)
        print_summary(summary, tier)
        print_best_k_folds(fold_df, tier, best_k)
        fold_parts.append(fold_df)
        k_tested[tier] = k_max
        best_f = summary.loc[summary["k"] == best_k, "f_measure_mean"].iloc[0]
        best_rows.append({"tier": tier, "n_features": len(features), "best_k": best_k,
                          "cv_mean_f_measure": best_f, "k_min_tested": K_MIN,
                          "k_max_tested": k_max})

        # 5. refit on the full training set and score once on the hold-out test set
        model = build_pipeline(features, best_k).fit(X_train[features], y_train)
        y_pred = model.predict(test[features])
        y_score = model.predict_proba(test[features])[:, 1]
        test_rows.append({"tier": tier, "k": best_k, "n_test": len(test),
                          **fold_metrics(test[TARGET], y_pred, y_score)})
        pred_parts.append(pd.DataFrame({"tier": tier, "CustomerID": test["CustomerID"],
                                        "y_true": test[TARGET], "y_score": y_score,
                                        "y_pred": y_pred.astype(int)}))
        t = test_rows[-1]
        print(f"\n{tier} TEST (k = {best_k}): [[TP, FP], [FN, TN]] = "
              f"[[{t['tp']}, {t['fp']}], [{t['fn']}, {t['tn']}]]  "
              f"Precision = {t['precision']:.3f}  Recall = {t['recall']:.3f}  "
              f"F-measure = {t['f_measure']:.3f}  ROC-AUC = {t['roc_auc']:.3f}  "
              f"PR-AUC = {t['pr_auc']:.3f}")

    # 6. save everything to output/
    fold_results = pd.concat(fold_parts, ignore_index=True)
    fold_results.to_csv(OUTPUT_DIR / "fold_results.csv", index=False)
    summarize(fold_results).to_csv(OUTPUT_DIR / "k_sweep_summary.csv", index=False)
    pd.DataFrame(best_rows).to_csv(OUTPUT_DIR / "best_k.csv", index=False)
    pd.DataFrame(test_rows).to_csv(OUTPUT_DIR / "test_results.csv", index=False)
    pd.concat(pred_parts, ignore_index=True).to_csv(OUTPUT_DIR / "test_predictions.csv",
                                                    index=False)
    config = {"seed": SEED, "n_folds": N_FOLDS, "test_size": TEST_SIZE,
              "k_min": K_MIN, "k_max_initial": K_MAX, "k_extend_step": K_EXTEND_STEP,
              "k_cap": K_CAP, "k_max_tested": k_tested, "n_train": len(train),
              "n_test": len(test), "sklearn": sklearn.__version__,
              "pandas": pd.__version__}
    (OUTPUT_DIR / "run_config.json").write_text(json.dumps(config, indent=2))
    print(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
