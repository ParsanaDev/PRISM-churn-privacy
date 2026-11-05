"""PRISM main pipeline. Run from the repository root:

    python src/main.py                 # clean data, k-NN baseline, logistic regression, comparison
    python src/main.py --models knn    # only the k-NN baseline (fast)
"""
import argparse
import json
import platform
import time

import numpy as np
import pandas as pd
import sklearn

from cmn.config import N_FOLDS, OUTPUT_DIR, PREPROCESSED_DIR, SEED, TEST_SIZE
from cmn.data import holdout_split, load_and_clean, make_folds

MODEL_CHOICES = ["knn", "lr"]


def parse_args():
    parser = argparse.ArgumentParser(description="PRISM: privacy-tiered churn classification")
    parser.add_argument("--models", nargs="+", choices=MODEL_CHOICES, default=MODEL_CHOICES,
                        help="models to run (default: both)")
    parser.add_argument("--no-compare", action="store_true",
                        help="skip the k-NN vs logistic regression comparison")
    return parser.parse_args()


def main():
    args = parse_args()
    start = time.time()
    OUTPUT_DIR.mkdir(exist_ok=True)
    PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # 1. load and clean the raw data once, for every model
    df, report = load_and_clean()
    df.to_csv(PREPROCESSED_DIR / "cleaned_data.csv", index=False)
    (OUTPUT_DIR / "cleaning_report.json").write_text(json.dumps(report, indent=2))
    print("Cleaning report:", report)

    # 2. one stratified 80/20 split and one set of 5 CV folds, shared by both models
    train, test = holdout_split(df)
    folds = make_folds(train)
    print(f"Train = {len(train)}, Test = {len(test)}, {N_FOLDS} folds, seed = {SEED}")

    # 3. k-NN baseline, then logistic regression
    if "knn" in args.models:
        from knn.run_experiment import run as run_knn
        run_knn(train, test, folds)
    if "lr" in args.models:
        from logistic_regression.run_experiment import run as run_lr
        run_lr(train, test, folds)

    # 4. compare the two models on the same hold-out test set
    if not args.no_compare:
        from compare import MODELS, run as run_compare
        missing = [str(p) for p in MODELS.values() if not (p / "test_results.csv").exists()]
        if missing:
            print(f"\nSkipping comparison, results missing in: {', '.join(missing)}")
        else:
            run_compare()

    config = {"seed": SEED, "n_folds": N_FOLDS, "test_size": TEST_SIZE,
              "n_train": len(train), "n_test": len(test),
              "python": platform.python_version(), "numpy": np.__version__,
              "pandas": pd.__version__, "sklearn": sklearn.__version__}
    (OUTPUT_DIR / "run_config.json").write_text(json.dumps(config, indent=2))
    print(f"\nDone in {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
