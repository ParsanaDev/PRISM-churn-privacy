"""Logistic regression settings (shared seed, split and folds live in cmn/config.py)."""
from cmn.config import OUTPUT_DIR, RESULTS_DIR

THRESHOLD = 0.5      # probability threshold for predicting churn

# hyperparameter grid (from the course Cross-Fold-Validation-LR notebook);
# batch size 1 is replaced by 32 because ~4,500 rows per fold is too slow in pure NumPy
LEARNING_RATES = [0.01, 0.1, 0.5]
EPOCHS = [50, 200]
BATCH_SIZES = [32, 128, 512]
LOSSES = ["absolute_error"]

LR_OUTPUT_DIR = OUTPUT_DIR / "logistic_regression"
LR_RESULTS_DIR = RESULTS_DIR / "logistic_regression"
