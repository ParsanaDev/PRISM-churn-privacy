from pathlib import Path

SEED = 2026          # same random seed as the k-NN experiment (split, folds, batch order)
N_FOLDS = 5          # 5-fold stratified CV on the training set
TEST_SIZE = 0.2      # stratified hold-out test split
THRESHOLD = 0.5      # probability threshold for predicting churn

# Dataset-scale grid adapted from the course Cross-Fold-Validation-LR notebook.
# The toy notebook uses epochs [100, 500, 1000] and batches [1, 4, 8];
# batch size 1 is replaced by 32 because ~4,500 rows per fold is too slow in pure NumPy
LEARNING_RATES = [0.01, 0.1, 0.5]
EPOCHS = [50, 200]
BATCH_SIZES = [32, 128, 512]
LOSSES = ["hossein"]  # use the error function trained in the supplied course notebook

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "Telco_customer_churn.csv"
OUTPUT_DIR = ROOT / "output" / "logistic_regression"
