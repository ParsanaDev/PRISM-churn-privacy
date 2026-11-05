"""Settings shared by every model, so all models see the same rows, split and folds."""
from pathlib import Path

SEED = 2026          # one random seed for the hold-out split, the CV folds and LR training
N_FOLDS = 5          # 5-fold stratified cross-validation on the training set
TEST_SIZE = 0.2      # stratified 80/20 hold-out split

ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = ROOT / "data" / "raw" / "Telco_customer_churn.csv"
PREPROCESSED_DIR = ROOT / "data" / "preprocessed"
OUTPUT_DIR = ROOT / "output"
RESULTS_DIR = ROOT / "evaluation" / "results"
