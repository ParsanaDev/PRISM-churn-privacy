from pathlib import Path

SEED = 2026          # same random seed for every split and run
N_FOLDS = 5          # 5-fold stratified CV (each fold is an 80/20 split)
TEST_SIZE = 0.2      # stratified hold-out test split

# k range for the sweep; extended if the best k sits at the upper edge
K_MIN = 1
K_MAX = 20
K_EXTEND_STEP = 10
K_CAP = 100

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "Telco_customer_churn.csv"
OUTPUT_DIR = ROOT / "output"
