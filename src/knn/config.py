"""k-NN baseline settings (shared seed, split and folds live in cmn/config.py)."""
from cmn.config import OUTPUT_DIR, RESULTS_DIR

# k range for the sweep; extended if the best k sits at the upper edge
K_MIN = 1
K_MAX = 20
K_EXTEND_STEP = 10
K_CAP = 100

KNN_OUTPUT_DIR = OUTPUT_DIR / "knn"
KNN_RESULTS_DIR = RESULTS_DIR / "knn"
