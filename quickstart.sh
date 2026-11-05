#!/usr/bin/env bash
# PRISM quickstart: create an environment, install pinned dependencies and run the pipeline.
#
#   git clone https://github.com/ParsanaDev/PRISM-churn-privacy.git
#   cd PRISM-churn-privacy
#   bash quickstart.sh            # full run: k-NN baseline + logistic regression + comparison
#   bash quickstart.sh --fast     # k-NN baseline only (about a minute)
set -euo pipefail

cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
VENV=".venv"

# 1. dataset (committed to the repo; fetched from GitHub if it is missing)
DATA="data/raw/Telco_customer_churn.csv"
if [ ! -f "$DATA" ]; then
    echo "Dataset not found, downloading it to $DATA"
    mkdir -p data/raw
    wget -q -O "$DATA" \
        https://raw.githubusercontent.com/ParsanaDev/PRISM-churn-privacy/main/data/Telco_customer_churn.csv
fi

# 2. virtual environment with the exact pinned versions
if [ ! -d "$VENV" ]; then
    echo "Creating virtual environment in $VENV"
    "$PYTHON" -m venv "$VENV"
fi
"$VENV/bin/python" -m pip install --quiet --upgrade pip
"$VENV/bin/python" -m pip install --quiet -r requirements.txt

# 3. run the pipeline
if [ "${1:-}" = "--fast" ]; then
    "$VENV/bin/python" src/main.py --models knn
else
    "$VENV/bin/python" src/main.py
fi

echo
echo "Results:  output/   (CSV and JSON per model, comparison in output/comparison/)"
echo "Charts:   evaluation/results/comparison/model_comparison.png"
