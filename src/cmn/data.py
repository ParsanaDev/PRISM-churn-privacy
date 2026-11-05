"""Loading, cleaning, preprocessing and splitting the Telco churn data (shared by all models)."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from cmn.config import N_FOLDS, RAW_DATA_PATH, SEED, TEST_SIZE
from cmn.tiers import TARGET, TIER_3

NUMERIC = ["Tenure Months", "Monthly Charges", "Total Charges"]

YES_NO = ["Yes", "No"]
# allowed values for every categorical feature
VALID_VALUES = {
    "Contract": ["Month-to-month", "One year", "Two year"],
    "Payment Method": ["Mailed check", "Electronic check",
                       "Bank transfer (automatic)", "Credit card (automatic)"],
    "Paperless Billing": YES_NO,
    "Gender": ["Male", "Female"],
    "Senior Citizen": YES_NO,
    "Partner": YES_NO,
    "Dependents": YES_NO,
    "Phone Service": YES_NO,
    "Multiple Lines": YES_NO + ["No phone service"],
    "Internet Service": ["DSL", "Fiber optic", "No"],
    **{c: YES_NO + ["No internet service"] for c in
       ["Online Security", "Online Backup", "Device Protection",
        "Tech Support", "Streaming TV", "Streaming Movies"]},
}


def load_and_clean(path=RAW_DATA_PATH):
    """Load the churn CSV, clean it, and return (clean_df, report)."""
    df = pd.read_csv(path)
    report = {"rows_loaded": len(df)}

    # keep only the id, the features used by the tiers and the target
    df = df[["CustomerID"] + TIER_3 + [TARGET]].copy()

    # strip stray whitespace in text columns
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].str.strip()

    # Total Charges is stored as text; blanks are new customers (tenure 0) -> 0
    total = pd.to_numeric(df["Total Charges"], errors="coerce")
    blank_new = total.isna() & (df["Tenure Months"] == 0)
    total[blank_new] = 0.0
    df["Total Charges"] = total
    report["total_charges_blank_set_to_0"] = int(blank_new.sum())

    # remove duplicate customers
    before = len(df)
    df = df.drop_duplicates().drop_duplicates(subset="CustomerID")
    report["duplicates_removed"] = before - len(df)

    # remove rows with missing or invalid values
    invalid = df[TIER_3 + [TARGET]].isna().any(axis=1)
    invalid |= (df[NUMERIC] < 0).any(axis=1)
    invalid |= ~df[TARGET].isin([0, 1])
    for col, allowed in VALID_VALUES.items():
        invalid |= ~df[col].isin(allowed)
    report["invalid_rows_removed"] = int(invalid.sum())
    df = df[~invalid].reset_index(drop=True)

    report["rows_after_cleaning"] = len(df)
    report["churn_rate"] = round(float(df[TARGET].mean()), 4)
    return df, report


def build_preprocessor(features):
    """z-score the numeric features and one-hot encode the categorical ones.

    Always fitted on training rows only, so no information leaks from validation or test rows.
    """
    numeric = [c for c in features if c in NUMERIC]
    categorical = [c for c in features if c not in NUMERIC]
    return ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore",
                              sparse_output=False), categorical),
    ])


def holdout_split(df):
    """Stratified 80/20 train/test split; the same split is used for every model and tier."""
    train, test = train_test_split(df, test_size=TEST_SIZE, stratify=df[TARGET],
                                   random_state=SEED)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def make_folds(train):
    """Stratified CV folds on the training set; reused for every model, tier and setting."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    return list(skf.split(train, train[TARGET]))
