"""Privacy tiers: each tier adds more personal information to the one before it."""

# Tier 1 (4 features): administrative account terms only
TIER_1 = ["Tenure Months", "Contract", "Payment Method", "Paperless Billing"]

# Tier 2 (6 features): + billing amounts
TIER_2 = TIER_1 + ["Monthly Charges", "Total Charges"]

# Tier 3 (19 features): + demographics and subscribed services
TIER_3 = TIER_2 + [
    "Gender", "Senior Citizen", "Partner", "Dependents",
    "Phone Service", "Multiple Lines", "Internet Service",
    "Online Security", "Online Backup", "Device Protection", "Tech Support",
    "Streaming TV", "Streaming Movies",
]

TARGET = "Churn Value"   # 1 = churned (positive class), 0 = stayed

TIERS = {"Tier 1": TIER_1, "Tier 2": TIER_2, "Tier 3": TIER_3}
