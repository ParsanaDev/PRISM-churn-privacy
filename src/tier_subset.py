TIER_1 = ["Tenure Months", "Contract", "Payment Method", "Paperless Billing"]
TIER_2 = TIER_1 + ["Monthly Charges", "Total Charges"]
TIER_3 = TIER_2 + [
    "Gender", "Senior Citizen", "Partner", "Dependents",
    "Phone Service", "Multiple Lines", "Internet Service",
    "Online Security", "Online Backup", "Device Protection", "Tech Support",
    "Streaming TV", "Streaming Movies",
]

TARGET = "Churn Value"

TIERS = {"Tier 1": TIER_1, "Tier 2": TIER_2, "Tier 3": TIER_3}