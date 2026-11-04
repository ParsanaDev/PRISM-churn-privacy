# PRISM feature tiers (global, shared by all models)

Source of truth: `src/tier_subset.py`. Dataset: `data/Telco_customer_churn.csv` (7,043 rows).
Target: `Churn Value` (1 = churned, 26.5%).

| Tier | Adds | Columns |
|---|---|---|
| **Tier 1** (4): administrative | account terms | Tenure Months, Contract, Payment Method, Paperless Billing |
| **Tier 2** (6): + billing | charges | Tier 1 + Monthly Charges, Total Charges |
| **Tier 3** (19): + demographics & services | profile + subscriptions | Tier 2 + Gender, Senior Citizen, Partner, Dependents, Phone Service, Multiple Lines, Internet Service, Online Security, Online Backup, Device Protection, Tech Support, Streaming TV, Streaming Movies |

**Never used (any tier):**
- **Identifiers / constants:** CustomerID, Count
- **Location:** Country, State, City, Zip Code, Lat Long, Latitude, Longitude
- **Leakage (derived from or describing the outcome):** Churn Label, Churn Score, CLTV, Churn Reason

**Shared setup for every model:**
- **Split:** 80/20 stratified on Churn Value, `random_state=42`.
- **Preprocessing:** fitted on training data only. `Total Charges` is blank for 11 customers
  with tenure 0; it is converted to a number and the blanks are filled with the training median.
