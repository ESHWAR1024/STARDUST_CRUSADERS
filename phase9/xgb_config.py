"""
Phase 9 — XGBoost Behavioral Layer Configuration

L2 in the detection stack (from the honest detection map):
  - PRIMARY:  mule behavior detection
  - PARTIAL:  layering, fan-in, fan-out, smurfing, collection accounts
  - BLIND:    fraud rings (structural, GraphSAGE handles that)

Feature groups and all tunable hyperparameters live here.
"""

# ---------------------------------------------------------------------------
# Model hyperparameters
# ---------------------------------------------------------------------------
XGB_PARAMS = {
    "n_estimators":      400,
    "max_depth":         6,
    "learning_rate":     0.05,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "min_child_weight":  3,
    "gamma":             0.1,
    "reg_alpha":         0.1,       # L1 — encourage sparsity (many zero-weight features)
    "reg_lambda":        1.0,       # L2
    "scale_pos_weight":  5.0,       # class imbalance: ~5:1 normal:fraud ratio
    "eval_metric":       "logloss",
    "use_label_encoder": False,
    "random_state":      42,
    "n_jobs":            -1,
}

# Early stopping rounds (used when a validation set is available)
EARLY_STOPPING_ROUNDS = 30

# Cross-validation folds (used when no ground truth split is available)
CV_FOLDS = 5

# Probability threshold above which an account is classed as suspicious
SUSPICIOUS_THRESHOLD = 0.40   # lower than 0.5 to favour recall (investigators prefer false +)

# ---------------------------------------------------------------------------
# Feature groups — used to explain which features drove each prediction
# ---------------------------------------------------------------------------
FEATURE_GROUPS = {
    "volume":      ["txn_count", "debit_count", "credit_count",
                    "total_debit", "total_credit", "net_flow"],
    "amount":      ["mean_debit", "std_debit", "max_debit",
                    "mean_credit", "std_credit", "max_credit",
                    "debit_credit_ratio", "max_single_txn_ratio"],
    "temporal":    ["active_days", "txn_per_day", "odd_hour_ratio",
                    "weekend_ratio", "burst_count"],
    "structural":  ["unique_counterparties", "counterparty_diversity",
                    "fan_in_score", "fan_out_score",
                    "round_trip_flag", "layering_flag"],
    "phase8":      ["flag_round_trip", "flag_layering", "flag_fan_in",
                    "flag_fan_out", "flag_smurfing", "flag_odd_hour",
                    "flag_velocity", "flag_high_value", "flag_balance_breach"],
    "channel":     ["upi_ratio", "neft_ratio", "rtgs_ratio",
                    "cash_ratio", "imps_ratio"],
    "zscore":      ["amount_zscore_max", "amount_zscore_mean",
                    "balance_volatility"],
}

# Flat list of all features (order determines column order in the feature matrix)
ALL_FEATURES = [f for group in FEATURE_GROUPS.values() for f in group]

# ---------------------------------------------------------------------------
# Mule behavior scoring weights
# (used by the standalone mule_score function for explainability)
# ---------------------------------------------------------------------------
MULE_SIGNAL_WEIGHTS = {
    "flag_fan_out":          3.0,    # primary mule signal: receive and distribute
    "flag_fan_in":           2.5,
    "flag_velocity":         2.0,    # rapid successive outbound transfers
    "fan_out_score":         1.5,
    "fan_in_score":          1.5,
    "flag_round_trip":       1.5,
    "odd_hour_ratio":        1.0,    # late-night activity
    "debit_credit_ratio":    1.0,    # near-equal debit and credit volumes
    "counterparty_diversity":1.0,    # many unique counterparties = mule hub
    "burst_count":           0.8,
}

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
MODEL_SAVE_PATH  = "xgb_model.pkl"
REPORT_SAVE_PATH = "xgb_report.json"
SCORES_SAVE_PATH = "xgb_scores.csv"
