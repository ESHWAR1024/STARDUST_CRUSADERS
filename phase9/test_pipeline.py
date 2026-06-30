"""
Phase 9 — Quick self-test
Generates a minimal synthetic dataset, runs feature engineering + scoring,
and confirms the outputs are correct shapes and values.

Run: python test_pipeline.py
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from feature_engineer import build_feature_matrix
from xgb_model import train_and_score
from xgb_config import ALL_FEATURES

# ─────────────────────────────────────────────────────────────────────────────
# Build a tiny synthetic dataset
# ─────────────────────────────────────────────────────────────────────────────

ACCOUNTS = {
    "ACC000001": {"holder": "Normal Salaried",   "is_fraud": False},
    "ACC000002": {"holder": "Mule Account",       "is_fraud": True},
    "ACC000003": {"holder": "Collector Account",  "is_fraud": True},
    "ACC000004": {"holder": "Normal Student",     "is_fraud": False},
    "ACC000005": {"holder": "Business Owner",     "is_fraud": False},
}

def _make_txns(acc_id: str, is_fraud: bool) -> list[dict]:
    rows = []
    base_date = "2025-01-"
    if is_fraud:
        # Mule pattern: receive from many, send to many, rapid bursts, odd hours
        for i in range(1, 31):
            day = f"{i:02d}"
            # receive credit from multiple sources at odd hours
            rows.append({
                "account_id": acc_id, "account_holder": ACCOUNTS[acc_id]["holder"],
                "bank_name": "Test Bank",
                "date": f"2025-01-{day}", "time": f"02:{i%60:02d}:00",
                "narration": "UPI CREDIT FROM ACC99999",
                "channel": "UPI",
                "debit": 0, "credit": round(9500 + i * 100, 2),
                "balance": round(10000 + i * 100, 2),
                "utr_ref": f"UTR{i:04d}", "counterparty_name": f"ACC{9000+i:06d}",
                "clean_flags": "", "is_duplicate": False, "is_balance_breach": False,
                "is_high_value_flag": False, "is_ocr_row": False, "is_velocity_flag": True,
                "is_round_trip": False, "is_layering": False, "is_fan_in": True,
                "is_fan_out": True, "is_smurfing": False, "is_odd_hour": True,
                "analytics_flags": "FAN_IN | FAN_OUT | ODD_HOUR",
            })
            # immediately debit out to different account
            rows.append({
                "account_id": acc_id, "account_holder": ACCOUNTS[acc_id]["holder"],
                "bank_name": "Test Bank",
                "date": f"2025-01-{day}", "time": f"02:{(i+5)%60:02d}:00",
                "narration": "NEFT TO ACC88888",
                "channel": "NEFT",
                "debit": round(9400 + i * 100, 2), "credit": 0,
                "balance": round(600, 2),
                "utr_ref": f"UTR{i:04d}D", "counterparty_name": f"ACC{8000+i:06d}",
                "clean_flags": "", "is_duplicate": False, "is_balance_breach": False,
                "is_high_value_flag": True, "is_ocr_row": False, "is_velocity_flag": True,
                "is_round_trip": False, "is_layering": False, "is_fan_in": True,
                "is_fan_out": True, "is_smurfing": False, "is_odd_hour": True,
                "analytics_flags": "FAN_IN | FAN_OUT | ODD_HOUR",
            })
    else:
        # Normal account: salary in on 1st, regular bills/UPI out
        for i in range(1, 15):
            day = f"{i:02d}"
            if i == 1:
                rows.append({
                    "account_id": acc_id, "account_holder": ACCOUNTS[acc_id]["holder"],
                    "bank_name": "Test Bank",
                    "date": f"2025-01-{day}", "time": "10:00:00",
                    "narration": "SALARY CREDIT",
                    "channel": "NEFT",
                    "debit": 0, "credit": 50000, "balance": 55000,
                    "utr_ref": f"UTR{i:04d}", "counterparty_name": "EMPLOYER_CO",
                    "clean_flags": "", "is_duplicate": False, "is_balance_breach": False,
                    "is_high_value_flag": False, "is_ocr_row": False, "is_velocity_flag": False,
                    "is_round_trip": False, "is_layering": False, "is_fan_in": False,
                    "is_fan_out": False, "is_smurfing": False, "is_odd_hour": False,
                    "analytics_flags": "",
                })
            rows.append({
                "account_id": acc_id, "account_holder": ACCOUNTS[acc_id]["holder"],
                "bank_name": "Test Bank",
                "date": f"2025-01-{day}", "time": "14:30:00",
                "narration": f"UPI PAYMENT SWIGGY{i}",
                "channel": "UPI",
                "debit": round(200 + i * 50, 2), "credit": 0,
                "balance": round(55000 - i * 500, 2),
                "utr_ref": f"UTR{i:04d}N", "counterparty_name": "Swiggy",
                "clean_flags": "", "is_duplicate": False, "is_balance_breach": False,
                "is_high_value_flag": False, "is_ocr_row": False, "is_velocity_flag": False,
                "is_round_trip": False, "is_layering": False, "is_fan_in": False,
                "is_fan_out": False, "is_smurfing": False, "is_odd_hour": False,
                "analytics_flags": "",
            })
    return rows


def build_test_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_rows = []
    for acc_id, meta in ACCOUNTS.items():
        all_rows.extend(_make_txns(acc_id, meta["is_fraud"]))

    txn_df = pd.DataFrame(all_rows)
    for col in ("debit", "credit", "balance"):
        txn_df[col] = txn_df[col].astype(float)

    # Minimal risk_scores.csv mock
    risk_rows = []
    for acc_id, meta in ACCOUNTS.items():
        is_f = meta["is_fraud"]
        risk_rows.append({
            "account_id":         acc_id,
            "account_holder":     meta["holder"],
            "bank_name":          "Test Bank",
            "risk_score":         75.0 if is_f else 10.0,
            "risk_tier":          "CRITICAL" if is_f else "LOW",
            "flag_round_trip":    False,
            "flag_layering":      False,
            "flag_fan_in":        is_f,
            "flag_fan_out":       is_f,
            "flag_smurfing":      False,
            "flag_odd_hour":      is_f,
            "flag_velocity":      is_f,
            "flag_high_value":    is_f,
            "flag_balance_breach":False,
            "flag_new_hv_bene":   False,
            "active_patterns":    "FAN_IN | FAN_OUT" if is_f else "NONE",
            "risk_reasoning":     "mule" if is_f else "clean",
        })
    risk_df = pd.DataFrame(risk_rows)

    # Ground truth labels
    label_rows = [
        {"account_id": acc_id, "is_fraud": int(meta["is_fraud"])}
        for acc_id, meta in ACCOUNTS.items()
    ]
    labels_df = pd.DataFrame(label_rows)

    return txn_df, risk_df, labels_df


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_feature_matrix():
    txn_df, risk_df, _ = build_test_data()
    feat = build_feature_matrix(txn_df, risk_df)
    assert len(feat) == 5, f"Expected 5 accounts, got {len(feat)}"
    assert set(ALL_FEATURES).issubset(feat.columns), "Missing features"
    assert not feat.isnull().any().any(), "Feature matrix contains NaN"
    assert not np.isinf(feat.values).any(), "Feature matrix contains Inf"
    print("  [PASS] test_feature_matrix")

    # Mule accounts should have higher fan_out_score, odd_hour_ratio
    mule   = feat.loc["ACC000002"]
    normal = feat.loc["ACC000001"]
    assert mule["odd_hour_ratio"] > normal["odd_hour_ratio"], "Mule should have higher odd_hour_ratio"
    assert mule["flag_fan_out"]   > normal["flag_fan_out"],   "Mule should have fan_out flag"
    print("  [PASS] test_feature_matrix - mule vs normal differentiation")


def test_unsupervised_scoring():
    import tempfile
    txn_df, risk_df, _ = build_test_data()
    with tempfile.TemporaryDirectory() as tmpdir:
        txn_path  = os.path.join(tmpdir, "txn.csv")
        risk_path = os.path.join(tmpdir, "risk.csv")
        txn_df.to_csv(txn_path,  index=False)
        risk_df.to_csv(risk_path, index=False)

        scores = train_and_score(txn_path, risk_path, out_dir=tmpdir)

        assert len(scores) == 5, f"Expected 5 rows, got {len(scores)}"
        assert "xgb_fraud_probability" in scores.columns
        assert "xgb_mule_score" in scores.columns
        assert "xgb_tier" in scores.columns
        assert not scores["xgb_fraud_probability"].isnull().any()

        # Verify outputs exist
        assert os.path.exists(os.path.join(tmpdir, "xgb_scores.csv"))
        assert os.path.exists(os.path.join(tmpdir, "xgb_report.json"))

        print("  [PASS] test_unsupervised_scoring")


def test_supervised_training():
    import tempfile
    txn_df, risk_df, labels_df = build_test_data()
    with tempfile.TemporaryDirectory() as tmpdir:
        txn_path    = os.path.join(tmpdir, "txn.csv")
        risk_path   = os.path.join(tmpdir, "risk.csv")
        labels_path = os.path.join(tmpdir, "labels.csv")
        txn_df.to_csv(txn_path,    index=False)
        risk_df.to_csv(risk_path,  index=False)
        labels_df.to_csv(labels_path, index=False)

        scores = train_and_score(txn_path, risk_path, labels_path, out_dir=tmpdir)

        assert len(scores) == 5
        assert "ground_truth_label" in scores.columns
        assert os.path.exists(os.path.join(tmpdir, "xgb_model.pkl"))
        assert os.path.exists(os.path.join(tmpdir, "xgb_report.json"))

        import json
        with open(os.path.join(tmpdir, "xgb_report.json")) as f:
            report = json.load(f)
        assert report["mode"] == "supervised"
        assert "roc_auc" in report

        print("  [PASS] test_supervised_training")
        print(f"         ROC-AUC: {report.get('roc_auc', 'N/A'):.4f}")


if __name__ == "__main__":
    print("\n=== Phase 9 Self-Test ===\n")
    test_feature_matrix()
    test_unsupervised_scoring()
    test_supervised_training()
    print("\n=== All tests passed ===\n")
