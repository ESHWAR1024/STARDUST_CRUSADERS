"""
Phase 9 — Supervised Training Script

Generates pseudo-labels from Phase 8 risk scores, trains XGBoost,
evaluates with cross-validation, and saves the model.

Usage:
    python train.py --txn   ../phase8/analytics_training/analytics_transactions.csv
                    --risk  ../phase8/analytics_training/risk_scores.csv
                    --out   xgb_output_trained/

Optional — if real ground truth exists:
    python train.py ... --labels path/to/account_labels.csv
"""

import argparse
import json
import os
import pickle
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    precision_recall_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)

# ── import from same directory ───────────────────────────────────────────────
import sys
sys.path.insert(0, os.path.dirname(__file__))
from feature_engineer import build_feature_matrix
from xgb_config import (
    XGB_PARAMS, CV_FOLDS, SUSPICIOUS_THRESHOLD,
    MODEL_SAVE_PATH, REPORT_SAVE_PATH, SCORES_SAVE_PATH, ALL_FEATURES,
)

# ─────────────────────────────────────────────────────────────────────────────
# Label generation from Phase 8 risk scores (when no ground truth)
# ─────────────────────────────────────────────────────────────────────────────

# Accounts scoring >= this threshold from Phase 8 get label=1 (suspicious)
# Set at 40 → catches accounts with 3+ independent fraud signals stacked.
# Realistic AML hit rate for a mixed dataset: ~5-10% of accounts.
PSEUDO_LABEL_THRESHOLD = 40


def generate_pseudo_labels(risk_df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive binary fraud labels from Phase 8 risk scores.
    Accounts scoring >= PSEUDO_LABEL_THRESHOLD are labelled fraud=1.
    """
    labels = risk_df[["account_id", "risk_score", "risk_tier", "active_patterns"]].copy()
    labels["is_fraud"] = (
        pd.to_numeric(labels["risk_score"], errors="coerce").fillna(0) >= PSEUDO_LABEL_THRESHOLD
    ).astype(int)

    n_fraud = labels["is_fraud"].sum()
    n_legit = len(labels) - n_fraud
    rate    = n_fraud / len(labels) * 100

    print(f"\n  Label generation (threshold ≥ {PSEUDO_LABEL_THRESHOLD}):")
    print(f"    Fraud  : {n_fraud:3d} accounts ({rate:.1f}%)")
    print(f"    Legit  : {n_legit:3d} accounts ({100-rate:.1f}%)")
    print(f"    Ratio  : 1 : {n_legit//max(n_fraud,1)}")

    if n_fraud < 3:
        raise ValueError(
            f"Only {n_fraud} fraud accounts at threshold {PSEUDO_LABEL_THRESHOLD}. "
            "Lower PSEUDO_LABEL_THRESHOLD or provide real labels with --labels."
        )

    # Print who was labelled fraud
    print("\n  Fraud-labelled accounts:")
    for _, row in labels[labels["is_fraud"] == 1].iterrows():
        print(f"    {row['account_id']:25s}  score={row['risk_score']:.0f}  "
              f"patterns={row['active_patterns']}")

    return labels[["account_id", "is_fraud"]]


# ─────────────────────────────────────────────────────────────────────────────
# Main training pipeline
# ─────────────────────────────────────────────────────────────────────────────

def train(
    txn_path:    str,
    risk_path:   str,
    out_dir:     str,
    labels_path: str | None = None,
):
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*65}")
    print("  Phase 9 — XGBoost Supervised Training")
    print(f"{'='*65}")

    # ── Load Phase 8 outputs ─────────────────────────────────────────────
    print("\n  [1/5] Loading data ...")
    txn_df  = pd.read_csv(txn_path,  dtype=str)
    risk_df = pd.read_csv(risk_path, dtype=str)
    risk_df["risk_score"] = pd.to_numeric(risk_df["risk_score"], errors="coerce").fillna(0.0)

    print(f"        Transactions : {len(txn_df):,}")
    print(f"        Accounts     : {txn_df['account_id'].nunique()}")

    # ── Feature engineering ──────────────────────────────────────────────
    print("\n  [2/5] Engineering features ...")
    feature_df = build_feature_matrix(txn_df, risk_df)
    print(f"        Feature matrix : {feature_df.shape[0]} accounts × {feature_df.shape[1]} features")

    # ── Labels ───────────────────────────────────────────────────────────
    print("\n  [3/5] Building labels ...")
    if labels_path and os.path.exists(labels_path):
        labels_df = pd.read_csv(labels_path)[["account_id", "is_fraud"]]
        labels_df["is_fraud"] = labels_df["is_fraud"].astype(bool).astype(int)
        print(f"        Using ground truth labels from {os.path.basename(labels_path)}")
        n_fraud = labels_df["is_fraud"].sum()
        n_legit = len(labels_df) - n_fraud
        print(f"        Fraud: {n_fraud} | Legit: {n_legit}")
    else:
        print("        No ground truth provided — generating pseudo-labels from Phase 8 scores")
        labels_df = generate_pseudo_labels(risk_df)

    # ── Merge features + labels ──────────────────────────────────────────
    merged = (
        feature_df.reset_index()
        .merge(labels_df, on="account_id", how="left")
    )
    merged["is_fraud"] = merged["is_fraud"].fillna(0).astype(int)
    merged = merged.set_index("account_id")

    X = merged[ALL_FEATURES].values.astype(np.float32)
    y = merged["is_fraud"].values

    n_fraud  = int(y.sum())
    n_legit  = int((y == 0).sum())
    spw      = round(n_legit / max(n_fraud, 1), 2)

    # ── Train with cross-validation ──────────────────────────────────────
    actual_folds = min(CV_FOLDS, max(2, min(n_fraud, n_legit)))
    if actual_folds < CV_FOLDS:
        print(f"\n        Note: reduced to {actual_folds}-fold CV (small fraud class)")

    print(f"\n  [4/5] Training ({actual_folds}-fold CV) ...")
    print(f"        SPW (scale_pos_weight) = {spw}")

    params = {k: v for k, v in XGB_PARAMS.items() if k != "use_label_encoder"}
    params["scale_pos_weight"] = spw

    # Cross-val for unbiased evaluation
    model_cv = XGBClassifier(**params)
    cv        = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=42)
    y_prob_cv = cross_val_predict(model_cv, X, y, cv=cv, method="predict_proba")[:, 1]
    y_pred_cv = (y_prob_cv >= SUSPICIOUS_THRESHOLD).astype(int)

    clf_rep  = classification_report(y, y_pred_cv, output_dict=True, zero_division=0)
    cm       = confusion_matrix(y, y_pred_cv).tolist()
    roc_auc  = float(roc_auc_score(y, y_prob_cv))  if n_fraud > 1 else 0.0
    avg_prec = float(average_precision_score(y, y_prob_cv)) if n_fraud > 1 else 0.0

    print(f"\n        ── Evaluation ──────────────────────────────")
    print(f"        ROC-AUC            : {roc_auc:.4f}")
    print(f"        Avg Precision (AP) : {avg_prec:.4f}")
    print(f"        Fraud Recall       : {clf_rep.get('1',{}).get('recall',0):.4f}")
    print(f"        Fraud Precision    : {clf_rep.get('1',{}).get('precision',0):.4f}")
    print(f"        Fraud F1           : {clf_rep.get('1',{}).get('f1-score',0):.4f}")
    print(f"        Confusion matrix   : {cm}")

    # ── Train final model on ALL data ────────────────────────────────────
    print("\n  [5/5] Training final model on full dataset ...")
    final_model = XGBClassifier(**params)
    final_model.fit(X, y)

    # Feature importance
    importances = final_model.feature_importances_
    feat_imp    = pd.DataFrame({
        "feature":    ALL_FEATURES,
        "importance": importances,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    print("\n        Top 10 features by importance:")
    for _, row in feat_imp.head(10).iterrows():
        bar = "█" * int(row["importance"] * 200)
        print(f"        {row['feature']:<30} {row['importance']:.4f}  {bar}")

    # Save model
    model_path = os.path.join(out_dir, MODEL_SAVE_PATH)
    with open(model_path, "wb") as f:
        pickle.dump(final_model, f)
    print(f"\n        Model saved → {model_path}")

    # ── Build scores for all accounts ────────────────────────────────────
    y_prob_final = final_model.predict_proba(X)[:, 1]
    feat_imp_dict = feat_imp.set_index("feature")["importance"].to_dict()

    score_rows = []
    for i, acc_id in enumerate(merged.index):
        prob  = float(y_prob_final[i])
        tier  = _tier(prob)
        feats = merged.iloc[i][ALL_FEATURES]

        top = sorted(
            [(f, feat_imp_dict.get(f, 0) * abs(float(feats[f]))) for f in ALL_FEATURES],
            key=lambda x: x[1], reverse=True
        )
        top_str = " | ".join(
            f"{f}={float(feats[f]):.3g}" for f, s in top[:5] if s > 0
        )
        exp = (
            f"{'High' if prob>=0.5 else 'Moderate' if prob>=0.25 else 'Low'} "
            f"suspicion ({prob:.1%}) — top signals: "
            + ", ".join(f"{f}={float(feats[f]):.3g}" for f, s in top[:3] if s > 0)
        )
        score_rows.append({
            "account_id":           acc_id,
            "xgb_fraud_probability":round(prob, 4),
            "xgb_tier":             tier,
            "is_suspicious":        prob >= SUSPICIOUS_THRESHOLD,
            "ground_truth_label":   int(y[i]),
            "top_features":         top_str,
            "shap_explanation":     exp,
        })

    scores_df = pd.DataFrame(score_rows).sort_values(
        "xgb_fraud_probability", ascending=False
    )
    scores_df.to_csv(os.path.join(out_dir, SCORES_SAVE_PATH), index=False)

    # ── Save report ───────────────────────────────────────────────────────
    report = {
        "run_timestamp":          datetime.now().isoformat(),
        "mode":                   "supervised",
        "label_source":           os.path.basename(labels_path) if labels_path else "pseudo_labels_phase8",
        "pseudo_label_threshold": PSEUDO_LABEL_THRESHOLD if not labels_path else None,
        "accounts_total":         len(merged),
        "fraud_accounts":         n_fraud,
        "legit_accounts":         n_legit,
        "features":               len(ALL_FEATURES),
        "cv_folds":               actual_folds,
        "roc_auc":                roc_auc,
        "average_precision":      avg_prec,
        "fraud_recall":           clf_rep.get("1", {}).get("recall", 0),
        "fraud_precision":        clf_rep.get("1", {}).get("precision", 0),
        "fraud_f1":               clf_rep.get("1", {}).get("f1-score", 0),
        "confusion_matrix":       cm,
        "suspicious_threshold":   SUSPICIOUS_THRESHOLD,
        "top_10_features":        feat_imp.head(10)["feature"].tolist(),
        "feature_importances":    feat_imp.set_index("feature")["importance"].round(4).to_dict(),
    }

    with open(os.path.join(out_dir, REPORT_SAVE_PATH), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    # ── Final summary ─────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  TRAINING COMPLETE")
    print(f"{'='*65}")
    print(f"  Accounts scored   : {len(scores_df)}")
    print(f"  CRITICAL (≥0.75)  : {(scores_df['xgb_tier']=='CRITICAL').sum()}")
    print(f"  HIGH     (≥0.50)  : {(scores_df['xgb_tier']=='HIGH').sum()}")
    print(f"  MEDIUM   (≥0.25)  : {(scores_df['xgb_tier']=='MEDIUM').sum()}")
    print(f"  LOW      (<0.25)  : {(scores_df['xgb_tier']=='LOW').sum()}")
    print(f"  ROC-AUC           : {roc_auc:.4f}")
    print(f"  Avg Precision     : {avg_prec:.4f}")
    print(f"\n  Outputs → {os.path.abspath(out_dir)}/")
    print(f"    • {MODEL_SAVE_PATH:<30} ← trained model")
    print(f"    • {SCORES_SAVE_PATH:<30} ← all account scores")
    print(f"    • {REPORT_SAVE_PATH:<30} ← full metrics report")
    print(f"{'='*65}\n")

    return scores_df, report


def _tier(prob: float) -> str:
    if prob >= 0.75: return "CRITICAL"
    if prob >= 0.50: return "HIGH"
    if prob >= 0.25: return "MEDIUM"
    return "LOW"


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Phase 9 — XGBoost Supervised Training")
    ap.add_argument("--txn",    required=True,  help="analytics_transactions.csv (Phase 8)")
    ap.add_argument("--risk",   required=True,  help="risk_scores.csv (Phase 8)")
    ap.add_argument("--labels", default=None,   help="account_labels.csv (optional ground truth)")
    ap.add_argument("--out",    default="xgb_output_trained", help="Output directory")
    args = ap.parse_args()
    train(args.txn, args.risk, args.out, args.labels)
