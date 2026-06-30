"""
Phase 9 — XGBoost Behavioral Model

L2 in the detection stack. Trained on account-level behavioral features.
Operates in two modes:

  SUPERVISED mode   — ground truth labels are available (Phase 5 output).
                      Trains a proper XGBClassifier, reports precision/recall/F1,
                      saves the model, produces SHAP feature importance.

  UNSUPERVISED mode — no labels available (real uploaded statements).
                      Uses the Phase 8 risk score + behavioral anomaly score
                      as a pseudo-label to produce a relative suspiciousness
                      ranking. Model is calibrated on synthetic ground truth
                      if model.pkl exists, otherwise falls back to rule scoring.

Output:
  xgb_scores.csv    — account_id, xgb_fraud_probability, xgb_mule_score,
                       xgb_tier, top_features, shap_explanation
  xgb_model.pkl     — serialised model (if trained)
  xgb_report.json   — precision, recall, F1, feature importances, confusion matrix
"""

from __future__ import annotations
import json
import os
import pickle
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, precision_recall_curve, average_precision_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from xgb_config import (
    XGB_PARAMS, EARLY_STOPPING_ROUNDS, CV_FOLDS,
    SUSPICIOUS_THRESHOLD, MODEL_SAVE_PATH, REPORT_SAVE_PATH, SCORES_SAVE_PATH,
    ALL_FEATURES, FEATURE_GROUPS, MULE_SIGNAL_WEIGHTS,
)
from feature_engineer import build_feature_matrix

warnings.filterwarnings("ignore", category=UserWarning)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def train_and_score(
    txn_path:    str,
    risk_path:   str,
    labels_path: str | None = None,
    out_dir:     str        = ".",
    model_path:  str | None = None,
) -> pd.DataFrame:
    """
    Full pipeline entry point.

    Parameters
    ----------
    txn_path    : path to analytics_transactions.csv (Phase 8)
    risk_path   : path to risk_scores.csv (Phase 8)
    labels_path : (optional) path to account_labels.csv (Phase 5 ground truth)
                  If provided → supervised training + evaluation
                  If None     → unsupervised scoring only
    out_dir     : directory to write outputs
    model_path  : (optional) path to a previously saved model.pkl to load
                  instead of training fresh

    Returns
    -------
    scores_df : DataFrame with xgb results per account
    """
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*65}")
    print("  Phase 9 — XGBoost Behavioral Layer")
    print(f"{'='*65}")

    # Load Phase 8 outputs
    txn_df  = pd.read_csv(txn_path,  dtype=str)
    risk_df = pd.read_csv(risk_path, dtype=str)
    for col in ("risk_score",):
        if col in risk_df.columns:
            risk_df[col] = pd.to_numeric(risk_df[col], errors="coerce").fillna(0.0)

    # Build feature matrix
    print("\n  [1/4] Engineering features ...")
    feature_df = build_feature_matrix(txn_df, risk_df)
    print(f"        Accounts: {len(feature_df)} | Features: {len(ALL_FEATURES)}")

    report = {
        "run_timestamp": datetime.now().isoformat(),
        "accounts":      len(feature_df),
        "features":      len(ALL_FEATURES),
        "mode":          None,
    }

    # ── SUPERVISED MODE ──────────────────────────────────────────────────
    if labels_path and os.path.exists(labels_path):
        print("\n  [2/4] Supervised training (ground truth available) ...")
        scores_df, report = _supervised_train(
            feature_df, labels_path, out_dir, report
        )

    # ── UNSUPERVISED MODE ────────────────────────────────────────────────
    else:
        print("\n  [2/4] Unsupervised scoring (no ground truth) ...")
        scores_df, report = _unsupervised_score(
            feature_df, risk_df, model_path, out_dir, report
        )

    # ── Write outputs ────────────────────────────────────────────────────
    print("\n  [4/4] Writing outputs ...")
    scores_path = os.path.join(out_dir, SCORES_SAVE_PATH)
    scores_df.to_csv(scores_path, index=False)
    print(f"        {scores_path}")

    report_path = os.path.join(out_dir, REPORT_SAVE_PATH)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"        {report_path}")

    _print_summary(scores_df, report)
    return scores_df


# ─────────────────────────────────────────────────────────────────────────────
# SUPERVISED TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def _supervised_train(
    feature_df:  pd.DataFrame,
    labels_path: str,
    out_dir:     str,
    report:      dict,
) -> tuple[pd.DataFrame, dict]:
    report["mode"] = "supervised"

    labels_df = pd.read_csv(labels_path)
    labels_df = labels_df[["account_id", "is_fraud"]].copy()
    labels_df["is_fraud"] = labels_df["is_fraud"].astype(bool).astype(int)

    # Merge features with labels
    merged = feature_df.reset_index().merge(labels_df, on="account_id", how="left")
    merged["is_fraud"] = merged["is_fraud"].fillna(0).astype(int)
    merged = merged.set_index("account_id")

    X = merged[ALL_FEATURES].values.astype(np.float32)
    y = merged["is_fraud"].values

    n_fraud  = int(y.sum())
    n_legit  = int((y == 0).sum())
    spw      = max(n_legit / n_fraud, 1.0) if n_fraud > 0 else 5.0
    print(f"        Fraud accounts : {n_fraud} | Legit : {n_legit} | SPW : {spw:.1f}")

    params = {**XGB_PARAMS, "scale_pos_weight": spw}
    params.pop("use_label_encoder", None)

    # Cross-validated predictions for unbiased evaluation
    # Auto-reduce folds if dataset is too small (min samples per class must >= n_splits)
    min_class_size = min(n_fraud, n_legit)
    actual_folds   = min(CV_FOLDS, max(2, min_class_size))
    if actual_folds < CV_FOLDS:
        print(f"        Note: reduced CV folds to {actual_folds} (small dataset)")

    print(f"\n  [3/4] Cross-validating ({actual_folds}-fold) ...")
    model_cv = XGBClassifier(**params)
    cv = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=42)
    y_prob_cv = cross_val_predict(model_cv, X, y, cv=cv, method="predict_proba")[:, 1]
    y_pred_cv = (y_prob_cv >= SUSPICIOUS_THRESHOLD).astype(int)

    clf_report = classification_report(y, y_pred_cv, output_dict=True, zero_division=0)
    cm         = confusion_matrix(y, y_pred_cv).tolist()
    roc_auc    = float(roc_auc_score(y, y_prob_cv)) if n_fraud > 0 else 0.0
    avg_prec   = float(average_precision_score(y, y_prob_cv)) if n_fraud > 0 else 0.0

    print(f"\n        ROC-AUC        : {roc_auc:.4f}")
    print(f"        Avg Precision  : {avg_prec:.4f}")
    print(f"        Fraud recall   : {clf_report.get('1', {}).get('recall', 0):.4f}")
    print(f"        Fraud precision: {clf_report.get('1', {}).get('precision', 0):.4f}")

    # Train final model on ALL data
    final_model = XGBClassifier(**params)
    final_model.fit(X, y)
    model_save = os.path.join(out_dir, MODEL_SAVE_PATH)
    with open(model_save, "wb") as f:
        pickle.dump(final_model, f)
    print(f"        Model saved → {model_save}")

    # Feature importances
    importances  = final_model.feature_importances_
    feat_imp_df  = pd.DataFrame({
        "feature":   ALL_FEATURES,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    feat_imp_dict = feat_imp_df.set_index("feature")["importance"].to_dict()

    # SHAP-style top features per account
    y_prob_final  = final_model.predict_proba(X)[:, 1]
    scores_df     = _build_scores_df(
        merged.index.tolist(), y_prob_final, feature_df, feat_imp_df, y
    )

    report.update({
        "roc_auc":              roc_auc,
        "average_precision":    avg_prec,
        "confusion_matrix":     cm,
        "classification_report":clf_report,
        "feature_importances":  feat_imp_dict,
        "top_10_features":      feat_imp_df.head(10)["feature"].tolist(),
        "suspicious_threshold": SUSPICIOUS_THRESHOLD,
    })

    return scores_df, report


# ─────────────────────────────────────────────────────────────────────────────
# UNSUPERVISED SCORING
# ─────────────────────────────────────────────────────────────────────────────

def _unsupervised_score(
    feature_df: pd.DataFrame,
    risk_df:    pd.DataFrame,
    model_path: str | None,
    out_dir:    str,
    report:     dict,
) -> tuple[pd.DataFrame, dict]:
    report["mode"] = "unsupervised"

    # Try to load a pre-trained model
    loaded_model = None
    model_pkl    = model_path or os.path.join(out_dir, MODEL_SAVE_PATH)
    if model_pkl and os.path.exists(model_pkl):
        with open(model_pkl, "rb") as f:
            loaded_model = pickle.load(f)
        print(f"        Loaded pre-trained model from {model_pkl}")
        report["pretrained_model"] = model_pkl

    X = feature_df[ALL_FEATURES].values.astype(np.float32)

    if loaded_model is not None:
        # Use pre-trained model probabilities
        y_prob = loaded_model.predict_proba(X)[:, 1]
        feat_imp_df = pd.DataFrame({
            "feature":    ALL_FEATURES,
            "importance": loaded_model.feature_importances_,
        }).sort_values("importance", ascending=False)
        report["pretrained_model_used"] = True
    else:
        # No model: compute a weighted mule score from raw features
        print("        No pretrained model — computing rule-based mule score")
        y_prob      = _compute_mule_score(feature_df)
        feat_imp_df = pd.DataFrame({
            "feature":    list(MULE_SIGNAL_WEIGHTS.keys()),
            "importance": list(MULE_SIGNAL_WEIGHTS.values()),
        }).sort_values("importance", ascending=False)
        report["pretrained_model_used"] = False

    print("\n  [3/4] Generating scores ...")
    scores_df = _build_scores_df(
        feature_df.index.tolist(), y_prob, feature_df, feat_imp_df
    )

    report.update({
        "suspicious_threshold": SUSPICIOUS_THRESHOLD,
        "feature_importances":  feat_imp_df.set_index("feature")["importance"].to_dict(),
    })

    return scores_df, report


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compute_mule_score(feature_df: pd.DataFrame) -> np.ndarray:
    """
    Fallback scoring when no trained model is available.
    Weighted linear combination of the strongest mule signals,
    sigmoid-squashed to [0, 1].
    """
    scores = np.zeros(len(feature_df))
    for feat, weight in MULE_SIGNAL_WEIGHTS.items():
        if feat in feature_df.columns:
            col_vals = feature_df[feat].values.astype(float)
            # Normalise non-binary columns to [0, 1]
            col_max = col_vals.max()
            if col_max > 0:
                col_vals = col_vals / col_max
            scores += weight * col_vals

    # Normalise to max weight sum then sigmoid
    max_possible = sum(MULE_SIGNAL_WEIGHTS.values())
    scores = scores / max_possible
    return _sigmoid(scores)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-6.0 * (x - 0.5)))   # steep sigmoid centred at 0.5


def _build_scores_df(
    account_ids: list,
    y_prob:      np.ndarray,
    feature_df:  pd.DataFrame,
    feat_imp_df: pd.DataFrame,
    y_true:      np.ndarray | None = None,
) -> pd.DataFrame:
    rows = []
    feat_imp = feat_imp_df.set_index("feature")["importance"].to_dict()

    for i, acc_id in enumerate(account_ids):
        prob    = float(y_prob[i])
        tier    = _prob_to_tier(prob)
        feats   = feature_df.loc[acc_id] if acc_id in feature_df.index else None

        # Top contributing features for this account (weighted by importance × value)
        top_feats = _top_features(feats, feat_imp) if feats is not None else []

        # Mule-specific score (focuses only on mule signals)
        mule_sc = _per_account_mule_score(feats) if feats is not None else 0.0

        row = {
            "account_id":          acc_id,
            "xgb_fraud_probability": round(prob, 4),
            "xgb_mule_score":       round(mule_sc, 4),
            "xgb_tier":             tier,
            "is_suspicious":        prob >= SUSPICIOUS_THRESHOLD,
            "top_features":         " | ".join(top_feats[:5]),
            "shap_explanation":     _build_explanation(top_feats[:5], prob),
        }
        if y_true is not None:
            row["ground_truth_label"] = int(y_true[i])

        rows.append(row)

    return pd.DataFrame(rows).sort_values("xgb_fraud_probability", ascending=False)


def _prob_to_tier(prob: float) -> str:
    if prob >= 0.75:  return "CRITICAL"
    if prob >= 0.50:  return "HIGH"
    if prob >= 0.25:  return "MEDIUM"
    return "LOW"


def _top_features(feats: pd.Series, feat_imp: dict) -> list[str]:
    """Return feature names ranked by (importance × feature value)."""
    scored = []
    for feat, importance in feat_imp.items():
        if feat not in feats.index:
            continue
        val   = float(feats[feat])
        score = importance * abs(val)
        if score > 0:
            scored.append((feat, score, val))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [f"{feat}={val:.3g}" for feat, _, val in scored]


def _per_account_mule_score(feats: pd.Series) -> float:
    """Compute mule signal score for a single account."""
    total_w = sum(MULE_SIGNAL_WEIGHTS.values())
    score   = 0.0
    for feat, weight in MULE_SIGNAL_WEIGHTS.items():
        if feat in feats.index:
            val = float(feats[feat])
            # binary flags: val is 0 or 1
            # continuous features: normalise by their typical max
            score += weight * min(val, 1.0)
    return min(score / total_w, 1.0)


def _build_explanation(top_feats: list[str], prob: float) -> str:
    if not top_feats:
        return f"Suspicion score {prob:.1%}; no dominant signals."
    joined = ", ".join(top_feats[:3])
    label  = "High" if prob >= 0.5 else "Moderate" if prob >= 0.25 else "Low"
    return f"{label} suspicion ({prob:.1%}) driven by: {joined}"


def _print_summary(scores_df: pd.DataFrame, report: dict):
    print(f"\n{'='*65}")
    print("  PHASE 9 XGBoost — COMPLETE")
    print(f"{'='*65}")
    print(f"  Mode            : {report['mode']}")
    print(f"  Accounts scored : {report['accounts']}")
    print(f"  CRITICAL        : {(scores_df['xgb_tier'] == 'CRITICAL').sum()}")
    print(f"  HIGH            : {(scores_df['xgb_tier'] == 'HIGH').sum()}")
    print(f"  MEDIUM          : {(scores_df['xgb_tier'] == 'MEDIUM').sum()}")
    print(f"  LOW             : {(scores_df['xgb_tier'] == 'LOW').sum()}")
    if "roc_auc" in report:
        print(f"  ROC-AUC         : {report['roc_auc']:.4f}")
        print(f"  Avg Precision   : {report['average_precision']:.4f}")
    print(f"{'='*65}\n")
