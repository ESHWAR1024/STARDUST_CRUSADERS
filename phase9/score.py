"""
Phase 9 — Main entry point

Usage examples
--------------

  # Unsupervised (real statements, no ground truth):
  python score.py --txn   ../phase8/analytics/analytics_transactions.csv
                  --risk  ../phase8/analytics/risk_scores.csv
                  --out   xgb_output/

  # Supervised (synthetic data with known labels):
  python score.py --txn      ../phase8/analytics/analytics_transactions.csv
                  --risk     ../phase8/analytics/risk_scores.csv
                  --labels   ../phase5/output/account_labels.csv
                  --out      xgb_output/

  # Use a pre-trained model on new data:
  python score.py --txn    ../phase8/analytics/analytics_transactions.csv
                  --risk   ../phase8/analytics/risk_scores.csv
                  --model  xgb_output/xgb_model.pkl
                  --out    xgb_output/
"""

import argparse
from xgb_model import train_and_score


def main():
    ap = argparse.ArgumentParser(description="Phase 9 — XGBoost Behavioral Layer")
    ap.add_argument("--txn",    required=True,
                    help="analytics_transactions.csv (Phase 8 output)")
    ap.add_argument("--risk",   required=True,
                    help="risk_scores.csv (Phase 8 output)")
    ap.add_argument("--labels", default=None,
                    help="account_labels.csv (Phase 5 ground truth, optional)")
    ap.add_argument("--model",  default=None,
                    help="Pre-trained xgb_model.pkl to load (optional)")
    ap.add_argument("--out",    default="xgb_output",
                    help="Output directory (default: xgb_output/)")
    args = ap.parse_args()

    train_and_score(
        txn_path    = args.txn,
        risk_path   = args.risk,
        labels_path = args.labels,
        model_path  = args.model,
        out_dir     = args.out,
    )


if __name__ == "__main__":
    main()
