"""
neuroshield/train.py
─────────────────────
One-shot training script. Run this first before starting the API.

Usage
─────
  python train.py                  # train both models
  python train.py --url-only       # URL model only
  python train.py --email-only     # Email model only
  python train.py --url-sample 50000  # use 50k URL rows

Prerequisites
─────────────
  1. pip install -r requirements.txt
  2. Place kaggle.json at ~/.kaggle/kaggle.json
     (download from https://www.kaggle.com/settings → API)
"""

import argparse
import logging
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("neuroshield.train")


def train_url(sample: int) -> None:
    log.info("Starting URL model training …")
    t0 = time.time()
    from url_model import train
    result = train(sample=sample, save=True, verbose=True)
    elapsed = round(time.time() - t0, 1)
    m = result["metrics"]
    log.info(
        "URL model done  (%.1fs)  accuracy=%.4f  f1=%.4f  auc=%.4f",
        elapsed, m["accuracy"], m["f1"], m["roc_auc"],
    )


def train_email() -> None:
    log.info("Starting email model training …")
    t0 = time.time()
    from email_model import train
    result = train(save=True, verbose=True)
    elapsed = round(time.time() - t0, 1)
    m = result["metrics"]
    log.info(
        "Email model done  (%.1fs)  accuracy=%.4f  f1=%.4f  auc=%.4f",
        elapsed, m["accuracy"], m["f1"], m["roc_auc"],
    )


def main():
    parser = argparse.ArgumentParser(description="NeuroShield model training")
    parser.add_argument("--url-only",    action="store_true", help="Train URL model only")
    parser.add_argument("--email-only",  action="store_true", help="Train email model only")
    parser.add_argument("--url-sample",  type=int, default=200_000,
                        help="Max URL rows to use (default 200k)")
    args = parser.parse_args()

    print("\n" + "═" * 58)
    print("  NeuroShield — Model Training Pipeline")
    print("═" * 58)

    if args.email_only:
        train_email()
    elif args.url_only:
        train_url(args.url_sample)
    else:
        train_url(args.url_sample)
        print()
        train_email()

    print("\n" + "═" * 58)
    print("  Training complete. Start the API with:")
    print("  uvicorn api:app --host 0.0.0.0 --port 8000")
    print("═" * 58 + "\n")


if __name__ == "__main__":
    main()
