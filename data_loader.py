"""
neuroshield/data_loader.py
──────────────────────────
Downloads the two Kaggle datasets on first run, caches them locally,
and returns clean, label-encoded DataFrames ready for training.

Datasets
────────
  Email  →  jackksoncsie/spam-email-dataset
  URL    →  sid321axn/malicious-urls-dataset

Setup
─────
  1. Install the Kaggle CLI:  pip install kaggle
  2. Create API token at https://www.kaggle.com/settings → API → Create New Token
  3. Place the downloaded kaggle.json at  ~/.kaggle/kaggle.json
  4. Run this file directly to pre-cache both datasets:
       python data_loader.py

Returns
───────
  load_email_dataset() → pd.DataFrame  columns: text, label (0=ham, 1=spam)
  load_url_dataset()   → pd.DataFrame  columns: url, label (0=benign, 1=malicious)
"""

from __future__ import annotations

import os
import zipfile
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

# ── Cache directory ────────────────────────────────────────────────────────
CACHE_DIR = Path(__file__).parent / ".dataset_cache"
CACHE_DIR.mkdir(exist_ok=True)


# ── Kaggle download helper ─────────────────────────────────────────────────

def _kaggle_download(dataset: str, dest: Path) -> None:
    """
    Download a Kaggle dataset ZIP and extract it into *dest*.
    Skips if *dest* already contains files.
    """
    if dest.exists() and any(dest.iterdir()):
        log.info("Cache hit — skipping download for %s", dataset)
        return

    dest.mkdir(parents=True, exist_ok=True)

    try:
        from kaggle.api.kaggle_api_extended import KaggleApiExtended
        api = KaggleApiExtended()
        api.authenticate()
        log.info("Downloading %s …", dataset)
        api.dataset_download_files(dataset, path=str(dest), unzip=True, quiet=False)
        log.info("Downloaded and extracted → %s", dest)
    except ImportError:
        raise RuntimeError(
            "kaggle package not found. Run: pip install kaggle\n"
            "Then place your API token at ~/.kaggle/kaggle.json"
        )
    except Exception as exc:
        # Try unzipping any leftover zip before re-raising
        zips = list(dest.glob("*.zip"))
        if zips:
            with zipfile.ZipFile(zips[0]) as z:
                z.extractall(dest)
            log.info("Extracted %s", zips[0])
        else:
            raise RuntimeError(
                f"Kaggle download failed for '{dataset}': {exc}\n\n"
                "Ensure ~/.kaggle/kaggle.json exists and the dataset slug is correct."
            ) from exc


# ── Email dataset ─────────────────────────────────────────────────────────

EMAIL_DATASET  = "jackksoncsie/spam-email-dataset"
EMAIL_CACHE    = CACHE_DIR / "email"

# Candidate file names the dataset might ship under
_EMAIL_CANDIDATES = [
    "emails.csv",
    "spam.csv",
    "email.csv",
    "spam_ham_dataset.csv",
]


def load_email_dataset() -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
      text   – full email body (str)
      label  – 0 = ham, 1 = spam (int)
    """
    _kaggle_download(EMAIL_DATASET, EMAIL_CACHE)

    # Find the CSV
    csv_path: Path | None = None
    for name in _EMAIL_CANDIDATES:
        p = EMAIL_CACHE / name
        if p.exists():
            csv_path = p
            break

    if csv_path is None:
        # Fall back to first CSV found anywhere in the cache dir
        candidates = list(EMAIL_CACHE.rglob("*.csv"))
        if not candidates:
            raise FileNotFoundError(
                f"No CSV found in {EMAIL_CACHE}. "
                "Check that the Kaggle download succeeded."
            )
        csv_path = candidates[0]
        log.warning("Using fallback CSV: %s", csv_path)

    log.info("Loading email dataset from %s", csv_path)
    df = pd.read_csv(csv_path, encoding="latin-1")

    # ── Normalise columns ────────────────────────────────────────────────
    # The dataset ships with varying column names across versions.
    # Common layouts:
    #   (v1) "v1" = label string, "v2" = text
    #   (v2) "Category", "Message"
    #   (v3) "label", "text"  /  "spam", "text"

    col_lower = {c.lower(): c for c in df.columns}

    # Text column
    text_col = (
        col_lower.get("text") or col_lower.get("message") or
        col_lower.get("v2")   or col_lower.get("body")    or
        col_lower.get("email")
    )
    # Label column
    label_col = (
        col_lower.get("label") or col_lower.get("category") or
        col_lower.get("v1")    or col_lower.get("spam")     or
        col_lower.get("class")
    )

    if text_col is None or label_col is None:
        raise ValueError(
            f"Cannot identify text/label columns. Found: {list(df.columns)}"
        )

    df = df[[text_col, label_col]].copy()
    df.columns = ["text", "label"]
    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str).str.strip()

    # Map string labels → int
    label_map = {
        "spam": 1, "ham": 0,
        "1": 1, "0": 0,
        1: 1, 0: 0,
        "yes": 1, "no": 0,
    }
    df["label"] = df["label"].map(
        lambda x: label_map.get(str(x).strip().lower(), x)
    )
    df = df[df["label"].isin([0, 1])].copy()
    df["label"] = df["label"].astype(int)

    log.info(
        "Email dataset: %d rows  spam=%d  ham=%d",
        len(df), df["label"].sum(), (df["label"] == 0).sum(),
    )
    return df.reset_index(drop=True)


# ── URL dataset ───────────────────────────────────────────────────────────

URL_DATASET  = "sid321axn/malicious-urls-dataset"
URL_CACHE    = CACHE_DIR / "url"

_URL_CANDIDATES = [
    "malicious_phish.csv",
    "urls.csv",
    "url_data.csv",
    "malicious_urls.csv",
    "dataset.csv",
]

# The Kaggle dataset ships with four type labels; we collapse them:
#   benign           → 0
#   phishing/malware/defacement → 1
_URL_LABEL_MAP = {
    "benign":       0,
    "phishing":     1,
    "malware":      1,
    "defacement":   1,
}


def load_url_dataset(sample: int | None = 200_000) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
      url    – raw URL string
      label  – 0 = benign, 1 = malicious (int)

    The full dataset has ~650k rows. Pass sample=None to load everything.
    """
    _kaggle_download(URL_DATASET, URL_CACHE)

    csv_path: Path | None = None
    for name in _URL_CANDIDATES:
        p = URL_CACHE / name
        if p.exists():
            csv_path = p
            break

    if csv_path is None:
        candidates = list(URL_CACHE.rglob("*.csv"))
        if not candidates:
            raise FileNotFoundError(
                f"No CSV found in {URL_CACHE}. "
                "Check that the Kaggle download succeeded."
            )
        csv_path = candidates[0]
        log.warning("Using fallback CSV: %s", csv_path)

    log.info("Loading URL dataset from %s", csv_path)
    df = pd.read_csv(csv_path)

    # Normalise columns
    col_lower = {c.lower(): c for c in df.columns}
    url_col   = col_lower.get("url") or col_lower.get("urls") or col_lower.get("address")
    type_col  = col_lower.get("type") or col_lower.get("label") or col_lower.get("class")

    if url_col is None or type_col is None:
        raise ValueError(
            f"Cannot identify url/type columns. Found: {list(df.columns)}"
        )

    df = df[[url_col, type_col]].copy()
    df.columns = ["url", "label"]
    df = df.dropna(subset=["url", "label"])
    df["url"] = df["url"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip().str.lower().map(_URL_LABEL_MAP)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    if sample and len(df) > sample:
        # Stratified sample to preserve class balance
        df = (
            df.groupby("label", group_keys=False)
              .apply(lambda x: x.sample(min(len(x), sample // 2), random_state=42))
              .reset_index(drop=True)
        )

    log.info(
        "URL dataset: %d rows  malicious=%d  benign=%d",
        len(df), df["label"].sum(), (df["label"] == 0).sum(),
    )
    return df.reset_index(drop=True)


# ── CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    print("\n── Email dataset ──")
    email_df = load_email_dataset()
    print(email_df.head(3).to_string())
    print(f"\nShape: {email_df.shape}")

    print("\n── URL dataset ──")
    url_df = load_url_dataset(sample=10_000)
    print(url_df.head(3).to_string())
    print(f"\nShape: {url_df.shape}")
