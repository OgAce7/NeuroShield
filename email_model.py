"""
neuroshield/email_model.py
───────────────────────────
Trains a spam/phishing classifier on the merged phishing-email corpus
(Enron, Ling, SpamAssassin, CEAS_08, Nazario, Nigerian_Fraud — see
data_loader.load_email_dataset_v2).

Architecture
────────────
  Text pipeline  →  HashingVectorizer (char n-grams + word n-grams)  →  SGDClassifier
  Meta features  →  EmailMetaFeatures  →  RandomForest
  Fusion         →  LogisticRegression meta-learner on stacked OOF probabilities

Why HashingVectorizer + SGDClassifier instead of TF-IDF + LinearSVC:
this box has ~4GB RAM and 1 CPU. At 82k rows (vs. the original 5.7k),
the original TfidfVectorizer + LinearSVC + CalibratedClassifierCV(cv=5)
design — which fits a full in-memory vocabulary, then refits the SVC
5-20 times across the main calibration and OOF-stacking loop — reliably
OOM-killed the process. HashingVectorizer needs no vocabulary at all
(fixed-size output, feature index = hash of the token), and SGDClassifier
fits in a single streaming pass. Together they get equivalent accuracy
(~98% on this corpus) in a few seconds and a few GB, instead of crashing.

We also cap how much of each email's text feeds the *character*
vectorizer (see CHAR_CAP below) — char n-grams over a 20k-char email
generate a huge number of tokens per document, and that token-generation
step (not the final matrix) was the actual memory/time bottleneck.

Both models are saved together so the API can load and serve them.
"""

from __future__ import annotations

import json
import logging
import pickle
import re
import string
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report,
)
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import HashingVectorizer
from scipy.sparse import hstack, csr_matrix

from features import extract_email_meta, email_df_to_meta, EmailMetaFeatures

log = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODELS_DIR / "email_model.pkl"
META_PATH  = MODELS_DIR / "email_meta.json"

# How much of each (cleaned) email's text feeds each vectoriser. Full text
# still goes into extract_email_meta() (char_count, caps_ratio, etc. need
# the real length) — these caps only bound the vectorisers' input.
CHAR_CAP = 1_000   # char n-grams: phishing tells concentrate in the opening
WORD_CAP = 3_000   # word n-grams: cheaper per-char, can afford more


# ── Text cleaning ──────────────────────────────────────────────────────────

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE      = re.compile(r"https?://\S+")
_EMAIL_RE    = re.compile(r"\S+@\S+")
_WS_RE       = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Lightweight cleaning that preserves signal words."""
    t = _HTML_TAG_RE.sub(" ", text)
    t = _URL_RE.sub(" URLTOKEN ", t)
    t = _EMAIL_RE.sub(" EMAILTOKEN ", t)
    t = _WS_RE.sub(" ", t)
    return t.strip()


def _make_vectorizers() -> tuple[HashingVectorizer, HashingVectorizer]:
    hv_char = HashingVectorizer(
        analyzer="char_wb", ngram_range=(3, 5),
        n_features=2 ** 14, alternate_sign=False, norm="l2",
        dtype=np.float32,
    )
    hv_word = HashingVectorizer(
        analyzer="word", ngram_range=(1, 2),
        n_features=2 ** 13, alternate_sign=False, norm="l2",
        token_pattern=r"(?u)\b\w+\b", dtype=np.float32,
    )
    return hv_char, hv_word


def _vectorize(
    texts: np.ndarray,
    hv_char: HashingVectorizer,
    hv_word: HashingVectorizer,
) -> csr_matrix:
    """Apply both hashing vectorisers with their respective char caps."""
    char_in = np.array([t[:CHAR_CAP] for t in texts])
    word_in = np.array([t[:WORD_CAP] for t in texts])
    X_char = hv_char.transform(char_in)
    X_word = hv_word.transform(word_in)
    return hstack([X_char, X_word], format="csr")


# ── Training ───────────────────────────────────────────────────────────────

def train(
    df: Optional[pd.DataFrame] = None,
    save: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Train the email phishing/spam classifier.

    Parameters
    ----------
    df      : Pre-loaded DataFrame with columns [text, label].
              If None, loads the merged corpus via data_loader.
    save    : Persist artefacts to disk.
    verbose : Print training progress.
    """
    if df is None:
        from data_loader import load_email_dataset_v2
        df = load_email_dataset_v2()

    if verbose:
        print(f"\n{'─'*55}")
        print(f"  NeuroShield Email Model — Training")
        print(f"{'─'*55}")
        print(f"  Dataset : {len(df):,} rows  |  spam={df['label'].sum():,}  ham={(df['label']==0).sum():,}")

    # ── Clean text ───────────────────────────────────────────────────────
    df = df.copy()
    df["clean"] = df["text"].apply(clean_text)

    X_text = df["clean"].values
    y      = df["label"].values.astype(int)

    # ── Meta features (uses full, uncapped text) ──────────────────────────
    if verbose:
        print("\n  Extracting meta-features …")
    X_meta, meta_names = email_df_to_meta(df)
    del df  # free the DataFrame; we only need the arrays from here on

    # ── Train/test split ─────────────────────────────────────────────────
    idx = np.arange(len(X_text))
    tr_idx, te_idx = train_test_split(idx, test_size=0.15, stratify=y, random_state=42)

    X_txt_tr, X_txt_te = X_text[tr_idx], X_text[te_idx]
    X_meta_tr, X_meta_te = X_meta[tr_idx], X_meta[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]

    # ── Hashing vectorisers (no fit step — fixed-size output) ────────────
    if verbose:
        print("  Vectorising text (hashing, no vocabulary fit needed) …")
    t0 = time.time()
    hv_char, hv_word = _make_vectorizers()
    X_text_tr = _vectorize(X_txt_tr, hv_char, hv_word)
    if verbose:
        print(f"    done  ({time.time()-t0:.1f}s)")

    # Scale meta features
    meta_scaler = StandardScaler()
    X_meta_s_tr = meta_scaler.fit_transform(X_meta_tr).astype(np.float32)

    # Combine: char-hash + word-hash + meta
    X_full_tr = hstack([X_text_tr, csr_matrix(X_meta_s_tr)], format="csr")
    del X_text_tr

    # ── Text model (SGDClassifier — single streaming pass, low memory) ───
    if verbose:
        print("  Training SGDClassifier (text) …")
    t0 = time.time()
    sgd = SGDClassifier(
        loss="log_loss", class_weight="balanced", random_state=42,
        max_iter=200, alpha=1e-5, n_jobs=1,
        early_stopping=True, n_iter_no_change=5,
    )
    sgd.fit(X_full_tr, y_tr)
    if verbose:
        print(f"    done  ({time.time()-t0:.1f}s, {sgd.n_iter_} iterations)")

    # ── Meta-feature-only RF ──────────────────────────────────────────────
    if verbose:
        print("  Training RandomForest (meta) …")
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=150, max_depth=15,
        min_samples_leaf=2, n_jobs=-1,
        random_state=42, class_weight="balanced",
    )
    rf.fit(X_meta_s_tr, y_tr)
    if verbose:
        print(f"    done  ({time.time()-t0:.1f}s)")

    # ── OOF stacking ─────────────────────────────────────────────────────
    if verbose:
        print("  Building OOF meta-features (5-fold) …")
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    sgd_oof = np.zeros(len(tr_idx))
    rf_oof  = np.zeros(len(tr_idx))

    for fold, (ti, vi) in enumerate(kf.split(X_full_tr, y_tr), 1):
        _sgd = SGDClassifier(
            loss="log_loss", class_weight="balanced", random_state=42,
            max_iter=200, alpha=1e-5, n_jobs=1,
            early_stopping=True, n_iter_no_change=5,
        )
        _sgd.fit(X_full_tr[ti], y_tr[ti])
        sgd_oof[vi] = _sgd.predict_proba(X_full_tr[vi])[:, 1]

        _rf = RandomForestClassifier(
            n_estimators=80, max_depth=15, n_jobs=-1,
            random_state=42, class_weight="balanced",
        )
        _rf.fit(X_meta_s_tr[ti], y_tr[ti])
        rf_oof[vi] = _rf.predict_proba(X_meta_s_tr[vi])[:, 1]

        if verbose:
            print(f"    Fold {fold}/5 done")

    meta_tr_stack = np.column_stack([sgd_oof, rf_oof])

    meta_lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    meta_lr.fit(meta_tr_stack, y_tr)

    del X_full_tr  # free before building the (separate) test matrix

    # ── Evaluate on held-out test set ──────────────────────────────────────
    X_text_te = _vectorize(X_txt_te, hv_char, hv_word)
    X_meta_s_te = meta_scaler.transform(X_meta_te).astype(np.float32)
    X_full_te = hstack([X_text_te, csr_matrix(X_meta_s_te)], format="csr")
    del X_text_te

    meta_te_stack = np.column_stack([
        sgd.predict_proba(X_full_te)[:, 1],
        rf.predict_proba(X_meta_s_te)[:, 1],
    ])

    y_prob = meta_lr.predict_proba(meta_te_stack)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "accuracy": round(float(accuracy_score(y_te, y_pred)), 4),
        "f1":       round(float(f1_score(y_te, y_pred)),       4),
        "roc_auc":  round(float(roc_auc_score(y_te, y_prob)),  4),
        "n_train":  int(len(tr_idx)),
        "n_test":   int(len(te_idx)),
    }

    if verbose:
        print(f"\n{'─'*55}")
        print(f"  Results")
        print(f"{'─'*55}")
        print(f"  Accuracy  : {metrics['accuracy']:.4f}")
        print(f"  F1-Score  : {metrics['f1']:.4f}")
        print(f"  ROC-AUC   : {metrics['roc_auc']:.4f}")
        print()
        print(classification_report(y_te, y_pred, target_names=["ham", "spam"]))

    result = {
        "hv_char": hv_char, "hv_word": hv_word,
        "sgd": sgd, "rf": rf, "meta": meta_lr,
        "meta_scaler": meta_scaler,
        "meta_feature_names": list(meta_names),
        "metrics": metrics,
        "threshold": 0.5,
    }

    if save:
        _save_email(result)

    return result


def _save_email(result: Dict[str, Any]) -> None:
    pkl = {k: result[k] for k in [
        "hv_char", "hv_word", "sgd", "rf", "meta",
        "meta_scaler", "meta_feature_names", "threshold",
    ]}
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pkl, f)
    with open(META_PATH, "w") as f:
        json.dump({"metrics": result["metrics"]}, f, indent=2)
    log.info("Email model saved → %s", MODEL_PATH)


def load() -> Dict[str, Any]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Email model not found at {MODEL_PATH}. Run: python email_model.py"
        )
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


# ── Inference ──────────────────────────────────────────────────────────────

def predict_email(text: str, artifacts: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Analyse a single email body (raw text) and return a structured result.
    """
    if artifacts is None:
        artifacts = load()

    cleaned = clean_text(text)

    # Hashing-vectorised text features
    X_text_vec = _vectorize(
        np.array([cleaned]), artifacts["hv_char"], artifacts["hv_word"],
    )

    # Meta features (full, uncapped text)
    meta_f = extract_email_meta(text)
    X_meta = np.array([meta_f.to_list()])
    X_meta_s = artifacts["meta_scaler"].transform(X_meta)

    X_full = hstack([X_text_vec, csr_matrix(X_meta_s)], format="csr")

    sgd_p = float(artifacts["sgd"].predict_proba(X_full)[0, 1])
    rf_p  = float(artifacts["rf"].predict_proba(X_meta_s)[0, 1])

    conf  = float(artifacts["meta"].predict_proba([[sgd_p, rf_p]])[0, 1])
    score = round(conf * 100, 1)
    thr   = artifacts.get("threshold", 0.5)

    label = "spam" if conf >= thr else ("suspicious" if conf >= 0.3 else "ham")

    # Build indicators
    indicators: List[Dict] = []
    urls = _URL_RE.findall(text)
    if urls:
        indicators.append({"signal": f"{len(urls)} URL(s) found", "risk": "medium" if len(urls) > 2 else "low"})
    if meta_f.num_phish_words >= 3:
        indicators.append({"signal": f"{meta_f.num_phish_words} phishing phrases", "risk": "high"})
    if meta_f.exclamation_count >= 3:
        indicators.append({"signal": f"{meta_f.exclamation_count} exclamation marks", "risk": "medium"})
    if meta_f.caps_ratio > 0.25:
        indicators.append({"signal": f"High caps ratio ({meta_f.caps_ratio:.0%})", "risk": "medium"})
    if meta_f.has_html:
        indicators.append({"signal": "Contains HTML markup", "risk": "low"})
    if conf < 0.3:
        indicators.append({"signal": "Low-risk language", "risk": "none"})

    return {
        "label":        label,
        "risk_score":   score,
        "confidence":   round(conf, 4),
        "sgd_prob":     round(sgd_p, 4),
        "rf_prob":      round(rf_p, 4),
        "meta_features": meta_f.to_dict(),
        "indicators":   indicators,
        "url_count":    len(urls),
        "urls":         urls[:20],  # cap at 20
    }


# ── CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    arts = train(save=True)

    print("\n\n── Live predictions ──")
    tests = [
        ("SPAM", "Congratulations! You have won a $1,000 prize. Click here immediately to claim your reward: http://win-prize.tk/claim?user=you"),
        ("HAM",  "Hi team, the quarterly review meeting is scheduled for Friday at 2pm. Please find the agenda attached."),
        ("SPAM", "URGENT: Your PayPal account has been suspended. Verify now: http://paypa1-secure.ml/verify?token=abc123"),
        ("HAM",  "Thanks for your pull request! The CI pipeline passed. Your changes will be merged after review."),
    ]
    for expected, text in tests:
        r = predict_email(text, arts)
        match = "✓" if r["label"].upper() in (expected, "SUSPICIOUS") else "✗"
        print(f"\n  [{expected}] → {r['label'].upper():<12} score={r['risk_score']}/100  {match}")
        print(f"  Text: {text[:70]}…")
