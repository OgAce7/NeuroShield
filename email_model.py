"""
neuroshield/email_model.py
───────────────────────────
Trains a spam/phishing classifier on the Kaggle spam-email dataset.

Architecture
────────────
  Text pipeline  →  TF-IDF (subword + word n-grams)  →  LinearSVC  (calibrated)
  Meta features  →  EmailMetaFeatures  →  RandomForest
  Fusion         →  LogisticRegression meta-learner on stacked OOF probabilities

Both models are saved together so the API can load and serve them.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix

from features import extract_email_meta, email_df_to_meta, EmailMetaFeatures

log = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODELS_DIR / "email_model.pkl"
META_PATH  = MODELS_DIR / "email_meta.json"


# ── Text cleaning ──────────────────────────────────────────────────────────

import re
import string

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
              If None, loads from Kaggle via data_loader.
    save    : Persist artefacts to disk.
    verbose : Print training progress.
    """
    if df is None:
        from data_loader import load_email_dataset
        df = load_email_dataset()

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

    # ── Meta features ────────────────────────────────────────────────────
    if verbose:
        print("\n  Extracting meta-features …")
    X_meta, meta_names = email_df_to_meta(df)

    # ── Train/test split ─────────────────────────────────────────────────
    idx = np.arange(len(df))
    tr_idx, te_idx = train_test_split(idx, test_size=0.15, stratify=y, random_state=42)

    X_txt_tr, X_txt_te = X_text[tr_idx], X_text[te_idx]
    X_meta_tr, X_meta_te = X_meta[tr_idx], X_meta[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]

    # ── TF-IDF vectoriser ────────────────────────────────────────────────
    if verbose:
        print("  Fitting TF-IDF vectoriser …")

    tfidf = TfidfVectorizer(
        analyzer="char_wb",         # character n-grams (robust to typos)
        ngram_range=(3, 5),
        max_features=60_000,
        sublinear_tf=True,
        strip_accents="unicode",
        min_df=3,
    )
    # Word-level on top (stacked)
    tfidf_word = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=40_000,
        sublinear_tf=True,
        strip_accents="unicode",
        min_df=2,
        token_pattern=r"(?u)\b\w+\b",
    )

    X_char_tr = tfidf.fit_transform(X_txt_tr)
    X_word_tr = tfidf_word.fit_transform(X_txt_tr)
    X_char_te = tfidf.transform(X_txt_te)
    X_word_te = tfidf_word.transform(X_txt_te)

    # Scale meta features
    meta_scaler = StandardScaler()
    X_meta_s_tr = meta_scaler.fit_transform(X_meta_tr)
    X_meta_s_te = meta_scaler.transform(X_meta_te)

    # Combine: char-TFIDF + word-TFIDF + meta
    X_full_tr = hstack([X_char_tr, X_word_tr, csr_matrix(X_meta_s_tr)])
    X_full_te = hstack([X_char_te, X_word_te, csr_matrix(X_meta_s_te)])

    # ── Text model (LinearSVC, calibrated for probabilities) ─────────────
    if verbose:
        print("  Training LinearSVC (text) …")
    t0 = time.time()
    svc_base = LinearSVC(C=0.8, max_iter=3000, class_weight="balanced", random_state=42)
    svc = CalibratedClassifierCV(svc_base, cv=5, method="sigmoid")
    svc.fit(X_full_tr, y_tr)
    if verbose:
        print(f"    done  ({time.time()-t0:.1f}s)")

    # ── Meta-feature-only RF ──────────────────────────────────────────────
    if verbose:
        print("  Training RandomForest (meta) …")
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=None,
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
    svc_oof = np.zeros(len(tr_idx))
    rf_oof  = np.zeros(len(tr_idx))

    for fold, (ti, vi) in enumerate(kf.split(X_full_tr, y_tr), 1):
        # SVC
        _svc = CalibratedClassifierCV(
            LinearSVC(C=0.8, max_iter=2000, class_weight="balanced", random_state=42),
            cv=3, method="sigmoid",
        )
        _svc.fit(X_full_tr[ti], y_tr[ti])
        svc_oof[vi] = _svc.predict_proba(X_full_tr[vi])[:, 1]

        # RF on meta only
        _rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42, class_weight="balanced")
        _rf.fit(X_meta_s_tr[ti], y_tr[ti])
        rf_oof[vi] = _rf.predict_proba(X_meta_s_tr[vi])[:, 1]

        if verbose:
            print(f"    Fold {fold}/5 done")

    meta_tr_stack = np.column_stack([svc_oof, rf_oof])
    meta_te_stack = np.column_stack([
        svc.predict_proba(X_full_te)[:, 1],
        rf.predict_proba(X_meta_s_te)[:, 1],
    ])

    meta_lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    meta_lr.fit(meta_tr_stack, y_tr)

    # ── Evaluate ─────────────────────────────────────────────────────────
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
        "tfidf": tfidf, "tfidf_word": tfidf_word,
        "svc": svc, "rf": rf, "meta": meta_lr,
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
        "tfidf", "tfidf_word", "svc", "rf", "meta",
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
    from scipy.sparse import hstack, csr_matrix
    if artifacts is None:
        artifacts = load()

    cleaned = clean_text(text)

    # TF-IDF features
    X_char = artifacts["tfidf"].transform([cleaned])
    X_word = artifacts["tfidf_word"].transform([cleaned])

    # Meta features
    meta_f = extract_email_meta(text)
    X_meta = np.array([meta_f.to_list()])
    X_meta_s = artifacts["meta_scaler"].transform(X_meta)

    X_full = hstack([X_char, X_word, csr_matrix(X_meta_s)])

    svc_p = float(artifacts["svc"].predict_proba(X_full)[0, 1])
    rf_p  = float(artifacts["rf"].predict_proba(X_meta_s)[0, 1])

    conf  = float(artifacts["meta"].predict_proba([[svc_p, rf_p]])[0, 1])
    score = round(conf * 100, 1)
    thr   = artifacts.get("threshold", 0.5)

    label = "spam" if conf >= thr else ("suspicious" if conf >= 0.3 else "ham")

    # Build indicators
    indicators: List[Dict] = []
    import re as _re
    urls = _re.findall(r"https?://\S+", text, _re.IGNORECASE)
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
        "svc_prob":     round(svc_p, 4),
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
