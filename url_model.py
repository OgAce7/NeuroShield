"""
neuroshield/url_model.py
─────────────────────────
Trains a stacked ensemble on the Kaggle malicious-URLs dataset.

Architecture
────────────
  Layer 1  →  RandomForest  +  GradientBoosting  (base learners)
  Layer 2  →  LogisticRegression  (meta-learner on OOF probabilities)

Artefacts saved
───────────────
  models/url_model.pkl   – trained pipeline
  models/url_meta.json   – metrics + feature importances
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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix,
)

from features import extract_url_features, url_df_to_features, URLFeatures

log = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODELS_DIR / "url_model.pkl"
META_PATH  = MODELS_DIR / "url_meta.json"


# ── Training ───────────────────────────────────────────────────────────────

def train(
    df: Optional[pd.DataFrame] = None,
    sample: int = 200_000,
    save: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Train the URL detection model.

    Parameters
    ----------
    df      : Pre-loaded DataFrame with columns [url, label].
              If None, loads from Kaggle via data_loader.
    sample  : Max rows to use (stratified). Ignored if df is provided.
    save    : Persist model artefacts to disk.
    verbose : Print training progress.

    Returns
    -------
    dict with keys: rf, gb, meta, scaler, feature_names, metrics, importance
    """
    if df is None:
        from data_loader import load_url_dataset
        df = load_url_dataset(sample=sample)

    if verbose:
        print(f"\n{'─'*55}")
        print(f"  NeuroShield URL Model — Training")
        print(f"{'─'*55}")
        print(f"  Dataset : {len(df):,} rows  |  malicious={df['label'].sum():,}  benign={(df['label']==0).sum():,}")

    X, feature_names = url_df_to_features(df)
    y = df["label"].values.astype(int)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )

    scaler    = StandardScaler()
    X_tr_s    = scaler.fit_transform(X_tr)
    X_te_s    = scaler.transform(X_te)

    # ── Base learners ────────────────────────────────────────────────────
    rf = RandomForestClassifier(
        n_estimators=400, max_depth=None, min_samples_leaf=2,
        n_jobs=-1, random_state=42, class_weight="balanced",
    )
    gb = GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.08,
        subsample=0.8, min_samples_leaf=3, random_state=42,
    )

    if verbose:
        print("\n  Training base learners …")
    t0 = time.time()
    rf.fit(X_tr_s, y_tr)
    if verbose:
        print(f"    RandomForest  done  ({time.time()-t0:.1f}s)")
    t1 = time.time()
    gb.fit(X_tr_s, y_tr)
    if verbose:
        print(f"    GradientBoost done  ({time.time()-t1:.1f}s)")

    # ── OOF meta-features ────────────────────────────────────────────────
    kf    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rf_oof = np.zeros(len(X_tr_s))
    gb_oof = np.zeros(len(X_tr_s))

    if verbose:
        print("\n  Building OOF meta-features (5-fold) …")

    for fold, (ti, vi) in enumerate(kf.split(X_tr_s, y_tr), 1):
        _rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42, class_weight="balanced")
        _gb = GradientBoostingClassifier(n_estimators=80, random_state=42)
        _rf.fit(X_tr_s[ti], y_tr[ti])
        _gb.fit(X_tr_s[ti], y_tr[ti])
        rf_oof[vi] = _rf.predict_proba(X_tr_s[vi])[:, 1]
        gb_oof[vi] = _gb.predict_proba(X_tr_s[vi])[:, 1]
        if verbose:
            print(f"    Fold {fold}/5 done")

    meta_tr = np.column_stack([rf_oof, gb_oof])
    meta_te = np.column_stack([
        rf.predict_proba(X_te_s)[:, 1],
        gb.predict_proba(X_te_s)[:, 1],
    ])

    meta = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    meta.fit(meta_tr, y_tr)

    # ── Evaluate ─────────────────────────────────────────────────────────
    y_prob = meta.predict_proba(meta_te)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "accuracy": round(float(accuracy_score(y_te, y_pred)), 4),
        "f1":       round(float(f1_score(y_te, y_pred)),       4),
        "roc_auc":  round(float(roc_auc_score(y_te, y_prob)),  4),
        "n_train":  int(len(X_tr)),
        "n_test":   int(len(X_te)),
    }

    if verbose:
        print(f"\n{'─'*55}")
        print(f"  Results")
        print(f"{'─'*55}")
        print(f"  Accuracy  : {metrics['accuracy']:.4f}")
        print(f"  F1-Score  : {metrics['f1']:.4f}")
        print(f"  ROC-AUC   : {metrics['roc_auc']:.4f}")
        print()
        print(classification_report(y_te, y_pred, target_names=["benign", "malicious"]))

    # ── Feature importance ────────────────────────────────────────────────
    importance = {
        fn: round(float(iv), 6)
        for fn, iv in zip(feature_names, rf.feature_importances_)
    }
    top = sorted(importance.items(), key=lambda x: -x[1])
    if verbose:
        print("  Top-10 features:")
        for name, imp in top[:10]:
            print(f"    {name:<30} {imp:.5f}")

    result = {
        "rf": rf, "gb": gb, "meta": meta, "scaler": scaler,
        "feature_names": list(feature_names),
        "metrics": metrics,
        "importance": importance,
        "threshold": 0.5,
    }

    if save:
        _save_url(result)

    return result


def _save_url(result: Dict[str, Any]) -> None:
    pkl = {k: result[k] for k in ["rf", "gb", "meta", "scaler", "feature_names", "threshold"]}
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pkl, f)
    with open(META_PATH, "w") as f:
        json.dump({"metrics": result["metrics"], "importance": result["importance"]}, f, indent=2)
    log.info("URL model saved → %s", MODEL_PATH)


def load() -> Dict[str, Any]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"URL model not found at {MODEL_PATH}. Run: python url_model.py"
        )
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


# ── Inference ──────────────────────────────────────────────────────────────

def predict_url(url: str, artifacts: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Analyse a single URL and return a structured result dict.

    Returns
    -------
    {
      url, label, risk_score (0–100), confidence,
      rf_prob, gb_prob, features, indicators
    }
    """
    if artifacts is None:
        artifacts = load()

    feats  = extract_url_features(url)
    fd     = feats.to_dict()
    fn     = artifacts["feature_names"]
    X      = np.array([[fd.get(k, 0.0) for k in fn]])
    X_s    = artifacts["scaler"].transform(X)

    rf_p   = float(artifacts["rf"].predict_proba(X_s)[0, 1])
    gb_p   = float(artifacts["gb"].predict_proba(X_s)[0, 1])
    conf   = float(artifacts["meta"].predict_proba([[rf_p, gb_p]])[0, 1])

    score  = round(conf * 100, 1)
    thr    = artifacts.get("threshold", 0.5)

    if conf >= thr:
        label = "malicious"
    elif conf >= 0.3:
        label = "suspicious"
    else:
        label = "benign"

    # Human-readable indicators
    indicators: List[Dict] = []
    if not feats.has_https:
        indicators.append({"signal": "No HTTPS",              "risk": "high"})
    if feats.has_ip:
        indicators.append({"signal": "IP address in URL",     "risk": "high"})
    if feats.has_risky_tld:
        indicators.append({"signal": f"Risky TLD",            "risk": "high"})
    if feats.brand_not_apex:
        indicators.append({"signal": "Brand impersonation",   "risk": "high"})
    if feats.num_sus_words >= 2:
        indicators.append({"signal": f"{feats.num_sus_words} suspicious keywords", "risk": "medium"})
    if feats.has_redirect:
        indicators.append({"signal": "Redirect parameter",    "risk": "medium"})
    if feats.has_hex:
        indicators.append({"signal": "Hex-encoded characters","risk": "medium"})
    if feats.num_subdomains >= 3:
        indicators.append({"signal": f"{feats.num_subdomains} subdomain levels", "risk": "medium"})
    if feats.has_shortener:
        indicators.append({"signal": "URL shortener",         "risk": "medium"})
    if feats.has_https and not feats.has_risky_tld and feats.num_sus_words == 0:
        indicators.append({"signal": "HTTPS secure",          "risk": "none"})
        indicators.append({"signal": "Clean domain",          "risk": "none"})

    return {
        "url":         url,
        "label":       label,
        "risk_score":  score,
        "confidence":  round(conf, 4),
        "rf_prob":     round(rf_p, 4),
        "gb_prob":     round(gb_p, 4),
        "features":    fd,
        "indicators":  indicators,
    }


def predict_batch(urls: List[str], artifacts: Optional[Dict] = None) -> List[Dict]:
    if artifacts is None:
        artifacts = load()
    return [predict_url(u, artifacts) for u in urls]


# ── CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    arts = train(save=True)

    print("\n\n── Live predictions ──")
    tests = [
        "https://www.google.com",
        "http://paypal-secure-login.tk/verify?user=john&token=abc",
        "http://192.168.1.10/banking/login.php",
        "https://github.com/python/cpython",
        "http://amazon-update-billing.ml/signin?redirect=http://evil.ru",
        "https://stackoverflow.com/questions/1234567",
    ]
    for u in tests:
        r = predict_url(u, arts)
        print(f"\n  {u}")
        print(f"  → {r['label'].upper():<12} score={r['risk_score']}/100  conf={r['confidence']:.4f}")
        for ind in r["indicators"][:3]:
            print(f"     • {ind['signal']}")
