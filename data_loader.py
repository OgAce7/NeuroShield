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


# ── Email dataset v2 (merged phishing-email corpus) ────────────────────────
#
# The original 5.7k-row Kaggle set above is small (~4.9k train rows) and
# entirely 2000s-era Enron/stock spam — no modern phishing patterns
# (fake MFA prompts, invoice fraud, OAuth consent phishing, etc.), and
# too little data for the 100k-feature TF-IDF vectoriser to generalise
# well. This loader merges six raw source files (Enron, Ling,
# SpamAssassin, CEAS_08, Nazario, Nigerian_Fraud — the same sources
# behind several published phishing-detection benchmarks) into one
# ~82k-row, roughly balanced dataset spanning old-school spam through
# modern-style phishing/scam text.
#
# Unlike a pre-merged/lowercased version of this corpus, we rebuild the
# merge from the raw per-source files so we keep original casing, HTML,
# and punctuation — features.py's meta-features (caps_ratio, has_html,
# exclamation_count, url_count) need that raw signal; a lowercased/
# stripped merge would zero most of them out.

EMAIL_V2_CACHE = CACHE_DIR / "email_v2"

# (filename, columns-to-use, label-value-if-single-class-file)
# Nazario and Nigerian_Fraud are single-class files (100% phishing) in
# this corpus, so they have no "label" column ambiguity — hardcode it.
_EMAIL_V2_SOURCES = [
    ("CEAS_08.csv",       None),
    ("Enron.csv",         None),
    ("Ling.csv",          None),
    ("Nazario.csv",       1),
    ("Nigerian_Fraud.csv", 1),
    ("SpamAssasin.csv",   None),
]


def load_email_dataset_v2(dedupe: bool = True) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
      text   – "{subject}\n\n{body}" (str)
      label  – 0 = legitimate, 1 = phishing/spam (int)

    Loads and merges the six raw CSVs cached under
    .dataset_cache/email_v2/ (see _EMAIL_V2_SOURCES). These are expected
    to already be present (copied in manually / via a prior download) —
    this loader does not fetch them itself, since they come from a
    Kaggle dataset that requires auth and isn't wired into
    _kaggle_download's simple slug-download path.
    """
    if not EMAIL_V2_CACHE.exists():
        raise FileNotFoundError(
            f"{EMAIL_V2_CACHE} not found. Populate it with the raw source "
            "CSVs (CEAS_08.csv, Enron.csv, Ling.csv, Nazario.csv, "
            "Nigerian_Fraud.csv, SpamAssasin.csv) from "
            "kaggle.com/datasets/naserabdullahalam/phishing-email-dataset"
        )

    frames = []
    for fname, forced_label in _EMAIL_V2_SOURCES:
        path = EMAIL_V2_CACHE / fname
        if not path.exists():
            log.warning("Skipping missing source file: %s", fname)
            continue

        raw = pd.read_csv(path)
        col_lower = {c.lower(): c for c in raw.columns}

        subject_col = col_lower.get("subject")
        body_col    = col_lower.get("body") or col_lower.get("message")
        label_col   = col_lower.get("label")

        if body_col is None:
            log.warning("Skipping %s: no body/message column found", fname)
            continue

        subj = raw[subject_col].astype(str).fillna("") if subject_col else ""
        body = raw[body_col].astype(str).fillna("")
        text = (subj + "\n\n" + body) if subject_col else body

        if forced_label is not None:
            label = pd.Series(forced_label, index=raw.index)
        elif label_col is not None:
            label = pd.to_numeric(raw[label_col], errors="coerce")
        else:
            log.warning("Skipping %s: no label column and no forced label", fname)
            continue

        part = pd.DataFrame({"text": text, "label": label, "source": fname})
        frames.append(part)
        log.info("  %-20s %6d rows  (label counts: %s)",
                  fname, len(part), part["label"].value_counts().to_dict())

    if not frames:
        raise FileNotFoundError(f"No usable source CSVs found in {EMAIL_V2_CACHE}")

    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""]
    df["label"] = df["label"].astype(int)
    df = df[df["label"].isin([0, 1])]

    if dedupe:
        before = len(df)
        df = df.drop_duplicates(subset=["text"])
        log.info("Deduped %d → %d rows", before, len(df))

    # ── Cap outlier-length text ───────────────────────────────────────────
    # A handful of rows (mostly CEAS_08 digest-style newsletters or raw
    # MIME dumps that slipped through with embedded content) run up to
    # ~4.6M characters — three orders of magnitude past the ~99th
    # percentile (~2k chars). These few rows dominate TF-IDF fitting time
    # and memory (and, on this box, cause an OOM kill) for no real signal
    # gain — a phishing/spam classifier doesn't need the full text of a
    # multi-megabyte digest email, the signal is front-loaded anyway.
    # Truncate instead of dropping, so we keep the row (and its label)
    # but bound the cost.
    MAX_CHARS = 20_000
    over = df["text"].str.len() > MAX_CHARS
    if over.any():
        log.info("Truncating %d rows longer than %d chars", over.sum(), MAX_CHARS)
        df.loc[over, "text"] = df.loc[over, "text"].str.slice(0, MAX_CHARS)

    df = df.drop(columns=["source"]).reset_index(drop=True)

    log.info(
        "Email dataset v2: %d rows  phishing/spam=%d  legit=%d",
        len(df), df["label"].sum(), (df["label"] == 0).sum(),
    )
    return df


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


CRUX_PATH = CACHE_DIR / "crux_top_domains.csv"


def _add_top_domains(df: pd.DataFrame, n: int = 4_000) -> pd.DataFrame:
    """
    Blend real top-visited domains into the benign class, sampled across
    the popularity spectrum (not just top-1000 mega-brands) from Chrome's
    CrUX top-sites list. See the long comment at the call site for why
    this is needed — the base dataset's benign class is almost entirely
    deep content pages, so root-domain URLs are (backwards) treated as a
    malicious signal.

    n=4_000 is deliberately modest: root-domain URLs make up only ~1.8%
    of the malicious class here, so flooding benign with tens of
    thousands of root-domain examples would just flip the bias in the
    other direction (root domain ⇒ benign). ~4k brings benign's
    root-domain share to roughly 10% — enough to break the "bare domain
    looks suspicious" inversion without creating an equally artificial
    "bare domain looks safe" shortcut in its place.

    If the CrUX cache file isn't present, this is a no-op — training
    still works, just with the structural bias intact.
    """
    if not CRUX_PATH.exists():
        log.warning(
            "%s not found — skipping top-domain blend. Root-domain URLs "
            "will remain under-represented in the benign class.",
            CRUX_PATH,
        )
        return df

    crux = pd.read_csv(CRUX_PATH)
    if len(crux) > n:
        # Sample per rank bucket (not a flat .sample()) so the smaller
        # blend still spans the popularity spectrum rather than
        # over-representing whichever bucket happens to be biggest.
        buckets = crux["rank"].unique()
        per_bucket = max(1, n // len(buckets))
        parts = [
            crux[crux["rank"] == r].sample(
                min((crux["rank"] == r).sum(), per_bucket), random_state=42
            )
            for r in buckets
        ]
        crux = pd.concat(parts, ignore_index=True)
    top = pd.DataFrame({
        "url": crux["origin"].astype(str).str.strip(),
        "label": 0,
    })
    before = (df["label"] == 0).sum()
    df = pd.concat([df, top], ignore_index=True)
    df = df.drop_duplicates(subset=["url"])
    after = (df["label"] == 0).sum()
    log.info(
        "Blended in %d top-domain URLs as benign (benign: %d → %d)",
        len(top), before, after,
    )
    return df


SYNTHETIC_PATHS_PATH = CACHE_DIR / "synthetic_legit_paths.csv"


def _add_synthetic_legit_paths(df: pd.DataFrame) -> pd.DataFrame:
    """
    Blend in benign URLs with short/medium realistic paths on real,
    popular domains (e.g. "https://<real-top-domain>/login").

    Why this exists: even after _add_top_domains() fixes the pure
    root-domain bias, a second layer of the same problem remains — the
    base dataset's benign *paths* have a median length of ~79 chars
    (long descriptive slugs from news/forum content), while malicious
    paths have a median of ~44 chars. A short, completely ordinary path
    like "/login" or "/torvalds/linux" sits far outside what the model
    has ever seen labeled benign, so it gets pulled toward "malicious"
    on path length alone — the same shape-based shortcut as the
    root-domain issue, just one level deeper.

    IMPORTANT — what these rows are and aren't: the *domains* are real
    entries from Chrome's CrUX top-sites list. The *paths*
    ("/login", "/product/{id}/reviews", etc.) are synthetic —
    programmatically generated from common, generic path patterns, not
    scraped or verified to exist on that exact domain. We are not
    claiming "this exact URL is safe" — we're teaching the model that
    ordinary short/medium path *shapes* on legitimate domains aren't
    inherently suspicious, which is a structural gap the base dataset
    has no way to fill (it has almost no short-path benign examples at
    all). If SYNTHETIC_PATHS_PATH is missing, this is a no-op.
    """
    if not SYNTHETIC_PATHS_PATH.exists():
        log.warning(
            "%s not found — skipping synthetic-path blend. Short-path "
            "URLs will remain under-represented in the benign class.",
            SYNTHETIC_PATHS_PATH,
        )
        return df

    synth = pd.read_csv(SYNTHETIC_PATHS_PATH)
    top = pd.DataFrame({
        "url": synth["url"].astype(str).str.strip(),
        "label": 0,
    })
    before = (df["label"] == 0).sum()
    df = pd.concat([df, top], ignore_index=True)
    df = df.drop_duplicates(subset=["url"])
    after = (df["label"] == 0).sum()
    log.info(
        "Blended in %d synthetic-path URLs as benign (benign: %d → %d)",
        len(top), before, after,
    )
    return df


def load_url_dataset(sample: int | None = 90_000) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
      url    – raw URL string
      label  – 0 = benign, 1 = malicious (int)

    The full base dataset has ~650k rows, but ~463k of those have no
    scheme prefix and are dropped (see the scheme-filtering note above)
    because scheme presence is confounded with the label in this
    dataset. That leaves ~188k usable rows (35k benign / 152k
    malicious). We then blend in:
      - ~4k real top-visited domains (_add_top_domains) to fix a bias
        where root-domain URLs were ~95% malicious in the base data
      - ~6k synthetic-path URLs on real domains (_add_synthetic_legit_paths)
        to fix a second bias where benign paths were almost always much
        longer than malicious paths, so any short/ordinary path (like
        "/login") looked anomalous
    bringing benign to ~45k. Default sample=90_000 asks for a balanced
    45k/45k split, the most benign data we actually have — pass
    sample=None to load everything usable (unbalanced ~45k/143k, rely
    on class_weight="balanced").
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

    # ── Drop rows with no real scheme ────────────────────────────────────
    # This dataset stores URLs inconsistently: ~92% of *benign* rows have
    # no "http(s)://" prefix at all, while ~96-100% of malware/defacement
    # rows do. That's a data-collection artifact (different classes came
    # from different source lists that formatted URLs differently), not a
    # real phishing signal. If we guessed a default scheme for the
    # scheme-less rows we'd just be injecting a fabricated has_https value
    # for ~90% of the benign class — same problem, relocated.
    # Instead we keep only rows where the original source already
    # included a real scheme, so has_https and every downstream feature
    # (hostname parsing, TLD, subdomain count, etc.) reflects genuine
    # signal rather than a class-correlated formatting quirk. This still
    # leaves ~188k rows (35k benign / 152k malicious across the 3 bad
    # types), enough to train on without the artifact.
    df = df[df["url"].str.contains("://", regex=False)].copy()

    df["label"] = df["label"].astype(str).str.strip().str.lower().map(_URL_LABEL_MAP)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    # ── Blend in real top-domain URLs (structural bias fix) ──────────────
    # Beyond the scheme artifact above, this dataset has a second, deeper
    # bias: its "benign" class is almost entirely deep content pages
    # (articles, forum threads, product pages — avg ~115 chars), while
    # its "malicious" class skews toward bare hosts/IPs (avg ~80 chars).
    # Root-domain URLs with no path are 95% malicious in this dataset,
    # which is backwards from reality and teaches the model that a bare
    # domain like "https://google.com" looks suspicious. Filtering rows
    # doesn't fix this — it's baked into the sampling design of the whole
    # benign class. Instead we blend in real top-visited domains (Chrome
    # CrUX top-sites list, sampled across the full popularity spectrum,
    # not just mega-brands) labeled benign, so root-domain URLs stop
    # being a near-exclusive signal of "malicious" in training.
    df = _add_top_domains(df)
    df = _add_synthetic_legit_paths(df)

    if sample and len(df) > sample:
        # Balanced sample: cap each class at sample // 2, but don't let the
        # majority class overshoot just because the minority class is
        # smaller than requested (benign is now the limiting class after
        # the scheme filter above, at ~35k rows).
        # NOTE: avoid groupby(...).apply(...) here — pandas >= 2.2 changed
        # how the grouping column is (or isn't) passed into the apply
        # function ("include_groups"), and pandas 3.0 removes it from the
        # sub-frame entirely, which breaks any downstream code expecting
        # "label" to still be a column. Sampling per label value directly
        # sidesteps that version-dependent behaviour.
        per_class = sample // 2
        counts = df["label"].value_counts()
        cap = min(per_class, counts.min())
        parts = [
            df[df["label"] == lbl].sample(cap, random_state=42)
            for lbl in counts.index
        ]
        df = pd.concat(parts, ignore_index=True)

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
