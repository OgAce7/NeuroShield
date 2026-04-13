"""
neuroshield/features.py
────────────────────────
Feature extraction for URLs and email text.

URL features  → 35 hand-crafted lexical / structural signals
Email features → TF-IDF vectorisation + 8 meta-features (length, caps ratio, etc.)
"""

from __future__ import annotations

import math
import re
import string
import urllib.parse
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# tldextract gives us reliable TLD parsing without hitting DNS
try:
    import tldextract
    _HAS_TLDEXTRACT = True
except ImportError:
    _HAS_TLDEXTRACT = False


# ── URL constants ─────────────────────────────────────────────────────────

SUSPICIOUS_WORDS = [
    "login", "signin", "verify", "update", "secure", "account", "banking",
    "confirm", "wallet", "password", "credential", "support", "helpdesk",
    "paypal", "amazon", "google", "apple", "microsoft", "ebay", "netflix",
    "chase", "hsbc", "wellsfargo", "dhl", "fedex", "ups", "instagram",
    "facebook", "twitter", "linkedin", "dropbox", "icloud",
    "free", "win", "prize", "click", "urgent", "alert", "suspended",
    "limited", "unusual", "activity", "validate",
]

RISKY_TLDS = {
    "tk", "ml", "ga", "cf", "gq", "xyz", "top", "club", "info",
    "online", "site", "website", "space", "buzz", "icu", "cyou",
    "rest", "vip", "win", "loan", "download",
}

BRAND_NAMES = [
    "paypal", "amazon", "google", "apple", "microsoft", "facebook",
    "instagram", "twitter", "netflix", "ebay", "chase", "citibank",
    "wellsfargo", "bankofamerica", "hsbc", "dhl", "fedex", "ups",
    "linkedin", "dropbox", "icloud", "outlook", "yahoo",
]

TRUSTED_TLDS = {"com", "org", "edu", "gov", "net", "io", "co"}


# ── URL feature dataclass ─────────────────────────────────────────────────

@dataclass
class URLFeatures:
    # Lengths
    url_length:       int   = 0
    hostname_length:  int   = 0
    path_length:      int   = 0
    query_length:     int   = 0

    # Character counts
    num_dots:         int   = 0
    num_hyphens:      int   = 0
    num_underscores:  int   = 0
    num_slashes:      int   = 0
    num_question:     int   = 0
    num_equals:       int   = 0
    num_at:           int   = 0
    num_ampersand:    int   = 0
    num_digits:       int   = 0
    num_params:       int   = 0   # query param count

    # Security
    has_https:        int   = 0
    has_ip:           int   = 0
    has_port:         int   = 0
    has_double_slash: int   = 0

    # Suspicion
    num_subdomains:   int   = 0
    has_risky_tld:    int   = 0
    is_trusted_tld:   int   = 0
    has_brand_in_sub: int   = 0
    brand_not_apex:   int   = 0   # brand in subdomain but apex ≠ brand.com
    num_sus_words:    int   = 0
    has_sus_word:     int   = 0

    # Encoding / obfuscation
    has_hex:          int   = 0
    has_data_uri:     int   = 0
    has_redirect:     int   = 0
    has_shortener:    int   = 0   # bit.ly, tinyurl, etc.

    # Entropy / ratios
    url_entropy:      float = 0.0
    host_entropy:     float = 0.0
    digit_ratio_url:  float = 0.0
    digit_ratio_host: float = 0.0
    special_ratio:    float = 0.0

    # Composite heuristic
    heuristic_score:  float = 0.0

    def to_list(self) -> List[float]:
        return [float(v) for v in self.__dict__.values()]

    def to_dict(self) -> Dict[str, float]:
        return {k: float(v) for k, v in self.__dict__.items()}

    @property
    def feature_names(self) -> List[str]:
        return list(self.__dict__.keys())


_SHORTENERS = {
    "bit.ly", "tinyurl.com", "goo.gl", "ow.ly", "t.co", "buff.ly",
    "adf.ly", "short.link", "cutt.ly", "rb.gy", "lnkd.in",
}

_IP_RE = re.compile(
    r"^(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)$"
)


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    freq: Dict[str, int] = {}
    for ch in s:
        freq[ch] = freq.get(ch, 0) + 1
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in freq.values())


def extract_url_features(url: str) -> URLFeatures:
    f = URLFeatures()

    # ── Parse ────────────────────────────────────────────────────────────
    raw = url.strip()
    if "://" not in raw:
        raw = "http://" + raw
    try:
        p = urllib.parse.urlparse(raw)
    except Exception:
        return f

    scheme   = p.scheme.lower()
    hostname = (p.hostname or "").lower()
    path     = p.path or ""
    query    = p.query or ""
    fragment = p.fragment or ""

    # ── Lengths ──────────────────────────────────────────────────────────
    f.url_length      = len(url)
    f.hostname_length = len(hostname)
    f.path_length     = len(path)
    f.query_length    = len(query)

    # ── Character counts ─────────────────────────────────────────────────
    f.num_dots         = url.count(".")
    f.num_hyphens      = url.count("-")
    f.num_underscores  = url.count("_")
    f.num_slashes      = url.count("/")
    f.num_question     = url.count("?")
    f.num_equals       = url.count("=")
    f.num_at           = url.count("@")
    f.num_ampersand    = url.count("&")
    f.num_digits       = sum(c.isdigit() for c in url)
    f.num_params       = len(urllib.parse.parse_qs(query))

# ── Security signals ─────────────────────────────────────────────────
    f.has_https        = int(scheme == "https")
    f.has_ip           = int(bool(_IP_RE.match(hostname)))
    
    try:
        f.has_port     = int(p.port is not None)
    except ValueError:
        # If the port is garbage characters, treat it as no valid port
        f.has_port     = 0
        
    f.has_double_slash = int("//" in path)

    # ── TLD & domain ─────────────────────────────────────────────────────
    if _HAS_TLDEXTRACT:
        ext = tldextract.extract(raw)
        tld = ext.suffix.lower()
        apex = f"{ext.domain}.{ext.suffix}".lower()
        subdomain = ext.subdomain.lower()
        num_sub = len([s for s in subdomain.split(".") if s]) if subdomain else 0
    else:
        parts = hostname.split(".")
        tld   = parts[-1] if parts else ""
        apex  = ".".join(parts[-2:]) if len(parts) >= 2 else hostname
        subdomain = ".".join(parts[:-2]) if len(parts) > 2 else ""
        num_sub = len(parts) - 2 if len(parts) > 2 else 0

    f.num_subdomains  = max(0, num_sub)
    f.has_risky_tld   = int(tld in RISKY_TLDS)
    f.is_trusted_tld  = int(tld in TRUSTED_TLDS)

    # Brand-in-subdomain (typosquatting)
    brand_in_sub = any(b in subdomain for b in BRAND_NAMES)
    f.has_brand_in_sub = int(brand_in_sub)
    if brand_in_sub:
        matched = next(b for b in BRAND_NAMES if b in subdomain)
        f.brand_not_apex = int(apex != f"{matched}.com" and apex != f"{matched}.net")

    # ── Suspicious words ─────────────────────────────────────────────────
    url_lower = url.lower()
    f.num_sus_words = sum(1 for w in SUSPICIOUS_WORDS if w in url_lower)
    f.has_sus_word  = int(f.num_sus_words > 0)

    # ── Encoding / obfuscation ────────────────────────────────────────────
    f.has_hex      = int(bool(re.search(r"%[0-9a-fA-F]{2}", url)))
    f.has_data_uri = int(url.lower().startswith("data:"))
    redirect_kw    = ["redirect", "url=", "return=", "goto=", "next=", "dest="]
    f.has_redirect = int(any(kw in url_lower for kw in redirect_kw))
    f.has_shortener = int(hostname in _SHORTENERS)

    # ── Entropy & ratios ─────────────────────────────────────────────────
    f.url_entropy      = round(_entropy(url), 4)
    f.host_entropy     = round(_entropy(hostname), 4)
    f.digit_ratio_url  = round(f.num_digits / max(len(url), 1), 4)
    f.digit_ratio_host = round(
        sum(c.isdigit() for c in hostname) / max(len(hostname), 1), 4
    )
    special = sum(c in string.punctuation for c in url)
    f.special_ratio = round(special / max(len(url), 1), 4)

    # ── Composite heuristic score (0–1) ──────────────────────────────────
    score = 0.0
    score += 0.20 * (1 - f.has_https)
    score += 0.15 * f.has_ip
    score += 0.12 * min(f.num_sus_words / 4, 1.0)
    score += 0.10 * f.has_risky_tld
    score += 0.10 * (f.brand_not_apex or f.has_brand_in_sub)
    score += 0.08 * min(f.num_subdomains / 4, 1.0)
    score += 0.07 * min(f.url_length / 200, 1.0)
    score += 0.06 * f.has_hex
    score += 0.06 * f.has_redirect
    score += 0.03 * f.has_port
    score += 0.03 * f.has_shortener
    f.heuristic_score = round(min(score, 1.0), 4)

    return f


# ── Email feature extraction ───────────────────────────────────────────────

# Phishing-specific word lists
_PHISH_WORDS = [
    "verify", "account", "suspended", "limited", "click here", "urgent",
    "immediately", "banking", "credentials", "password", "login", "confirm",
    "update your", "unusual activity", "security alert", "validate",
    "dear customer", "dear user", "prize", "won", "congratulations",
    "free gift", "act now", "expires", "24 hours", "48 hours",
]

_URL_RE   = re.compile(r"https?://\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[a-z]{2,}", re.IGNORECASE)


@dataclass
class EmailMetaFeatures:
    char_count:       int   = 0
    word_count:       int   = 0
    url_count:        int   = 0
    has_html:         int   = 0
    caps_ratio:       float = 0.0
    digit_ratio:      float = 0.0
    special_ratio:    float = 0.0
    num_phish_words:  int   = 0
    avg_word_length:  float = 0.0
    line_count:       int   = 0
    exclamation_count:int   = 0
    sender_mismatch:  int   = 0   # set externally if header info available

    def to_list(self) -> List[float]:
        return [float(v) for v in self.__dict__.values()]

    def to_dict(self) -> Dict[str, float]:
        return {k: float(v) for k, v in self.__dict__.items()}

    @property
    def feature_names(self) -> List[str]:
        return list(self.__dict__.keys())


def extract_email_meta(text: str) -> EmailMetaFeatures:
    f = EmailMetaFeatures()
    if not text:
        return f

    f.char_count  = len(text)
    words         = text.split()
    f.word_count  = len(words)
    f.url_count   = len(_URL_RE.findall(text))
    f.has_html    = int(bool(re.search(r"<[a-z][\s\S]*?>", text, re.IGNORECASE)))
    f.line_count  = text.count("\n") + 1
    f.exclamation_count = text.count("!")

    upper  = sum(c.isupper() for c in text)
    digits = sum(c.isdigit() for c in text)
    special= sum(c in string.punctuation for c in text)
    n      = max(len(text), 1)

    f.caps_ratio    = round(upper   / n, 4)
    f.digit_ratio   = round(digits  / n, 4)
    f.special_ratio = round(special / n, 4)

    text_lower = text.lower()
    f.num_phish_words = sum(1 for pw in _PHISH_WORDS if pw in text_lower)

    if words:
        f.avg_word_length = round(sum(len(w) for w in words) / len(words), 4)

    return f


# ── Batch helpers ─────────────────────────────────────────────────────────

def url_df_to_features(df, url_col: str = "url"):
    """Vectorise a URL DataFrame into a numpy feature matrix."""
    import numpy as np
    rows = [extract_url_features(u).to_list() for u in df[url_col]]
    names = URLFeatures().feature_names
    return np.array(rows, dtype=float), names


def email_df_to_meta(df, text_col: str = "text"):
    """Extract meta-features from an email DataFrame."""
    import numpy as np
    rows = [extract_email_meta(t).to_list() for t in df[text_col]]
    names = EmailMetaFeatures().feature_names
    return np.array(rows, dtype=float), names
