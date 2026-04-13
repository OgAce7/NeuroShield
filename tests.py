"""
neuroshield/tests.py
─────────────────────
Offline test suite — no Kaggle download required.
Uses synthetic data for model tests so it runs in CI without credentials.

Run
───
  python tests.py
  # or with pytest:
  pytest tests.py -v
"""

from __future__ import annotations

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction tests
# ─────────────────────────────────────────────────────────────────────────────

class TestURLFeatures:

    def _f(self, url):
        from features import extract_url_features
        return extract_url_features(url)

    def test_https_detected(self):
        assert self._f("https://www.google.com").has_https == 1

    def test_http_flagged(self):
        assert self._f("http://evil.tk/login").has_https == 0

    def test_ip_detected(self):
        assert self._f("http://192.168.1.1/banking").has_ip == 1

    def test_no_ip_on_domain(self):
        assert self._f("https://www.amazon.com").has_ip == 0

    def test_risky_tld(self):
        assert self._f("http://site.tk").has_risky_tld == 1
        assert self._f("http://site.xyz").has_risky_tld == 1

    def test_trusted_tld(self):
        assert self._f("https://university.edu").is_trusted_tld == 1

    def test_brand_in_subdomain(self):
        assert self._f("http://paypal.evil.com/signin").has_brand_in_sub == 1

    def test_sus_words(self):
        f = self._f("http://x.com/verify/account/login")
        assert f.num_sus_words >= 2

    def test_hex_encoding(self):
        assert self._f("http://x.com/path?q=%61%62%63").has_hex == 1

    def test_redirect_param(self):
        assert self._f("http://x.com/go?redirect=http://evil.com").has_redirect == 1

    def test_feature_count(self):
        from features import URLFeatures
        f = self._f("https://example.com")
        assert len(f.to_list()) == len(URLFeatures().to_list())

    def test_heuristic_high_for_phishing(self):
        f = self._f("http://192.168.1.1/paypal/login.php?verify=1")
        assert f.heuristic_score > 0.35

    def test_heuristic_low_for_legit(self):
        f = self._f("https://www.wikipedia.org/wiki/Python")
        assert f.heuristic_score < 0.25

    def test_to_dict_keys(self):
        from features import URLFeatures
        f = self._f("https://example.com")
        assert set(f.to_dict().keys()) == set(URLFeatures().feature_names)


class TestEmailMetaFeatures:

    def _f(self, text):
        from features import extract_email_meta
        return extract_email_meta(text)

    def test_url_count(self):
        text = "Visit http://evil.tk and http://phish.ml for your prize"
        assert self._f(text).url_count == 2

    def test_html_detected(self):
        assert self._f("<html><body>Click <a href='x'>here</a></body></html>").has_html == 1

    def test_exclamation_count(self):
        assert self._f("Win now! Act fast! Limited offer!").exclamation_count == 3

    def test_caps_ratio(self):
        f = self._f("URGENT: YOUR ACCOUNT IS SUSPENDED")
        assert f.caps_ratio > 0.5

    def test_phish_words(self):
        f = self._f("Please verify your account and confirm your password immediately")
        assert f.num_phish_words >= 2

    def test_word_count(self):
        assert self._f("hello world foo bar").word_count == 4


# ─────────────────────────────────────────────────────────────────────────────
# Model tests (synthetic data — no Kaggle needed)
# ─────────────────────────────────────────────────────────────────────────────

def _make_url_df(n=400):
    """Generate a small balanced synthetic URL dataset."""
    rng = np.random.default_rng(0)
    phish = [
        f"http://paypal-{rng.integers(0,999)}.tk/verify?user={i}&token={''.join(rng.choice(list('abc123'), 8))}"
        for i in range(n // 2)
    ]
    legit = [
        f"https://www.{''.join(rng.choice(list('abcdefg'), 6))}.com/page/{i}"
        for i in range(n // 2)
    ]
    urls   = phish + legit
    labels = [1] * (n // 2) + [0] * (n // 2)
    return pd.DataFrame({"url": urls, "label": labels}).sample(frac=1, random_state=0).reset_index(drop=True)


def _make_email_df(n=400):
    """Generate a small balanced synthetic email dataset."""
    rng = np.random.default_rng(0)
    spam_templates = [
        "URGENT: Your account has been suspended. Verify immediately: http://paypa1.tk/verify",
        "Congratulations! You won a $1000 prize. Click here to claim: http://win.ml/claim",
        "Dear customer, confirm your password now or account will close: http://secure.xyz/update",
    ]
    ham_templates = [
        "Hi team, the meeting is scheduled for Friday. Please review the attached agenda.",
        "Your order has been shipped. Estimated delivery: 3-5 business days.",
        "Thanks for your contribution! The pull request has been approved and merged.",
    ]
    spam = [spam_templates[i % 3] + f" ID:{rng.integers(0,9999)}" for i in range(n // 2)]
    ham  = [ham_templates[i % 3]  + f" Ref:{rng.integers(0,9999)}" for i in range(n // 2)]
    texts  = spam + ham
    labels = [1] * (n // 2) + [0] * (n // 2)
    return pd.DataFrame({"text": texts, "label": labels}).sample(frac=1, random_state=0).reset_index(drop=True)


class TestURLModel:

    def setup_method(self):
        from url_model import train, predict_url
        self.arts     = train(df=_make_url_df(400), save=False, verbose=False)
        self.predict  = lambda u: predict_url(u, self.arts)

    def test_metrics_present(self):
        m = self.arts["metrics"]
        for k in ("accuracy", "f1", "roc_auc"):
            assert k in m, f"Missing metric: {k}"

    def test_accuracy_reasonable(self):
        assert self.arts["metrics"]["accuracy"] >= 0.75

    def test_phishing_url_flagged(self):
        r = self.predict("http://paypal-login.tk/verify?user=x&token=abc")
        assert r["label"] in ("malicious", "suspicious")
        assert r["risk_score"] >= 30

    def test_legit_url_clean(self):
        r = self.predict("https://www.github.com/python/cpython")
        assert r["label"] in ("benign", "suspicious")
        assert r["risk_score"] < 80

    def test_result_schema(self):
        r = self.predict("https://example.com")
        for key in ("url", "label", "risk_score", "confidence", "features", "indicators"):
            assert key in r, f"Missing key: {key}"

    def test_risk_score_range(self):
        for u in ["https://google.com", "http://evil.ml/phish", "http://192.0.0.1/x"]:
            r = self.predict(u)
            assert 0 <= r["risk_score"] <= 100

    def test_confidence_range(self):
        r = self.predict("https://stackoverflow.com")
        assert 0.0 <= r["confidence"] <= 1.0

    def test_indicators_list(self):
        r = self.predict("http://paypal.evil.xyz/signin?verify=1")
        assert isinstance(r["indicators"], list)


class TestEmailModel:

    def setup_method(self):
        from email_model import train, predict_email
        self.arts    = train(df=_make_email_df(400), save=False, verbose=False)
        self.predict = lambda t: predict_email(t, self.arts)

    def test_metrics_present(self):
        m = self.arts["metrics"]
        for k in ("accuracy", "f1", "roc_auc"):
            assert k in m

    def test_accuracy_reasonable(self):
        assert self.arts["metrics"]["accuracy"] >= 0.70

    def test_spam_detected(self):
        r = self.predict("URGENT: Your account has been suspended. Verify immediately at http://paypa1.tk/verify")
        assert r["label"] in ("spam", "suspicious")

    def test_ham_clean(self):
        r = self.predict("Hi, the meeting is at 3pm. Please bring your laptop.")
        assert r["label"] in ("ham", "suspicious")

    def test_result_schema(self):
        r = self.predict("Test email body")
        for k in ("label", "risk_score", "confidence", "indicators"):
            assert k in r

    def test_risk_score_range(self):
        r = self.predict("Normal email about a project update.")
        assert 0 <= r["risk_score"] <= 100


# ─────────────────────────────────────────────────────────────────────────────
# API tests (uses TestClient — no live server needed)
# ─────────────────────────────────────────────────────────────────────────────

class TestAPI:

    def setup_method(self):
        # Patch registry before importing the app
        import api
        from url_model import train as url_train
        from email_model import train as email_train

        arts_url   = url_train(df=_make_url_df(400),   save=False, verbose=False)
        arts_email = email_train(df=_make_email_df(400), save=False, verbose=False)

        api.registry.url_artifacts   = arts_url
        api.registry.email_artifacts = arts_email

        from fastapi.testclient import TestClient
        self.client = TestClient(api.app)

    def test_health(self):
        r = self.client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_url_predict_200(self):
        r = self.client.post("/api/url/predict", json={"url": "https://www.google.com"})
        assert r.status_code == 200
        data = r.json()
        assert "label" in data
        assert "risk_score" in data

    def test_url_predict_phishing(self):
        r = self.client.post("/api/url/predict", json={"url": "http://paypal.evil.tk/verify?u=me"})
        assert r.status_code == 200
        data = r.json()
        assert data["risk_score"] >= 30

    def test_url_batch(self):
        r = self.client.post("/api/url/batch", json={"urls": ["https://google.com", "http://evil.ml/phish"]})
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 2
        assert len(data["results"]) == 2

    def test_url_batch_too_large(self):
        r = self.client.post("/api/url/batch", json={"urls": ["http://x.com"] * 101})
        assert r.status_code == 422

    def test_email_predict_200(self):
        r = self.client.post("/api/email/predict", json={"text": "Hello, your account needs verification."})
        assert r.status_code == 200

    def test_email_parse(self):
        raw = (
            "From: evil@paypa1.tk\n"
            "Subject: Urgent: Verify now\n\n"
            "Your account is suspended. Visit http://paypa1.tk/verify to restore access."
        )
        r = self.client.post("/api/email/parse", json={"raw": raw})
        assert r.status_code == 200
        data = r.json()
        assert "parsed" in data
        assert "email_analysis" in data
        assert "combined_score" in data

    def test_model_status(self):
        r = self.client.get("/api/models/status")
        assert r.status_code == 200
        data = r.json()
        assert "url_model" in data
        assert "email_model" in data


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import traceback

    test_classes = [
        TestURLFeatures,
        TestEmailMetaFeatures,
        TestURLModel,
        TestEmailModel,
        TestAPI,
    ]

    total = passed = failed = 0

    for cls in test_classes:
        instance = cls()
        methods = [m for m in dir(cls) if m.startswith("test_")]
        group_pass = group_fail = 0

        print(f"\n{'─'*55}")
        print(f"  {cls.__name__}")
        print(f"{'─'*55}")

        for method in methods:
            total += 1
            if hasattr(instance, "setup_method"):
                try:
                    instance.setup_method()
                except Exception as e:
                    print(f"  [SETUP FAIL]  {method}")
                    traceback.print_exc()
                    failed += 1
                    group_fail += 1
                    continue
            try:
                getattr(instance, method)()
                print(f"  ✓  {method}")
                passed += 1
                group_pass += 1
            except Exception as e:
                print(f"  ✗  {method}")
                print(f"     {type(e).__name__}: {e}")
                failed += 1
                group_fail += 1

        print(f"\n  {group_pass}/{group_pass+group_fail} passed")

    print(f"\n{'═'*55}")
    print(f"  Total: {passed}/{total} passed  ({failed} failed)")
    print(f"{'═'*55}\n")

    sys.exit(0 if failed == 0 else 1)
