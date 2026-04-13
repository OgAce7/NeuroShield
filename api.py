"""
neuroshield/api.py
───────────────────
FastAPI backend serving the URL and email phishing detection models.

Run
───
  uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Endpoints
─────────
  POST  /api/url/predict         Analyse a single URL
  POST  /api/url/batch           Analyse up to 100 URLs
  POST  /api/email/predict       Analyse email text
  POST  /api/email/parse         Parse raw forwarded email + analyse
  GET   /api/models/status       Health + model metadata
  GET   /api/models/url/meta     URL model metrics + feature importances
  GET   /api/models/email/meta   Email model metrics
  GET   /health                  Liveness probe
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator, model_validator

log = logging.getLogger("neuroshield.api")
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

MODELS_DIR = Path(__file__).parent / "models"


# ── Model registry ─────────────────────────────────────────────────────────

class ModelRegistry:
    url_artifacts:   Optional[Dict] = None
    email_artifacts: Optional[Dict] = None
    url_meta:        Dict = {}
    email_meta:      Dict = {}

registry = ModelRegistry()


def _load_url_model() -> None:
    try:
        from url_model import load as url_load
        registry.url_artifacts = url_load()
        meta_path = MODELS_DIR / "url_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                registry.url_meta = json.load(f)
        log.info("URL model loaded ✓")
    except FileNotFoundError:
        log.warning("URL model not found — run: python url_model.py")


def _load_email_model() -> None:
    try:
        from email_model import load as email_load
        registry.email_artifacts = email_load()
        meta_path = MODELS_DIR / "email_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                registry.email_meta = json.load(f)
        log.info("Email model loaded ✓")
    except FileNotFoundError:
        log.warning("Email model not found — run: python email_model.py")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_url_model()
    _load_email_model()
    yield


# ── App ────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="NeuroShield API",
    description="Real-time phishing detection for URLs and emails",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Middleware: request timing ─────────────────────────────────────────────

@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    t0 = time.perf_counter()
    resp = await call_next(request)
    ms = round((time.perf_counter() - t0) * 1000, 2)
    resp.headers["X-Response-Time-Ms"] = str(ms)
    log.info("%s %s  %d  %.1fms", request.method, request.url.path, resp.status_code, ms)
    return resp


# ── Request / response schemas ─────────────────────────────────────────────

class URLRequest(BaseModel):
    url: str

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 4:
            raise ValueError("URL too short")
        if len(v) > 2048:
            raise ValueError("URL exceeds 2048 characters")
        return v


class URLBatchRequest(BaseModel):
    urls: List[str]

    @model_validator(mode="after")
    def check_batch_size(self):
        if len(self.urls) > 100:
            raise ValueError("Batch size cannot exceed 100 URLs")
        return self


class EmailRequest(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 10:
            raise ValueError("Email text too short")
        if len(v) > 500_000:
            raise ValueError("Email text too long (max 500k chars)")
        return v


class ParseEmailRequest(BaseModel):
    """Accepts a raw forwarded email (with or without headers)."""
    raw: str

    @field_validator("raw")
    @classmethod
    def validate_raw(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 10:
            raise ValueError("Raw email too short")
        return v


# ── URL endpoints ──────────────────────────────────────────────────────────

@app.post("/api/url/predict", tags=["URL"])
async def predict_url(req: URLRequest):
    if registry.url_artifacts is None:
        raise HTTPException(status_code=503, detail="URL model not loaded")
    from url_model import predict_url as _predict
    t0 = time.perf_counter()
    result = _predict(req.url, registry.url_artifacts)
    result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    return result


@app.post("/api/url/batch", tags=["URL"])
async def predict_url_batch(req: URLBatchRequest):
    if registry.url_artifacts is None:
        raise HTTPException(status_code=503, detail="URL model not loaded")
    from url_model import predict_url as _predict
    t0 = time.perf_counter()
    results = []
    for url in req.urls:
        try:
            r = _predict(url.strip(), registry.url_artifacts)
        except Exception as exc:
            r = {"url": url, "label": "error", "risk_score": -1, "error": str(exc)}
        results.append(r)
    return {
        "results": results,
        "total": len(results),
        "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
    }


# ── Email endpoints ────────────────────────────────────────────────────────

def _parse_raw_email(raw: str) -> Dict[str, Any]:
    """
    Parse a raw forwarded email into structured fields.
    Handles emails with or without RFC headers.
    """
    import re

    lines = raw.split("\n")
    headers: Dict[str, str] = {}
    body_lines: List[str] = []
    in_body = False

    for line in lines:
        if in_body:
            body_lines.append(line)
        elif line.strip() == "":
            in_body = True
        else:
            # Try header pattern
            m = re.match(r"^([\w-]+)\s*:\s*(.+)$", line, re.IGNORECASE)
            if m:
                headers[m.group(1).lower()] = m.group(2).strip()
            else:
                # No headers found — treat everything as body
                in_body = True
                body_lines.append(line)

    # If no body was separated, use the full text
    body = "\n".join(body_lines).strip() or raw

    # Extract URLs from entire raw text
    urls = list({m.group(0) for m in re.finditer(r"https?://\S+", raw, re.IGNORECASE)})

    return {
        "from_":    headers.get("from", ""),
        "to":       headers.get("to", ""),
        "subject":  headers.get("subject", ""),
        "date":     headers.get("date", ""),
        "reply_to": headers.get("reply-to", ""),
        "body":     body,
        "urls":     urls,
        "has_headers": bool(headers),
    }


@app.post("/api/email/predict", tags=["Email"])
async def predict_email(req: EmailRequest):
    if registry.email_artifacts is None:
        raise HTTPException(status_code=503, detail="Email model not loaded")
    from email_model import predict_email as _predict
    t0 = time.perf_counter()
    result = _predict(req.text, registry.email_artifacts)
    result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    return result


@app.post("/api/email/parse", tags=["Email"])
async def parse_and_predict_email(req: ParseEmailRequest):
    """
    Parse a full forwarded email (headers + body), then run both
    the email classifier and URL analyser on every link found.
    """
    if registry.email_artifacts is None:
        raise HTTPException(status_code=503, detail="Email model not loaded")

    from email_model import predict_email as _predict_email
    t0 = time.perf_counter()

    parsed  = _parse_raw_email(req.raw)
    email_r = _predict_email(parsed["body"] or req.raw, registry.email_artifacts)

    # Analyse each URL found in the email
    url_results: List[Dict] = []
    if registry.url_artifacts is not None and parsed["urls"]:
        from url_model import predict_url as _predict_url
        for u in parsed["urls"][:30]:  # cap at 30
            try:
                url_results.append(_predict_url(u, registry.url_artifacts))
            except Exception:
                pass

    # Aggregate risk (email score + worst URL score)
    worst_url_score = max((r["risk_score"] for r in url_results), default=0)
    combined_score  = round(max(email_r["risk_score"], worst_url_score * 0.6), 1)

    return {
        "parsed": {
            "from":    parsed["from_"],
            "to":      parsed["to"],
            "subject": parsed["subject"],
            "date":    parsed["date"],
            "body_preview": parsed["body"][:400] + ("…" if len(parsed["body"]) > 400 else ""),
        },
        "email_analysis":  email_r,
        "url_analysis":    url_results,
        "combined_score":  combined_score,
        "combined_label":  (
            "phishing" if combined_score >= 65 else
            "suspicious" if combined_score >= 35 else
            "clean"
        ),
        "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
    }


# ── Model metadata ─────────────────────────────────────────────────────────

@app.get("/api/models/status", tags=["Models"])
async def model_status():
    return {
        "url_model":   {"loaded": registry.url_artifacts is not None,   "meta": registry.url_meta},
        "email_model": {"loaded": registry.email_artifacts is not None, "meta": registry.email_meta},
    }


@app.get("/api/models/url/meta", tags=["Models"])
async def url_model_meta():
    if not registry.url_meta:
        raise HTTPException(status_code=404, detail="URL model metadata not available")
    return registry.url_meta


@app.get("/api/models/email/meta", tags=["Models"])
async def email_model_meta():
    if not registry.email_meta:
        raise HTTPException(status_code=404, detail="Email model metadata not available")
    return registry.email_meta


# ── Health ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health():
    return {
        "status": "ok",
        "models": {
            "url":   registry.url_artifacts is not None,
            "email": registry.email_artifacts is not None,
        },
    }


# ── Global error handler ───────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_error(request: Request, exc: Exception):
    log.error("Unhandled error on %s: %s", request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Check server logs."},
    )


# ── Dev server ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
