"""FastAPI demo backend for Certified Numeric Verification, adapted from the
original api/server.py for a stateless Vercel serverless deployment.

Differences from the original (docs/SDD.md-contracted) API, deliberate and
scoped to this public demo:
  - Reads/writes are confined to /tmp (Vercel's function filesystem is
    read-only elsewhere) and never persist across cold starts.
  - Calibrated thresholds are bundled read-only from calibration/ instead of
    a results/ directory produced by the offline eval pipeline.
  - The response exposes BOTH the raw majority-vote verdict and whether it
    clears the paper's calibrated threshold at the chosen alpha, instead of
    collapsing an uncertified claim into a bare "ABSTAIN" with no verdict at
    all. The original API hides this because its UiPath contract only cares
    about the certified answer/abstain decision; a public demo is more
    useful if it still shows what the model actually computed.
  - A best-effort per-IP rate limit and input-size cap guard the public
    endpoint against runaway LLM spend. The limiter resets on cold start,
    so treat it as a deterrent, not a hard guarantee -- set a billing cap on
    the Gemini API key itself for real protection.
"""
from __future__ import annotations

import io
import json
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from src.conformal import apply_threshold
from src.llm_client import LLMClient, load_config
from src.review_queue import append_review_row
from src.table_utils import load_generic_table
from src.verifier import verify_claim_df

INDEX_HTML_PATH = PROJECT_ROOT / "index.html"
CALIBRATION_PATH = PROJECT_ROOT / "calibration" / "conformal_thresholds.json"
MAX_CSV_CHARS = 20_000
MAX_ROWS = 200
MAX_COLS = 40
MAX_CLAIM_CHARS = 500
RATE_LIMIT_PER_MINUTE = 8

app = FastAPI(title="Certified Numeric Verification — Demo API", version="1.0-demo")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"],
)

CONFIG = load_config()
CLIENT = LLMClient(CONFIG)
DEFAULT_MODEL = CONFIG["models"]["default"]
DEFAULT_K = CONFIG["experiment"]["k_checks"]
DEFAULT_TEMP = CONFIG["models"]["available"][0]["temperature_checks"]
REVIEW_QUEUE_PATH = str(Path(CONFIG["paths"]["results_dir"]) / "review_queue.xlsx")

# Best-effort, per-instance only (resets on cold start / not shared across
# concurrent instances) -- a deterrent against casual abuse, not a hard cap.
_hits: dict[str, list[float]] = defaultdict(list)


def _rate_limited(client_ip: str) -> bool:
    now = time.time()
    window = _hits[client_ip] = [t for t in _hits[client_ip] if now - t < 60]
    if len(window) >= RATE_LIMIT_PER_MINUTE:
        return True
    window.append(now)
    return False


def _load_thresholds() -> dict:
    if CALIBRATION_PATH.exists():
        return json.loads(CALIBRATION_PATH.read_text())
    return {}


class VerifyRequest(BaseModel):
    claim: str
    table_csv: str
    claim_id: Optional[str] = None
    table_id: Optional[str] = None
    alpha: float = Field(default=0.10, description="Target risk level; must be one of the calibrated alphas.")


class VerifyResponse(BaseModel):
    claim_id: str
    verdict: str
    raw_verdict: str
    certified: bool
    agreement_score: float
    executed_checks: list
    computed_values: list
    reason: str
    latency_ms: float
    sent_to_review: bool


@app.get("/", response_class=HTMLResponse)
@app.get("/index.html", response_class=HTMLResponse)
async def root():
    # Vercel's static-asset auto-serving didn't pick up the root index.html
    # alongside a Python function in this project layout, so the app serves
    # its own frontend directly rather than depending on that detection.
    return INDEX_HTML_PATH.read_text()


@app.get("/api/health")
@app.get("/health")
async def health():
    return {"status": "ok", "llm_available": CLIENT.available(), "model": DEFAULT_MODEL}


@app.post("/api/verify", response_model=VerifyResponse)
@app.post("/verify", response_model=VerifyResponse)
async def verify(req: VerifyRequest, request: Request):
    start = time.time()
    client_ip = request.client.host if request.client else "unknown"
    if _rate_limited(client_ip):
        raise HTTPException(status_code=429, detail="rate limit exceeded: max 8 verify calls/minute per IP on this demo")

    claim_id = req.claim_id or str(uuid.uuid4())
    table_id = req.table_id or "inline_table"

    if not req.claim or not req.claim.strip():
        raise HTTPException(status_code=400, detail="`claim` is missing or empty")
    if len(req.claim) > MAX_CLAIM_CHARS:
        raise HTTPException(status_code=400, detail=f"`claim` too long (max {MAX_CLAIM_CHARS} chars)")
    if not req.table_csv or not req.table_csv.strip():
        raise HTTPException(status_code=400, detail="`table_csv` is required")
    if len(req.table_csv) > MAX_CSV_CHARS:
        raise HTTPException(status_code=400, detail=f"`table_csv` too large (max {MAX_CSV_CHARS} chars) -- this is a demo, not a bulk-upload tool")

    try:
        df = load_generic_table(io.StringIO(req.table_csv))
        if df.shape[1] == 0 or df.shape[0] == 0:
            raise HTTPException(status_code=422, detail="table parsed to zero rows/columns")
        if df.shape[0] > MAX_ROWS or df.shape[1] > MAX_COLS:
            raise HTTPException(status_code=422, detail=f"table too large for the demo (max {MAX_ROWS} rows x {MAX_COLS} cols)")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"could not parse table: {e}")

    if not CLIENT.available():
        raise HTTPException(status_code=503, detail="LLM unavailable: GEMINI_API_KEY is not set on this deployment")

    try:
        result = await verify_claim_df(CLIENT, DEFAULT_MODEL, req.claim, df, k=DEFAULT_K,
                                        temperature=DEFAULT_TEMP, method="ours")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"LLM unavailable: {e}")

    agreement = result.get("agreement_score", 0.0)
    raw_verdict = result["verdict"]
    thresholds = _load_thresholds().get("ours_agreement", {})
    lam = thresholds.get(str(req.alpha))

    if lam is None:
        lam = 1.0
        reason_suffix = " (no calibrated threshold for this alpha: conservative fallback requires full agreement)"
    else:
        reason_suffix = f" (alpha={req.alpha}, calibrated lambda={lam:.3f})"

    certified = bool(apply_threshold([agreement], lam)[0]) if raw_verdict != "UNVERIFIABLE" else False
    sent_to_review = not certified

    computed_values = [c["result"].get("computed_value") for c in result.get("executed_checks", [])
                        if c.get("result", {}).get("ok")]

    if certified:
        reason = "certified: agreement >= calibrated threshold" + reason_suffix
    else:
        reason = ("below calibrated confidence at this alpha" + reason_suffix
                   if raw_verdict != "UNVERIFIABLE" else "no valid executed check (UNVERIFIABLE)")
        try:
            append_review_row(REVIEW_QUEUE_PATH, claim_id, req.claim, table_id, raw_verdict,
                               agreement, result.get("executed_checks", []), reason)
        except Exception:
            pass  # /tmp review queue is best-effort in this demo; never fail the request over it

    latency_ms = (time.time() - start) * 1000
    return VerifyResponse(
        claim_id=claim_id,
        verdict=raw_verdict if certified else "ABSTAIN",
        raw_verdict=raw_verdict,
        certified=certified,
        agreement_score=agreement,
        executed_checks=result.get("executed_checks", []),
        computed_values=computed_values,
        reason=reason,
        latency_ms=latency_ms,
        sent_to_review=sent_to_review,
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc):
    return JSONResponse(status_code=500, content={"detail": f"internal error: {exc}"})


from starlette.exceptions import HTTPException as StarletteHTTPException


@app.exception_handler(StarletteHTTPException)
async def debug_404_handler(request, exc):
    # TEMPORARY diagnostic: a bare {"detail": "Not Found"} on Vercel with no
    # other information doesn't say whether the rewrite is forwarding the
    # real request path to this app. Echo it back so we can see what this
    # ASGI app actually received, then remove this once routing is confirmed.
    return JSONResponse(status_code=exc.status_code, content={
        "detail": exc.detail,
        "path_seen_by_app": str(request.url.path),
        "full_url_seen_by_app": str(request.url),
        "registered_routes": [getattr(r, "path", str(r)) for r in app.routes],
    })
