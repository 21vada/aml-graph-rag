# app/main.py
import asyncio
import json
import os
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from neo4j import GraphDatabase
from pydantic import BaseModel

from app.features import build_features, THRESHOLD, HEADLINE
from app.retriever import AMLEvidenceRetriever


# ============================================================
# Schemas
# ============================================================
class Transaction(BaseModel):
    source: str
    target: str
    amount_paid: float
    timestamp: str


class ScoreResponse(BaseModel):
    risk_score: float
    flagged: bool
    threshold: float
    model: str


class ExplainResponse(BaseModel):
    explanation: str
    evidence_context: str
    model_used: str


class AnalyzeResponse(BaseModel):
    score: ScoreResponse
    explanation: Optional[str] = None
    evidence_context: Optional[str] = None
    explain_model: Optional[str] = None


# ============================================================
# Lifespan — load model, connect Neo4j (with retry), init Gemini
# ============================================================
_models = {}

def _model_path() -> str:
    """Resolve the XGBoost artifact to serve.

    Priority:
      1. $MODEL_PATH if set
      2. models/xgb_tabular.json (the real trained model; not committed)
      3. models/demo_xgb.json   (local demo model from scripts/make_demo_model.py)
    """
    env_path = os.getenv("MODEL_PATH", "").strip()
    if env_path:
        return env_path
    base = os.path.join(os.path.dirname(__file__), "..", "models")
    for name in ("xgb_tabular.json", "demo_xgb.json"):
        p = os.path.abspath(os.path.join(base, name))
        if os.path.exists(p):
            return p
    # Nothing found — return the canonical path so the error message is clear.
    return os.path.abspath(os.path.join(base, "xgb_tabular.json"))

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---- 1. XGBoost ----
    model_path = _model_path()
    if not os.path.exists(model_path):
        raise RuntimeError(
            f"Model not found at {model_path}. Generate the local demo model with:\n"
            f"    python scripts/make_demo_model.py\n"
            f"or place the trained artifact (from Colab) at models/xgb_tabular.json."
        )
    clf = xgb.XGBClassifier()
    clf.load_model(model_path)
    _models["xgb"] = clf
    print(f"✅ Loaded XGBoost model from {os.path.basename(model_path)} ({HEADLINE})")

    # ---- 2. Neo4j (retry up to 60s, non-blocking) ----
    uri = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
    password = os.getenv("NEO4J_PASSWORD", "")
    if not password:
        print("⚠️  NEO4J_PASSWORD not set. Set it in .env (see .env.example).")
    driver = GraphDatabase.driver(
        uri,
        auth=(
            os.getenv("NEO4J_USER", "neo4j"),
            password,
        ),
        keep_alive=True,
        connection_acquisition_timeout=5,
    )

    connected = False
    last_err = None
    for attempt in range(1, 31):          # 30 × 2s = 60s max
        try:
            await asyncio.to_thread(driver.verify_connectivity)
            connected = True
            print(f"✅ Connected to Neo4j ({uri}) after ~{attempt * 2}s")
            break
        except Exception as e:
            last_err = e
            if attempt == 1 or attempt % 5 == 0:
                print(f"  Neo4j not ready (attempt {attempt}/30): {type(e).__name__}")
            await asyncio.sleep(2)

    if connected:
        _models["driver"] = driver
        _models["retriever"] = AMLEvidenceRetriever(driver)
    else:
        print(f"⚠️  Neo4j unreachable after 60s ({last_err}). /explain will 503.")
        _models["driver"] = None
        _models["retriever"] = None

    # ---- 3. Gemini ----
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    if gemini_key:
        try:
            from google import genai
            _models["gemini"] = genai.Client(api_key=gemini_key)
            print("✅ Gemini client ready")
        except Exception as e:
            print(f"⚠️  Gemini client init failed: {e}")
            _models["gemini"] = None
    else:
        print("⚠️  GEMINI_API_KEY not set — /explain will 503")
        _models["gemini"] = None

    print("INFO:     Application startup complete.")

    yield

    if _models.get("driver"):
        _models["driver"].close()
    _models.clear()


# ============================================================
# App
# ============================================================
app = FastAPI(title="AML Graph RAG API", lifespan=lifespan)


# ============================================================
# Meta endpoints
# ============================================================
@app.get("/")
def root():
    return {
        "service": "AML Graph RAG API",
        "docs": "/docs",
        "endpoints": ["/health", "/model-info", "/score", "/explain", "/analyze"],
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": "xgb" in _models,
        "neo4j_connected": _models.get("driver") is not None,
        "gemini_ready": _models.get("gemini") is not None,
    }


@app.get("/model-info")
def model_info():
    if "xgb" not in _models:
        raise HTTPException(503, "model not loaded")
    return {
        "headline": HEADLINE,
        "threshold": THRESHOLD,
        "n_features": int(_models["xgb"].n_features_in_),
    }


# ============================================================
# Internal helpers
# ============================================================
def _score_one(tx: Transaction) -> ScoreResponse:
    if "xgb" not in _models:
        raise HTTPException(503, "model not loaded")
    X = build_features(tx.model_dump())
    raw_logit = float(_models["xgb"].predict(X, output_margin=True)[0])
    prob = float(1.0 / (1.0 + np.exp(-raw_logit)))
    # Scale consistency: model_meta.json's threshold is fitted on the
    # validation-set PROBABILITY scale (see notebooks/graphrag_ex_aml.ipynb).
    # If the shipped threshold is outside [0, 1] it was exported from raw
    # margins; convert it to a probability so `flagged` and `risk_score`
    # are always reported on the same scale.
    thr_prob = THRESHOLD if 0.0 <= THRESHOLD <= 1.0 else float(
        1.0 / (1.0 + np.exp(-THRESHOLD))
    )
    return ScoreResponse(
        risk_score=prob,
        flagged=prob >= thr_prob,
        threshold=thr_prob,
        model=HEADLINE,
    )


GEMINI_PRIMARY = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
GEMINI_FALLBACKS = ["gemini-2.5-flash", "gemini-2.0-flash"]

AML_PROMPT = """You are a senior AML compliance analyst reviewing a transaction
flagged by our model.

EVIDENCE:
{context}

Write a 3-4 sentence explanation that:
1. States the suspicious flow (who sent what to whom, in what format).
2. Cites specific network context from the evidence.
3. Ends with a verdict on its own line: [HIGH RISK: CONFIRMED] or [HIGH RISK: INCONCLUSIVE].

Rules:
- Do NOT invent facts.
- Do not mention XGBoost, GNN, or "evidence".
- Speak like a human analyst."""


def _explain_one(tx: Transaction) -> ExplainResponse:
    if _models.get("retriever") is None:
        raise HTTPException(503, "Neo4j not connected — cannot retrieve evidence")
    if _models.get("gemini") is None:
        raise HTTPException(503, "GEMINI_API_KEY not configured")

    query = json.dumps({
        "source_id":   tx.source,
        "target_id":   tx.target,
        "timestamp":   tx.timestamp,
        "amount_paid": tx.amount_paid,
    })
    result = _models["retriever"].search(query_text=query)
    if not result["items"]:
        raise HTTPException(
            404,
            f"Transaction {tx.source} -> {tx.target} @ {tx.timestamp} "
            f"not found in graph.",
        )
    context = result["items"][0]["content"]
    prompt = AML_PROMPT.format(context=context)

    last_err = None
    for model_id in [GEMINI_PRIMARY] + GEMINI_FALLBACKS:
        try:
            resp = _models["gemini"].models.generate_content(
                model=model_id,
                contents=prompt,
                config={"temperature": 0.0, "max_output_tokens": 400},
            )
            return ExplainResponse(
                explanation=resp.text,
                evidence_context=context,
                model_used=model_id,
            )
        except Exception as e:
            last_err = e
            continue

    raise HTTPException(502, f"All Gemini models failed. Last error: {last_err}")


# ============================================================
# Public endpoints
# ============================================================
@app.post("/score", response_model=ScoreResponse)
def score(tx: Transaction):
    return _score_one(tx)


@app.post("/explain", response_model=ExplainResponse)
def explain(tx: Transaction):
    return _explain_one(tx)


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(tx: Transaction, explain_if_flagged: bool = True):
    """Score first; explain only if flagged (unless explicitly requested)."""
    s = _score_one(tx)
    # Explain when the transaction is flagged, or when the caller opted out
    # of the flag-only behaviour (explain_if_flagged=False → always explain).
    if s.flagged or not explain_if_flagged:
        try:
            e = _explain_one(tx)
            return AnalyzeResponse(
                score=s,
                explanation=e.explanation,
                evidence_context=e.evidence_context,
                explain_model=e.model_used,
            )
        except HTTPException as eh:
            # Score is valid; explanation failed — return score with a note
            return AnalyzeResponse(
                score=s,
                explanation=f"[explain unavailable: {eh.detail}]",
            )
    return AnalyzeResponse(score=s)