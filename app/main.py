# app/main.py
import json
import os
import time
from contextlib import asynccontextmanager

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
    explanation: str = None
    evidence_context: str = None
    explain_model: str = None


# ============================================================
# Lifespan — load model, connect Neo4j (with retry), init Gemini
# ============================================================
_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---- 1. XGBoost ----
    model_path = os.path.join(
        os.path.dirname(__file__), "..", "models", "xgb_tabular.json"
    )
    if not os.path.exists(model_path):
        raise RuntimeError(f"Model not found at {model_path}")
    clf = xgb.XGBClassifier()
    clf.load_model(model_path)
    _models["xgb"] = clf
    print(f"✅ Loaded XGBoost model ({HEADLINE})")

    # ---- 2. Neo4j (retry up to 60s) ----
    uri = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
    driver = GraphDatabase.driver(
        uri,
        auth=(
            os.getenv("NEO4J_USER", "neo4j"),
            os.getenv("NEO4J_PASSWORD", "amlpassword123"),
        ),
        keep_alive=True,
        connection_acquisition_timeout=5,
    )

    connected = False
    last_err = None
    for attempt in range(1, 31):          # 30 × 2s = 60s max
        try:
            driver.verify_connectivity()
            connected = True
            print(f"✅ Connected to Neo4j ({uri}) after ~{attempt * 2}s")
            break
        except Exception as e:
            last_err = e
            if attempt == 1 or attempt % 5 == 0:
                print(f"  Neo4j not ready (attempt {attempt}/30): {type(e).__name__}")
            time.sleep(2)

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
            print(f"✅ Gemini client ready (key starts: {gemini_key[:6]}...)")
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
    prob = 1.0 / (1.0 + np.exp(-raw_logit))
    return ScoreResponse(
        risk_score=prob,
        flagged=raw_logit >= THRESHOLD,
        threshold=THRESHOLD,
        model=HEADLINE,
    )


GEMINI_PRIMARY = "gemini-3.5-flash-lite"
GEMINI_FALLBACKS = ["gemini-3.6-flash", "gemini-2.5-flash"]

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
    if not s.flagged and not explain_if_flagged:
        return AnalyzeResponse(score=s)

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