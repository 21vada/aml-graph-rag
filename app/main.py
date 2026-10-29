# app/main.py
import os
from fastapi import FastAPI
from pydantic import BaseModel


class Transaction(BaseModel):
    source: str
    target: str
    amount_paid: float
    timestamp: str


class ScoreResponse(BaseModel):
    risk_score: float
    flagged: bool


app = FastAPI(title="AML Scoring API")


@app.get("/")
def root():
    return {
        "service": "AML Scoring API",
        "docs": "/docs",
        "endpoints": ["/health", "/score"],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/score", response_model=ScoreResponse)
def score(tx: Transaction):
    # placeholder — real model loading comes in Step 2
    dummy_prob = min(tx.amount_paid / 10000.0, 1.0)
    return ScoreResponse(
        risk_score=dummy_prob,
        flagged=dummy_prob >= 0.5,
    )