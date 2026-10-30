# app/features.py
"""
Feature engineering for single-transaction inference.

For production, rolling-window features (e.g. s_out_sum_24h) would be
fetched from a feature store or queried directly from Neo4j. For this
demo, they default to 0.0 so the service can run standalone.

The ORDER of the returned vector MUST match feature_names.json exactly,
or XGBoost will predict garbage.
"""
import json
import os
from datetime import datetime

import numpy as np

# Load artifacts once at import time
_MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

def _load_json(name):
    with open(os.path.join(_MODELS_DIR, name)) as f:
        return json.load(f)

FEATURE_NAMES  = _load_json("feature_names.json")
LOGGED_NAMES   = set(_load_json("logged_features.json"))
_MODEL_META    = _load_json("model_meta.json")
THRESHOLD      = _MODEL_META["threshold"]
HEADLINE       = _MODEL_META["headline"]


def _static_features(tx: dict) -> dict:
    """Compute the features we CAN compute from a single transaction."""
    amt = float(tx.get("amount_paid", 0.0))
    ts  = tx.get("timestamp", "")

    # Cyclical time-of-day (matches training: sin_hour, cos_hour)
    try:
        dt = datetime.strptime(ts, "%Y-%m-%d %H:%M")
        hour_frac = (dt.hour * 3600 + dt.minute * 60) / 86400.0
    except (ValueError, TypeError):
        hour_frac = 0.0
    angle = hour_frac * 2 * np.pi

    return {
        "log_amount": np.log1p(max(amt, 0.0)),
        "sin_hour":   float(np.sin(angle)),
        "cos_hour":   float(np.cos(angle)),
        "self_loop":  int(tx.get("source") == tx.get("target")),
    }


def build_features(tx: dict) -> np.ndarray:
    """
    Build a (1, n_features) float32 array ordered exactly per feature_names.json.

    Rolling-window features default to 0.0 when not provided — the model still
    produces a valid prediction, just without the graph-history signal.
    """
    static = _static_features(tx)

    row = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
    for i, name in enumerate(FEATURE_NAMES):
        if name in static:
            val = static[name]
        else:
            # Accept rolling features from the payload if provided
            # (allows future integration with a feature store)
            val = float(tx.get(name, 0.0))
        # Apply the same log1p transform used at training time
        if name in LOGGED_NAMES:
            val = np.log1p(max(val, 0.0))
        row[i] = val

    return row.reshape(1, -1)