# scripts/make_demo_model.py
"""Generate a small, honest demo XGBoost model so the API can run without
the real trained artifact (models/xgb_tabular.json, which is gitignored).

The demo model uses exactly the feature set from models/feature_names.json
and computes causal rolling/pair features directly from data/sample_graph.csv
with a chronological 80/20 split. It is NOT the Colab-trained model — it is
a stand-in so `docker compose up` / local serving works out of the box.

Output: models/demo_xgb.json (gitignored; app/main.py prefers
models/xgb_tabular.json automatically when it exists).

Usage:
    python scripts/make_demo_model.py
"""
import json
import os

import numpy as np
import pandas as pd
import xgboost as xgb

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..")
CSV_PATH = os.path.join(ROOT, "data", "sample_graph.csv")
OUT_PATH = os.path.join(ROOT, "models", "demo_xgb.json")

FEATURE_NAMES = json.load(open(os.path.join(ROOT, "models", "feature_names.json")))


def build_frame(df: pd.DataFrame) -> np.ndarray:
    """Build the full feature matrix ordered per feature_names.json.

    All features are computed causally (only past rows visible), mirroring
    the training pipeline's intent. This is a demo artifact, not a
    reproduction of the Colab pipeline.
    """
    df = df.copy()
    df["ts"] = pd.to_datetime(df["timestamp"], format="%Y/%m/%d %H:%M", errors="coerce")
    df = df.sort_values("ts").reset_index(drop=True)

    n = len(df)
    ts_sec = df["ts"].astype("int64").to_numpy() // 10**9
    src = df["source"].to_numpy()
    tgt = df["target"].to_numpy()
    amt_np = df["amount_paid"].astype(float).to_numpy()

    col_map: dict[str, np.ndarray] = {
        "log_amount": np.log1p(np.clip(amt_np, 0, None)),
        "sin_hour": np.sin(ts_sec % 86400 / 86400 * 2 * np.pi),
        "cos_hour": np.cos(ts_sec % 86400 / 86400 * 2 * np.pi),
        "self_loop": (src == tgt).astype(float),
    }

    WINDOWS = {"": None, "_1h": 1, "_24h": 24, "_96h": 96, "_168h": 168}

    def account_stats(prefix: str, self_col, other_col):
        """Causal per-account out/in stats for one side of each edge.

        prefix 's_' → row's sender; 'r_' → row's receiver.
        {p}out_* = history of transactions sent BY this account,
        {p}in_*  = history of transactions received BY this account.
        """
        hist_out: dict[str, list[tuple[int, float, str]]] = {}
        hist_in: dict[str, list[tuple[int, float, str]]] = {}
        first_seen: dict[str, int] = {}
        store = {}
        for suffix in WINDOWS:
            for kind in ("out_sum", "out_cnt", "in_sum", "in_cnt",
                         "out_nuniq", "in_nuniq"):
                name = f"{prefix}{kind}{suffix}"
                if name in FEATURE_NAMES:
                    store[name] = np.zeros(n)
        gap_out = np.zeros(n); gap_in = np.zeros(n); age = np.zeros(n)

        for i in range(n):
            t = ts_sec[i]
            me, other = self_col[i], other_col[i]
            if me not in first_seen:
                first_seen[me] = t
            age[i] = (t - first_seen[me]) / 3600.0
            seen_out = hist_out.get(me, [])
            seen_in = hist_in.get(me, [])
            prior_t = [tt for tt, _, _ in seen_out]
            gap_out[i] = (t - max(prior_t)) / 3600.0 if prior_t else 0.0
            prior_ti = [tt for tt, _, _ in seen_in]
            gap_in[i] = (t - max(prior_ti)) / 3600.0 if prior_ti else 0.0
            for suffix, w in WINDOWS.items():
                cutoff = -np.inf if w is None else t - w * 3600
                wo = [(tt, a, c) for tt, a, c in seen_out if tt >= cutoff]
                wi = [(tt, a, c) for tt, a, c in seen_in if tt >= cutoff]
                k = f"{prefix}out_sum{suffix}"
                if k in store: store[k][i] = sum(a for _, a, _ in wo)
                k = f"{prefix}out_cnt{suffix}"
                if k in store: store[k][i] = len(wo)
                k = f"{prefix}in_sum{suffix}"
                if k in store: store[k][i] = sum(a for _, a, _ in wi)
                k = f"{prefix}in_cnt{suffix}"
                if k in store: store[k][i] = len(wi)
                k = f"{prefix}out_nuniq{suffix}"
                if k in store: store[k][i] = len({c for _, _, c in wo})
                k = f"{prefix}in_nuniq{suffix}"
                if k in store: store[k][i] = len({c for _, _, c in wi})
            hist_out.setdefault(me, []).append((t, amt_np[i], other))
            hist_in.setdefault(other, []).append((t, amt_np[i], me))

        col_map.update(store)
        # *_nuniq_all are the distinct-counterparty counts over full history.
        for kind in ("out_nuniq", "in_nuniq"):
            name = f"{prefix}{kind}_all"
            if name in FEATURE_NAMES and f"{prefix}{kind}" not in store:
                col_map[name] = store.get(f"{prefix}{kind}_", None) if False else \
                    np.array([len({c for _, _, c in hist_out.get(self_col[i], [])})
                              if kind == "out_nuniq"
                              else len({c for _, _, c in hist_in.get(self_col[i], [])})
                              for i in range(n)], dtype=float)
        for name, arr in ((f"{prefix}gap_out", gap_out),
                          (f"{prefix}gap_in", gap_in),
                          (f"{prefix[:-1]}_age", age)):
            if name in FEATURE_NAMES:
                col_map[name] = arr

    account_stats("s_", src, tgt)   # sender-side features
    account_stats("r_", tgt, src)   # receiver-side features

    # Pair-level features
    pair_fwd: dict[tuple, list[int]] = {}
    pair_rev: dict[tuple, list[tuple[int, float]]] = {}
    pf_cnt = np.zeros(n); pf_96 = np.zeros(n)
    pr_cnt = np.zeros(n); pr_sum = np.zeros(n); pr_96 = np.zeros(n)
    for i in range(n):
        key = (src[i], tgt[i]); rev = (tgt[i], src[i])
        fwd_hist = pair_fwd.get(key, [])
        pf_cnt[i] = len(fwd_hist)
        pf_96[i] = sum(1 for tt in fwd_hist if tt >= ts_sec[i] - 96 * 3600)
        rev_hist = pair_rev.get(rev, [])
        pr_cnt[i] = len(rev_hist)
        pr_sum[i] = sum(v for _, v in rev_hist)
        pr_96[i] = sum(1 for tt, _ in rev_hist if tt >= ts_sec[i] - 96 * 3600)
        pair_fwd.setdefault(key, []).append(ts_sec[i])
        pair_rev.setdefault(rev, []).append((ts_sec[i], amt_np[i]))
    col_map.update({
        "pair_fwd_cnt": pf_cnt, "pair_fwd_cnt_96h": pf_96,
        "pair_rev_cnt": pr_cnt, "pair_rev_sum": pr_sum, "pair_rev_cnt_96h": pr_96,
    })

    # Amount vs own historical average
    def avg_ratio(self_col):
        hist: dict[str, list[float]] = {}
        cur = np.zeros(n)
        for i in range(n):
            h = hist.get(self_col[i], [])
            avg = np.mean(h) if h else 0.0
            cur[i] = amt_np[i] / (avg + 1.0)
            hist.setdefault(self_col[i], []).append(amt_np[i])
        return cur

    col_map["amt_vs_s_avg"] = avg_ratio(src)
    col_map["amt_vs_r_avg"] = avg_ratio(tgt)

    X = np.zeros((n, len(FEATURE_NAMES)), dtype=np.float32)
    missing = []
    for j, name in enumerate(FEATURE_NAMES):
        if name in col_map:
            X[:, j] = col_map[name]
        else:
            missing.append(name)
    if missing:
        print(f"  (demo note: {len(missing)} features left at 0: {missing})")
    return X


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    y = df["label"].astype(int).to_numpy()
    X = build_frame(df)

    n = len(df)
    split = int(n * 0.8)  # chronological: last 20% held out
    clf = xgb.XGBClassifier(
        n_estimators=150, max_depth=5, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="aucpr", tree_method="hist", random_state=42,
    )
    clf.fit(X[:split], y[:split])
    clf.save_model(OUT_PATH)

    print(f"✅ Wrote demo model to {OUT_PATH}")
    print(f"   rows={n} train={split} holdout={n - split} positives={int(y.sum())}")
    print("   NOTE: this is a DEMO model. For real results use the Colab-trained")
    print("   artifact at models/xgb_tabular.json (app/main.py prefers it automatically).")


if __name__ == "__main__":
    main()
