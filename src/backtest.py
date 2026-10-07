from pathlib import Path
import json, re, warnings
import numpy as np
import pandas as pd
import polars as pl
from huggingface_hub import snapshot_download
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss

warnings.filterwarnings("ignore")

REPO_ID = "ibrahimdaud/binance-btcusdt"
OUT = Path("results")
OUT.mkdir(exist_ok=True)

TEST_YEARS = [2022, 2023, 2024, 2025, 2026]
HORIZONS = [4, 24]
RANDOM_STATE = 172

def norm(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())

def find_col(columns, exact=(), contains=()):
    nm = {norm(c): c for c in columns}
    for x in exact:
        if norm(x) in nm:
            return nm[norm(x)]
    for c in columns:
        nc = norm(c)
        if all(norm(x) in nc for x in contains):
            return c
    return None

print("Downloading public feature parquet only...")
root = snapshot_download(
    repo_id=REPO_ID,
    repo_type="dataset",
    allow_patterns=[
        "features/BTCUSDT/*.parquet",
        "features/BTCUSDT/**/*.parquet",
        "features/**/*.parquet",
    ],
    max_workers=16,
)

files = sorted(Path(root).glob("features/**/*.parquet"))
files = [p for p in files if "aggtrades" not in str(p).lower()]
if not files:
    raise RuntimeError("No feature parquet files found.")

print("feature_files", len(files))
print("feature_size_mb", round(sum(p.stat().st_size for p in files) / 1024**2, 1))

df = pl.read_parquet([str(p) for p in files]).to_pandas()
print("raw_rows", len(df))
print("columns", df.columns.tolist())

ts_col = find_col(
    df.columns,
    exact=("bar_time_ms", "timestamp", "datetime", "time", "open_time", "openTime", "ts"),
    contains=("time",),
)
close_col = find_col(
    df.columns,
    exact=("close", "close_price", "closePrice", "price"),
    contains=("close",),
)
if ts_col is None or close_col is None:
    raise RuntimeError(f"Timestamp/close columns not resolved. ts={ts_col}, close={close_col}")

def to_utc(s):
    if pd.api.types.is_numeric_dtype(s):
        x = pd.to_numeric(s, errors="coerce")
        med = x.dropna().abs().median()
        if med > 1e17:
            return pd.to_datetime(x, unit="ns", utc=True, errors="coerce")
        if med > 1e14:
            return pd.to_datetime(x, unit="us", utc=True, errors="coerce")
        if med > 1e11:
            return pd.to_datetime(x, unit="ms", utc=True, errors="coerce")
        return pd.to_datetime(x, unit="s", utc=True, errors="coerce")
    return pd.to_datetime(s, utc=True, errors="coerce")

df["_ts"] = to_utc(df[ts_col])
df["_close"] = pd.to_numeric(df[close_col], errors="coerce")
df = (
    df.dropna(subset=["_ts", "_close"])
      .sort_values("_ts")
      .drop_duplicates("_ts", keep="last")
      .set_index("_ts")
)

wanted = {
    "vpin_50": (("vpin_50", "vpin50"), ("vpin", "50")),
    "vpin_bucket_imbalance": (("vpin_bucket_imbalance",), ("vpin", "imbalance")),
    "hawkes_buy_intensity": (("hawkes_buy_intensity",), ("hawkes", "buy")),
    "hawkes_sell_intensity": (("hawkes_sell_intensity",), ("hawkes", "sell")),
    "hawkes_net": (("hawkes_net",), ("hawkes", "net")),
    "oi_btc": (("oi_btc", "open_interest_btc"), ("oi", "btc")),
    "oi_change_1h": (("oi_change_1h",), ("oi", "change", "1h")),
    "avg_trade_size_5m": (("avg_trade_size_5m",), ("avg", "trade", "size")),
    "trade_count_5m": (("trade_count_5m",), ("trade", "count")),
    "taker_buy_ratio_5m": (("taker_buy_ratio_5m",), ("taker", "buy", "ratio")),
    "ls_count_ratio": (("ls_count_ratio",), ("ls", "count", "ratio")),
    "taker_ls_vol_ratio": (("taker_ls_vol_ratio",), ("taker", "ls", "vol", "ratio")),
    "depth_imbalance_1pct": (("depth_imbalance_1pct",), ("depth", "imbalance")),
}

resolved = {}
for key, (exacts, contains) in wanted.items():
    c = find_col(df.columns, exact=exacts, contains=contains)
    resolved[key] = c
    if c is not None:
        df[key] = pd.to_numeric(df[c], errors="coerce")

print("resolved_columns", json.dumps(resolved, indent=2))

work = pd.DataFrame(index=df.index)
work["close"] = df["_close"]
ret5 = work["close"].pct_change()

# M0 PRICE
for h, n in [(1, 12), (4, 48), (12, 144), (24, 288)]:
    work[f"ret_{h}h"] = work["close"].pct_change(n)
for h, n in [(1, 12), (4, 48), (24, 288)]:
    work[f"rv_{h}h"] = ret5.rolling(n, min_periods=n).std() * np.sqrt(n)

groups = {
    "M0_PRICE": ["ret_1h", "ret_4h", "ret_12h", "ret_24h", "rv_1h", "rv_4h", "rv_24h"]
}

def add_rolls(src, prefix, use_mean=True, use_max=True, use_sum=False):
    if src not in df.columns:
        return []
    out = []
    work[f"{prefix}_now"] = df[src]
    out.append(f"{prefix}_now")
    for n, label in [(12, "1h"), (48, "4h"), (288, "24h")]:
        mp = max(3, n // 3)
        if use_mean:
            name = f"{prefix}_{label}_mean"
            work[name] = df[src].rolling(n, min_periods=mp).mean()
            out.append(name)
        if use_max:
            name = f"{prefix}_{label}_max"
            work[name] = df[src].rolling(n, min_periods=mp).max()
            out.append(name)
        if use_sum:
            name = f"{prefix}_{label}_sum"
            work[name] = df[src].rolling(n, min_periods=mp).sum()
            out.append(name)
    return out

m1 = add_rolls("vpin_50", "vpin") + add_rolls("vpin_bucket_imbalance", "vpinimb")
groups["M1_VPIN"] = groups["M0_PRICE"] + m1

m2 = (
    add_rolls("hawkes_net", "hawkesnet")
    + add_rolls("hawkes_buy_intensity", "hawkesbuy")
    + add_rolls("hawkes_sell_intensity", "hawkessell")
)
if "hawkes_buy_intensity" in df.columns and "hawkes_sell_intensity" in df.columns:
    den = (df["hawkes_buy_intensity"].abs() + df["hawkes_sell_intensity"].abs()).replace(0, np.nan)
    df["hawkes_pressure"] = (df["hawkes_buy_intensity"] - df["hawkes_sell_intensity"]) / den
    m2 += add_rolls("hawkes_pressure", "hawkespressure")
groups["M2_HAWKES"] = groups["M1_VPIN"] + m2

m3 = add_rolls("oi_change_1h", "oichg", use_mean=True, use_max=False, use_sum=True)
if "oi_btc" in df.columns:
    work["oi_4h_pct"] = df["oi_btc"].pct_change(48)
    work["oi_24h_pct"] = df["oi_btc"].pct_change(288)
    m3 += ["oi_4h_pct", "oi_24h_pct"]
groups["M3_OI"] = groups["M2_HAWKES"] + m3

m4 = (
    add_rolls("avg_trade_size_5m", "avgtradesize")
    + add_rolls("trade_count_5m", "tradecount", use_mean=True, use_max=False, use_sum=True)
)
groups["M4_TRADE_SIZE"] = groups["M3_OI"] + m4

m5 = add_rolls("taker_buy_ratio_5m", "takerbuy")
groups["M5_TAKER"] = groups["M4_TRADE_SIZE"] + m5

m6 = (
    add_rolls("ls_count_ratio", "lsratio")
    + add_rolls("taker_ls_vol_ratio", "takerls")
)
groups["M6_POSITIONING"] = groups["M5_TAKER"] + m6

m7 = add_rolls("depth_imbalance_1pct", "depthimb")
groups["M7_ALL"] = groups["M6_POSITIONING"] + m7

for k in groups:
    groups[k] = list(dict.fromkeys(groups[k]))

# exact +4H/+24H targets
for h in HORIZONS:
    idx = work.index + pd.Timedelta(hours=h)
    future = work["close"].reindex(idx)
    future.index = work.index
    work[f"fwd_{h}h"] = future / work["close"] - 1
    work[f"y_{h}h"] = (work[f"fwd_{h}h"] > 0).astype(float)

# Dataset timestamps are 5m bar OPEN times.
# 00:55-01:00 UTC is the last completed bar available at 10:00 KST.
snap = work[(work.index.hour == 0) & (work.index.minute == 55)].copy()
print("snapshots", len(snap), snap.index.min(), snap.index.max())

def model_set():
    return {
        "LOGIT": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.5, max_iter=2000, random_state=RANDOM_STATE),
        ),
        "HGB": HistGradientBoostingClassifier(
            learning_rate=0.04,
            max_iter=120,
            max_depth=2,
            min_samples_leaf=25,
            l2_regularization=1.0,
            random_state=RANDOM_STATE,
        ),
    }

def evaluate(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    pred = (p >= 0.5).astype(int)
    out = {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "log_loss": float(log_loss(y, np.c_[1-p, p], labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
    }
    for th in [0.55, 0.60, 0.65]:
        mask = np.maximum(p, 1-p) >= th
        out[f"conf_{int(th*100)}_n"] = int(mask.sum())
        out[f"conf_{int(th*100)}_coverage"] = float(mask.mean()) if len(mask) else np.nan
        out[f"conf_{int(th*100)}_accuracy"] = (
            float(accuracy_score(y[mask], pred[mask])) if mask.any() else np.nan
        )
    return out

summary_rows = []
prediction_rows = []

for horizon in HORIZONS:
    ycol = f"y_{horizon}h"
    purge = pd.Timedelta(hours=horizon)

    for stage, features in groups.items():
        usable = [f for f in features if f in snap.columns]
        if len(usable) < 4:
            continue

        for model_name, model in model_set().items():
            chunks = []

            for test_year in TEST_YEARS:
                test_start = pd.Timestamp(f"{test_year}-01-01", tz="UTC")
                test_end = pd.Timestamp(f"{test_year+1}-01-01", tz="UTC")

                train = snap[
                    (snap.index + purge < test_start) & snap[ycol].notna()
                ].dropna(subset=usable + [ycol])

                test = snap[
                    (snap.index >= test_start)
                    & (snap.index < test_end)
                    & snap[ycol].notna()
                ].dropna(subset=usable + [ycol])

                if len(train) < 250 or len(test) == 0:
                    continue

                X_train = train[usable].replace([np.inf, -np.inf], np.nan).dropna()
                y_train = train.loc[X_train.index, ycol].astype(int)
                X_test = test[usable].replace([np.inf, -np.inf], np.nan).dropna()
                y_test = test.loc[X_test.index, ycol].astype(int)

                if len(X_train) < 250 or len(X_test) == 0:
                    continue

                model.fit(X_train, y_train)
                p = model.predict_proba(X_test)[:, 1]

                chunks.append(pd.DataFrame({
                    "timestamp": X_test.index,
                    "y_true": y_test.values,
                    "p_up": p,
                    "test_year": test_year,
                }))

            if not chunks:
                continue

            pred_df = pd.concat(chunks).sort_values("timestamp")
            m = evaluate(pred_df["y_true"], pred_df["p_up"])
            summary_rows.append({
                **m,
                "horizon": f"{horizon}H",
                "stage": stage,
                "model": model_name,
                "feature_count": len(usable),
                "scope": "ALL",
            })

            for yr, g in pred_df.groupby("test_year"):
                ym = evaluate(g["y_true"], g["p_up"])
                summary_rows.append({
                    **ym,
                    "horizon": f"{horizon}H",
                    "stage": stage,
                    "model": model_name,
                    "feature_count": len(usable),
                    "scope": f"YEAR_{yr}",
                })

            pred_df["horizon"] = f"{horizon}H"
            pred_df["stage"] = stage
            pred_df["model"] = model_name
            prediction_rows.append(pred_df)

summary = pd.DataFrame(summary_rows)
if not prediction_rows:
    raise RuntimeError("No OOS predictions were produced.")

predictions = pd.concat(prediction_rows, ignore_index=True)
overall = summary[summary["scope"] == "ALL"].copy()

verdict_rows = []
for (horizon, model_name), g in overall.groupby(["horizon", "model"]):
    base = g[g["stage"] == "M0_PRICE"]
    if base.empty:
        continue
    base = base.iloc[0]
    for _, r in g.iterrows():
        c55 = r["conf_55_accuracy"]
        c60 = r["conf_60_accuracy"]
        pass_conf = (
            (pd.isna(c55) or c55 >= r["accuracy"])
            and (pd.isna(c60) or r["conf_60_n"] < 20 or c60 >= r["accuracy"])
        )
        verdict_rows.append({
            "horizon": horizon,
            "model": model_name,
            "stage": r["stage"],
            "n": int(r["n"]),
            "accuracy": r["accuracy"],
            "delta_acc_vs_M0": r["accuracy"] - base["accuracy"],
            "log_loss": r["log_loss"],
            "delta_logloss_vs_M0": r["log_loss"] - base["log_loss"],
            "brier": r["brier"],
            "delta_brier_vs_M0": r["brier"] - base["brier"],
            "conf55_n": int(r["conf_55_n"]),
            "conf55_acc": c55,
            "conf60_n": int(r["conf_60_n"]),
            "conf60_acc": c60,
            "PASS_CORE": bool(
                r["accuracy"] >= 0.55
                and r["log_loss"] < base["log_loss"]
                and r["brier"] < base["brier"]
                and pass_conf
            ),
        })

verdict = pd.DataFrame(verdict_rows)

summary.to_csv(OUT / "summary.csv", index=False)
predictions.to_csv(OUT / "predictions.csv", index=False)
verdict.to_csv(OUT / "verdict.csv", index=False)

with open(OUT / "feature_groups.json", "w", encoding="utf-8") as f:
    json.dump(groups, f, ensure_ascii=False, indent=2)

with open(OUT / "resolved_columns.json", "w", encoding="utf-8") as f:
    json.dump(resolved, f, ensure_ascii=False, indent=2)

lines = [
    "# BTC_TRUE_MICRO_V1 Stage 3",
    "",
    f"Data: {REPO_ID}",
    f"Feature files: {len(files)}",
    f"Daily 10:00 KST snapshots: {len(snap)}",
    "",
    "## Frozen core verdict",
    "",
    verdict.to_markdown(index=False),
]
for h in ["4H", "24H"]:
    q = verdict[verdict["horizon"] == h].sort_values(["model", "log_loss"])
    if len(q):
        lines += ["", f"## {h} best rows", "", q.head(10).to_markdown(index=False)]

(OUT / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")

print(verdict.to_string(index=False))
print("PASS_CORE")
print(verdict[verdict["PASS_CORE"]].to_string(index=False))
