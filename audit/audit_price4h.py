from pathlib import Path
import json, warnings, math
import numpy as np
import pandas as pd
import polars as pl
from huggingface_hub import snapshot_download
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss, roc_auc_score, matthews_corrcoef
from scipy.stats import binomtest
from statsmodels.stats.contingency_tables import mcnemar

warnings.filterwarnings("ignore")
OUT=Path("audit_results"); OUT.mkdir(exist_ok=True)
REPO_ID="ibrahimdaud/binance-btcusdt"
TEST_YEARS=[2022,2023,2024,2025,2026]
H=4
RANDOM_STATE=172

root=snapshot_download(
    repo_id=REPO_ID, repo_type="dataset",
    allow_patterns=["features/BTCUSDT/*.parquet","features/BTCUSDT/**/*.parquet","features/**/*.parquet"],
    max_workers=16
)
files=sorted(Path(root).glob("features/**/*.parquet"))
files=[p for p in files if "aggtrades" not in str(p).lower()]
if not files: raise RuntimeError("No feature parquet files found")
df=pl.read_parquet([str(p) for p in files]).to_pandas()

# Frozen source columns used by original Stage 3.
df["ts"]=pd.to_datetime(pd.to_numeric(df["bar_time_ms"],errors="coerce"),unit="ms",utc=True)
df["close"]=pd.to_numeric(df["close"],errors="coerce")
df=df.dropna(subset=["ts","close"]).sort_values("ts").drop_duplicates("ts").set_index("ts")

w=pd.DataFrame(index=df.index)
w["close"]=df["close"]
r5=w.close.pct_change()
for h,n in [(1,12),(4,48),(12,144),(24,288)]:
    w[f"ret_{h}h"]=w.close.pct_change(n)
for h,n in [(1,12),(4,48),(24,288)]:
    w[f"rv_{h}h"]=r5.rolling(n,min_periods=n).std()*np.sqrt(n)

FEATURES=["ret_1h","ret_4h","ret_12h","ret_24h","rv_1h","rv_4h","rv_24h"]

future_idx=w.index+pd.Timedelta(hours=4)
future=w.close.reindex(future_idx); future.index=w.index
w["future_close"]=future
w["fwd_ret"]=w.future_close/w.close-1
w["y"]=(w.fwd_ret>0).astype(float)

# 00:55 UTC = 5m bar that completes at 01:00 UTC / 10:00 KST.
snap=w[(w.index.hour==0)&(w.index.minute==55)].copy()
snap["sma200"]=snap.close.rolling(200,min_periods=200).mean()
snap["bull"]=snap.close>=snap.sma200

def make_model():
    return HistGradientBoostingClassifier(
        learning_rate=.04,max_iter=120,max_depth=2,min_samples_leaf=25,
        l2_regularization=1.0,random_state=RANDOM_STATE
    )

rows=[]
for yr in TEST_YEARS:
    st=pd.Timestamp(f"{yr}-01-01",tz="UTC")
    en=pd.Timestamp(f"{yr+1}-01-01",tz="UTC")
    purge=pd.Timedelta(hours=4)

    tr=snap[(snap.index+purge<st)&snap.y.notna()].dropna(subset=FEATURES+["y"])
    te=snap[(snap.index>=st)&(snap.index<en)&snap.y.notna()].dropna(subset=FEATURES+["y","fwd_ret"])
    if len(tr)<250 or len(te)==0: continue

    model=make_model()
    model.fit(tr[FEATURES],tr.y.astype(int))
    p=model.predict_proba(te[FEATURES])[:,1]
    pred=(p>=.5).astype(int)

    train_rate=float(tr.y.mean())
    majority=1 if train_rate>=.5 else 0
    momentum=(te["ret_4h"]>0).astype(int).values
    meanrev=1-momentum

    # Past-only volatility state: compare today's rv24h to trailing 365 snapshot distribution.
    vol_state=[]
    for t in te.index:
        hist=snap.loc[(snap.index<t)&(snap.index>=t-pd.Timedelta(days=365)),"rv_24h"].dropna()
        if len(hist)<100:
            vol_state.append("UNKNOWN")
        else:
            q1,q2=hist.quantile([1/3,2/3]).values
            v=te.loc[t,"rv_24h"]
            vol_state.append("LOW" if v<=q1 else ("MID" if v<=q2 else "HIGH"))

    z=pd.DataFrame({
        "timestamp":te.index,
        "year":yr,
        "y_true":te.y.astype(int).values,
        "p_up":p,
        "model_pred":pred,
        "always_up":1,
        "always_down":0,
        "majority_train":majority,
        "momentum_4h":momentum,
        "meanrev_4h":meanrev,
        "fwd_ret":te.fwd_ret.values,
        "ret_4h":te.ret_4h.values,
        "rv_24h":te.rv_24h.values,
        "bull":te.bull.values,
        "vol_state":vol_state,
    })
    rows.append(z)

oos=pd.concat(rows,ignore_index=True).sort_values("timestamp").reset_index(drop=True)
oos["model_hit"]=(oos.model_pred==oos.y_true).astype(int)

def metric_for_pred(pred_col):
    y=oos.y_true.values.astype(int)
    pr=oos[pred_col].values.astype(int)
    return {
        "n":len(y),
        "accuracy":float((pr==y).mean()),
        "mcc":float(matthews_corrcoef(y,pr)),
    }

def model_metrics(df0=oos):
    y=df0.y_true.values.astype(int); p=df0.p_up.values; pr=(p>=.5).astype(int)
    return {
        "n":len(y),
        "accuracy":float(accuracy_score(y,pr)),
        "auc":float(roc_auc_score(y,p)),
        "mcc":float(matthews_corrcoef(y,pr)),
        "log_loss":float(log_loss(y,np.c_[1-p,p],labels=[0,1])),
        "brier":float(brier_score_loss(y,p)),
    }

main=model_metrics()
hits=int(oos.model_hit.sum()); n=len(oos)
bt=binomtest(hits,n,.5,alternative="greater")
ci=binomtest(hits,n).proportion_ci(.95,method="wilson")

# Month-block bootstrap, including paired differences to baselines.
oos["month"]=pd.to_datetime(oos.timestamp).dt.strftime("%Y-%m")
month_groups=[]
for m,g in oos.groupby("month"):
    month_groups.append({
        "n":len(g),
        "model_hits":int((g.model_pred==g.y_true).sum()),
        "mom_hits":int((g.momentum_4h==g.y_true).sum()),
        "maj_hits":int((g.majority_train==g.y_true).sum()),
        "up_hits":int((g.always_up==g.y_true).sum()),
    })
mg=pd.DataFrame(month_groups)
rng=np.random.default_rng(RANDOM_STATE)
B=30000
idx=rng.integers(0,len(mg),size=(B,len(mg)))
den=mg.n.to_numpy()[idx].sum(axis=1)
ma=mg.model_hits.to_numpy()[idx].sum(axis=1)/den
mom=mg.mom_hits.to_numpy()[idx].sum(axis=1)/den
maj=mg.maj_hits.to_numpy()[idx].sum(axis=1)/den
up=mg.up_hits.to_numpy()[idx].sum(axis=1)/den

def qci(x):
    return [float(v) for v in np.quantile(x,[.025,.5,.975])]

# McNemar exact paired tests: model vs baseline.
def mc(pred_col):
    a=(oos.model_pred==oos.y_true).values
    b=(oos[pred_col]==oos.y_true).values
    table=[[int((a&b).sum()),int((a&~b).sum())],[int((~a&b).sum()),int((~a&~b).sum())]]
    r=mcnemar(table,exact=True)
    return {"table":table,"statistic":float(r.statistic),"pvalue":float(r.pvalue)}

# Calibration bins.
cal=oos.copy()
cal["bin"]=pd.cut(cal.p_up,bins=np.linspace(0,1,11),include_lowest=True)
calib=cal.groupby("bin",observed=False).agg(
    n=("y_true","size"),mean_p=("p_up","mean"),actual_up=("y_true","mean")
).reset_index()
calib["abs_gap"]=(calib.mean_p-calib.actual_up).abs()
ece=float((calib.n/calib.n.sum()*calib.abs_gap.fillna(0)).sum())

# Confidence accuracy.
conf=[]
for th in [.55,.60,.65]:
    mask=np.maximum(oos.p_up,1-oos.p_up)>=th
    conf.append({
        "threshold":th,"n":int(mask.sum()),"coverage":float(mask.mean()),
        "accuracy":float((oos.loc[mask,"model_pred"]==oos.loc[mask,"y_true"]).mean()) if mask.any() else np.nan
    })

# Year, bull/bear, volatility, month diagnostics.
def split_table(col):
    out=[]
    for k,g in oos.groupby(col,dropna=False):
        if len(g)==0: continue
        m=model_metrics(g)
        out.append({col:str(k),**m})
    return pd.DataFrame(out)

by_year=split_table("year")
by_bull=split_table("bull")
by_vol=split_table("vol_state")
by_month=split_table("month")

# Latest-window robustness, explicitly NOT a pristine holdout.
latest=[]
for count in [60,120,180]:
    g=oos.tail(min(count,len(oos)))
    latest.append({"window_last_n":len(g),**model_metrics(g)})
latest=pd.DataFrame(latest)

# Economic diagnostic; non-overlapping daily 4h windows.
# Long/cash: hold only when model predicts UP. Costs in round-trip bps per active trade.
econ=[]
for bps in [0,10,20,30]:
    cost=bps/10000
    r=np.where(oos.model_pred.values==1,oos.fwd_ret.values-cost,0.0)
    mom_r=np.where(oos.momentum_4h.values==1,oos.fwd_ret.values-cost,0.0)
    all_long=oos.fwd_ret.values-cost
    econ.append({
        "roundtrip_bps":bps,
        "model_long_cash_mean":float(np.mean(r)),
        "model_long_cash_total":float(np.prod(1+r)-1),
        "momentum_long_cash_mean":float(np.mean(mom_r)),
        "momentum_long_cash_total":float(np.prod(1+mom_r)-1),
        "always_long_4h_mean":float(np.mean(all_long)),
        "always_long_4h_total":float(np.prod(1+all_long)-1),
        "model_active_share":float((oos.model_pred==1).mean()),
    })
econ=pd.DataFrame(econ)

baselines=pd.DataFrame([
    {"baseline":"MODEL_HGB_PRICE",**main},
    {"baseline":"ALWAYS_UP",**metric_for_pred("always_up")},
    {"baseline":"ALWAYS_DOWN",**metric_for_pred("always_down")},
    {"baseline":"MAJORITY_TRAIN",**metric_for_pred("majority_train")},
    {"baseline":"MOMENTUM_4H",**metric_for_pred("momentum_4h")},
    {"baseline":"MEAN_REVERSION_4H",**metric_for_pred("meanrev_4h")},
])

audit={
    "model":main,
    "hits":hits,
    "n":n,
    "wilson95":[float(ci.low),float(ci.high)],
    "binom_p_gt_50":float(bt.pvalue),
    "base_up_rate":float(oos.y_true.mean()),
    "constant_50_logloss":float(-math.log(.5)),
    "constant_50_brier":0.25,
    "block_bootstrap_accuracy95":qci(ma),
    "block_bootstrap_diff_vs_momentum95":qci(ma-mom),
    "block_bootstrap_diff_vs_majority95":qci(ma-maj),
    "block_bootstrap_diff_vs_always_up95":qci(ma-up),
    "mcnemar_vs_momentum":mc("momentum_4h"),
    "mcnemar_vs_majority":mc("majority_train"),
    "mcnemar_vs_always_up":mc("always_up"),
    "ece_10bin":ece,
    "pristine_final_holdout_available":False,
    "holdout_note":"2022-2026 OOS results have already been inspected; no pristine final holdout remains in the frozen dataset. Latest-window results below are robustness diagnostics only."
}

oos.to_csv(OUT/"oos_predictions.csv",index=False)
baselines.to_csv(OUT/"baselines.csv",index=False)
calib.to_csv(OUT/"calibration.csv",index=False)
pd.DataFrame(conf).to_csv(OUT/"confidence.csv",index=False)
by_year.to_csv(OUT/"by_year.csv",index=False)
by_bull.to_csv(OUT/"by_bull_bear.csv",index=False)
by_vol.to_csv(OUT/"by_volatility.csv",index=False)
by_month.to_csv(OUT/"by_month.csv",index=False)
latest.to_csv(OUT/"latest_windows.csv",index=False)
econ.to_csv(OUT/"economic_diagnostic.csv",index=False)
with open(OUT/"audit.json","w") as f: json.dump(audit,f,indent=2)

lines=[
"# BTC 4H PRICE-only Independent Audit",
"",
"Frozen model: Stage 3 M0_PRICE / HGB / 10:00 KST decision / +4H direction.",
"",
"## Core statistics",
"",
f"- OOS n: {n}",
f"- Accuracy: {main['accuracy']:.4%}",
f"- Wilson 95% CI: {ci.low:.4%} ~ {ci.high:.4%}",
f"- Exact binomial p-value vs 50% (one-sided): {bt.pvalue:.6g}",
f"- AUC: {main['auc']:.4f}",
f"- MCC: {main['mcc']:.4f}",
f"- Log loss: {main['log_loss']:.6f} (50% constant = {-math.log(.5):.6f})",
f"- Brier: {main['brier']:.6f} (50% constant = 0.250000)",
f"- ECE (10 bins): {ece:.6f}",
"",
"## Baselines",
"",
baselines.to_markdown(index=False),
"",
"## Confidence",
"",
pd.DataFrame(conf).to_markdown(index=False),
"",
"## Year",
"",
by_year.to_markdown(index=False),
"",
"## Bull/Bear",
"",
by_bull.to_markdown(index=False),
"",
"## Volatility",
"",
by_vol.to_markdown(index=False),
"",
"## Latest-window robustness (NOT pristine holdout)",
"",
latest.to_markdown(index=False),
"",
"## Economic diagnostic",
"",
econ.to_markdown(index=False),
"",
"## Statistical comparison",
"",
f"- Month-block bootstrap accuracy 95%: {qci(ma)}",
f"- Accuracy diff vs 4H momentum 95%: {qci(ma-mom)}",
f"- Accuracy diff vs training-majority 95%: {qci(ma-maj)}",
f"- Accuracy diff vs always-UP 95%: {qci(ma-up)}",
f"- McNemar vs momentum: {audit['mcnemar_vs_momentum']}",
f"- McNemar vs majority: {audit['mcnemar_vs_majority']}",
f"- McNemar vs always-UP: {audit['mcnemar_vs_always_up']}",
"",
"## Holdout status",
"",
audit["holdout_note"],
]
(OUT/"AUDIT.md").write_text("\n".join(lines),encoding="utf-8")
print((OUT/"AUDIT.md").read_text())
