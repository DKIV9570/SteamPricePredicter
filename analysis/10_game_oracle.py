"""Upper bound: how accurate could we be on not-yet-discounted games if we KNEW each game's
own pricing strategy?

"Strategy" is revealed by the game's actual sales AFTER the prediction window (L, L+H]:
  oracle-near  sales in the year right after the window   (generous: close in time)
  oracle-far   sales in the second year after the window  (conservative)
These are not the label (which is about the window itself), but since only ~15% of discount
variation is sale-to-sale noise, they reveal the game's persistent strategy.
All models are compared on the same rows: those with two full years observed after the window.

    python analysis/10_game_oracle.py
"""
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pricepredicter.config import PROCESSED  # noqa: E402
from pricepredicter.reach import MONOTONE  # noqa: E402
from pricepredicter.reach_data import build_reach_dataset  # noqa: E402

YEAR = np.timedelta64(365, "D").astype("timedelta64[ns]")

rows, FEATS, parts = build_reach_dataset()
seg = rows[(rows["n_sales_so_far"] <= rows["had_launch_disc"]) & (rows["reached_before"] == 0)].copy()
del rows
sales = pd.read_parquet(PROCESSED / "sales.parquet")
data_end = sales["start"].max().tz_convert("US/Pacific").tz_localize(None)
seg["W"] = seg["L"] + pd.to_timedelta(seg["horizon_days"].astype(float), unit="D")
seg = seg[seg["W"] + 2 * pd.Timedelta(days=365) <= data_end].reset_index(drop=True)
print(f"rows with 2 years observed after the window: {len(seg):,}", seg["split"].value_counts().to_dict())

s = sales[sales["kind"] == "regular"].copy()
s["start"] = s["start"].dt.tz_convert("US/Pacific").dt.tz_localize(None)
s = s.sort_values(["appid", "start"])
by_game = {a: (d["start"].to_numpy("datetime64[ns]"), d["max_cut"].to_numpy(float)) for a, d in s.groupby("appid")}

names = {"near": (0, 1), "far": (1, 2)}
out = {f"orc_{n}_{k}": np.full(len(seg), np.nan) for n in names for k in ["n", "mean", "max"]}
for appid, idx in seg.groupby("appid").indices.items():
    st, cut = by_game.get(appid, (np.array([], "datetime64[ns]"), np.array([])))
    W = seg["W"].to_numpy("datetime64[ns]")[idx]
    for n, (a, b) in names.items():
        lo, hi = np.searchsorted(st, W + a * YEAR, "right"), np.searchsorted(st, W + b * YEAR, "right")
        cnt = hi - lo
        out[f"orc_{n}_n"][idx] = cnt
        for i, (l, h) in enumerate(zip(lo, hi)):
            if h > l:
                out[f"orc_{n}_mean"][idx[i]] = cut[l:h].mean()
                out[f"orc_{n}_max"][idx[i]] = cut[l:h].max()
for k, v in out.items():
    seg[k] = v

ORC = {n: [f"orc_{n}_{k}" for k in ["n", "mean", "max"]] for n in names}
sets = {"当前模型": FEATS, "保守上限(窗口后第2年)": FEATS + ORC["far"], "宽松上限(窗口后第1年)": FEATS + ORC["near"]}
tr, va, te = (seg[seg.split == x] for x in ["train", "valid", "test"])
bucket = lambda x: x.threshold.astype(int).astype(str) + "|" + x.horizon.astype(str) + "|" + x.lw.astype(int).astype(str)  # noqa: E731
te = te.assign(查表=bucket(te).map(tr.groupby(bucket(tr))["reached"].mean()).fillna(tr["reached"].mean()))
for name, fs in sets.items():
    params = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_child_samples=200,
                  feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1, verbose=-1, num_threads=14,
                  monotone_constraints=[MONOTONE.get(f, 0) for f in fs], monotone_constraints_method="advanced")
    m = lgb.train(params, lgb.Dataset(tr[fs], tr["reached"]), 3000, valid_sets=[lgb.Dataset(va[fs], va["reached"])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    te[name] = m.predict(te[fs], num_iteration=m.best_iteration)
    print(f"trained {name}: {m.best_iteration} trees", flush=True)

cols = ["查表"] + list(sets)


def metrics(x):
    r = {}
    for k in cols:
        r[(k, "AUC")] = roc_auc_score(x.reached, x[k]) if x.reached.nunique() > 1 else np.nan
        r[(k, "准确率")] = np.mean((x[k] >= 0.5) == x.reached)
    r[("", "n")] = len(x)
    return pd.Series(r)


pd.set_option("display.width", 250)
price = np.expm1(te["log_price"])
for title, col in [("全部", None), ("按发行商", "pub_group")]:
    print(f"\n=== {title} ===")
    r = pd.DataFrame({"全部": metrics(te)}).T if col is None else te.groupby(col, observed=True).apply(metrics)
    print(r.round(3).to_string())
print("\n=== 按首发价 ===")
print(pd.DataFrame({f"≥${p}": metrics(te[price >= p - 0.01]) for p in [10, 15, 20]}
                   | {"<$10": metrics(te[price < 9.99])}).T.round(3).to_string())
te[["appid", "lw", "threshold", "horizon", "reached", "pub_group", "log_price"] + cols].to_parquet(
    PROCESSED / "game_oracle_preds.parquet", index=False)
