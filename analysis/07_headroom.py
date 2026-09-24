"""Is the remaining error a similarity problem or irreducible noise?

On games that have NOT had a regular sale yet at the moment of asking, compare:
  lookup     historical rate by (target, horizon, age, ...) — the baseline
  no-sim     model without any similarity features (kNN games, similar publishers, tag embedding)
  full       current model
  oracle     full + PERFECT publisher knowledge: how deep ALL of the publisher's other games
             eventually went (including future games) — i.e. publisher "clustering" done perfectly.
If oracle >> full, better similarity/clustering has room to help; if not, it's noise.

    python analysis/07_headroom.py
"""
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pricepredicter.reach import AGES_WEEKS, MONOTONE  # noqa: E402
from pricepredicter.reach_data import build_reach_dataset  # noqa: E402

rows, FEATS, parts = build_reach_dataset()
g, best = parts["g"], parts["best"]
data_end = g["t0"].max()

# Games that haven't had a regular sale yet (a launch discount doesn't count)
seg = rows[(rows["n_sales_so_far"] <= rows["had_launch_disc"]) & (rows["reached_before"] == 0)].copy()
del rows
print(f"segment rows: {len(seg):,}", seg["split"].value_counts().to_dict())

# Oracle: leave-one-out, all-time mean depth reached by age A over the publisher's OTHER games
d = g[["appid", "publisher", "t0"]].merge(best, on="appid")
orc = d[["appid"]].copy()
for a in AGES_WEEKS:
    ok = d["t0"] + pd.Timedelta(weeks=a) < data_end  # only games old enough to have an answer
    v = d[f"best_{a}w"].where(ok)
    s = v.groupby(d["publisher"]).transform("sum")
    n = v.notna().groupby(d["publisher"]).transform("sum")
    orc[f"orc_pub_best_{a}w"] = (s - v.fillna(0)) / (n - v.notna()).replace(0, np.nan)
seg = seg.merge(orc, on="appid", how="left")
ORC = [c for c in orc.columns if c != "appid"]
print("oracle coverage (games whose publisher has other games):",
      seg.drop_duplicates("appid")[ORC[1]].notna().groupby(seg.drop_duplicates("appid")["pub_group"]).mean().round(2).to_dict())

SIM = [f for f in FEATS if f.startswith(("knn_", "pknn_", "emb_"))]
sets = {"no-sim": [f for f in FEATS if f not in SIM], "full": FEATS, "oracle": FEATS + ORC}

tr, va, te = (seg[seg.split == s] for s in ["train", "valid", "test"])
bucket = lambda x: (x.threshold.astype(int).astype(str) + "|" + x.horizon.astype(str) + "|"  # noqa: E731
                    + x.lw.astype(int).astype(str))
te = te.assign(p_lookup=bucket(te).map(tr.groupby(bucket(tr))["reached"].mean()).fillna(tr["reached"].mean()))
for name, fs in sets.items():
    params = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_child_samples=200,
                  feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1, verbose=-1, num_threads=14,
                  monotone_constraints=[MONOTONE.get(f, 0) for f in fs], monotone_constraints_method="advanced")
    m = lgb.train(params, lgb.Dataset(tr[fs], tr["reached"]), 3000, valid_sets=[lgb.Dataset(va[fs], va["reached"])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    te[f"p_{name}"] = m.predict(te[fs], num_iteration=m.best_iteration)
    print(f"trained {name}: {m.best_iteration} trees", flush=True)


def metrics(x):
    r = {}
    for k in ["lookup", "no-sim", "full", "oracle"]:
        p = x[f"p_{k}"]
        r[(k, "AUC")] = roc_auc_score(x.reached, p)
        r[(k, "准确率")] = np.mean((p >= 0.5) == x.reached)
    r[("", "n")] = len(x)
    return pd.Series(r)


pd.set_option("display.width", 250)
res = pd.concat([metrics(te).rename("全部")] + [metrics(v).rename(k) for k, v in te.groupby("pub_group")], axis=1).T
print(res.round(3).to_string())
