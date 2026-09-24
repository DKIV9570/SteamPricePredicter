"""Hypothesis: publishers discount when the sales curve's decline steepens.
Test: add review-velocity features (sales proxy: level, 1st/2nd derivative, decay from launch/peak)
and compare on games that have not had a regular sale yet.

    python analysis/08_velocity_test.py
"""
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pricepredicter.reach import MONOTONE, VELOCITY  # noqa: E402
from pricepredicter.reach_data import build_reach_dataset  # noqa: E402

rows, FEATS, _ = build_reach_dataset()
seg = rows[(rows["n_sales_so_far"] <= rows["had_launch_disc"]) & (rows["reached_before"] == 0)].copy()
del rows
print(f"segment rows: {len(seg):,}; with velocity: {seg['rv_rate_last'].notna().mean():.1%}")

tr, va, te = (seg[seg.split == s] for s in ["train", "valid", "test"])
sets = {"不含销量变化": [f for f in FEATS if f not in VELOCITY], "含销量变化": FEATS}
models = {}
for name, fs in sets.items():
    params = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_child_samples=200,
                  feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1, verbose=-1, num_threads=14,
                  monotone_constraints=[MONOTONE.get(f, 0) for f in fs], monotone_constraints_method="advanced")
    m = lgb.train(params, lgb.Dataset(tr[fs], tr["reached"]), 3000, valid_sets=[lgb.Dataset(va[fs], va["reached"])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    te[name] = m.predict(te[fs], num_iteration=m.best_iteration)
    models[name] = m
    print(f"trained {name}: {m.best_iteration} trees", flush=True)


def metrics(x):
    r = {}
    for k in sets:
        r[(k, "AUC")] = roc_auc_score(x.reached, x[k]) if x.reached.nunique() > 1 else np.nan
        r[(k, "准确率")] = np.mean((x[k] >= 0.5) == x.reached)
    r[("", "n")] = len(x)
    return pd.Series(r)


te["时点"] = pd.cut(te.lw, [-1, 1, 6, 13, 26, 1000], labels=["≤1周", "2-6周", "3个月", "半年", "半年以上"])
te["有评测时间线"] = np.where(te["rv_rate_last"].notna(), "有", "无")
pd.set_option("display.width", 250)
for title, col in [("全部", None), ("按提问时点", "时点"), ("是否有销量数据", "有评测时间线"), ("按发行商", "pub_group")]:
    print(f"\n=== {title} ===")
    r = pd.DataFrame({"全部": metrics(te)}).T if col is None else te.groupby(col, observed=True).apply(metrics)
    print(r.round(3).to_string())

imp = pd.Series(models["含销量变化"].feature_importance("gain"), index=models["含销量变化"].feature_name())
imp = imp / imp.sum()
print("\n销量变化特征的重要性:", imp[VELOCITY].round(3).to_dict())
print("排名:", {f: int((imp > imp[f]).sum()) + 1 for f in VELOCITY})
