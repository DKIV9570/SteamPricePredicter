"""Model the user question: P(price is at >= X% off at some point within H | what's known now).
Trains, evaluates on unseen later games, and saves everything the live predictor needs.

    python analysis/05_reach_model.py
"""
import json
import pickle
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pricepredicter.config import MODELS, PROCESSED  # noqa: E402
from pricepredicter.reach import MONOTONE  # noqa: E402
from pricepredicter.reach_data import build_reach_dataset  # noqa: E402

rows, FEATS, parts = build_reach_dataset()
g, vz, STATIC = parts["g"], parts["vectorizer"], parts["static"]
print(f"rows: {len(rows):,}", rows["split"].value_counts().to_dict())
print("label rate by threshold:", rows.groupby("threshold")["reached"].mean().round(3).to_dict())
print("re-reach rows share:", round(rows["reached_before"].mean(), 3))

mono = [MONOTONE.get(f, 0) for f in FEATS]
tr, va, te = (rows[rows.split == s] for s in ["train", "valid", "test"])
params = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_child_samples=200,
              feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1, monotone_constraints=mono,
              monotone_constraints_method="advanced", verbose=-1, num_threads=14)
m = lgb.train(params, lgb.Dataset(tr[FEATS], tr["reached"], free_raw_data=True), 3000,
              valid_sets=[lgb.Dataset(va[FEATS], va["reached"])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
print(f"trees: {m.best_iteration}")
te = te.assign(p=m.predict(te[FEATS], num_iteration=m.best_iteration))

# Artifacts for the live predictor and the website export
out = MODELS
out.mkdir(exist_ok=True)
m.save_model(str(out / "model.txt"), num_iteration=m.best_iteration)
(out / "features.json").write_text(json.dumps({
    "features": FEATS, "static": STATIC,
    "itad_tags": [c.removeprefix("tag_") for c in g.columns if c.startswith("tag_")]}, ensure_ascii=False))
(out / "vectorizer.pkl").write_bytes(pickle.dumps(vz))


# Baseline: historical rate among similar situations in train
def bucket(d):
    return pd.DataFrame({"threshold": d.threshold, "horizon": d.horizon.astype(str), "lw": d.lw,
                         "rb": d.reached_before, "bc": pd.cut(d.best_cut_so_far, [-1, 0, 19, 39, 100]).astype(str)})


key = lambda d: bucket(d).astype(str).agg("|".join, axis=1)  # noqa: E731
rate = tr.groupby(key(tr))["reached"].mean()
te = te.assign(p_base=key(te).map(rate).fillna(tr["reached"].mean()))


def report(d):
    out = {}
    for k in ["base", "model"]:
        p = d["p_base"] if k == "base" else d["p"]
        out[f"{k}_AUC"] = roc_auc_score(d.reached, p) if d.reached.nunique() > 1 else np.nan
        out[f"{k}_Brier"] = brier_score_loss(d.reached, p)
        out[f"{k}_决策准确率"] = np.mean((p >= 0.5) == d.reached)
    out["n"] = len(d)
    return pd.Series(out)


pd.set_option("display.width", 250)
te["时点"] = pd.cut(te.lw, [-1, 0, 1, 13, 52, 1000], labels=["上架时", "上架1周", "2周-3个月", "半年-1年", "1年以上"])
te["类型"] = np.where(te.reached_before == 1, "再次到达", "首次到达")
for name, col in [("全部", None), ("按提问时点", "时点"), ("首次/再次", "类型"), ("按时间范围", "horizon"),
                  ("按发行商", "pub_group"), ("按目标折扣", "threshold")]:
    print(f"\n=== {name} ===")
    r = pd.DataFrame({"全部": report(te)}).T if col is None else te.groupby(col, observed=True).apply(report)
    print(r.round(3).to_string())

te["bin"] = pd.cut(te.p, np.linspace(0, 1, 11))
print("\n=== 校准（模型说的概率 vs 实际发生比例）===")
print(te.groupby("bin", observed=True).agg(预测=("p", "mean"), 实际=("reached", "mean"), n=("p", "size")).round(3).to_string())

imp = pd.Series(m.feature_importance("gain"), index=m.feature_name())
print("\ntop 特征:", (imp / imp.sum()).sort_values(ascending=False).head(20).round(3).to_dict())
te.drop(columns=["bin"]).to_parquet(PROCESSED / "reach_test_preds.parquet", index=False)
