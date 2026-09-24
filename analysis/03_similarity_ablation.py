"""Ablation: how much do content, similar-game and similar-publisher features add,
especially for cold-start publishers?

    python analysis/03_similarity_ablation.py [--rebuild]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pricepredicter.modeling import feature_sets, load_model_table, make_panels, run_experiment  # noqa: E402

g, majors, data_end = load_model_table(rebuild="--rebuild" in sys.argv)
print("games by split:", g["split"].value_counts().to_dict())
panels = make_panels(g, majors, data_end)
print(f"panel rows: train {len(panels.train):,}  valid {len(panels.valid):,}  test(full) {len(panels.test_full):,}")

# Rule baselines for reference
test = g[g["split"] == "test"]
obs = test.dropna(subset=["days_to_first_sale"])
st = np.sort(majors["start"].to_numpy("datetime64[ns]"))
i = np.searchsorted(st, obs["t0"].to_numpy("datetime64[ns]") + np.timedelta64(30, "D"))
rule_days = np.where(i < len(st), (st[np.minimum(i, len(st) - 1)] - obs["t0"].to_numpy("datetime64[ns]"))
                     / np.timedelta64(1, "D"), np.nan)
hit = pd.Series(np.abs(rule_days - obs["days_to_first_sale"]) <= 7, index=obs.index)
tr = g[(g.split == "train")].dropna(subset=["first_sale_cut"])
bins = [0, 5, 10, 20, 30, 45, 1000]
med = tr.groupby(pd.cut(tr.launch_price, bins), observed=True).first_sale_cut.median()
dte = test.dropna(subset=["first_sale_cut"])
rule_cut = pd.cut(dte.launch_price, bins).map(med).astype(float)
derr = (rule_cut - dte.first_sale_cut).abs()

rows = {"规则": {
    **{f"时间±7天命中 | {k}": v for k, v in
       ({"全部": hit.mean()} | hit.groupby(obs.pub_group).mean().to_dict()).items()},
    **{f"折扣MAE | {k}": v for k, v in
       ({"全部": derr.mean()} | derr.groupby(dte.pub_group).mean().to_dict()).items()},
}}
for name, feats in feature_sets(g).items():
    r = run_experiment(g, panels, feats)
    rows[name] = ({f"时间±7天命中 | {k}": v for k, v in r["timing_hit7"].items()}
                  | {f"真实周±1周概率 | {k}": v for k, v in r["timing_mass"].items()}
                  | {f"折扣MAE | {k}": v for k, v in r["depth_mae"].items()})
    print(f"done: {name}", flush=True)

res = pd.DataFrame(rows).T
groups = ["全部", "新发行商(0)", "少量(1-2)", "成熟(3+)"]
pd.set_option("display.width", 250)
for metric, fmt in [("时间±7天命中", "{:.1%}"), ("真实周±1周概率", "{:.1%}"), ("折扣MAE", "{:.2f}")]:
    cols = [f"{metric} | {k}" for k in groups]
    print(f"\n=== {metric} ===")
    print(res.reindex(columns=cols).rename(columns=lambda c: c.split(" | ")[1])
          .map(lambda v: "" if pd.isna(v) else fmt.format(v)).to_string())
print("\n测试集样本数:", r["n"])

hz, dm = r["models"]
for label, m in [("时间模型", hz), ("折扣模型", dm)]:
    imp = pd.Series(m.feature_importance("gain"), index=m.feature_name())
    print(f"{label} top 特征:", (imp / imp.sum()).sort_values(ascending=False).head(15).round(3).to_dict())
res.to_csv(ROOT / "reports" / "ablation_similarity.csv")
