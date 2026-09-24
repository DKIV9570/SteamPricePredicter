"""End-to-end evaluation of what a user would see at launch: "first sale around <date>, at <cut>%".
Depth is predicted from the predicted week, not the true one.

    python analysis/04_end_to_end_eval.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pricepredicter.modeling import feature_sets, load_model_table, make_panels, run_experiment  # noqa: E402

g, majors, data_end = load_model_table()
panels = make_panels(g, majors, data_end)
r = run_experiment(g, panels, feature_sets(g)["E +首月评测"])
t = r["test"].dropna(subset=["days_to_first_sale", "first_sale_cut"])
t["w"] = t["waitlisted"].fillna(0)

d_err = (t["pred_days"] - t["days_to_first_sale"]).abs()
c_err = (t["pred_cut_e2e"] - t["first_sale_cut"]).abs()
rows = {}
for name, m in [("全部", slice(None))] + [(k, t["pub_group"] == k) for k in ["新发行商(0)", "少量(1-2)", "成熟(3+)"]]:
    d, c, w = d_err[m], c_err[m], t["w"][m]
    both = (d <= 7) & (c <= 5)
    rows[name] = {
        "时间±7天命中": f"{np.mean(d <= 7):.1%}",
        "折扣MAE(用真实时间)": f"{r['depth_mae'][name]:.1f}",
        "折扣MAE(端到端)": f"{c.mean():.1f}",
        "折扣±5内": f"{np.mean(c <= 5):.1%}",
        "两者都对": f"{both.mean():.1%}",
        "两者都对(按关注度加权)": f"{(both * w).sum() / w.sum():.1%}",
        "样本": int(len(d)),
    }
print(pd.DataFrame(rows).T.to_string())
