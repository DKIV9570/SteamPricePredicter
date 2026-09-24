"""Accuracy of the one-line verdict the user actually sees (test set).

Verdict = earliest window (next major / 3 / 6 / 12 months) with p >= 0.5, else "buy now".
  "wait until H" is right if the target price appeared within H;
  "buy now"      is right if it did not appear within a year.

    python analysis/06_verdict_accuracy.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
te = pd.read_parquet(ROOT / "data" / "processed" / "reach_test_preds.parquet")
te["horizon"] = te["horizon"].astype(str)
ORDER = ["next_major", "90", "180", "365"]


def verdicts(d: pd.DataFrame, col: str) -> pd.DataFrame:
    w = d.pivot_table(index=["appid", "lw", "threshold"], columns="horizon", values=[col, "reached"])
    p, y = w[col].reindex(columns=ORDER), w["reached"].reindex(columns=ORDER)
    p = p.cummax(axis=1)  # same monotone fix as the live tool
    first = (p >= 0.5).to_numpy()
    idx = np.where(first.any(1), first.argmax(1), -1)
    out = pd.DataFrame(index=w.index)
    out["verdict"] = np.where(idx >= 0, np.array(ORDER)[np.clip(idx, 0, 3)], "buy")
    yv = y.to_numpy()
    # label for the chosen window (NaN if that window was censored)
    chosen = np.where(idx >= 0, yv[np.arange(len(yv)), np.clip(idx, 0, 3)], 1 - yv[:, 3])
    out["correct"] = chosen
    return out.dropna(subset=["correct"])


meta = te.groupby(["appid", "lw", "threshold"]).agg(
    pub_group=("pub_group", "first"), waitlisted=("waitlisted", "first"), price=("log_price", "first"),
    fresh=("n_sales_so_far", "first"), launch=("had_launch_disc", "first"))
meta["price"] = np.expm1(meta["price"])
meta["fresh"] = meta["fresh"] <= meta["launch"]  # no regular sale yet at the moment of asking
tiers = {"全部价位": 0, "≥$10": 10, "≥$15": 15, "≥$20": 20}
out = {}
for name, col in [("查表", "p_base"), ("模型", "p")]:
    v = verdicts(te, col).join(meta)
    for tier, lo in tiers.items():
        x = v[v["price"] >= lo - 0.01]
        w = x["waitlisted"].fillna(0)
        out[(tier, name)] = {
            "全部": x.correct.mean(),
            "还没打过折的游戏": x.correct[x.fresh].mean(),
            "已打过折的游戏": x.correct[~x.fresh].mean(),
            "按关注度加权": (x.correct * w).sum() / w.sum(),
            "新发行商": x.correct[x.pub_group == "新发行商(0)"].mean(),
            "游戏数": x.index.get_level_values("appid").nunique(),
            "查询数": len(x),
        }
res = pd.DataFrame(out)
pd.set_option("display.width", 220)
print(res.map(lambda x: f"{x:.1%}" if x <= 1 else f"{x:,.0f}").to_string())
