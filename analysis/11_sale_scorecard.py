"""Real-world check: predictions saved before a major sale vs. what actually happened.

Takes a site-data snapshot (site/data as exported before the sale) and current raw ITAD
price histories, and scores the "by the end of the next major sale" probabilities:
did the price reach >= X% off (vs launch price) between the snapshot and --until?

    python analysis/11_sale_scorecard.py --snapshot <site/data dir> --itad <raw/itad dir> [--until 2026-10-09]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

sys.stdout.reconfigure(encoding="utf-8")
ap = argparse.ArgumentParser()
ap.add_argument("--snapshot", required=True)
ap.add_argument("--itad", required=True)
ap.add_argument("--window", default="next_major")
ap.add_argument("--until", default=None, help="end of the observed period (UTC date); default now")
args = ap.parse_args()

snap = Path(args.snapshot)
meta = json.loads((snap / "meta.json").read_text(encoding="utf-8"))
t0 = pd.Timestamp(meta["updated"])
sale = meta["sales"][0]
sale_end = pd.Timestamp(sale["start"], tz="US/Pacific") + pd.Timedelta(days=14)
until = min(pd.Timestamp(args.until, tz="UTC") if args.until else pd.Timestamp.now(tz="UTC"), sale_end)
w = meta["windows"].index(args.window)
popular = {row[0] for row in json.loads((snap / "index.json").read_text(encoding="utf-8"))[:2000]}

rows = []
for shard in (snap / "g").glob("*.json"):
    for appid, rec in json.loads(shard.read_text(encoding="utf-8")).items():
        path = Path(args.itad) / f"{appid}.json"
        if not path.exists():
            continue
        hist = json.loads(path.read_text(encoding="utf-8"))["history"]
        prices = [h["deal"]["price"]["amount"] for h in hist
                  if t0 < pd.Timestamp(h["timestamp"]) <= until]
        low = min(prices) if prices else np.inf
        for cut, p in zip(meta["cuts"], rec["p"]):
            if p is None or p[w] is None:
                continue  # price was already at the target when predicted
            rows.append({"appid": int(appid), "lp": rec["lp"], "cut": cut, "p": p[w] / 100,
                         "reached": int(low <= rec["lp"] * (1 - (cut - 1) / 100) + 1e-6),
                         "popular": int(appid) in popular, "never_disc": rec["ns"] == 0})
d = pd.DataFrame(rows)
print(f"snapshot {t0:%Y-%m-%d %H:%M}Z -> {sale['name']} sale from {sale['start']}, outcomes observed until "
      f"{until:%Y-%m-%d %H:%M}Z; {d.appid.nunique():,} games, {len(d):,} (game, target) predictions\n")


def score(x):
    pred = x.p >= 0.5
    return pd.Series({
        "预测平均": x.p.mean(), "实际发生率": x.reached.mean(),
        "AUC": roc_auc_score(x.reached, x.p) if x.reached.nunique() > 1 else np.nan,
        "Brier": brier_score_loss(x.reached, x.p),
        "判断准确率": np.mean(pred == x.reached),
        "说'会'时真发生": x.reached[pred].mean() if pred.any() else np.nan,
        "说'不会'时真没发生": 1 - x.reached[~pred].mean() if (~pred).any() else np.nan,
        "n": len(x)})


pd.set_option("display.width", 220)
groups = {"全部": d, "首发价≥$20": d[d.lp >= 19.99], "最受关注的 2000 款": d[d.popular],
          "还没打过折的游戏": d[d.never_disc], "目标 5 折": d[d.cut == 50], "目标 8 折": d[d.cut == 20]}
res = pd.DataFrame({k: score(v) for k, v in groups.items()}).T
print(res.to_string(formatters={c: "{:.1%}".format for c in ["预测平均", "实际发生率", "判断准确率",
                                                             "说'会'时真发生", "说'不会'时真没发生"]}
                    | {"AUC": "{:.3f}".format, "Brier": "{:.3f}".format, "n": "{:,.0f}".format}))

d["bin"] = pd.cut(d.p, np.linspace(0, 1, 11))
print("\n校准（预测概率 vs 实际发生比例）:")
print(d.groupby("bin", observed=True).agg(预测=("p", "mean"), 实际=("reached", "mean"), n=("p", "size"))
      .to_string(formatters={"预测": "{:.1%}".format, "实际": "{:.1%}".format}))
