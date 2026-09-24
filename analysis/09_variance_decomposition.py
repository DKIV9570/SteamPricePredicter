"""Where does the variation in discounting come from?

Nested variance decomposition of discount outcomes:
  publisher  -> differences between publishers' average behaviour
  game       -> differences between games of the same publisher
  sale       -> differences between sales of the same game (at similar age)
Only publishers with >= 5 games, so publisher means are estimable.

    python analysis/09_variance_decomposition.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
P = ROOT / "data" / "processed"
g = pd.read_parquet(P / "model_table.parquet")
sales = pd.read_parquet(P / "sales.parquet")


def two_level(d: pd.DataFrame, y: str, group: str) -> dict:
    """Share of variance between groups vs within groups (unbiased: subtracts the
    sampling noise of group means)."""
    d = d.dropna(subset=[y, group])
    tot = d[y].var()
    st = d.groupby(group)[y].agg(["mean", "var", "count"])
    within = (st["var"] * (st["count"] - 1)).sum() / (st["count"] - 1).sum()
    n_bar = st["count"].mean()
    between = max(0.0, st["mean"].var() - within / n_bar)
    return {"between": between / (between + within), "within": within / (between + within), "n": len(d),
            "groups": len(st), "total_var": tot}


big = g.groupby("publisher")["appid"].transform("count") >= 5
gb = g[big].copy()
gb["log_days"] = np.log(gb["days_to_first_sale"])

rows = {}
for y, label in [("first_sale_cut", "首次打折幅度"), ("log_days", "首次打折时间(log)"),
                 ("launch_cut", "首发折扣幅度")]:
    r = two_level(gb, y, "publisher")
    rows[label] = {"发行商之间": r["between"], "同发行商·不同游戏": r["within"], "游戏数": r["n"], "发行商数": r["groups"]}

# Depth reached by age A (the reach model's core quantity)
tl_sales = sales[sales["kind"] == "regular"].merge(g[["appid", "publisher", "launch_price"]], on="appid")
for a_lo, a_hi, label in [(0, 182, "半年内最深折扣"), (182, 365, "半年~1年最深折扣"), (365, 730, "第2年最深折扣")]:
    s = tl_sales[(tl_sales.days_from_release >= a_lo) & (tl_sales.days_from_release < a_hi)]
    per_game = s.groupby(["appid", "publisher"], as_index=False)["max_cut"].max()
    per_game = per_game[per_game["publisher"].isin(gb["publisher"].unique())]
    r = two_level(per_game, "max_cut", "publisher")
    rows[label] = {"发行商之间": r["between"], "同发行商·不同游戏": r["within"], "游戏数": r["n"], "发行商数": r["groups"]}
res = pd.DataFrame(rows).T

# Third level: sale-to-sale variation within the same game, at similar age (year 2)
s2 = tl_sales[(tl_sales.days_from_release >= 365) & (tl_sales.days_from_release < 730)
              & tl_sales["publisher"].isin(gb["publisher"].unique())]
s2 = s2[s2.groupby("appid")["max_cut"].transform("count") >= 3]
lvl_sale = two_level(s2, "max_cut", "appid")  # between games vs within game
game_means = s2.groupby(["appid", "publisher"], as_index=False)["max_cut"].mean()
lvl_pub = two_level(game_means, "max_cut", "publisher")
tot = s2["max_cut"].var()
v_sale = lvl_sale["within"] * tot
v_game_total = tot - v_sale
three = {"发行商之间": lvl_pub["between"] * v_game_total / tot,
         "同发行商·不同游戏": lvl_pub["within"] * v_game_total / tot,
         "同一游戏·不同次打折": v_sale / tot}

pd.set_option("display.width", 200)
print("=== 两层分解：差异来自发行商之间，还是同一发行商的不同游戏之间 ===")
print(res.assign(**{c: res[c].map("{:.0%}".format) for c in ["发行商之间", "同发行商·不同游戏"]}).to_string())
print(f"\n=== 三层分解：第 2 年的每次打折幅度（{s2.appid.nunique()} 款游戏，{len(s2)} 次打折）===")
print({k: f"{v:.0%}" for k, v in three.items()})
