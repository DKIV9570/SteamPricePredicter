"""EDA: is discount strategy driven by publisher?

Outputs figures to reports/figures/ and prints how much of the first-sale
timing / depth each single factor explains (out-of-fold target encoding).

    python analysis/01_publisher_eda.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parent.parent
P = ROOT / "data" / "processed"
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# Reference palette (dataviz skill): light surface, recessive chrome
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
    "font.family": ["Segoe UI", "Microsoft YaHei", "sans-serif"],
})

games = pd.read_parquet(P / "games.parquet")
sales = pd.read_parquet(P / "sales.parquet")
u = games[games["usable"]].copy()
reg = sales[(sales["kind"] == "regular") & sales["appid"].isin(u["appid"])].merge(
    u[["appid", "publisher"]], on="appid")
reg["age_m"] = reg["days_from_release"] / 30.44

# ---------- Figure 1: discount depth vs game age, per publisher ----------
PUBS = ["capcom", "ubisoft", "electronic arts", "bandai namco", "square enix", "sega",
        "koei tecmo", "sony", "xbox", "paradox", "devolver", "thq nordic"]
EDGES = [0, 3, 6, 9, 12, 18, 24, 36, 48, 72]
MIDS = [(a + b) / 2 for a, b in zip(EDGES[:-1], EDGES[1:])]


def median_curve(df):
    b = pd.cut(df["age_m"], EDGES, labels=MIDS)
    return df.groupby(b, observed=True)["max_cut"].median()


overall = median_curve(reg)
fig, axes = plt.subplots(3, 4, figsize=(13, 8.5), sharex=True, sharey=True)
for ax, pub in zip(axes.flat, PUBS):
    d = reg[reg["publisher"] == pub]
    n_games = d["appid"].nunique()
    ax.scatter(d["age_m"].clip(upper=72), d["max_cut"], s=6, color=BLUE, alpha=0.12, linewidths=0)
    ax.plot(overall.index.astype(float), overall.values, color=MUTED, lw=1.5, ls="--")
    c = median_curve(d)
    ax.plot(c.index.astype(float), c.values, color=BLUE, lw=2, marker="o", ms=4)
    ax.set_title(f"{pub}  ·  {n_games} games", loc="left", fontsize=10, color=INK)
    ax.set_xlim(0, 72); ax.set_ylim(0, 100)
    ax.set_xticks([0, 12, 24, 36, 48, 60, 72])
for ax in axes[-1]:
    ax.set_xlabel("发售后月数")
for ax in axes[:, 0]:
    ax.set_ylabel("折扣 %")
fig.suptitle("各发行商：折扣深度随游戏年龄的变化", x=0.01, ha="left", fontsize=13, color=INK)
fig.text(0.01, 0.945, "点 = 每次打折（不含首发折扣）  蓝线 = 该发行商中位数  灰虚线 = 全体游戏中位数",
         fontsize=9, color=INK2)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(FIG / "publisher_discount_curves.png", dpi=130)
plt.close(fig)

# ---------- Figure 2: when does the first sale happen (calendar) ----------
fs = u.dropna(subset=["first_sale_date"])
fig, ax = plt.subplots(figsize=(11, 3.6))
doy = fs["first_sale_date"].dt.dayofyear
ax.hist(doy, bins=range(1, 368, 3), color=BLUE, edgecolor=SURFACE, linewidth=0.5)
top = ax.get_ylim()[1]
for d0, name in [(76, "春促"), (176, "夏促"), (300, "万圣节"), (330, "秋促"), (355, "冬促")]:
    ax.annotate(name, (d0, top * 0.97), ha="center", va="top", color=INK2, fontsize=9)
ax.set_xticks([1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335],
              ["1月", "2月", "3月", "4月", "5月", "6月", "7月", "8月", "9月", "10月", "11月", "12月"])
ax.set_ylabel("游戏数")
ax.set_title(f"首次打折落在一年中的哪一天（{len(fs):,} 款游戏，3 天一格）", loc="left", fontsize=12, color=INK)
fig.tight_layout()
fig.savefig(FIG / "first_sale_calendar.png", dpi=130)
plt.close(fig)

# ---------- Single-factor explanatory power ----------
u = u.dropna(subset=["first_sale_date"]).copy()
u["log_days"] = np.log(u["days_to_first_sale"])
u["price_tier"] = pd.cut(u["launch_price"], [0, 5, 10, 15, 20, 30, 40, 50, 60, 1000]).astype(str)
u["top_tag"] = u["tags"].str.split("|").str[0].fillna("none")
u["release_month"] = u["steam_release"].dt.month.astype(str)
u["release_year"] = u["steam_release"].dt.year.astype(str)
u["had_launch_disc"] = u["launch_cut"].notna().astype(str)
u["late_to_steam"] = (u["days_late_to_steam"] > 30).astype(str)


def oof_r2(df, col, target, k=20):
    """R² of a smoothed out-of-fold mean encoding — honest for high-cardinality columns."""
    pred = np.zeros(len(df))
    for tr, te in KFold(5, shuffle=True, random_state=0).split(df):
        t = df.iloc[tr]
        gm = t[target].mean()
        st = t.groupby(col)[target].agg(["mean", "count"])
        enc = (st["mean"] * st["count"] + gm * k) / (st["count"] + k)
        pred[te] = df.iloc[te][col].map(enc).fillna(gm).to_numpy()
    y = df[target].to_numpy()
    return 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()


cols = ["publisher", "developer", "price_tier", "top_tag", "release_month", "release_year",
        "had_launch_disc", "late_to_steam"]
res = pd.DataFrame({t: {c: oof_r2(u, c, t) for c in cols} for t in ["log_days", "first_sale_cut"]})
res.columns = ["R² 首次打折时间(log天)", "R² 首次折扣幅度"]
print(res.sort_values(res.columns[0], ascending=False).round(3).to_string())

# publisher effect restricted to publishers with a track record
big = u.groupby("publisher")["appid"].transform("count") >= 10
print(f"\n只看有 >=10 款游戏的发行商 ({big.sum()} 款游戏):")
for t in ["log_days", "first_sale_cut"]:
    print(f"  publisher R² on {t}: {oof_r2(u[big], 'publisher', t):.3f}")
print(f"游戏数 <10 的发行商占比: {(~big).mean():.1%}")
