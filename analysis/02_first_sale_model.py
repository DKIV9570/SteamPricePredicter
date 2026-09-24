"""v1 (no similarity features yet): first-sale timing + depth.

Timing: discrete-time hazard model (LightGBM on the game x week panel).
Depth:  LightGBM regression on the first sale's cut.
Both compared against rule baselines, overall and for cold-start publishers.

    python analysis/02_first_sale_model.py
"""
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pricepredicter.features import FIRST_WEEK, MAX_WEEKS, build_game_table, build_panel  # noqa: E402

P = ROOT / "data" / "processed"
TRAIN_END, VALID_END = pd.Timestamp("2022-07-01"), pd.Timestamp("2023-07-01")

games = pd.read_parquet(P / "games.parquet")
sales = pd.read_parquet(P / "sales.parquet")
g, majors = build_game_table(games, sales)
data_end = sales["start"].max().tz_convert("US/Pacific").tz_localize(None).floor("D")
g["split"] = np.where(g["t0"] < TRAIN_END, "train", np.where(g["t0"] < VALID_END, "valid", "test"))
g["pub_group"] = pd.cut(g["pub_n_known"].fillna(0), [-1, 0, 2, 1e9], labels=["新发行商(0)", "少量(1-2)", "成熟(3+)"])
print(g["split"].value_counts().to_dict())

STATIC = (["log_price", "had_launch_disc", "launch_cut", "early_access", "late_to_steam",
           "release_month", "release_year", "self_published"]
          + [c for c in g.columns if c.startswith(("tag_", "pub_", "dev_")) and c != "pub_group"])
# Review count/score are TODAY's values, unknown at launch -> leak. Opt-in for ablation only.
if "--with-reviews" in sys.argv:
    STATIC += ["log_reviews", "steam_score"]
TIME = ["week", "major_now", "major_name", "majors_passed", "days_to_next_major"]

# ---------------- timing: hazard model ----------------
panel = build_panel(g, majors, data_end).merge(g[["appid", "split"] + STATIC], on="appid")
print(f"panel rows: {len(panel):,}")
X = lambda d: d[TIME + STATIC]  # noqa: E731
tr, va = panel[panel.split == "train"], panel[panel.split == "valid"]
params = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_child_samples=100,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1)
hz = lgb.train(params, lgb.Dataset(X(tr), tr.event), 2000, valid_sets=[lgb.Dataset(X(va), va.event)],
               callbacks=[lgb.early_stopping(100, verbose=False)])
print(f"hazard model: {hz.best_iteration} trees")

test = g[g.split == "test"].copy()
full = build_panel(test, majors, data_end, full=True).merge(test[["appid"] + STATIC], on="appid")
full["h"] = hz.predict(X(full), num_iteration=hz.best_iteration)
full["surv"] = full.groupby("appid")["h"].transform(lambda h: (1 - h).cumprod())
# P(first sale happens in week w) = P(no sale before w) * hazard(w)
full["pmf"] = full["h"] * full.groupby("appid")["surv"].shift(1).fillna(1.0)
# Point prediction = most likely week. (The week where the CDF crosses 50% tends to land
# between two majors, i.e. on a week when almost nothing starts a sale.)
pred_week = full.loc[full.groupby("appid")["pmf"].idxmax()].set_index("appid")["week"]
test["pred_days_model"] = test["appid"].map(pred_week * 7 + 3.5)
# Probabilistic score: mass the model put within +-1 week of the true first-sale week
truth = test.set_index("appid")["days_to_first_sale"] // 7
full["near"] = (full["week"] - full["appid"].map(truth)).abs() <= 1
mass = full[full.near].groupby("appid")["pmf"].sum()

# Baseline A: the first major sale starting at least 30 days after launch
st = np.sort(majors["start"].to_numpy())
def next_major_days(t0, min_days):  # noqa: E302
    i = np.searchsorted(st, np.datetime64(t0 + pd.Timedelta(days=min_days)))
    return (st[i] - np.datetime64(t0)) / np.timedelta64(1, "D") if i < len(st) else np.nan
test["pred_days_rule"] = test["t0"].map(lambda t: next_major_days(t, 30))
# Baseline B: publisher's typical wait, snapped to the next major after it
test["pred_days_pub"] = [next_major_days(t, max(14, np.exp(m) - 3)) if not np.isnan(m) else r
                         for t, m, r in zip(test.t0, test.pub_log_days_mean, test.pred_days_rule)]

obs = test.dropna(subset=["days_to_first_sale"])
def timing_report(d):  # noqa: E302
    out = {}
    for k in ["rule", "pub", "model"]:
        err = (d[f"pred_days_{k}"] - d["days_to_first_sale"]).abs()
        out[k] = f"±7天命中 {np.mean(err <= 7):.1%} | 中位误差 {err.median():.0f}天"
    return pd.Series(out)
print("\n=== 首次打折时间（测试集，已打折的游戏）===")
print(pd.DataFrame({"全部": timing_report(obs)} |
                   {str(k): timing_report(v) for k, v in obs.groupby("pub_group", observed=True)}).T.to_string())
print("样本数:", obs.groupby("pub_group", observed=True).size().to_dict())
print(f"模型给真实首折周 ±1 周的平均概率: {obs.appid.map(mass).fillna(0).mean():.1%}")

# ---------------- depth: first-sale cut ----------------
g["sale_week"] = g["days_to_first_sale"] // 7
g["sale_in_major"] = g["first_sale_in_major"].astype(float)
DEPTH_X = STATIC + ["sale_week", "sale_in_major"]
d = g.dropna(subset=["first_sale_cut"])
dtr, dva, dte = (d[d.split == s] for s in ["train", "valid", "test"])
dp = dict(objective="l1", learning_rate=0.03, num_leaves=31, min_child_samples=50,
          feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1)
dm = lgb.train(dp, lgb.Dataset(dtr[DEPTH_X], dtr.first_sale_cut), 3000,
               valid_sets=[lgb.Dataset(dva[DEPTH_X], dva.first_sale_cut)],
               callbacks=[lgb.early_stopping(150, verbose=False)])
dte = dte.copy()
dte["pred_model"] = dm.predict(dte[DEPTH_X], num_iteration=dm.best_iteration)
tier_median = dtr.groupby(pd.cut(dtr.launch_price, [0, 5, 10, 20, 30, 45, 1000]), observed=True).first_sale_cut.median()
dte["pred_rule"] = pd.cut(dte.launch_price, [0, 5, 10, 20, 30, 45, 1000]).map(tier_median).astype(float)
dte["pred_pub"] = dte["pub_cut_mean"].fillna(dte["pred_rule"])

def depth_report(x):  # noqa: E302
    return pd.Series({k: f"MAE {np.mean(np.abs(x[f'pred_{k}'] - x.first_sale_cut)):.1f} 个百分点"
                      for k in ["rule", "pub", "model"]})
print("\n=== 首次折扣幅度（测试集）===")
print(pd.DataFrame({"全部": depth_report(dte)} |
                   {str(k): depth_report(v) for k, v in dte.groupby("pub_group", observed=True)}).T.to_string())

imp = pd.Series(hz.feature_importance("gain"), index=hz.feature_name()).sort_values(ascending=False)
print("\n时间模型 top 特征:", (imp / imp.sum()).head(12).round(3).to_dict())
imp = pd.Series(dm.feature_importance("gain"), index=dm.feature_name()).sort_values(ascending=False)
print("折扣模型 top 特征:", (imp / imp.sum()).head(12).round(3).to_dict())
