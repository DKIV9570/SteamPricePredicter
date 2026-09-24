"""Shared model-table assembly, training and evaluation.

    table = load_model_table()               # cached in data/processed/model_table.parquet
    res = run_experiment(table, features)    # timing + depth metrics by publisher group
"""
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import PROCESSED
from .features import build_game_table, build_panel
from .sale_calendar import all_majors
from .similarity import game_knn_features, game_vectors, publisher_knn_features

TRAIN_END, VALID_END = pd.Timestamp("2022-07-01"), pd.Timestamp("2023-07-01")
TABLE = PROCESSED / "model_table.parquet"
N_EMB_FEATS = 16  # leading SVD dims fed to the model directly

TIME = ["week", "major_now", "major_name", "majors_passed", "days_to_next_major"]
BASE = ["log_price", "had_launch_disc", "launch_cut", "early_access", "late_to_steam",
        "release_month", "release_year", "self_published"]


def load_model_table(rebuild: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    games = pd.read_parquet(PROCESSED / "games.parquet")
    sales = pd.read_parquet(PROCESSED / "sales.parquet")
    data_end = sales["start"].max().tz_convert("US/Pacific").tz_localize(None).floor("D")
    if TABLE.exists() and not rebuild:
        return pd.read_parquet(TABLE), all_majors(sales), data_end

    g, majors = build_game_table(games, sales)
    g = g.reset_index(drop=True)
    content = pd.read_parquet(PROCESSED / "content.parquet")
    g = g.merge(content.drop(columns=["description", "spy_genre", "store_genres"]), on="appid", how="left")
    g["log_m1_reviews"] = np.log1p(g["m1_reviews_up"] + g["m1_reviews_down"])
    g["m1_score"] = g["m1_reviews_up"] / (g["m1_reviews_up"] + g["m1_reviews_down"]).replace(0, np.nan)

    vec = game_vectors(g, pd.read_parquet(PROCESSED / "tags.parquet"))
    for i in range(N_EMB_FEATS):
        g[f"emb_{i}"] = vec[:, i]
    g = pd.concat([g, game_knn_features(g, vec), publisher_knn_features(g, vec)], axis=1)
    g["split"] = np.where(g["t0"] < TRAIN_END, "train", np.where(g["t0"] < VALID_END, "valid", "test"))
    g["pub_group"] = pd.cut(g["pub_n_known"].fillna(0), [-1, 0, 2, 1e9],
                            labels=["新发行商(0)", "少量(1-2)", "成熟(3+)"]).astype(str)
    g.to_parquet(TABLE, index=False)
    return g, majors, data_end


def feature_sets(g: pd.DataFrame) -> dict[str, list[str]]:
    track = [c for c in g.columns if c.startswith(("pub_", "dev_")) and c != "pub_group"]
    itad_tags = [c for c in g.columns if c.startswith("tag_")]
    content = (["n_languages", "metacritic", "required_age"]
               + [c for c in g.columns if c.startswith("cat_")]
               + [f"emb_{i}" for i in range(N_EMB_FEATS)])
    gknn = [c for c in g.columns if c.startswith("knn_")]
    pknn = [c for c in g.columns if c.startswith("pknn_")]
    v1 = BASE + track + itad_tags
    return {
        "A v1 基线": v1,
        "B +内容": v1 + content,
        "C +相似游戏": v1 + content + gknn,
        "D +相似发行商": v1 + content + gknn + pknn,
        "E +首月评测": v1 + content + gknn + pknn + ["log_m1_reviews", "m1_score"],
    }


@dataclass
class Panels:
    train: pd.DataFrame
    valid: pd.DataFrame
    test_full: pd.DataFrame


def make_panels(g, majors, data_end) -> Panels:
    p = build_panel(g, majors, data_end).merge(g[["appid", "split"]], on="appid")
    test = g[g["split"] == "test"]
    return Panels(p[p.split == "train"], p[p.split == "valid"],
                  build_panel(test, majors, data_end, full=True))


def _fit(params, Xtr, ytr, Xva, yva, rounds=3000, stop=150):
    return lgb.train(params, lgb.Dataset(Xtr, ytr), rounds, valid_sets=[lgb.Dataset(Xva, yva)],
                     callbacks=[lgb.early_stopping(stop, verbose=False)])


HAZARD_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_child_samples=200,
                     feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1)
DEPTH_PARAMS = dict(objective="l1", learning_rate=0.03, num_leaves=31, min_child_samples=50,
                    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1)


def run_experiment(g: pd.DataFrame, panels: Panels, feats: list[str]) -> dict:
    gs = g.set_index("appid")
    X = lambda p: pd.concat([p[TIME].reset_index(drop=True),  # noqa: E731
                             gs.loc[p["appid"], feats].reset_index(drop=True)], axis=1)

    # ---- timing ----
    hz = _fit(HAZARD_PARAMS, X(panels.train), panels.train["event"].to_numpy(),
              X(panels.valid), panels.valid["event"].to_numpy(), 2000, 100)
    full = panels.test_full.copy()
    full["h"] = hz.predict(X(full), num_iteration=hz.best_iteration)
    full["surv"] = full.groupby("appid")["h"].transform(lambda h: (1 - h).cumprod())
    full["pmf"] = full["h"] * full.groupby("appid")["surv"].shift(1).fillna(1.0)
    pred_week = full.loc[full.groupby("appid")["pmf"].idxmax()].set_index("appid")["week"]

    test = g[g["split"] == "test"].copy()
    test["pred_days"] = test["appid"].map(pred_week * 7 + 3.5)
    truth = test.set_index("appid")["days_to_first_sale"] // 7
    near = (full["week"] - full["appid"].map(truth)).abs() <= 1
    test["mass"] = test["appid"].map(full[near].groupby("appid")["pmf"].sum()).fillna(0)
    obs = test.dropna(subset=["days_to_first_sale"])
    err = (obs["pred_days"] - obs["days_to_first_sale"]).abs()

    # ---- depth ----
    g2 = g.assign(sale_week=g["days_to_first_sale"] // 7, sale_in_major=g["first_sale_in_major"].astype(float))
    dx = feats + ["sale_week", "sale_in_major"]
    d = g2.dropna(subset=["first_sale_cut"])
    dtr, dva, dte = (d[d.split == s] for s in ["train", "valid", "test"])
    dm = _fit(DEPTH_PARAMS, dtr[dx], dtr["first_sale_cut"], dva[dx], dva["first_sale_cut"])
    dte = dte.assign(pred_cut=dm.predict(dte[dx], num_iteration=dm.best_iteration))
    derr = (dte["pred_cut"] - dte["first_sale_cut"]).abs()

    # End-to-end, as a user would get it: depth predicted from the PREDICTED sale week
    pw = full.loc[full.groupby("appid")["pmf"].idxmax()].set_index("appid")
    e2e = test.assign(sale_week=test["appid"].map(pw["week"]),
                      sale_in_major=test["appid"].map(pw["major_now"]).astype(float))
    test["pred_cut_e2e"] = dm.predict(e2e[dx], num_iteration=dm.best_iteration)

    def by_group(s, frame):
        r = {"全部": s.mean()}
        r.update(s.groupby(frame.loc[s.index, "pub_group"]).mean().to_dict())
        return r

    return {
        "test": test,
        "timing_hit7": by_group(err <= 7, obs),
        "timing_mass": by_group(obs["mass"], obs),
        "depth_mae": by_group(derr, dte),
        "models": (hz, dm),
        "n": {"timing": obs["pub_group"].value_counts().to_dict(), "depth": dte["pub_group"].value_counts().to_dict()},
    }
