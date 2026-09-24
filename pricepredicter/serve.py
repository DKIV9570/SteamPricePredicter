"""Shared inference, used by the CLI (predict.py) and the website export (export_site.py),
so both compute features exactly the way the model was trained.

    art = load_artifacts()
    ref = load_reference()
    feats = game_features(ref, art)                      # all games, or only={appid}
    rows = predict_now(feats, ref, art, now, thresholds)  # one row per game x target x window
"""
import json
import pickle
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import MODELS, PROCESSED
from .features import build_game_table
from .reach import (CUT_TOL, VELOCITY, attach_features, best_cut_by_age, expand_rows, landmark_states,
                    price_timeline, static_extras, velocity_features)
from .sale_calendar import all_majors
from .similarity import game_knn_features, publisher_knn_features

ART = MODELS
N_EMB_FEATS = 16
PREDICT_CHUNK = 500_000


@dataclass
class Artifacts:
    model: lgb.Booster
    features: list
    static: list
    itad_tags: list
    vectorizer: object


def load_artifacts() -> Artifacts:
    meta = json.loads((ART / "features.json").read_text(encoding="utf-8"))
    return Artifacts(lgb.Booster(model_file=str(ART / "model.txt")), meta["features"], meta["static"],
                     meta["itad_tags"], pickle.loads((ART / "vectorizer.pkl").read_bytes()))


def load_reference() -> dict:
    ref = {n: pd.read_parquet(PROCESSED / f"{f}.parquet") for n, f in
           [("games", "games"), ("sales", "sales"), ("changes", "price_changes"), ("content", "content"),
            ("tags", "tags"), ("months", "review_months")]}
    ref["games"] = ref["games"][ref["games"]["usable"]]
    ref["majors"] = all_majors(ref["sales"])
    return ref


def game_features(ref: dict, art: Artifacts, only: set | None = None) -> dict:
    """Per-game features as of each game's launch, built the same way as the training table.
    only: compute the expensive per-game parts just for these appids (live queries)."""
    g, _ = build_game_table(ref["games"], ref["sales"], majors=ref["majors"], tag_list=art.itad_tags,
                            only_appids=only)
    g = g.reset_index(drop=True)
    pos = None if only is None else np.flatnonzero(g["appid"].isin(only))
    g = g.merge(ref["content"].drop(columns=["description", "spy_genre", "store_genres"], errors="ignore"),
                on="appid", how="left")
    g["log_m1_reviews"] = np.log1p(g["m1_reviews_up"] + g["m1_reviews_down"])
    g["m1_score"] = g["m1_reviews_up"] / (g["m1_reviews_up"] + g["m1_reviews_down"]).replace(0, np.nan)

    vec = art.vectorizer.transform(g, ref["tags"])
    for i in range(N_EMB_FEATS):
        g[f"emb_{i}"] = vec[:, i]
    g = pd.concat([g, game_knn_features(g, vec, rows=pos), publisher_knn_features(g, vec, rows=pos)], axis=1)
    tl = price_timeline(g, ref["changes"])
    extra = static_extras(g, best_cut_by_age(g, tl), vec, rows=pos)
    static = g[["appid"] + art.static].merge(extra, on="appid")
    if only is not None:
        g, static = g[g["appid"].isin(only)], static[static["appid"].isin(only)]
        tl = tl[tl["appid"].isin(only)]
    return {"g": g, "tl": tl, "static": static}


def predict_now(feats: dict, ref: dict, art: Artifacts, now: pd.Timestamp, thresholds) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Probabilities that each game's price is at >= X% off (vs launch price) within each window,
    asked at `now` (US/Pacific, tz-naive). Returns (rows, states). Targets the price already
    meets are absent from rows."""
    g, tl = feats["g"], feats["tl"]
    sales = ref["sales"][ref["sales"]["appid"].isin(g["appid"])]
    states = landmark_states(g, tl, sales, now, at=now)
    rows = expand_rows(states, tl, ref["majors"], now, thresholds=thresholds, predict=True)
    if rows.empty:
        return pd.DataFrame(columns=["appid", "threshold", "horizon", "horizon_days", "p"]), states
    rows = attach_features(rows, feats["static"], ref["content"])
    if any(f in art.features for f in VELOCITY):
        vel = velocity_features(states.merge(g[["appid", "t0"]], on="appid"), ref["months"], now)
        rows = rows.merge(vel, on=["appid", "L"], how="left")
    rows["p"] = np.concatenate([art.model.predict(rows[art.features].iloc[i:i + PREDICT_CHUNK].astype(np.float32))
                                for i in range(0, len(rows), PREDICT_CHUNK)])
    # A longer window can't be less likely; a deeper cut can't be more likely
    rows = rows.sort_values(["appid", "threshold", "horizon_days"])
    rows["p"] = rows.groupby(["appid", "threshold"])["p"].cummax()
    rows = rows.sort_values(["appid", "horizon", "threshold"])
    rows["p"] = rows.groupby(["appid", "horizon"], observed=True)["p"].cummin()
    return rows[["appid", "threshold", "horizon", "horizon_days", "p"]], states


def upcoming_majors(majors: pd.DataFrame, now: pd.Timestamp, n: int = 2) -> pd.DataFrame:
    return majors[majors["start"] > now].sort_values("start").head(n)


__all__ = ["CUT_TOL", "Artifacts", "load_artifacts", "load_reference", "game_features", "predict_now",
           "upcoming_majors"]
