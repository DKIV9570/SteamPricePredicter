"""Build the (game, landmark, threshold, horizon) training table for the reach model.
Shared by analysis/05_reach_model.py and the diagnostics."""
import numpy as np
import pandas as pd

from .config import PROCESSED
from .modeling import feature_sets, load_model_table
from .reach import (EARLY_REVIEWS, STATE, TASK, VELOCITY, add_major_participation, attach_features,
                    best_cut_by_age, expand_rows, landmark_states, price_timeline, regular_sale_depths,
                    static_extras, velocity_features)
from .similarity import GameVectorizer


def build_reach_dataset():
    """Returns rows, feature list, and pieces the artifacts need (g, vectorizer, best)."""
    g, majors, data_end = load_model_table()
    tags = pd.read_parquet(PROCESSED / "tags.parquet")
    vz = GameVectorizer().fit(g, tags)
    vec = vz.transform(g, tags)
    tl = price_timeline(g, pd.read_parquet(PROCESSED / "price_changes.parquet"))
    best = best_cut_by_age(g, tl)
    extra = static_extras(g, best, vec)
    sales = pd.read_parquet(PROCESSED / "sales.parquet")
    states = add_major_participation(landmark_states(g, tl, sales, data_end), g, sales, majors)
    rows = expand_rows(states, tl, majors, data_end, hist=regular_sale_depths(g, sales))

    static_cols = feature_sets(g)["E +首月评测"]
    age = [c for c in extra.columns if c != "appid"]
    feats = TASK + STATE + age + static_cols + EARLY_REVIEWS + VELOCITY
    static = g[["appid", "split", "pub_group", "waitlisted", "had_launch_disc"]
               + [c for c in static_cols if c != "had_launch_disc"]].merge(extra, on="appid")
    rows = attach_features(rows, static, pd.read_parquet(PROCESSED / "content.parquet"))
    vel = velocity_features(states.merge(g[["appid", "t0"]], on="appid"),
                            pd.read_parquet(PROCESSED / "review_months.parquet"), data_end)
    rows = rows.merge(vel, on=["appid", "L"], how="left")
    num = [f for f in feats if rows[f].dtype.kind in "fiub"]
    rows[num] = rows[num].astype(np.float32)
    return rows, feats, {"g": g, "vectorizer": vz, "best": best, "static": static_cols}
