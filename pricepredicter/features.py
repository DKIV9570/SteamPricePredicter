"""Model inputs.

build_game_table   one row per usable game: static features known at Steam launch,
                   plus publisher/developer track records computed ONLY from games
                   whose outcome was already visible at this game's launch.
build_panel        discrete-time survival panel: one row per game per week since
                   launch, until its first regular sale (event) or censoring.
"""
import numpy as np
import pandas as pd

from .build_dataset import LAUNCH_WINDOW_DAYS
from .sale_calendar import all_majors

MAX_WEEKS = 156  # model the first 3 years
FIRST_WEEK = LAUNCH_WINDOW_DAYS // 7  # sales inside the launch window aren't "first sales"
N_TAGS = 60


def _pacific_day(ts: pd.Series) -> pd.Series:
    return ts.dt.tz_convert("US/Pacific").dt.tz_localize(None).dt.floor("D")


def _track_record(df: pd.DataFrame, key: str, prefix: str, only_appids=None) -> pd.DataFrame:
    """Per game: stats over earlier games of the same publisher/developer whose
    first sale (or 1 year without one) had already happened by this game's launch.
    only_appids: compute just these games (their publishers' histories are still used)."""
    out = []
    df = df.dropna(subset=[key])
    if only_appids is not None:
        df = df[df[key].isin(df.loc[df["appid"].isin(only_appids), key])]
    for _, g in df.groupby(key):
        g = g.sort_values("t0")
        t0 = g["t0"].to_numpy()
        known_at = g["first_sale_day"].fillna(g["t0"] + pd.Timedelta(days=365)).to_numpy()
        days = g["days_to_first_sale"].to_numpy()
        cut = g["first_sale_cut"].to_numpy()
        in_major = g["first_sale_in_major"].to_numpy()
        launch = g["launch_cut"].notna().to_numpy()
        for i in range(len(g)):
            if only_appids is not None and g["appid"].iat[i] not in only_appids:
                continue
            prior = t0 < t0[i]
            known = prior & (known_at < t0[i])
            k_d = known & ~np.isnan(days)
            out.append({
                "appid": g["appid"].iat[i],
                f"{prefix}_n_prior": int(prior.sum()),
                f"{prefix}_n_known": int(known.sum()),
                f"{prefix}_log_days_mean": np.log(days[k_d]).mean() if k_d.any() else np.nan,
                f"{prefix}_cut_mean": cut[k_d].mean() if k_d.any() else np.nan,
                f"{prefix}_cut_max": cut[k_d].max() if k_d.any() else np.nan,
                f"{prefix}_major_share": in_major[k_d].mean() if k_d.any() else np.nan,
                f"{prefix}_never_share": 1 - k_d.sum() / known.sum() if known.any() else np.nan,
                f"{prefix}_launch_disc_share": launch[prior].mean() if prior.any() else np.nan,
            })
    cols = ["appid"] + [f"{prefix}_{c}" for c in ["n_prior", "n_known", "log_days_mean", "cut_mean",
                                                  "cut_max", "major_share", "never_share", "launch_disc_share"]]
    return pd.DataFrame(out, columns=cols)


def build_game_table(games: pd.DataFrame, sales: pd.DataFrame, majors: pd.DataFrame | None = None,
                     tag_list=None, only_appids=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """tag_list: reuse the reference tag columns (live prediction); only_appids: compute
    publisher/developer track records just for these games."""
    majors = all_majors(sales) if majors is None else majors
    g = games[games["usable"]].copy()
    g["t0"] = _pacific_day(g["steam_release"])
    g["first_sale_day"] = _pacific_day(g["first_sale_date"])
    st = majors["start"].to_numpy()
    lag = g["first_sale_day"].to_numpy()[:, None] - st[None, :]
    g["first_sale_in_major"] = ((lag >= np.timedelta64(0, "D")) & (lag <= np.timedelta64(2, "D"))).any(axis=1)
    g.loc[g["first_sale_day"].isna(), "first_sale_in_major"] = np.nan

    g["log_price"] = np.log1p(g["launch_price"])
    g["had_launch_disc"] = g["launch_cut"].notna().astype(int)
    g["launch_cut"] = g["launch_cut"].fillna(0)
    g["early_access"] = g["early_access"].astype(float)
    g["late_to_steam"] = g["days_late_to_steam"].clip(lower=0)
    g["release_month"] = g["t0"].dt.month
    g["release_year"] = g["t0"].dt.year
    g["log_reviews"] = np.log1p(g["steam_reviews"])  # NB: today's count, not launch-time

    tags = g["tags"].str.split("|")
    top = tags.explode().value_counts().head(N_TAGS).index if tag_list is None else tag_list
    for t in top:
        g[f"tag_{t}"] = tags.map(lambda ts, t=t: int(t in ts))

    for key, prefix in [("publisher", "pub"), ("developer", "dev")]:
        g = g.merge(_track_record(g, key, prefix, only_appids), on="appid", how="left")
    g[["pub_n_prior", "dev_n_prior"]] = g[["pub_n_prior", "dev_n_prior"]].fillna(0)
    g["self_published"] = (g["publisher"] == g["developer"]).astype(int)
    return g, majors


def build_panel(g: pd.DataFrame, majors: pd.DataFrame, data_end: pd.Timestamp,
                full: bool = False) -> pd.DataFrame:
    """One row per (game, week) from FIRST_WEEK until first sale or censoring.
    full=True keeps every week up to MAX_WEEKS (for predicting whole survival curves)."""
    m = majors.sort_values("start")
    st = m["start"].to_numpy("datetime64[ns]")
    names = np.append(m["name"].to_numpy(object), "none")
    t0 = g["t0"].to_numpy("datetime64[ns]")
    days = g["days_to_first_sale"].to_numpy(float)
    event_week = np.where(np.isnan(days), 10**9, np.floor(np.nan_to_num(days) / 7)).astype(np.int64)
    if full:
        last = np.full(len(g), MAX_WEEKS - 1)
    else:
        obs_weeks = ((np.datetime64(data_end) - t0) / np.timedelta64(7, "D")).astype(np.int64)
        last = np.minimum.reduce([np.full(len(g), MAX_WEEKS - 1), obs_weeks - 1, event_week])
    n = np.clip(last - FIRST_WEEK + 1, 0, None)

    gi = np.repeat(np.arange(len(g)), n)
    week = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n) + FIRST_WEEK
    ws = t0[gi] + week.astype("timedelta64[D]") * 7
    i = np.searchsorted(st, ws)
    ic = np.minimum(i, len(st) - 1)
    major_now = (i < len(st)) & (st[ic] < ws + np.timedelta64(7, "D"))
    j = i + major_now
    nxt_days = np.where(j < len(st), (st[np.minimum(j, len(st) - 1)] - ws) / np.timedelta64(1, "D"), np.nan)
    passed = i - np.searchsorted(st, t0 + np.timedelta64(LAUNCH_WINDOW_DAYS, "D"))[gi]
    p = pd.DataFrame({
        "appid": g["appid"].to_numpy()[gi], "week": week, "major_now": major_now.astype(int),
        "major_name": pd.Categorical(np.where(major_now, names[ic], "none")),
        "majors_passed": passed, "days_to_next_major": nxt_days,
        "event": (week == event_week[gi]).astype(int)})
    return p
