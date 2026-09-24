"""The user-facing question: "will the price reach my target discount within H?"

Discounts are measured against the LAUNCH price (1 - price / launch_price), so
permanent price cuts count too — that's what matters to a buyer.

Rows are (game, landmark, threshold, horizon):
  landmark  the moment a user asks (0 = launch week, 13 = three months in, ...)
  threshold target discount X (%), unless the price already meets it at the landmark
  horizon   end of the next major sale / 90 / 180 / 365 days after the landmark
  label     reached = price is at >= X% off at some point within (landmark, landmark + horizon],
            whether for the first time or again (reached_before tells the two apart)
Rows whose horizon runs past the end of the data without reaching X are censored and dropped.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

THRESHOLDS = [20, 30, 40, 50, 60, 70, 80]
LANDMARK_WEEKS = [0, 1, 2, 6, 13, 26, 39, 52, 78, 104, 156]
FIXED_HORIZONS = [90, 180, 365]
MAJOR_SALE_DAYS = 14  # "by the end of the next major sale" ~ its start + two weeks
AGES_WEEKS = [13, 26, 52, 104]  # for "how deep did it get by age A" track records
CUT_TOL = 1.0  # 49.5% off counts as 50%
LANDMARK_OFFSET_DAYS = 1


STATE = ["lw", "best_cut_so_far", "price_ratio_now", "regular_ratio_now", "on_sale_now", "n_sales_so_far",
         "last_sale_cut", "prev_cut", "cut_trend", "weeks_since_sale", "sales_since_deeper"]
TASK = ["threshold", "gap_to_target", "reached_before", "horizon_days", "majors_in_horizon", "days_to_next_major"]
EARLY_REVIEWS = ["log_w1_reviews", "w1_score"]
MONOTONE = {"threshold": -1, "gap_to_target": -1, "horizon_days": 1, "majors_in_horizon": 1}


def attach_features(rows: pd.DataFrame, static: pd.DataFrame, content: pd.DataFrame) -> pd.DataFrame:
    """Join per-game features onto (game, landmark, threshold, horizon) rows, hiding
    review counts whose window hadn't closed yet at the landmark."""
    w1 = content[["appid"]].assign(
        log_w1_reviews=np.log1p(content["w1_reviews"]) if "w1_reviews" in content else np.nan,
        w1_score=(content["w1_positive"] / content["w1_reviews"].replace(0, np.nan))
        if "w1_reviews" in content else np.nan)
    rows = rows.merge(static, on="appid").merge(w1, on="appid", how="left")
    rows.loc[rows["lw"] * 7 < 30, ["log_m1_reviews", "m1_score"]] = np.nan
    rows.loc[rows["lw"] < 1, EARLY_REVIEWS] = np.nan
    return rows


def static_extras(g: pd.DataFrame, best: pd.DataFrame, vec: np.ndarray, rows=None) -> pd.DataFrame:
    """Age-based track records (publisher/developer) + similar games' depth by age.
    rows: positions in g to compute (default all)."""
    only = None if rows is None else set(g["appid"].iloc[rows])
    extra = (g[["appid"]]
             .merge(age_track_record(g, best, "publisher", "pub", only), on="appid", how="left")
             .merge(age_track_record(g, best, "developer", "dev", only), on="appid", how="left"))
    return pd.concat([extra, knn_age_features(g, best, vec, rows=rows)], axis=1)


VELOCITY = ["rv_rate_last", "rv_slope", "rv_accel", "rv_decay", "rv_from_peak", "rv_log_cum"]


def velocity_features(keys: pd.DataFrame, months: pd.DataFrame, data_end: pd.Timestamp) -> pd.DataFrame:
    """Sales-curve proxy from monthly review counts, using only months that had fully ended
    before each moment L (the month containing L would include future reviews).

    keys: appid, t0, L.  Rates are reviews/day; slopes are in log space, so they mean
    "how fast is the curve falling" regardless of how big the game is:
      rv_rate_last  log review rate in the last complete month (current sales level)
      rv_slope      change vs the month before (1st derivative)
      rv_accel      change of that change (2nd derivative: is the decline speeding up?)
      rv_decay      last month vs launch month (how much launch heat is left)
      rv_from_peak  last month vs the best month so far
      rv_log_cum    total reviews so far (size)
    """
    first = keys.groupby("appid")["t0"].min()
    m = months[months["appid"].isin(first.index)]
    # Zero-filled month grid from launch month to the end of the data
    start = first.dt.to_period("M")
    n = (pd.Period(data_end, "M") - start).map(lambda x: x.n) + 1
    grid = pd.DataFrame({"appid": np.repeat(first.index, n.clip(lower=1))})
    grid["k"] = grid.groupby("appid").cumcount()
    grid["month"] = (start.reindex(grid["appid"]).to_numpy() + grid["k"].to_numpy())
    grid["month"] = pd.PeriodIndex(grid["month"]).to_timestamp()
    grid = grid.merge(m, on=["appid", "month"], how="left").fillna({"reviews": 0})
    t0 = grid["appid"].map(first)
    month_end = grid["month"] + pd.offsets.MonthBegin(1)
    days = np.where(grid["k"] == 0, (month_end - t0).dt.days.clip(lower=1), grid["month"].dt.days_in_month)
    lr = np.log1p(grid["reviews"] / days)
    grp = lr.groupby(grid["appid"])
    grid["rv_rate_last"] = lr
    grid["rv_slope"] = lr - grp.shift(1)
    grid["rv_accel"] = grid["rv_slope"] - grid["rv_slope"].groupby(grid["appid"]).shift(1)
    grid["rv_decay"] = lr - grp.transform("first")
    grid["rv_from_peak"] = lr - grp.cummax()
    grid["rv_log_cum"] = np.log1p(grid.groupby("appid")["reviews"].cumsum())

    k = keys[["appid", "L"]].drop_duplicates().copy()
    k["month"] = (k["L"].dt.to_period("M") - 1).dt.to_timestamp()  # last fully ended month
    return k.merge(grid[["appid", "month"] + VELOCITY], on=["appid", "month"], how="left").drop(columns="month")


def price_timeline(g: pd.DataFrame, changes: pd.DataFrame) -> pd.DataFrame:
    c = changes[changes["appid"].isin(g["appid"])].merge(
        g[["appid", "t0", "launch_price"]], on="appid")
    c["ts"] = c["ts"].dt.tz_convert("US/Pacific").dt.tz_localize(None)
    c = c[c["ts"] >= c["t0"] - pd.Timedelta(days=1)].sort_values(["appid", "ts"])
    c["eff_cut"] = (1 - c["price"] / c["launch_price"]) * 100
    c["best_cut"] = c.groupby("appid")["eff_cut"].cummax()
    return c


def best_cut_by_age(g: pd.DataFrame, tl: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"appid": g["appid"]})
    for a in AGES_WEEKS:
        m = tl[tl["ts"] < tl["t0"] + pd.Timedelta(weeks=a)].groupby("appid")["eff_cut"].max()
        out[f"best_{a}w"] = out["appid"].map(m).fillna(0).clip(lower=0)
    return out


def age_track_record(g: pd.DataFrame, best: pd.DataFrame, key: str, prefix: str,
                     only_appids=None) -> pd.DataFrame:
    """Mean depth reached by age A over earlier games of the same publisher/developer,
    counting only games that were already A old at this game's launch."""
    d = g[["appid", key, "t0"]].merge(best, on="appid")
    d = d.dropna(subset=[key])
    if only_appids is not None:
        d = d[d[key].isin(d.loc[d["appid"].isin(only_appids), key])]
    res = []
    for _, grp in d.groupby(key):
        grp = grp.sort_values("t0")
        t0 = grp["t0"].to_numpy()
        vals = {a: grp[f"best_{a}w"].to_numpy() for a in AGES_WEEKS}
        for i in range(len(grp)):
            if only_appids is not None and grp["appid"].iat[i] not in only_appids:
                continue
            row = {"appid": grp["appid"].iat[i]}
            for a in AGES_WEEKS:
                ok = t0 + np.timedelta64(a * 7, "D") < t0[i]
                row[f"{prefix}_best_{a}w"] = vals[a][ok].mean() if ok.any() else np.nan
            res.append(row)
    return pd.DataFrame(res, columns=["appid"] + [f"{prefix}_best_{a}w" for a in AGES_WEEKS])


def knn_age_features(g: pd.DataFrame, best: pd.DataFrame, vec: np.ndarray, k: int = 20,
                     chunk: int = 2000, rows=None) -> pd.DataFrame:
    """Similarity-weighted depth reached by age A among the K most similar games that
    were already A old at this game's launch. rows: positions to compute (default all)."""
    vn = normalize(vec)
    t0 = g["t0"].to_numpy("datetime64[ns]")
    b = g[["appid"]].merge(best, on="appid", how="left")
    rows = np.arange(len(g)) if rows is None else np.asarray(rows)
    feats = {f"knn_best_{a}w": np.full(len(g), np.nan) for a in AGES_WEEKS}
    for s0 in range(0, len(rows), chunk):
        s = rows[s0:s0 + chunk]
        sim = vn[s] @ vn.T
        for a in AGES_WEEKS:
            valid = (t0 + np.timedelta64(a * 7, "D"))[None, :] < t0[s, None]
            sa = np.where(valid, sim, -np.inf)
            top = np.argpartition(-sa, k, axis=1)[:, :k]
            ts = np.take_along_axis(sa, top, 1)
            w = np.where(np.isfinite(ts), np.clip(ts, 0, None) + 1e-6, 0.0)
            v = b[f"best_{a}w"].to_numpy()[top]
            ws = w.sum(1)
            feats[f"knn_best_{a}w"][s] = np.where(ws > 0, (v * w).sum(1) / np.where(ws > 0, ws, 1), np.nan)
    return pd.DataFrame(feats, index=g.index)


def landmark_states(g: pd.DataFrame, tl: pd.DataFrame, sales: pd.DataFrame,
                    data_end: pd.Timestamp, weeks=LANDMARK_WEEKS, at: pd.Timestamp | None = None) -> pd.DataFrame:
    """Price history summary strictly before each landmark.
    at: a single arbitrary moment (live prediction) instead of the fixed landmark weeks."""
    if at is None:
        lm = g[["appid", "t0", "launch_price"]].merge(pd.DataFrame({"lw": weeks}), how="cross")
        # +1 day: a launch discount is already visible to anyone looking at launch
        lm["L"] = lm["t0"] + pd.to_timedelta(lm["lw"] * 7 + LANDMARK_OFFSET_DAYS, unit="D")
        lm = lm[lm["L"] < data_end]
    else:
        lm = g[["appid", "t0", "launch_price"]].assign(L=at)
        lm["lw"] = ((lm["L"] - lm["t0"]).dt.total_seconds() / 86400 - LANDMARK_OFFSET_DAYS).clip(lower=0) / 7
    lm["L"] = lm["L"].astype("datetime64[ns]")  # merge_asof needs matching resolutions
    lm = lm.sort_values("L")

    st = tl[["appid", "ts", "price", "regular", "best_cut"]].sort_values("ts")
    lm = pd.merge_asof(lm, st, left_on="L", right_on="ts", by="appid",
                       direction="backward", allow_exact_matches=False)
    lm["best_cut_so_far"] = lm["best_cut"].fillna(0).clip(lower=0)
    lm["price_ratio_now"] = (lm["price"] / lm["launch_price"]).fillna(1.0)
    lm["regular_ratio_now"] = (lm["regular"] / lm["launch_price"]).fillna(1.0)
    lm["on_sale_now"] = (lm["price"] < lm["regular"] - 0.01).astype(int)

    s = sales[sales["appid"].isin(g["appid"])][["appid", "start", "max_cut"]].copy()
    s["start"] = s["start"].dt.tz_convert("US/Pacific").dt.tz_localize(None)
    s = s.sort_values(["appid", "start"])
    s["n_sales"] = s.groupby("appid").cumcount() + 1
    s["prev_cut"] = s.groupby("appid")["max_cut"].shift(1)
    s["first_cut"] = s.groupby("appid")["max_cut"].transform("first")
    # Plateau: how many sales in a row have failed to beat the deepest cut so far
    run_best = s.groupby("appid")["max_cut"].cummax()
    new_best = run_best > run_best.groupby(s["appid"]).shift(1).fillna(-1)
    s["sales_since_deeper"] = s.groupby(["appid", new_best.cumsum()]).cumcount()
    lm = pd.merge_asof(lm.drop(columns=["ts"]), s.sort_values("start"), left_on="L", right_on="start",
                       by="appid", direction="backward", allow_exact_matches=False)
    lm["n_sales_so_far"] = lm["n_sales"].fillna(0)
    lm["last_sale_cut"] = lm["max_cut"]
    lm["cut_trend"] = lm["max_cut"] - lm["first_cut"]  # deepening so far?
    lm["weeks_since_sale"] = (lm["L"] - lm["start"]).dt.days / 7
    keep = ["appid", "lw", "L", "best_cut_so_far", "price_ratio_now", "regular_ratio_now", "on_sale_now",
            "n_sales_so_far", "last_sale_cut", "prev_cut", "cut_trend", "weeks_since_sale", "sales_since_deeper"]
    return lm[keep]


def expand_rows(states: pd.DataFrame, tl: pd.DataFrame, majors: pd.DataFrame, data_end: pd.Timestamp,
                thresholds=THRESHOLDS, predict: bool = False) -> pd.DataFrame:
    """(landmark state) x thresholds x horizons, labelled with whether the price is at
    >= X% off at some point within (L, L + horizon] — first time or again.
    Thresholds the price already meets at L are skipped (the answer is "buy now").
    predict=True keeps unobserved future windows (no labels)."""
    st = np.sort(majors["start"].to_numpy("datetime64[ns]"))
    x = states.sort_values("L").reset_index(drop=True)
    L = x["L"].to_numpy("datetime64[ns]")
    i = np.searchsorted(st, L, side="right")
    nxt = np.where(i < len(st), st[np.minimum(i, len(st) - 1)], np.datetime64("NaT"))
    x["days_to_next_major"] = (nxt - L) / np.timedelta64(1, "D")
    end_days = (np.datetime64(data_end) - L) / np.timedelta64(1, "D")
    cut_now = (1 - x["price_ratio_now"].to_numpy()) * 100

    horizons = [("next_major", x["days_to_next_major"].to_numpy() + MAJOR_SALE_DAYS)] + \
               [(str(h), np.full(len(x), float(h))) for h in FIXED_HORIZONS]
    if predict:
        # "by the end of the major sale after next" — a window length the model already
        # covers through horizon_days / majors_in_horizon
        j = np.minimum(i + 1, len(st) - 1)
        second = np.where(i + 1 < len(st), (st[j] - L) / np.timedelta64(1, "D"), np.nan)
        horizons.append(("second_major", second + MAJOR_SALE_DAYS))
    base_cols = list(states.columns) + ["days_to_next_major"]
    parts = []
    for thr in thresholds:
        hits = tl.loc[tl["eff_cut"] >= thr - CUT_TOL, ["appid", "ts"]].sort_values("ts")
        nh = pd.merge_asof(x[["appid", "L"]], hits, left_on="L", right_on="ts", by="appid",
                           direction="forward", allow_exact_matches=False)["ts"]
        days_to_hit = ((nh - x["L"]).dt.total_seconds() / 86400).to_numpy()
        open_ = cut_now < thr - CUT_TOL
        for hname, hd in horizons:
            reached = days_to_hit <= hd
            m = open_ & ~np.isnan(hd) & (predict | reached | (hd <= end_days))
            if not m.any():
                continue
            p = x.loc[m, base_cols].copy()
            p["threshold"] = thr
            p["horizon"] = hname
            p["horizon_days"] = hd[m]
            # how many major sales fall inside the window
            lo = np.searchsorted(st, L[m], side="right")
            hi = np.searchsorted(st, L[m] + (hd[m] * 86400e9).astype("timedelta64[ns]"), side="right")
            p["majors_in_horizon"] = hi - lo
            p["gap_to_target"] = thr - p["best_cut_so_far"]
            p["reached_before"] = (p["best_cut_so_far"] >= thr - CUT_TOL).astype(int)
            if not predict:
                p["reached"] = reached[m].astype(int)
            parts.append(p)
    if not parts:  # every target already met (or none asked)
        return pd.DataFrame(columns=base_cols + ["threshold", "horizon", "horizon_days", "majors_in_horizon",
                                                 "gap_to_target", "reached_before"])
    out = pd.concat(parts, ignore_index=True)
    out["horizon"] = pd.Categorical(out["horizon"], ["next_major", "second_major"] + [str(h) for h in FIXED_HORIZONS])
    return out
