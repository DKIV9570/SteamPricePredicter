"""Game and publisher similarity features.

Game vector:  TF-IDF over SteamSpy tag votes -> 64-d SVD, L2-normalised, plus launch price.
Game kNN:     for each game, the K most similar games whose first-sale outcome was already
              known at its launch; their outcomes, similarity-weighted, become features.
Publisher kNN: publishers are profiled by the mean vector of their games. A game's own
              publisher profile (its prior games + itself) is matched against established
              publishers as of Jan 1 of the launch year, borrowing their track records.
              Works for brand-new publishers too: their profile is just the game itself.
"""
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize

EMB_DIM = 64
PRICE_WEIGHT = 0.5  # how much launch price counts next to the (unit-length) tag vector
CHUNK = 2000


class GameVectorizer:
    """Fitted on the reference games; transform() places any game (e.g. a live query)
    in the same space. Games without tags get a zero tag vector (price only)."""

    def fit(self, g: pd.DataFrame, tags: pd.DataFrame) -> "GameVectorizer":
        t = tags[tags["appid"].isin(g["appid"])]
        self.vocab = pd.Index(sorted(t["tag"].unique()))
        self.tfidf = TfidfTransformer().fit(self._counts(g, t))
        self.svd = TruncatedSVD(EMB_DIM, random_state=0).fit(self.tfidf.transform(self._counts(g, t)))
        lp = np.log1p(g["launch_price"].to_numpy(float))
        self.price_mean, self.price_std = lp.mean(), lp.std()
        return self

    def _counts(self, g, tags):
        idx = pd.Index(g["appid"])
        t = tags[tags["appid"].isin(idx) & tags["tag"].isin(self.vocab)]
        return sparse.csr_matrix((np.log1p(t["votes"].to_numpy(float)),
                                  (idx.get_indexer(t["appid"]), self.vocab.get_indexer(t["tag"]))),
                                 shape=(len(g), len(self.vocab)))

    def transform(self, g: pd.DataFrame, tags: pd.DataFrame) -> np.ndarray:
        emb = normalize(self.svd.transform(self.tfidf.transform(self._counts(g, tags))))
        price = (np.log1p(g["launch_price"].to_numpy(float)) - self.price_mean) / self.price_std
        return np.hstack([emb, PRICE_WEIGHT * price[:, None]]).astype(np.float32)


def game_vectors(g: pd.DataFrame, tags: pd.DataFrame) -> np.ndarray:
    """Rows aligned with g."""
    return GameVectorizer().fit(g, tags).transform(g, tags)


def _outcomes(g):
    days = g["days_to_first_sale"].to_numpy(float)
    known_at = g["first_sale_day"].fillna(g["t0"] + pd.Timedelta(days=365)).to_numpy()
    return {
        "log_days": np.where(np.isnan(days), np.log(365), np.log(np.maximum(days, 1))),
        "cut": g["first_sale_cut"].to_numpy(float),
        "major": g["first_sale_in_major"].astype(float).to_numpy(),
        "never1y": (np.isnan(days) | (days > 365)).astype(float),
        "launch_disc": g["had_launch_disc"].to_numpy(float),
    }, known_at


def _wavg(vals, w):
    m = ~np.isnan(vals)
    ws = (w * m).sum(1)
    return np.where(ws > 0, np.nansum(vals * w, 1) / np.where(ws > 0, ws, 1), np.nan)


def game_knn_features(g: pd.DataFrame, vec: np.ndarray, k: int = 20, rows=None) -> pd.DataFrame:
    """rows: positions to compute (default all); others are left NaN."""
    out, known_at = _outcomes(g)
    t0 = g["t0"].to_numpy()
    rows = np.arange(len(g)) if rows is None else np.asarray(rows)
    # Cosine-like similarity; the price component makes it "same kind of game at a similar price"
    vn = normalize(vec)
    feats = {f"knn_{n}": np.full(len(g), np.nan) for n in out}
    feats["knn_sim_mean"] = np.full(len(g), np.nan)
    for s in range(0, len(rows), CHUNK):
        r = rows[s:s + CHUNK]
        sim = vn[r] @ vn.T
        valid = known_at[None, :] < t0[r, None]  # outcome visible at launch
        sim = np.where(valid, sim, -np.inf)
        top = np.argpartition(-sim, k, axis=1)[:, :k]
        ts = np.take_along_axis(sim, top, 1)
        w = np.where(np.isfinite(ts), np.clip(ts, 0, None) + 1e-6, 0.0)
        for n, v in out.items():
            feats[f"knn_{n}"][r] = _wavg(v[top], w)
        feats["knn_sim_mean"][r] = np.where(np.isfinite(ts), ts, np.nan).mean(1)
    return pd.DataFrame(feats, index=g.index)


def _own_publisher_profiles(g: pd.DataFrame, vec: np.ndarray) -> np.ndarray:
    """Per game: mean vector of its publisher's games launched up to and including it.
    Content only (no outcomes), so using same-day siblings leaks nothing."""
    prof = vec.copy()
    t0 = g["t0"].to_numpy()
    keys = pd.Series(g["publisher"].fillna("").to_numpy())
    for key, ix in keys.groupby(keys).groups.items():
        ix = np.asarray(ix)
        if key == "" or len(ix) < 2:
            continue
        ix = ix[np.argsort(t0[ix], kind="stable")]
        prof[ix] = np.cumsum(vec[ix], 0) / np.arange(1, len(ix) + 1)[:, None]
    return normalize(prof)


PKNN_OUTCOMES = ["log_days", "cut", "major", "never1y"]


def publisher_knn_features(g: pd.DataFrame, vec: np.ndarray, k: int = 10,
                           min_games: int = 3, rows=None) -> pd.DataFrame:
    """rows: positions to compute (default all); others are left NaN."""
    out, known_at = _outcomes(g)
    t0 = pd.to_datetime(g["t0"])
    pub = g["publisher"].fillna("__none__").to_numpy()
    own_all = _own_publisher_profiles(g, vec)
    feats = {f"pknn_{n}": np.full(len(g), np.nan) for n in PKNN_OUTCOMES}
    feats["pknn_sim_mean"] = np.full(len(g), np.nan)
    wanted = np.zeros(len(g), bool)
    wanted[np.arange(len(g)) if rows is None else np.asarray(rows)] = True

    for year in sorted(t0[wanted].dt.year.unique()):
        cutoff = np.datetime64(pd.Timestamp(f"{year}-01-01"))
        known = known_at < cutoff
        # Established publishers as of the cutoff: profile + track record
        kp = pd.DataFrame({"pub": pub[known], "i": np.flatnonzero(known)})
        kp = kp[kp["pub"] != "__none__"]
        cnt = kp.groupby("pub")["i"].count()
        est = cnt[cnt >= min_games].index
        if len(est) < k:
            continue
        kp = kp[kp["pub"].isin(est)]
        grp = kp.groupby("pub")["i"].apply(np.array)
        prof = normalize(np.vstack([vec[ix].mean(0) for ix in grp]))
        rec = {n: np.array([np.nanmean(out[n][ix]) for ix in grp]) for n in PKNN_OUTCOMES}

        rows = np.flatnonzero((t0.dt.year == year).to_numpy() & wanted)
        sim = own_all[rows] @ prof.T
        # never match a game to its own publisher: that's the plain track-record feature's job
        self_idx = pd.Index(grp.index).get_indexer(pub[rows])
        has_self = self_idx >= 0
        sim[np.flatnonzero(has_self), self_idx[has_self]] = -np.inf
        top = np.argpartition(-sim, k, axis=1)[:, :k]
        ts = np.take_along_axis(sim, top, 1)
        w = np.clip(ts, 0, None) + 1e-6
        for n, v in rec.items():
            feats[f"pknn_{n}"][rows] = _wavg(v[top], w)
        feats["pknn_sim_mean"][rows] = ts.mean(1)
    return pd.DataFrame(feats, index=g.index)
