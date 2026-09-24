"""Step 3: turn raw JSON into analysis tables in data/processed/.

games.parquet          one row per app: metadata + first-sale target
price_changes.parquet  ITAD price-change log (Steam, USD)
sales.parquet          contiguous discount periods merged from the log

    python -m pricepredicter.build_dataset
"""
import json
import re

import pandas as pd

from .config import ITAD_RAW, PROCESSED, RAW
from .rawio import iter_json

# ITAD only started tracking around 2012; earlier releases have no usable first-sale info
MIN_RELEASE = "2014-01-01"
# Sales starting this close to release count as launch discounts, not the "first real sale"
LAUNCH_WINDOW_DAYS = 14

# Stripped only from the END of a name, repeatedly: legal forms, regions, generic words.
# "SA Industry" keeps its leading SA; "Ubisoft Entertainment" -> "ubisoft".
_TRAILING = {
    "co", "corp", "corporation", "inc", "incorporated", "ltd", "limited", "llc", "gmbh", "sa", "s a",
    "ab", "plc", "kk", "k k", "srl", "sl", "bv", "pty", "sas", "sro", "s r o", "oy", "as", "spa",
    "usa", "u s a", "us", "america", "europe", "eu", "japan", "jp", "asia", "uk",
    "entertainment", "interactive", "games", "game", "studios", "studio", "publishing", "digital",
}
_TRAILING_RE = re.compile(r"\s(" + "|".join(sorted(map(re.escape, _TRAILING), key=len, reverse=True)) + r")$")
# Same company under different names that suffix-stripping can't merge
_ALIASES = {"warner bros": "warner bros", "wb": "warner bros", "playstation": "sony",
            "sony interactive": "sony", "xbox game": "xbox", "microsoft": "xbox"}


def normalize_publisher(name: str | None) -> str | None:
    """'CAPCOM CO. LTD' / 'Capcom (JP)' / 'Ubisoft Entertainment' -> 'capcom' / 'capcom' / 'ubisoft'."""
    if not name:
        return None
    s = re.sub(r"\(.*?\)", " ", name.lower())
    s = re.sub(r"[^\w\s&]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    while True:
        t = _TRAILING_RE.sub("", s).strip()
        if t == s or not t:
            break
        s = t
    return _ALIASES.get(s, s) or None


def primary_publisher(steamspy_pub: str | None, itad_pubs: list) -> str | None:
    # SteamSpy mirrors the Steam store page, which is who actually sets the price;
    # ITAD merges names from several sources, so it is only a fallback
    if steamspy_pub:
        return normalize_publisher(steamspy_pub.split(",")[0])
    if itad_pubs:
        return normalize_publisher(itad_pubs[0]["name"])
    return None


def merge_sales(changes: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    """Collapse consecutive cut>0 log entries into sale periods (per app)."""
    rows = []
    for appid, g in changes.sort_values(["appid", "ts"]).groupby("appid", sort=False):
        ts, cut, regular = g["ts"].tolist(), g["cut"].tolist(), g["regular"].tolist()
        i = 0
        while i < len(ts):
            if cut[i] <= 0:
                i += 1
                continue
            j = i
            while j + 1 < len(ts) and cut[j + 1] > 0:
                j += 1
            end = ts[j + 1] if j + 1 < len(ts) else pd.NaT
            rows.append({"appid": appid, "start": ts[i], "end": end,
                         "ongoing": j + 1 >= len(ts),
                         "max_cut": max(cut[i:j + 1]), "first_cut": cut[i],
                         "regular": regular[i]})
            i = j + 1
    # Explicit columns/dtypes so a game that has never been discounted yields an empty but usable table
    sales = pd.DataFrame(rows, columns=["appid", "start", "end", "ongoing", "max_cut", "first_cut", "regular"])
    sales = sales.astype({"appid": "int64", "ongoing": bool, "max_cut": "int64", "first_cut": "int64",
                          "regular": float})
    for c in ["start", "end"]:
        sales[c] = pd.to_datetime(sales[c], utc=True)
    sales["duration_days"] = ((sales["end"].fillna(now) - sales["start"]).dt.total_seconds() / 86400)
    return sales


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    catalog = json.loads((RAW / "catalog.json").read_text(encoding="utf-8"))
    names_path = RAW / "names.json"
    steam_release = {int(k): v["release"] for k, v in json.loads(names_path.read_text(encoding="utf-8")).items()
                     if v.get("release")} if names_path.exists() else {}
    records = (rec for _, rec in iter_json(ITAD_RAW))
    games, changes, sales = process_records(((r, catalog.get(str(r["appid"]), {})) for r in records),
                                            release_fallback=steam_release)

    games.to_parquet(PROCESSED / "games.parquet", index=False)
    changes.to_parquet(PROCESSED / "price_changes.parquet", index=False)
    sales.to_parquet(PROCESSED / "sales.parquet", index=False)

    u = games[games["usable"]]
    print(f"games: {len(games)}  usable: {len(u)}  sales periods: {len(sales)}")
    print(f"usable & discounted: {(~u['never_discounted']).sum()}  never discounted (censored): {u['never_discounted'].sum()}")
    print(f"distinct publishers (usable): {u['publisher'].nunique()}")


def process_records(pairs, now: pd.Timestamp | None = None, release_fallback: dict | None = None):
    """(ITAD record, SteamSpy app dict) pairs -> (games, changes, sales) tables.
    `now` is the observation time (defaults to the newest price change seen).
    release_fallback: {appid: unix time} Steam release dates for games ITAD has none for."""
    games, changes = [], []
    for rec, spy in pairs:
        info, appid = rec["info"], rec["appid"]
        steam_rev = next((r for r in info.get("reviews") or [] if r["source"] == "Steam"), {})
        games.append({
            "appid": appid,
            "title": info.get("title"),
            "type": info.get("type"),
            "release_date": info.get("releaseDate"),
            "early_access": info.get("earlyAccess"),
            "tags": "|".join(info.get("tags") or []),
            "publisher": primary_publisher(spy.get("publisher"), info.get("publishers") or []),
            "publisher_raw": spy.get("publisher"),
            "developer": normalize_publisher((spy.get("developer") or "").split(",")[0]),
            "steam_score": steam_rev.get("score"),
            "steam_reviews": steam_rev.get("count"),
            "waitlisted": (info.get("stats") or {}).get("waitlisted"),
            "owners_band": spy.get("owners"),
        })
        for h in rec["history"]:
            d = h["deal"]
            changes.append({"appid": appid, "ts": h["timestamp"], "price": d["price"]["amount"],
                            "regular": d["regular"]["amount"], "cut": d["cut"]})

    games = pd.DataFrame(games)
    games["release_date"] = pd.to_datetime(games["release_date"], errors="coerce", utc=True)
    if release_fallback:
        fb = pd.to_datetime(games["appid"].map(release_fallback), unit="s", utc=True)
        games["release_date"] = games["release_date"].fillna(fb)
    changes = pd.DataFrame(changes)
    changes["ts"] = pd.to_datetime(changes["ts"], utc=True, format="ISO8601")
    now = changes["ts"].max() if now is None else now
    sales = merge_sales(changes, now)

    # --- first-sale target per game ---
    first_seen = changes.groupby("appid")["ts"].min().rename("history_start")
    games = games.merge(first_seen, on="appid", how="left")
    # ITAD's release date is the game's first release on any platform. Epic exclusives,
    # PlayStation ports etc. reach Steam months later, and a Steam price log that starts
    # after the release date marks exactly that moment (pre-orders start it earlier).
    games["steam_release"] = games[["release_date", "history_start"]].max(axis=1)
    games["days_late_to_steam"] = (games["steam_release"] - games["release_date"]).dt.days
    s = sales.merge(games[["appid", "steam_release"]], on="appid")
    s["days_from_release"] = (s["start"] - s["steam_release"]).dt.total_seconds() / 86400
    s["kind"] = pd.cut(s["days_from_release"], [-1e9, -1, LAUNCH_WINDOW_DAYS, 1e9],
                       labels=["prerelease", "launch", "regular"])
    sales = sales.merge(s[["appid", "start", "days_from_release", "kind"]], on=["appid", "start"])

    launch = s[s["kind"] == "launch"].groupby("appid")["max_cut"].max().rename("launch_cut")
    first = (s[s["kind"] == "regular"].sort_values("start").groupby("appid").first()
             [["start", "days_from_release", "max_cut"]]
             .rename(columns={"start": "first_sale_date", "days_from_release": "days_to_first_sale",
                              "max_cut": "first_sale_cut"}))
    games = games.merge(launch, on="appid", how="left").merge(first, on="appid", how="left")

    # Regular price in effect at launch: the last change up to launch day (a pre-order price
    # often simply carries over), else the first change after it
    c = changes.merge(games[["appid", "steam_release"]], on="appid").sort_values("ts")
    upto = c[c["ts"] <= c["steam_release"] + pd.Timedelta(days=1)].groupby("appid")["regular"].last()
    after = c[c["ts"] > c["steam_release"] + pd.Timedelta(days=1)].groupby("appid")["regular"].first()
    games["launch_price"] = games["appid"].map(upto).fillna(games["appid"].map(after))

    games["never_discounted"] = games["first_sale_date"].isna()
    games["observed_days"] = (now - games["steam_release"]).dt.total_seconds() / 86400
    # release_date >= MIN_RELEASE (not steam_release) so pre-2014 games whose log merely
    # starts when ITAD began tracking aren't mistaken for late Steam arrivals
    games["usable"] = (
        (games["type"] == "game") & (games["release_date"] >= MIN_RELEASE)
        & games["history_start"].notna() & games["launch_price"].gt(0))
    return games, changes, sales


if __name__ == "__main__":
    main()
