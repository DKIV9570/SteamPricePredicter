"""Step 6: content tables from SteamSpy details + Steam store data.

tags.parquet      appid, tag, votes            (long format, SteamSpy user tags)
content.parquet   one row per app: languages, store categories, genres,
                  metacritic, description, first-month reviews

    python -m pricepredicter.build_content
"""
import json

import pandas as pd

from .config import PROCESSED, RAW

# Store categories worth keeping as flags (the rest are Steam-feature noise)
CATEGORIES = {
    "Single-player": "cat_single", "Multi-player": "cat_multi", "Online PvP": "cat_pvp",
    "Co-op": "cat_coop", "Online Co-op": "cat_online_coop", "Shared/Split Screen": "cat_split",
    "MMO": "cat_mmo", "In-App Purchases": "cat_iap", "Full controller support": "cat_controller",
    "Steam Workshop": "cat_workshop", "VR Only": "cat_vr_only", "VR Support": "cat_vr",
}
FIRST_MONTH_DAYS = 30


def first_month_reviews(rollups, t0):
    """Reviews in the first ~30 days after Steam launch.
    Steam returns weekly buckets for younger games and monthly ones for older games.
    Each bucket contributes the share of it that overlaps [t0, t0+30d); the bucket
    holding t0 counts fully, since no reviews exist before launch."""
    if not rollups or pd.isna(t0):
        return None, None
    end = t0 + pd.Timedelta(days=FIRST_MONTH_DAYS)
    up = down = 0.0
    for i, r in enumerate(rollups):
        b0 = pd.Timestamp(r["date"], unit="s", tz="UTC")
        b1 = (pd.Timestamp(rollups[i + 1]["date"], unit="s", tz="UTC") if i + 1 < len(rollups)
              else b0 + (b0 - pd.Timestamp(rollups[i - 1]["date"], unit="s", tz="UTC") if i else pd.Timedelta(days=7)))
        if b1 <= t0 or b0 >= end:
            continue
        lo = max(b0, t0) if b0 > t0 else b0  # launch bucket: everything in it is post-launch
        frac = min(1.0, max(0.0, (min(b1, end) - lo) / (b1 - lo)))
        up += r["recommendations_up"] * frac
        down += r["recommendations_down"] * frac
    return round(up), round(down)


def review_months(rollups, appid) -> pd.DataFrame:
    """Monthly review counts (the sales proxy). Steam returns weekly buckets for younger
    games and monthly ones for older games; weeks are summed into the month they start in,
    so every game is on the same monthly footing."""
    if not rollups:
        return pd.DataFrame(columns=["appid", "month", "reviews"])
    d = pd.DataFrame(rollups)
    d["month"] = pd.to_datetime(d["date"], unit="s").dt.to_period("M").dt.to_timestamp()
    d["reviews"] = d["recommendations_up"] + d["recommendations_down"]
    out = d.groupby("month", as_index=False)["reviews"].sum()
    out.insert(0, "appid", appid)
    return out


def main():
    games = pd.read_parquet(PROCESSED / "games.parquet")[["appid", "steam_release"]]
    t0 = games.set_index("appid")["steam_release"]

    tag_rows, content = [], {}
    for p in (RAW / "steamspy_details").glob("*.json"):
        d = json.loads(p.read_text(encoding="utf-8"))
        appid = int(p.stem)
        tags = d.get("tags") or {}
        if isinstance(tags, dict):  # SteamSpy returns [] when an app has no tags
            tag_rows += [(appid, t, int(v)) for t, v in tags.items()]
        langs = [x for x in (d.get("languages") or "").split(",") if x.strip()]
        content[appid] = {"appid": appid, "n_languages": len(langs) or None,
                          "spy_genre": d.get("genre")}

    for p in (RAW / "steam_store").glob("*.json"):
        rec = json.loads(p.read_text(encoding="utf-8"))
        appid = rec["appid"]
        row = content.setdefault(appid, {"appid": appid})
        det = rec.get("details") or {}
        row["has_store"] = bool(det)
        cats = {c["description"] for c in det.get("categories") or []}
        for name, col in CATEGORIES.items():
            row[col] = int(name in cats) if det else None
        row["store_genres"] = "|".join(g["description"] for g in det.get("genres") or [])
        row["metacritic"] = (det.get("metacritic") or {}).get("score")
        row["required_age"] = pd.to_numeric(det.get("required_age"), errors="coerce") if det else None
        row["description"] = det.get("short_description")
        up, down = first_month_reviews(rec.get("review_rollups"), t0.get(appid))
        row["m1_reviews_up"], row["m1_reviews_down"] = up, down

    for p in (RAW / "steam_reviews_w1").glob("*.json"):
        rec = json.loads(p.read_text(encoding="utf-8"))
        row = content.setdefault(rec["appid"], {"appid": rec["appid"]})
        row["w1_reviews"], row["w1_positive"] = rec.get("total"), rec.get("positive")

    tags = pd.DataFrame(tag_rows, columns=["appid", "tag", "votes"])
    content = pd.DataFrame(list(content.values()))
    tags.to_parquet(PROCESSED / "tags.parquet", index=False)
    content.to_parquet(PROCESSED / "content.parquet", index=False)

    timeline = pd.concat(
        [review_months(json.loads(p.read_text(encoding="utf-8")).get("review_rollups"), int(p.stem))
         for p in (RAW / "steam_store").glob("*.json")], ignore_index=True)
    timeline.to_parquet(PROCESSED / "review_months.parquet", index=False)
    print(f"review timeline: {timeline.appid.nunique():,} apps, {len(timeline):,} app-months")
    print(f"tags: {len(tags):,} rows, {tags.appid.nunique():,} apps, {tags.tag.nunique()} distinct tags")
    print(f"content: {len(content):,} apps, with store data: {content.get('has_store', pd.Series()).eq(True).sum():,}")


if __name__ == "__main__":
    main()
