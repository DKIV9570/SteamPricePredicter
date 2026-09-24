"""Step 7: number of Steam reviews in each game's first week on Steam.

Early review volume is the best public proxy for "is this a hit?" — hits discount later
and shallower. The appreviews endpoint filters by exact date range, so the count means the
same thing for a 2016 game as for a 2026 one (the review histogram only gives monthly
buckets for older games).

One JSON per app in data/raw/steam_reviews_w1/<appid>.json (resumable), in random order so
that a partial run is a uniform sample (usable for training before the fetch finishes).

    python -m pricepredicter.fetch_first_week_reviews [--interval 1.0]
"""
import argparse
import json

import pandas as pd
from tqdm import tqdm

from .config import PROCESSED, RAW
from .fetch_steam_store import Paced

OUT = RAW / "steam_reviews_w1"
WINDOW_DAYS = 7


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=1.0)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    games = pd.read_parquet(PROCESSED / "games.parquet")
    u = games[games["usable"]].sample(frac=1.0, random_state=0)
    # The first week must be over, or we'd store a partial count and never revisit it
    week_over = u["steam_release"] < pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=WINDOW_DAYS + 1)
    todo = u[week_over & [not (OUT / f"{a}.json").exists() for a in u["appid"]]]
    client = Paced(args.interval)
    for appid, rel in tqdm(zip(todo["appid"], todo["steam_release"]), total=len(todo), desc="first-week reviews"):
        start = int(rel.floor("D").timestamp())
        end = start + WINDOW_DAYS * 86400
        res = client.get(f"https://store.steampowered.com/appreviews/{appid}",
                         json=1, filter="all", language="all", purchase_type="all", num_per_page=0,
                         review_type="all", day_range=9223372036854775807,
                         start_date=start, end_date=end, date_range_type="include")
        qs = (res or {}).get("query_summary")
        if qs is None:
            continue  # retried on a rerun
        (OUT / f"{appid}.json").write_text(json.dumps({
            "appid": int(appid), "start": start, "end": end, "total": qs.get("total_reviews"),
            "positive": qs.get("total_positive"), "negative": qs.get("total_negative")}), encoding="utf-8")


if __name__ == "__main__":
    main()
