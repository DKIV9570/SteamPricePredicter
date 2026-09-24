"""Step 5: Steam store data per game.

- appdetails: description text, categories (single/multi/co-op...), genres, DLC count,
  Steam's own release date, metacritic, languages
- appreviewhistogram: monthly up/down review counts over the game's life — a leak-free
  popularity signal (reviews known at any point in time), unlike today's totals

One JSON per app in data/raw/steam_store/<appid>.json (resumable).
Steam doesn't publish a quota (~200 req / 5 min is the folk number), so we pace
conservatively and back off hard on 429/403.

    python -m pricepredicter.fetch_steam_store [--interval 1.6]
"""
import argparse
import json
import time

import pandas as pd
import requests
from tqdm import tqdm

from .config import PROCESSED, RAW, USER_AGENT

OUT = RAW / "steam_store"
KEEP = ["type", "name", "short_description", "categories", "genres", "dlc", "release_date",
        "metacritic", "supported_languages", "platforms", "controller_support", "required_age",
        "content_descriptors", "developers", "publishers", "is_free", "price_overview"]


class Paced:
    def __init__(self, interval=1.6):
        self.s = requests.Session()
        self.s.headers["User-Agent"] = USER_AGENT
        self.interval, self.last = interval, 0.0

    def get(self, url, **params):
        for attempt in range(8):
            time.sleep(max(0.0, self.interval - (time.monotonic() - self.last)))
            self.last = time.monotonic()
            try:
                r = self.s.get(url, params=params, timeout=30)
            except requests.RequestException:
                time.sleep(10 * 2 ** attempt)
                continue
            if r.status_code in (429, 403):
                # Steam's rate-limit window is ~5 minutes; slow down for good
                self.interval = min(self.interval * 1.25, 6.0)
                tqdm.write(f"rate limited ({r.status_code}); sleeping 5 min, interval -> {self.interval:.2f}s")
                time.sleep(300)
                continue
            if r.status_code >= 500:
                time.sleep(10 * 2 ** attempt)
                continue
            try:
                return r.json()
            except ValueError:
                return None
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=1.6)
    ap.add_argument("--appids", type=int, nargs="*", help="only these (default: all usable games)")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    games = pd.read_parquet(PROCESSED / "games.parquet")
    appids = args.appids or sorted(games.loc[games["usable"], "appid"], reverse=True)
    todo = [a for a in appids if not (OUT / f"{a}.json").exists()]
    client = Paced(args.interval)
    for appid in tqdm(todo, desc="steam store"):
        det = client.get("https://store.steampowered.com/api/appdetails",
                         appids=appid, cc="us", l="english")
        hist = client.get(f"https://store.steampowered.com/appreviewhistogram/{appid}",
                          l="all", review_score_preference=0)
        if det is None and hist is None:
            continue  # retried later on a rerun
        entry = (det or {}).get(str(appid)) or {}
        data = {k: entry["data"].get(k) for k in KEEP} if entry.get("success") else None
        if data and data.get("dlc") is not None:
            data["dlc"] = len(data["dlc"])
        rollups = ((hist or {}).get("results") or {}).get("rollups")
        (OUT / f"{appid}.json").write_text(json.dumps(
            {"appid": appid, "details": data, "review_rollups": rollups}), encoding="utf-8")


if __name__ == "__main__":
    main()
