"""Step 4: per-game SteamSpy details — the full user-tag list with vote counts,
genres and languages, used to build game/publisher similarity.

One JSON per app in data/raw/steamspy_details/<appid>.json (resumable).
SteamSpy allows 1 appdetails request per second.

    python -m pricepredicter.fetch_steamspy_details [--include-catalog] [--interval 1.15]
"""
import argparse
import json
import time

import pandas as pd
import requests
from tqdm import tqdm

from .config import PROCESSED, RAW, USER_AGENT

URL = "https://steamspy.com/api.php"
OUT = RAW / "steamspy_details"


def main():
    ap = argparse.ArgumentParser()
    # Also fetch catalog apps not yet processed by ITAD (after the usable ones)
    ap.add_argument("--include-catalog", action="store_true")
    # 1/s is SteamSpy's limit; the default margin leaves room for a concurrent
    # `request=all` catalog page (1/min)
    ap.add_argument("--interval", type=float, default=1.15)
    # Re-fetch these even if a file exists (young games whose tags are still forming);
    # the old file is only replaced once the new response arrives
    ap.add_argument("--refresh", type=int, nargs="*", default=[])
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    games = pd.read_parquet(PROCESSED / "games.parquet")
    appids = sorted(games.loc[games["usable"], "appid"], reverse=True)
    if args.include_catalog:
        catalog = json.loads((RAW / "catalog.json").read_text(encoding="utf-8"))
        seen = set(appids)
        appids += sorted((int(a) for a in catalog if int(a) >= 200000 and int(a) not in seen), reverse=True)
    todo = list(dict.fromkeys(args.refresh + [a for a in appids if not (OUT / f"{a}.json").exists()]))

    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    last = 0.0
    for appid in tqdm(todo, desc="steamspy details"):
        for attempt in range(5):
            time.sleep(max(0.0, args.interval - (time.monotonic() - last)))
            last = time.monotonic()
            try:
                r = s.get(URL, params={"request": "appdetails", "appid": appid}, timeout=60)
                r.raise_for_status()
                data = r.json()
                break
            except (requests.RequestException, ValueError):
                time.sleep(5 * 2 ** attempt)
        else:
            continue  # give up on this app; a rerun will retry it
        (OUT / f"{appid}.json").write_text(json.dumps(data), encoding="utf-8")


if __name__ == "__main__":
    main()
