"""Localized store names (Simplified / Traditional Chinese) so the site can be searched
as "鬼武者" or "艾爾登法環", plus Steam's own release date (a fallback for games ITAD has
no release date for), via the store's batch endpoint (no API key, many apps per call).

Writes data/raw/names.json: {appid: {"schinese": ..., "tchinese": ..., "release": unix}}
(resumable; merges).

    python -m pricepredicter.fetch_names [--appids 1 2 3]
"""
import argparse
import json
import time

import pandas as pd
import requests
from tqdm import tqdm

from .config import PROCESSED, RAW, USER_AGENT

URL = "https://api.steampowered.com/IStoreBrowseService/GetItems/v1/"
LANGS = {"schinese": "CN", "tchinese": "TW"}
BATCH = 100
OUT = RAW / "names.json"


def fetch_items(appids, lang, country, session) -> dict:
    """{appid: (name, steam_release_unix or None)}"""
    inp = {"ids": [{"appid": int(a)} for a in appids],
           "context": {"language": lang, "country_code": country, "steam_realm": 1},
           "data_request": {"include_release": True}}
    for attempt in range(5):
        try:
            r = session.get(URL, params={"input_json": json.dumps(inp)}, timeout=30)
            if r.status_code == 429:
                time.sleep(60)
                continue
            r.raise_for_status()
            return {it["appid"]: (it["name"], (it.get("release") or {}).get("steam_release_date"))
                    for it in r.json()["response"].get("store_items", []) if it.get("appid") and it.get("name")}
        except (requests.RequestException, ValueError, KeyError):
            time.sleep(5 * 2 ** attempt)
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--appids", type=int, nargs="*", help="only these (default: all usable games)")
    args = ap.parse_args()

    names = {int(k): v for k, v in json.loads(OUT.read_text(encoding="utf-8")).items()} if OUT.exists() else {}
    if args.appids:
        todo = args.appids
    else:
        g = pd.read_parquet(PROCESSED / "games.parquet")
        todo = [a for a in g.loc[g["usable"], "appid"] if a not in names]
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    for i in tqdm(range(0, len(todo), BATCH), desc="names"):
        batch = todo[i:i + BATCH]
        for lang, cc in LANGS.items():
            for appid, (name, release) in fetch_items(batch, lang, cc, s).items():
                names.setdefault(appid, {})[lang] = name
                if release:
                    names[appid]["release"] = release
            time.sleep(0.5)
        for appid in batch:
            names.setdefault(appid, {})  # mark as tried, even if the store returned nothing
        if i // BATCH % 20 == 0:
            OUT.write_text(json.dumps(names, ensure_ascii=False), encoding="utf-8")
    OUT.write_text(json.dumps(names, ensure_ascii=False), encoding="utf-8")
    print(f"names for {sum(bool(v) for v in names.values()):,} apps")


if __name__ == "__main__":
    main()
