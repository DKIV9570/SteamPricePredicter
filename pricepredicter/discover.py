"""Find games worth adding: Steam's "popular new releases" and "top sellers" lists, minus
what the catalog already has. New finds get a SteamSpy appdetails record (publisher,
developer, price, tags), which goes both into catalog.json and data/raw/steamspy_details/.

    python -m pricepredicter.discover [--top-sellers 500]
"""
import argparse
import json
import re
import time

import requests

from .config import RAW, USER_AGENT

SEARCH = "https://store.steampowered.com/search/results/"
PAGE = 100


def steam_list(filter_: str, limit: int, session) -> list[int]:
    ids = []
    for start in range(0, limit, PAGE):
        r = session.get(SEARCH, params={"filter": filter_, "category1": 998, "json": 1, "infinite": 1,
                                        "start": start, "count": PAGE, "cc": "us", "l": "english"}, timeout=30)
        r.raise_for_status()
        d = r.json()
        page = [int(x) for x in re.findall(r'data-ds-appid="(\d+)"', d.get("results_html", ""))]
        ids += page
        if len(page) < PAGE or start + PAGE >= d.get("total_count", 0):
            break
        time.sleep(1.5)
    return ids


def main(top_sellers: int = 500) -> list[int]:
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    found = steam_list("popularnew", 1000, s) + steam_list("topsellers", top_sellers, s)
    catalog_path = RAW / "catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    new = [a for a in dict.fromkeys(found) if str(a) not in catalog]
    print(f"listed {len(set(found))} games, {len(new)} not in the catalog")

    details_dir = RAW / "steamspy_details"
    details_dir.mkdir(parents=True, exist_ok=True)
    added = []
    for appid in new:
        time.sleep(1.05)  # SteamSpy: 1 request / second
        try:
            d = s.get("https://steamspy.com/api.php", params={"request": "appdetails", "appid": appid},
                      timeout=60).json()
        except (requests.RequestException, ValueError):
            continue
        if not d.get("name") or int(d.get("initialprice") or 0) <= 0:
            continue  # unknown to SteamSpy yet, or free-to-play
        catalog[str(appid)] = {k: d.get(k) for k in ["appid", "name", "developer", "publisher", "owners",
                                                      "price", "initialprice", "discount"]}
        (details_dir / f"{appid}.json").write_text(json.dumps(d), encoding="utf-8")
        added.append(appid)
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    print(f"added {len(added)} paid games to the catalog")
    return added


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-sellers", type=int, default=500)
    main(ap.parse_args().top_sellers)
