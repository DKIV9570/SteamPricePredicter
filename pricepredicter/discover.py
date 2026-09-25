"""Find games worth adding and put them in the catalog.

SteamSpy's owner estimates (what the original catalog was ranked by) badly undercount games
from the last year or two — Anno 117 shows "0 .. 20,000" — so recent hits were missing.
Candidates now come from:
  - Steam "popular new releases" and the full Steam top-sellers list
  - IsThereAnyDeal's most-waitlisted games: people waiting for a discount, i.e. our users
Each new paid game gets a SteamSpy appdetails record (publisher, developer, price, tags),
stored in catalog.json and data/raw/steamspy_details/. Candidates that turn out free (or
unknown to SteamSpy) are remembered and not re-checked for a while.

    python -m pricepredicter.discover [--waitlisted 30000]
"""
import argparse
import json
import re
import time

import requests

from .config import RAW, STEAM_SHOP_ID, USER_AGENT
from .itad import ITADClient

SEARCH = "https://store.steampowered.com/search/results/"
PAGE = 100
REJECTED = RAW / "discover_rejected.json"
RECHECK_DAYS = 30


def steam_list(filter_: str, limit: int, session) -> list[int]:
    """Appids from a Steam store search list. Rate limits end the listing early rather than
    failing it: whatever was collected is still useful."""
    ids = []
    for start in range(0, limit, PAGE):
        d = None
        for attempt in range(4):
            try:
                r = session.get(SEARCH, params={"filter": filter_, "category1": 998, "json": 1, "infinite": 1,
                                                "start": start, "count": PAGE, "cc": "us", "l": "english"},
                                timeout=30)
                if r.status_code == 429:
                    time.sleep(60 * (attempt + 1))
                    continue
                r.raise_for_status()
                d = r.json()
                break
            except (requests.RequestException, ValueError):
                time.sleep(10 * (attempt + 1))
        if d is None:
            print(f"steam '{filter_}' list stopped at {start}")
            break
        page = [int(x) for x in re.findall(r'data-ds-appid="(\d+)"', d.get("results_html", ""))]
        ids += page
        # A page can hold fewer than PAGE apps (bundles carry no appid), so only an empty page
        # or the reported total ends the listing
        if not page or start + PAGE >= d.get("total_count", 0):
            break
        time.sleep(2)
    return ids


def itad_most_waitlisted(limit: int) -> list[int]:
    """Steam appids of ITAD's most-waitlisted *games* (DLC excluded), most wanted first."""
    client = ITADClient()
    gids = []
    for offset in range(0, limit, 50):
        page = client._request("GET", "/stats/most-waitlisted/v1", params={"offset": offset, "limit": 50})
        if not page:
            break
        # untyped entries are kept; build_dataset keeps only ITAD type "game" anyway
        gids += [x["id"] for x in page if x.get("type") in ("game", None)]
    apps = []
    for i in range(0, len(gids), 200):
        res = client._request("POST", f"/lookup/shop/{STEAM_SHOP_ID}/id/v1", json=gids[i:i + 200])
        for gid in gids[i:i + 200]:
            # a game can map to several Steam apps (editions); keep them all
            apps += [int(s.split("/")[1]) for s in (res.get(gid) or []) if s.startswith("app/")]
    return apps


def main(waitlisted: int = 30000) -> list[int]:
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    # the store serves ~3,000 top sellers before it starts refusing deeper pages
    found = steam_list("popularnew", 1000, s) + steam_list("topsellers", 3000, s)
    try:
        found += itad_most_waitlisted(waitlisted)
    except Exception as e:  # one source failing shouldn't stop the others
        print(f"ITAD waitlist ranking unavailable: {e}")
    catalog_path = RAW / "catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    rejected = json.loads(REJECTED.read_text(encoding="utf-8")) if REJECTED.exists() else {}
    now = time.time()
    new = [a for a in dict.fromkeys(found) if str(a) not in catalog
           and now - rejected.get(str(a), 0) > RECHECK_DAYS * 86400]
    print(f"listed {len(set(found))} games, {len(new)} new candidates")

    details_dir = RAW / "steamspy_details"
    details_dir.mkdir(parents=True, exist_ok=True)
    added = []
    for n, appid in enumerate(new, 1):
        time.sleep(1.05)  # SteamSpy: 1 request / second
        try:
            d = s.get("https://steamspy.com/api.php", params={"request": "appdetails", "appid": appid},
                      timeout=60).json()
        except (requests.RequestException, ValueError):
            continue  # transient; tried again next run
        if not d.get("name") or int(d.get("initialprice") or 0) <= 0:
            rejected[str(appid)] = now  # free-to-play, or not known to SteamSpy yet
            continue
        catalog[str(appid)] = {k: d.get(k) for k in ["appid", "name", "developer", "publisher", "owners",
                                                      "price", "initialprice", "discount"]}
        (details_dir / f"{appid}.json").write_text(json.dumps(d), encoding="utf-8")
        added.append(appid)
        if n % 200 == 0:  # checkpoint long backfills
            catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
            REJECTED.write_text(json.dumps(rejected), encoding="utf-8")
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    REJECTED.write_text(json.dumps(rejected), encoding="utf-8")
    print(f"added {len(added)} paid games to the catalog")
    return added


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--waitlisted", type=int, default=30000)
    main(ap.parse_args().waitlisted)
