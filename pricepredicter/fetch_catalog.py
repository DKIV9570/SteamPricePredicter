"""Step 1: build the candidate game list from SteamSpy.

SteamSpy's `request=all` returns 1000 apps per page ordered by owners,
and allows one such request per minute.

    python -m pricepredicter.fetch_catalog --pages 20
"""
import argparse
import json
import time

import requests

from .config import RAW, USER_AGENT

URL = "https://steamspy.com/api.php"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=20)
    args = ap.parse_args()

    out_dir = RAW / "steamspy"
    out_dir.mkdir(parents=True, exist_ok=True)

    for page in range(args.pages):
        path = out_dir / f"page_{page:03d}.json"
        if path.exists():
            continue
        r = requests.get(URL, params={"request": "all", "page": page},
                         headers={"User-Agent": USER_AGENT}, timeout=120)
        r.raise_for_status()
        data = r.json()
        path.write_text(json.dumps(data), encoding="utf-8")
        print(f"page {page}: {len(data)} apps")
        if not data:
            break
        if page < args.pages - 1:
            time.sleep(61)

    catalog = {}
    for path in sorted(out_dir.glob("page_*.json")):
        for app in json.loads(path.read_text(encoding="utf-8")).values():
            # Free games never go on sale; drop them early
            if int(app.get("initialprice") or 0) > 0:
                catalog[app["appid"]] = app
    (RAW / "catalog.json").write_text(json.dumps(catalog), encoding="utf-8")
    print(f"catalog: {len(catalog)} paid apps")


if __name__ == "__main__":
    main()
