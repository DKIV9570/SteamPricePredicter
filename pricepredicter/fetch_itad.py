"""Step 2: fetch ITAD game info + Steam price history for every catalog app.

One JSON per app in data/raw/itad/<appid>.json; already-fetched apps are
skipped, so the script can be stopped and resumed at any time.

    python -m pricepredicter.fetch_itad [--limit N]
"""
import argparse
import json

from tqdm import tqdm

from .config import ITAD_RAW, RAW
from .itad import ITADClient

LOOKUP_BATCH = 200


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    # appids grow roughly with store-page creation date; <200000 is almost all pre-2014,
    # which ITAD has no launch-era history for anyway
    ap.add_argument("--min-appid", type=int, default=200000)
    args = ap.parse_args()

    catalog = json.loads((RAW / "catalog.json").read_text(encoding="utf-8"))
    # Newest first: with a tight API quota, the most relevant games land first
    appids = sorted((int(a) for a in catalog if int(a) >= args.min_appid), reverse=True)
    ITAD_RAW.mkdir(parents=True, exist_ok=True)
    client = ITADClient()

    ids_path = RAW / "itad_ids.json"
    ids = {int(k): v for k, v in json.loads(ids_path.read_text()).items()} if ids_path.exists() else {}
    missing = [a for a in appids if a not in ids]
    for i in tqdm(range(0, len(missing), LOOKUP_BATCH), desc="lookup"):
        ids.update(client.lookup_steam_appids(missing[i:i + LOOKUP_BATCH]))
        ids_path.write_text(json.dumps(ids))

    todo = [a for a in appids if ids.get(a) and not (ITAD_RAW / f"{a}.json").exists()]
    if args.limit:
        todo = todo[:args.limit]
    for appid in tqdm(todo, desc="info+history"):
        gid = ids[appid]
        record = {"appid": appid, "itad_id": gid,
                  "info": client.info(gid), "history": client.history(gid)}
        (ITAD_RAW / f"{appid}.json").write_text(json.dumps(record), encoding="utf-8")


if __name__ == "__main__":
    main()
