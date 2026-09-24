"""Daily price refresh: ask ITAD for every game's current Steam price in batches of 200
and append any change to that game's stored history, so the full history never has to be
re-downloaded. ITAD's `timestamp` says when the current price started; a newer timestamp
than the last stored change means the price moved.

A price that changes and changes back between two runs is missed; Steam sales last days,
so a daily run catches practically all of them.

    python -m pricepredicter.refresh_prices
"""
import json

import pandas as pd
from tqdm import tqdm

from .config import ITAD_RAW, RAW, STEAM_SHOP_ID
from .itad import ITADClient

BATCH = 200


def main() -> int:
    ids = {int(k): v for k, v in json.loads((RAW / "itad_ids.json").read_text()).items() if v}
    have = {int(p.stem) for p in ITAD_RAW.glob("*.json")}
    by_gid = {gid: appid for appid, gid in ids.items() if appid in have}
    gids = list(by_gid)
    client = ITADClient()
    changed = 0
    for i in tqdm(range(0, len(gids), BATCH), desc="prices"):
        res = client._request("POST", "/games/prices/v3",
                              params={"country": "US", "shops": STEAM_SHOP_ID, "deals": "false"},
                              json=gids[i:i + BATCH])
        for item in res:
            deal = next((d for d in item.get("deals") or [] if d["shop"]["id"] == STEAM_SHOP_ID), None)
            appid = by_gid.get(item["id"])
            if deal is None or appid is None:
                continue
            path = ITAD_RAW / f"{appid}.json"
            try:
                rec = json.loads(path.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                continue  # unreadable file; skipped until it's re-fetched
            hist = rec["history"]  # newest first
            ts = pd.Timestamp(deal["timestamp"])
            last = hist[0] if hist else None
            same_state = last and last["deal"]["price"]["amount"] == deal["price"]["amount"] \
                and last["deal"]["regular"]["amount"] == deal["regular"]["amount"]
            if last and (same_state or pd.Timestamp(last["timestamp"]) >= ts):
                continue
            hist.insert(0, {"timestamp": deal["timestamp"], "shop": deal["shop"],
                            "deal": {"price": deal["price"], "regular": deal["regular"], "cut": deal["cut"]}})
            path.write_text(json.dumps(rec), encoding="utf-8")
            changed += 1
    print(f"price changes recorded: {changed}")
    return changed


if __name__ == "__main__":
    main()
