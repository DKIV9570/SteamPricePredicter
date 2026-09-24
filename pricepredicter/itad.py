"""Minimal IsThereAnyDeal API client with rate limiting and retries.

Docs: https://docs.isthereanydeal.com/
The quota is per account (see the app page); set ITAD_RATE_PER_5MIN in .env if yours differs.
"""
import time

import requests

from .config import ITAD_API_KEY, ITAD_RATE_PER_5MIN, STEAM_SHOP_ID, USER_AGENT

BASE = "https://api.isthereanydeal.com"


class ITADClient:
    def __init__(self, key: str = ITAD_API_KEY, rate_per_5min: int = ITAD_RATE_PER_5MIN):
        if not key:
            raise RuntimeError("ITAD_API_KEY missing — put it in .env")
        self.s = requests.Session()
        # ITAD rejects Python's default User-Agent with 403
        self.s.headers.update({"ITAD-API-Key": key, "User-Agent": USER_AGENT})
        # Even spacing with a 5% margin, so we never hit the window limit
        self.min_interval = 300 / rate_per_5min * 1.05
        self._last = 0.0

    def _request(self, method, path, **kw):
        for attempt in range(6):
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            try:
                r = self.s.request(method, BASE + path, timeout=30, **kw)
            except requests.RequestException:
                time.sleep(2 ** attempt)
                continue
            if r.status_code == 429:
                time.sleep(float(r.headers.get("Retry-After", 30)))
                continue
            if r.status_code >= 500:
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            return r.json()
        raise RuntimeError(f"ITAD request failed after retries: {method} {path}")

    def lookup_steam_appids(self, appids):
        """Map Steam appids to ITAD game ids (None if unknown)."""
        body = [f"app/{a}" for a in appids]
        res = self._request("POST", f"/lookup/id/shop/{STEAM_SHOP_ID}/v1", json=body)
        return {int(k.split("/", 1)[1]): v for k, v in res.items()}

    def info(self, game_id):
        return self._request("GET", "/games/info/v2", params={"id": game_id})

    def history(self, game_id, since="2010-01-01T00:00:00Z", country="US"):
        """Steam-only price change log, newest first."""
        return self._request("GET", "/games/history/v2", params={
            "id": game_id, "shops": STEAM_SHOP_ID, "country": country, "since": since})
