"""The daily update, run by GitHub Actions (or by hand):

  discover new games -> their ITAD history -> refresh every game's current price ->
  rebuild tables -> localized names / store data / first-week reviews for games missing them ->
  re-fetch SteamSpy tags for young games -> content tables -> website data.

Collection steps are best-effort (a failure means less new data today); the rebuild and
export steps must succeed. Backlogs are capped per run so the job stays well inside the
Actions time limit.

    python -m pricepredicter.daily
"""
import json
import os
import subprocess
import sys
import time

import pandas as pd

from .config import PROCESSED, RAW

STORE_PER_RUN = 1200        # Steam store fetches (2 requests each)
W1_PER_RUN = 3000           # first-week review lookups (1 request each)
YOUNG_DAYS = 60             # re-fetch SteamSpy tags while a game is this young...
YOUNG_REFRESH_DAYS = 7      # ...at most once a week


FAILED: list[str] = []


def run(module: str, *args, required: bool = False) -> bool:
    t = time.time()
    print(f"\n=== {module} {' '.join(map(str, args))[:80]}", flush=True)
    ok = subprocess.run([sys.executable, "-m", f"pricepredicter.{module}", *map(str, args)]).returncode == 0
    print(f"=== {module}: {'ok' if ok else 'FAILED'} in {time.time() - t:.0f}s", flush=True)
    if required and not ok:
        raise SystemExit(f"{module} failed")
    if not ok:
        FAILED.append(module)
    return ok


def missing(dirname: str) -> list[int]:
    """Usable games without a file in data/raw/<dirname>, newest first."""
    g = pd.read_parquet(PROCESSED / "games.parquet")
    have = {int(p.stem) for p in (RAW / dirname).glob("*.json")}
    return [a for a in sorted(g.loc[g["usable"], "appid"], reverse=True) if a not in have]


def young_games_to_retag() -> list[int]:
    g = pd.read_parquet(PROCESSED / "games.parquet")
    now = pd.Timestamp.now(tz="UTC")
    young = g[g["usable"] & (g["steam_release"] > now - pd.Timedelta(days=YOUNG_DAYS))]
    stale = []
    for a in young["appid"]:
        p = RAW / "steamspy_details" / f"{a}.json"
        if not p.exists() or time.time() - p.stat().st_mtime > YOUNG_REFRESH_DAYS * 86400:
            stale.append(int(a))
        elif not (json.loads(p.read_text(encoding="utf-8")).get("tags") or None):
            stale.append(int(a))  # SteamSpy hadn't tagged it yet last time
    return stale


def main():
    t0 = time.time()
    run("discover")
    run("fetch_itad")
    run("refresh_prices")
    run("build_dataset", required=True)
    run("fetch_names")
    todo = missing("steam_store")[:STORE_PER_RUN]
    if todo:
        run("fetch_steam_store", "--interval", 1.5, "--appids", *todo)
    run("fetch_first_week_reviews", "--limit", W1_PER_RUN)
    retag = young_games_to_retag()
    run("fetch_steamspy_details", "--interval", 1.05, *(["--refresh", *retag] if retag else []))
    run("build_content", required=True)
    run("export_site", required=True)
    print(f"\ndaily update done in {(time.time() - t0) / 60:.0f} min"
          + (f"; FAILED (best-effort): {', '.join(FAILED)}" if FAILED else ""))
    # The site still gets published, but the workflow flags the run so failures don't go unnoticed
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"degraded={' '.join(FAILED)}\n")


if __name__ == "__main__":
    main()
