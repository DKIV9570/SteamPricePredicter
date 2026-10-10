"""Archive the day's predictions (site/data) so they can be scored once the future arrives
(analysis/11_sale_scorecard.py). Keeps one snapshot a week, plus one in the days right
before each major sale.

    python -m pricepredicter.snapshot      # prints the archive path, or nothing if not due today
"""
import datetime as dt
import json
import tarfile

from .config import ROOT

SITE_DATA = ROOT / "site" / "data"
PRE_SALE_DAYS = 3


def due(today: dt.date, next_sale: dt.date | None) -> bool:
    if today.weekday() == 0:  # Monday
        return True
    return next_sale is not None and 0 < (next_sale - today).days <= PRE_SALE_DAYS


def main():
    meta = json.loads((SITE_DATA / "meta.json").read_text(encoding="utf-8"))
    today = dt.datetime.fromisoformat(meta["updated"].replace("Z", "+00:00")).date()
    nxt = dt.date.fromisoformat(meta["sales"][0]["start"]) if meta["sales"] else None
    if not due(today, nxt):
        return
    out = ROOT / f"preds-{today:%Y-%m-%d}.tar.gz"
    with tarfile.open(out, "w:gz") as tar:
        tar.add(SITE_DATA, arcname="data")
    print(out)


if __name__ == "__main__":
    main()
