"""Detect Steam's major sale events (Summer, Winter, Autumn, Spring, Lunar New Year, ...)
from the data itself: days on which thousands of games start a discount at once.

Valve moves the dates around (e.g. the 2025 Autumn Sale started in late September),
so a hardcoded calendar would drift. Upcoming events for live predictions come from
Valve's published schedule instead — see UPCOMING.
"""
import pandas as pd

# A day is a major-sale start if its sale-start count reaches this share of the year's peak
MAJOR_SHARE_OF_YEAR_PEAK = 0.3
MAJOR_LENGTH_DAYS = 8  # majors run ~1-2 weeks; used only to mark the window

# Announced by Valve at https://partner.steamgames.com/doc/marketing/upcoming_events
# (checked 2026-09-22); extend as new dates are published
UPCOMING = [
    "2026-10-01",  # Autumn Sale
    "2026-12-17",  # Winter Sale
    "2027-03-18",  # Spring Sale
    "2027-06-24",  # Summer Sale
]


def detect_majors(sales: pd.DataFrame) -> pd.DataFrame:
    """sales: sales.parquet rows. Returns one row per major event: start (tz-naive date), name."""
    s = sales[sales["kind"] == "regular"]
    # Steam sales start 10:00 Pacific, so bucket days in Pacific time
    day = s["start"].dt.tz_convert("US/Pacific").dt.tz_localize(None).dt.floor("D")
    cnt = day.value_counts().sort_index()
    cnt = cnt[cnt.index >= "2014-06-01"]
    peak = cnt.groupby(cnt.index.year).transform("max")
    starts = cnt[cnt >= MAJOR_SHARE_OF_YEAR_PEAK * peak].index.sort_values()
    # Collapse starts within a few days of each other into one event
    keep = [d for i, d in enumerate(starts) if i == 0 or (d - starts[i - 1]).days > 5]
    ev = pd.DataFrame({"start": keep})
    ev["name"] = ev["start"].map(_name)
    return ev


def _name(d: pd.Timestamp) -> str:
    m = d.month
    if m in (1, 2):
        return "lunar_new_year"
    if m == 3:
        return "spring"
    if m in (6, 7):
        return "summer"
    if m == 9 or (m == 10 and d.day <= 7) or (m == 11 and d.day >= 15):
        return "autumn"
    if m == 10 or m == 11:
        return "halloween"
    if m == 12:
        return "winter"
    return "other"


def all_majors(sales: pd.DataFrame) -> pd.DataFrame:
    ev = detect_majors(sales)
    up = pd.DataFrame({"start": pd.to_datetime(UPCOMING)})
    up = up[up["start"] > ev["start"].max()]
    up["name"] = up["start"].map(_name)
    return pd.concat([ev, up], ignore_index=True)
