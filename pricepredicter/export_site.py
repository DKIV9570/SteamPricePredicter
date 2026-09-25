"""Precompute predictions for every game and write the static website's data files.

site/data/meta.json        update time, targets, windows, upcoming major sales
site/data/index.json       [[appid, title, chinese_display_or_"", *search_aliases], ...] most-wanted
                           first; aliases (names.py) are matched but not shown
site/data/g/<k>.json       per-game records, sharded by appid % N_SHARDS

    python -m pricepredicter.export_site
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import RAW
from .names import display_and_aliases
from .serve import game_features, load_artifacts, load_reference, predict_now, upcoming_majors

SITE_DATA = Path(__file__).resolve().parent.parent / "site" / "data"
EXPORT_CUTS = [20, 25, 30, 33, 40, 50, 60, 67, 70, 75, 80]
WINDOWS = ["next_major", "second_major", "90", "180", "365"]
N_SHARDS = 128


def main():
    art, ref = load_artifacts(), load_reference()
    now = pd.Timestamp.now(tz="US/Pacific").tz_localize(None)
    feats = game_features(ref, art)
    rows, states = predict_now(feats, ref, art, now, EXPORT_CUTS)
    print(f"predicted {len(rows):,} (game, target, window) rows for {rows.appid.nunique():,} games")

    # probability table per game: targets x windows, in percent; None = price already there
    rows["pct"] = np.rint(rows["p"] * 100).astype(int)
    tab = rows.pivot_table(index=["appid", "threshold"], columns="horizon", values="pct", observed=True)
    tab = tab.reindex(columns=WINDOWS)

    g = feats["g"].set_index("appid")
    st = states.set_index("appid")
    names_path = RAW / "names.json"
    names = {int(k): v for k, v in json.loads(names_path.read_text(encoding="utf-8")).items()} \
        if names_path.exists() else {}
    records = {}
    for appid in g.index:
        s = st.loc[appid]
        probs = []
        for cut in EXPORT_CUTS:
            key = (appid, cut)
            probs.append(None if key not in tab.index else
                         [None if pd.isna(v) else int(v) for v in tab.loc[key].to_numpy()])
        zh, _ = display_and_aliases(g.at[appid, "title"], names.get(int(appid), {}))
        records[int(appid)] = {
            "t": g.at[appid, "title"],
            **({"zh": zh} if zh else {}),
            "lp": round(float(g.at[appid, "launch_price"]), 2),
            "cp": round(float(g.at[appid, "launch_price"] * s["price_ratio_now"]), 2),
            "best": int(round(s["best_cut_so_far"])),
            "ns": int(s["n_sales_so_far"]),
            "rd": g.at[appid, "t0"].strftime("%Y-%m-%d"),
            "p": probs,
        }

    (SITE_DATA / "g").mkdir(parents=True, exist_ok=True)
    shards: dict[int, dict] = {}
    for appid, rec in records.items():
        shards.setdefault(appid % N_SHARDS, {})[appid] = rec
    for k in range(N_SHARDS):
        (SITE_DATA / "g" / f"{k}.json").write_text(
            json.dumps(shards.get(k, {}), ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    order = g["waitlisted"].fillna(0).sort_values(ascending=False).index
    index = []
    for a in order:
        zh, aliases = display_and_aliases(g.at[a, "title"], names.get(int(a), {}))
        entry = [int(a), g.at[a, "title"]]
        if zh or aliases:
            entry += [zh, *aliases]
        index.append(entry)
    (SITE_DATA / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    up = upcoming_majors(ref["majors"], now)
    (SITE_DATA / "meta.json").write_text(json.dumps({
        "updated": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%MZ"),
        "cuts": EXPORT_CUTS, "windows": WINDOWS, "shards": N_SHARDS,
        "sales": [{"name": r["name"], "start": r["start"].strftime("%Y-%m-%d")} for _, r in up.iterrows()],
        "games": len(records),
    }), encoding="utf-8")
    size = sum(p.stat().st_size for p in SITE_DATA.rglob("*.json")) / 1e6
    print(f"wrote {len(records):,} games, {size:.1f} MB to {SITE_DATA}")


if __name__ == "__main__":
    main()
