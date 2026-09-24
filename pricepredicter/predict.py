"""Live prediction: should I wait for a discount, or buy now?

    python -m pricepredicter.predict https://store.steampowered.com/app/1245620 --target 50
    python -m pricepredicter.predict 1245620 --target-price 29.99

Fetches the game's current data, computes the same features the model was trained on
(against the reference dataset in data/processed), and prints a one-line verdict plus
the chance of reaching the target by the next major sale / 3 / 6 / 12 months.
Price data: IsThereAnyDeal (https://isthereanydeal.com). Prices are Steam US (USD).
"""
import argparse
import re
import sys

import pandas as pd
import requests

from .build_content import CATEGORIES, first_month_reviews, review_months
from .build_dataset import process_records
from .config import USER_AGENT
from .fetch_first_week_reviews import WINDOW_DAYS
from .fetch_steam_store import Paced
from .itad import ITADClient
from .serve import CUT_TOL, game_features, load_artifacts, load_reference, predict_now, upcoming_majors

SALE_NAMES = {"summer": "夏促", "winter": "冬促", "autumn": "秋促", "spring": "春促",
              "lunar_new_year": "春节促销", "halloween": "万圣节促销", "other": "大促"}


def parse_appid(s: str) -> int:
    m = re.search(r"/app/(\d+)", s) or re.fullmatch(r"\s*(\d+)\s*", s)
    if not m:
        raise SystemExit(f"无法识别的 Steam 链接或 appid: {s}")
    return int(m.group(1))


# ---------------------------------------------------------------- live data
def fetch_live(appid: int) -> dict:
    itad = ITADClient()
    gid = itad.lookup_steam_appids([appid]).get(appid)
    if not gid:
        raise SystemExit("IsThereAnyDeal 上找不到这个游戏")
    rec = {"appid": appid, "itad_id": gid, "info": itad.info(gid), "history": itad.history(gid)}

    spy = requests.get("https://steamspy.com/api.php", params={"request": "appdetails", "appid": appid},
                       headers={"User-Agent": USER_AGENT}, timeout=60).json()
    store = Paced(0.2)
    det = (store.get("https://store.steampowered.com/api/appdetails", appids=appid, cc="us", l="english")
           or {}).get(str(appid)) or {}
    hist = store.get(f"https://store.steampowered.com/appreviewhistogram/{appid}",
                     l="all", review_score_preference=0) or {}
    return {"rec": rec, "spy": spy, "store": det.get("data") if det.get("success") else None,
            "rollups": (hist.get("results") or {}).get("rollups"), "client": store}


def query_content(appid, live, t0_utc) -> tuple[pd.DataFrame, pd.DataFrame]:
    spy, det = live["spy"], live["store"] or {}
    tags = spy.get("tags") or {}
    tags = pd.DataFrame([(appid, t, int(v)) for t, v in (tags.items() if isinstance(tags, dict) else [])],
                        columns=["appid", "tag", "votes"])
    langs = [x for x in (spy.get("languages") or "").split(",") if x.strip()]
    row = {"appid": appid, "n_languages": len(langs) or None, "has_store": bool(det)}
    cats = {c["description"] for c in det.get("categories") or []}
    for name, col in CATEGORIES.items():
        row[col] = int(name in cats) if det else None
    row["metacritic"] = (det.get("metacritic") or {}).get("score")
    row["required_age"] = pd.to_numeric(det.get("required_age"), errors="coerce") if det else None
    row["m1_reviews_up"], row["m1_reviews_down"] = first_month_reviews(live["rollups"], t0_utc)
    start = int(t0_utc.floor("D").timestamp())
    if pd.Timestamp.now(tz="UTC").timestamp() > start + (WINDOW_DAYS + 1) * 86400:
        qs = (live["client"].get(f"https://store.steampowered.com/appreviews/{appid}",
                                 json=1, filter="all", language="all", purchase_type="all", num_per_page=0,
                                 review_type="all", day_range=9223372036854775807, start_date=start,
                                 end_date=start + WINDOW_DAYS * 86400, date_range_type="include")
              or {}).get("query_summary") or {}
        row["w1_reviews"], row["w1_positive"] = qs.get("total_reviews"), qs.get("total_positive")
    return pd.DataFrame([row]), tags


# ---------------------------------------------------------------- prediction
def prepare(appid: int) -> dict:
    """Fetch live data, merge it into the reference data, and compute the game's features."""
    art = load_artifacts()
    now = pd.Timestamp.now(tz="UTC")
    live = fetch_live(appid)
    q_games, q_changes, q_sales = process_records([(live["rec"], live["spy"])], now=now)
    q = q_games.iloc[0]
    if pd.isna(q["steam_release"]) or q_changes.empty or not q["launch_price"] > 0:
        raise SystemExit("这个游戏还没有 Steam 价格记录（可能未发售或免费），暂时无法预测")
    # Always predict for the queried game, even if it wouldn't qualify as training data
    q_games["usable"] = True
    q_content, q_tags = query_content(appid, live, q["steam_release"])

    ref = load_reference()
    drop = lambda d: d[d["appid"] != appid]  # noqa: E731  (live data replaces any stale copy)
    for key, new in [("games", q_games), ("sales", q_sales), ("changes", q_changes), ("content", q_content),
                     ("tags", q_tags), ("months", review_months(live["rollups"], appid))]:
        ref[key] = pd.concat([drop(ref[key]), new], ignore_index=True)

    feats = game_features(ref, art, only={appid})
    now_pac = now.tz_convert("US/Pacific").tz_localize(None)
    _, states = predict_now(feats, ref, art, now_pac, thresholds=[])
    up = upcoming_majors(ref["majors"], now_pac)
    sale_label = lambda r: f"{SALE_NAMES.get(r['name'], '大促')}（{r['start']:%m/%d} 开始）"  # noqa: E731
    labels = {"next_major": sale_label(up.iloc[0]),
              "second_major": sale_label(up.iloc[1]) if len(up) > 1 else "下下次大促",
              "90": "3 个月内", "180": "半年内", "365": "一年内"}
    launch_price = float(q["launch_price"])
    ratio = states["price_ratio_now"].iat[0]
    return {"appid": appid, "title": q["title"], "launch_price": launch_price, "labels": labels,
            "current_price": launch_price * ratio, "cut_now": (1 - ratio) * 100,
            "n_sales": int(states["n_sales_so_far"].iat[0]), "_ctx": (feats, ref, art, now_pac)}


def probabilities(game: dict, thresholds) -> pd.DataFrame:
    """threshold x horizon probability table; NaN where the price already meets the target."""
    feats, ref, art, now_pac = game["_ctx"]
    rows, _ = predict_now(feats, ref, art, now_pac, thresholds)
    if rows.empty:
        return pd.DataFrame(index=pd.Index(thresholds, name="threshold"))
    t = rows.pivot_table(index="threshold", columns="horizon", values="p", observed=True).reindex(sorted(thresholds))
    return t[rows.groupby("horizon", observed=True)["horizon_days"].first().sort_values().index]


def predict(appid: int, target: float | None = None, target_price: float | None = None) -> dict:
    game = prepare(appid)
    lp = game["launch_price"]
    target = (1 - target_price / lp) * 100 if target is None else target
    result = {k: v for k, v in game.items() if k != "_ctx"} | {"target": target, "target_price": lp * (1 - target / 100)}
    if target <= 0:
        return result | {"verdict": f"目标价不低于首发价 ${lp:.2f}，现在就可以买"}
    if game["cut_now"] >= target - CUT_TOL:
        return result | {"verdict": "现在的价格已经达到你的目标，直接买"}
    p = probabilities(game, [target]).loc[target].drop("second_major", errors="ignore")
    labels = game["labels"]
    return result | {"probs": [(labels[h], float(v)) for h, v in p.items()], "verdict": verdict(p.to_dict(), labels)}


def _likely(p: float) -> str:
    return "很可能" if p >= 0.8 else "大概率" if p >= 0.6 else "有一半以上的机会"


def verdict(p: dict, labels: dict) -> str:
    """Earliest window that more likely than not reaches the target, with honest hedging."""
    nm, p90, p180, p365 = (p.get(k, 0) for k in ["next_major", "90", "180", "365"])
    if nm >= 0.5:
        return f"建议等：{labels['next_major']}{_likely(nm)}能买到"
    if p90 >= 0.5:
        return f"建议等：3 个月内{_likely(p90)}降到这个价"
    if p180 >= 0.5:
        return f"要等 3～6 个月（{_likely(p180)}）；不想等的话可以直接买"
    if p365 >= 0.5:
        return f"要等半年到一年（{_likely(p365)}）；不想等这么久的话建议直接买"
    if p365 >= 0.35:
        return "一年内有机会但说不准，不想赌的话建议直接买"
    return "一年内降到这个价的可能性不大，想玩的话建议直接买"


TABLE_CUTS = [20, 25, 30, 40, 50, 60]


def _pad(s: str, width: int, right: bool = False) -> str:
    """Pad to a terminal width, counting CJK characters as two columns."""
    w = sum(2 if ord(ch) > 0x2E7F else 1 for ch in s)
    fill = " " * max(0, width - w)
    return fill + s if right else s + fill


def print_table(appid: int):
    game = prepare(appid)
    t = probabilities(game, TABLE_CUTS)
    cols = [c for c in ["next_major", "second_major"] if c in t.columns]
    lp, labels = game["launch_price"], game["labels"]
    print(f"\n{game['title']}")
    status = "还没打过折" if game["n_sales"] == 0 else f"已打过 {game['n_sales']} 次折"
    print(f"首发价 ${lp:.2f} · 现价 ${game['current_price']:.2f} · {status}\n")
    print(_pad("到大促结束时至少打到", 22) + "".join(_pad(labels[c], 24, right=True) for c in cols))
    for cut in TABLE_CUTS:
        label = f"  {10 - cut / 10:g} 折（${lp * (1 - cut / 100):.2f}）"
        cells = "".join(_pad("已达到" if pd.isna(t.loc[cut, c]) else f"{t.loc[cut, c]:.0%}", 24, right=True)
                        for c in cols)
        print(_pad(label, 22) + cells)
    print("\n第二列为累计概率：到那次大促结束为止是否打到过该折扣")


def main():
    ap = argparse.ArgumentParser(description="Steam 折扣预测：值不值得等？")
    ap.add_argument("game", help="Steam 商店链接或 appid")
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--target", type=float, help="想要的折扣，例如 50 表示五折（相对首发价）")
    grp.add_argument("--target-price", type=float, help="想要的价格（美元）")
    grp.add_argument("--table", action="store_true", help="各档折扣在下次 / 下下次大促的概率")
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if args.table:
        print_table(parse_appid(args.game))
        print("价格数据来自 IsThereAnyDeal · 预测仅供参考")
        return
    r = predict(parse_appid(args.game), args.target, args.target_price)
    print(f"\n{r['title']}")
    print(f"首发价 ${r['launch_price']:.2f} · 现价 ${r['current_price']:.2f} · "
          f"目标 ${r['target_price']:.2f}（{r['target']:.0f}% off）\n")
    print(f"结论：{r['verdict']}\n")
    for label, p in r.get("probs", []):
        print(f"  {label:<16} {p:>4.0%}")
    print("\n价格数据来自 IsThereAnyDeal · 预测仅供参考")


if __name__ == "__main__":
    main()
