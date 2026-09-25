"""Search names for the website: what to display, and hidden aliases to match against.

Many Japanese games have no Chinese store name, but Chinese players know them by a Chinese
reading of the Japanese title — 空の軌跡 is searched as 空之轨迹. So Japanese and Traditional
Chinese names are also converted to Simplified Chinese (with の -> 之) as search aliases.
"""
from opencc import OpenCC

_t2s = OpenCC("t2s")
# Japanese shinjitai that traditional->simplified conversion leaves alone
_SHINJITAI = str.maketrans({
    "伝": "传", "戦": "战", "楽": "乐", "験": "验", "変": "变", "剣": "剑", "竜": "龙", "闘": "斗",
    "帰": "归", "険": "险", "図": "图", "黒": "黑", "転": "转", "覚": "觉", "桜": "樱", "気": "气",
    "獣": "兽", "蔵": "藏", "読": "读", "満": "满", "継": "继", "総": "总", "対": "对", "実": "实",
    "悪": "恶", "鉄": "铁", "発": "发", "広": "广", "県": "县", "駅": "驿", "姫": "姬", "廃": "废",
    "来": "来", "巻": "卷", "塁": "垒", "聖": "圣", "録": "录", "撃": "击", "弾": "弹", "営": "营",
    "労": "劳", "鬪": "斗", "譲": "让", "暦": "历", "闇": "暗", "拡": "扩", "権": "权", "様": "样",
})


def to_simplified(name: str) -> str:
    return _t2s.convert(name.translate(_SHINJITAI)).replace("の", "之")


def display_and_aliases(title: str, names: dict) -> tuple[str, list[str]]:
    """names: {"schinese": .., "tchinese": .., "japanese": ..} from fetch_names.
    Returns (Chinese display name or "", extra search aliases)."""
    sc, tc, ja = (names.get(k) or "" for k in ["schinese", "tchinese", "japanese"])
    display = sc if sc and sc != title else (tc if tc and tc != title else "")
    seen = {title, display}
    aliases = []
    for n in [sc, tc, to_simplified(tc) if tc else "", ja, to_simplified(ja) if ja else ""]:
        if n and n not in seen:
            seen.add(n)
            aliases.append(n)
    return display, aliases
