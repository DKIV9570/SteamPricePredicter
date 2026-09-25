"""Render site/og.png (1200x630), the preview image shown when the site's link is shared.

    python tools/make_og_image.py
"""
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parent.parent / "site" / "og.png"

HTML = """<!doctype html><meta charset="utf-8"><style>
* { box-sizing: border-box; margin: 0; }
body { width: 1200px; height: 630px; background: #f7f6f3; color: #16161a; display: flex; align-items: center;
       gap: 56px; padding: 0 72px; font-family: "Segoe UI", "Microsoft YaHei", "PingFang SC", sans-serif; }
.left { flex: 1; }
.logo { font-size: 72px; }
h1 { font-size: 84px; letter-spacing: 2px; margin-top: 8px; }
h1 span { color: #8a8a92; font-weight: 400; font-size: 44px; margin-left: 16px; letter-spacing: 0; }
.tag { font-size: 38px; color: #55555c; margin-top: 18px; }
.sub { font-size: 24px; color: #8a8a92; margin-top: 28px; line-height: 1.5; }
.card { width: 470px; background: #fff; border: 1px solid #e4e3de; border-radius: 22px; padding: 30px 32px;
        box-shadow: 0 12px 40px rgba(0,0,0,.08); }
.game { font-size: 24px; font-weight: 650; }
.target { font-size: 18px; color: #8a8a92; margin-top: 6px; }
.verdict { font-size: 27px; font-weight: 700; color: #2a78d6; margin: 22px 0 18px; }
.row { display: grid; grid-template-columns: 120px 1fr 56px; gap: 14px; align-items: center; margin-top: 13px;
       font-size: 19px; color: #55555c; }
.track { height: 12px; background: #ecebe7; border-radius: 99px; overflow: hidden; }
.fill { height: 100%; background: #2a78d6; border-radius: 99px; }
.pct { text-align: right; font-weight: 700; color: #16161a; }
</style>
<div class="left">
  <div class="logo">⏳</div>
  <h1>等不等<span>Wait or Buy</span></h1>
  <div class="tag">想要的折扣，要等多久？</div>
  <div class="sub">Steam 折扣预测 · 4 万款游戏 · 每日更新<br>How long until the Steam discount you want?</div>
</div>
<div class="card">
  <div class="game">Warhammer 40,000: Rogue Trader</div>
  <div class="target">我想等到 5 折</div>
  <div class="verdict">建议等：秋促很可能能买到</div>
  <div class="row"><span>秋促</span><span class="track"><span class="fill" style="display:block;width:96%"></span></span><span class="pct">96%</span></div>
  <div class="row"><span>3 个月内</span><span class="track"><span class="fill" style="display:block;width:98%"></span></span><span class="pct">98%</span></div>
  <div class="row"><span>半年内</span><span class="track"><span class="fill" style="display:block;width:99%"></span></span><span class="pct">99%</span></div>
</div>"""

with sync_playwright() as p:
    browser = p.chromium.launch(channel="chrome")
    page = browser.new_page(viewport={"width": 1200, "height": 630})
    page.set_content(HTML)
    page.screenshot(path=str(OUT))
    browser.close()
print(f"wrote {OUT}")
