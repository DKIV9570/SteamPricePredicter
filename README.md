# 等不等 · Wait or Buy

**Steam 折扣预测：你想要的折扣，要等多久？**
**Steam discount forecasts: how long until the price you want?**

🌐 **https://dkiv9570.github.io/SteamPricePredicter/**

[中文](#中文) · [English](#english)

---

## 中文

想买一款游戏，但觉得现价太贵？输入游戏和你的心理价位，网站告诉你：

> **鬼武者 Way of the Sword**　想等到 8 折（示例，数字每天更新）
> 秋促（10/1）5% · 冬促（12/17）54% · 半年内 …
> **结论：冬促大约一半机会；想五折以下买的话要等很久，想玩不如现在买。**

- 支持中英文游戏名（包括日系游戏的中文叫法，如"空之轨迹"）、appid 或直接粘贴 Steam 商店链接
- 可以选折扣档位（8 折～2 折），也可以直接输入目标价格
- 覆盖约 4.2 万款 2014 年以后发售的付费游戏，每天自动更新价格、加入新游戏

### 准不准？

在模型训练时**完全没见过**的游戏（2023 年 7 月以后发售的约 1.07 万款）上测试，模拟用户在不同时间点查询：

| 给出的结论正确的比例 | 查表基线 | 本模型 |
|---|---|---|
| 全部游戏 | 68.6% | **75.6%** |
| 首发价 ≥ $20 的游戏 | 72.3% | **79.3%** |
| 其中：还没打过折的游戏 | 67.3% | **76.9%** |
| 按用户关注度加权（热门游戏权重更高） | 71.8% | **80.5%** |

- "结论正确"指：说"下次大促能买到"时真的在那之前降到了目标价；说"建议直接买"时一年内确实没降到
- **概率是可信的**：模型说 X% 的时候，实际发生的比例与 X 相差不到 2 个百分点
- 查表基线：按目标折扣、时间范围、游戏年龄、历史最低折扣分组，取历史上的达成率
- 完整结果见 [reports/](reports/)（AUC 0.920，查表基线 0.876）

### 原理

- **数据**：[IsThereAnyDeal](https://isthereanydeal.com) 的 Steam 价格历史，SteamSpy 用户标签，Steam 商店信息与评测数量。收录范围来自 Steam 热销榜、热门新品和 ITAD 愿望单榜
- **模型**：LightGBM 预测"在某个时间窗口内，价格（相对首发价）会不会至少降到过一次目标折扣"。主要特征：
  - 大促日历，以及这款游戏以往参加各类大促的情况（比如从不参加秋促）
  - 当前的折扣状态、历史最深和最浅折扣、以往打折达到目标折扣的比例
  - 发行商和开发商的历史折扣习惯，相似游戏（按用户标签）的折扣轨迹
  - 首发价格、首周评测数、评测数随时间的变化（销量的替代指标）
- **大促日历**从价格数据中自动识别历年大促，未来日期取自 Valve 公布的日程

更多分析见 [analysis/](analysis/)，比如方差分解表明：打几折这件事，发行商和游戏本身各占一半左右；同一款游戏每次打折之间的随机波动只占约 15%。

### 本地运行

原始数据来自第三方（IsThereAnyDeal、SteamSpy、Steam），不在本仓库中公开，需要用自己的 ITAD key 采集。全部采集完大约需要一天（主要受各接口的限速影响），之后每天增量更新只要几分钟。

```bash
pip install -r requirements.txt
echo "ITAD_API_KEY=<你的 key>" > .env        # 在 https://isthereanydeal.com/apps/my/ 申请

# 采集（均可中断后续跑）
python -m pricepredicter.fetch_catalog --pages 60      # SteamSpy 游戏目录
python -m pricepredicter.fetch_itad                    # ITAD 价格历史
python -m pricepredicter.build_dataset
python -m pricepredicter.fetch_steamspy_details        # 用户标签
python -m pricepredicter.fetch_steam_store             # 商店信息、评测时间线
python -m pricepredicter.fetch_first_week_reviews      # 首周评测数
python -m pricepredicter.fetch_names                   # 中文名
python -m pricepredicter.build_dataset && python -m pricepredicter.build_content

# 训练（models/ 里已有训练好的模型，可跳过）
python analysis/05_reach_model.py

# 命令行查询
python -m pricepredicter.predict https://store.steampowered.com/app/2186680 --target 50
python -m pricepredicter.predict 2638890 --table

# 生成网站数据并本地预览
python -m pricepredicter.export_site
python -m http.server 8000 -d site
```

### 自动更新与维护

[.github/workflows/daily.yml](.github/workflows/daily.yml) 每天运行 `python -m pricepredicter.daily`：发现新游戏、刷新价格、补抓数据、重新预测，然后发布到 GitHub Pages。原始数据以压缩包形式保存在一个**私有**仓库的 Release 里。

需要的仓库密钥（Settings → Secrets and variables → Actions）：
- `ITAD_API_KEY`
- `DATA_REPO_TOKEN`：fine-grained token，只对数据仓库开放 Contents 读写权限。**过期后每日任务会失败并邮件通知**，重新生成后运行 `gh secret set DATA_REPO_TOKEN` 更新即可

---

## English

Want a game but the price feels too high? Pick the game and the discount (or price) you'd wait for, and get a verdict:

> **Onimusha: Way of the Sword**, waiting for 20% off (example; numbers update daily)
> Autumn Sale (Oct 1) 5% · Winter Sale (Dec 17) 54% · within 6 months …
> **Verdict: a coin flip by the Winter Sale; 50% off is a long way off — if you want to play it, buy now.**

- Search by English, Chinese or Japanese title, appid, or a pasted Steam store link
- ~42k paid games released since 2014; prices and new releases update daily

### How accurate?

Tested on ~10.7k games released after July 2023 that the model never saw during training, asking at different points in each game's life:

| Verdict correct | Lookup baseline | Model |
|---|---|---|
| All games | 68.6% | **75.6%** |
| Launch price ≥ $20 | 72.3% | **79.3%** |
| …not yet discounted | 67.3% | **76.9%** |
| Weighted by interest (ITAD waitlists) | 71.8% | **80.5%** |

- A verdict is correct if "wait for the next sale" is followed by the target price by then, or "buy now" by no such price within a year
- **The probabilities are calibrated**: when it says X%, it happens within 2 points of X% of the time
- Lookup baseline: the historical hit rate for the same target, window, game age and discount history
- Full results in [reports/](reports/) (AUC 0.920 vs. 0.876 for the baseline)

### How it works

- **Data**: Steam price history from [IsThereAnyDeal](https://isthereanydeal.com), SteamSpy user tags, Steam store details and review counts; games are picked from Steam's top sellers and popular new releases and ITAD's most-waitlisted list
- **Model**: LightGBM estimating whether the price (vs. launch price) hits the target at least once within a window, from:
  - the sale calendar and which kinds of major sales this game has joined before (e.g. never the Autumn Sale)
  - its discount state, deepest and shallowest past sales, and how many past sales reached the target
  - the publisher's and developer's past behaviour, and how similar games (by user tags) were discounted
  - launch price, first-week reviews, and reviews over time (a proxy for sales)
- **Sale calendar**: past major sales are detected from the price data; upcoming dates come from Valve's published schedule

### Run locally

The raw data is third-party (IsThereAnyDeal, SteamSpy, Steam) and isn't redistributed here; collect it with your own ITAD API key using the commands in the Chinese section above (a full collection takes about a day because of rate limits; daily updates take minutes).

### Automation

[.github/workflows/daily.yml](.github/workflows/daily.yml) runs `python -m pricepredicter.daily` every day and publishes to GitHub Pages. Raw data is kept as an archive on a release of a **private** repo. Secrets: `ITAD_API_KEY`, and `DATA_REPO_TOKEN` (fine-grained, Contents read/write on the data repo only — when it expires the workflow fails with an email; renew it with `gh secret set DATA_REPO_TOKEN`).

---

Price data from [IsThereAnyDeal](https://isthereanydeal.com). Not affiliated with Valve, Steam or IsThereAnyDeal.
Forecasts, not promises. Released under the [MIT License](LICENSE).
