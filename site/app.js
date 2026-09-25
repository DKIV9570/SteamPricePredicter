"use strict";
// Static front end: all predictions are precomputed daily into data/*.json.

const I18N = {
  zh: {
    title: "等不等", tagline: "想要的折扣，要等多久？", searchLabel: "搜索游戏",
    searchPlaceholder: "游戏名、appid 或 Steam 商店链接",
    emptyHint: "搜索一款游戏，看看你想要的折扣大概什么时候会来。", try: "试试：",
    notFound: "数据库里还没有这个游戏。目前收录约 4 万款 2014 年以后发售的付费游戏。",
    noMatch: "没找到这个游戏。目前收录约 4 万款 2014 年以后发售的付费游戏，新游戏会在发售后自动加入。",
    targetLabel: "我想等到", orPrice: "或者目标价",
    cumulative: "概率为累计值：到该时间点为止，价格是否至少降到过一次目标价。折扣按首发价计算，永久降价也算。",
    data: "价格数据来自", usd: "美区价格（美元）；Steam 大促的折扣比例各区基本一致",
    updated: "数据更新于", disclaimer: "预测仅供参考",
    cut: c => `${+(10 - c / 10).toFixed(1)} 折`,
    prices: (lp, cp) => cp < lp - 0.005 ? `首发价 <b>$${lp.toFixed(2)}</b> · 现价 <b>$${cp.toFixed(2)}</b>（-${Math.round((1 - cp / lp) * 100)}%）`
                                         : `首发价 <b>$${lp.toFixed(2)}</b> · 现价 <b>$${cp.toFixed(2)}</b>`,
    status: (r) => (r.ns === 0 && r.best === 0 ? "还没打过折" : `已打过 ${r.ns} 次折 · 最低到过 ${r.best}% off`) + ` · 发售于 ${r.rd}`,
    sale: { summer: "夏促", winter: "冬促", autumn: "秋促", spring: "春促", lunar_new_year: "春节促销", halloween: "万圣节促销", other: "大促" },
    saleLabel: (name, d) => `${name}（${d.getMonth() + 1}/${d.getDate()} 开始）`,
    within: { "90": "3 个月内", "180": "半年内", "365": "一年内" },
    likely: p => p >= 80 ? "很可能" : p >= 60 ? "大概率" : "有一半以上的机会",
    vSale: (label, lk) => `建议等：${label}${lk}能买到`,
    v90: lk => `建议等：3 个月内${lk}降到这个价`,
    v180: lk => `要等 3～6 个月（${lk}）；不想等的话可以直接买`,
    v365: lk => `要等半年到一年（${lk}）；不想等这么久的话建议直接买`,
    vMaybe: "一年内有机会但说不准，不想赌的话建议直接买",
    vNo: "一年内降到这个价的可能性不大，想玩的话建议直接买",
    vNow: "现在的价格已经达到你的目标，直接买",
    vAbove: "目标价不低于首发价，现在就可以买",
    vRange: "目标太深了：只预测到 2 折（80% off）",
    targetPrice: (p, c) => `目标 $${p.toFixed(2)}（${c}% off）`,
  },
  en: {
    title: "Wait or Buy", tagline: "How long until the discount you want?", searchLabel: "Search games",
    searchPlaceholder: "Game name, appid or Steam store link",
    emptyHint: "Search a game to see when the discount you want is likely to arrive.", try: "Try:",
    notFound: "This game isn't in the database yet. It covers ~40k paid games released since 2014.",
    noMatch: "No match. The database covers ~40k paid games released since 2014; new releases are added automatically.",
    targetLabel: "I'd wait for", orPrice: "or a target price",
    cumulative: "Probabilities are cumulative: will the price hit the target at least once by then. Discounts are measured against the launch price, so permanent price cuts count.",
    data: "Price data from", usd: "US prices (USD); Steam sale percentages are mostly the same in every region",
    updated: "Updated", disclaimer: "Forecasts, not promises",
    cut: c => `${c}% off`,
    prices: (lp, cp) => cp < lp - 0.005 ? `Launch <b>$${lp.toFixed(2)}</b> · Now <b>$${cp.toFixed(2)}</b> (-${Math.round((1 - cp / lp) * 100)}%)`
                                         : `Launch <b>$${lp.toFixed(2)}</b> · Now <b>$${cp.toFixed(2)}</b>`,
    status: (r) => (r.ns === 0 && r.best === 0 ? "Never discounted yet" : `Discounted ${r.ns} times · deepest ${r.best}% off`) + ` · Released ${r.rd}`,
    sale: { summer: "Summer Sale", winter: "Winter Sale", autumn: "Autumn Sale", spring: "Spring Sale", lunar_new_year: "Lunar New Year Sale", halloween: "Halloween Sale", other: "Major sale" },
    saleLabel: (name, d) => `${name} (from ${d.toLocaleDateString("en-US", { month: "short", day: "numeric" })})`,
    within: { "90": "Within 3 months", "180": "Within 6 months", "365": "Within a year" },
    likely: p => p >= 80 ? "very likely" : p >= 60 ? "likely" : "more likely than not",
    vSale: (label, lk) => `Wait: ${lk} by the ${label}`,
    v90: lk => `Wait: ${lk} within 3 months`,
    v180: lk => `Expect 3–6 months (${lk}); buy now if you don't want to wait`,
    v365: lk => `Expect 6–12 months (${lk}); buy now if that's too long`,
    vMaybe: "Possible within a year, but uncertain — buy now if you don't want to gamble",
    vNo: "Unlikely within a year — if you want to play it, buy now",
    vNow: "The price is already at your target — buy now",
    vAbove: "That's not below the launch price — buy now",
    vRange: "Too deep: forecasts go up to 80% off",
    targetPrice: (p, c) => `target $${p.toFixed(2)} (${c}% off)`,
  },
};

const $ = id => document.getElementById(id);
const state = { lang: "zh", meta: null, index: null, shards: {}, appid: null, rec: null, cut: 50, price: null };
try { state.lang = localStorage.getItem("lang") || (navigator.language.startsWith("zh") ? "zh" : "en"); } catch (_) {}
const t = () => I18N[state.lang];

// ---------------------------------------------------------------- data
const getJSON = url => fetch(url).then(r => { if (!r.ok) throw new Error(url); return r.json(); });
async function loadIndex() {
  if (!state.index) {
    // [appid, title, chinese display name or "", ...hidden search aliases]
    state.index = (await getJSON("data/index.json")).map(([id, title, zh = "", ...aliases]) => {
      const names = [title, zh, ...aliases].filter(Boolean).map(norm);
      return { id, title, zh, names, compact: names.map(n => n.replace(/ /g, "")) };
    });
  }
  return state.index;
}
async function loadGame(appid) {
  const k = appid % state.meta.shards;
  if (!state.shards[k]) state.shards[k] = await getJSON(`data/g/${k}.json`);
  return state.shards[k][appid] || null;
}

// ---------------------------------------------------------------- search
const norm = s => s.toLowerCase().normalize("NFKD").replace(/[^\p{L}\p{N}]+/gu, " ").trim();
function parseQuery(q) {
  const m = q.match(/\/app\/(\d+)/) || q.match(/^\s*(\d+)\s*$/);
  return m ? +m[1] : null;
}
// Rank: 0 = a name starts with the query, 1 = a word starts with it, 2 = found inside, -1 = no match.
// Spaces are optional ("darksouls", "艾尔登 法环"); ties keep the popularity order of the index.
function matchRank(row, q, tokens) {
  const qc = q.replace(/ /g, "");
  let best = -1;
  row.names.forEach((n, i) => {
    let r = -1;
    if (n.startsWith(q) || row.compact[i].startsWith(qc)) r = 0;
    else if (tokens.every(tk => (" " + n).includes(" " + tk))) r = 1;
    else if (tokens.every(tk => n.includes(tk)) || row.compact[i].includes(qc)) r = 2;
    if (r >= 0 && (best < 0 || r < best)) best = r;
  });
  return best;
}
async function suggest(q) {
  const box = $("suggest");
  const id = parseQuery(q);
  const nq = norm(q);
  if (!nq || id) { box.hidden = true; $("nomatch").hidden = true; return; }
  const idx = await loadIndex();
  const tokens = nq.split(" ");
  const buckets = [[], [], []];
  for (const row of idx) {
    const r = matchRank(row, nq, tokens);
    if (r >= 0 && buckets[r].length < 8) buckets[r].push(row);
    if (buckets[0].length >= 8) break;
  }
  const hits = buckets.flat().slice(0, 8);
  box.innerHTML = "";
  hits.forEach((row, i) => {
    const li = document.createElement("li");
    li.setAttribute("role", "option");
    li.setAttribute("aria-selected", i === 0 ? "true" : "false");
    li.dataset.appid = row.id;
    li.innerHTML = `<span class="names"><span class="primary"></span><span class="alt"></span></span><span class="id">${row.id}</span>`;
    const [primary, alt] = state.lang === "zh" && row.zh ? [row.zh, row.title] : [row.title, row.zh];
    li.querySelector(".primary").textContent = primary;
    li.querySelector(".alt").textContent = alt || "";
    li.addEventListener("mousedown", e => { e.preventDefault(); pick(row.id); });
    box.appendChild(li);
  });
  box.hidden = hits.length === 0;
  $("nomatch").hidden = hits.length > 0;
}
function pick(appid) {
  $("suggest").hidden = true;
  $("nomatch").hidden = true;
  $("q").blur();
  setHash(appid, state.cut, null);
}

// ---------------------------------------------------------------- prediction logic
function windows() {
  const today = new Date();
  const sales = state.meta.sales.map(s => ({ ...s, date: new Date(s.start + "T00:00:00") }));
  const days = d => (d - today) / 864e5 + 14;  // "by the end of the sale" ~ start + 2 weeks
  return state.meta.windows.map((w, i) => {
    if (w === "next_major" || w === "second_major") {
      const s = sales[w === "next_major" ? 0 : 1];
      return s ? { i, w, days: days(s.date), label: t().saleLabel(t().sale[s.name] || t().sale.other, s.date), sale: true }
               : null;
    }
    return { i, w, days: +w, label: t().within[w], sale: false };
  }).filter(Boolean).sort((a, b) => a.days - b.days);
}

// probabilities (percent) per window index for any target cut; null = already there
function probsFor(rec, cut) {
  const cuts = state.meta.cuts;
  const at = c => rec.p[cuts.indexOf(c)];
  if (cuts.includes(cut)) return at(cut);
  const hi = cuts.find(c => c > cut), lo = [...cuts].reverse().find(c => c < cut);
  if (hi === undefined) return undefined;
  const pHi = at(hi);
  const pLo = lo === undefined ? null : at(lo);
  if (pHi === null) return null;
  const w = lo === undefined ? 1 : (cut - lo) / (hi - lo);
  return pHi.map((v, i) => v === null ? null : Math.round((pLo === null ? 100 : pLo[i] ?? 100) * (1 - w) + v * w));
}

function verdict(rec, cut) {
  if (cut <= 0) return [t().vAbove, "now"];
  const curCut = (1 - rec.cp / rec.lp) * 100;
  if (curCut >= cut - 1) return [t().vNow, "now"];
  const p = probsFor(rec, cut);
  if (p === undefined) return [t().vRange, "buy"];
  if (p === null) return [t().vNow, "now"];
  for (const w of windows()) {
    const v = p[w.i];
    if (v === null || v < 50) continue;
    const lk = t().likely(v);
    if (w.sale) return [t().vSale(w.label, lk), "wait"];
    if (w.w === "90") return [t().v90(lk), "wait"];
    if (w.w === "180") return [t().v180(lk), "buy"];
    return [t().v365(lk), "buy"];
  }
  const p365 = p[state.meta.windows.indexOf("365")] ?? 0;
  return [p365 >= 35 ? t().vMaybe : t().vNo, "buy"];
}

// ---------------------------------------------------------------- render
function renderStatic() {
  document.documentElement.lang = state.lang === "zh" ? "zh-CN" : "en";
  document.title = state.lang === "zh" ? "等不等 · Steam 折扣预测" : "Wait or Buy · Steam discount forecasts";
  document.querySelectorAll("[data-i18n]").forEach(el => { el.textContent = t()[el.dataset.i18n]; });
  document.querySelectorAll("[data-i18n-placeholder]").forEach(el => { el.placeholder = t()[el.dataset.i18nPlaceholder]; });
  $("lang").textContent = state.lang === "zh" ? "EN" : "中文";
  if (state.meta) {
    $("updated").textContent = new Date(state.meta.updated).toLocaleString(state.lang === "zh" ? "zh-CN" : "en-US",
      { year: "numeric", month: "short", day: "numeric" });
  }
}

function renderResult() {
  const rec = state.rec;
  $("empty").hidden = !!state.appid;
  $("notfound").hidden = !(state.appid && !rec);
  $("result").hidden = !rec;
  if (!rec) return;

  const store = `https://store.steampowered.com/app/${state.appid}/`;
  $("cover").src = `https://cdn.cloudflare.steamstatic.com/steam/apps/${state.appid}/header.jpg`;
  const [primary, alt] = state.lang === "zh" && rec.zh ? [rec.zh, rec.t] : [rec.t, rec.zh];
  $("title").textContent = primary;
  $("title").href = store;
  $("alttitle").textContent = alt || "";
  $("alttitle").hidden = !alt;
  $("prices").innerHTML = t().prices(rec.lp, rec.cp);
  $("status").textContent = t().status(rec);

  const cut = state.price !== null ? Math.round((1 - state.price / rec.lp) * 100) : state.cut;
  const cuts = $("cuts");
  cuts.innerHTML = "";
  for (const c of state.meta.cuts) {
    const b = document.createElement("button");
    b.type = "button";
    b.setAttribute("role", "radio");
    b.setAttribute("aria-checked", state.price === null && c === state.cut ? "true" : "false");
    b.textContent = t().cut(c);
    b.title = `$${(rec.lp * (1 - c / 100)).toFixed(2)}`;
    b.addEventListener("click", () => setHash(state.appid, c, null));
    cuts.appendChild(b);
  }
  if (document.activeElement !== $("tprice")) {
    $("tprice").value = (state.price ?? rec.lp * (1 - cut / 100)).toFixed(2);
  }

  const [text, cls] = verdict(rec, cut);
  $("verdict").textContent = text;
  $("verdict").className = `verdict ${cls}`;

  const bars = $("bars");
  bars.innerHTML = "";
  const p = cls === "now" ? null : probsFor(rec, cut);
  if (p) {
    for (const w of windows()) {
      const v = p[w.i];
      if (v === null || v === undefined) continue;
      const li = document.createElement("li");
      li.innerHTML = `<span class="label"></span><span class="track"><span class="fill"></span></span><span class="pct"></span>`;
      li.querySelector(".label").textContent = w.label;
      li.querySelector(".fill").style.width = `${v}%`;
      li.querySelector(".pct").textContent = `${v}%`;
      bars.appendChild(li);
    }
  }
}

// ---------------------------------------------------------------- routing
function setHash(appid, cut, price) {
  const h = `#app=${appid}` + (price !== null ? `&price=${price}` : `&cut=${cut}`);
  if (location.hash !== h) location.hash = h; else route();
}
async function route() {
  const h = new URLSearchParams(location.hash.slice(1));
  const appid = +h.get("app") || null;
  state.cut = +h.get("cut") || state.cut;
  state.price = h.has("price") ? +h.get("price") : null;
  state.appid = appid;
  state.rec = appid ? await loadGame(appid) : null;
  renderResult();
}

// ---------------------------------------------------------------- wiring
async function init() {
  renderStatic();
  state.meta = await getJSON("data/meta.json");
  renderStatic();
  const q = $("q");
  let timer;
  q.addEventListener("input", () => { clearTimeout(timer); timer = setTimeout(() => suggest(q.value), 120); });
  q.addEventListener("focus", () => { loadIndex(); if (q.value) suggest(q.value); });
  q.addEventListener("blur", () => setTimeout(() => { $("suggest").hidden = true; }, 150));
  q.addEventListener("keydown", e => {
    const items = [...$("suggest").querySelectorAll("li")];
    const cur = items.findIndex(li => li.getAttribute("aria-selected") === "true");
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      if (!items.length) return;
      const next = (cur + (e.key === "ArrowDown" ? 1 : -1) + items.length) % items.length;
      items.forEach((li, i) => li.setAttribute("aria-selected", i === next ? "true" : "false"));
    } else if (e.key === "Enter") {
      const id = parseQuery(q.value);
      if (id) pick(id);
      else if (items[cur]) pick(+items[cur].dataset.appid);
    } else if (e.key === "Escape") {
      $("suggest").hidden = true;
    }
  });
  $("tprice").addEventListener("change", e => {
    const v = parseFloat(e.target.value);
    if (state.appid && v >= 0) setHash(state.appid, state.cut, +v.toFixed(2));
  });
  $("lang").addEventListener("click", () => {
    state.lang = state.lang === "zh" ? "en" : "zh";
    try { localStorage.setItem("lang", state.lang); } catch (_) {}
    renderStatic();
    renderResult();
  });
  window.addEventListener("hashchange", route);
  route();
}
init();
