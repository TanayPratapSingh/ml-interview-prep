/* ML Interview Prep: a static, no build single page app.
   Content comes from content/<module>/*.md and cards.json; the daily feed from data/daily and data/jobs.
   Card progress lives in localStorage under "mip.v1" and can be exported from #/progress. */
(() => {
  "use strict";

  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => [...el.querySelectorAll(s)];
  const main = $("#main");
  const EXTRA_CARD_SOURCES = ["lectures"];
  const LEVELS = ["core", "advanced", "expert"];
  const KINDS = ["concept", "why", "compare", "formula", "example", "code", "interview"];
  const RUBRIC = ["Led with the answer in the first sentence", "Gave one concrete example or number",
    "Named a trade off or a caveat", "Stopped on time without a recap", "No filler openers (so, um, basically)"];

  // ------------------------------------------------------------ state
  const KEY = "mip.v1";
  const blank = () => ({ cards: {}, drills: [], reviews: {}, theme: null });
  let state = blank();
  try { state = Object.assign(blank(), JSON.parse(localStorage.getItem(KEY)) || {}); } catch (e) { /* private mode */ }
  const save = () => { try { localStorage.setItem(KEY, JSON.stringify(state)); } catch (e) { /* ignore */ } };

  function applyTheme() {
    if (state.theme) document.documentElement.dataset.theme = state.theme;
    else delete document.documentElement.dataset.theme;
  }
  $("#theme").addEventListener("click", () => {
    const cur = state.theme || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    state.theme = cur === "dark" ? "light" : "dark";
    save(); applyTheme();
  });
  applyTheme();

  // ------------------------------------------------------------ helpers
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const safeUrl = (u) => (/^https?:\/\//i.test(u || "") ? u : "#");
  const link = (u, text, cls = "") => `<a class="${cls}" href="${esc(safeUrl(u))}" target="_blank" rel="noopener">${esc(text)}</a>`;
  const plural = (n, w) => `${n} ${w}${n === 1 ? "" : "s"}`;
  const slug = (s) => s.toLowerCase().replace(/<[^>]+>/g, "").replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
  const pad = (n) => String(n).padStart(2, "0");
  const today = () => { const d = new Date(); return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`; };
  const addDays = (s, n) => { const [y, m, d] = s.split("-").map(Number); return new Date(Date.UTC(y, m - 1, d + n)).toISOString().slice(0, 10); };
  const dayIndex = (s) => { const [y, m, d] = s.split("-").map(Number); return Math.round((Date.UTC(y, m - 1, d) - Date.UTC(2026, 0, 1)) / 864e5); };
  const shuffle = (a) => { a = [...a]; for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; };
  const fmtTime = (iso) => { if (!iso) return ""; const d = new Date(iso); return isNaN(d) ? "" : d.toLocaleString([], { dateStyle: "medium", timeStyle: "short" }); };

  let cleanups = [];
  const onCleanup = (fn) => cleanups.push(fn);

  // ------------------------------------------------------------ data
  const cache = new Map();
  function getJSON(path, fresh = false) {
    if (!fresh && cache.has(path)) return cache.get(path);
    const p = fetch(fresh ? `${path}?v=${Date.now()}` : path, { cache: fresh ? "no-store" : "default" })
      .then((r) => { if (!r.ok) throw new Error(`${r.status} for ${path}`); return r.json(); });
    cache.set(path, p);
    p.catch(() => cache.delete(path));
    return p;
  }
  function getText(path) {
    if (cache.has(path)) return cache.get(path);
    const p = fetch(path).then((r) => (r.ok ? r.text() : null)).catch(() => null);
    cache.set(path, p);
    return p;
  }
  const modules = () => getJSON("data/modules.json");
  async function moduleCards(id) {
    try { return (await getJSON(`content/${id}/cards.json`)).map((c) => ({ ...c, module: id })); } catch (e) { return []; }
  }
  async function allCards() {
    const ids = (await modules()).filter((m) => m.ready).map((m) => m.id).concat(EXTRA_CARD_SOURCES);
    const lists = await Promise.all(ids.map(moduleCards));
    return lists.flat().sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
  }
  // Curriculum order: modules in data/modules.json order, cards in id order inside each module.
  async function curriculum() {
    const mods = await modules();
    const order = Object.fromEntries(mods.map((m, i) => [m.id, i]).concat([["lectures", mods.length]]));
    const path = state.path && state.path.length ? new Set(state.path) : null;
    return (await allCards()).filter((c) => !path || path.has(c.module))
      .sort((a, b) => (order[a.module] - order[b.module]) || (a.id < b.id ? -1 : 1));
  }

  async function moduleTitles() {
    const t = { lectures: "Lecture notes" };
    (await modules()).forEach((m) => { t[m.id] = m.title; });
    return t;
  }

  // Same function as drill_pick in scripts/digest.py, so the issue and the page agree.
  function drillPick(cards, date, n = 5) {
    const out = [], used = new Set(), day = dayIndex(date);
    let i = 0;
    while (out.length < Math.min(n, cards.length)) {
      const idx = (((day * 7919 + i * 104729) % cards.length) + cards.length) % cards.length;
      i++;
      if (used.has(idx)) continue;
      used.add(idx); out.push(cards[idx]);
    }
    return out;
  }

  // ------------------------------------------------------------ markdown with math
  if (window.marked) marked.setOptions({ gfm: true, breaks: false });
  function md(src) {
    if (!src) return "";
    if (!window.marked) return `<pre>${esc(src)}</pre>`;
    const maths = [];
    const parts = src.split(/(```[\s\S]*?```|`[^`\n]+`)/g);
    for (let i = 0; i < parts.length; i += 2) {
      parts[i] = parts[i]
        .replace(/\$\$([\s\S]+?)\$\$/g, (_, tex) => `@@M${maths.push({ tex, display: true }) - 1}@@`)
        .replace(/(^|[^\\$\w])\$(?!\s)([^$\n]+?)(?<!\s)\$(?!\d)/g, (_, pre, tex) => `${pre}@@M${maths.push({ tex, display: false }) - 1}@@`);
    }
    let html = marked.parse(parts.join(""));
    html = html.replace(/@@M(\d+)@@/g, (_, i) => {
      const m = maths[+i];
      if (!window.katex) return esc(m.tex);
      try { return katex.renderToString(m.tex, { displayMode: m.display, throwOnError: false }); } catch (e) { return esc(m.tex); }
    });
    return html;
  }
  const CALLOUTS = { "interview trap": "trap", "say it like this": "say", "worked example": "example", "watch": "watch", "remember": "remember" };
  function enhance(root) {
    $$("blockquote", root).forEach((b) => {
      const s = b.querySelector("strong");
      if (!s) return;
      const label = s.textContent.replace(/:$/, "").trim().toLowerCase();
      if (CALLOUTS[label]) b.classList.add(CALLOUTS[label]);
    });
    const used = new Set();
    $$("h2, h3", root).forEach((h) => {
      let id = slug(h.textContent) || "section";
      while (used.has(id)) id += "-x";
      used.add(id); h.id = id;
    });
    $$("a[href^='http']", root).forEach((a) => { a.target = "_blank"; a.rel = "noopener"; });
    if (window.hljs) $$("pre code", root).forEach((c) => { try { hljs.highlightElement(c); } catch (e) { /* ignore */ } });
  }

  // ------------------------------------------------------------ spaced repetition
  const cardState = (id) => state.cards[id];
  const isDue = (id, d = today()) => { const s = cardState(id); return !!s && s.due <= d; };
  const isNew = (id) => !cardState(id);
  const isLearned = (id) => { const s = cardState(id); return !!s && s.interval >= 7; };
  function schedule(id, rating) {
    const s = Object.assign({ reps: 0, ease: 2.5, interval: 0, lapses: 0 }, cardState(id));
    if (rating === 1) { s.reps = 0; s.interval = 0; s.lapses += 1; s.ease = Math.max(1.3, s.ease - 0.2); }
    else if (rating === 2) { s.interval = Math.max(1, Math.round(s.interval * 1.2)); s.ease = Math.max(1.3, s.ease - 0.15); s.reps += 1; }
    else if (rating === 3) { s.interval = s.reps === 0 ? 1 : s.reps === 1 ? 3 : Math.round(s.interval * s.ease); s.reps += 1; }
    else { s.interval = s.reps === 0 ? 3 : Math.round(Math.max(1, s.interval) * s.ease * 1.3); s.ease += 0.15; s.reps += 1; }
    s.due = addDays(today(), s.interval);
    s.last = today();
    state.cards[id] = s;
    state.reviews[today()] = (state.reviews[today()] || 0) + 1;
    save();
  }
  function streak() {
    const active = new Set(Object.keys(state.reviews).concat(state.drills.map((d) => d.date)));
    let n = 0, d = today();
    if (!active.has(d)) d = addDays(d, -1);
    while (active.has(d)) { n++; d = addDays(d, -1); }
    return n;
  }

  // ------------------------------------------------------------ router
  const routes = {};
  async function route() {
    cleanups.forEach((fn) => { try { fn(); } catch (e) { /* ignore */ } });
    cleanups = [];
    if (window.speechSynthesis) speechSynthesis.cancel();
    const hash = location.hash.slice(1) || "/cards";
    const [path, query] = hash.split("?");
    const parts = path.split("/").filter(Boolean).map(decodeURIComponent);
    const params = new URLSearchParams(query || "");
    const name = parts[0] || "today";
    const navName = { module: "study", guide: "study", day: "archive", progress: "" }[name] ?? name;
    $$(".nav a").forEach((a) => a.classList.toggle("on", a.dataset.route === navName));
    main.innerHTML = '<p class="loading">Loading…</p>';
    try {
      await (routes[name] || routes.notfound)(parts.slice(1), params);
    } catch (e) {
      console.error(e);
      main.innerHTML = `<div class="notice">This page could not load: ${esc(e.message)}</div>`;
    }
    if (!params.get("s")) window.scrollTo(0, 0);
    main.focus({ preventScroll: true });
  }
  window.addEventListener("hashchange", route);

  routes.notfound = async () => { main.innerHTML = `<h1>Not found</h1><p><a href="#/today">Go to Today</a></p>`; };

  // ------------------------------------------------------------ today and day views
  const SPONSOR = {
    sponsors: ["good", "Sponsorship stated"], no_sponsorship: ["warn", "No sponsorship"],
    citizens_only: ["bad", "Citizens or clearance only"], not_stated: ["", "Not stated"],
  };
  const sponsorBadge = (s) => { const [c, t] = SPONSOR[s] || SPONSOR.not_stated; return `<span class="badge ${c}">${t}</span>`; };

  function jobRows(jobs) {
    return jobs.map((j) => `<tr>
      <td><b>${esc(j.company)}</b></td>
      <td>${esc(j.title)}${j.early_career ? ' <span class="badge accent">early career</span>' : ""}${j.comp ? `<div class="muted small">${esc(j.comp)}</div>` : ""}</td>
      <td class="small">${esc(j.location || (j.remote ? "Remote" : ""))}</td>
      <td class="nowrap small">${esc(j.posted || "n/a")}</td>
      <td>${sponsorBadge(j.sponsorship)}</td>
      <td class="nowrap">${link(j.apply_url || j.url, "Apply", "btn small primary")}</td></tr>`).join("");
  }
  const jobTable = (jobs) => `<div class="table-wrap"><table><thead><tr><th>Company</th><th>Role</th><th>Location</th><th>Posted</th><th>Sponsorship</th><th></th></tr></thead><tbody>${jobRows(jobs)}</tbody></table></div>`;

  const feedList = (items, meta) => items.length ? `<ul class="feed">${items.map((x) => `<li><div class="t">${link(x.url, x.title)}</div><div class="m">${meta(x)}</div>${x.summary ? `<div class="s">${esc(x.summary)}</div>` : ""}</li>`).join("")}</ul>` : `<p class="empty">Nothing new today.</p>`;

  async function renderDay(date, isToday) {
    const [day, notes, cards, titles] = await Promise.all([getJSON(`data/daily/${date}.json`, true), getText(`data/daily/${date}.notes.md`), allCards(), moduleTitles()]);
    const picks = drillPick(cards, date);
    const due = cards.filter((c) => isDue(c.id)).length;
    const c = day.counts || {};
    const jobs = day.jobs_new || [];
    main.innerHTML = `
      <h1>${isToday ? "Today" : esc(date)}</h1>
      <p class="lede">${isToday ? `Daily update for ${esc(date)}, built ${esc(fmtTime(day.generated_at))}.` : `Archived update, built ${esc(fmtTime(day.generated_at))}.`}
        New roles, today's spoken drill, and what changed in the field.</p>
      <div class="stats">
        <div class="stat"><b>${c.jobs_new ?? 0}</b><span>new roles</span></div>
        <div class="stat"><b>${c.jobs_open ?? 0}</b><span>open roles tracked</span></div>
        <div class="stat"><b>${due}</b><span>cards due</span></div>
        <div class="stat"><b>${streak()}</b><span>day streak</span></div>
      </div>
      <div class="row" style="margin-bottom:8px"><a class="btn primary" href="#/cards/learn">Continue learning</a><a class="btn" href="#/drill/today">Today's spoken drill</a></div>
      ${notes ? `<section class="panel prose" id="notes"><h2 style="margin-top:0">What matters today</h2>${md(notes)}</section>` : ""}
      <h2>Today's spoken drill</h2>
      ${picks.length ? `<ol>${picks.map((p) => `<li>${esc(p.q)} <span class="muted small">(${esc(titles[p.module] || p.module)}, ${esc(p.level)})</span></li>`).join("")}</ol>
        <div class="row"><a class="btn primary" href="#/drill/today">Start the drill</a><a class="btn" href="#/cards/learn">Review ${plural(due, "due card")}</a><a class="btn" href="#/cards/today">Flip today's 5 as cards</a></div>`
        : `<p class="empty">Cards appear here once the first modules are published.</p>`}
      <h2>New roles</h2>
      ${jobs.length ? jobTable(jobs.slice(0, 30)) + `<p class="small muted">${jobs.length > 30 ? `${jobs.length - 30} more new today. ` : ""}<a href="#/jobs">All ${c.jobs_open ?? ""} open roles with filters</a></p>` : `<p class="empty">No new matching roles since the last run. <a href="#/jobs">Browse all open roles</a>.</p>`}
      <div class="two">
        <section><h2>Papers people are reading</h2>${feedList(day.papers || [], (p) => `${p.upvotes} upvotes on Hugging Face, ${link(p.arxiv, "arXiv")}`)}</section>
        <section><h2>New from the reading list</h2>${feedList((day.posts || []).slice(0, 15), (p) => `${esc(p.source)}, ${esc((p.published || "").slice(0, 10))}`)}</section>
      </div>
      <div class="two">
        <section><h2>Hacker News</h2>${feedList(day.hn || [], (h) => `${h.points} points, ${link(h.discussion, plural(h.comments, "comment"))}`)}</section>
        <section><h2>Lectures and videos</h2>${feedList(day.videos || [], (v) => `${esc(v.source)}${v.new_in_playlist ? ", new in a playlist you follow" : ""}`)}
          <h2>Interview intel</h2>${feedList(day.interview_intel || [], (x) => esc(x.source))}</section>
      </div>
      ${(day.errors || []).length ? `<details class="errors"><summary>${plural(day.errors.length, "source")} failed this run and were skipped</summary><ul>${day.errors.map((e) => `<li><code>${esc(e.source)}</code>: ${esc(e.error)}</li>`).join("")}</ul></details>` : ""}`;
    enhance(main);
  }

  routes.today = async () => {
    let idx = null;
    try { idx = await getJSON("data/daily/index.json", true); } catch (e) { /* first run pending */ }
    if (!idx || !idx.latest) {
      main.innerHTML = `<h1>Today</h1><div class="notice">The first daily run has not finished yet. The study modules and cards already work: <a href="#/study">start studying</a>.</div>`;
      return;
    }
    await renderDay(idx.latest, true);
  };
  routes.day = async ([date]) => renderDay(date, false);
  routes.archive = async () => {
    const idx = await getJSON("data/daily/index.json", true).catch(() => ({ dates: [] }));
    main.innerHTML = `<h1>Archive</h1><p class="lede">Every daily update, newest first.</p>
      ${idx.dates.length ? `<ul class="feed">${idx.dates.map((d) => `<li><a href="#/day/${esc(d)}">${esc(d)}</a></li>`).join("")}</ul>` : `<p class="empty">No updates yet.</p>`}`;
  };

  // ------------------------------------------------------------ study
  routes.study = async () => {
    const mods = await modules();
    const counts = await Promise.all(mods.map((m) => (m.ready ? moduleCards(m.id) : [])));
    const groups = [...new Set(mods.map((m) => m.group))];
    const tile = (m, cards) => {
      const ready = cards.length > 0;
      const learned = cards.filter((c) => isLearned(c.id)).length;
      const pct = ready ? Math.round((100 * learned) / cards.length) : 0;
      return `<div class="panel tile">
        <h3>${esc(m.title)}</h3><p>${esc(m.summary)}</p>
        ${ready ? `<div class="meter" title="${learned} of ${cards.length} cards learned"><span style="width:${pct}%"></span></div>
          <div class="small muted">${cards.length} cards, ${learned} learned</div>
          <div class="row"><a class="btn small" href="#/module/${m.id}?d=cheat">Cheatsheet</a><a class="btn small" href="#/module/${m.id}?d=plain">Plain</a><a class="btn small" href="#/module/${m.id}?d=deep">Deep</a><a class="btn small primary" href="#/cards/m/${m.id}">Cards</a></div>`
        : `<div class="small muted">Being written. Check back after the next update.</div>`}
      </div>`;
    };
    main.innerHTML = `<h1>Study</h1>
      <p class="lede">Each module comes at three depths. The cheatsheet is the ten minute refresher, Plain explains it to a smart friend, and Deep goes as far as a research lab interviewer will push. The headings match, so you can switch depth mid read.</p>
      <div class="row" style="margin-bottom:8px"><a class="btn" href="#/guide/question-bank/study-plan">Study plan</a><a class="btn" href="#/guide/question-bank/interview-loops">How interview loops work</a><a class="btn" href="#/guide/question-bank/resources">Best resources</a><a class="btn" href="#/progress">Your progress</a></div>
      ${groups.map((g) => `<h2>${esc(g)}</h2><div class="grid">${mods.map((m, i) => (m.group === g ? tile(m, counts[i]) : "")).join("")}</div>`).join("")}`;
  };

  const DEPTH = { cheat: ["cheatsheet.md", "Cheatsheet"], plain: ["plain.md", "Plain"], deep: ["deep.md", "Deep"] };
  async function renderDoc(path, { title, head = "", cls = "" }) {
    const text = await getText(path);
    if (!text) { main.innerHTML = `<h1>${esc(title)}</h1><div class="notice">This page is still being written.</div>`; return null; }
    main.innerHTML = `<div class="reader ${cls}"><nav class="toc" aria-label="On this page"></nav><div><div class="reader-head noprint">${head}</div><article class="prose">${md(text)}</article></div></div>`;
    const art = $("article", main);
    enhance(art);
    const heads = $$("h2, h3", art);
    const toc = $(".toc", main);
    toc.innerHTML = heads.map((h) => `<a class="${h.tagName.toLowerCase()}" href="#" data-id="${h.id}">${esc(h.textContent)}</a>`).join("");
    toc.addEventListener("click", (e) => {
      const a = e.target.closest("a[data-id]");
      if (!a) return;
      e.preventDefault();
      document.getElementById(a.dataset.id)?.scrollIntoView({ behavior: "smooth", block: "start" });
    });
    if ("IntersectionObserver" in window && heads.length) {
      const io = new IntersectionObserver((entries) => {
        entries.forEach((en) => {
          if (en.isIntersecting) $$(".toc a", main).forEach((a) => a.classList.toggle("on", a.dataset.id === en.target.id));
        });
      }, { rootMargin: "0px 0px -75% 0px" });
      heads.forEach((h) => io.observe(h));
      onCleanup(() => io.disconnect());
    }
    return art;
  }
  function currentSection(art) {
    let cur = null;
    $$("h2", art).forEach((h) => { if (h.getBoundingClientRect().top < 120) cur = h.id; });
    return cur;
  }

  routes.module = async ([id], params) => {
    const mods = await modules();
    const m = mods.find((x) => x.id === id);
    if (!m) return routes.notfound();
    const depth = DEPTH[params.get("d")] ? params.get("d") : "plain";
    const head = `<h1 style="margin:0">${esc(m.title)}</h1><span class="spacer"></span>
      <div class="seg" role="group" aria-label="Depth">${Object.entries(DEPTH).map(([k, [, label]]) => `<button type="button" data-d="${k}" aria-pressed="${k === depth}">${label}</button>`).join("")}</div>
      <a class="btn primary" href="#/cards/m/${id}">Practise the cards</a>${depth === "cheat" ? `<button class="btn" type="button" id="print">Print</button>` : ""}`;
    const art = await renderDoc(`content/${id}/${DEPTH[depth][0]}`, { title: m.title, head, cls: depth === "cheat" ? "cheat" : "" });
    if (!art) return;
    $$(".seg button", main).forEach((b) => b.addEventListener("click", () => {
      const s = currentSection(art);
      location.hash = `#/module/${id}?d=${b.dataset.d}${s ? `&s=${s}` : ""}`;
    }));
    $("#print", main)?.addEventListener("click", () => window.print());
    const s = params.get("s");
    if (s) requestAnimationFrame(() => document.getElementById(s)?.scrollIntoView({ block: "start" }));
  };

  routes.guide = async (parts) => {
    const [folder, name] = parts;
    if (!/^[a-z0-9-]+$/.test(folder || "") || !/^[a-z0-9-]+$/.test(name || "")) return routes.notfound();
    const titles = { "study-plan": "Study plan", "interview-loops": "How interview loops work", resources: "Best resources" };
    await renderDoc(`content/${folder}/${name}.md`, { title: titles[name] || name, head: `<a class="btn small" href="#/study">Back to Study</a>` });
  };

  // ------------------------------------------------------------ flashcards
  async function buildDeck(parts, params) {
    const cards = await allCards();
    const kind = parts[0] || "";
    const t = today();
    let deck, title;
    if (kind === "due" || kind === "learn") {
      const cur = await curriculum();
      const due = cur.filter((c) => isDue(c.id, t)).sort((a, b) => (cardState(a.id).due < cardState(b.id).due ? -1 : 1));
      deck = due.concat(cur.filter((c) => isNew(c.id)).slice(0, state.newPerDay || 20));
      title = "Continue learning";
    } else if (kind === "today") {
      const idx = await getJSON("data/daily/index.json", true).catch(() => null);
      deck = drillPick(cards, (idx && idx.latest) || t); title = "Today's five";
    } else if (kind === "m") {
      deck = cards.filter((c) => c.module === parts[1]);
      const due = deck.filter((c) => isDue(c.id, t)), fresh = deck.filter((c) => isNew(c.id)), rest = deck.filter((c) => !isDue(c.id, t) && !isNew(c.id));
      deck = params.get("o") === "all" ? deck : due.concat(fresh, rest);
      title = (await moduleTitles())[parts[1]] || parts[1];
    } else if (kind === "weak") {
      deck = cards.filter((c) => cardState(c.id)).sort((a, b) => cardState(a.id).ease - cardState(b.id).ease).slice(0, 20);
      title = "Weakest 20";
    } else {
      const mods = (params.get("m") || "").split(",").filter(Boolean);
      const lv = (params.get("l") || LEVELS.join(",")).split(",");
      const kd = (params.get("k") || "").split(",").filter(Boolean);
      deck = cards.filter((c) => (!mods.length || mods.includes(c.module)) && lv.includes(c.level) && (!kd.length || kd.includes(c.kind || "concept")));
      if (params.get("o") === "shuffle") deck = shuffle(deck);
      title = "Custom deck";
    }
    return { deck, title };
  }

  routes.cards = async (parts, params) => {
    if (!parts.length) return deckPicker();
    const { deck, title } = await buildDeck(parts, params);
    if (!deck.length) {
      main.innerHTML = `<h1>${esc(title)}</h1><p class="empty">No cards in this deck right now. <a href="#/cards">Pick another deck</a>.</p>`;
      return;
    }
    runSession(deck, title);
  };

  async function deckPicker() {
    const [cards, mods] = await Promise.all([allCards(), modules()]);
    const cur = await curriculum();
    const due = cur.filter((c) => isDue(c.id)).length;
    const fresh = cur.filter((c) => isNew(c.id)).length;
    const perDay = state.newPerDay || 20;
    const seen = cards.filter((c) => cardState(c.id)).length;
    const learned = cards.filter((c) => isLearned(c.id)).length;
    const ready = mods.filter((m) => m.ready).concat(cards.some((c) => c.module === "lectures") ? [{ id: "lectures", title: "Lecture notes", ready: true }] : []);
    const inPath = (id) => !state.path || !state.path.length || state.path.includes(id);
    const by = (id) => cards.filter((c) => c.module === id);
    if (!cards.length) {
      main.innerHTML = `<h1>Learn</h1><div class="notice">The first flashcard modules are being written right now. They appear here as soon as each one passes its checks.</div>
        <p>Meanwhile: <a href="#/today">today's jobs and papers</a>, <a href="#/lectures">the lectures</a>.</p>`;
      return;
    }
    main.innerHTML = `<h1>Learn</h1>
      <p class="lede">Learn everything from the cards. Each module's cards run in order, from the first idea to the interview questions that combine them. Every card teaches: a plain answer, a deep answer, a version to say out loud, the points to mention, follow ups and traps. Rate each card honestly and the schedule brings it back just before you would forget it.</p>
      <div class="stats">
        <div class="stat"><b>${due}</b><span>reviews due</span></div>
        <div class="stat"><b>${Math.min(perDay, fresh)}</b><span>new cards today</span></div>
        <div class="stat"><b>${learned} / ${cards.length}</b><span>learned</span></div>
        <div class="stat"><b>${streak()}</b><span>day streak</span></div>
      </div>
      <div class="row"><a class="btn primary" href="#/cards/learn">Continue learning (${due + Math.min(perDay, fresh)} cards)</a>
        <label class="check">New cards per day <select id="perday">${[10, 20, 30, 50, 100].map((n) => `<option ${n === perDay ? "selected" : ""}>${n}</option>`).join("")}</select></label>
        <a class="btn" href="#/cards/weak">Weakest 20</a><a class="btn" href="#/drill/today">Today's spoken drill</a></div>
      <h2>Modules</h2>
      <p class="small muted">Ticked modules are in your learning path, in this order. Or learn one module on its own.</p>
      <div class="grid">${ready.map((m) => {
        const cs = by(m.id), l = cs.filter((c) => isLearned(c.id)).length, s = cs.filter((c) => cardState(c.id)).length;
        return `<div class="panel tile"><div class="row"><h3 style="flex:1">${esc(m.title)}</h3><label class="check" title="Include in my learning path"><input type="checkbox" data-path="${m.id}" ${inPath(m.id) ? "checked" : ""}> path</label></div>
          <div class="meter" title="${l} of ${cs.length} learned"><span style="width:${cs.length ? Math.round((100 * l) / cs.length) : 0}%"></span></div>
          <div class="small muted">${cs.length} cards, ${s} seen, ${l} learned</div>
          <div class="row"><a class="btn small primary" href="#/cards/m/${m.id}">Learn in order</a>${m.id !== "lectures" ? `<a class="btn small" href="#/module/${m.id}?d=plain">Read</a>` : ""}<a class="btn small" href="#/drill/m/${m.id}">Drill</a></div></div>`;
      }).join("")}</div>
      <h2>Custom deck</h2>
      <form id="deck" class="panel">
        <div class="row">${KINDS.map((k) => `<label class="check"><input type="checkbox" name="k" value="${k}"> ${k}</label>`).join("")}</div>
        <div class="row" style="margin-top:10px">${LEVELS.map((l) => `<label class="check"><input type="checkbox" name="l" value="${l}" checked> ${l}</label>`).join("")}
          <label class="check"><input type="checkbox" name="o" value="shuffle" checked> shuffle</label>
          <span class="spacer"></span><button class="btn primary" type="submit">Start</button></div>
        <p class="hint">Uses the modules ticked above. Leave every kind unticked to include all kinds. ${seen} of ${cards.length} cards seen so far.</p>
      </form>`;
    $("#perday").addEventListener("change", (e) => { state.newPerDay = +e.target.value; save(); route(); });
    $$("[data-path]", main).forEach((cb) => cb.addEventListener("change", () => {
      state.path = $$("[data-path]", main).filter((x) => x.checked).map((x) => x.dataset.path);
      if (state.path.length === ready.length) state.path = null;
      save(); route();
    }));
    $("#deck").addEventListener("submit", (e) => {
      e.preventDefault();
      const f = new FormData(e.target);
      const m = state.path && state.path.length ? state.path : [];
      const q = new URLSearchParams({ m: m.join(","), l: f.getAll("l").join(","), k: f.getAll("k").join(","), o: f.get("o") || "" });
      location.hash = `#/cards/custom?${q}`;
    });
  }

  function speak(text) {
    if (!window.speechSynthesis) return;
    speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    u.rate = 1; u.lang = "en-US";
    speechSynthesis.speak(u);
  }

  async function runSession(deck, title) {
    const titles = await moduleTitles();
    const queue = deck.map((c) => c.id);
    const byId = Object.fromEntries(deck.map((c) => [c.id, c]));
    const again = new Set();
    let done = 0, shown = 0;
    const total = queue.length;

    const show = () => {
      if (!queue.length) {
        main.innerHTML = `<div class="study"><h1>Done</h1><p class="lede">You went through ${plural(total, "card")} in "${esc(title)}". Anything you rated Again comes back first next time.</p>
          <div class="row"><a class="btn primary" href="#/drill">Say a few out loud in the drill</a><a class="btn" href="#/cards">Another deck</a><a class="btn" href="#/today">Back to Today</a></div></div>`;
        return;
      }
      const c = byId[queue[0]];
      const st = cardState(c.id);
      shown = 0;
      main.innerHTML = `<div class="study">
        <div class="progress-line"><span>${esc(title)}</span><div class="meter"><span style="width:${Math.round((100 * done) / total)}%"></span></div><span>${done} / ${total}</span></div>
        <div class="card">
          <div class="meta"><span class="badge">${esc(titles[c.module] || c.module)}</span><span class="badge">${esc(c.topic)}</span><span class="badge ${c.level === "expert" ? "warn" : c.level === "advanced" ? "accent" : ""}">${esc(c.level)}</span>${c.kind ? `<span class="badge">${esc(c.kind)}</span>` : ""}${!st ? '<span class="badge good">new</span>' : ""}</div>
          <p class="q">${esc(c.q)}</p>
          <div id="reveal"></div>
          <div class="actions" id="acts">
            <button class="btn primary" type="button" data-a="plain">Show answer <kbd>space</kbd></button>
            <button class="btn" type="button" data-a="deep">Go deep <kbd>d</kbd></button>
            <button class="btn" type="button" data-a="say">Say it <kbd>s</kbd></button>
            <button class="btn" type="button" data-a="fu">Follow ups <kbd>f</kbd></button>
            <a class="btn" href="#/drill/q/${esc(c.id)}">Drill this aloud</a>
          </div>
          <div class="rate" id="rate" hidden>
            <button type="button" class="again" data-r="1">Again<small>1</small></button>
            <button type="button" data-r="2">Hard<small>2</small></button>
            <button type="button" class="good" data-r="3">Good<small>3</small></button>
            <button type="button" data-r="4">Easy<small>4</small></button>
          </div>
          <p class="hint">Answer in your head or out loud first, then reveal. Rate after you see the answer.</p>
        </div></div>`;
      const reveal = $("#reveal", main);
      const sections = {};
      const add = (key, html) => {
        if (sections[key]) { sections[key].scrollIntoView({ block: "nearest", behavior: "smooth" }); return; }
        const div = document.createElement("div");
        div.className = "section"; div.innerHTML = html;
        reveal.appendChild(div); enhance(div); sections[key] = div;
        $("#rate", main).hidden = false; shown++;
      };
      const actions = {
        plain: () => add("plain", `<h4>Plain answer</h4><div class="prose">${md(c.plain)}</div>`),
        deep: () => add("deep", `<h4>Deep answer</h4><div class="prose">${md(c.deep)}</div>${(c.traps || []).length ? `<h4 style="margin-top:14px">Traps</h4><ul class="mm">${c.traps.map((t) => `<li>${esc(t)}</li>`).join("")}</ul>` : ""}${c.source ? `<p class="small muted">Seen asked: ${link(c.source.url, c.source.title)}</p>` : ""}`),
        say: () => {
          add("say", `<h4>Say it like this (about ${Math.round(c.say.split(/\s+/).length / 2.6)} seconds)</h4><div class="say">${esc(c.say)}</div>
            <div class="row" style="margin-top:8px"><button class="btn small" type="button" data-speak>Hear it</button></div>
            <h4 style="margin-top:14px">Must mention</h4><ul class="mm">${(c.must_mention || []).map((t) => `<li>${esc(t)}</li>`).join("")}</ul>`);
          $("[data-speak]", sections.say)?.addEventListener("click", () => speak(c.say));
        },
        fu: () => add("fu", `<h4>Follow ups</h4>${(c.followups || []).map((f) => `<details class="fu"><summary>${esc(f.q)}</summary><p>${esc(f.a)}</p></details>`).join("")}`),
      };
      $("#acts", main).addEventListener("click", (e) => { const b = e.target.closest("button[data-a]"); if (b) actions[b.dataset.a](); });
      const rate = (r) => {
        if (!shown) return;
        schedule(c.id, r);
        queue.shift();
        if (r === 1 && !again.has(c.id)) { again.add(c.id); queue.push(c.id); } else done++;
        show();
      };
      $("#rate", main).addEventListener("click", (e) => { const b = e.target.closest("button[data-r]"); if (b) rate(+b.dataset.r); });
      onKey = (e) => {
        if (e.target.closest("input, textarea, select") || e.metaKey || e.ctrlKey || e.altKey) return;
        if (e.key === " " || e.key === "Enter") { e.preventDefault(); actions.plain(); }
        else if (e.key === "d") actions.deep();
        else if (e.key === "s") actions.say();
        else if (e.key === "f") actions.fu();
        else if (["1", "2", "3", "4"].includes(e.key)) rate(+e.key);
      };
    };
    let onKey = null;
    const handler = (e) => onKey && onKey(e);
    document.addEventListener("keydown", handler);
    onCleanup(() => document.removeEventListener("keydown", handler));
    show();
  }

  // ------------------------------------------------------------ narration drill
  const STOP = new Set("the a an and or of to in on for with is are be by as at it its this that from into than then when what which your you".split(" "));
  const words = (s) => (s || "").toLowerCase().replace(/[^a-z0-9\s]/g, " ").split(/\s+/).filter(Boolean);
  function mentioned(phrase, transcript) {
    const said = new Set(words(transcript).map((w) => w.slice(0, 5)));
    const key = words(phrase).filter((w) => w.length >= 3 && !STOP.has(w)).map((w) => w.slice(0, 5));
    if (!key.length) return false;
    const hits = key.filter((w) => said.has(w)).length;
    return hits / key.length >= 0.6;
  }

  routes.drill = async (parts, params) => {
    if (!parts.length) return drillHome();
    const cards = await allCards();
    let items = [], title = "Drill";
    if (parts[0] === "today") {
      const idx = await getJSON("data/daily/index.json", true).catch(() => null);
      items = drillPick(cards, (idx && idx.latest) || today()); title = "Today's drill";
    } else if (parts[0] === "q") {
      items = cards.filter((c) => c.id === parts[1]); title = "Single question";
    } else if (parts[0] === "m") {
      items = shuffle(cards.filter((c) => c.module === parts[1])).slice(0, +(params.get("n") || 5)); title = (await moduleTitles())[parts[1]] || "Drill";
    } else if (parts[0] === "bank") {
      const qs = await getJSON("content/question-bank/questions.json").catch(() => []);
      const q = qs.find((x) => x.id === parts[1]);
      if (q) items = [{ id: q.id, q: q.q, module: q.module, level: q.level, topic: q.kind, must_mention: [], followups: [], say: "", bank: q }];
      title = "Question bank";
    }
    if (!items.length) { main.innerHTML = `<h1>Drill</h1><p class="empty">Nothing to drill here yet. <a href="#/drill">Back</a></p>`; return; }
    runDrill(items, title, +(params.get("t") || state.drillSeconds || 45));
  };

  async function drillHome() {
    const mods = await modules();
    const counts = await Promise.all(mods.map((m) => (m.ready ? moduleCards(m.id) : [])));
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    const sec = state.drillSeconds || 45;
    main.innerHTML = `<h1>Spoken drill</h1>
      <p class="lede">Interviews are spoken, so practise speaking. You get a question and a timer. Answer out loud, then compare against the points a strong answer must hit and the model spoken answer. Record yourself to hear your pace and filler words.</p>
      <div class="panel">
        <div class="row"><span>Answer length</span><div class="seg" role="group" aria-label="Answer length">
          ${[45, 120, 300].map((s) => `<button type="button" data-s="${s}" aria-pressed="${s === sec}">${s < 60 ? `${s} s` : `${s / 60} min`}</button>`).join("")}</div></div>
        <p class="small muted" style="margin:10px 0 0">45 seconds for concept questions, 2 minutes for a project summary, 5 minutes for a deep dive or a case.</p>
        <div class="row" style="margin-top:12px">
          <label class="check"><input type="checkbox" id="rec" ${state.drillRecord ? "checked" : ""}> Record my answer (stays in this tab, never uploaded)</label>
          ${SR ? `<label class="check"><input type="checkbox" id="stt" ${state.drillTranscript ? "checked" : ""}> Live transcript and auto ticking (your browser sends audio to its speech service)</label>` : `<span class="small muted">Live transcript needs Chrome, Edge or Safari.</span>`}
        </div>
      </div>
      <h2>Pick a drill</h2>
      <div class="row"><a class="btn primary" href="#/drill/today">Today's five</a><a class="btn" href="#/questions">From the question bank</a></div>
      <div class="grid" style="margin-top:14px">${mods.map((m, i) => (counts[i].length ? `<a class="panel tile" style="text-decoration:none;color:inherit" href="#/drill/m/${m.id}"><h3>${esc(m.title)}</h3><p>5 random questions from ${counts[i].length}</p></a>` : "")).join("")}</div>`;
    $$(".seg button", main).forEach((b) => b.addEventListener("click", () => {
      state.drillSeconds = +b.dataset.s; save();
      $$(".seg button", main).forEach((x) => x.setAttribute("aria-pressed", x === b));
    }));
    $("#rec", main)?.addEventListener("change", (e) => { state.drillRecord = e.target.checked; save(); });
    $("#stt", main)?.addEventListener("change", (e) => { state.drillTranscript = e.target.checked; save(); });
  }

  function runDrill(items, title, seconds) {
    let i = 0;
    let timer = null, recorder = null, stream = null, recog = null, audioUrl = null;
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    const stopAll = () => {
      clearInterval(timer); timer = null;
      try { recog && recog.stop(); } catch (e) { /* ignore */ }
      try { recorder && recorder.state !== "inactive" && recorder.stop(); } catch (e) { /* ignore */ }
      if (stream) stream.getTracks().forEach((t) => t.stop());
      stream = null;
    };
    onCleanup(() => { stopAll(); if (audioUrl) URL.revokeObjectURL(audioUrl); });

    const ask = (c, prompt, onDone) => {
      let transcript = "", left = seconds, started = 0;
      const box = $("#stage", main);
      box.innerHTML = `<p class="q" style="font:600 22px/1.35 var(--sans)">${esc(prompt)}</p>
        <div class="row"><button class="btn primary" type="button" id="go">Start speaking</button><span class="timer" id="clock">${seconds < 60 ? `0:${pad(seconds)}` : `${Math.floor(seconds / 60)}:${pad(seconds % 60)}`}</span><span id="recstate"></span></div>
        <div class="transcript" id="tx" hidden></div>`;
      const clock = $("#clock", box), tx = $("#tx", box);
      const finish = () => {
        stopAll();
        onDone({ transcript, elapsed: Math.round((Date.now() - started) / 1000) });
      };
      $("#go", box).addEventListener("click", async (e) => {
        if (timer) { finish(); return; }
        e.target.textContent = "Done";
        started = Date.now();
        if (state.drillRecord && navigator.mediaDevices && window.MediaRecorder) {
          try {
            stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            const chunks = [];
            recorder = new MediaRecorder(stream);
            recorder.ondataavailable = (ev) => chunks.push(ev.data);
            recorder.onstop = () => {
              if (audioUrl) URL.revokeObjectURL(audioUrl);
              audioUrl = URL.createObjectURL(new Blob(chunks, { type: recorder.mimeType }));
              const slot = $("#playback", main);
              if (slot) slot.innerHTML = `<h4>Your answer</h4><audio controls src="${audioUrl}"></audio>`;
            };
            recorder.start();
            $("#recstate", box).innerHTML = `<span class="rec-dot" aria-hidden="true"></span> recording`;
          } catch (err) { $("#recstate", box).textContent = "Microphone blocked, continuing without recording."; }
        }
        if (state.drillTranscript && SR) {
          try {
            recog = new SR(); recog.lang = "en-US"; recog.continuous = true; recog.interimResults = true;
            recog.onresult = (ev) => {
              let fin = "", interim = "";
              for (const r of ev.results) (r.isFinal ? (fin += r[0].transcript + " ") : (interim += r[0].transcript));
              transcript = fin + interim; tx.hidden = false; tx.textContent = transcript;
            };
            recog.start();
          } catch (err) { /* recognition unavailable */ }
        }
        timer = setInterval(() => {
          left--;
          const a = Math.abs(left);
          clock.textContent = `${left < 0 ? "+" : ""}${Math.floor(a / 60)}:${pad(a % 60)}`;
          clock.classList.toggle("over", left < 0);
        }, 1000);
      });
    };

    const review = (c, res) => {
      const box = $("#stage", main);
      const mm = c.must_mention || [];
      const hits = mm.map((p) => mentioned(p, res.transcript));
      box.innerHTML = `
        <p class="muted small">${esc(c.q)}</p>
        <p>You spoke for ${res.elapsed} seconds${res.elapsed > seconds * 1.15 ? ", which ran long. Cut the setup and lead with the answer." : "."}</p>
        <div id="playback"></div>
        ${res.transcript ? `<h4>What you said</h4><div class="transcript">${esc(res.transcript)}</div>` : ""}
        ${mm.length ? `<h4 style="margin-top:16px">Did you mention</h4><ul class="checklist">${mm.map((p, k) => `<li class="${hits[k] ? "hit" : ""}"><input type="checkbox" data-mm ${hits[k] ? "checked" : ""} aria-label="${esc(p)}"> <span>${esc(p)}</span></li>`).join("")}</ul>` : ""}
        <h4 style="margin-top:16px">Delivery check</h4><ul class="checklist">${RUBRIC.map((r) => `<li><input type="checkbox" data-rb aria-label="${esc(r)}"> <span>${esc(r)}</span></li>`).join("")}</ul>
        ${c.say ? `<h4 style="margin-top:16px">Model spoken answer</h4><div class="say" style="background:var(--accent-soft);border-radius:10px;padding:14px 16px;font:17px/1.65 var(--serif)">${esc(c.say)}</div><div class="row" style="margin-top:8px"><button class="btn small" type="button" id="hear">Hear it</button>${c.module && !c.bank ? `<a class="btn small" href="#/module/${esc(c.module)}?d=plain">Read the module</a>` : ""}</div>` : `<p class="small muted">No model answer for this one yet. ${c.bank && c.bank.source ? `Source: ${link(c.bank.source.url, c.bank.source.title)}` : ""}</p>`}
        ${(c.followups || []).length ? `<h4 style="margin-top:16px">The interviewer follows up</h4><div class="row"><button class="btn" type="button" id="fu">Answer the follow up aloud</button></div>` : ""}
        <h4 style="margin-top:16px">How clear was it?</h4>
        <div class="row" id="score">${[1, 2, 3, 4, 5].map((n) => `<button class="btn" type="button" data-n="${n}">${n}</button>`).join("")}<span class="muted small">1 rambling, 5 crisp</span></div>`;
      if (audioUrl) $("#playback", box).innerHTML = `<h4>Your answer</h4><audio controls src="${audioUrl}"></audio>`;
      $("#hear", box)?.addEventListener("click", () => speak(c.say));
      $("#fu", box)?.addEventListener("click", () => {
        const f = c.followups[0];
        ask(c, f.q, (r2) => {
          const b2 = $("#stage", main);
          b2.innerHTML = `<p class="muted small">Follow up: ${esc(f.q)}</p><p>You spoke for ${r2.elapsed} seconds.</p>${r2.transcript ? `<div class="transcript">${esc(r2.transcript)}</div>` : ""}
            <h4 style="margin-top:14px">A strong answer</h4><div class="say" style="background:var(--accent-soft);border-radius:10px;padding:14px 16px">${esc(f.a)}</div>
            <div class="row" style="margin-top:14px"><button class="btn primary" type="button" id="next">Next question</button></div>`;
          $("#next", b2).addEventListener("click", next);
        });
      });
      $("#score", box).addEventListener("click", (e) => {
        const b = e.target.closest("button[data-n]");
        if (!b) return;
        state.drills.push({ id: c.id, date: today(), seconds: res.elapsed, score: +b.dataset.n,
          hits: $$("[data-mm]", box).filter((x) => x.checked).length, total: mm.length });
        save(); next();
      });
    };

    const next = () => { i++; render(); };
    const render = () => {
      if (i >= items.length) {
        const mine = state.drills.slice(-items.length);
        const avg = mine.length ? (mine.reduce((a, d) => a + d.score, 0) / mine.length).toFixed(1) : "n/a";
        main.innerHTML = `<div class="study"><h1>Drill done</h1><p class="lede">${plural(items.length, "question")} answered out loud. Average clarity ${avg} out of 5.</p>
          <div class="row"><a class="btn primary" href="#/cards/learn">Continue learning</a><a class="btn" href="#/drill">Another drill</a><a class="btn" href="#/progress">Progress</a></div></div>`;
        return;
      }
      const c = items[i];
      main.innerHTML = `<div class="study"><div class="progress-line"><span>${esc(title)}</span><div class="meter"><span style="width:${Math.round((100 * i) / items.length)}%"></span></div><span>${i + 1} / ${items.length}</span></div>
        <div class="card" id="stage"></div></div>`;
      ask(c, c.q, (res) => review(c, res));
    };
    render();
  }

  // ------------------------------------------------------------ lectures
  routes.lectures = async () => {
    const pls = await getJSON("data/playlists.json");
    const notes = await Promise.all(pls.map((p) => getText(`content/lectures/${p.key}.md`)));
    const mins = (d) => d.split(":").map(Number).reduce((a, x) => a * 60 + x, 0) / 60;
    main.innerHTML = `<h1>Lectures</h1>
      <p class="lede">The four playlists this site is built around. The study modules link the lecture that teaches each section, and each playlist has its own notes page once it is written.</p>
      ${pls.map((p, k) => `<section><h2>${esc(p.title)}</h2>
        <div class="row small muted" style="margin-bottom:10px">${plural(p.videos.length, "video")}, about ${Math.round(p.videos.reduce((a, v) => a + mins(v.duration || "0"), 0) / 60)} hours. ${link(p.url, "Open on YouTube")}
          ${notes[k] ? `<a class="btn small primary" href="#/guide/lectures/${p.key}">Read the notes</a>` : `<span class="badge">notes being written</span>`}</div>
        <div class="table-wrap"><table><thead><tr><th>#</th><th>Lecture</th><th>Length</th><th></th></tr></thead><tbody>
        ${p.videos.map((v) => `<tr><td>${v.n}</td><td>${esc(v.title.replace(/^Stanford [A-Z]+\d+[A-Z]?[^|]*\|\s*[^|]*\|\s*/, ""))}</td><td class="nowrap">${esc(v.duration)}</td><td>${link(v.url, "Watch", "btn small")}</td></tr>`).join("")}
        </tbody></table></div></section>`).join("")}`;
  };

  // ------------------------------------------------------------ question bank
  routes.questions = async () => {
    let qs;
    try { qs = await getJSON("content/question-bank/questions.json"); } catch (e) {
      main.innerHTML = `<h1>Questions</h1><div class="notice">The question bank is still being compiled.</div>`; return;
    }
    const titles = await moduleTitles();
    const opts = (vals, label) => `<option value="">${label}</option>` + [...new Set(vals)].sort().map((v) => `<option value="${esc(v)}">${esc(titles[v] || v)}</option>`).join("");
    main.innerHTML = `<h1>Questions</h1>
      <p class="lede">${qs.length} real interview questions gathered from question banks, books, course material and interview write ups, each linked to where it was seen. Practise any of them aloud.</p>
      <div class="filters">
        <input type="search" id="q" placeholder="Search questions" aria-label="Search questions">
        <select id="fm" aria-label="Module">${opts(qs.map((x) => x.module), "All modules")}</select>
        <select id="fk" aria-label="Kind">${opts(qs.map((x) => x.kind), "All kinds")}</select>
        <select id="fr" aria-label="Role">${opts(qs.flatMap((x) => x.roles || []), "All roles")}</select>
        <select id="fl" aria-label="Level">${opts(qs.map((x) => x.level), "All levels")}</select>
      </div>
      <p class="small muted" id="count"></p><ul class="feed" id="list"></ul>`;
    const draw = () => {
      const t = $("#q").value.toLowerCase(), m = $("#fm").value, k = $("#fk").value, r = $("#fr").value, l = $("#fl").value;
      const hit = qs.filter((x) => (!t || x.q.toLowerCase().includes(t)) && (!m || x.module === m) && (!k || x.kind === k) && (!r || (x.roles || []).includes(r)) && (!l || x.level === l));
      $("#count").textContent = `${hit.length} of ${qs.length} questions`;
      $("#list").innerHTML = hit.slice(0, 300).map((x) => `<li><div class="t">${esc(x.q)}</div>
        <div class="m row"><span class="badge">${esc(titles[x.module] || x.module)}</span><span class="badge">${esc(x.kind)}</span><span class="badge">${esc(x.level)}</span>
        ${x.seen ? `<span>${esc(x.seen)}</span>` : ""}${x.source ? link(x.source.url, x.source.title) : ""}<a class="btn small" href="#/drill/bank/${esc(x.id)}">Practise aloud</a></div></li>`).join("");
    };
    $$(".filters input, .filters select", main).forEach((el) => el.addEventListener("input", draw));
    draw();
  };

  // ------------------------------------------------------------ jobs
  routes.jobs = async () => {
    let data;
    try { data = await getJSON("data/jobs/open.json", true); } catch (e) {
      main.innerHTML = `<h1>Jobs</h1><div class="notice">The first daily run has not finished yet.</div>`; return;
    }
    const GROUPS = { "": "All sources", "application-log": "Companies you applied to or tracked", "ai-startup": "AI startups", "new-grad-list": "New grad list (SimplifyJobs)", "hn-who-is-hiring": "HN Who is hiring", larger: "Larger companies" };
    main.innerHTML = `<h1>Jobs</h1>
      <p class="lede">${data.count} open roles matching ML, AI, data science and analytics titles in the US or remote, from ${Object.keys(GROUPS).length - 1} kinds of source. Senior, staff, lead and manager titles are filtered out. Updated ${esc(fmtTime(data.generated_at))}.</p>
      <div class="filters">
        <input type="search" id="q" placeholder="Search company, title or location" aria-label="Search jobs">
        <select id="g" aria-label="Source">${Object.entries(GROUPS).map(([k, v]) => `<option value="${k}">${esc(v)}</option>`).join("")}</select>
        <label class="check"><input type="checkbox" id="early"> Early career only</label>
        <label class="check"><input type="checkbox" id="hidecit" checked> Hide citizens or clearance only</label>
        <label class="check"><input type="checkbox" id="hideno"> Hide "no sponsorship"</label>
      </div>
      <p class="small muted">Sponsorship comes from keyword matching on the posting text. "Not stated" means the posting did not say either way, so check before you apply.</p>
      <p class="small muted" id="count"></p><div id="tbl"></div>`;
    const draw = () => {
      const t = $("#q").value.toLowerCase(), g = $("#g").value;
      const hit = data.jobs.filter((j) => (!t || `${j.company} ${j.title} ${j.location}`.toLowerCase().includes(t)) && (!g || j.group === g) &&
        (!$("#early").checked || j.early_career) && (!$("#hidecit").checked || j.sponsorship !== "citizens_only") && (!$("#hideno").checked || j.sponsorship !== "no_sponsorship"));
      $("#count").textContent = `Showing ${Math.min(hit.length, 400)} of ${hit.length} matching roles`;
      $("#tbl").innerHTML = hit.length ? jobTable(hit.slice(0, 400)) : `<p class="empty">No roles match these filters.</p>`;
    };
    $$(".filters input, .filters select", main).forEach((el) => el.addEventListener("input", draw));
    draw();
  };

  // ------------------------------------------------------------ progress
  routes.progress = async () => {
    const cards = await allCards();
    const seen = cards.filter((c) => cardState(c.id)).length;
    const learned = cards.filter((c) => isLearned(c.id)).length;
    const due = cards.filter((c) => isDue(c.id)).length;
    const last7 = [...Array(7)].map((_, k) => addDays(today(), -k)).reduce((a, d) => a + (state.reviews[d] || 0), 0);
    const avg = state.drills.length ? (state.drills.reduce((a, d) => a + d.score, 0) / state.drills.length).toFixed(1) : "n/a";
    main.innerHTML = `<h1>Progress</h1>
      <p class="lede">Your progress lives in this browser only. Export it to move to another device or to keep a backup.</p>
      <div class="stats">
        <div class="stat"><b>${seen} / ${cards.length}</b><span>cards seen</span></div>
        <div class="stat"><b>${learned}</b><span>learned (interval of a week or more)</span></div>
        <div class="stat"><b>${due}</b><span>due now</span></div>
        <div class="stat"><b>${last7}</b><span>reviews in the last 7 days</span></div>
        <div class="stat"><b>${state.drills.length}</b><span>spoken drills, average clarity ${avg}</span></div>
        <div class="stat"><b>${streak()}</b><span>day streak</span></div>
      </div>
      <div class="row"><button class="btn" type="button" id="exp">Export progress</button>
        <label class="btn" for="imp">Import progress</label><input type="file" id="imp" accept="application/json" hidden>
        <button class="btn" type="button" id="reset">Reset everything</button></div>`;
    $("#exp").addEventListener("click", () => {
      const a = document.createElement("a");
      a.href = URL.createObjectURL(new Blob([JSON.stringify(state, null, 1)], { type: "application/json" }));
      a.download = `ml-interview-prep-progress-${today()}.json`;
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 1000);
    });
    $("#imp").addEventListener("change", async (e) => {
      const f = e.target.files[0];
      if (!f) return;
      try { state = Object.assign(blank(), JSON.parse(await f.text())); save(); route(); } catch (err) { alert("That file is not a progress export."); }
    });
    $("#reset").addEventListener("click", () => {
      if (confirm("Delete all card progress and drill history in this browser?")) { state = Object.assign(blank(), { theme: state.theme }); save(); route(); }
    });
  };

  route();
})();
