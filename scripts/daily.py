"""Daily scrape: jobs, papers, posts, videos and interview intel.

Usage:
    python3 scripts/daily.py                    # today, UTC
    python3 scripts/daily.py --date 2026-10-13
    python3 scripts/daily.py --skip youtube,reddit --out /tmp/test-data

Reads config/*.json and data/playlists.json. Writes data/daily/<date>.json,
data/daily/latest.json, data/daily/index.json, data/jobs/open.json and the two
seen files. Every source is fetched independently; a failure is recorded under
"errors" in the day file and never stops the run. Stdlib only.
"""
from __future__ import annotations

import concurrent.futures as cf
import datetime as dt
import email.utils
import hashlib
import html
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UA = "ml-interview-prep-daily/1.0 (+https://github.com/TanayPratapSingh/ml-interview-prep)"
BROWSER_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"
ATOM = "{http://www.w3.org/2005/Atom}"
YT = "{http://www.youtube.com/xml/schemas/2015}"

ERRORS: list[dict] = []


# ---------------------------------------------------------------- helpers

def fetch(url: str, *, data: bytes | None = None, headers: dict | None = None, timeout: int = 30,
          retries: int = 2, ua: str = UA) -> bytes:
    hdrs = {"User-Agent": ua, "Accept-Language": "en-US,en;q=0.8"}
    hdrs.update(headers or {})
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, data=data, headers=hdrs)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (403, 404, 406):
                break
            time.sleep(4 * (attempt + 1) if e.code == 429 else 1.5 * (attempt + 1))
        except Exception as e:  # network errors, timeouts
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"{type(last).__name__}: {last}")


def fetch_json(url: str, **kw):
    return json.loads(fetch(url, **kw))


def record_error(source: str, err: Exception | str) -> None:
    ERRORS.append({"source": source, "error": str(err)[:300]})


TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")


def clean(s: str | None, limit: int = 280) -> str:
    """Strip tags, unescape entities, normalise dashes and shorten on a word boundary."""
    if not s:
        return ""
    s = html.unescape(TAG_RE.sub(" ", html.unescape(s)))
    s = s.replace("\u2014", ", ").replace("\u2013", " to ").replace("\xa0", " ")
    s = WS_RE.sub(" ", s).strip()
    if len(s) > limit:
        s = s[:limit].rsplit(" ", 1)[0].rstrip(",.;:") + " ..."
    return s


def parse_date(s: str | None) -> dt.datetime | None:
    if not s:
        return None
    s = s.strip()
    try:
        d = email.utils.parsedate_to_datetime(s)
        return d if d.tzinfo else d.replace(tzinfo=dt.timezone.utc)
    except (TypeError, ValueError):
        pass
    try:
        d = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
        return d if d.tzinfo else d.replace(tzinfo=dt.timezone.utc)
    except ValueError:
        return None


def iso(d: dt.datetime | None) -> str:
    return d.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if d else ""


def uid(*parts: str) -> str:
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]


def load(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def save(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- learning feeds

def parse_feed(raw: bytes, source: str, topic: str) -> list[dict]:
    root = ET.fromstring(raw)
    items = []
    for it in root.iter("item"):
        items.append({"title": clean(it.findtext("title"), 200), "url": (it.findtext("link") or "").strip(),
                      "published": iso(parse_date(it.findtext("pubDate") or it.findtext("{http://purl.org/dc/elements/1.1/}date"))),
                      "summary": clean(it.findtext("description")), "source": source, "topic": topic})
    for e in root.iter(f"{ATOM}entry"):
        link = ""
        for ln in e.findall(f"{ATOM}link"):
            if ln.get("rel") in (None, "alternate"):
                link = ln.get("href", "")
                break
        summary = e.findtext(f"{ATOM}summary") or e.findtext(f"{ATOM}content")
        items.append({"title": clean(e.findtext(f"{ATOM}title"), 200), "url": link,
                      "published": iso(parse_date(e.findtext(f"{ATOM}published") or e.findtext(f"{ATOM}updated"))),
                      "summary": clean(summary), "source": source, "topic": topic})
    return [i for i in items if i["title"] and i["url"]]


def blog_posts(cfg: dict) -> list[dict]:
    out: list[dict] = []

    def one(f):
        try:
            return parse_feed(fetch(f["url"], timeout=25), f["name"], f["topic"])
        except Exception as e:
            record_error(f"feed:{f['name']}", e)
            return []

    with cf.ThreadPoolExecutor(12) as ex:
        for items in ex.map(one, cfg["feeds"]):
            out += items
    return out


def hf_papers(cfg: dict, date: str) -> list[dict]:
    try:
        data = fetch_json(f"https://huggingface.co/api/daily_papers?date={date}")
        if not data:
            data = fetch_json("https://huggingface.co/api/daily_papers")
    except Exception as e:
        record_error("huggingface:daily_papers", e)
        return []
    papers = []
    for x in data:
        p = x.get("paper", {})
        pid = p.get("id", "")
        papers.append({"title": clean(p.get("title") or x.get("title"), 220), "url": f"https://huggingface.co/papers/{pid}",
                       "arxiv": f"https://arxiv.org/abs/{pid}", "upvotes": p.get("upvotes", 0), "github": p.get("githubRepo") or "",
                       "summary": clean(p.get("summary"), 420), "published": iso(parse_date(p.get("publishedAt"))),
                       "source": "Hugging Face Daily Papers"})
    papers.sort(key=lambda p: -p["upvotes"])
    return papers[: cfg["hf_papers"]["top_n"]]


def kw_regex(words: list[str]) -> re.Pattern:
    # Short all caps words (AI, ML, RAG) match case sensitively as whole words.
    parts = []
    for w in words:
        if w.isupper() and len(w) <= 4:
            parts.append(rf"(?-i:\b{re.escape(w)}\b)")
        else:
            parts.append(rf"\b{re.escape(w)}")
    return re.compile("|".join(parts), re.I)


def hn_stories(cfg: dict, since: int) -> list[dict]:
    hn = cfg["hn"]
    rx = kw_regex(hn["keywords"])
    url = ("https://hn.algolia.com/api/v1/search?tags=story&hitsPerPage=200&numericFilters="
           + urllib.parse.quote(f"created_at_i>{since},points>{hn['min_points']}"))
    try:
        hits = fetch_json(url)["hits"]
    except Exception as e:
        record_error("hackernews", e)
        return []
    out = []
    for h in hits:
        title = h.get("title") or ""
        if not rx.search(title):
            continue
        out.append({"title": clean(title, 200), "url": h.get("url") or f"https://news.ycombinator.com/item?id={h['objectID']}",
                    "discussion": f"https://news.ycombinator.com/item?id={h['objectID']}", "points": h.get("points", 0),
                    "comments": h.get("num_comments", 0), "published": iso(parse_date(h.get("created_at"))), "source": "Hacker News"})
    out.sort(key=lambda x: -x["points"])
    return out[:20]


def reddit_posts(cfg: dict) -> list[dict]:
    out = []
    for sub in cfg["reddit"]:
        try:
            raw = fetch(f"https://www.reddit.com/r/{sub}/top/.rss?t=day&limit=8", ua=UA, retries=1, timeout=20)
            for it in parse_feed(raw, f"r/{sub}", "community")[:6]:
                out.append(it)
        except Exception as e:
            record_error(f"reddit:{sub}", e)
        time.sleep(3)
    return out


# ---------------------------------------------------------------- youtube

def yt_channel(ch: dict) -> list[dict]:
    raw = fetch(f"https://www.youtube.com/feeds/videos.xml?channel_id={ch['channel_id']}", ua=BROWSER_UA, timeout=20)
    root = ET.fromstring(raw)
    rx = kw_regex(ch["keywords"]) if ch.get("keywords") else None
    vids = []
    for e in root.iter(f"{ATOM}entry"):
        title = e.findtext(f"{ATOM}title") or ""
        if rx and not rx.search(title):
            continue
        vid = e.findtext(f"{YT}videoId") or ""
        vids.append({"title": clean(title, 200), "url": f"https://www.youtube.com/watch?v={vid}", "id": vid,
                     "published": iso(parse_date(e.findtext(f"{ATOM}published"))), "source": ch["name"]})
    return vids


def _walk(o, key):
    if isinstance(o, dict):
        for k, v in o.items():
            if k == key:
                yield v
            yield from _walk(v, key)
    elif isinstance(o, list):
        for v in o:
            yield from _walk(v, key)


def yt_playlist_ids(pid: str) -> list[tuple[str, str]]:
    """Return (video id, title) for every video in a playlist by reading the page data."""
    page = fetch(f"https://www.youtube.com/playlist?list={pid}", ua=BROWSER_UA, timeout=25).decode("utf-8", "replace")
    m = re.search(r"var ytInitialData = (\{.*?\});</script>", page, re.S)
    if not m:
        raise RuntimeError("no playlist data on page (consent or bot check)")
    data = json.loads(m.group(1))
    key = re.search(r'"INNERTUBE_API_KEY":"([^"]+)"', page)
    ver = re.search(r'"INNERTUBE_CLIENT_VERSION":"([^"]+)"', page)

    def lockups(d):
        res = []
        for lk in _walk(d, "lockupViewModel"):
            title = lk.get("metadata", {}).get("lockupMetadataViewModel", {}).get("title", {}).get("content", "")
            if lk.get("contentId"):
                res.append((lk["contentId"], title))
        for v in _walk(d, "playlistVideoRenderer"):
            res.append((v.get("videoId"), "".join(r.get("text", "") for r in v.get("title", {}).get("runs", []))))
        return res

    def token(d):
        for c in _walk(d, "continuationCommand"):
            if c.get("request") == "CONTINUATION_REQUEST_TYPE_BROWSE":
                return c.get("token")
        return None

    vids, tok, n = lockups(data), token(data), 0
    while tok and key and ver and n < 10:
        n += 1
        body = json.dumps({"context": {"client": {"clientName": "WEB", "clientVersion": ver.group(1), "hl": "en"}},
                           "continuation": tok}).encode()
        more = json.loads(fetch(f"https://www.youtube.com/youtubei/v1/browse?key={key.group(1)}", data=body, ua=BROWSER_UA,
                                headers={"Content-Type": "application/json"}))
        new = lockups(more)
        if not new:
            break
        vids += new
        tok = token(more)
    seen, out = set(), []
    for v, t in vids:
        if v and v not in seen:
            seen.add(v)
            out.append((v, t))
    return out


def youtube(cfg: dict) -> list[dict]:
    known = {v["id"] for p in load(ROOT / "data/playlists.json", []) for v in p["videos"]}
    out = []
    for pl in cfg["youtube_playlists"]:
        try:
            for vid, title in yt_playlist_ids(pl["playlist_id"]):
                if vid not in known:
                    out.append({"title": clean(title, 200), "url": f"https://www.youtube.com/watch?v={vid}", "id": vid,
                                "published": "", "source": pl["name"], "new_in_playlist": pl["key"]})
        except Exception as e:
            record_error(f"youtube:playlist:{pl['key']}", e)
    for ch in cfg["youtube_channels"]:
        try:
            out += yt_channel(ch)
        except Exception as e:
            record_error(f"youtube:channel:{ch['name']}", e)
    return out


# ---------------------------------------------------------------- jobs

class JobFilter:
    def __init__(self, cfg: dict):
        c = lambda xs: re.compile("|".join(xs), re.I)
        self.inc, self.exc = c(cfg["title_include"]), c(cfg["title_exclude"])
        self.early = c(cfg["early_career"])
        self.loc_inc = [re.compile(x, re.I) for x in cfg["location_include"]]
        self.loc_exc = c(cfg["location_exclude"])
        s = cfg["sponsorship_signals"]
        self.sp = {k: c(v) for k, v in s.items()}

    def title_ok(self, title: str) -> bool:
        return bool(self.inc.search(title)) and not self.exc.search(title)

    def us_ok(self, loc: str, remote: bool) -> bool:
        loc = loc or ""
        us = any(rx.search(loc) for rx in self.loc_inc if rx.pattern != "remote")
        if us:
            return True
        if self.loc_exc.search(loc):
            return False
        return remote or "remote" in loc.lower() or not loc.strip()

    def sponsorship(self, text: str) -> str:
        t = text or ""
        if self.sp["citizens_only"].search(t):
            return "citizens_only"
        if self.sp["no_sponsorship"].search(t):
            return "no_sponsorship"
        if self.sp["sponsors"].search(t):
            return "sponsors"
        return "not_stated"


def ashby_jobs(b: dict) -> list[dict]:
    d = fetch_json(f"https://api.ashbyhq.com/posting-api/job-board/{b['slug']}?includeCompensation=true", timeout=30)
    out = []
    for j in d.get("jobs", []):
        if j.get("isListed") is False:
            continue
        locs = [j.get("location") or ""] + [s.get("location", "") for s in j.get("secondaryLocations") or []]
        comp = (j.get("compensation") or {}).get("compensationTierSummary") or ""
        out.append({"jid": j.get("id"), "title": j.get("title", ""), "location": "; ".join(x for x in locs if x),
                    "remote": bool(j.get("isRemote")), "url": j.get("jobUrl", ""), "apply_url": j.get("applyUrl") or j.get("jobUrl", ""),
                    "posted": (j.get("publishedAt") or "")[:10], "comp": comp, "text": j.get("descriptionPlain") or ""})
    return out


def greenhouse_jobs(b: dict) -> list[dict]:
    d = fetch_json(f"https://boards-api.greenhouse.io/v1/boards/{b['slug']}/jobs?content=true", timeout=40)
    out = []
    for j in d.get("jobs", []):
        loc = (j.get("location") or {}).get("name", "")
        out.append({"jid": str(j.get("id")), "title": j.get("title", ""), "location": loc, "remote": "remote" in loc.lower(),
                    "url": j.get("absolute_url", ""), "apply_url": j.get("absolute_url", ""),
                    "posted": (j.get("first_published") or j.get("updated_at") or "")[:10], "comp": "",
                    "text": clean(j.get("content", ""), 20000)})
    return out


def lever_jobs(b: dict) -> list[dict]:
    d = fetch_json(f"https://api.lever.co/v0/postings/{b['slug']}?mode=json", timeout=30)
    out = []
    for j in d:
        cats = j.get("categories") or {}
        loc = "; ".join(cats.get("allLocations") or [cats.get("location") or ""])
        text = " ".join([j.get("descriptionPlain") or "", j.get("additionalPlain") or ""] +
                        [clean(x.get("content"), 5000) for x in j.get("lists") or []])
        created = j.get("createdAt")
        out.append({"jid": j.get("id"), "title": j.get("text", ""), "location": loc,
                    "remote": (j.get("workplaceType") or "") == "remote", "url": j.get("hostedUrl", ""),
                    "apply_url": j.get("applyUrl") or j.get("hostedUrl", ""),
                    "posted": dt.datetime.fromtimestamp(created / 1000, dt.timezone.utc).strftime("%Y-%m-%d") if created else "",
                    "comp": (j.get("salaryRange") or {}).get("currency", "") and
                            f"{j['salaryRange'].get('min', '')} to {j['salaryRange'].get('max', '')} {j['salaryRange'].get('currency', '')}",
                    "text": text})
    return out


ATS = {"ashby": ashby_jobs, "greenhouse": greenhouse_jobs, "lever": lever_jobs}


def board_jobs(boards: list[dict], flt: JobFilter) -> list[dict]:
    out: list[dict] = []

    def one(b):
        try:
            rows = ATS[b["ats"]](b)
        except Exception as e:
            record_error(f"jobs:{b['ats']}:{b['slug']}", e)
            return []
        keep = []
        for r in rows:
            if not flt.title_ok(r["title"]) or not flt.us_ok(r["location"], r["remote"]):
                continue
            keep.append({"id": f"{b['ats']}:{b['slug']}:{r['jid']}", "company": b["name"], "title": clean(r["title"], 160),
                         "location": clean(r["location"], 160), "remote": r["remote"], "url": r["url"], "apply_url": r["apply_url"],
                         "posted": r["posted"], "comp": clean(r["comp"], 120),
                         "early_career": bool(flt.early.search(r["title"])), "sponsorship": flt.sponsorship(r["text"]),
                         "group": b["group"], "source": b["ats"]})
        return keep

    with cf.ThreadPoolExecutor(16) as ex:
        for rows in ex.map(one, boards):
            out += rows
    return out


SIMPLIFY = "https://raw.githubusercontent.com/SimplifyJobs/New-Grad-Positions/dev/.github/scripts/listings.json"
SIMPLIFY_SPONSOR = {"Offers Sponsorship": "sponsors", "Does Not Offer Sponsorship": "no_sponsorship",
                    "U.S. Citizenship is Required": "citizens_only"}


def simplify_jobs(flt: JobFilter, max_age_days: int = 30) -> list[dict]:
    try:
        rows = fetch_json(SIMPLIFY, timeout=60)
    except Exception as e:
        record_error("jobs:simplify", e)
        return []
    now = time.time()
    out = []
    for r in rows:
        if not r.get("active") or not r.get("is_visible", True):
            continue
        if now - (r.get("date_posted") or 0) > max_age_days * 86400:
            continue
        cat = r.get("category") or ""
        if "AI" not in cat and "Data" not in cat and not flt.title_ok(r.get("title", "")):
            continue
        if flt.exc.search(r.get("title", "")):
            continue
        loc = "; ".join(r.get("locations") or [])
        if not flt.us_ok(loc, False):
            continue
        out.append({"id": f"simplify:{r['id']}", "company": r.get("company_name", ""), "title": clean(r.get("title"), 160),
                    "location": clean(loc, 160), "remote": "remote" in loc.lower(), "url": r.get("url", ""), "apply_url": r.get("url", ""),
                    "posted": dt.datetime.fromtimestamp(r["date_posted"], dt.timezone.utc).strftime("%Y-%m-%d"), "comp": "",
                    "early_career": True, "sponsorship": SIMPLIFY_SPONSOR.get(r.get("sponsorship"), "not_stated"),
                    "group": "new-grad-list", "source": "simplify"})
    return out


HN_ML = re.compile(r"machine learning|\bML\b|\bMLE\b|\bAI\b|\bLLM|data scien|research engineer|applied scien|analytics engineer|data analyst", re.I)
HN_JUNIOR = re.compile(r"new grad|junior|entry[- ]level|early[- ]career|recent grad|intern|0[-+ ]?(to|-)? ?[12] years", re.I)
HREF = re.compile(r'href="([^"]+)"')


def hn_hiring(flt: JobFilter) -> list[dict]:
    try:
        stories = fetch_json("https://hn.algolia.com/api/v1/search_by_date?tags=story,author_whoishiring&hitsPerPage=10")["hits"]
        story = next(s for s in stories if (s.get("title") or "").startswith("Ask HN: Who is hiring"))
        sid = story["objectID"]
        hits = fetch_json(f"https://hn.algolia.com/api/v1/search?tags=comment,story_{sid}&hitsPerPage=1000", timeout=60)["hits"]
    except Exception as e:
        record_error("jobs:hn_who_is_hiring", e)
        return []
    out = []
    for h in hits:
        if str(h.get("parent_id")) != str(sid):
            continue
        raw = h.get("comment_text") or ""
        text = clean(raw, 4000)
        first = text.split(" | ")
        if not HN_ML.search(text) or not HN_JUNIOR.search(text):
            continue
        if not flt.us_ok(text[:300], "remote" in text[:300].lower()):
            continue
        links = [html.unescape(u) for u in HREF.findall(raw) if "ycombinator.com" not in u]
        permalink = f"https://news.ycombinator.com/item?id={h['objectID']}"
        out.append({"id": f"hn:{h['objectID']}", "company": clean(first[0], 60), "title": clean(" | ".join(first[1:3]), 160) or "See post",
                    "location": "", "remote": "remote" in text.lower(), "url": permalink, "apply_url": links[0] if links else permalink,
                    "posted": (h.get("created_at") or "")[:10], "comp": "", "early_career": True,
                    "sponsorship": flt.sponsorship(text), "group": "hn-who-is-hiring", "source": "hn", "excerpt": clean(text, 400)})
    return out


# ---------------------------------------------------------------- main

def main(argv: list[str]) -> int:
    date = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    if "--date" in argv:
        date = argv[argv.index("--date") + 1]
    skip = set(argv[argv.index("--skip") + 1].split(",")) if "--skip" in argv else set()
    out_dir = Path(argv[argv.index("--out") + 1]).resolve() if "--out" in argv else ROOT / "data"
    now = dt.datetime.now(dt.timezone.utc)
    window = now - dt.timedelta(days=3)
    cfg = load(ROOT / "config/sources.json", {})
    boards = load(ROOT / "config/job_boards.json", [])
    flt = JobFilter(load(ROOT / "config/job_filters.json", {}))

    # Jobs
    jobs = board_jobs(boards, flt) + simplify_jobs(flt) + hn_hiring(flt)
    dedup: dict[str, dict] = {}
    for j in jobs:
        dedup.setdefault(j["id"], j)
    jobs = list(dedup.values())
    seen_path = out_dir / "jobs/seen.json"
    seen: dict[str, str] = load(seen_path, {})
    first_run = not seen
    recent_cut = (now - dt.timedelta(days=14)).strftime("%Y-%m-%d")
    new_jobs = []
    for j in jobs:
        if j["id"] not in seen:
            seen[j["id"]] = date
            fresh = not j["posted"] or j["posted"] >= recent_cut
            if fresh and (not first_run or j["posted"] >= (now - dt.timedelta(days=3)).strftime("%Y-%m-%d")):
                new_jobs.append(j)
        j["first_seen"] = seen[j["id"]]
    open_ids = {j["id"] for j in jobs}
    prune_cut = (now - dt.timedelta(days=120)).strftime("%Y-%m-%d")
    seen = {k: v for k, v in seen.items() if k in open_ids or v >= prune_cut}
    group_order = {"application-log": 0, "new-grad-list": 1, "ai-startup": 2, "hn-who-is-hiring": 3, "larger": 4}

    def rank(j):
        return (not j["early_career"], j["sponsorship"] in ("no_sponsorship", "citizens_only"), group_order.get(j["group"], 5))

    for lst in (new_jobs, jobs):
        lst.sort(key=lambda j: j["posted"] or "", reverse=True)
        lst.sort(key=rank)

    # Learning feeds
    seen_items_path = out_dir / "seen_items.json"
    seen_items: dict[str, str] = load(seen_items_path, {})

    def fresh(items: list[dict], need_date: bool = True) -> list[dict]:
        out = []
        for it in items:
            key = uid(it["url"])
            pub = parse_date(it.get("published"))
            if need_date and (pub is None or pub < window):
                continue
            if key in seen_items and seen_items[key] != date:
                continue
            seen_items[key] = date
            out.append(it)
        return out

    since = int((now - dt.timedelta(hours=30)).timestamp())
    papers = hf_papers(cfg, date)
    posts = fresh(blog_posts(cfg))
    posts.sort(key=lambda p: p["published"], reverse=True)
    hn = hn_stories(cfg, since)
    reddit = [] if "reddit" in skip else fresh(reddit_posts(cfg), need_date=False)
    videos = [] if "youtube" in skip else fresh(youtube(cfg), need_date=False)
    ikw = kw_regex(cfg.get("interview_keywords", []))
    intel = [x for x in reddit + hn if ikw.search(x["title"])]
    seen_items = {k: v for k, v in seen_items.items() if v >= prune_cut}

    day = {"date": date, "generated_at": iso(now),
           "counts": {"jobs_open": len(jobs), "jobs_new": len(new_jobs), "papers": len(papers), "posts": len(posts),
                      "hn": len(hn), "reddit": len(reddit), "videos": len(videos), "interview_intel": len(intel), "errors": len(ERRORS)},
           "jobs_new": new_jobs[:80], "papers": papers, "posts": posts[:40], "hn": hn, "reddit": reddit, "videos": videos,
           "interview_intel": intel, "errors": ERRORS}
    save(out_dir / f"daily/{date}.json", day)
    save(out_dir / "daily/latest.json", day)
    idx = load(out_dir / "daily/index.json", {"dates": []})
    dates = sorted(set(idx.get("dates", [])) | {date}, reverse=True)
    save(out_dir / "daily/index.json", {"latest": dates[0], "dates": dates})
    save(out_dir / "jobs/open.json", {"generated_at": iso(now), "count": len(jobs), "jobs": jobs})
    save(seen_path, seen)
    save(seen_items_path, seen_items)
    print(json.dumps(day["counts"]))
    for e in ERRORS:
        print("source error:", e["source"], e["error"][:120], file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
