"""Render the daily digest issue body from data/daily/latest.json.

Usage:
    python3 scripts/digest.py > digest.md
    python3 scripts/digest.py --date 2026-10-14 > digest.md

Every job line carries its direct apply link. Stdlib only.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = os.environ.get("SITE_URL", "https://tanaypratapsingh.github.io/ml-interview-prep/")
OWNER = os.environ.get("DIGEST_MENTION", "@TanayPratapSingh")

SPONSOR_LABEL = {"sponsors": "sponsorship stated", "no_sponsorship": "no sponsorship", "citizens_only": "US citizens or clearance only",
                 "not_stated": "sponsorship not stated"}


def load(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def all_cards() -> list[dict]:
    """Same card list as allCards() in assets/app.js: every module in data/modules.json plus lectures."""
    ids = [m["id"] for m in load(ROOT / "data/modules.json", []) if m.get("ready")] + ["lectures"]
    cards = []
    for mid in ids:
        for c in load(ROOT / "content" / mid / "cards.json", []):
            c["module"] = mid
            cards.append(c)
    return sorted(cards, key=lambda c: c["id"])


def drill_pick(cards: list[dict], date: str, n: int = 5) -> list[dict]:
    """Deterministic daily pick. assets/app.js implements the same function."""
    if not cards:
        return []
    y, m, d = (int(x) for x in date.split("-"))
    import datetime as dt
    day = (dt.date(y, m, d) - dt.date(2026, 1, 1)).days
    picks, used = [], set()
    i = 0
    while len(picks) < min(n, len(cards)):
        idx = (day * 7919 + i * 104729) % len(cards)
        i += 1
        if idx in used:
            continue
        used.add(idx)
        picks.append(cards[idx])
    return picks


def esc(s: str) -> str:
    return (s or "").replace("|", "/").replace("\n", " ").strip()


def main(argv: list[str]) -> int:
    day = load(ROOT / "data/daily/latest.json", {})
    if "--date" in argv:
        day = load(ROOT / f"data/daily/{argv[argv.index('--date') + 1]}.json", day)
    if not day:
        print("No daily data found.")
        return 1
    date = day["date"]
    modules = {m["id"]: m["title"] for m in load(ROOT / "data/modules.json", [])}
    c = day.get("counts", {})
    out = [f"{OWNER} here is today's update. Full page: {SITE}#/today", ""]

    notes = ROOT / f"data/daily/{date}.notes.md"
    if notes.exists():
        out += ["## What matters today", "", notes.read_text(encoding="utf-8").strip(), ""]

    jobs = day.get("jobs_new", [])
    out += [f"## New roles ({len(jobs)} new, {c.get('jobs_open', 0)} open in total)", ""]
    if jobs:
        out += ["| Company | Role | Where | Posted | Sponsorship | Apply |", "| --- | --- | --- | --- | --- | --- |"]
        for j in jobs[:40]:
            role = esc(j["title"]) + (" (early career)" if j.get("early_career") else "")
            if j.get("comp"):
                role += f", {esc(j['comp'])}"
            out.append(f"| {esc(j['company'])} | {role} | {esc(j.get('location', ''))[:60] or ('Remote' if j.get('remote') else '')} | "
                       f"{j.get('posted') or 'n/a'} | {SPONSOR_LABEL.get(j.get('sponsorship'), 'unknown')} | [Apply]({j['apply_url']}) |")
        if len(jobs) > 40:
            out.append(f"\n{len(jobs) - 40} more on the site: {SITE}#/jobs")
    else:
        out.append("No new matching roles since yesterday.")
    out.append("")

    picks = drill_pick(all_cards(), date)
    if picks:
        out += ["## Today's spoken drill", "", f"Answer each out loud in under 45 seconds, then check yourself: {SITE}#/drill/today", ""]
        for p in picks:
            out.append(f"- {p['q']} ({modules.get(p['module'], p['module'])}, {p['level']})")
        out.append("")

    if day.get("papers"):
        out += ["## Papers people are reading", ""]
        for p in day["papers"][:5]:
            out.append(f"- [{p['title']}]({p['url']}) ({p['upvotes']} upvotes). {p['summary'][:220]}")
        out.append("")
    if day.get("posts"):
        out += ["## New posts from the reading list", ""]
        for p in day["posts"][:10]:
            out.append(f"- [{p['title']}]({p['url']}), {p['source']}")
        out.append("")
    if day.get("hn"):
        out += ["## Hacker News", ""]
        for h in day["hn"][:6]:
            out.append(f"- [{h['title']}]({h['url']}) ({h['points']} points, [discussion]({h['discussion']}))")
        out.append("")
    if day.get("videos"):
        out += ["## New lectures and videos", ""]
        for v in day["videos"][:8]:
            tag = " (new in a playlist you follow)" if v.get("new_in_playlist") else ""
            out.append(f"- [{v['title']}]({v['url']}), {v['source']}{tag}")
        out.append("")
    if day.get("interview_intel"):
        out += ["## Interview intel", ""]
        for x in day["interview_intel"][:8]:
            out.append(f"- [{x['title']}]({x['url']}), {x['source']}")
        out.append("")
    if day.get("errors"):
        names = ", ".join(sorted({e["source"] for e in day["errors"]}))
        out += [f"Sources that failed today and were skipped: {names}.", ""]
    out.append(f"Generated {day.get('generated_at', '')} by the daily workflow.")
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
