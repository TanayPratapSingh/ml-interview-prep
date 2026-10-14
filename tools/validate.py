"""Check content against docs/STYLE.md and docs/CONTENT_SPEC.md.

Usage:
    python3 tools/validate.py                 # everything under content/, docs/, README.md
    python3 tools/validate.py content/ml-coding
    python3 tools/validate.py --strict        # warnings count as errors

Exit code 1 when any error is found. Stdlib only.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DASHES = {"—": "em dash", "–": "en dash", "‒": "figure dash", "―": "horizontal bar"}

BANNED = [
    r"\bdelve[sd]?\b", r"\bdelving\b", r"\bfoster(s|ed|ing)?\b",
    r"\bleverag(e|es|ed|ing)\b(?!\s+(point|points|score|scores|statistic|value|values))",
    r"\butili[sz](e|es|ed|ing)\b", r"\bfacilitat(e|es|ed|ing)\b", r"\bempower(s|ed|ing)?\b",
    r"\bstreamlin(e|es|ed|ing)\b",
    r"\brobust\b(?!\s+(to|standard|regression|loss|estimator|estimators|statistic|statistics|covariance|scaling|scaler))",
    r"\bcutting[- ]edge\b", r"\bparadigm shift", r"\bgame[- ]changer", r"\btapestry\b", r"\brealms?\b",
    r"\bbeacons?\b", r"\bmultifaceted\b", r"\bmeticulous(ly)?\b", r"\bintricate(ly)?\b", r"\bparamount\b",
    r"\btransformative\b", r"\belevat(e|es|ed|ing)\b", r"\bembark(s|ed|ing)?\b", r"\bsupercharg(e|es|ed|ing)\b",
    r"\bharness(es|ed|ing)?\s+the\b", r"\bharnessing\b", r"\bever[- ]evolving\b",
    r"\bit'?s worth noting\b", r"\bit is worth noting\b", r"\bit'?s important to note\b",
    r"\bit is important to note\b", r"\bin today'?s world\b", r"\blet'?s dive in\b", r"\bat its core\b",
    r"\bin conclusion\b", r"\bhere'?s the thing\b", r"\blet me be clear\b",
]
BANNED_RE = [re.compile(p, re.I) for p in BANNED]

HYPHENATED = [
    "real-time", "end-to-end", "zero-shot", "few-shot", "one-shot", r"fine-tun\w*", "self-attention",
    "self-supervised", "cross-entropy", "cross-validation", "cross-validate", "cross-attention", "multi-head",
    "multi-class", "multi-label", "multi-task", "multi-modal", "multi-step", "multi-turn", r"pre-train\w*",
    r"post-train\w*", "chain-of-thought", r"trade-offs?", "open-source", "state-of-the-art", "human-in-the-loop",
    "test-time", "train-time", "long-term", "short-term", "one-hot", r"mini-batch\w*", "in-context",
    "large-scale", "small-scale", "high-level", "low-level", "well-known", "data-driven", "high-dimensional",
    "low-dimensional", "low-rank", "top-down", "bottom-up", "out-of-distribution", "out-of-sample", "in-sample",
    "hold-out", "held-out", "non-linear", "non-parametric", "semi-supervised", "step-by-step", r"follow-ups?",
    "rule-based", "model-based", "model-free", "on-policy", "off-policy", "decoder-only", "encoder-only",
    "encoder-decoder", "sequence-to-sequence", "next-token", "next-word", "left-to-right", "bias-variance",
    "precision-recall", "mean-squared", "log-likelihood", "log-odds", "closed-form", "gradient-based",
    "tree-based", "first-order", "second-order", "data-set", "pre-processing", "post-processing",
    "time-series", "real-world", "built-in", "so-called", "long-tail", "long-context", "general-purpose",
]
HYPH_RE = re.compile(r"(?<![\w/.-])(" + "|".join(HYPHENATED) + r")(?![\w-])", re.I)

FENCE_RE = re.compile(r"```.*?```", re.S)
INLINE_CODE_RE = re.compile(r"`[^`\n]*`")
DISPLAY_MATH_RE = re.compile(r"\$\$.*?\$\$", re.S)
INLINE_MATH_RE = re.compile(r"(?<![\\$])\$(?!\s)[^$\n]+?(?<!\s)\$(?!\d)")
URL_RE = re.compile(r"https?://\S+|\]\([^)]*\)")

PREFIXES = {"prob", "exp", "sql", "pa", "cml", "dl", "llm", "post", "rag", "sysd", "code", "narr", "lec", "qb"}
LEVELS = {"core", "advanced", "expert"}
SOURCE_TYPES = {"course", "book", "paper", "blog", "docs", "video", "question-bank", "repo", "forum"}
CARD_KEYS = {"id", "topic", "level", "q", "plain", "deep", "say", "must_mention", "followups", "traps", "source", "kind"}
KINDS = {"concept", "why", "compare", "formula", "example", "code", "interview"}
REQUIRED_CARD_KEYS = {"id", "topic", "level", "q", "plain", "deep", "say", "must_mention", "followups"}

LENGTHS = {"cheatsheet.md": (900, 1800), "plain.md": (2500, 5000), "deep.md": (6000, 12000)}


class Report:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def err(self, where: str, msg: str) -> None:
        self.errors.append(f"ERROR {where}: {msg}")

    def warn(self, where: str, msg: str) -> None:
        self.warnings.append(f"warn  {where}: {msg}")


def strip_code(text: str) -> str:
    """Remove the parts of a text the prose rules do not apply to."""
    for rx in (FENCE_RE, DISPLAY_MATH_RE, INLINE_CODE_RE, INLINE_MATH_RE, URL_RE):
        text = rx.sub(" ", text)
    return text


def line_of(text: str, index: int) -> int:
    return text.count("\n", 0, index) + 1


# Files that quote the banned vocabulary on purpose. They still get the dash check.
WORD_LIST_FILES = {"docs/STYLE.md"}


def check_text(rep: Report, where: str, text: str) -> None:
    for ch, name in DASHES.items():
        i = text.find(ch)
        while i != -1:
            rep.err(f"{where}:{line_of(text, i)}", f"{name} found: ...{text[max(0, i - 30):i + 30]!r}...")
            i = text.find(ch, i + 1)
    if where in WORD_LIST_FILES:
        return
    prose = strip_code(text)
    for rx in BANNED_RE:
        for m in rx.finditer(prose):
            rep.err(where, f"banned phrase {m.group(0)!r}")
    for m in HYPH_RE.finditer(prose):
        rep.err(where, f"hyphenated compound {m.group(0)!r}, write it with spaces")


def word_count(md: str) -> int:
    return len(re.findall(r"[A-Za-z0-9']+", FENCE_RE.sub(" ", md)))


def h2s(md: str) -> list[str]:
    return [m.group(1).strip() for m in re.finditer(r"^## (.+)$", FENCE_RE.sub("", md), re.M)]


def check_markdown(rep: Report, path: Path) -> None:
    where = str(path.relative_to(ROOT))
    md = path.read_text(encoding="utf-8")
    check_text(rep, where, md)
    if not md.startswith("# "):
        rep.err(where, "must start with a '# Title' line")
    if md.count("$$") % 2:
        rep.err(where, "odd number of $$ display math delimiters")
    if md.count("```") % 2:
        rep.err(where, "odd number of ``` code fences")
    for h in h2s(md) + [m.group(1) for m in re.finditer(r"^### (.+)$", md, re.M)]:
        if len(h) > 70:
            rep.warn(where, f"heading longer than 70 chars: {h[:40]}...")
    lo_hi = LENGTHS.get(path.name)
    if lo_hi and path.parent.name not in {"lectures", "question-bank"}:
        n = word_count(md)
        if n < lo_hi[0] * 0.8 or n > lo_hi[1] * 1.25:
            rep.warn(where, f"{n} words, target {lo_hi[0]} to {lo_hi[1]}")


def check_cards(rep: Report, path: Path) -> list[str]:
    where = str(path.relative_to(ROOT))
    try:
        cards = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        rep.err(where, f"invalid JSON: {e}")
        return []
    if not isinstance(cards, list):
        rep.err(where, "top level must be a list")
        return []
    plain = path.parent / "plain.md"
    topics = set(h2s(plain.read_text(encoding="utf-8"))) if plain.exists() else None
    ids: list[str] = []
    levels = {k: 0 for k in LEVELS}
    for i, c in enumerate(cards):
        cw = f"{where}[{i}]"
        if not isinstance(c, dict):
            rep.err(cw, "card must be an object")
            continue
        cid = c.get("id", "?")
        cw = f"{where}[{cid}]"
        missing = REQUIRED_CARD_KEYS - c.keys()
        extra = c.keys() - CARD_KEYS
        if missing:
            rep.err(cw, f"missing keys {sorted(missing)}")
        if extra:
            rep.err(cw, f"unknown keys {sorted(extra)}")
        m = re.fullmatch(r"([a-z]+)-(\d{3})", str(cid))
        if not m or m.group(1) not in PREFIXES:
            rep.err(cw, "id must look like 'prob-001' with a known prefix")
        ids.append(str(cid))
        if c.get("level") not in LEVELS:
            rep.err(cw, f"level must be one of {sorted(LEVELS)}")
        else:
            levels[c["level"]] += 1
        if topics is not None and c.get("topic") not in topics:
            rep.err(cw, f"topic {c.get('topic')!r} is not a ## heading in plain.md")
        for k in ("q", "plain", "deep", "say"):
            if not isinstance(c.get(k), str) or not c.get(k, "").strip():
                rep.err(cw, f"{k} must be a non empty string")
        say = c.get("say", "") or ""
        n = len(say.split())
        if n < 45 or n > 160:
            rep.err(cw, f"say is {n} words, must be 60 to 130")
        elif n < 60 or n > 130:
            rep.warn(cw, f"say is {n} words, target 60 to 130")
        if "$" in say or "\\" in say:
            rep.err(cw, "say must not contain math markup, write it the way you would say it")
        mm = c.get("must_mention", [])
        if not isinstance(mm, list) or not 3 <= len(mm) <= 6 or not all(isinstance(x, str) for x in mm):
            rep.err(cw, "must_mention must be a list of 3 to 6 strings")
        fu = c.get("followups", [])
        if not isinstance(fu, list) or not 1 <= len(fu) <= 4 or not all(
                isinstance(x, dict) and set(x) == {"q", "a"} for x in fu):
            rep.err(cw, "followups must be 1 to 4 objects with exactly q and a")
        tr = c.get("traps", [])
        if not isinstance(tr, list) or len(tr) > 3:
            rep.err(cw, "traps must be a list of at most 3 strings")
        if "kind" in c and c["kind"] not in KINDS:
            rep.err(cw, f"kind must be one of {sorted(KINDS)}")
        src = c.get("source")
        if src is not None and (not isinstance(src, dict) or not str(src.get("url", "")).startswith("http")):
            rep.err(cw, "source must be an object with title and an http url")
        texts = [c.get(k, "") for k in ("q", "plain", "deep", "say")] + list(mm or []) + list(tr or [])
        texts += [x.get(k, "") for x in (fu or []) if isinstance(x, dict) for k in ("q", "a")]
        check_text(rep, cw, "\n".join(t for t in texts if isinstance(t, str)))
        for k in ("plain", "deep"):
            v = c.get(k, "") or ""
            if v.count("$$") % 2:
                rep.err(cw, f"odd number of $$ in {k}")
    dupes = {x for x in ids if ids.count(x) > 1}
    if dupes:
        rep.err(where, f"duplicate ids {sorted(dupes)}")
    total = len(cards)
    if path.parent.name not in {"lectures", "question-bank"} and not 80 <= total <= 160:
        rep.warn(where, f"{total} cards, target 80 to 150")
    if total:
        share = {k: round(100 * v / total) for k, v in levels.items()}
        if share["expert"] > 30 or share["core"] < 25:
            rep.warn(where, f"level mix {share}, target about 45 core / 40 advanced / 15 expert")
    return ids


def check_sources(rep: Report, path: Path) -> None:
    where = str(path.relative_to(ROOT))
    try:
        srcs = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        rep.err(where, f"invalid JSON: {e}")
        return
    if not isinstance(srcs, list):
        rep.err(where, "top level must be a list")
        return
    urls = []
    for i, s in enumerate(srcs):
        sw = f"{where}[{i}]"
        if not isinstance(s, dict):
            rep.err(sw, "entry must be an object")
            continue
        for k in ("title", "url", "type", "why", "checked"):
            if not s.get(k):
                rep.err(sw, f"missing {k}")
        if not str(s.get("url", "")).startswith("http"):
            rep.err(sw, "url must start with http")
        if s.get("type") not in SOURCE_TYPES:
            rep.err(sw, f"type must be one of {sorted(SOURCE_TYPES)}")
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(s.get("checked", ""))):
            rep.err(sw, "checked must be a YYYY-MM-DD date")
        urls.append(s.get("url"))
        check_text(rep, sw, f"{s.get('title', '')}\n{s.get('why', '')}")
    if len(set(urls)) != len(urls):
        rep.err(where, "duplicate urls")


def check_module(rep: Report, d: Path, all_ids: dict[str, str]) -> None:
    for name in ("cheatsheet.md", "plain.md", "deep.md"):
        p = d / name
        if p.exists():
            check_markdown(rep, p)
        elif d.name not in {"lectures", "question-bank"}:
            rep.err(str(d.relative_to(ROOT)), f"missing {name}")
    plain, deep = d / "plain.md", d / "deep.md"
    if plain.exists() and deep.exists():
        ph = h2s(plain.read_text(encoding="utf-8"))
        dh = [h for h in h2s(deep.read_text(encoding="utf-8")) if h not in {"What interviewers probe", "Sources"}]
        if ph != dh:
            rep.err(str(d.relative_to(ROOT)), f"plain.md and deep.md ## headings differ.\n  plain: {ph}\n  deep:  {dh}")
    for p in sorted(d.glob("*.md")):
        if p.name not in LENGTHS:
            check_markdown(rep, p)
    if (d / "cards.json").exists():
        for cid in check_cards(rep, d / "cards.json"):
            if cid in all_ids and all_ids[cid] != d.name:
                rep.err(str(d.relative_to(ROOT)), f"card id {cid} also used in {all_ids[cid]}")
            all_ids[cid] = d.name
    if (d / "sources.json").exists():
        check_sources(rep, d / "sources.json")
    elif d.name not in {"lectures", "question-bank"}:
        rep.err(str(d.relative_to(ROOT)), "missing sources.json")


def main(argv: list[str]) -> int:
    strict = "--strict" in argv
    targets = [Path(a).resolve() for a in argv if not a.startswith("--")]
    rep = Report()
    all_ids: dict[str, str] = {}
    if not targets:
        content = ROOT / "content"
        targets = sorted(p for p in content.iterdir() if p.is_dir() and any(p.iterdir())) if content.exists() else []
        targets += [ROOT / "docs", ROOT / "README.md", ROOT / "CLAUDE.md"]
    for t in targets:
        if not t.exists():
            continue
        if t.is_dir() and t.parent.name == "content":
            check_module(rep, t, all_ids)
        elif t.is_dir():
            for p in sorted(t.rglob("*.md")):
                check_text(rep, str(p.relative_to(ROOT)), p.read_text(encoding="utf-8"))
        elif t.suffix == ".md":
            check_text(rep, str(t.relative_to(ROOT)), t.read_text(encoding="utf-8"))
        elif t.suffix == ".json":
            check_text(rep, str(t.relative_to(ROOT)), t.read_text(encoding="utf-8"))
    for line in rep.warnings:
        print(line)
    for line in rep.errors:
        print(line)
    n_err = len(rep.errors) + (len(rep.warnings) if strict else 0)
    print(f"\n{len(rep.errors)} errors, {len(rep.warnings)} warnings")
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
