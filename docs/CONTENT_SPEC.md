# Content spec

Each study module lives in `content/<module-id>/` and has exactly these files.
The site reads them directly, so names and shapes matter.

| File | What it is | Target size |
| --- | --- | --- |
| `cheatsheet.md` | One page dense reference | 900 to 1,800 words |
| `plain.md` | Easy version, intuition first | 2,500 to 5,000 words |
| `deep.md` | Exhaustive technical version | 6,000 to 12,000 words |
| `cards.json` | Flashcards with spoken answers | 45 to 70 cards |
| `sources.json` | Every source used, all fetched and checked | 8 to 25 entries |

Read `docs/STYLE.md` first. The writing rules there are not optional.

## Markdown files

- Start with a single `# Title` line, then a one paragraph summary.
- Use `##` for sections and `###` for subsections. The site builds the table of
  contents from these, so keep headings short (under 60 characters).
- `plain.md` and `deep.md` cover the same sections in the same order, so a reader
  can switch depth and land in the same place. Use identical `##` headings in both.
- Callouts are blockquotes that start with a bold label, one of:
  `> **Interview trap:**`, `> **Say it like this:**`, `> **Worked example:**`,
  `> **Watch:**` (a lecture link), `> **Remember:**`.
- Link a lecture from `data/playlists.json` as
  `[CME295 Lecture 2](https://www.youtube.com/watch?v=yT84Y5zCnaA)`, and add a
  timestamp parameter (`&t=1234s`) only if you verified the time.
- Code blocks carry a language tag (`python`, `sql`, `text`).
- `deep.md` ends with `## What interviewers probe` (the follow up chains you should
  expect) and `## Sources` (the same items as `sources.json`, as a list of links).

## cards.json

A JSON array. Build it with a Python script and `json.dump(..., ensure_ascii=False,
indent=1)` so escaping is always right.

```json
{
  "id": "prob-001",
  "topic": "Hypothesis testing",
  "level": "core",
  "q": "What does a p value tell you, and what does it not tell you?",
  "plain": "Two to five sentences a non specialist understands. Markdown allowed.",
  "deep": "The full technical answer. Markdown and KaTeX allowed. As long as it needs to be.",
  "say": "The 60 to 130 word spoken answer, first person.",
  "must_mention": ["probability of data this extreme if the null is true", "not P(null | data)", "depends on sample size"],
  "followups": [{"q": "So what would you report instead?", "a": "An effect size with a confidence interval, plus the p value."}],
  "traps": ["Saying p = 0.03 means a 3 percent chance the null is true."],
  "source": {"title": "Where this question was seen asked", "url": "https://..."}
}
```

Field rules:

- `id`: module prefix plus three digits, unique. Prefixes: `prob`, `exp`, `sql`,
  `pa`, `cml`, `dl`, `llm`, `post`, `rag`, `sysd`, `code`, `narr`, `lec`.
- `topic`: one of the `##` headings in `plain.md`, copied exactly.
- `level`: `core` (asked in most screens), `advanced` (onsite depth), `expert`
  (research lab depth). Aim for roughly 45 / 40 / 15 percent.
- `q`: phrased the way an interviewer asks it.
- `say`: 60 to 130 words. See the spoken answer rules in `docs/STYLE.md`.
- `must_mention`: 3 to 6 short phrases. The narration drill shows these as a
  checklist after you answer out loud.
- `followups`: 1 to 4 objects with `q` and a short `a`.
- `traps`: 0 to 3 wrong answers or pitfalls interviewers listen for.
- `source`: optional. Include it when the question was seen in a real interview
  write up or question bank, and only with a URL you fetched.

## sources.json

```json
[{"title": "Stanford CS229 lecture notes", "url": "https://cs229.stanford.edu/main_notes.pdf",
  "type": "course", "why": "One line on what this source is best for.", "checked": "2026-10-13"}]
```

`type` is one of `course`, `book`, `paper`, `blog`, `docs`, `video`, `question-bank`,
`repo`, `forum`.

## Validation

Run `python3 tools/validate.py content/<module-id>` after writing. Fix every error.
Warnings are about length targets and are allowed only with a reason.
