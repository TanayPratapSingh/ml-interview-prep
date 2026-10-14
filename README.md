# ML Interview Prep

Study material and a daily feed for ML engineer, AI engineer, research engineer,
data scientist and analyst interviews. Live at
**https://tanaypratapsingh.github.io/ml-interview-prep/**

## What is in it

Twelve study modules, each written at three depths with matching headings, so you
can switch depth mid read and land in the same section:

- **Cheatsheet**: the dense one page refresher for the ten minutes before a call.
- **Plain**: the intuition, an analogy and one worked example, for a smart friend
  with no ML background.
- **Deep**: derivations, assumptions, failure modes, comparisons and the follow up
  chains a research lab interviewer will push on, with citations.

Every module also has flashcards. A card carries a plain answer, a deep answer,
a spoken answer of 60 to 130 words to say out loud, the points a strong answer must
mention, follow up questions with answers, and the traps interviewers listen for.

| Group | Modules |
| --- | --- |
| Foundations | Probability and statistics |
| Analytics | Experiments and causal inference, SQL and data wrangling, Product analytics and cases |
| Machine learning | Classical machine learning, Deep learning foundations |
| LLMs | Transformers and LLMs, Post training and reasoning, RAG, agents and evals |
| Systems | ML system design |
| Interview craft | ML coding, Narration and your projects |

The site also has:

- **Cards**: spaced repetition (a small SM-2 variant) with keyboard shortcuts.
  Progress stays in your browser; export it from the Progress page.
- **Drill**: a spoken answer trainer. You get a question and a timer, answer out
  loud, optionally record yourself, then check your answer against the must mention
  list, a delivery rubric and the model spoken answer, and answer the follow up.
- **Lectures**: the four playlists the material is built around, Stanford CME295
  (Transformers and LLMs, Autumn 2025), Stanford CS229 (Machine Learning, Spring
  2026), Stanford CS329A (Self Improving AI Agents) and Andrej Karpathy's Neural
  Networks: Zero to Hero. Modules link the lecture that teaches each section.
- **Questions**: a bank of real interview questions, each linked to where it was
  seen, filterable by module, kind, role and level.

## How it updates

`.github/workflows/daily.yml` runs every morning at 10:17 UTC on GitHub's servers:

1. `scripts/daily.py` pulls open roles from 106 company job boards (Ashby,
   Greenhouse and Lever public APIs), the SimplifyJobs new grad list and the
   current Hacker News "Who is hiring" thread, then filters to ML, AI, data and
   analytics titles in the US or remote, dropping senior, staff, lead and manager
   titles. It also pulls the top Hugging Face daily papers, new posts from 24
   blogs, AI stories from Hacker News, Reddit, and new videos in the four
   playlists.
2. The data is committed to `data/`, so every day is archived in git.
3. `scripts/digest.py` writes a digest and the workflow opens it as an issue
   labelled `daily`. GitHub emails the repo owner about new issues. Every role in
   the digest carries its direct apply link.
4. The site redeploys to GitHub Pages.

A failed source is logged in that day's file and skipped. It never stops the run.

### What the job data can and cannot tell you

- Postings come straight from the companies' own job board APIs or from the
  community maintained SimplifyJobs list. Nothing is generated.
- The sponsorship label is keyword matching on the posting text. "Not stated"
  means the posting says nothing either way. Check before you apply.
- Title filtering is a regular expression list in `config/job_filters.json`, so it
  will miss some roles and let a few odd ones through.

## Layout

```text
index.html, assets/        the site (plain JavaScript, no build step)
content/<module>/          cheatsheet.md, plain.md, deep.md, cards.json, sources.json
content/question-bank/     questions.json, interview-loops.md, resources.md, study-plan.md
content/lectures/          notes for the four playlists
data/                      modules.json, playlists.json, daily/, jobs/
config/                    feeds, job boards and job filters
scripts/                   daily.py (scrape), digest.py (issue body)
tools/validate.py          content and style checks, run in CI
docs/                      STYLE.md, CONTENT_SPEC.md, AUTHORING.md
```

## Run it locally

```bash
python3 -m http.server 8000
```

Then open http://localhost:8000. The scraper needs only the Python standard
library:

```bash
python3 scripts/daily.py --skip youtube --out /tmp/test-data
```

## Writing rules

Content follows `docs/STYLE.md`: no em or en dashes, no hyphens in ordinary
compound phrases, no filler vocabulary, and no number, source or URL that was not
checked. `python3 tools/validate.py` enforces the mechanical rules and CI runs it
on every push.

## License

MIT for the code. The lecture videos belong to Stanford and Andrej Karpathy; this
repo links to them and does not copy them.
