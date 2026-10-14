# ML Interview Prep: working context

A public study site plus a daily scraper, for one reader preparing for ML, AI,
data science and analytics interviews. Read `README.md`, then `docs/STYLE.md`,
`docs/CONTENT_SPEC.md` and `docs/AUTHORING.md` before changing content.

## House rules

1. **No em dashes or en dashes anywhere**: files, comments, commit messages, issue
   bodies, chat replies. Use commas, periods, colons or parentheses.
2. **No hyphens in ordinary compound phrases** ("real time", "end to end", "fine
   tuning"). Names and notation keep theirs (`k-means`, `GPT-4`).
3. **Never invent a number, a job posting, a source, a person or a URL.** Every
   job shown must come from a fetched posting and carry its direct apply link.
4. **Project numbers carry their caveats.** EvidenceOps figures come from a four
   article synthetic corpus run with a deterministic stub adapter, so the caveat
   goes in the same sentence as any EvidenceOps number. The BEIR replication
   numbers are real measurements and need no caveat.
5. **This repo is public.** No phone numbers, email addresses, visa or work
   authorisation details, or salary expectations, anywhere in it.
6. **Run `python3 tools/validate.py` before every commit.** CI blocks the deploy
   when it fails. Never weaken a rule in the validator to make content pass; fix
   the content.

## Daily Claude run

The GitHub Actions job (`.github/workflows/daily.yml`) does the mechanical scrape
at 10:17 UTC. A scheduled Claude run afterwards adds judgement:

1. Pull `main` and read `data/daily/<today>.json`. If it does not exist yet, stop.
2. Write `data/daily/<today>.notes.md`: at most 250 words, three to five bullets on
   what matters most today for someone interviewing for ML engineer, research
   engineer (retrieval, evals, agents) and data scientist roles at funded
   startups. Name the specific role, paper or post, link it, and say why it
   matters in one sentence. Prefer early career roles whose posting does not rule
   out sponsorship. Plain markdown, no heading line, no em dashes.
3. If a paper or post from today would make a good interview question, add at
   most three cards to the matching `content/<module>/cards.json`, following
   `docs/CONTENT_SPEC.md`, with `source` set to the item's URL.
4. Run `python3 tools/validate.py`, fix every error, then commit with the message
   `Daily notes <date>` and push.
5. Edit the open issue titled `Daily digest <date>` by adding a comment with the
   same notes, using `gh issue comment`.

Never apply to a job, email anyone, or change repo settings during the daily run.
