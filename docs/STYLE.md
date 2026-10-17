# Writing rules

Every file in this repo follows these rules. `python3 tools/validate.py` enforces
the mechanical ones and CI fails when they break.

## Hard rules (the validator checks these)

1. **No em dashes and no en dashes.** Not in prose, headings, tables, code comments
   or commit messages. Use a comma, a period, a colon or parentheses. Write numeric
   ranges as "2 to 4".
2. **No hyphens in ordinary compound phrases.** Write "real time", "end to end",
   "zero shot", "few shot", "fine tuning", "self attention", "cross entropy",
   "multi head", "pre training", "chain of thought", "trade off", "open source",
   "state of the art", "human in the loop", "test time", "long term", "one hot",
   "mini batch", "cross validation", "in context", "large scale", "high level".
   Hyphens stay where they are part of a name or notation: `k-means`, `t-test`,
   `t-SNE`, `GPT-4`, `Llama-3`, `bge-small`, URLs, code blocks and inline code.
3. **None of these words:** delve, foster, leverage, utilize, facilitate, empower,
   streamline, robust, cutting-edge, paradigm shift, game changer, tapestry, realm,
   beacon, multifaceted, meticulous, intricate, paramount, transformative, elevate,
   embark, supercharge, harness, ever-evolving, "it's worth noting", "it's important
   to note", "in today's world", "let's dive in", "at its core".
   Robust is allowed only as the statistics term: the validator accepts `robust to`,
   `robust standard`, `robust regression`, `robust loss`, `robust estimator`.
4. **Never invent a number, a source, a quote or a URL.** Every URL in a sources
   file was fetched during research. Every benchmark figure names the paper or page
   it came from.

## How the writing should sound

- Lead with the answer. The first sentence of any explanation says what the thing
  is or what the answer is. Background comes after.
- Concrete beats abstract. "A test with 80 percent power misses a real 2 point lift
  one time in five" beats "power measures sensitivity".
- Use worked examples with real numbers, and check the arithmetic by running it.
- Active voice, plain verbs. "The model memorises the training set" not "overfitting
  is experienced".
- Second person for teaching ("you"), first person in spoken answers ("I").
- No throat clearing, no recap endings, no "In conclusion", no "Here's the thing",
  no fake reveals ("The secret: ..."), no "not X, it's Y" constructions, no emoji.
- Bold only for a label at the start of a line ("**Interview trap:** ...") or a
  defined term the first time it appears. Never bold scattered mid sentence.
- Tables when you compare three or more things on the same attributes. Prose when
  you explain a mechanism.

## The three depths

| Depth | Reader | What it contains |
| --- | --- | --- |
| Cheatsheet | You, ten minutes before the interview | Dense one page reference: formulas, decision tables, "if asked X, say Y" lines, the numbers worth memorising |
| Plain | A smart friend with no ML background | Intuition first, an analogy, one worked example with numbers, where it breaks. Every technical term kept and explained in plain words where it first appears, short sentences, formulas said in words. Nothing technical is left out |
| Deep | An interviewer at a research lab | Exhaustive: definitions, derivations interviewers ask you to reproduce, assumptions, complexity, edge cases, comparisons, failure modes, what interviewers probe, citations |

## Spoken answers (the `say` field on every card)

The spoken answer is what you would say out loud in 30 to 45 seconds.

- 60 to 130 words. First person. Contractions are fine.
- Sentence one answers the question directly.
- Then one intuition or example, then one caveat or trade off, then stop.
- No bullet lists, no formulas you cannot say aloud (say "the square root of n",
  not `\sqrt{n}`), no "great question".
- It should sound like a person thinking clearly, not like a textbook read aloud.

## Math

KaTeX. Inline math between single dollar signs, display math between `$$` on their
own lines. In JSON strings escape the backslash (`\\frac`). Write money as
"USD 40" or "40 dollars" so it never parses as math.
