# Decision Eval v1 — oracle-labelled decision benchmark

`decision_eval_v1.jsonl` — **1,350 decision items across 225 policy domains**, each
with an exact **oracle-certified** correct action. A generic benchmark for scoring
any LLM on graded decision-making: accuracy, and — more importantly — the *safety
direction* of its errors (fail-open vs. over-cautious) against a fixed ground truth.

## Why oracle-anchored

Most LLM evals score against human labels or an LLM judge — both carry noise and
can be gamed. Here every item's correct action is fixed by the **AmberTrace
decision oracle**, independently of any model. That makes the errors *signed*: a
model choosing a *less* restrictive action than required is a safety-relevant
**fail-open**, while a *more* restrictive one is merely over-cautious — a
distinction a plain accuracy number discards.

## Schema (one JSON object per line)

| field | meaning |
|---|---|
| `id` | stable item id |
| `domain` | opaque domain id (many items per domain) |
| `prompt` | the full decision task: policy description + case facts + the allowed actions. **This is the only field shown to the model.** |
| `vocabulary` | the ordered action set: `{verb, rank, restrictive}` (lower `rank` = more restrictive; `restrictive` marks the safety-critical / fail-closed side) |
| `oracle` | the certified correct action — the **answer key**. Never put this in the model's prompt. |
| `undecidable` | `true` if the policy fixes no determinate action (v1: always `false` — see limits) |
| `difficulty` | `{structure}` — the rule form: `baseline` / `ratio` / `precedence` / `negation` / `multi_trigger` — for stratified reporting |

Score with `ambertrace_rlvr`:

```python
from ambertrace_rlvr import load_decision_corpus, judgments_for, parse_model_answer, score_deviation

items = load_decision_corpus("data/decision_eval_v1.jsonl")
judgments = judgments_for(items)                      # the fixed oracle truth
spec = items[0].spec()                                # per-domain vocabulary
answers = [parse_model_answer(model(it.prompt), it.label_space) for it in items]
report = score_deviation(judgments, answers, spec)
print(report.accuracy, report.over_permit_rate)       # alignment scores
```

## How the labels were made

Each correct action was certified by AmberTrace's verified decision oracle on the
item's fixed inputs. The oracle's *internals* are not part of this dataset — only
its labels ship.

## Limits (v1)

- **Decidable-only.** Every item has a determinate answer; there are no
  certified-undecidable items yet, so this set does not exercise abstention /
  overconfidence (`overconfidence_rate`). A later version will add them.
- English prompts; single graded action space; synthetic policy domains.

## Make your own

This set is fixed, but the pipeline is not: with the
[`ambertraceai`](https://pypi.org/project/ambertraceai/) SDK and an AmberTrace API
key you can **author your own decision domain, build its oracle, and generate an
eval set on your own data** — see `examples/generate_eval_set.py`. Sign up at
[ambertrace.ai](https://ambertrace.ai).

## License

Released under the repository's MIT license.
