# Swap-the-rule-set demo

`examples/cross_domain_demo.py` scores two very different domains —
**grant eligibility** and **ACMG variant classification** — through a single,
domain-agnostic function, `score_domain`. It is the concrete demonstration of
design principle 4: the same library trains on ≥2 domains by swapping only
**config + parser**, with no per-domain forks.

```bash
python examples/cross_domain_demo.py   # offline, network-free (FakeVerifier)
```

## What changes between domains

Only two things, and neither is code:

1. **The config** — `configs/grant_eligibility.yaml` vs `configs/acmg.yaml`.
   The config selects the platform, the reward weights, and the parser together
   with its `query_template`. Add a domain = write a new YAML (and a parser only
   if its completions are shaped differently).
2. **The recorded platform payloads** — the domain's certified "rule set",
   replayed offline by `FakeVerifier` so the demo needs no API key or network.
   On a live run these come from the platform; you swap `FakeVerifier` for
   `run.reward_function()` and the scoring code is unchanged.

## What stays constant (the one code path)

`score_domain` is identical for every domain:

- `load_run_config(config)` → a fully-wired `RunConfig` (parser + shaper + floor).
- `FakeVerifier(parser=…, shaper=…, floor=…).as_reward_function()` → the batch
  reward function (parser → verifier → `DefaultRewardShaper`), exactly the path
  real training uses.
- The reward contract: parse → verify → shape → clip; an unparseable completion
  fails closed to the floor and never out-scores a certified one.

Because the shaper and report shape are shared, the two domains produce the same
reward **spread** — correct (`+1.900`) > certified-but-wrong (`+0.750`) >
malformed floor (`-1.000`) — even though their completions, facts, queries, and
rule sets are all different. That identical spread from independent inputs is
the evidence that it is genuinely one code path.
