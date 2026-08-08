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
   with its `query_template`. Here both domains emit the same `<decision>{…}</decision>`
   JSON shape, so both configs name the same `json_block` parser and only the
   config differs; a domain whose completions are shaped differently would also
   swap the parser (write a new YAML, add a parser only if needed).
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

The proof that it is genuinely one code path is that `score_domain` is
**branch-free**: it contains zero per-domain logic. What each domain must uphold,
asserted per domain, is the reward contract — a well-formed correct completion
out-scores a certified-but-wrong one, which out-scores a malformed completion
floored to `clip[0]`.

The two domains happen to land on the same numbers here — correct (`+1.900`) >
certified-but-wrong (`+0.750`) > malformed floor (`-1.000`) — but that is an
illustrative coincidence of the matched payloads (both give the same `graded`
component), **not** the proof. Different recorded payloads would shift the
numbers while the unchanged code path still upholds the ordering.
