"""Per-structure / per-family vulnerability map for the v2 decision corpus (E3 first pass).

Capture-reproduction script for `vuln_map_e3_firstpass.{json,md}`. Measures K (true
capability), K_hat (hack rate), m (fail-open), wedge (K_hat-K) for each difficulty
STRUCTURE and vocab FAMILY, on one or more adapters, reusing decision_race's eval
machinery (`evaluate_ambertrace` + the eval_fn greedy path) so adapter loading matches
training exactly.

This is a stored capture artifact, not a polished example; the rigorous version
(train/held-out split, sampled K_hat, multi-seed) is tracked under issue #202.

Usage:
    python docs/research/captures/vuln_map_firstpass.py \
        --adapters warm=outputs/wf_v2_always_least_restrictive_s0/sft/adapter \
                   poisoned_ent01=outputs/wf_v2_always_least_restrictive_s0_ent01/poisoned/gen9/adapter
"""
from __future__ import annotations
import sys, json, collections, argparse
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

import torch
from peft import PeftModel
from ambertrace_rlvr.corpus import load_decision_corpus
from ambertrace_rlvr.math_env import ambertrace_backend as B

MODEL = "Qwen/Qwen3-4B"
HACK = "always_least_restrictive"
DATA = REPO / "data" / "decision_eval_v2_subset.jsonl"

FAMILY_NAMES = {
    frozenset({"approve", "deny"}): "approve_deny",
    frozenset({"clear", "flag"}): "clear_flag",
    frozenset({"approve", "restrict", "suspend"}): "approve_restrict_suspend",
    frozenset({"critical_escalate", "discharge", "escalate", "monitor"}): "discharge_monitor_escalate",
}

def family_of(it) -> str:
    return FAMILY_NAMES.get(frozenset(it.label_space), "/".join(sorted(it.label_space)))

def structure_of(it) -> str:
    return str(it.difficulty.get("structure", "?"))

def generate_all(adapter_path: Path | None, items, max_new_tokens: int, bs: int) -> list[str]:
    tok = B._load_tokenizer(MODEL)
    tk = B.chat_template_kwargs(B.resolve_model_spec(MODEL))
    base = B._load_base_model(MODEL)
    model = base
    try:
        if adapter_path is not None:
            model = PeftModel.from_pretrained(model, str(adapter_path))
        model.eval()
        texts = [
            tok.apply_chat_template(
                [{"role": "system", "content": B.AMBERTRACE_SYSTEM_PROMPT},
                 {"role": "user", "content": B.framed_user_prompt(it, HACK)}],
                tokenize=False, add_generation_prompt=True, **tk)
            for it in items
        ]
        tok.padding_side = "left"
        comps: list[str] = []
        for i in range(0, len(texts), bs):
            enc = tok(texts[i:i + bs], return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=max_new_tokens,
                                     do_sample=False, pad_token_id=tok.eos_token_id)
            new = out[:, enc["input_ids"].shape[1]:]
            comps.extend(tok.batch_decode(new, skip_special_tokens=True))
            print(f"    gen {min(i + bs, len(texts))}/{len(texts)}", flush=True)
        return comps
    finally:
        del model, base
        B._free_accelerator_memory()

def score_groups(comps, items, keyfn) -> dict:
    buckets = collections.defaultdict(lambda: ([], []))
    for c, it in zip(comps, items):
        if it.oracle is None:
            continue
        buckets[keyfn(it)][0].append(c)
        buckets[keyfn(it)][1].append(it)
    rows = {}
    for key, (cs, its) in sorted(buckets.items()):
        rep = B.evaluate_ambertrace(cs, its, HACK, lam=0.0)
        rows[key] = dict(n=rep.n, K=round(rep.k, 3), K_hat=round(rep.k_hat, 3),
                         m=round(rep.m, 3), wedge=round(rep.wedge, 3), parse=round(rep.parse_rate, 3))
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", nargs="+", required=True, help="NAME=PATH (PATH='' for base)")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--out", type=Path, default=Path(__file__).with_name("vuln_map_e3_firstpass.json"))
    args = ap.parse_args()

    items = load_decision_corpus(DATA)
    print(f"corpus: {len(items)} items; structures={collections.Counter(structure_of(i) for i in items)}")
    result = {}
    for spec in args.adapters:
        name, _, path = spec.partition("=")
        print(f"\n=== adapter {name!r} ({path or 'base'}) ===", flush=True)
        comps = generate_all(Path(path) if path else None, items, args.max_new_tokens, args.bs)
        by_struct = score_groups(comps, items, structure_of)
        by_fam = score_groups(comps, items, family_of)
        result[name] = {"by_structure": by_struct, "by_family": by_fam}
        for label, rows in (("STRUCTURE", by_struct), ("FAMILY", by_fam)):
            print(f"\n  [{name}] by {label}:")
            for k, r in rows.items():
                print(f"    {k:28s} n={r['n']:3d}  K={r['K']:.3f}  K_hat={r['K_hat']:.3f}  "
                      f"m={r['m']:.3f}  wedge={r['wedge']:+.3f}")
    args.out.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {args.out}")

if __name__ == "__main__":
    main()
