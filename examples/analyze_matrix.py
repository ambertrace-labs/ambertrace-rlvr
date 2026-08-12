"""Multi-angle analysis of the full-run alignment data (local research helper).

Reconstructs each model's AlignmentRow from row_full_*.json and computes: CAS under
all three schemes, the signed safety direction, reasoning-taxonomy / size / speed /
lab effects, per-structure & action-count difficulty, confusion structure from the
per-item answers, and the two-build Qwen comparison."""
from __future__ import annotations
import glob, json, os, statistics as st
from collections import Counter, defaultdict
from pathlib import Path
import sys
sys.path.insert(0, "src")
from ambertrace_rlvr.deviation import (DeviationReport, SAFETY_FIRST, BALANCED, CAPITAL_ADEQUACY,
                                       SAFETY_FIRST_SEVERITY, BALANCED_SEVERITY, CAPITAL_ADEQUACY_SEVERITY)
from ambertrace_rlvr.matrix import AlignmentRow, score_matrix_cas

REPO = Path(__file__).resolve().parent.parent
CTOR = ("correct", "over_permit", "over_deny", "refusal_on_certified",
        "parse_fail_on_certified", "overconfident", "mutual_abstain", "unverifiable")
# key -> (display, lab, params_B, flag)  (params as a number for correlation)
META = {
    "muse-glimmer-30b": ("Muse-Glimmer-30B", "Meta", 30, "‡"),
    "olmo-3-32b-think": ("OLMo-3-32B-Think", "Allen AI", 32, "‡"),
    "olmo-3.1-32b": ("OLMo-3.1-Instruct", "Allen AI", 32, ""),
    "qwen3.6-35b-a3b": ("Qwen3.6-35B-A3B", "Alibaba", 35, "†"),
    "qwen3.6-27b": ("Qwen3.6-27B", "Alibaba", 27, "†"),
    "qwen3.5-9b": ("Qwen3.5-9B", "Alibaba", 9, "†"),
    "glm-4.7-flash": ("GLM-4.7-Flash", "Zhipu/Z.ai", 30, "†"),
    "glm-4-9b-0414": ("GLM-4-9B-0414", "Zhipu/Z.ai", 9, ""),
    "kimi-linear-48b-a3b": ("Kimi-Linear-48B", "Moonshot AI", 48, ""),
    "phi-4": ("Phi-4", "Microsoft", 14, ""),
    "gemma-4-e4b": ("Gemma-4-E4B", "Google", 4, ""),
    "gemma-2-9b": ("Gemma-2-9B", "Google", 9, ""),
    "mistral-small-3.2": ("Mistral-Small-3.2", "Mistral", 24, ""),
    "mistral-7b-v0.3": ("Mistral-7B-v0.3", "Mistral", 7, ""),
    "deepseek-coder-v2-lite": ("DeepSeek-Coder-V2-Lite", "DeepSeek", 16, ""),
    "llama-3.2-3b": ("Llama-3.2-3B", "Meta", 3, ""),
    "yi-1.5-9b": ("Yi-1.5-9B", "01.AI", 9, ""),
}
STRUCTS = ["baseline", "ratio", "precedence", "negation", "multi_trigger_disjunction"]
SCHEMES = {"BALANCED": (BALANCED, BALANCED_SEVERITY), "SAFETY_FIRST": (SAFETY_FIRST, SAFETY_FIRST_SEVERITY),
           "CAPITAL_ADEQUACY": (CAPITAL_ADEQUACY, CAPITAL_ADEQUACY_SEVERITY)}


def norm(m):
    m = m.lower().split("/")[-1]
    for s in ("-mlx-4bit", "-mlx-6bit", "-mlx", "-gguf", "-instruct", "-chat", "-bq4km", "-it"):
        m = m.replace(s, "")
    return m.replace("meta-llama-", "llama-").strip("-")


def meta(k):
    if k in META: return META[k]
    for mk in sorted(META, key=len, reverse=True):
        if mk in k or k in mk: return META[mk]
    return (k, "?", 0, "")


def rep(bd): return DeviationReport(**{c: bd.get(c, 0) for c in CTOR})


def load():
    best = {}
    for f in glob.glob("outputs/row_full_*.json"):
        d = json.load(open(f)); r = d["rows"][0]; k = norm(r["model"])
        mt = os.path.getmtime(f)
        if k not in best or mt > best[k][0]:
            best[k] = (mt, r, d.get("seconds"))
    return best


def row_of(r):
    return AlignmentRow(model=r["model"], report=rep(r["report"]),
                        by_band={b: rep(bd) for b, bd in r["by_band"].items()})


def pearson(xs, ys):
    n = len(xs); mx, my = sum(xs)/n, sum(ys)/n
    cov = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
    sx = sum((x-mx)**2 for x in xs)**.5; sy = sum((y-my)**2 for y in ys)**.5
    return cov/(sx*sy) if sx and sy else 0.0


data = load()
ranked = {k: (r, sec) for k, (mt, r, sec) in data.items() if (r["parse_rate"] or 0) >= 0.85}
print(f"# Full-run analysis — {len(ranked)} ranked models (of {len(data)} run)\n")

# 1) CAS under three schemes + ranking robustness
print("## 1. CAS under three schemes (does the ranking flip?)\n")
rows = {}
for k, (r, sec) in ranked.items():
    ar = row_of(r)
    rows[k] = {s: score_matrix_cas(ar, scheme=w, severity=sv).cas for s, (w, sv) in SCHEMES.items()}
order_bal = sorted(ranked, key=lambda k: -rows[k]["BALANCED"])
print(f"{'model':24}{'BAL':>7}{'SAFE':>7}{'CAP':>7}  rank shift (BAL->SAFE)")
rank_safe = {k: i for i, k in enumerate(sorted(ranked, key=lambda k: -rows[k]['SAFETY_FIRST']))}
for i, k in enumerate(order_bal):
    d = meta(k)[0]; shift = i - rank_safe[k]
    print(f"{d:24}{rows[k]['BALANCED']:7.3f}{rows[k]['SAFETY_FIRST']:7.3f}{rows[k]['CAPITAL_ADEQUACY']:7.3f}  {shift:+d}")

# 2) signed safety direction
print("\n## 2. Signed safety direction (fail-open danger vs over-caution)\n")
print(f"{'model':24}{'signed':>8}{'FO_restr':>9}{'over-cau':>9}  tilt")
for k in order_bal:
    r = ranked[k][0]
    tilt = "DANGER (fail-open)" if r["signed_bias"] > 0.03 else ("paranoid (over-cautious)" if r["signed_bias"] < -0.03 else "balanced")
    print(f"{meta(k)[0]:24}{r['signed_bias']:+8.2f}{r['fail_open_restrictive']:9.1%}{r['over_cautious_rate']:9.1%}  {tilt}")

# 3) reasoning taxonomy effect
print("\n## 3. Does reasoning help alignment? (taxonomy group means)\n")
grp = defaultdict(list)
for k in ranked:
    grp[meta(k)[3] or "plain"].append(rows[k]["BALANCED"])
for g in ("‡", "†", "plain"):
    v = grp.get(g, [])
    if v: print(f"  {g:6} n={len(v)}  mean CAS={sum(v)/len(v):.3f}  range {min(v):.3f}-{max(v):.3f}")
# natural experiment: Olmo Think vs Instruct
if "olmo-3-32b-think" in rows and "olmo-3.1-32b" in rows:
    print(f"  OLMo Think {rows['olmo-3-32b-think']['BALANCED']:.3f} vs Instruct {rows['olmo-3.1-32b']['BALANCED']:.3f} "
          f"(Δ {rows['olmo-3-32b-think']['BALANCED']-rows['olmo-3.1-32b']['BALANCED']:+.3f})")

# 4) size & speed correlations
print("\n## 4. Size and speed vs CAS\n")
ks = list(ranked)
cas = [rows[k]["BALANCED"] for k in ks]
size = [meta(k)[2] for k in ks]
spi = [ranked[k][1]/1350 for k in ks]
print(f"  Pearson(params, CAS)   = {pearson(size, cas):+.2f}")
print(f"  Pearson(s/item, CAS)   = {pearson(spi, cas):+.2f}   (speed-alignment tradeoff)")
print("  fastest vs slowest by s/item:")
for k in sorted(ks, key=lambda k: ranked[k][1])[:3] + sorted(ks, key=lambda k: -ranked[k][1])[:3]:
    print(f"    {meta(k)[0]:24}{ranked[k][1]/1350:6.1f}s/item  CAS {rows[k]['BALANCED']:.3f}")

# 5) lab standings
print("\n## 5. Lab standings (mean CAS)\n")
labg = defaultdict(list)
for k in ranked: labg[meta(k)[1]].append(rows[k]["BALANCED"])
for lab, v in sorted(labg.items(), key=lambda kv: -sum(kv[1])/len(kv[1])):
    print(f"  {lab:14} n={len(v)}  mean {sum(v)/len(v):.3f}  best {max(v):.3f}")

# 6) per-structure & action difficulty (mean accuracy across models)
print("\n## 6. Which decisions are hardest (mean accuracy across ranked models)\n")
for s in STRUCTS:
    accs = [ranked[k][0]["by_structure"].get(s, {}).get("accuracy") for k in ranked]
    accs = [a for a in accs if a is not None]
    print(f"  {s:26} mean acc {sum(accs)/len(accs):.1%}")
print("  --- by action count ---")
for c in ("2", "3", "4"):
    accs = [ranked[k][0]["by_action_count"].get(c, {}).get("accuracy") for k in ranked]
    accs = [a for a in accs if a is not None]
    print(f"  {c}-verb                     mean acc {sum(accs)/len(accs):.1%}")

# 7) confusion structure from per-item answers (aggregate over all ranked models)
print("\n## 7. Confusion structure (oracle -> model, aggregate over ranked models)\n")
# need oracle per item; load corpus
from ambertrace_rlvr import load_decision_corpus
items = {it.id: it for it in load_decision_corpus(REPO/"data"/"decision_eval_v1.jsonl")}
conf = Counter(); band_err = Counter()
for k in ranked:
    for a in ranked[k][0]["answers"]:
        it = items.get(a["id"])
        if not it or not a["parse_ok"] or a["value"] is None: continue
        oracle = it.oracle
        if oracle and a["value"] != oracle:
            spec = it.spec()
            # restrictiveness rank to see direction
            conf[(oracle, a["value"])] += 1
print("  top 10 (oracle -> chosen) errors:")
for (o, c), n in conf.most_common(10):
    print(f"    {o:20} -> {c:20} {n}")

# 8) two-build Qwen3.6-27B comparison
print("\n## 8. Build effect — Qwen3.6-27B (two builds)\n")
for f in glob.glob("outputs/row_full_*qwen3.6-27b*.json"):
    d = json.load(open(f)); r = d["rows"][0]
    print(f"  {r['model']:22} CAS {score_matrix_cas(row_of(r)).cas:.3f}  acc {r['accuracy']:.1%}  FO_restr {r['fail_open_restrictive']:.1%}")

# 9) per-model reasoning-type strengths/weaknesses (relative to own mean & field)
print("\n## 9. Reasoning-type strengths & weaknesses (acc by structure; Δ vs own mean)\n")
field = {}
for s in STRUCTS:
    a = [ranked[k][0]["by_structure"].get(s, {}).get("accuracy") for k in ranked]
    a = [x for x in a if x is not None]; field[s] = sum(a)/len(a)
print("  field mean acc per structure: " + "  ".join(f"{s[:9]}={field[s]:.0%}" for s in STRUCTS))
print()
hdr = f"{'model':22}" + "".join(f"{s[:5]:>7}" for s in STRUCTS) + "   strongest / weakest (vs own mean)"
print(hdr)
for k in sorted(ranked, key=lambda k: -rows[k]["BALANCED"]):
    bs = ranked[k][0]["by_structure"]
    accs = {s: bs.get(s, {}).get("accuracy") for s in STRUCTS}
    valid = {s: a for s, a in accs.items() if a is not None}
    mean = sum(valid.values())/len(valid)
    strong = max(valid, key=lambda s: valid[s]-field[s]-(0))   # highest vs field
    # relative to own mean, to find intra-model spread
    hi = max(valid, key=lambda s: valid[s]); lo = min(valid, key=lambda s: valid[s])
    cells = "".join((f"{accs[s]:6.0%} " if accs[s] is not None else "   -- ") for s in STRUCTS)
    print(f"{meta(k)[0]:22}{cells}  +{hi[:12]} / -{lo[:12]} (spread {valid[hi]-valid[lo]:.0%})")

# 10) action-count decay per model (graceful vs cliff)
print("\n## 10. Degradation from 2->4 verbs (who holds up as choices multiply)\n")
print(f"{'model':22}{'2v':>6}{'3v':>6}{'4v':>6}   drop 2->4")
for k in sorted(ranked, key=lambda k: (ranked[k][0]['by_action_count'].get('4',{}).get('accuracy') or 0)-(ranked[k][0]['by_action_count'].get('2',{}).get('accuracy') or 0)):
    bac = ranked[k][0]["by_action_count"]
    a2 = bac.get("2",{}).get("accuracy"); a4 = bac.get("4",{}).get("accuracy"); a3=bac.get("3",{}).get("accuracy")
    drop = (a2-a4) if (a2 is not None and a4 is not None) else None
    print(f"{meta(k)[0]:22}{(a2 or 0):6.0%}{(a3 or 0):6.0%}{(a4 or 0):6.0%}   {('-%.0f%%'%(drop*100)) if drop is not None else 'n/a'}")
