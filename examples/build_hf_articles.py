"""Build paste-ready Hugging Face Blog Articles from the research corpus (#109).

Cross-posting the `docs/research/` corpus as HF community Articles reaches the ML
community a repo README never will. HF Articles have **no publish API** (web-UI only,
`huggingface.co/new-blog`) and org-namespace publishing needs a Team/Enterprise plan,
so this script does the reproducible part: it transforms each source doc into a
self-contained, paste-ready article and leaves the (manual) publish to a human.

Per source doc it:
  * lifts the H1 into front-matter `title` and drops it from the body (HF renders
    the title itself);
  * rewrites every repo-relative image to an absolute **jsDelivr** URL pinned to a
    commit SHA (GitHub-raw SVGs render, but a pinned CDN URL never rots), and every
    repo-relative link to an absolute GitHub `blob/<sha>` URL — an article on
    huggingface.co has no repo context, so nothing relative may survive;
  * appends the house **"Reproduce this"** CTA linking the repo, the roadmap, and
    the HF dataset/model/Space that piece is about (issue #109 criterion);
  * fails loud if any relative link survives or the CTA lost its repo/roadmap links,
    and warns on claudisms.

Writes to ``dist/hf/articles/<slug>.md`` (git-ignored). See ``docs/hf-articles/
PUBLISHING.md`` for the manual publish playbook.

Run:  ``python examples/build_hf_articles.py``          (writes dist/hf/articles/)
      ``python examples/build_hf_articles.py --check``  (dry-run: transform + assert)
"""
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DOCS = REPO / "docs"
DEFAULT_OUT = REPO / "dist" / "hf" / "articles"

GH = "https://github.com/ambertrace-labs/ambertrace-rlvr"
CDN = "https://cdn.jsdelivr.net/gh/ambertrace-labs/ambertrace-rlvr"
REPO_URL = GH
PYPI_URL = "https://pypi.org/project/ambertrace-rlvr/"

# HF surfaces this corpus links to (all live under the org).
HF = {
    "space": "https://huggingface.co/spaces/AmberTraceLabs/certified-alignment-matrix",
    "matrix_ds": "https://huggingface.co/datasets/AmberTraceLabs/alignment-matrix-results",
    "decision_ds": "https://huggingface.co/datasets/AmberTraceLabs/decision-eval",
    "airtrack_ds": "https://huggingface.co/datasets/AmberTraceLabs/air-track-triage",
    "faith_model": "https://huggingface.co/AmberTraceLabs/olmo-3-7b-air-track-faithfulness",
}

# Phrases the house voice avoids ([[research-writing-voice]]). Warn, don't fail.
CLAUDISMS = ["load-bearing", "it's worth noting", "it's important to note",
             "delve", "dive into", "in today's fast-paced", "at the end of the day",
             "when it comes to", "a testament to", "underscore", "leverage"]

ARTICLES = [
    {
        "src": "research/why-verifiable-rewards.md",
        "slug": "verifiable-rewards-beyond-maths-and-code",
        "thumbnail": "faithfulness_mechanism.svg",
        "cta": [
            ("🏆 Certified alignment leaderboard (Space)", HF["space"]),
            ("🧩 Faithfulness adapter (Model)", HF["faith_model"]),
            ("📚 Eval corpus (Dataset)", HF["decision_ds"]),
        ],
    },
    {
        "src": "research/alignment-matrix.md",
        "slug": "direction-of-error-open-weight-decision-models",
        "thumbnail": "alignment_cas_1350.svg",
        "cta": [
            ("🏆 Live leaderboard (Space)", HF["space"]),
            ("📊 Results table (Dataset)", HF["matrix_ds"]),
            ("📚 The 1,350-item eval corpus (Dataset)", HF["decision_ds"]),
        ],
    },
    {
        "src": "research/quantisation-safety-direction.md",
        "slug": "quantisation-and-the-safety-direction-of-decisions",
        "thumbnail": "quant_precision_scatter.svg",
        "cta": [
            ("📚 The eval corpus (Dataset)", HF["decision_ds"]),
            ("🏆 How full-precision models rank (Space)", HF["space"]),
        ],
    },
    {
        "src": "research/faithfulness-under-rlvr.md",
        "slug": "faithfulness-of-stated-reasoning-under-rlvr",
        "thumbnail": "faithfulness_overview.svg",
        "cta": [
            ("🧩 The trained adapter + checkpoints (Model)", HF["faith_model"]),
            ("📚 The air-track triage domain (Dataset)", HF["airtrack_ds"]),
        ],
    },
]


def _sha() -> str:
    return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"]).decode().strip()


def _rewrite_links(body: str, sha: str) -> str:
    """Make every repo-relative markdown link/image absolute (nothing relative may
    survive onto huggingface.co)."""
    def repl(m: re.Match) -> str:
        alt, target = m.group(1), m.group(2).strip()
        if target.startswith(("http://", "https://", "#", "mailto:")):
            return m.group(0)
        # images: ../assets/x.svg -> pinned jsDelivr CDN (renders + immutable)
        am = re.search(r"assets/([^)]+)$", target)
        if am and m.group(0).startswith("!"):
            return f"![{alt}]({CDN}@{sha}/docs/assets/{am.group(1)})"
        # doc/code links -> absolute GitHub blob at the pinned SHA
        path = target
        if path.startswith("../../"):
            path = path[len("../../"):]                      # repo root
        elif path.startswith("../"):
            path = "docs/" + path[len("../"):]               # docs/ root
        else:
            path = "docs/research/" + path                   # sibling research doc
        return f"[{alt}]({GH}/blob/{sha}/{path})"

    return re.sub(r"!?\[([^\]]*)\]\(([^)]+)\)", repl, body)


def _cta(article: dict) -> str:
    lines = ["", "---", "", "## Reproduce this", "",
             "Everything here is open, and every number links to the capture behind "
             "it. The repo ships the methodology, acceptance gates, drift/OOD probes, "
             "and reproduction recipes; the roadmap lists what is specified and "
             "waiting on compute.", ""]
    for label, url in article["cta"]:
        lines.append(f"- {label} — {url}")
    lines += [
        "",
        f"- 🔬 **Repo** — {REPO_URL} · **PyPI** — `pip install ambertrace-rlvr` "
        f"({PYPI_URL})",
        f"- 🗺️ **Roadmap** — {GH}/blob/main/ROADMAP.md",
        "",
        "*Ambertrace Labs. Researched and drafted by Ambertrace's AI systems under "
        "the editorial oversight of Peter Chatwell, Founder/CEO.*",
        "",
    ]
    return "\n".join(lines)


def build_one(article: dict, sha: str) -> dict:
    src = (DOCS / article["src"]).read_text()
    lines = src.splitlines()
    # Lift the H1 into the title; drop it from the body.
    title = next((l[2:].strip() for l in lines if l.startswith("# ")), article["slug"])
    body = "\n".join(l for l in lines if not l.startswith("# ")).lstrip("\n")
    body = _rewrite_links(body, sha)
    front = (f"---\ntitle: \"{title}\"\n"
             f"thumbnail: {CDN}@{sha}/docs/assets/{article['thumbnail']}\n"
             f"authors:\n  - user: AmberTraceLabs\n---\n\n")
    md = front + body + _cta(article)
    return {"slug": article["slug"], "title": title, "md": md}


def _validate(built: dict) -> list[str]:
    """Return warnings; raise SystemExit on hard failures (leaked relative links,
    missing CTA anchors)."""
    md = built["md"]
    body = md.split("---", 2)[-1]                            # skip front-matter
    leaked = re.findall(r"!?\[[^\]]*\]\((\.\.?/[^)]+|docs/[^)]+|[a-z0-9-]+\.md)\)", body)
    if leaked:
        raise SystemExit(f"{built['slug']}: relative link(s) survived: {leaked[:5]}")
    for must in (REPO_URL, "ROADMAP.md"):
        if must not in md:
            raise SystemExit(f"{built['slug']}: CTA missing required link {must!r}")
    low = md.lower()
    return [f"{built['slug']}: claudism {c!r}" for c in CLAUDISMS if c in low]


def build_all(out_dir: Path, *, write: bool) -> tuple[list[dict], list[str]]:
    sha = _sha()
    built, warnings = [], []
    for article in ARTICLES:
        b = build_one(article, sha)
        warnings += _validate(b)
        built.append(b)
        if write:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"{b['slug']}.md").write_text(b["md"])
    return built, warnings


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="dry-run: transform + assert, write nothing")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output dir (default: dist/hf/articles)")
    args = ap.parse_args()

    built, warnings = build_all(args.out, write=not args.check)
    for b in built:
        print(f"  {b['slug']:52} ({len(b['md'].splitlines())} lines)  {b['title']}")
    for w in warnings:
        print(f"  ⚠ {w}")
    print(f"\n{len(built)} articles. Publish is manual (huggingface.co/new-blog) — "
          "see docs/hf-articles/PUBLISHING.md.")
    if args.check:
        print("Dry-run — validated, nothing written.")
    else:
        print(f"Wrote {args.out}/")


if __name__ == "__main__":
    main()
