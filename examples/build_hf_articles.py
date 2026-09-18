"""Build paste-ready Hugging Face Blog Articles from the research corpus (#109).

Cross-posting the `docs/research/` corpus as HF community Articles reaches the ML
community a repo README never will. HF Articles have **no publish API** (web-UI only,
`huggingface.co/new-blog`) and org-namespace publishing needs a Team/Enterprise plan,
so this script does the reproducible part: it transforms each source doc into a
self-contained, paste-ready article and leaves the (manual) publish to a human.

Per source doc it:
  * drops the H1 (HF's editor has a separate title field) and rewrites every
    repo-relative image to an absolute **jsDelivr** URL pinned to a commit SHA
    (GitHub-raw SVGs render, but a pinned CDN URL never rots), and every
    repo-relative link to an absolute GitHub `blob/<sha>` URL — an article on
    huggingface.co has no repo context, so nothing relative may survive;
  * appends the house **"Reproduce this"** CTA linking the repo, the roadmap, and
    the HF dataset/model/Space that piece is about (issue #109 criterion);
  * fails loud if any relative link survives or the CTA lost its repo/roadmap links,
    and warns on claudisms.

The `huggingface.co/new-blog` editor takes title / slug / thumbnail / authors as
**separate fields** and only the **body** in the markdown pane, so this writes:
  * ``dist/hf/articles/<slug>.md`` — the paste-ready **body** (no front-matter);
  * ``dist/hf/articles/thumbnails/<slug>.png`` — a branded 1200×648 cover to upload
    (the editor's thumbnail is an upload, not a URL); needs Pillow, skipped with a
    warning if absent;
  * ``dist/hf/articles/PUBLISH.md`` — a per-article sheet of the field values.

See ``docs/hf-articles/PUBLISHING.md`` for the manual publish playbook.

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
    """A durable, publicly reachable commit SHA to pin image URLs to.

    Prefers ``origin/main`` — a feature-branch HEAD can be garbage-collected after a
    squash-merge, which would break every jsDelivr image URL in a published article.
    ``main`` is never GC'd and already carries the assets. Falls back to ``HEAD``."""
    for ref in ("origin/main", "HEAD"):
        try:
            return subprocess.check_output(
                ["git", "-C", str(REPO), "rev-parse", ref],
                stderr=subprocess.DEVNULL).decode().strip()
        except subprocess.CalledProcessError:
            continue
    raise SystemExit("could not resolve a commit SHA to pin image URLs")


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
    # The H1 becomes the editor's title field; drop it from the body. The italic
    # line under it is the subtitle (used on the cover).
    title = next((l[2:].strip() for l in lines if l.startswith("# ")), article["slug"])
    body = "\n".join(l for l in lines if not l.startswith("# ")).lstrip("\n")
    subtitle = next((l.strip("* ").strip() for l in body.splitlines()
                     if l.startswith("*") and l.rstrip().endswith("*")), "")
    body = _rewrite_links(body, sha) + _cta(article)
    # A front-matter block is kept only as a reference view for tests / humans;
    # the body written to disk is front-matter-free (HF takes fields separately).
    front = (f"---\ntitle: \"{title}\"\n"
             f"thumbnail: thumbnails/{article['slug']}.png\n"
             f"authors:\n  - user: AmberTraceLabs\n---\n\n")
    return {"slug": article["slug"], "title": title, "subtitle": subtitle,
            "body": body, "md": front + body}


# --- Branded 1200x648 cover (the editor's thumbnail is an upload, not a URL) ---
_COVER = (1200, 648)
_PALETTE = {"paper": "#F7F6F3", "line": "#E7E4DC", "ink": "#1B1A17",
            "muted": "#7A776E", "amber": "#E0982E"}
_FONTS = [  # macOS system faces, first that loads wins
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial.ttf",
]
_FONTS_BOLD = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]


def _font(size: int, *, bold: bool):
    from PIL import ImageFont
    for path in (_FONTS_BOLD if bold else _FONTS):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _wrap(draw, text: str, font, max_w: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if draw.textlength(trial, font=font) <= max_w:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def _make_cover(title: str, subtitle: str, out_path: Path) -> None:
    from PIL import Image, ImageDraw
    W, H = _COVER
    img = Image.new("RGB", (W, H), _PALETTE["paper"])
    d = ImageDraw.Draw(img)
    d.rectangle([12, 12, W - 13, H - 13], outline=_PALETTE["line"], width=2)
    m = 80
    d.text((m, 92), "A M B E R T R A C E   ·   R E S E A R C H",
           font=_font(24, bold=True), fill=_PALETTE["amber"])
    y = 150
    for line in _wrap(d, title, _font(66, bold=True), W - 2 * m):
        d.text((m, y), line, font=_font(66, bold=True), fill=_PALETTE["ink"])
        y += 78
    if subtitle:
        y += 14
        for line in _wrap(d, subtitle, _font(30, bold=False), W - 2 * m)[:3]:
            d.text((m, y), line, font=_font(30, bold=False), fill=_PALETTE["muted"])
            y += 42
    d.text((m, H - 78), "ambertrace-rlvr  ·  verifiable rewards for rule-governed decisions",
           font=_font(24, bold=False), fill=_PALETTE["muted"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


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


def _publish_sheet(built: list[dict]) -> str:
    rows = ["# Publish sheet — paste-ready HF Articles (#109)", "",
            "For each row: at `huggingface.co/new-blog` set **Owner** = `AmberTraceLabs`, "
            "the **Title**, **Slug**, upload the **thumbnail** PNG, then paste the "
            "**body** file into the markdown pane (no front-matter to skip — the file "
            "is body-only). Authors: your HF handle.", "",
            "| Title | Slug | Thumbnail (upload) | Body (paste) |",
            "|---|---|---|---|"]
    for b in built:
        rows.append(f"| {b['title']} | `{b['slug']}` | "
                    f"`thumbnails/{b['slug']}.png` | `{b['slug']}.md` |")
    return "\n".join(rows) + "\n"


def build_all(out_dir: Path, *, write: bool) -> tuple[list[dict], list[str]]:
    sha = _sha()
    built, warnings = [], []
    have_pil = True
    try:
        import PIL  # noqa: F401
    except ImportError:
        have_pil = False
    for article in ARTICLES:
        b = build_one(article, sha)
        warnings += _validate(b)
        built.append(b)
        if write:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"{b['slug']}.md").write_text(b["body"])  # body only
            if have_pil:
                _make_cover(b["title"], b["subtitle"], out_dir / "thumbnails" / f"{b['slug']}.png")
    if write:
        (out_dir / "PUBLISH.md").write_text(_publish_sheet(built))
        if not have_pil:
            warnings.append("Pillow not installed — thumbnails skipped (pip install pillow)")
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
