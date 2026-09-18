"""The HF Articles builder (#109): every research doc transforms into a
self-contained, paste-ready article with no relative link surviving onto
huggingface.co, correct front-matter, and the "reproduce this" CTA."""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
BUILD = REPO / "examples" / "build_hf_articles.py"


def _load():
    spec = importlib.util.spec_from_file_location("build_hf_articles", BUILD)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def built():
    mod = _load()
    articles, warnings = mod.build_all(REPO / "dist" / "hf" / "articles", write=False)
    return mod, articles, warnings


def test_every_source_doc_builds(built):
    mod, articles, _ = built
    assert len(articles) == len(mod.ARTICLES)
    for a in articles:
        assert a["title"] and a["body"]


def test_body_starts_with_h1_title(built):
    """HF uses the leading H1 as the article title, so it must lead the body."""
    _, articles, _ = built
    for a in articles:
        assert a["body"].lstrip().startswith("# "), a["slug"]
        assert a["subtitle"]  # cover subtitle was extracted


def test_no_relative_link_survives(built):
    """An article on huggingface.co has no repo context — nothing relative may leak."""
    _, articles, _ = built
    for a in articles:
        leaked = re.findall(r"!?\[[^\]]*\]\((\.\.?/[^)]+|docs/[^)]+|[a-z0-9-]+\.md)\)", a["body"])
        assert not leaked, f"{a['slug']}: leaked {leaked}"


def test_images_use_pinned_cdn(built):
    """Figures must resolve absolutely (pinned jsDelivr), not GitHub-relative."""
    mod, articles, _ = built
    for a in articles:
        for img in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", a["body"]):
            assert img.startswith(mod.CDN + "@"), f"{a['slug']}: non-CDN image {img}"


def test_reflow_unwraps_and_preserves_blocks(built):
    """Paragraphs/list items/blockquotes collapse to one line; code/tables/headings
    stay intact."""
    mod, _, _ = built
    src = ("# Title\n\nA paragraph that the source\nhard-wrapped over\nthree lines.\n\n"
           "> a quote wrapped\n> across two lines\n\n"
           "- an item wrapped\n  onto a second line\n- second item\n\n"
           "| a | b |\n|---|---|\n\n```\ncode  stays\n  verbatim\n```\n")
    out = mod._reflow(src)
    assert "A paragraph that the source hard-wrapped over three lines." in out
    assert "> a quote wrapped across two lines" in out
    assert "- an item wrapped onto a second line" in out
    assert "- second item" in out
    assert "| a | b |" in out and "|---|---|" in out
    assert "code  stays\n  verbatim" in out          # code fence untouched
    assert out.startswith("# Title")


def test_cta_has_repo_roadmap_and_hf_surface(built):
    _, articles, _ = built
    for a in articles:
        assert "## Reproduce this" in a["body"]
        assert "github.com/ambertrace-labs/ambertrace-rlvr" in a["body"]
        assert "ROADMAP.md" in a["body"]
        assert "huggingface.co/" in a["body"].split("## Reproduce this")[1]


def test_no_claudisms(built):
    _, _, warnings = built
    assert warnings == [], f"claudisms flagged: {warnings}"
