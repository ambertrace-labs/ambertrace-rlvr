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
        assert a["title"] and a["md"]


def test_no_relative_link_survives(built):
    """An article on huggingface.co has no repo context — nothing relative may leak."""
    _, articles, _ = built
    for a in articles:
        body = a["md"].split("---", 2)[-1]
        leaked = re.findall(r"!?\[[^\]]*\]\((\.\.?/[^)]+|docs/[^)]+|[a-z0-9-]+\.md)\)", body)
        assert not leaked, f"{a['slug']}: leaked {leaked}"


def test_images_use_pinned_cdn(built):
    """Figures must resolve absolutely (pinned jsDelivr), not GitHub-relative."""
    mod, articles, _ = built
    for a in articles:
        for img in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", a["md"]):
            assert img.startswith(mod.CDN + "@"), f"{a['slug']}: non-CDN image {img}"


def test_front_matter_and_title(built):
    _, articles, _ = built
    for a in articles:
        assert a["md"].startswith("---\ntitle: \"")
        assert "thumbnail: https://" in a["md"]
        # H1 was lifted out of the body
        assert "\n# " not in a["md"]


def test_cta_has_repo_roadmap_and_hf_surface(built):
    _, articles, _ = built
    for a in articles:
        assert "## Reproduce this" in a["md"]
        assert "github.com/ambertrace-labs/ambertrace-rlvr" in a["md"]
        assert "ROADMAP.md" in a["md"]
        assert "huggingface.co/" in a["md"].split("## Reproduce this")[1]


def test_no_claudisms(built):
    _, _, warnings = built
    assert warnings == [], f"claudisms flagged: {warnings}"
