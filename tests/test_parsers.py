"""Parser behaviour on well-formed, malformed, and adversarial completions."""

from __future__ import annotations

from ambertrace_rlvr.parsers import JSONBlockParser, RegexBlockParser

JSON_COMPLETION = """
<reasoning>the applicant is a strong candidate</reasoning>
<decision>
{"classification": "permit", "facts": {"credit_score": 818, "loan_type": "unsecured"}}
</decision>
"""


def test_json_parser_happy_path():
    p = JSONBlockParser(answer_key="classification", facts_key="facts",
                        query_template="Assess: {facts}")
    parsed = p.parse("prompt", JSON_COMPLETION)
    assert parsed is not None
    assert parsed.proposed_answer == "permit"
    assert parsed.facts["credit_score"] == 818
    assert "credit_score" in parsed.query


def test_json_parser_returns_none_on_malformed_block():
    p = JSONBlockParser()
    assert p.parse("prompt", "<decision>{not valid json</decision>") is None


def test_json_parser_returns_none_when_no_block():
    p = JSONBlockParser()
    assert p.parse("prompt", "I think the answer is permit.") is None


def test_json_parser_returns_none_when_facts_missing():
    p = JSONBlockParser(facts_key="facts")
    assert p.parse("prompt", '<decision>{"classification": "permit"}</decision>') is None


def test_json_parser_ignores_non_object_json():
    p = JSONBlockParser()
    assert p.parse("prompt", "<decision>[1, 2, 3]</decision>") is None


def test_regex_parser_coerces_types():
    completion = "<decision>\nanswer: permit\nfact credit_score: 818\nfact ok: true\n</decision>"
    parsed = RegexBlockParser().parse("prompt", completion)
    assert parsed is not None
    assert parsed.proposed_answer == "permit"
    assert parsed.facts["credit_score"] == 818
    assert parsed.facts["ok"] is True


def test_both_parsers_populate_prompt():
    prompt = "the applicant income is 25000"
    json_parsed = JSONBlockParser().parse(prompt, JSON_COMPLETION)
    assert json_parsed is not None
    assert json_parsed.prompt == prompt
    regex_completion = "<decision>\nanswer: permit\nfact x: 1\n</decision>"
    regex_parsed = RegexBlockParser().parse(prompt, regex_completion)
    assert regex_parsed is not None
    assert regex_parsed.prompt == prompt
