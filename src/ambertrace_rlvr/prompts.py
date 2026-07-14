"""System-prompt templates and the format contract the parser enforces.

The policy is prompted to emit a reasoning trace followed by a machine-readable
``<decision>`` block. Keeping the contract here (rather than scattered in examples)
means the parser and the prompt stay in sync.
"""

from __future__ import annotations

DECISION_OPEN = "<decision>"
DECISION_CLOSE = "</decision>"

SYSTEM_PROMPT_TEMPLATE = """\
You are solving a task in the "{domain}" domain.

Think step by step inside a <reasoning> ... </reasoning> block, then output your
final answer as a single JSON object inside a <decision> ... </decision> block.

The JSON object MUST contain:
  - "{answer_key}": your final answer
  - "{facts_key}": an object of the structured facts your answer rests on

Only assert facts supported by the input. Emit exactly one <decision> block.

Example:
<reasoning>
... your chain of thought ...
</reasoning>
<decision>
{{"{answer_key}": "<answer>", "{facts_key}": {{"<name>": <value>}}}}
</decision>
"""


def build_system_prompt(domain: str, answer_key: str = "classification",
                        facts_key: str = "facts") -> str:
    """Render the system prompt for a domain + the parser's key names."""
    return SYSTEM_PROMPT_TEMPLATE.format(
        domain=domain, answer_key=answer_key, facts_key=facts_key,
    )


def has_decision_block(completion: str) -> bool:
    """Cheap well-formedness check for a format reward (the parser is authoritative)."""
    return DECISION_OPEN in completion and DECISION_CLOSE in completion
