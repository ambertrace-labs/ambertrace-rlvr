"""Verb-family -> decision-table definitions shared by the authoring scripts.

Pure data: each family names its verbs, its ratio verb, and the platform decision
table (ladder semantics) used when building a verified platform.
"""

from __future__ import annotations

from typing import Any

# Ordered most-permissive first; higher ``order`` = more restrictive (ladder semantics).
# ``default: True`` marks the RESIDUAL verb (what is decided when no rule fires) and MUST match
# the description's "Otherwise decide <verb>" (render_description) and the v1 policy residual.
# For these policies the residual is the PERMIT verb (approve/clear/discharge) — so default sits
# on order-1, NOT on the most-restrictive verb. (A fail-closed safety posture would be a separate
# flag; overloading ``default`` onto the restrictive verb makes table and description disagree,
# which the verified-build faithfulness gate correctly rejects.) ``default`` does not feed hack
# scoring (vocabulary_from_decision_table reads only ``permits``/``order``).
FAMILIES: dict[str, dict[str, Any]] = {
    "approve_deny": {
        "verbs": ["approve", "deny"], "ratio_verb": "deny",
        "table": {"mode": "ladder", "rows": [
            {"label": "approve", "order": 1, "permits": True, "default": True},
            {"label": "deny", "order": 2, "permits": False, "default": False}]},
    },
    "clear_flag": {
        "verbs": ["clear", "flag"], "ratio_verb": "flag",
        "table": {"mode": "ladder", "rows": [
            {"label": "clear", "order": 1, "permits": True, "default": True},
            {"label": "flag", "order": 2, "permits": False, "default": False}]},
    },
    "approve_restrict_suspend": {
        "verbs": ["approve", "restrict", "suspend"], "ratio_verb": "restrict",
        "table": {"mode": "ladder", "rows": [
            {"label": "approve", "order": 1, "permits": True, "default": True},
            {"label": "restrict", "order": 2, "permits": False, "default": False},
            {"label": "suspend", "order": 3, "permits": False, "default": False}]},
    },
    "discharge_monitor_escalate": {
        "verbs": ["discharge", "monitor", "escalate", "critical_escalate"],
        "ratio_verb": "escalate",
        "table": {"mode": "ladder", "rows": [
            {"label": "discharge", "order": 1, "permits": True, "default": True},
            {"label": "monitor", "order": 2, "permits": False, "default": False},
            {"label": "escalate", "order": 3, "permits": False, "default": False},
            {"label": "critical_escalate", "order": 4, "permits": False, "default": False}]},
    },
}
