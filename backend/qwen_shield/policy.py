"""Deterministic authorization. The security model's assessment is one input; permission comes only from the
application's grants and the user's request. Every decision names the rule that produced it."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from qwen_shield.provenance import Ledger
from qwen_shield.tools import BULK_EXPORT, CREDENTIALS, EXTERNAL_COMM, READ, TOOLS, WRITE_INTERNAL

Decision = Literal["allow", "block", "approve"]

ALWAYS = frozenset({READ, WRITE_INTERNAL})       # any task may do these
NEVER = frozenset({CREDENTIALS})                  # no task may do these
SENSITIVE = frozenset({EXTERNAL_COMM, BULK_EXPORT, CREDENTIALS})


@dataclass(frozen=True)
class Verdict:
    decision: Decision
    rule: str
    reason: str


def authorize(tool: str, args: dict, ledger: Ledger, granted: frozenset[str], assessment: dict | None,
              *, rules: bool = True, use_assessment: bool = True) -> Verdict:
    """`granted`: capabilities the user's request authorizes, set by the application, never by content.
    `assessment`: the validated assessment of untrusted content seen in this task (None if unavailable).
    `rules` / `use_assessment` switch the two layers off for the benchmark's ablations."""
    if tool not in TOOLS:
        return Verdict("block", "unknown_tool", f"{tool} is not a registered tool")
    cap = TOOLS[tool].capability
    if rules:
        if cap in NEVER:
            return Verdict("block", "never_grantable", f"{tool} needs {cap}, which no task may use")
        if cap not in ALWAYS and cap not in granted:
            return Verdict("block", "not_granted", f"the user's request did not authorize {cap}")
        for a in TOOLS[tool].sensitive_args:              # the target of a sensitive action must come from the user
            origin = ledger.origin_of(str(args.get(a, "")))
            if origin != "trusted":
                where = "untrusted content" if origin == "untrusted" else "nowhere the user said"
                return Verdict("block", "tainted_argument", f"{a}={args.get(a)!r} comes from {where}")
        if cap in SENSITIVE and use_assessment and assessment is None:
            return Verdict("block", "fail_closed", "no valid security assessment for a sensitive action")
    # Rules cannot see intent inside a permitted channel (an authorized email that also carries the customer list, an
    # internal ticket that asks for a password reset): there the assessment of the content decides.
    if use_assessment and cap != READ and assessment and assessment.get("classification") == "malicious" \
            and assessment.get("target_capability") in (cap, "any"):
        return Verdict("block", "assessed_malicious",
                       f"content was assessed as {assessment.get('attack_type')} targeting {cap}")
    return Verdict("allow", "granted", f"{cap} is permitted for this request")
