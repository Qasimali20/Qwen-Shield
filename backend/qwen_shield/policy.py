"""Deterministic authorization. The security model's assessment is one input; permission comes only from the
application's grants and the user's request. Every decision names the rule that produced it."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from qwen_shield.provenance import Ledger
from qwen_shield.tools import BULK_EXPORT, CREDENTIALS, EXTERNAL_COMM, READ, TOOLS, WRITE_INTERNAL

Decision = Literal["allow", "block", "approve"]

ALWAYS = frozenset({READ, WRITE_INTERNAL})       # any task may do these
NEVER = frozenset({CREDENTIALS})                  # no task may do these
SENSITIVE = frozenset({EXTERNAL_COMM, BULK_EXPORT, CREDENTIALS})


_INDICATOR = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"            # email addresses
                        r"|(?:https?|s?ftp|s3)://[^\s'\"<>)]+"           # URLs
                        r"|\b[\w-]+(?:\.[\w-]+)*\.[a-z]{2,}\b"          # domains and dotted names (faisal.admin)
                        r"|\b[a-z]+[-_][\w-]*\d[\w-]*\b", re.I)          # host-like names (prod-db-01)
_WORD = re.compile(r"\w+")


def _grams(text: str, n: int = 4) -> set[tuple[str, ...]]:
    w = _WORD.findall(text.lower())
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def carries_attack(args: dict, assessment: dict) -> str | None:
    """What part of the attack, if any, an action's arguments carry: an indicator from the evidence (address, URL,
    domain, account number) or a 4-word phrase of it. The document the agent read has already been sanitized, so a
    clean action means the user's work goes through while the attack does not."""
    text = " ".join(str(v) for v in args.values()).lower()
    if not text.strip():
        return None
    digits = re.sub(r"\D", "", text)
    for ev in assessment.get("evidence", []):
        for ind in _INDICATOR.findall(ev):
            if len(ind) >= 6 and ind.lower().rstrip(".,;") in text:
                return ind
        for run in re.findall(r"\d[\d\s-]{6,}\d", ev):            # account and phone numbers, spacing-insensitive
            if len(d := re.sub(r"\D", "", run)) >= 8 and d in digits:
                return run
        if shared := _grams(ev) & _grams(text):
            return " ".join(next(iter(shared)))
    return None


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
        if cap != READ and use_assessment and assessment is None:     # any write or send built from unchecked content
            return Verdict("block", "fail_closed", "content read in this task has no valid security assessment")
    # Rules cannot see intent inside a permitted channel (an authorized email that also carries a phishing link, an
    # internal ticket that asks for admin rights): there the assessment decides. The injected text was removed from
    # what the agent read, so the action is blocked when it still carries the attack, and goes through when clean.
    if use_assessment and cap != READ and assessment and assessment.get("classification") == "malicious" \
            and assessment.get("target_capability") in (cap, "any"):
        if cap in NEVER:
            return Verdict("block", "assessed_malicious", f"content tried to obtain {cap}")
        if hit := carries_attack(args, assessment):
            return Verdict("block", "assessed_malicious",
                           f"the {tool} call carries injected content ({hit!r}) from a {assessment.get('attack_type')}")
        return Verdict("allow", "sanitized_clean", f"{cap} is permitted and the call carries none of the injected content")
    return Verdict("allow", "granted", f"{cap} is permitted for this request")
