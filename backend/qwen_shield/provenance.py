"""Provenance: where every piece of text the agent saw came from, and how far it is trusted.

Only the system configuration and the user's own request carry authority. Documents, web pages and tool outputs are
content: they may inform an answer, never authorize an action."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

TRUSTED = frozenset({"system", "user"})
UNTRUSTED = frozenset({"retrieved_document", "web_page", "tool_output", "agent"})
_WS = re.compile(r"\s+")


def norm(s: str) -> str:
    return _WS.sub(" ", str(s).strip().lower())


@dataclass(frozen=True)
class Segment:
    source: str
    text: str
    ref: str = ""

    @property
    def trusted(self) -> bool:
        return self.source in TRUSTED


@dataclass
class Ledger:
    """Everything that entered the agent's context, in order, with its source."""
    segments: list[Segment] = field(default_factory=list)

    def add(self, source: str, text: str, ref: str = "") -> Segment:
        if source not in TRUSTED | UNTRUSTED:
            raise ValueError(f"unknown source {source!r}")
        s = Segment(source, text, ref)
        self.segments.append(s)
        return s

    def origin_of(self, value: str) -> str:
        """'trusted' if the user or system stated the value, 'untrusted' if it appears only in content, 'novel' if it
        appears nowhere (the agent produced it)."""
        v = norm(value)
        if not v:
            return "trusted"
        if any(v in norm(s.text) for s in self.segments if s.trusted):
            return "trusted"
        if any(v in norm(s.text) for s in self.segments if not s.trusted):
            return "untrusted"
        return "novel"
