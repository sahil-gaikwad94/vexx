"""CRANE protocol phases: reasoning, experiment cells, and strict verified output.

Vexx already ships :class:`~vexx.crane_mode.CraneModeController`, which handles
``<think>`` / ``</think>`` and ``<emit>`` / ``</emit>``.  This module *extends* that
behaviour rather than replacing it: it adds the ``<experiment>`` phase and, crucially,
incremental marker recognition.

Why incremental matters: markers are split across token boundaries.  A controller that
calls ``str.rfind()`` on the accumulated buffer cannot tell a malformed nesting from a
chunk split, and it cannot decide a marker that is only half-present.  The tracker here
buffers a bounded suffix and re-matches the combined suffix plus the next token, so the
verdict is independent of how the text was chunked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

__all__ = [
    "ProtocolConfig",
    "ProtocolPhase",
    "ProtocolError",
    "PhaseTransition",
    "ProtocolTracker",
]


class ProtocolPhase(str, Enum):
    """Where the model is in the CRANE protocol."""

    #: Outside any span.  Host-authorized output is expected here.
    STRICT_EMIT = "strict_emit"
    #: Inside ``<think>``: free-form reasoning, no code grammar applied.
    REASONING = "reasoning"
    #: Inside ``<experiment>``: executable, policy-checked Python.
    EXPERIMENT = "experiment"
    #: An explicit end-of-output marker was seen.
    COMPLETE = "complete"


class ProtocolError(ValueError):
    """A malformed protocol sequence (unmatched or nested markers)."""


@dataclass(frozen=True)
class ProtocolConfig:
    """Marker strings and the transitions the host authorizes.

    Defaults match both the Maya protocol constants and Vexx's existing CRANE markers,
    so a host can adopt the code engine without changing its tokenizer.
    """

    think_start: str = "<think>"
    think_end: str = "</think>"
    emit_start: str = "<emit>"
    emit_end: str = "</emit>"
    experiment_start: str = "<experiment>"
    experiment_end: str = "</experiment>"
    #: Optional explicit end-of-output marker.  Empty means "no explicit terminator".
    output_end: str = ""
    #: When false, a start marker seen without host authorization is a protocol error.
    require_authorized_transitions: bool = True

    def __post_init__(self) -> None:
        markers = {
            "think_start": self.think_start,
            "think_end": self.think_end,
            "emit_start": self.emit_start,
            "emit_end": self.emit_end,
            "experiment_start": self.experiment_start,
            "experiment_end": self.experiment_end,
        }
        for name, value in markers.items():
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if self.output_end and not isinstance(self.output_end, str):
            raise ValueError("output_end must be a string")
        # A marker that is a prefix of another would make matching ambiguous.
        values = [v for v in markers.values()] + ([self.output_end] if self.output_end else [])
        for i, first in enumerate(values):
            for second in values[i + 1 :]:
                if first.startswith(second) or second.startswith(first):
                    raise ValueError(f"protocol markers must not prefix each other: {first!r} / {second!r}")

    @property
    def markers(self) -> tuple[str, ...]:
        base = [
            self.think_start, self.think_end,
            self.emit_start, self.emit_end,
            self.experiment_start, self.experiment_end,
        ]
        if self.output_end:
            base.append(self.output_end)
        return tuple(base)

    @property
    def max_marker_len(self) -> int:
        return max(len(m) for m in self.markers)

    def marker_kind(self, marker: str) -> str:
        for name in (
            "think_start", "think_end", "emit_start", "emit_end",
            "experiment_start", "experiment_end",
        ):
            if getattr(self, name) == marker:
                return name
        if self.output_end and marker == self.output_end:
            return "output_end"
        return ""


@dataclass(frozen=True)
class PhaseTransition:
    """One authorized phase change, for the audit trail."""

    phase: ProtocolPhase
    marker: str
    kind: str
    offset: int

    def as_dict(self) -> dict:
        return {
            "phase": self.phase.value,
            "marker": self.marker,
            "kind": self.kind,
            "offset": self.offset,
        }


@dataclass
class ProtocolTracker:
    """Incremental CRANE phase tracker.

    Feed it tokens or characters in any chunking; the phase verdict depends only on the
    concatenated text.  ``feed`` returns the text that is *settled* (no longer a possible
    marker prefix) plus the transitions that occurred, so a caller can commit text to a
    cache without risking a half-emitted marker.
    """

    config: ProtocolConfig = field(default_factory=ProtocolConfig)
    phase: ProtocolPhase = ProtocolPhase.STRICT_EMIT
    #: Authorized start markers; a start marker outside this set is a protocol error
    #: when ``require_authorized_transitions`` is true.
    authorized_starts: frozenset[str] = field(default_factory=frozenset)
    transitions: list[PhaseTransition] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    #: Raw text seen so far, for audit and phase re-derivation.
    consumed: str = ""
    #: Text held back because it may still become a marker.
    _buffer: str = ""
    #: Currently open span kinds (``think_start`` / ``emit_start`` / ``experiment_start``).
    _open_spans: set[str] = field(default_factory=set)
    #: Reasoning text accumulated while inside ``<think>``.
    reasoning: str = ""
    #: Experiment-cell text accumulated while inside ``<experiment>``.
    experiment: str = ""

    def __post_init__(self) -> None:
        self.authorized_starts = frozenset(self.authorized_starts)

    # ------------------------------------------------------------- helpers

    @property
    def open_spans(self) -> frozenset[str]:
        """The currently open span kinds (empty when not inside any span)."""
        return frozenset(self._open_spans)

    @property
    def in_reasoning(self) -> bool:
        return self.phase is ProtocolPhase.REASONING

    @property
    def in_experiment(self) -> bool:
        return self.phase is ProtocolPhase.EXPERIMENT

    @property
    def strict(self) -> bool:
        """True when code grammar constraints apply."""
        return self.phase in (ProtocolPhase.STRICT_EMIT, ProtocolPhase.EXPERIMENT)

    @property
    def complete(self) -> bool:
        return self.phase is ProtocolPhase.COMPLETE

    def _held_len(self) -> int:
        """Characters at the end of the buffer that could still become a marker."""
        return self._held_len_from(self._buffer)

    def _apply_marker(self, marker: str, offset: int) -> None:
        kind = self.config.marker_kind(marker)
        cfg = self.config
        if kind == "output_end":
            self.phase = ProtocolPhase.COMPLETE
            self.transitions.append(PhaseTransition(ProtocolPhase.COMPLETE, marker, kind, offset))
            return
        if kind in ("think_start", "emit_start", "experiment_start"):
            if cfg.require_authorized_transitions and marker not in self.authorized_starts:
                self.errors.append(f"unauthorized transition: {marker!r}")
                return
            target = {
                "think_start": ProtocolPhase.REASONING,
                "emit_start": ProtocolPhase.STRICT_EMIT,
                "experiment_start": ProtocolPhase.EXPERIMENT,
            }[kind]
            # Track the open span explicitly.  Comparing phases is not enough: ``</think>``
            # and ``<emit>`` both lead to STRICT_EMIT, so a phase check cannot tell a
            # legitimate switch from a duplicate open.
            if kind in self._open_spans:
                self.errors.append(f"duplicate open marker: {marker!r}")
                return
            if self._open_spans:
                # A second span opened before the first one closed is nesting, which the
                # protocol forbids (``<experiment>`` cells must not nest).
                self.errors.append(
                    f"nested protocol span: {marker!r} inside {sorted(self._open_spans)}"
                )
                return
            self._open_spans.add(kind)
            self.phase = target
            self.transitions.append(PhaseTransition(target, marker, kind, offset))
            return
        # close markers
        expected = {
            "think_end": "think_start",
            "emit_end": "emit_start",
            "experiment_end": "experiment_start",
        }[kind]
        if expected not in self._open_spans:
            self.errors.append(f"close marker {marker!r} without matching open")
            return
        self._open_spans.discard(expected)
        self.phase = ProtocolPhase.STRICT_EMIT
        self.transitions.append(PhaseTransition(ProtocolPhase.STRICT_EMIT, marker, kind, offset))

    # ---------------------------------------------------------------- core

    def feed(self, text: str) -> tuple[str, list[PhaseTransition]]:
        """Consume ``text``; return ``(settled_text, transitions)``.

        ``settled_text`` excludes any trailing characters that could still become part of
        a marker, so it is always safe to commit to a token cache.
        """
        if not text:
            return "", []
        if self.phase is ProtocolPhase.COMPLETE:
            return "", []

        self.consumed += text
        buffer = self._buffer + text
        self._buffer = ""

        before = len(self.transitions)
        settled: list[str] = []
        pending = buffer

        while pending:
            hits: list[tuple[int, str]] = []
            for marker in self.config.markers:
                index = pending.find(marker)
                if index != -1:
                    hits.append((index, marker))
            if not hits:
                break
            index, marker = min(hits, key=lambda item: item[0])
            # The body preceding a marker belongs to the phase that was active *before*
            # the marker, so it must be recorded before the transition is applied.
            settled.append(pending[:index])
            self._record_body(pending[:index])
            self._apply_marker(marker, len(self.consumed) - len(pending) + index)
            pending = pending[index + len(marker) :]
            if self.phase is ProtocolPhase.COMPLETE:
                break

        if self.phase is not ProtocolPhase.COMPLETE:
            tail = pending
            hold = self._held_len_from(tail)
            if hold:
                settled.append(tail[: len(tail) - hold])
                self._record_body(tail[: len(tail) - hold])
                self._buffer = tail[len(tail) - hold :]
            else:
                settled.append(tail)
                self._record_body(tail)
        else:
            self._buffer = ""

        return "".join(settled), self.transitions[before:]

    def _held_len_from(self, text: str) -> int:
        """Length of the longest suffix of ``text`` that is a *proper* marker prefix.

        Only proper prefixes are held: a complete marker must never be deferred, and
        ``</think>`` must not be mistaken for a prefix of ``<think>`` (the prefix test
        is anchored at the end of the text, so the two differ at the first character).
        """
        longest = self.config.max_marker_len
        for size in range(min(longest, len(text)), 0, -1):
            tail = text[-size:]
            for marker in self.config.markers:
                if len(tail) < len(marker) and marker.startswith(tail):
                    return size
        return 0

    def _record_body(self, body: str) -> None:
        if not body:
            return
        if self.phase is ProtocolPhase.REASONING:
            self.reasoning += body
        elif self.phase is ProtocolPhase.EXPERIMENT:
            self.experiment += body

    # -------------------------------------------------------------- queries

    def flush(self) -> str:
        """Release any held suffix (end of generation)."""
        held, self._buffer = self._buffer, ""
        self._record_body(held)
        return held

    def reset(self) -> None:
        self.phase = ProtocolPhase.STRICT_EMIT
        self.transitions = []
        self.errors = []
        self.consumed = ""
        self.reasoning = ""
        self.experiment = ""
        self._buffer = ""
        self._open_spans = set()

    def phase_at(self, text: str) -> ProtocolPhase:
        """Phase a *complete* text would leave the tracker in (diagnostic helper).

        Every marker is authorized here: this answers "where does this text end up",
        not "was this text a legal host-driven transition".
        """
        probe = ProtocolTracker(self.config, authorized_starts=frozenset(self.config.markers))
        probe.feed(text)
        return probe.phase

    def as_dict(self) -> dict:
        return {
            "phase": self.phase.value,
            "authorized_starts": sorted(self.authorized_starts),
            "transitions": [t.as_dict() for t in self.transitions],
            "errors": list(self.errors),
            "buffered": self._buffer,
        }