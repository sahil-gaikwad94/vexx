"""Resumable Python lexer state for incremental in-flight masking.

Tier-1 masking must decide, for every vocabulary entry, whether the token is
syntactically legal after the current prefix.  Re-lexing the whole prefix for all
``|V|`` candidates costs ``O(|V| * L)`` character steps per decode step, which at
``L = 4096`` is over 10^8 Python loop iterations per token.  The state below scans
only the appended characters, so each candidate costs ``O(len(token))``.

:func:`advance_lexer_state` is chunk-boundary invariant: feeding a complete string at
once yields the same final state as feeding it one character at a time.  That property
is what makes it safe to clone a state per vocabulary candidate and discard the clone.

**Difference from the Maya reference implementation.**  Maya rejected *any* quote
inside an active f-string expression (``quote_in_fstring_expression``).  That rejects
valid Python such as ``f"{format_value('x')}"``, which is ordinary nested-string code
and is explicitly allowed by PEP 701 on the Python versions this package targets.  This
implementation tracks f-string expressions with a real substate stack, so nested quotes
and nested replacement fields are accepted while genuinely malformed input (unterminated
expression, unbalanced brace, stray closer) still fails closed.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "LexerState",
    "advance_lexer_state",
    "CALLEE_KEYWORDS",
    "SAFE_STANDARD_METHODS",
    "FORBIDDEN_IDENTIFIERS",
    "iter_lexer_tokens",
]

#: Words that can precede ``(`` without being a callable of interest.
CALLEE_KEYWORDS = frozenset(
    {
        "if", "while", "for", "return", "elif", "assert", "not", "and", "or",
        "in", "def", "class", "lambda", "with", "yield", "del", "await", "from",
        "raise", "import", "except", "finally", "match", "case", "global",
        "print",
    }
)

#: Identifiers that must never be called or referenced by generated code.
FORBIDDEN_IDENTIFIERS = frozenset(
    {
        "os", "sys", "subprocess", "socket", "shutil", "pathlib", "open",
        "eval", "exec", "compile", "__import__", "globals", "locals",
        "getattr", "setattr", "delattr", "vars", "input", "breakpoint",
    }
)

#: String methods that are safe on a *literal* receiver.  ``format`` / ``format_map``
#: stay out: a format string can walk attribute paths into a rendering of the object
#: graph.
SAFE_LITERAL_ATTRIBUTES = frozenset(
    {
        "capitalize", "casefold", "center", "count", "endswith",
        "expandtabs", "find", "index", "isalnum", "isalpha", "isascii",
        "isdecimal", "isdigit", "isidentifier", "islower", "isnumeric",
        "isprintable", "isspace", "istitle", "isupper", "join", "ljust",
        "lower", "lstrip", "partition", "removeprefix", "removesuffix",
        "replace", "rfind", "rindex", "rjust", "rpartition", "rsplit",
        "rstrip", "split", "splitlines", "startswith", "strip",
        "swapcase", "title", "upper", "zfill",
    }
)

#: Standard container / string methods that are safe on a local variable, a subscript
#: or an attribute chain.  Without this, Tier-1 masking rejected every method call whose
#: receiver was not literally ``self`` / ``cls``, so Python generation could not emit
#: ``stack.append(x)``, ``adj[u].append(v)`` or ``d.get(k, 0)`` at all.
SAFE_STANDARD_METHODS = frozenset(
    {
        "append", "appendleft", "add", "clear", "copy", "count",
        "difference", "difference_update", "discard", "extend", "extendleft",
        "fromkeys", "get", "index", "insert", "intersection",
        "intersection_update", "issubset", "issuperset", "items", "keys",
        "pop", "popleft", "popitem", "remove", "reverse", "rotate",
        "setdefault", "sort", "symmetric_difference",
        "symmetric_difference_update", "union", "update", "values",
    }
) | SAFE_LITERAL_ATTRIBUTES

_BRACKET_PAIRS = {("(", ")"), ("[", "]"), ("{", "}")}
_OPEN_TO_CLOSE = {"(": ")", "[": "]", "{": "}"}
_STRING_START_RE = re.compile(r"^([rRbBuUfF]{0,3})('''|\"\"\"|'|\")")
#: A trailing run that might still grow into a longer string opener/closer.  A doubled
#: quote is held so that ``""`` + ``"`` can form a triple-quoted string across chunks;
#: the decision is deferred to the next chunk (or to EOF).
_PARTIAL_STRING_RE = re.compile(r"[rRbBuUfF]{0,3}(?:'''|\"\"\"|''|\"\"|'|\")\Z")
_STRING_PREFIX_RE = re.compile(r"[rRbBuUfF]{1,3}")
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _is_fstring(in_str: str | None, f_mode: bool) -> bool:
    return bool(f_mode) and in_str not in (None, "#")


@dataclass
class LexerState:
    """Resumable Python lexer state.

    ``stack`` holds open delimiters as ``(delimiter, callee, comma_count)`` where
    ``callee`` is the dotted name immediately before an opening ``(``.
    ``closed_frames`` reports only the frames closed while scanning the most recent
    chunk, which is exactly what a token-closing candidate needs.

    ``fstring_stack`` tracks nested replacement fields inside an f-string.  Each entry
    is the delimiter stack (``[...]``) that was open when the field started, so a ``}``
    that closes a replacement field restores the correct outer context instead of being
    mistaken for a dict close.
    """

    valid: bool = True
    reason: str | None = None
    in_str: str | None = None
    raw_str: bool = False
    f_mode: bool = False
    #: Stack of suspended f-string bodies: each entry is ``(delimiter_stack, quote, raw, f_mode)``.
    fstring_stack: list[tuple[list[tuple[str, str | None, int]], str, bool, bool]] = field(default_factory=list)
    #: Outer string modes suspended while lexing a nested string inside an f-expression.
    nested_string_stack: list[tuple[str | None, bool, bool]] = field(default_factory=list)
    #: Quote that would end the string literal currently being scanned.
    stack: list[tuple[str, str | None, int]] = field(default_factory=list)
    closed_frames: list[tuple[str, str | None, int]] = field(default_factory=list)
    last_ident: str = ""
    prev_non_ws: str = ""
    last_line: str = ""
    #: Text carried over from a chunk that ended mid string-opener.
    pending: str = ""
    string_open_at: int = -1
    #: Source accumulated by this lexer state; used for deterministic recovery/replay.
    source: str = ""

    # ------------------------------------------------------------- helpers

    @property
    def in_fstring_expression(self) -> bool:
        """True while scanning the replacement-field expression of an f-string."""
        return bool(self.fstring_stack)

    @property
    def open_depth(self) -> int:
        return len(self.stack)

    @property
    def f_brace_depth(self) -> int:
        return len(self.fstring_stack)

    def clone(self) -> LexerState:
        """Copy the state, dropping per-chunk ``closed_frames``."""
        return LexerState(
            valid=self.valid,
            reason=self.reason,
            in_str=self.in_str,
            raw_str=self.raw_str,
            f_mode=self.f_mode,
            fstring_stack=[(list(fs), q, r, fm) for (fs, q, r, fm) in self.fstring_stack],
            nested_string_stack=list(self.nested_string_stack),
            stack=list(self.stack),
            closed_frames=[],
            last_ident=self.last_ident,
            prev_non_ws=self.prev_non_ws,
            last_line=self.last_line,
            pending=self.pending,
            string_open_at=self.string_open_at,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serializable view used by telemetry, debugging and native parity tests."""
        return {
            "valid": self.valid,
            "reason": self.reason,
            "stack": list(self.stack),
            "closed_frames": list(self.closed_frames),
            "in_str": self.in_str,
            "raw_str": self.raw_str,
            "f_mode": self.f_mode,
            "fstring_depth": len(self.fstring_stack),
            "f_brace_depth": len(self.fstring_stack),
            "last_ident": self.last_ident,
            "prev_non_ws": self.prev_non_ws,
            "pending": self.pending,
        }


def _close_frame(st: LexerState, ch: str) -> bool:
    """Consume a closing delimiter.  Returns False (and marks invalid) on mismatch."""
    if not st.stack:
        st.valid = False
        st.reason = "unmatched_close"
        return False
    open_ch, callee, count = st.stack[-1]
    if _OPEN_TO_CLOSE.get(open_ch) != ch:
        st.valid = False
        st.reason = "mismatched_close"
        return False
    st.closed_frames.append(st.stack.pop())
    st.last_ident = ""
    st.prev_non_ws = ch
    st.last_line = (st.last_line + ch)[-256:]
    return True


def _open_frame(st: LexerState, ch: str) -> None:
    callee: str | None = None
    if ch == "(" and st.last_ident and st.last_ident not in CALLEE_KEYWORDS:
        callee = st.last_ident
    st.stack.append((ch, callee, 0))
    st.last_ident = ""
    st.prev_non_ws = ch
    st.last_line = (st.last_line + ch)[-256:]


def _scan_chars(st: LexerState, text: str, limit: int) -> int:
    """Scan ``text[:limit]``; returns the number of characters consumed."""
    n = min(len(text), limit)
    i = 0
    while i < n and st.valid:
        ch = text[i]

        # ---------------------------------------------------------- comment
        if st.in_str == "#":
            if ch == "\n":
                st.in_str = None
                st.prev_non_ws = "\n"
                st.last_line = ""
            i += 1
            continue

        # ------------------------------------------------------ f-string body
        if st.in_str is not None and _is_fstring(st.in_str, st.f_mode) and not st.fstring_stack:
            if ch == "{":
                if i + 1 < n and text[i + 1] == "{":
                    i += 2
                    continue
                # Enter a replacement field: suspend the f-string body and scan
                # an ordinary expression.  The suspended quote is restored on close.
                st.fstring_stack.append((list(st.stack), st.in_str, st.raw_str, st.f_mode))
                st.stack = []
                st.in_str = None
                st.f_mode = False
                st.raw_str = False
                st.last_ident = ""
                i += 1
                continue
            if ch == "}":
                if i + 1 < n and text[i + 1] == "}":
                    i += 2
                    continue
                st.valid = False
                st.reason = "unmatched_close"
                break
            if ch == "\\" and not st.raw_str and i + 1 < n:
                i += 2
                continue
            if text.startswith(st.in_str, i):
                i += len(st.in_str)
                st.in_str = None
                st.f_mode = False
                st.raw_str = False
                continue
            i += 1
            continue

        # ------------------------------------------- f-string expression body
        if st.fstring_stack:
            if st.in_str is not None:
                # Nested string literal inside the replacement field (PEP 701).
                if ch == "\\" and i + 1 < n:
                    i += 2
                    continue
                if text.startswith(st.in_str, i):
                    i += len(st.in_str)
                    if st.nested_string_stack:
                        st.in_str, st.raw_str, st.f_mode = st.nested_string_stack.pop()
                    else:
                        st.in_str = None
                        st.f_mode = False
                        st.raw_str = False
                    continue
                i += 1
                continue
            if ch == "}" and not st.stack:
                # Only a ``}`` at the field's own nesting level closes the field.
                fs, quote, raw, fm = st.fstring_stack.pop()
                st.stack = fs
                st.in_str = quote
                st.raw_str = raw
                st.f_mode = fm
                st.prev_non_ws = "}"
                i += 1
                continue
            # A nested string literal inside the expression is legal (PEP 701).
            match = _STRING_START_RE.match(text[i:])
            if match is not None:
                st.nested_string_stack.append((st.in_str, st.raw_str, st.f_mode))
                _enter_string(st, match, i)
                i += len(match.group(0))
                continue
            if ch in "([{":
                _open_frame(st, ch)
                i += 1
                continue
            if ch in ")]":
                if not _close_frame(st, ch):
                    break
                i += 1
                continue
            if ch == "}" and st.stack:
                # A brace that does not match the innermost open delimiter is invalid.
                if not _close_frame(st, ch):
                    break
                i += 1
                continue
            if ch == "#":
                st.in_str = "#"
                i += 1
                continue
            if ch == "," and st.stack:
                delim, call, count = st.stack[-1]
                st.stack[-1] = (delim, call, count + 1)
                st.last_ident = ""
                i += 1
                continue
            if ch.isalnum() or ch in "_.":
                st.last_ident = (st.last_ident + ch)[-160:]
            elif not ch.isspace():
                st.last_ident = ""
            if ch == "\n":
                st.last_line = ""
            else:
                st.last_line = (st.last_line + ch)[-256:]
            if not ch.isspace():
                st.prev_non_ws = ch
            i += 1
            continue

        # ------------------------------------------------- ordinary string
        if st.in_str is not None:
            if ch == "\n" and not st.raw_str and isinstance(st.in_str, str) and len(st.in_str) == 1:
                st.valid = False
                st.reason = "unterminated_string_literal"
                break
            if ch == "\\" and i + 1 < n:
                i += 2
                continue
            if text.startswith(st.in_str, i):
                i += len(st.in_str)
                if st.nested_string_stack:
                    st.in_str, st.raw_str, st.f_mode = st.nested_string_stack.pop()
                else:
                    st.in_str = None
                    st.f_mode = False
                    st.raw_str = False
                continue
            i += 1
            continue

        # ------------------------------------------------------- code body
        if ch == "#":
            st.in_str = "#"
            i += 1
            continue

        match = _STRING_START_RE.match(text[i:])
        if match is not None:
            _enter_string(st, match, i)
            i += len(match.group(0))
            continue

        if ch in "([{":
            _open_frame(st, ch)
            i += 1
            continue

        if ch in ")]}":
            if not _close_frame(st, ch):
                break
            i += 1
            continue

        if ch == "," and st.stack:
            delim, call, count = st.stack[-1]
            st.stack[-1] = (delim, call, count + 1)
            st.last_ident = ""
            i += 1
            continue

        if ch.isalnum() or ch in "_.":
            st.last_ident = (st.last_ident + ch)[-160:]
        elif not ch.isspace():
            st.last_ident = ""

        if ch == "\n":
            st.last_line = ""
        else:
            st.last_line = (st.last_line + ch)[-256:]
        if not ch.isspace():
            st.prev_non_ws = ch
        i += 1

    return i


def _enter_string(st: LexerState, match: re.Match[str], position: int) -> None:
    flags = match.group(1).lower()
    st.in_str = match.group(2)
    st.f_mode = "f" in flags
    st.raw_str = "r" in flags
    st.string_open_at = position
    # The prefix (``f``, ``rb``, ...) was just consumed as part of the opener, so it
    # must not linger in ``last_ident`` and be mistaken for a callable name.
    st.last_ident = ""


def _split_string_prefix(st: LexerState) -> str:
    """Move a dangling string prefix out of ``last_ident`` and into ``pending``."""
    match = _STRING_PREFIX_RE.fullmatch(st.last_ident)
    if match is None:
        return ""
    prefix = match.group(0)
    st.last_ident = ""
    return prefix


def advance_lexer_state(state: LexerState, chunk: str, *, reset_closed: bool = True) -> LexerState:
    """Advance ``state`` over ``chunk`` in ``O(len(chunk))`` character steps.

    Only a genuinely incomplete trailing string opener (``f"`` / ``""`` / ``'''``
    split across chunk boundaries) is carried into ``state.pending`` and re-scanned with
    the next chunk, so token boundaries cannot silently change how a quote is read.
    """
    st = state.clone()
    if reset_closed:
        st.closed_frames = []
    if not st.valid or not chunk:
        return st

    text = (st.pending + chunk) if st.pending else chunk
    # A dangling string prefix from the previous chunk (``f`` with the quote still to
    # come) is part of the text being decided, so it must survive into the rewind state.
    carry_prefix = _split_string_prefix(st)
    st.pending = ""

    # A trailing quote run may still grow into a triple-quoted string, so the scan stops
    # before it and the decision is deferred to the next chunk.  Only the *last line* can
    # hold an undecided opener: a quote on an earlier line belongs to a string that has
    # already been terminated, and treating it as pending would corrupt the state.
    hold = -1
    hold_state: LexerState | None = None
    line_start = text.rfind("\n") + 1
    partial = _PARTIAL_STRING_RE.search(text, line_start)
    if partial is not None:
        run = partial.group(0)
        quote = run[-1]
        if len(run) - len(run.rstrip(quote)) < 3:
            hold = partial.start()
            # Rewind to just before the opener: the entry state when the opener sits at
            # the start of the chunk, otherwise the state after scanning the head.
            head = state.clone()
            head.pending = ""
            head.closed_frames = []
            if hold > 0:
                _scan_chars(head, text, hold)
            hold_state = head

    # Scan only up to the undecided tail, so the tail is never consumed by accident.
    limit = hold if hold >= 0 else len(text)
    consumed = _scan_chars(st, text, limit)
    if hold >= 0 and consumed >= hold:
        # Carry the undecided tail (and any dangling string prefix) into ``pending``.
        st = hold_state if hold_state is not None else st
        st.closed_frames = []
        if st.valid:
            st.pending = carry_prefix + text[hold:]
    elif carry_prefix and st.valid:
        # The prefix did not turn into a string opener after all; keep it as pending so
        # the next chunk can still complete it.
        st.pending = carry_prefix
    return st


def iter_lexer_tokens(state: LexerState, chunk: str) -> Iterator[str]:
    """Yield the significant characters of ``chunk`` (used by diagnostics/tests)."""
    for char in chunk:
        if not char.isspace():
            yield char