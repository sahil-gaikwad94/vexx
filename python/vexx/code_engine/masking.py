"""Tier-0 / Tier-1 in-flight logit masking for Vexx code generation.

This module is framework-neutral: it operates on a 1-D logits array plus a
vocabulary map and a :class:`CodeSession`.  Backend adapters (Hugging Face,
vLLM) wrap it.
"""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

try:
    import numpy as np
except ImportError:
    np = None

from .config import DEFAULT_ALLOWED_IMPORTS, CodeEngineConfig
from .lexer import LexerState, advance_lexer_state
from .protocol import ProtocolConfig, ProtocolPhase, ProtocolTracker


class DeadEnd(Exception):
    """Raised (or returned) when every candidate token is masked."""


@dataclass
class MaskResult:
    logits: np.ndarray
    masked_count: int
    total: int
    dead_end: bool
    warnings: list[str] = field(default_factory=list)


#: Characters that make a token's legality depend on the current lexer context.
_STRUCTURAL_CHARS = frozenset("\"'\\#()[]{}\n\r")


@dataclass(frozen=True)
class _VocabPlan:
    """Static, per-tokenizer classification of a vocabulary.

    Decode-time masking only has to inspect :attr:`dynamic` tokens (typically a few percent
    of a BPE vocabulary: quote, escape, delimiter, comment and forbidden-name tokens).
    Everything else is legal in every lexer context and is left untouched.
    """

    drop_ids: np.ndarray
    keep_ids: np.ndarray
    dynamic: tuple[tuple[int, str], ...]
    marker_candidates: tuple[tuple[int, str], ...]


class _VocabCache:
    """Thread-safe, bounded, immutable cache of static vocabulary classification."""

    def __init__(self, max_entries: int = 32) -> None:
        self._lock = threading.RLock()
        self._max = max_entries
        self._store: dict[str, _VocabPlan] = {}
        #: Identity-keyed plans.  The entry keeps a strong reference to the mapping, so its
        #: ``id`` cannot be recycled while the plan is cached.
        self._by_identity: dict[int, tuple[Mapping[int, str], int, str | None, _VocabPlan]] = {}

    def for_vocab(
        self, vocab_map: Mapping[int, str], tokenizer_hash: str | None = None
    ) -> _VocabPlan:
        """Return the plan for ``vocab_map`` without re-hashing 128k tokens per step.

        The cache is keyed by object identity (plus size and the optional tokenizer hash), and
        the cached tuple holds the mapping itself, which both pins the object and lets us
        confirm it is still the same vocabulary.
        """
        identity = id(vocab_map)
        with self._lock:
            entry = self._by_identity.get(identity)
            if (
                entry is not None
                and entry[0] is vocab_map
                and entry[1] == len(vocab_map)
                and entry[2] == tokenizer_hash
            ):
                return entry[3]
        plan = self.get_or_build(self.key_for(vocab_map, tokenizer_hash), vocab_map)
        with self._lock:
            if len(self._by_identity) >= self._max:
                self._by_identity.pop(next(iter(self._by_identity)), None)
            self._by_identity[identity] = (vocab_map, len(vocab_map), tokenizer_hash, plan)
        return plan

    def key_for(self, vocab_map: Mapping[int, str], tokenizer_hash: str | None) -> str:
        if tokenizer_hash:
            # Caller asserts the hash covers vocabulary contents; we still fold in size.
            return f"hash:{tokenizer_hash}:n={len(vocab_map)}"
        # Collision-resistant digest over sorted (id, text) pairs.
        h = hashlib.sha256()
        for tid in sorted(vocab_map):
            h.update(str(int(tid)).encode())
            h.update(b"\0")
            h.update(str(vocab_map[tid]).encode("utf-8", "replace"))
            h.update(b"\n")
        return f"digest:{h.hexdigest()}"

    def get_or_build(self, key: str, vocab_map: Mapping[int, str]) -> _VocabPlan:
        with self._lock:
            cached = self._store.get(key)
            if cached is not None:
                return cached
            drop: list[int] = []
            keep: list[int] = []
            dynamic: list[tuple[int, str]] = []
            markers: list[tuple[int, str]] = []
            for tid, tok in vocab_map.items():
                if isinstance(tid, bool) or not isinstance(tid, (int, np.integer)):
                    raise ValueError(f"vocabulary key {tid!r} is not an integer id")
                tid_int = int(tid)
                if tid_int < 0:
                    raise ValueError(f"negative vocabulary id {tid_int}")
                text = str(tok)
                if text in {"", "\x00"}:
                    drop.append(tid_int)
                    continue
                if "<" in text or ">" in text:
                    markers.append((tid_int, text))
                if _is_static_safe(text):
                    keep.append(tid_int)
                else:
                    dynamic.append((tid_int, text))
            result = _VocabPlan(
                np.asarray(sorted(set(drop)), dtype=np.int64),
                np.asarray(sorted(set(keep)), dtype=np.int64),
                tuple(dynamic),
                tuple(markers),
            )
            if len(self._store) >= self._max:
                # Evict oldest (insertion-ordered dict).
                self._store.pop(next(iter(self._store)))
            self._store[key] = result
            return result

    def invalidate(self) -> None:
        with self._lock:
            self._store.clear()
            self._by_identity.clear()


_VOCAB_CACHE = _VocabCache()

#: ``(mapping, size, validated_max_id)`` per identity; keeps the mapping alive so the id is
#: stable and lets us notice a resized vocabulary.
_VALIDATED: dict[int, tuple[Mapping[int, str], int, int]] = {}
_VALIDATED_MAX = 16


def validate_vocab(
    vocab_map: Mapping[int, str], vocab_size: int
) -> None:
    """Validate that every id is an int in ``[0, vocab_size)``.

    A 128K-token vocabulary is walked once, not once per decode step: the identity-keyed memo
    below remembers the highest id already proven valid.
    """
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    size = len(vocab_map)
    memo = _VALIDATED.get(id(vocab_map))
    if (
        memo is not None
        and memo[0] is vocab_map
        and memo[1] == size
        and 0 <= memo[2] < vocab_size
    ):
        return
    for tid in vocab_map:
        if isinstance(tid, bool) or not isinstance(tid, (int, np.integer)):
            raise ValueError(f"vocabulary key {tid!r} is not an integer")
        tid_int = int(tid)
        if tid_int < 0 or tid_int >= vocab_size:
            raise ValueError(
                f"vocabulary id {tid_int} out of range [0, {vocab_size})"
            )
    # Detect duplicate decoded text? Not required; allow sparse ids.
    if len(_VALIDATED) >= _VALIDATED_MAX:
        _VALIDATED.pop(next(iter(_VALIDATED)), None)
    _VALIDATED[id(vocab_map)] = (vocab_map, size, max((int(k) for k in vocab_map), default=-1))


class CodeSession:
    """Per-generation mutable state: lexer, protocol, symbols, bounds."""

    def __init__(
        self,
        config: CodeEngineConfig | None = None,
        protocol_config: ProtocolConfig | None = None,
        symbol_table: dict[str, Any] | None = None,
        domain_bounds: dict[str, Any] | None = None,
        **_ignored: Any,
    ) -> None:
        if config is None:
            from .config import CodeEngineConfig as _CEC
            config = _CEC()
        self.config = config
        self.protocol_config = protocol_config or ProtocolConfig()
        self.protocol = ProtocolTracker(self.protocol_config, authorized_starts=frozenset())
        self.lexer_state = LexerState()
        self.symbols: dict[str, Any] = dict(symbol_table or {})
        self._accumulated_symbols: dict[str, Any] = dict(self.symbols)
        self._symbol_cache: dict[str, Any] = {}
        self._revision = 0
        if domain_bounds:
            self.initial_bounds = {k: (float(v[0]), float(v[1])) for k, v in domain_bounds.items()}
        else:
            self.initial_bounds = {}
        self.bounds: dict[str, tuple[float, float]] = dict(self.initial_bounds)
        self.checkpoints: list[dict[str, Any]] = []
        self.experiment_preamble: list[str] = []
        self._episode_id = 0
        self.warnings: list[str] = []
        self.dpo_buffer: list[dict[str, Any]] = []
        self.last_exact_fraction = None
        #: Incremental lexer prefix cache: ``(prefix, state)`` so a decode step lexes only
        #: the appended token instead of the whole document for every candidate.
        self._lex_prefix: str = ""
        self._lex_state: LexerState = LexerState()
        #: Incremental protocol phase cache, mirroring ``_lex_prefix``.
        self._phase_prefix: str = ""
        self._phase_tracker: ProtocolTracker | None = None
        self._phase: ProtocolPhase = ProtocolPhase.STRICT_EMIT

    # ------------------------------------------------------- incremental state

    def lexer_state_for(self, text: str) -> LexerState:
        """Return the lexer state after ``text``, advancing incrementally."""
        if text == self._lex_prefix:
            return self._lex_state
        if self._lex_prefix and text.startswith(self._lex_prefix):
            state = advance_lexer_state(self._lex_state, text[len(self._lex_prefix):])
        else:
            state = advance_lexer_state(LexerState(), text)
        self._lex_prefix = text
        self._lex_state = state
        return state

    def phase_for(self, text: str) -> ProtocolPhase:
        """Return the CRANE phase after ``text``, feeding only the appended suffix."""
        if text == self._phase_prefix:
            return self._phase
        tracker = self._phase_tracker
        if tracker is None or not self._phase_prefix or not text.startswith(self._phase_prefix):
            tracker = ProtocolTracker(
                self.protocol_config,
                authorized_starts=frozenset(self.protocol_config.markers),
            )
            delta = text
        else:
            delta = text[len(self._phase_prefix):]
        try:
            if delta:
                tracker.feed(delta)
            self._phase = tracker.phase
        except Exception:
            pass
        self._phase_tracker = tracker
        self._phase_prefix = text
        return self._phase

    def reset(self) -> None:
        self.protocol = ProtocolTracker(self.protocol_config, authorized_starts=frozenset())
        self.lexer_state = LexerState()
        self.symbols = {}
        self._accumulated_symbols = {}
        self._symbol_cache.clear()
        self._revision = 0
        self.bounds = dict(self.initial_bounds)
        self.checkpoints.clear()
        self.experiment_preamble.clear()
        self.warnings.clear()
        self._lex_prefix = ""
        self._lex_state = LexerState()
        self._phase_prefix = ""
        self._phase_tracker = None
        self._phase = ProtocolPhase.STRICT_EMIT
        self.new_episode()

    def new_episode(self) -> int:
        self._episode_id += 1
        self.experiment_preamble.clear()
        return self._episode_id

    @property
    def episode_id(self) -> int:
        return self._episode_id

    def commit_text(self, text: str) -> None:
        """Feed committed text once to the persistent lexer state."""
        self.lexer_state = advance_lexer_state(self.lexer_state, text)
        # Any committed text invalidates the per-step prefix caches.
        self._lex_prefix = ""
        self._phase_prefix = ""
        self._revision += 1
        # Update symbol table from a bounded tail window using a collision-resistant key.
        self._refresh_symbols()

    def _refresh_symbols(self) -> None:
        # We re-extract from a bounded tail; full definitions are preserved via accumulated.
        # Use the full active code if available; here we only have lexer state.
        # Symbol extraction is driven by masking.py which passes active_code.
        pass

    def extract_symbols(self, active_code: str) -> dict[str, Any]:
        """Return contracts for in-flight code, session-isolated, digest-keyed."""
        window = self.config.signature_scan_window
        tail = active_code[-window:] if len(active_code) > window else active_code
        key_src = f"{len(active_code)}\x00{tail}"
        key = hashlib.sha256(key_src.encode("utf-8", "replace")).hexdigest()
        cached = self._symbol_cache.get(key)
        if cached is not None:
            return cached
        from .contracts import build_contracts
        merged = dict(self._accumulated_symbols)
        # Try full parse first.
        try:
            grouped = build_contracts(active_code)
        except (SyntaxError, ValueError):
            grouped = None
        if grouped is not None:
            for contracts in grouped.values():
                for c in contracts:
                    name = c.symbol_id.rsplit("::", 1)[-1].rsplit(".", 1)[-1]
                    entry = c.to_symbol_table_entry()
                    entry["partial"] = c.confidence != "exact_ast"
                    existing = merged.get(name)
                    if existing is None:
                        merged[name] = entry
                    else:
                        # merge widest bounds, mark ambiguous
                        maxes = [existing.get("max_arity"), entry.get("max_arity")]
                        existing["min_arity"] = min(existing.get("min_arity", 0), entry.get("min_arity", 0))
                        existing["max_arity"] = None if any(m is None for m in maxes) else max(int(m) for m in maxes)
                        existing["has_var_keyword"] = bool(existing.get("has_var_keyword")) or bool(entry.get("has_var_keyword"))
                        existing["has_var_positional"] = bool(existing.get("has_var_positional")) or bool(entry.get("has_var_positional"))
                        existing["keyword_only"] = sorted(set(existing.get("keyword_only", [])) | set(entry.get("keyword_only", [])))
                        existing["required_keywords"] = []
                        existing["ambiguous"] = True
                        existing["partial"] = bool(existing.get("partial")) or bool(entry.get("partial"))
            self._accumulated_symbols = merged
        else:
            # Fall back to header scan on the tail.
            import re as _re
            for match in _re.finditer(r"(?:^|\n)[ \t]*(?:async[ \t]+)?def[ \t]+[A-Za-z_]\w*[ \t]*\(", tail):
                from .contracts import header_contract_from_text
                contract = header_contract_from_text(tail[match.start():], symbol_id="<inline>")
                if contract is not None and contract.confidence != "unknown":
                    name = contract.symbol_id.rsplit("::", 1)[-1].rsplit(".", 1)[-1]
                    merged[name] = contract.to_symbol_table_entry()
            self._accumulated_symbols = merged

        if len(self._symbol_cache) >= 8:
            self._symbol_cache.pop(next(iter(self._symbol_cache)))
        self._symbol_cache[key] = merged
        self.symbols = merged
        return merged


def _classify_callee(callee: str, live_syms: dict[str, Any], config: CodeEngineConfig) -> tuple[str | None, bool]:
    """Resolve a call target. Returns (key, is_safe_standard_method). None key => block."""
    if "." in callee:
        recv, meth = callee.rsplit(".", 1)
        if not _is_ident(meth):
            return None, False
        if meth.startswith("__") or meth in _FORBIDDEN:
            return None, False
        if recv in _ALLOWED_MODULE_ROOTS:
            # ``math.sqrt(``, ``heapq.heappush(`` ... the module itself is allow-listed by
            # the AST policy and the child runtime, so its members are callable.  Arity is
            # not modelled for the standard library, so the contract gate is skipped.
            return f"{recv}.{meth}", True
        if recv in {"self", "cls"}:
            if live_syms and meth not in live_syms:
                return None, False
            return meth, False
        if meth in live_syms:
            return meth, False
        if meth in _SAFE_STANDARD_METHODS:
            return meth, True
        return None, False
    if callee in _FORBIDDEN:
        return None, False
    return callee, False


def _is_ident(name: str) -> bool:
    return bool(name) and name.isidentifier()


_FORBIDDEN = frozenset({
    "os", "sys", "subprocess", "socket", "shutil", "pathlib", "open",
    "eval", "exec", "compile", "__import__", "globals", "locals",
    "getattr", "setattr", "delattr", "vars", "input", "breakpoint",
})

#: ``stripped.startswith(prefixes)`` replaces a per-token generator expression over every
#: forbidden name; masking a 4k-token structural subset was spending ~40% of its time there.
_FORBIDDEN_PREFIXES: tuple[str, ...] = tuple(f"{name}." for name in _FORBIDDEN)
_OPENING_CHARS = frozenset("([{")

#: Allow-listed standard-library roots (mirrors ``config.DEFAULT_ALLOWED_IMPORTS``).
_ALLOWED_MODULE_ROOTS = frozenset(DEFAULT_ALLOWED_IMPORTS)

#: Python builtins that are callable inside the sandbox.  Tier-1 masking must know these:
#: without them every ``range(``, ``len(``, ``sorted(`` and ``enumerate(`` call was masked
#: to ``-inf``, which made Python generation impossible.
SAFE_BUILTIN_CALLABLES: dict[str, dict[str, Any]] = {
    "abs": {"min_arity": 1, "max_arity": 1, "return_type": "any"},
    "all": {"min_arity": 1, "max_arity": 1, "return_type": "bool"},
    "any": {"min_arity": 1, "max_arity": 1, "return_type": "bool"},
    "bool": {"min_arity": 0, "max_arity": 1, "return_type": "bool"},
    "chr": {"min_arity": 1, "max_arity": 1, "return_type": "str"},
    "dict": {"min_arity": 0, "max_arity": None, "has_var_keyword": True, "return_type": "dict"},
    "divmod": {"min_arity": 2, "max_arity": 2, "return_type": "tuple"},
    "enumerate": {"min_arity": 1, "max_arity": 2, "return_type": "any"},
    "filter": {"min_arity": 2, "max_arity": 2, "return_type": "any"},
    "float": {"min_arity": 0, "max_arity": 1, "return_type": "float"},
    "int": {"min_arity": 0, "max_arity": 2, "return_type": "int"},
    "isinstance": {"min_arity": 2, "max_arity": 2, "return_type": "bool"},
    "len": {"min_arity": 1, "max_arity": 1, "return_type": "int"},
    "list": {"min_arity": 0, "max_arity": 1, "return_type": "list"},
    "map": {"min_arity": 2, "max_arity": None, "has_var_positional": True, "return_type": "any"},
    "max": {"min_arity": 1, "max_arity": None, "has_var_positional": True, "has_var_keyword": True, "return_type": "any"},
    "min": {"min_arity": 1, "max_arity": None, "has_var_positional": True, "has_var_keyword": True, "return_type": "any"},
    "next": {"min_arity": 1, "max_arity": 2, "return_type": "any"},
    "ord": {"min_arity": 1, "max_arity": 1, "return_type": "int"},
    "pow": {"min_arity": 2, "max_arity": 3, "return_type": "any"},
    "range": {"min_arity": 1, "max_arity": 3, "return_type": "any"},
    "reversed": {"min_arity": 1, "max_arity": 1, "return_type": "any"},
    "round": {"min_arity": 1, "max_arity": 2, "return_type": "any"},
    "set": {"min_arity": 0, "max_arity": 1, "return_type": "set"},
    "sorted": {"min_arity": 1, "max_arity": 1, "has_var_keyword": True, "return_type": "list"},
    "str": {"min_arity": 0, "max_arity": 3, "return_type": "str"},
    "sum": {"min_arity": 1, "max_arity": 2, "return_type": "any"},
    "tuple": {"min_arity": 0, "max_arity": 1, "return_type": "tuple"},
    "zip": {"min_arity": 0, "max_arity": None, "has_var_positional": True, "has_var_keyword": True, "return_type": "any"},
}


def _is_static_safe(text: str) -> bool:
    """True when a token is legal in *every* lexer context.

    Structural characters (quotes, escapes, comments, delimiters, newlines) and forbidden
    names are the only tokens whose legality depends on where the decode currently is.
    """
    stripped = text.strip()
    if not stripped:
        return True
    if any(ch in _STRUCTURAL_CHARS for ch in stripped):
        return False
    if stripped in _FORBIDDEN or stripped.startswith(_FORBIDDEN_PREFIXES):
        return False
    return True


_SAFE_STANDARD_METHODS = frozenset({
    "append", "appendleft", "add", "clear", "copy", "count", "difference",
    "difference_update", "discard", "extend", "extendleft", "fromkeys", "get",
    "index", "insert", "intersection", "intersection_update", "issubset",
    "issuperset", "items", "keys", "pop", "popleft", "popitem", "remove",
    "reverse", "rotate", "setdefault", "sort", "symmetric_difference",
    "symmetric_difference_update", "union", "update", "values",
    "capitalize", "casefold", "center", "endswith", "expandtabs",
    "find", "isalnum", "isalpha", "isascii", "isdecimal", "isdigit",
    "isidentifier", "islower", "isnumeric", "isprintable", "isspace", "istitle",
    "isupper", "join", "ljust", "lower", "lstrip", "partition", "removeprefix",
    "removesuffix", "replace", "rfind", "rindex", "rjust", "rpartition", "rsplit",
    "rstrip", "split", "splitlines", "startswith", "strip", "swapcase", "title",
    "upper", "zfill",
})


class VexxCodeController:
    def __init__(
        self,
        vocabulary: Mapping[int, str] | None = None,
        tokenizer: Any = None,
        config: CodeEngineConfig | None = None,
        protocol_config: ProtocolConfig | None = None,
        domain_bounds: dict[str, Any] | None = None,
        symbol_table: dict[str, Any] | None = None,
        **_ignored: Any,
    ) -> None:
        self.config = config or CodeEngineConfig()
        self.protocol_config = protocol_config or ProtocolConfig()
        self.vocabulary = dict(vocabulary) if vocabulary is not None else None
        self.tokenizer = tokenizer
        self._sessions: list[CodeSession] = []
        if domain_bounds:
            self.initial_bounds = {k: (float(v[0]), float(v[1])) for k, v in domain_bounds.items()}
        else:
            self.initial_bounds = {}
        self.bounds: dict[str, tuple[float, float]] = dict(self.initial_bounds)
        self.checkpoints: list[dict[str, Any]] = []
        self.experiment_preamble: list[str] = []
        self.symbol_table: dict[str, Any] = dict(symbol_table or {})
        self._symbol_cache: dict[Any, dict[str, dict[str, Any]]] = {}
        self._accumulated_symbols: dict[str, dict[str, Any]] = dict(self.symbol_table)
        self.last_exact_fraction = None
        self.dpo_buffer: list[dict[str, Any]] = []

    def new_session(
        self,
        prefix: str = "",
        mode: str = "python",
        authorized_starts: frozenset[str] | None = None,
        symbol_table: dict[str, Any] | None = None,
    ) -> CodeSession:
        session = CodeSession(self.config, self.protocol_config, symbol_table=symbol_table if symbol_table is not None else self.symbol_table)
        session.protocol = ProtocolTracker(
            self.protocol_config,
            authorized_starts=authorized_starts or frozenset(),
        )
        session.symbols = dict(session.symbols or {})
        session._accumulated_symbols = dict(session.symbols)
        if prefix:
            session.commit_text(prefix)
            session.extract_symbols(prefix)
        self._sessions.append(session)
        return session

    def _token_allowed(
        self,
        tok_str: str,
        base_state: LexerState,
        live_syms: dict[str, Any],
        phase: ProtocolPhase,
        remaining_budget: int,
        open_depth: int,
        expected_return_type: str | None,
    ) -> bool:
        """Whether one context-dependent token is legal after ``base_state``."""
        if remaining_budget <= open_depth + 1 and not _OPENING_CHARS.isdisjoint(tok_str):
            return False
        stripped = tok_str.strip()
        # A forbidden *name* is only forbidden in code; ``"open the door"`` is text.
        # A trailing quote run is held in ``pending`` while the lexer waits to see whether it
        # grows into a triple quote, so a half-typed string opener must count as string
        # context too (otherwise ``msg = "`` blocked the token ``open``).
        in_string = base_state.in_str is not None or bool(base_state.pending)
        if not in_string and (
            stripped in _FORBIDDEN or stripped.startswith(_FORBIDDEN_PREFIXES)
        ):
            return False
        st = advance_lexer_state(base_state, tok_str)
        if not st.valid:
            return False

        # Opening a call: the callee must be a known callable (or an allow-listed stdlib
        # member / safe container method).
        if "(" in tok_str and st.in_str is None and st.stack:
            _, callee, _ = st.stack[-1]
            if callee is not None:
                callee_key, safe_method = _classify_callee(callee, live_syms, self.config)
                if callee_key is None:
                    return False
                if not safe_method and phase != ProtocolPhase.EXPERIMENT:
                    if callee_key not in live_syms:
                        return False
                    ret = live_syms[callee_key].get("return_type", "any")
                    if expected_return_type and ret not in {expected_return_type, "any"}:
                        return False

        # Closing calls: check the argument shape recorded by the lexer for that frame.
        # (The previous implementation sliced source between the last ``(`` and the ``)``,
        # which grabbed the *inner* call on ``foo(bar(1), 2)``.)
        if ")" in tok_str and st.closed_frames:
            for (closed_delim, closed_callee, comma_cnt), meta in zip(st.closed_frames, st.closed_meta):
                if closed_delim != "(" or closed_callee is None:
                    continue
                callee_key, safe_method = _classify_callee(closed_callee, live_syms, self.config)
                if callee_key is None:
                    return False
                if safe_method or callee_key not in live_syms or phase == ProtocolPhase.EXPERIMENT:
                    continue
                if meta.get("saw_star") or meta.get("saw_equals"):
                    continue
                if meta.get("trailing_comma"):
                    return False
                n_args = (comma_cnt + 1) if meta.get("saw_content") else 0
                spec = live_syms.get(callee_key, {})
                min_a = int(spec.get("min_arity", 0) or 0)
                max_a = spec.get("max_arity")
                if n_args < min_a or (max_a is not None and n_args > int(max_a)):
                    return False
                if spec.get("keyword_only") and n_args > min_a:
                    return False
        return True

    def _mask_one(
        self,
        logits: np.ndarray,
        session: CodeSession,
        vocab_map: Mapping[int, str],
        vocab_size: int,
        active_code: str,
        remaining_budget: int,
        mode: str,
        expected_return_type: str | None,
        tokenizer_hash: str | None = None,
    ) -> np.ndarray:
        validate_vocab(vocab_map, vocab_size)
        if logits.ndim != 1:
            raise ValueError(f"logits must be 1-D, got shape {tuple(logits.shape)}")
        masked = np.array(logits, dtype=np.float64, copy=True)
        plan = _VOCAB_CACHE.for_vocab(vocab_map, tokenizer_hash)

        # Never-emit ids.
        if plan.drop_ids.size:
            valid_never = plan.drop_ids[(plan.drop_ids >= 0) & (plan.drop_ids < vocab_size)]
            masked[valid_never] = -np.inf

        phase = session.phase_for(active_code)

        # Protocol markers: mask unless the phase authorizes them.  Only tokens that could
        # be a marker (angle-bracket tokens) are inspected, not the whole vocabulary.
        marker_set = set(session.protocol_config.markers)
        for tid, tok_str in plan.marker_candidates:
            if tok_str not in marker_set or tid < 0 or tid >= vocab_size:
                continue
            allowed = False
            if phase == ProtocolPhase.REASONING and tok_str in (
                session.protocol_config.think_end,
                session.protocol_config.experiment_start,
            ):
                allowed = True
            if phase == ProtocolPhase.EXPERIMENT and tok_str == session.protocol_config.experiment_end:
                allowed = True
            if not allowed:
                masked[tid] = -np.inf

        if phase == ProtocolPhase.REASONING and remaining_budget > 4:
            return masked

        python_mode = mode == "python" or phase == ProtocolPhase.EXPERIMENT
        if not python_mode:
            return masked

        active = active_code
        if phase == ProtocolPhase.EXPERIMENT:
            # Use the experiment body: text after the last experiment start marker.
            marker = session.protocol_config.experiment_start
            idx = active.rfind(marker)
            if idx != -1:
                active = active[idx + len(marker):]

        live_syms = dict(SAFE_BUILTIN_CALLABLES)
        live_syms.update(session.extract_symbols(active_code))
        base_state = session.lexer_state_for(active)
        open_depth = len(base_state.stack)

        for tid, tok_str in plan.dynamic:
            if tid < 0 or tid >= vocab_size:
                continue
            if not self._token_allowed(
                tok_str, base_state, live_syms, phase, remaining_budget, open_depth,
                expected_return_type,
            ):
                masked[tid] = -np.inf
        return masked

    def mask_logits(
        self,
        logits: Any,
        session: CodeSession | None = None,
        prefix_text: str | None = None,
        vocab_map: Mapping[int, str] | None = None,
        remaining_token_budget: int = 256,
        mode: str = "python",
        expected_return_type: str | None = None,
        tokenizer_hash: str | None = None,
    ) -> np.ndarray:
        arr = np.asarray(logits)
        if arr.ndim != 1:
            raise ValueError(f"logits must be 1-D [vocab], got shape {tuple(arr.shape)}")
        if session is None:
            session = self.new_session(prefix=prefix_text or "", mode=mode)
        if vocab_map is None:
            vocab_map = self.vocabulary
        if vocab_map is None:
            raise ValueError("vocabulary is required for masking")
        vocab_size = int(arr.shape[0])
        validate_vocab(vocab_map, vocab_size)
        active_code = prefix_text if prefix_text is not None else ""
        # Update session lexer/symbols from active_code if provided
        if prefix_text is not None and prefix_text:
            # Only commit if not already; we use a lightweight approach: extract symbols.
            try:
                session.extract_symbols(prefix_text)
            except Exception:
                pass
        masked = self._mask_one(
            arr, session, vocab_map, vocab_size, active_code, remaining_token_budget, mode,
            expected_return_type, tokenizer_hash=tokenizer_hash,
        )
        # Dead end detection
        if not np.isfinite(masked).any():
            raise DeadEnd("all candidate tokens masked")
        return masked

    def filter_draft_tokens(
        self,
        draft_token_ids: list[int],
        session: CodeSession | None = None,
        prefix_text: str | None = None,
        vocab_map: Mapping[int, str] | None = None,
        remaining_token_budget: int = 256,
        mode: str = "python",
    ) -> list[int]:
        """Longest syntactically legal prefix of a proposed draft (syntax-only).

        Each draft token is checked in isolation with the single-token predicate, instead of
        running the full ``O(|V|)`` mask loop once per draft token.
        """
        if session is None:
            session = self.new_session(prefix=prefix_text or "", mode=mode)
        if vocab_map is None:
            vocab_map = self.vocabulary
        if vocab_map is None:
            raise ValueError("vocabulary is required")
        accepted: list[int] = []
        running = prefix_text or ""
        skip_grammar = False
        for offset, tid in enumerate(draft_token_ids):
            tok_str = vocab_map.get(tid)
            if tok_str is None:
                break
            if tid < 0 or tid >= (max(vocab_map) + 1 if vocab_map else 1):
                break
            if not skip_grammar:
                if _is_static_safe(tok_str):
                    accepted.append(tid)
                    running += tok_str
                    continue
                phase = session.phase_for(running)
                if mode != "python" and phase != ProtocolPhase.EXPERIMENT:
                    skip_grammar = True
                elif phase == ProtocolPhase.REASONING and remaining_token_budget > 4:
                    accepted.append(tid)
                    running += tok_str
                    continue
                else:
                    live_syms = dict(SAFE_BUILTIN_CALLABLES)
                    live_syms.update(session.extract_symbols(running))
                    base_state = session.lexer_state_for(running)
                    ok = self._token_allowed(
                        tok_str, base_state, live_syms, phase,
                        max(1, remaining_token_budget - offset), len(base_state.stack), None,
                    )
                    if not ok:
                        break
            accepted.append(tid)
            running += tok_str
        return accepted


# End of module.