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

from .config import CodeEngineConfig
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


class _VocabCache:
    """Thread-safe, bounded, immutable cache of static vocabulary classification."""

    def __init__(self, max_entries: int = 32) -> None:
        self._lock = threading.RLock()
        self._max = max_entries
        self._store: dict[str, tuple[np.ndarray, np.ndarray, tuple[tuple[int, str], ...]]] = {}

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

    def get_or_build(
        self, key: str, vocab_map: Mapping[int, str]
    ) -> tuple[np.ndarray, np.ndarray, tuple[tuple[int, str], ...]]:
        with self._lock:
            cached = self._store.get(key)
            if cached is not None:
                return cached
            never_ids: list[int] = []
            protocol_ids: list[int] = []
            entries: list[tuple[int, str]] = []
            for tid, tok in vocab_map.items():
                if not isinstance(tid, (int, np.integer)):
                    raise ValueError(f"vocabulary key {tid!r} is not an integer id")
                tid_int = int(tid)
                if tid_int < 0:
                    raise ValueError(f"negative vocabulary id {tid_int}")
                # We cannot know vocab size here; range check happens in mask_logits.
                text = str(tok)
                # Classify by simple heuristics; the host may override via config.
                if text in {"", "\x00"}:
                    never_ids.append(tid_int)
                else:
                    entries.append((tid_int, text))
            result = (
                np.asarray(sorted(set(never_ids)), dtype=np.int64),
                np.asarray(sorted(set(protocol_ids)), dtype=np.int64),
                tuple(entries),
            )
            if len(self._store) >= self._max:
                # Evict oldest (insertion-ordered dict).
                self._store.pop(next(iter(self._store)))
            self._store[key] = result
            return result

    def invalidate(self) -> None:
        with self._lock:
            self._store.clear()


_VOCAB_CACHE = _VocabCache()


def validate_vocab(
    vocab_map: Mapping[int, str], vocab_size: int
) -> None:
    """Validate that every id is an int in ``[0, vocab_size)``."""
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    for tid in vocab_map:
        if isinstance(tid, bool) or not isinstance(tid, (int, np.integer)):
            raise ValueError(f"vocabulary key {tid!r} is not an integer")
        tid_int = int(tid)
        if tid_int < 0 or tid_int >= vocab_size:
            raise ValueError(
                f"vocabulary id {tid_int} out of range [0, {vocab_size})"
            )
    # Detect duplicate decoded text? Not required; allow sparse ids.


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
    ) -> np.ndarray:
        validate_vocab(vocab_map, vocab_size)
        if logits.ndim != 1:
            raise ValueError(f"logits must be 1-D, got shape {tuple(logits.shape)}")
        masked = np.array(logits, dtype=np.float64, copy=True)
        # NaN/inf safety: keep as-is for finite entries.
        key = _VOCAB_CACHE.key_for(vocab_map, None)
        never_ids, _protocol_cached, entries = _VOCAB_CACHE.get_or_build(key, vocab_map)

        # Derive current protocol phase from active_code (throwaway tracker, all markers authorized).
        phase_tracker = ProtocolTracker(session.protocol_config, authorized_starts=frozenset(session.protocol_config.markers))
        try:
            phase_tracker.feed(active_code)
        except Exception:
            pass
        phase = phase_tracker.phase

        marker_set = set(session.protocol_config.markers)
        protocol_ids = [int(tid) for tid, tok in vocab_map.items() if str(tok) in marker_set]

        # Mask never-emit ids.
        if never_ids.size:
            valid_never = never_ids[(never_ids >= 0) & (never_ids < vocab_size)]
            masked[valid_never] = -np.inf

        # Protocol markers: mask unless the phase authorizes them.
        for tid in protocol_ids:
            if tid < 0 or tid >= vocab_size:
                continue
            tok_str = vocab_map.get(tid, "")
            allowed = False
            if phase == ProtocolPhase.REASONING and tok_str in (session.protocol_config.think_end, session.protocol_config.experiment_start):
                allowed = True
            if phase == ProtocolPhase.EXPERIMENT and tok_str == session.protocol_config.experiment_end:
                allowed = True
            if not allowed:
                masked[tid] = -np.inf

        in_reasoning = phase == ProtocolPhase.REASONING
        if in_reasoning and remaining_budget > 4:
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

        live_syms = session.extract_symbols(active_code)
        base_state = advance_lexer_state(LexerState(), active)
        open_depth = len(base_state.stack)

        for tid, tok_str in entries:
            if tid < 0 or tid >= vocab_size:
                continue
            if remaining_budget <= open_depth + 1 and any(c in tok_str for c in "([{"):
                masked[tid] = -np.inf
                continue
            # Block forbidden modules/attributes outright (not just on call).
            stripped = tok_str.strip()
            if stripped in _FORBIDDEN or any(stripped.startswith(f + ".") for f in _FORBIDDEN):
                masked[tid] = -np.inf
                continue
            st = advance_lexer_state(base_state, tok_str)
            if not st.valid:
                masked[tid] = -np.inf
                continue

            # call checks
            if "(" in tok_str and st.in_str is None and st.stack:
                _, callee, _ = st.stack[-1]
                if callee is not None:
                    callee_key, safe_method = _classify_callee(callee, live_syms, self.config)
                    if callee_key is None:
                        masked[tid] = -np.inf
                        continue
                    if not safe_method:
                        # require known symbol unless in experiment
                        if callee_key not in live_syms and phase != ProtocolPhase.EXPERIMENT:
                            masked[tid] = -np.inf
                            continue
                        if callee_key in live_syms:
                            ret = live_syms[callee_key].get("return_type", "any")
                            if expected_return_type and ret not in {expected_return_type, "any"}:
                                masked[tid] = -np.inf
                                continue

            if ")" in tok_str and base_state.valid:
                # check closed frames
                closed = st.closed_frames
                if closed:
                    closed_delim, closed_callee, comma_cnt = closed[-1]
                    if closed_delim == "(" and closed_callee is not None:
                        callee_key, safe_method = _classify_callee(closed_callee, live_syms, self.config)
                        if callee_key is None:
                            masked[tid] = -np.inf
                            continue
                        if not safe_method and callee_key in live_syms and phase != ProtocolPhase.EXPERIMENT:
                            # count args (Maya parity: inspect the inner call text)
                            tok_lparen = tok_str.rfind("(")
                            lparen = (
                                len(active) + tok_lparen
                                if tok_lparen != -1
                                else active.rfind("(")
                            )
                            rparen = len(active) + tok_str.rfind(")")
                            inner_call = (
                                (active + tok_str)[lparen + 1 : rparen].strip()
                                if rparen > lparen
                                else ""
                            )
                            if inner_call.endswith(",") and comma_cnt >= 1:
                                masked[tid] = -np.inf
                                continue
                            if "*" not in inner_call and "=" not in inner_call:
                                n_args = 0 if inner_call == "" else (comma_cnt + 1)
                                spec = live_syms.get(callee_key, {})
                                min_a = spec.get("min_arity", 0)
                                max_a = spec.get("max_arity", n_args)
                                if n_args < min_a or (max_a is not None and n_args > max_a):
                                    masked[tid] = -np.inf
                                    continue
                            # keyword-only
                            # best-effort: skip

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
            arr, session, vocab_map, vocab_size, active_code, remaining_token_budget, mode, expected_return_type
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
        """Longest syntactically legal prefix of a proposed draft (syntax-only)."""
        if session is None:
            session = self.new_session(prefix=prefix_text or "", mode=mode)
        if vocab_map is None:
            vocab_map = self.vocabulary
        if vocab_map is None:
            raise ValueError("vocabulary is required")
        accepted: list[int] = []
        running = prefix_text or ""
        for offset, tid in enumerate(draft_token_ids):
            if tid not in vocab_map:
                break
            if tid < 0 or tid >= (max(vocab_map) + 1 if vocab_map else 1):
                break
            dummy = np.zeros(max(vocab_map) + 1, dtype=np.float64)
            dummy[tid] = 1.0
            try:
                masked = self.mask_logits(
                    dummy, session=session, prefix_text=running, vocab_map=vocab_map,
                    remaining_token_budget=max(1, remaining_token_budget - offset), mode=mode,
                )
            except DeadEnd:
                break
            if not np.isfinite(masked[tid]):
                break
            accepted.append(tid)
            running = running + vocab_map[tid]
        return accepted


# End of module.