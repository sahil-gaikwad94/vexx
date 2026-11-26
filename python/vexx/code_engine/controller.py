"""VexxCodeController / CodeSession façade for Vexx 2.1."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

try:
    import numpy as np
except ImportError:
    np = None

from .lexer import LexerState, advance_lexer_state
from .masking import CodeSession
from .masking import VexxCodeController as _Base


class VexxCodeController(_Base):
    """Public façade. Extends the masking controller with convenience methods."""

    def verify_candidate(self, source: str, function_name: str, test_cases: list[dict[str, Any]]) -> Any:
        from .candidate import CandidateVerifier
        verifier = CandidateVerifier(self.config)
        return verifier.verify_candidate(source, function_name, test_cases)

    def select_candidate(
        self, candidates: list[str], function_name: str, inputs: list[list[Any]], timeout: float | None = None
    ) -> Any:
        from .candidate import CandidateVerifier
        verifier = CandidateVerifier(self.config)
        return verifier.select_candidate(candidates, function_name, inputs, timeout=timeout)

    def fuzz_against_oracle(
        self, candidate: str, reference: str, function_name: str,
        inputs: list[list[Any]] | None = None, timeout: float | None = None,
    ) -> Any:
        from .candidate import CandidateVerifier
        verifier = CandidateVerifier(self.config)
        return verifier.fuzz_against_oracle(candidate, reference, function_name, inputs=inputs, timeout=timeout)

    def filter_draft_tokens(
        self,
        draft_token_ids: list[int],
        session: CodeSession | None = None,
        prefix_text: str | None = None,
        vocab_map: Mapping[int, str] | None = None,
        remaining_token_budget: int = 256,
        mode: str = "python",
    ) -> list[int]:
        return super().filter_draft_tokens(
            draft_token_ids, session=session, prefix_text=prefix_text,
            vocab_map=vocab_map, remaining_token_budget=remaining_token_budget, mode=mode,
        )

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
        return super().mask_logits(
            logits, session=session, prefix_text=prefix_text, vocab_map=vocab_map,
            remaining_token_budget=remaining_token_budget, mode=mode,
            expected_return_type=expected_return_type, tokenizer_hash=tokenizer_hash,
        )

    # ------------------------------------------------------------------ maya parity

    @staticmethod
    def crane_phase(prefix_text: str) -> str:
        """CRANE phase name for ``prefix_text`` (delegates to the marker state machine).

        This used to be a second implementation built on ``str.rfind``, which cannot tell a
        half-emitted marker or a malformed nesting from a real transition.
        """
        from .protocol import ProtocolConfig, ProtocolTracker
        config = ProtocolConfig()
        tracker = ProtocolTracker(config, authorized_starts=frozenset(config.markers))
        try:
            tracker.feed(prefix_text)
        except Exception:
            pass
        return {
            "reasoning": "UNCONSTRAINED_THINK",
            "experiment": "EXPERIMENT_DESIGN",
        }.get(tracker.phase.value, "STRICT_VERIFIED")

    @staticmethod
    def _lex_code_state(code_prefix: str) -> dict:
        state = advance_lexer_state(LexerState(), code_prefix)
        # A quote that remains suspended inside an f-string replacement field at
        # end-of-input is not a legal nested string opener (Maya parity).
        if state.valid and state.fstring_stack and state.pending and state.pending[-1] in ("'", '"'):
            state.valid = False
            state.reason = "quote_in_fstring_expression"
            state.pending = ""
        return state.to_dict()

    def apply_in_flight_mask(
        self,
        logits: Any,
        prefix_text: str,
        vocab_map: Mapping[int, str],
        expected_return_type: str | None = None,
        remaining_token_budget: int = 256,
        mode: str = "python",
        tokenizer_hash: str | None = None,
    ) -> np.ndarray:
        return self.mask_logits(
            logits,
            prefix_text=prefix_text,
            vocab_map=vocab_map,
            expected_return_type=expected_return_type,
            remaining_token_budget=remaining_token_budget,
            mode=mode,
            tokenizer_hash=tokenizer_hash,
        )

    def filter_draft_tokens_by_grammar(
        self,
        prefix_text: str,
        draft_token_ids: list[int],
        vocab_map: Mapping[int, str],
        remaining_token_budget: int = 256,
        mode: str = "python",
    ) -> list[int]:
        accepted: list[int] = []
        running_text = prefix_text
        keys = list(vocab_map.keys()) if vocab_map else [0]
        dummy_logits = np.zeros((max(keys) + 1) if keys else 1, dtype=np.float64)
        for step_offset, tid in enumerate(draft_token_ids):
            budget = max(1, remaining_token_budget - step_offset)
            masked = self.apply_in_flight_mask(
                dummy_logits, running_text, vocab_map,
                remaining_token_budget=budget, mode=mode,
            )
            if tid >= len(masked) or not np.isfinite(masked[tid]):
                break
            accepted.append(tid)
            running_text = running_text + vocab_map.get(tid, "")
        return accepted

    def save_checkpoint(
        self,
        var_name: str,
        token_idx: int,
        hybrid_cache: Any,
        entropy: float,
        proposed_val: float,
    ) -> None:
        from ..cache_protocol import _contains_recurrent_state, clone_cache_generic
        if (
            isinstance(hybrid_cache, dict)
            and hybrid_cache.get("position") is not None
            and _contains_recurrent_state(hybrid_cache)
        ):
            position = int(hybrid_cache["position"])
            if position != int(token_idx):
                raise ValueError(
                    f"save_checkpoint for token {int(token_idx)} received a cache at "
                    f"position {position}; recurrent state cannot be rewound by slicing"
                )
        self.checkpoints.append(
            {
                "var": var_name,
                "token_idx": int(token_idx),
                "cache": clone_cache_generic(hybrid_cache),
                "bounds_snapshot": {k: v for k, v in self.bounds.items()},
                "entropy": float(entropy),
                "val": float(proposed_val),
            }
        )

    def single_row_bound_contractor(
        self,
        conflict_vars: set,
        constraint: dict[str, Any],
        assignment: dict[str, Any],
    ) -> tuple[int, Any, tuple[float, float]]:
        """Contract one bound from a conflicting row (delegates to ``conflict``).

        The duplicated copy here ignored ``constraint["relation"]``, so a ``>=`` row was
        contracted in the wrong direction.  :func:`conflict.contract_single_row` is the
        single source of truth.
        """
        from .conflict import CodeCheckpoint, contract_single_row
        candidates = [cp for cp in self.checkpoints if cp["var"] in conflict_vars]
        if not candidates:
            raise ValueError(f"No checkpoint found for conflict core {sorted(conflict_vars)}")
        checkpoints = [
            CodeCheckpoint(
                variable=cp["var"],
                token_index=int(cp["token_idx"]),
                cache_snapshot=cp["cache"],
                bounds_snapshot=dict(cp.get("bounds_snapshot") or {}),
                entropy=float(cp.get("entropy", 0.0)),
                proposed_value=float(cp.get("val", 0.0)),
            )
            for cp in self.checkpoints
        ]
        cut = contract_single_row(
            conflict_vars,
            dict(constraint["coeffs"]),
            constraint["rhs"],
            assignment,
            checkpoints=checkpoints,
            relation=str(constraint.get("relation", "<=")),
        )
        c_var = cut.variable
        tok_idx = cut.checkpoint.token_index if cut.checkpoint is not None else 0
        self.bounds = dict((cut.checkpoint.bounds_snapshot if cut.checkpoint else {}) or self.bounds)
        self.last_exact_fraction = cut.bound
        lo, hi = self.bounds.get(c_var, (-float("inf"), float("inf")))
        if cut.direction == "upper":
            self.bounds[c_var] = (lo, min(hi, cut.bound_float))
        elif cut.direction == "lower":
            self.bounds[c_var] = (max(lo, cut.bound_float), hi)
        else:
            self.bounds[c_var] = (cut.bound_float, cut.bound_float)
        self.checkpoints = [cp for cp in self.checkpoints if cp["token_idx"] <= tok_idx]
        return tok_idx, cut.rewound_cache, self.bounds[c_var]

    trigger_farkas_cdcl_jumpback = single_row_bound_contractor

    def execute_experiment_block(self, script_str: str, timeout: float = 3.0) -> str:
        from .policy import check_ast_policy
        from .sandbox import SandboxRunner
        try:
            check_ast_policy(script_str)
        except (SyntaxError, PermissionError, ValueError) as exc:
            return f"EXP_ERROR: {exc}"
        runner = SandboxRunner.from_config(self.config)
        combined = "\n".join([*self.experiment_preamble, script_str])
        if self.experiment_preamble:
            try:
                check_ast_policy(combined)
            except (SyntaxError, PermissionError, ValueError) as exc:
                return f"EXP_ERROR: {exc}"
        res = runner.run(combined, mode="experiment_script", timeout=timeout)
        if not res.ok:
            return f"EXP_ERROR: {res.error}"
        self._absorb_experiment_definitions(script_str)
        return str(res.repr) if res.repr is not None else str(res.result)

    def _absorb_experiment_definitions(self, script_str: str) -> None:
        import ast as _ast
        try:
            tree = _ast.parse(script_str)
        except SyntaxError:
            return
        for node in tree.body:
            if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef, _ast.ClassDef, _ast.Import, _ast.ImportFrom)):
                self.experiment_preamble.append(_ast.unparse(node))
            elif isinstance(node, (_ast.Assign, _ast.AnnAssign)):
                targets = node.targets if isinstance(node, _ast.Assign) else [node.target]
                if any(isinstance(t, _ast.Name) and t.id == "RESULT" for t in targets):
                    continue
                self.experiment_preamble.append(_ast.unparse(node))

    # static delegations to module-level helpers
    @staticmethod
    def verify_statistical_report(text: str, claims=None):
        from ..calibration import verify_statistical_report
        return verify_statistical_report(text, claims)

    @staticmethod
    def parse_unit_dimensions(text: str):
        from ..calibration import parse_unit_dimensions
        return parse_unit_dimensions(text)

    @staticmethod
    def verify_epistemic_calibration(text: str):
        from ..calibration import verify_epistemic_calibration
        return verify_epistemic_calibration(text)

    @staticmethod
    def _chi2_survival(statistic: float, df: int):
        from ..calibration import _chi2_survival
        return _chi2_survival(statistic, df)

    @staticmethod
    def _normal_two_sided_p(statistic: float):
        from ..calibration import _normal_two_sided_p
        return _normal_two_sided_p(statistic)

    @staticmethod
    def crane_cas_evaluate(expr_str: str, env=None):
        from ..cas import crane_cas_evaluate
        return crane_cas_evaluate(expr_str, env)

    @staticmethod
    def safe_sympy(expr: str, env=None):
        from ..cas import safe_sympy
        return safe_sympy(expr, env)

    @staticmethod
    def solve_linear_system(eqs, vars_):
        from ..cas import solve_linear_system
        return solve_linear_system(eqs, vars_)

    @staticmethod
    def extract_ast_symbol_table(code_text: str):
        from ..code_engine.contracts import extract_ast_symbol_table
        return extract_ast_symbol_table(code_text)

    def compute_rlvr_group_rewards(self, candidates, fn_name, test_cases):
        from ..rl.code_rewards import compute_code_rewards
        return compute_code_rewards(candidates, fn_name, test_cases, config=self.config)

    def select_verified(self, candidates, function_name, test_cases):
        from ..code_engine.candidate import CandidateVerifier
        return CandidateVerifier(self.config).select_verified(candidates, function_name, test_cases)

    @staticmethod
    def _contains_recurrent_state(node):
        from ..cache_protocol import _contains_recurrent_state
        return _contains_recurrent_state(node)

    import ast as _ast

    FORBIDDEN_AST_NODES = (_ast.Global,)
    ALLOWED_STDLIB_MODULES = frozenset(
        {
            "math", "heapq", "collections", "bisect", "itertools",
            "functools", "typing", "fractions", "decimal", "operator", "string",
        }
    )
    FORBIDDEN_IDENTIFIERS = {
        "os", "sys", "subprocess", "socket", "shutil", "pathlib", "open",
        "eval", "exec", "compile", "__import__", "globals", "locals",
        "getattr", "setattr", "delattr", "vars", "input", "breakpoint",
    }
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

    @classmethod
    def _check_import_policy(cls, node):
        import ast as _ast
        if isinstance(node, _ast.Import):
            modules = [alias.name for alias in node.names]
            imported_names = []
        elif isinstance(node, _ast.ImportFrom):
            if int(node.level or 0) != 0:
                raise PermissionError("relative imports are forbidden in the sandbox")
            modules = [node.module or ""]
            imported_names = [alias.name for alias in node.names]
        else:
            raise PermissionError(f"unexpected import node {type(node).__name__}")
        for name in modules:
            root = str(name).split(".")[0]
            if root not in cls.ALLOWED_STDLIB_MODULES:
                raise PermissionError(f"Forbidden import: {name}")
            if any(part.startswith("_") for part in str(name).split(".")):
                raise PermissionError(f"Forbidden private import: {name}")
        for name in imported_names:
            if str(name).split(".")[0].startswith("_"):
                raise PermissionError(f"Forbidden private import: {name}")
        for alias in node.names:
            if alias.asname and str(alias.asname).startswith("_"):
                raise PermissionError(f"Forbidden private import alias: {alias.asname}")

    @classmethod
    def _check_ast_policy(cls, code_str):
        """Delegate to :func:`vexx.code_engine.policy.check_ast_policy`.

        The previous copy here was missing the frame/format/size checks, so code rejected by
        the real policy could still pass through this path.
        """
        from .policy import check_ast_policy
        return check_ast_policy(code_str)

    @classmethod
    def _sandbox_dispatch(cls, code_str, fn_name="", args=None, timeout=5.0, mode="function_call"):
        from ..code_engine.sandbox import SandboxRunner
        runner = SandboxRunner(wall_timeout_s=timeout)
        result = runner.run(code_str, fn_name=fn_name, args=args or [], timeout=timeout, mode=mode)
        if not result.ok:
            return False, None, None, result.error
        # ``display`` is the bounded representation (never the raw object).
        if result.serialized_kind in {"json", "literal"}:
            try:
                import json
                display = json.dumps(result.result, sort_keys=True, default=str)
            except Exception:
                display = result.repr
        else:
            display = result.repr
        return True, display, result.input_after, None

    @classmethod
    def safe_execute_python(cls, code_str, fn_name, test_cases):
        """Verify ``code_str`` against ``test_cases`` (single implementation).

        This used to re-implement the verifier with a broken result contract: because
        ``SandboxRunner`` returned ``repr=None``, every return value was read as ``None``
        and every test failed with ``got None``.
        """
        from .candidate import CandidateVerifier
        from .config import CodeEngineConfig
        verifier = CandidateVerifier(CodeEngineConfig())
        report = verifier.verify_candidate(code_str, fn_name, test_cases)
        return report.passed, report.total, list(report.errors)
