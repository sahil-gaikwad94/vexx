"""Explicit, validated configuration for the Vexx 2.1 code engine.

Every limit the engine enforces lives here as a named field with a bounded default.
Maya hard-coded these as class constants scattered through the controller, which made
them impossible to audit or override per deployment.  Validation happens once, at
construction, so a bad configuration fails immediately instead of halfway through a
decode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["CodeEngineConfig", "DEFAULT_ALLOWED_IMPORTS", "CODE_MODES", "UNKNOWN_CALL_POLICIES"]

#: Purely algorithmic standard-library modules a candidate may import.  ``os``, ``sys``,
#: ``socket``, ``subprocess``, ``pathlib`` and friends stay blocked at both the AST
#: policy layer and the child runtime ``__import__`` layer.
DEFAULT_ALLOWED_IMPORTS: tuple[str, ...] = (
    "math",
    "heapq",
    "collections",
    "bisect",
    "itertools",
    "functools",
    "typing",
    "fractions",
    "decimal",
    "operator",
    "string",
)

#: ``python`` applies Python grammar; ``text`` applies protocol rules only.
CODE_MODES = ("python", "text")

#: What to do when a call target cannot be resolved to a known callable.
UNKNOWN_CALL_POLICIES = ("allow_with_warning", "block")


def _require_int(name: str, value: Any, minimum: int, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}, got {value}")
    return value


def _require_float(name: str, value: Any, minimum: float, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    value = float(value)
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError(f"{name} must be finite, got {value}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}, got {value}")
    return value


@dataclass
class CodeEngineConfig:
    """Validated policy and resource limits for the code engine.

    The defaults are deliberately conservative: they bound work per decode step, per
    candidate and per child process, so a hostile or merely confused model cannot turn
    generation into a denial-of-service against the host.
    """

    # -- protocol / grammar ------------------------------------------------
    mode: str = "python"
    enforce_grammar: bool = True
    enforce_known_call_contracts: bool = True
    unknown_call_policy: str = "allow_with_warning"

    # -- sandbox policy ----------------------------------------------------
    allowed_imports: tuple[str, ...] = DEFAULT_ALLOWED_IMPORTS
    #: AST nodes rejected outright.  ``nonlocal`` is *allowed*: it only rebinds names in
    #: an enclosing function frame and has no sandbox-escape surface, while blocking it
    #: rejects textbook Tarjan SCC solutions that use ``nonlocal`` for counters.
    forbidden_ast_nodes: tuple[str, ...] = ("Global",)
    forbidden_identifiers: tuple[str, ...] = (
        "os", "sys", "subprocess", "socket", "shutil", "pathlib", "open",
        "eval", "exec", "compile", "__import__", "globals", "locals",
        "getattr", "setattr", "delattr", "vars", "input", "breakpoint",
    )

    # -- expression / CAS bounds ------------------------------------------
    max_prefix_chars: int = 64_000
    signature_scan_window: int = 16_384
    max_expression_chars: int = 512
    max_expression_nodes: int = 200
    max_exponent: int = 10_000
    max_integer_bits: int = 4096
    max_factorial_argument: int = 5000
    max_binomial_argument: int = 100_000
    max_variadic_args: int = 64
    max_equations: int = 8
    max_system_variables: int = 8

    # -- sandbox resources -------------------------------------------------
    sandbox_wall_timeout_s: float = 5.0
    sandbox_cpu_seconds: int = 2
    sandbox_memory_bytes: int = 512 * 1024 * 1024
    sandbox_output_bytes: int = 1_000_000
    sandbox_max_files: int = 64
    sandbox_max_processes: int = 64
    #: When true, a profile that cannot be established fails closed instead of running
    #: with weaker isolation.  Never silently fall back to in-process ``exec``.
    sandbox_require_isolation: bool = False
    #: Advertised network isolation.  The engine only sets this true when the platform
    #: layer actually enforces it; it is never assumed.
    network_isolated: bool = False

    # -- experiments -------------------------------------------------------
    max_experiment_cells: int = 64
    max_experiment_source_chars: int = 24_000
    max_preamble_chars: int = 24_000

    # -- candidates --------------------------------------------------------
    max_candidate_count: int = 32
    max_probe_count: int = 256
    max_test_cases: int = 256
    max_input_depth: int = 8
    max_input_chars: int = 200_000

    # -- masking -----------------------------------------------------------
    #: Bound on the static vocabulary classification cache.
    vocab_cache_size: int = 8
    #: Remaining-token budget below which the mask stops opening new brackets, so a
    #: decode cannot end with unbalanced delimiters.
    min_budget_to_open: int = 1

    # -- rewards -----------------------------------------------------------
    reward_partial_scale: float = 0.3
    reward_syntax_penalty: float = -0.5
    reward_contract_penalty: float = -0.2
    reward_mutation_penalty: float = -0.2
    advantage_epsilon: float = 1e-6
    advantage_std_threshold: float = 1e-8

    def __post_init__(self) -> None:
        if self.mode not in CODE_MODES:
            raise ValueError(f"mode must be one of {CODE_MODES}, got {self.mode!r}")
        if self.unknown_call_policy not in UNKNOWN_CALL_POLICIES:
            raise ValueError(
                f"unknown_call_policy must be one of {UNKNOWN_CALL_POLICIES}, got {self.unknown_call_policy!r}"
            )
        for name in ("enforce_grammar", "enforce_known_call_contracts", "sandbox_require_isolation", "network_isolated"):
            value = getattr(self, name)
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be a bool, got {type(value).__name__}")

        self.allowed_imports = _validate_imports(self.allowed_imports)
        self.forbidden_identifiers = _validate_names(self.forbidden_identifiers, "forbidden_identifiers")
        self.forbidden_ast_nodes = _validate_names(self.forbidden_ast_nodes, "forbidden_ast_nodes")

        for name in (
            "max_prefix_chars", "signature_scan_window", "max_expression_chars",
            "max_expression_nodes", "max_exponent", "max_integer_bits",
            "max_factorial_argument", "max_binomial_argument", "max_variadic_args",
            "max_equations", "max_system_variables", "sandbox_cpu_seconds",
            "sandbox_memory_bytes", "sandbox_output_bytes", "sandbox_max_files",
            "sandbox_max_processes", "max_experiment_cells",
            "max_experiment_source_chars", "max_preamble_chars",
            "max_candidate_count", "max_probe_count", "max_test_cases",
            "max_input_depth", "max_input_chars", "vocab_cache_size",
            "min_budget_to_open",
        ):
            _require_int(name, getattr(self, name), 1)

        for name in ("sandbox_wall_timeout_s",):
            _require_float(name, getattr(self, name), 0.001, 3600.0)
        for name in ("reward_partial_scale", "advantage_epsilon"):
            _require_float(name, getattr(self, name), 0.0, 1.0)
        for name in ("reward_syntax_penalty", "reward_contract_penalty", "reward_mutation_penalty"):
            _require_float(name, getattr(self, name), -100.0, 0.0)
        _require_float("advantage_std_threshold", self.advantage_std_threshold, 0.0, 1.0)

    # ------------------------------------------------------------- helpers

    def with_overrides(self, **overrides: Any) -> CodeEngineConfig:
        """Return a validated copy with ``overrides`` applied."""
        payload = dict(self.__dict__)
        unknown = set(overrides) - set(payload)
        if unknown:
            raise TypeError(f"unknown CodeEngineConfig field(s): {sorted(unknown)}")
        payload.update(overrides)
        return CodeEngineConfig(**payload)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    def describe_limits(self) -> dict[str, Any]:
        """The resource/security limits, for receipts and audit logs."""
        return {
            "wall_timeout_s": self.sandbox_wall_timeout_s,
            "cpu_seconds": self.sandbox_cpu_seconds,
            "memory_bytes": self.sandbox_memory_bytes,
            "output_bytes": self.sandbox_output_bytes,
            "max_files": self.sandbox_max_files,
            "max_processes": self.sandbox_max_processes,
            "network_isolated": self.network_isolated,
            "allowed_imports": list(self.allowed_imports),
        }


def _validate_imports(imports: Any) -> tuple[str, ...]:
    if isinstance(imports, str) or not isinstance(imports, (list, tuple, set, frozenset)):
        raise TypeError("allowed_imports must be a sequence of module names")
    out: list[str] = []
    for item in imports:
        if not isinstance(item, str) or not item.isidentifier():
            raise ValueError(f"invalid module name in allowed_imports: {item!r}")
        if item.startswith("_"):
            raise ValueError(f"private module names are not allowed: {item!r}")
        out.append(item)
    return tuple(dict.fromkeys(out))


def _validate_names(names: Any, label: str) -> tuple[str, ...]:
    if isinstance(names, str) or not isinstance(names, (list, tuple, set, frozenset)):
        raise TypeError(f"{label} must be a sequence of strings")
    out: list[str] = []
    for item in names:
        if not isinstance(item, str) or not item:
            raise ValueError(f"invalid entry in {label}: {item!r}")
        out.append(item)
    return tuple(dict.fromkeys(out))