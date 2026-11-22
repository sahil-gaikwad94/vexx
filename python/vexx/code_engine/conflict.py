"""Rational single-row conflict contractor for Vexx 2.1."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

from ..cache_protocol import clone_cache_generic, truncate_cache_at_token


@dataclass
class CodeCheckpoint:
    variable: str
    token_index: int
    cache_snapshot: Any
    bounds_snapshot: dict[str, tuple[Fraction | float, Fraction | float]]
    entropy: float
    proposed_value: Fraction | float
    context_id: str = ""


@dataclass
class BoundCut:
    variable: str
    bound: Fraction
    bound_float: float
    relation: str
    strict: bool
    direction: str
    checkpoint: CodeCheckpoint | None = None
    rewound_cache: Any = None
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def contract_single_row(
    conflict_vars: set[str],
    coeffs: dict[str, Any],
    rhs: Any,
    assignment: dict[str, Any],
    checkpoints: list[CodeCheckpoint] | None = None,
    relation: str = "<=",
    context_id: str = "",
    selected_checkpoint: CodeCheckpoint | None = None,
) -> BoundCut:
    if relation not in ("<=", "<", ">=", ">", "=="):
        raise ValueError(f"unsupported relation {relation!r}")
    candidates = [cp for cp in (checkpoints or []) if cp.variable in conflict_vars and cp.context_id == context_id]
    if selected_checkpoint is None:
        if not candidates:
            raise ValueError(f"No checkpoint found for conflict core {sorted(conflict_vars)}")
        selected_checkpoint = max(candidates, key=lambda c: c.entropy)
    cp = selected_checkpoint
    if cp.variable not in conflict_vars:
        raise ValueError(f"selected checkpoint variable {cp.variable!r} is not in conflict core")
    if cp.context_id != context_id:
        raise ValueError("selected checkpoint belongs to a different context")
    c_var = cp.variable
    if c_var not in coeffs:
        raise ValueError(f"Conflict variable '{c_var}' absent from constraint coefficients")
    a_c = Fraction(str(coeffs[c_var]))
    if a_c == 0:
        raise ValueError("Bound contractor needs a non-zero coefficient for the culprit")

    rhs_f = Fraction(str(rhs))
    rem = rhs_f
    for var, coeff in coeffs.items():
        if var == c_var:
            continue
        if var not in assignment:
            raise ValueError(f"Constraint references '{var}' missing from assignment")
        rem -= Fraction(str(coeff)) * Fraction(str(assignment[var]))

    exact_bound = rem / a_c
    if relation in ("<=", "<"):
        direction = "upper" if a_c > 0 else "lower"
    elif relation in (">=", ">"):
        direction = "lower" if a_c > 0 else "upper"
    else:
        if len(coeffs) != 1:
            raise ValueError("equality row does not determine the culprit uniquely")
        direction = "exact"

    strict = relation in ("<", ">")

    bound_float = float(exact_bound)
    if direction == "upper":
        widened = math.nextafter(bound_float, float("inf"))
    elif direction == "lower":
        widened = math.nextafter(bound_float, -float("inf"))
    else:
        widened = bound_float

    rewound = None
    try:
        rewound = truncate_cache_at_token(clone_cache_generic(cp.cache_snapshot), cp.token_index)
    except Exception:
        rewound = None

    return BoundCut(
        variable=c_var,
        bound=exact_bound,
        bound_float=widened,
        relation=relation,
        strict=strict,
        direction=direction,
        checkpoint=cp,
        rewound_cache=rewound,
        reason=f"culprit={c_var} coeff={a_c} relation={relation}",
    )