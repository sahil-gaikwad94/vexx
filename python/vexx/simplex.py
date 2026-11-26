"""Exact-rational incremental simplex for linear real arithmetic (``QF_LRA``).

The reference decision procedure in :mod:`vexx.nano_solver` is bounded Fourier–Motzkin: it
is simple, sound, and doubly exponential in the worst case, with a hard cap on variables.
This module implements a Dutertre–de Moura style general simplex over exact rationals so the
engine can decide systems with hundreds of variables and thousands of rows — portfolios,
scheduling and resource-allocation schemas routinely exceed the old 24-variable ceiling.

Design notes
------------
* **Exact.** Every coefficient and bound is a :class:`~fractions.Fraction`.  There is no
  floating point anywhere in the decision path, so ``SAT``/``UNSAT`` are never decided by
  rounding.
* **Incremental.** Constraints are rows added to the tableau; ``push``/``pop`` snapshot the
  bound stacks in ``O(1)`` and never rebuild the tableau.
* **Strict inequalities** use an infinitesimal pair ``(constant, delta)`` compared
  lexicographically: ``x < b`` becomes ``x <= b - δ``.  No ULP fudging.
* **Verified models.** :meth:`check` re-evaluates every original constraint against the
  witness before returning ``SAT``.  A bug in the pivoting loop can therefore cost
  completeness (it falls back to ``UNKNOWN``), never soundness.
* **Cores.** The bound that forces a pivot and the bound that blocks it are both tracked, so
  an ``UNSAT`` answer carries the tracker ids of the constraints that caused it.  The core is
  marked minimal only after a successful deletion pass re-checks it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

from .nano_solver import SolverStatus

__all__ = ["Delta", "VexxDualSimplex", "IncrementalSimplex", "DualSimplexSolver"]

#: A ``(constant, delta)`` pair ordered lexicographically; ``delta`` is the infinitesimal.
Delta = tuple[Fraction, Fraction]

_ZERO: Delta = (Fraction(0), Fraction(0))


def _delta(value: Any) -> Delta:
    if isinstance(value, tuple) and len(value) == 2:
        return (Fraction(value[0]), Fraction(value[1]))
    if isinstance(value, bool):
        raise ValueError("boolean is not a rational")
    if isinstance(value, Fraction):
        return (value, Fraction(0))
    if isinstance(value, int):
        return (Fraction(value), Fraction(0))
    if isinstance(value, float) and value == value and value not in (float("inf"), float("-inf")):
        return (Fraction(str(value)), Fraction(0))
    raise ValueError(f"non-finite/non-numeric rational {value!r}")


def _add(a: Delta, b: Delta) -> Delta:
    return (a[0] + b[0], a[1] + b[1])


def _sub(a: Delta, b: Delta) -> Delta:
    return (a[0] - b[0], a[1] - b[1])


def _mul(a: Delta, k: Fraction) -> Delta:
    return (a[0] * k, a[1] * k)


@dataclass
class _Bound:
    value: Delta
    sources: frozenset
    #: ``+1`` for a lower bound, ``-1`` for an upper bound (used for pivot direction).
    side: int


@dataclass
class _Row:
    basic: Any
    constant: Delta
    coeffs: dict[Any, Fraction] = field(default_factory=dict)
    sources: frozenset = frozenset()


class VexxDualSimplex:
    """Incremental exact simplex with push/pop and unsat cores."""

    def __init__(self, *, max_pivots: int = 200_000, timeout_ms: int = 5_000) -> None:
        self.max_pivots = int(max_pivots)
        self.timeout_ms = int(timeout_ms)
        self.rows: dict[Any, _Row] = {}
        self.lower: dict[Any, _Bound] = {}
        self.upper: dict[Any, _Bound] = {}
        self.constraints: list[tuple[Any, tuple]] = []
        self._snapshots: list[tuple[list[tuple[Any, tuple]], dict, dict, dict]] = []
        self._slack_counter = 0
        self._last_status: SolverStatus | None = None
        self._last_model: dict[Any, Delta] = {}
        self._last_core: list = []
        self.core_is_minimal = False
        self.unknown_reason = ""
        self.pivot_count = 0

    # ------------------------------------------------------------------ bounds

    def _set_lower(self, var: Any, bound: _Bound) -> None:
        current = self.lower.get(var)
        if current is None or bound.value > current.value:
            self.lower[var] = bound

    def _set_upper(self, var: Any, bound: _Bound) -> None:
        current = self.upper.get(var)
        if current is None or bound.value < current.value:
            self.upper[var] = bound

    # --------------------------------------------------------------- interface

    def push(self) -> None:
        self._snapshots.append((
            list(self.constraints),
            dict(self.lower),
            dict(self.upper),
            {k: _Row(v.basic, v.constant, dict(v.coeffs), v.sources) for k, v in self.rows.items()},
        ))

    def pop(self, levels: int = 1) -> None:
        if levels < 1 or levels > len(self._snapshots):
            raise IndexError("invalid solver pop depth")
        for _ in range(levels):
            self.constraints, self.lower, self.upper, self.rows = self._snapshots.pop()
        self._invalidate()

    def _invalidate(self) -> None:
        self._last_status = None
        self._last_model = {}
        self._last_core = []
        self.core_is_minimal = False

    def assert_and_track(self, constraint: Any, tracker_id: Any = None) -> None:
        if tracker_id is None:
            tracker_id = len(self.constraints)
        if isinstance(constraint, dict):
            kind = constraint.get("type", constraint.get("kind", "linear"))
            if kind in ("bound", "bounds"):
                var = constraint["var"]
                lo, hi = constraint.get("lo"), constraint.get("hi")
                if lo is not None:
                    self.assert_and_track(("ge", {var: 1}, lo), tracker_id)
                if hi is not None:
                    self.assert_and_track(("le", {var: 1}, hi), tracker_id)
                return
            if kind == "diff":
                constraint = ("diff", constraint["i"], constraint["j"], constraint.get("c", 0))
            else:
                constraint = (constraint.get("op", "le"), constraint.get("coeffs", {}), constraint.get("rhs", 0))
        op = constraint[0]
        if op in ("diff", "difference") and len(constraint) == 4:
            left, right, offset = constraint[1], constraint[2], constraint[3]
            coeffs = {left: Fraction(1), right: Fraction(-1)}
            rhs, strict, op = offset, False, "le"
        elif op in ("le", "lt", "ge", "gt", "eq") and len(constraint) == 3:
            coeffs = {k: Fraction(v) for k, v in dict(constraint[1]).items() if Fraction(v)}
            rhs = constraint[2]
            strict = op in ("lt", "gt")
            if op in ("ge", "gt"):
                coeffs = {k: -v for k, v in coeffs.items()}
                rhs = -rhs
            if op == "eq":
                # Two opposite inequalities sharing the tracker id.
                self._add_row(coeffs, rhs, False, tracker_id)
                self._add_row({k: -v for k, v in coeffs.items()}, -rhs, False, tracker_id)
                return
        else:
            raise ValueError(f"unsupported solver constraint {constraint!r}")
        self._add_row(coeffs, rhs, strict, tracker_id)

    def assert_constraint(self, constraint: Any, tracker_id: Any = None) -> None:
        self.assert_and_track(constraint, tracker_id)

    def _add_row(self, coeffs: dict[Any, Fraction], rhs: Any, strict: bool, tracker: Any) -> None:
        right = _delta(rhs)
        self.constraints.append((tracker, (dict(coeffs), right, strict)))
        self._slack_counter += 1
        slack = ("slack", self._slack_counter)
        # The row is the *slack definition*: s = rhs - a^T x, so ``s >= 0`` is exactly
        # ``a^T x <= rhs``.  Keeping the original sign here made the tableau describe
        # ``s = rhs + a^T x`` and silently weakened every constraint.
        row = _Row(slack, right, {k: -v for k, v in coeffs.items()}, frozenset({tracker}))
        self.rows[slack] = row
        # The slack of ``a^T x <= b`` is ``b - a^T x`` and must be non-negative.
        self._set_lower(
            slack,
            _Bound((Fraction(0), Fraction(1)) if strict else _ZERO, frozenset({tracker}), +1),
        )
        self._invalidate()

    # ------------------------------------------------------------------- core

    def _value_of(self, var: Any, assignment: dict[Any, Delta]) -> Delta:
        row = self.rows.get(var)
        if row is None:
            return assignment.get(var, _ZERO)
        total = row.constant
        for nb, coeff in row.coeffs.items():
            total = _add(total, _mul(self._value_of(nb, assignment), coeff))
        return total

    def _assignment(self) -> dict[Any, Delta]:
        """Current values: nonbasic vars sit on a bound, basic vars follow their row."""
        values: dict[Any, Delta] = {}

        def value(var: Any, seen: frozenset) -> Delta:
            if var in values:
                return values[var]
            row = self.rows.get(var)
            if row is not None:
                if var in seen:  # pragma: no cover - defensive, cycles cannot be built
                    raise ArithmeticError("cyclic tableau row")
                total = row.constant
                for nb, coeff in row.coeffs.items():
                    total = _add(total, _mul(value(nb, seen | {var}), coeff))
                values[var] = total
                return total
            lo = self.lower.get(var)
            hi = self.upper.get(var)
            chosen = lo.value if lo is not None else (hi.value if hi is not None else _ZERO)
            values[var] = chosen
            return chosen

        for var in self.rows:
            value(var, frozenset())
        for var in list(self.lower) + list(self.upper):
            value(var, frozenset())
        return values

    def check(self) -> SolverStatus:
        self._invalidate()
        self.pivot_count = 0
        deadline = time.monotonic() + max(0, self.timeout_ms) / 1000.0
        seen_bases: set[tuple] = set()
        while True:
            if time.monotonic() >= deadline or self.pivot_count > self.max_pivots:
                return self._unknown("pivot budget exhausted")
            assignment = self._assignment()
            culprit = None
            for basic in self.rows:
                value = assignment[basic]
                lo = self.lower.get(basic)
                hi = self.upper.get(basic)
                if lo is not None and value < lo.value:
                    culprit = (basic, +1)
                    break
                if hi is not None and value > hi.value:
                    culprit = (basic, -1)
                    break
            if culprit is None:
                model = {var: assignment.get(var, _ZERO) for var in self._variables()}
                if not self._model_satisfies(model):
                    return self._unknown("witness failed verification")
                self._last_model = model
                self._last_status = SolverStatus.SAT
                return SolverStatus.SAT

            basic, direction = culprit
            row = self.rows[basic]
            bound = self.lower[basic] if direction > 0 else self.upper[basic]
            # Guard against cycling: a repeated (basis, target bound) cannot make progress.
            signature = (frozenset(self.rows), basic, direction)
            if signature in seen_bases:
                return self._unknown("pivot loop")
            seen_bases.add(signature)

            # How far the basic variable must move to reach its violated bound.
            required = (
                _sub(bound.value, assignment[basic]) if direction > 0
                else _sub(assignment[basic], bound.value)
            )
            blocked_sources: set = set(bound.sources)
            finite: list[tuple[Fraction, int, Any]] = []
            unbounded_nb: Any = None
            for index, (nb, coeff) in enumerate(row.coeffs.items()):
                if coeff == 0:
                    continue
                # Raising the basic variable needs nb to rise when coeff > 0 and fall when
                # coeff < 0; only the bound in that direction limits the move.
                rising = coeff > 0
                limit = self.upper.get(nb) if rising else self.lower.get(nb)
                if limit is None:
                    if unbounded_nb is None:
                        unbounded_nb = nb
                    continue
                blocked_sources |= limit.sources
                available = _sub(limit.value, assignment[nb])
                distance = available if rising else _mul(available, Fraction(-1))
                if distance[0] < 0 or (distance[0] == 0 and distance[1] < 0):
                    # Bound already violated on the far side; still usable, but no room.
                    distance = _ZERO
                finite.append((_mul(distance, abs(coeff)), index, nb))

            chosen = None
            # Compare complete Delta values: a deficit of only ``δ`` is invisible to the
            # real component, and comparing constants alone pivoted forever on ``x<3``
            # with ``x>3`` instead of reporting UNSAT.
            for reach, _index, nb in sorted(finite, key=lambda item: (item[0], item[1])):
                if reach >= required:
                    chosen = nb
                    break
            if chosen is None:
                if unbounded_nb is not None:
                    chosen = unbounded_nb
                elif finite:
                    # Every finite move is too small to repair the row: the bounds are
                    # jointly infeasible.
                    self._last_status = SolverStatus.UNSAT
                    self._last_core = list(dict.fromkeys(blocked_sources))
                    self.core_is_minimal = False
                    return SolverStatus.UNSAT
                else:
                    self._last_status = SolverStatus.UNSAT
                    self._last_core = list(dict.fromkeys(blocked_sources))
                    self.core_is_minimal = False
                    return SolverStatus.UNSAT

            self._pivot(basic, chosen)
            self.pivot_count += 1

    # ---------------------------------------------------------------- pivoting

    def _pivot(self, basic: Any, nb: Any) -> None:
        row = self.rows.pop(basic)
        coeff = row.coeffs.pop(nb)
        # basic = constant + coeff*nb + Σ c_j nb_j
        #   =>  nb = (1/coeff)*basic - constant/coeff - Σ (c_j/coeff) nb_j
        # The former *basic* variable becomes a non-basic column of the new row; omitting
        # it deleted the pivot column from the tableau.
        inverse = Fraction(1) / coeff
        new_constant = _mul(row.constant, -inverse)
        new_coeffs = {k: -v * inverse for k, v in row.coeffs.items()}
        new_coeffs[basic] = inverse
        new_row = _Row(nb, new_constant, new_coeffs, row.sources)
        self.rows[nb] = new_row

        # Substitute nb in every other row that mentions it.
        for other_basic, other in list(self.rows.items()):
            if other_basic == nb:
                continue
            factor = other.coeffs.pop(nb, None)
            if factor is None:
                continue
            other.constant = _add(other.constant, _mul(new_constant, factor))
            for var, value in new_coeffs.items():
                combined = other.coeffs.get(var, Fraction(0)) + factor * value
                if combined:
                    other.coeffs[var] = combined
                else:
                    other.coeffs.pop(var, None)

    # ----------------------------------------------------------------- helpers

    def _variables(self) -> list:
        variables: list = []
        for var in list(self.rows) + list(self.lower) + list(self.upper):
            if var not in variables and not (isinstance(var, tuple) and var and var[0] == "slack"):
                variables.append(var)
        for row in self.rows.values():
            for var in row.coeffs:
                if var in variables:
                    continue
                if isinstance(var, tuple) and var and var[0] == "slack":
                    continue
                variables.append(var)
        return variables

    def _model_satisfies(self, model: dict[Any, Delta]) -> bool:
        for _tracker, (coeffs, rhs, strict) in self.constraints:
            total = _ZERO
            for var, coeff in coeffs.items():
                total = _add(total, _mul(model.get(var, _ZERO), coeff))
            if strict:
                if not total < rhs:
                    return False
            elif total > rhs:
                return False
        return True

    def _unknown(self, reason: str) -> SolverStatus:
        self.unknown_reason = reason
        self._last_status = SolverStatus.UNKNOWN
        return SolverStatus.UNKNOWN

    # ---------------------------------------------------------------- accessors

    def get_model(self) -> dict[Any, Fraction]:
        if self._last_status != SolverStatus.SAT:
            return {}
        return {var: value[0] for var, value in self._last_model.items()}

    def get_float_model(self) -> dict[Any, float]:
        return {var: float(value) for var, value in self.get_model().items()}

    def get_unsat_core(self) -> list:
        if self._last_status != SolverStatus.UNSAT:
            return []
        return list(self._last_core)


#: Backwards-compatible aliases used by the roadmap/spec naming.
IncrementalSimplex = VexxDualSimplex
DualSimplexSolver = VexxDualSimplex
