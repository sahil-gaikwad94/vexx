"""Incremental exact simplex: soundness, push/pop, and scaling past the FM cap."""

from __future__ import annotations

import random

from vexx.nano_solver import SolverStatus, VexxNanoSolver
from vexx.simplex import VexxDualSimplex


def _solve(engine, constraints):
    solver = engine()
    for index, constraint in enumerate(constraints):
        solver.assert_and_track(constraint, index)
    return solver.check(), solver


def test_simple_unsat_core():
    status, solver = _solve(
        VexxDualSimplex, [("le", {"x": 1}, 3), ("ge", {"x": 1}, 5)]
    )
    assert status == SolverStatus.UNSAT
    assert set(solver.get_unsat_core()) == {0, 1}


def test_equality_is_two_inequalities():
    status, _ = _solve(VexxDualSimplex, [("eq", {"x": 1}, 5), ("le", {"x": 1}, 3)])
    assert status == SolverStatus.UNSAT


def test_sat_witness_satisfies_every_row():
    constraints = [
        ("eq", {"x": 1, "y": 1}, 4),
        ("ge", {"x": 1}, 0),
        ("le", {"y": 1}, 4),
        ("diff", "x", "y", 0),
    ]
    status, solver = _solve(VexxDualSimplex, constraints)
    assert status == SolverStatus.SAT
    model = solver.get_model()
    assert model["x"] == model["y"] == 2


def test_strict_bounds_are_decided_natively():
    assert _solve(VexxDualSimplex, [("lt", {"x": 1}, 3), ("gt", {"x": 1}, 3)])[0] == SolverStatus.UNSAT
    assert _solve(VexxDualSimplex, [("lt", {"x": 1}, 3), ("gt", {"x": 1}, 1)])[0] == SolverStatus.SAT


def test_push_pop_restores_the_bound_stack():
    solver = VexxDualSimplex()
    solver.assert_and_track(("le", {"x": 1}, 3), "upper")
    solver.push()
    solver.assert_and_track(("ge", {"x": 1}, 5), "lower")
    assert solver.check() == SolverStatus.UNSAT
    solver.pop()
    assert solver.check() == SolverStatus.SAT


def test_it_scales_past_the_fourier_motzkin_cap():
    constraints = [("ge", {f"x{i}": 1}, i) for i in range(60)]
    constraints.append(("le", {f"x{i}": 1 for i in range(60)}, 10**6))
    # 60 variables exceeds the reference engine's 24-variable FM cap.
    assert _solve(VexxNanoSolver, constraints)[0] == SolverStatus.SAT
    constraints.append(("le", {"x59": 1}, 10))
    assert _solve(VexxNanoSolver, constraints)[0] == SolverStatus.UNSAT


def test_differential_agreement_with_fourier_motzkin():
    rng = random.Random(20261126)
    decided = 0
    for _ in range(300):
        variables = rng.randint(2, 4)
        constraints = []
        for _ in range(rng.randint(2, 6)):
            coeffs = {
                f"x{i}": rng.choice([-2, -1, 1, 2])
                for i in range(variables)
                if rng.random() < 0.7
            } or {"x0": 1}
            constraints.append(
                (rng.choice(["le", "ge", "eq"]), coeffs, rng.randint(-5, 5))
            )
        reference, _ = _solve(VexxNanoSolver, constraints)
        candidate, _ = _solve(VexxDualSimplex, constraints)
        if reference == SolverStatus.UNKNOWN:
            # Fourier–Motzkin can fail to *witness* a strict system; only cases it decided
            # are a valid cross-check.
            continue
        assert reference == candidate, constraints
        decided += 1
    assert decided > 250
