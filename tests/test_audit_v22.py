"""Regressions found by the second end-to-end audit (Python <-> Rust parity and robustness).

Every test here failed, crashed, or reported a divergence before the corresponding fix:

* the incremental JSON automaton accepted dead number prefixes (``7.E``) and overflowing
  literals (``90e9376``, ``[1e400]``) that ``parse_json`` rejects, so the masker could emit
  a token the strict parser then refused;
* ``Interval.empty().sub(...)`` and ``inf - inf`` raised ``ValueError`` from the constructor;
* the numeric-prefix scanner did not treat ``\\r`` as a JSON terminator and pruned a
  reachable underflow-to-zero completion (``1e`` -> ``1e-999``);
* the native mask/solver/contractor surfaces disagreed with the Python reference.
"""

from __future__ import annotations

import random

import pytest

from vexx import native
from vexx.interval import Hc4AffineContractor, Interval
from vexx.json_pda import (
    JSONPushdownAutomaton,
    JSONSyntaxError,
    TokenMasker,
    is_complete_json,
    parse_json,
)

HAS_NATIVE = native.available()


def _accepts(text: str) -> bool:
    machine = JSONPushdownAutomaton()
    return bool(machine.feed(text) and machine.finish() and machine.accepts())


def test_dead_number_prefixes_are_rejected():
    for prefix in ("7.E", "5.e", "1.e5", "-0.E", "0.e+1"):
        machine = JSONPushdownAutomaton()
        assert not machine.feed(prefix), f"{prefix!r} has no valid completion"
        assert not is_complete_json(prefix)


def test_dead_prefix_really_has_no_completion():
    # Regression guard for the *reason* the prefix is dead: no suffix can rescue it.
    vocab = ["E", "e", "1", "0", "]", " ", "5"]
    masker = TokenMasker(vocab)
    mask = masker.valid_mask("7.")
    assert mask[vocab.index("E")] is False
    assert mask[vocab.index("1")] is True  # "7.1" is still a number


def test_overflowing_numbers_are_rejected():
    for text in ("90e9376", "[90e9376]", "1e400", "9" * 400):
        assert not _accepts(text), text
        machine = JSONPushdownAutomaton()
        assert not machine.feed(text + " ") or not machine.finish()
        with pytest.raises(JSONSyntaxError):
            parse_json(text)
    # Overflowing digits inside the mantissa can still be rescued by a negative exponent.
    rescued = "9" * 400 + "e-400"
    assert _accepts(rescued)
    assert 0.0 < float(rescued) <= 1.0


def test_valid_numbers_are_unchanged():
    for text in ("0", "-0", "1", "-1", "2.5", "1e5", "1E-5", "-3e-2", "0.0001", "1e+2"):
        assert _accepts(text), text
        machine = JSONPushdownAutomaton()
        assert machine.feed(text) and machine.finish() and machine.accepts()


def test_interval_empty_and_infinite_operations_never_raise():
    empty = Interval.empty()
    entire = Interval.entire()
    point = Interval(1.0, 1.0)
    assert empty.sub(entire).is_empty()
    assert empty.add(entire).is_empty()
    assert point.sub(empty).is_empty()
    # inf - inf is indeterminate: widening to the whole line is sound and must not raise.
    infinite = Interval(float("inf"), float("inf"))
    widened = infinite.sub(infinite)
    assert widened.lo == float("-inf") and widened.hi == float("inf")
    # inf / inf likewise.
    assert infinite.div(Interval(2.0, float("inf"))).lo == float("-inf")


def test_numeric_prefix_treats_carriage_return_as_a_terminator():
    closed = Interval._from_numeric_prefix_python("342\r", Interval.entire(), False)
    assert closed == Interval(342.0, 342.0)
    # A number followed by CR and a delimiter is pinned to the literal, not a huge span.
    pinned = Interval._from_numeric_prefix_python("12\r, 3", Interval.entire(), False)
    assert pinned == Interval(12.0, 12.0)


def test_underflow_completion_is_reachable():
    zero = Interval._from_numeric_prefix_python("1e-", Interval(-0.5, 0.0), True)
    assert zero.lo <= 0.0 <= zero.hi and not zero.is_empty()
    # A positive exponent cannot underflow.
    assert Interval._from_numeric_prefix_python("1e+", Interval(-0.5, 0.0), True).is_empty()


@pytest.mark.skipif(not HAS_NATIVE, reason="native extension not installed")
def test_native_prefix_status_matches_python():
    random.seed(2026)
    alphabet = list('{}[]",:0123456789.eE+-truefalsn \t\n\r\\/u')
    samples = ["", "7.E", "5.e", "[90e9376", "1e400", "1e-", "-3e-2", "{\"a\": 1}"]
    for _ in range(3000):
        samples.append("".join(random.choice(alphabet) for _ in range(random.randint(0, 12))))
    for text in samples:
        machine = JSONPushdownAutomaton()
        if not machine.feed(text):
            expected = "invalid"
        else:
            probe = machine.clone()
            ok = probe.finish()
            expected = ("complete" if probe.accepts() else "incomplete") if ok else "invalid"
        assert native.prefix_status(text) == expected, repr(text)


@pytest.mark.skipif(not HAS_NATIVE, reason="native extension not installed")
def test_native_numeric_prefix_matches_python():
    samples = ["1e-", "1e+", "342\r", "12\r, 3", "0.3", "-0.5", "01", "1e5", "", "-", "+"]
    domains = [(-0.5, 0.0), (0.0, 1.0), (-10.0, 10.0), (5.0, 6.0)]
    for text in samples:
        for lo, hi in domains:
            for allow_exp in (False, True):
                expected = Interval._from_numeric_prefix_python(text, Interval(lo, hi), allow_exp)
                got = native.numeric_prefix(text, Interval(lo, hi), allow_exp)
                # Both engines may spell the empty set differently (canonical vs not).
                def key(iv):
                    return None if iv.is_empty() else (iv.lo, iv.hi)
                assert key(got) == key(expected), (text, lo, hi, allow_exp, expected, got)


@pytest.mark.skipif(not HAS_NATIVE, reason="native extension not installed")
def test_native_mask_matches_pure_python(monkeypatch):
    random.seed(7)
    vocab = ["", "{", "}", "[", "]", '"a"', ":", ",", " ", "1", "12", "true", "null", "0", "e", "-", "."]
    prefixes = ["", "{", '{"a"', '{"a":', '{"a":1', '{"a":1}', "[1", "[1]", "1", "1.", "1e",
                "tru", "true", '"a', " ", "[1,", "7.E", "1.e", "-", "-0", "0", "4e3", "3e98"]
    for eos in (len(vocab), 0):
        native_masker = TokenMasker(vocab, eos_token=eos)
        saved, saved_error = native._NATIVE, native._LOAD_ERROR
        native._NATIVE, native._LOAD_ERROR = None, "forced fallback"
        try:
            python_masker = TokenMasker(vocab, eos_token=eos)
            for prefix in prefixes:
                try:
                    expected = python_masker.valid_mask(prefix)
                except JSONSyntaxError as exc:
                    expected = ("raise", type(exc).__name__)
                try:
                    got = native_masker.valid_mask(prefix)
                except JSONSyntaxError as exc:
                    got = ("raise", type(exc).__name__)
                assert got == expected, (prefix, eos)
        finally:
            native._NATIVE, native._LOAD_ERROR = saved, saved_error


def test_symbolic_exponent_is_not_an_integer_conversion():
    # ``2**x`` used to raise TypeError from ``int(Symbol)`` inside the size guard.
    from vexx.cas import safe_sympy

    expression = safe_sympy("2**x")
    assert str(expression) == "2**x"


def test_contractor_refutes_inequality_only_conflict():
    # x >= 0 and 2x <= -1: refutable only after directional inverse narrowing.
    contractor = Hc4AffineContractor({0: Interval(-2.0, 1.0)})
    contractor.add_constraint(("linear", {0: 1.0}, 0.0, "ge"))
    contractor.add_constraint(("linear", {0: 2.0}, -1.0, "le"))
    assert contractor.propagate() is False
