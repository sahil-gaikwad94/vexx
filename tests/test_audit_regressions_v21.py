"""Regression lock-down for the Vexx 2.1 principal engineering audit.

Every test here corresponds to a specific defect from the audit (BUG-01 … BUG-23) or to a
behaviour introduced by the 2.2 performance/production work.  The point of this file is
that these can never silently regress again.
"""

from __future__ import annotations

import http.client
import math

import numpy as np
import pytest
from pydantic import BaseModel

import vexx as vt
from vexx.calibration import (
    parse_unit_dimensions,
    verify_statistical_report,
)
from vexx.cartridges.quant import black_scholes_call, black_scholes_put, implied_volatility
from vexx.cartridges.sql import SqlCatalog, SqlVerifier
from vexx.code_engine.candidate import CandidateVerifier
from vexx.code_engine.config import CodeEngineConfig
from vexx.code_engine.controller import VexxCodeController
from vexx.code_engine.lexer import LexerState, advance_lexer_state
from vexx.code_engine.masking import (
    SAFE_BUILTIN_CALLABLES,
    CodeSession,
)
from vexx.code_engine.policy import CodePolicyError, check_ast_policy
from vexx.code_engine.sandbox import SandboxRunner
from vexx.interval import AffineForm, Interval
from vexx.mcp import UntrustedString, is_untainted, safe_format
from vexx.nano_solver import SolverStatus, VexxNanoSolver
from vexx.proof.certificate import ProofCertificate, build_certificate, verify_certificate
from vexx.rl.code_rewards import execution_reward
from vexx.serve import VexxServeServer, _open_upstream

# --------------------------------------------------------------------------- specs


class Portfolio(BaseModel):
    weights: dict[str, float]
    total: float


@vt.verified_schema()
def portfolio(p: Portfolio):
    vt.require(p.total == 1.0, tol=1e-9)


class Point(BaseModel):
    x: float
    y: float


@vt.verified_schema()
def point(p: Point):
    vt.require(p.x + p.y == 1.0, tol=1e-9)


CODE_VOCAB = {
    0: "def",
    1: " ",
    2: "f",
    3: "(",
    4: ")",
    5: ":",
    6: "\n",
    7: "    ",
    8: "return",
    9: "range",
    10: "len",
    11: "i",
    12: "1",
    13: ",",
    14: "for",
    15: "in",
    16: "open",
    17: '"',
    18: "x",
    19: "2",
    20: "3",
}


def _controller() -> VexxCodeController:
    return VexxCodeController(vocabulary=CODE_VOCAB)


def _mask(controller, prefix: str) -> np.ndarray:
    return controller.mask_logits(
        np.zeros(len(CODE_VOCAB)), prefix_text=prefix, vocab_map=CODE_VOCAB
    )


# ------------------------------------------------------------- BUG-01 solver equality


def test_equality_is_both_directions():
    solver = VexxNanoSolver()
    solver.assert_and_track(("eq", {"x": 1.0}, 5.0), "eq")
    solver.assert_and_track(("le", {"x": 1.0}, 3.0), "le")
    assert solver.check() == SolverStatus.UNSAT
    assert "eq" in solver.get_unsat_core()


def test_equality_witness_satisfies_the_equality():
    solver = VexxNanoSolver()
    solver.assert_and_track(("eq", {"x": 1.0, "y": 1.0}, 4.0), "eq")
    assert solver.check() == SolverStatus.SAT
    model = solver.get_float_model()
    assert math.isclose(model["x"] + model["y"], 4.0, abs_tol=1e-12)


# --------------------------------------------------------------- BUG-02 certificates


def test_certificate_with_dict_field_self_verifies():
    certificate = build_certificate(
        portfolio,
        {"weights": {"AAPL": 0.6, "MSFT": 0.4}, "total": 1.0},
    )
    # The witness keeps native containers; the SMT renderer and the live validator both
    # expect a dict, not a ``{"__pairs__": ...}`` tag.
    assert certificate.witness["weights"] == {"AAPL": 0.6, "MSFT": 0.4}
    verification = verify_certificate(certificate)
    assert verification.ok, verification.as_dict()
    assert verify_certificate(ProofCertificate.from_dict(certificate.to_dict())).ok


# -------------------------------------------------------------------- BUG-03 lexer


def test_callee_is_not_concatenated_across_whitespace():
    state = advance_lexer_state(LexerState(), "def f(x):\n    return helper(")
    assert state.stack[-1][1] == "helper"
    state = advance_lexer_state(LexerState(), "for i in range(")
    assert state.stack[-1][1] == "range"


def test_identifier_split_across_whitespace_is_not_a_string_prefix():
    state = advance_lexer_state(advance_lexer_state(LexerState(), "def f"), "(x):")
    assert state.pending == ""
    assert state.stack == []


# ------------------------------------------------------------------ BUG-04 masking


def test_builtin_calls_are_not_masked():
    controller = _controller()
    assert "range" in SAFE_BUILTIN_CALLABLES
    assert np.isfinite(_mask(controller, "for i in range")[3]), "range( was masked"
    assert np.isfinite(_mask(controller, "y = len")[3]), "len( was masked"


def test_forbidden_name_is_allowed_inside_a_string():
    controller = _controller()
    assert np.isfinite(_mask(controller, 'msg = "')[16]), 'open inside a string was masked'


# ------------------------------------------------------------------- BUG-05 policy


def test_allow_listed_module_is_not_a_forbidden_identifier():
    tree = check_ast_policy(
        "import heapq\n\ndef dijkstra(items):\n    heapq.heappush(items, 1)\n    return items\n"
    )
    assert tree is not None


@pytest.mark.parametrize(
    "source",
    [
        'f = "{0.__class__.__mro__}".format\nf(1)\n',
        'import string\nstring.Formatter().get_field("0.__class__", [1], {})\n',
    ],
)
def test_template_introspection_is_blocked(source):
    with pytest.raises(CodePolicyError):
        check_ast_policy(source)


# -------------------------------------------------------- BUG-06 sandbox contract


def test_safe_execute_python_sees_real_return_values():
    passed, total, errors = VexxCodeController.safe_execute_python(
        "def add(a, b):\n    return a + b", "add", [{"input": [2, 3], "expected": 5}]
    )
    assert (passed, total) == (1, 1), errors


def test_fuzzer_distinguishes_opaque_values():
    verifier = CandidateVerifier(CodeEngineConfig())
    report = verifier.fuzz_against_oracle(
        "def f():\n    return object()", "def f():\n    return None", "f", inputs=[[]]
    )
    assert report["agrees"] is False
    assert report["mismatches"]


def test_fuzzer_records_real_mismatch_values():
    verifier = CandidateVerifier(CodeEngineConfig())
    report = verifier.fuzz_against_oracle(
        "def f(x):\n    return x + 1", "def f(x):\n    return x + 2", "f", inputs=[[1]]
    )
    assert report["mismatches"][0]["expected"] == 3
    assert report["mismatches"][0]["actual"] == 2


def test_experiment_cell_reports_its_result():
    result = SandboxRunner().run_experiment("value = 1 + 1\nRESULT = value")
    assert result.ok and result.repr == "2"


# ------------------------------------------------------------------ BUG-07 units


def test_voltage_is_a_joule_per_coulomb():
    assert parse_unit_dimensions("V") == parse_unit_dimensions("J/C")
    assert parse_unit_dimensions("V") == {"M": 1, "L": 2, "T": -3, "I": -1}


# ------------------------------------------------- BUG-08 numeric prefix parity


@pytest.mark.parametrize("prefix", ["00", "-00", "01", "01.5", "inf", "NaN", "1e400"])
def test_dead_numeric_prefixes_are_empty(prefix):
    assert Interval.from_numeric_prefix(prefix, Interval(0, 10), allow_exp=True).is_empty()


def test_partial_exponent_is_reachable():
    assert not Interval.from_numeric_prefix("1e", Interval(0, 1000), allow_exp=True).is_empty()
    assert not Interval.from_numeric_prefix("1e1", Interval(1e9, 1e11), allow_exp=True).is_empty()
    assert Interval.from_numeric_prefix("1e2", Interval(1e9, 1e11), allow_exp=True).is_empty()


# ------------------------------------------------------------- BUG-09 sandbox limits


def test_configured_resource_limits_are_verified():
    runner = SandboxRunner(cpu_seconds=1, memory_bytes=256 * 1024 * 1024, max_files=32)
    result = runner.run("def f(x):\n    return x", "f", [1])
    assert result.ok
    assert result.limits_enforced["cpu"] is True
    assert result.limits_enforced["memory"] is True
    assert result.limits_enforced["nofile"] is True


def test_runaway_output_is_killed_not_buffered():
    runner = SandboxRunner(wall_timeout_s=5.0, output_bytes=4096)
    result = runner.run(
        "def f():\n    for _ in range(10 ** 6):\n        print('x' * 4096)\n    return 1", "f", []
    )
    assert not result.ok
    assert result.error and ("OutputLimit" in result.error or "Timeout" in result.error)


def test_infinite_loop_times_out():
    runner = SandboxRunner(wall_timeout_s=1.0)
    result = runner.run("def f():\n    while True:\n        pass", "f", [])
    assert not result.ok and "Timeout" in (result.error or "")


# ------------------------------------------------------- BUG-10 affine / interval


def test_affine_multiplication_width_is_not_double_counted():
    left = AffineForm.from_interval(Interval(1.0, 3.0), 1)
    right = AffineForm.from_interval(Interval(1.0, 3.0), 2)
    product = left.mul(right).to_interval()
    assert math.isclose(product.hi - product.lo, 10.0, abs_tol=1e-9)


# ----------------------------------------------------------------- BUG-11 options


def test_black_scholes_degenerate_inputs():
    assert black_scholes_call(0.0, 100.0, 0.05, 0.2, 1.0) == 0.0
    assert black_scholes_call(100.0, 0.0, 0.05, 0.2, 1.0) == 100.0
    assert black_scholes_put(0.0, 100.0, 0.05, 0.2, 1.0) == pytest.approx(100.0 * math.exp(-0.05))
    assert black_scholes_put(100.0, 0.0, 0.05, 0.2, 1.0) == 0.0


def test_implied_volatility_round_trips_a_deep_itm_put():
    price = black_scholes_put(50.0, 100.0, 0.10, 0.05, 2.0)
    recovered = implied_volatility(price, 50.0, 100.0, 0.10, 2.0, is_call=False)
    assert recovered is not None
    assert recovered == pytest.approx(0.05, abs=1e-3)


# ------------------------------------------------------------- BUG-12 statistics


def test_zero_p_value_is_a_conflict_not_a_crash():
    diagnostic = verify_statistical_report("", [{"kind": "z_test", "statistic": 2.0, "p_value": 0.0}])
    assert diagnostic is not None and diagnostic.verdict == "CONFLICT"


def test_inflated_t_test_p_value_is_flagged():
    diagnostic = verify_statistical_report(
        "", [{"kind": "t_test", "statistic": 5.0, "df": 50, "p_value": 0.85}]
    )
    assert diagnostic is not None and diagnostic.verdict == "CONFLICT"


# ------------------------------------------------------------------ BUG-13 serve


def test_https_upstreams_use_an_https_connection():
    connection, base_path = _open_upstream("https://api.openai.com/v1")
    try:
        assert isinstance(connection, http.client.HTTPSConnection)
        assert base_path == "/v1"
    finally:
        connection.close()
    plain, _ = _open_upstream("127.0.0.1:11434/v1")
    try:
        assert isinstance(plain, http.client.HTTPConnection)
        assert not isinstance(plain, http.client.HTTPSConnection)
    finally:
        plain.close()


def test_serve_is_threaded():
    import http.server

    assert issubclass(VexxServeServer, http.server.ThreadingHTTPServer)


# -------------------------------------------------------------- BUG-14 HF bridge


def test_hf_processor_preserves_the_numpy_container():
    from vexx.bridges.code_hf import VexxCodeLogitsProcessor

    controller = _controller()
    processor = VexxCodeLogitsProcessor(controller, CODE_VOCAB)
    scores = np.zeros((1, len(CODE_VOCAB)), dtype=np.float32)
    out = processor(np.array([[0, 1]]), scores)
    assert isinstance(out, np.ndarray) and out.dtype == np.float32
    assert processor.tokenizer_hash


# ------------------------------------------------------------------ BUG-15 rewards


def test_policy_violation_gets_the_syntax_penalty():
    reward, reason = execution_reward(
        {"syntax_ok": True, "policy_ok": False, "passed": 0, "total": 2}
    )
    assert (reward, reason) == (-0.5, "syntax_or_policy")


# -------------------------------------------------------------------- BUG-16 taint


@pytest.mark.parametrize("method", ["split", "rsplit", "splitlines"])
def test_string_splitting_keeps_taint(method):
    untrusted = UntrustedString("a; b", ["email"])
    pieces = getattr(untrusted, method)(";") if method != "splitlines" else getattr(untrusted, method)()
    assert all(isinstance(piece, UntrustedString) for piece in pieces)
    assert not any(is_untainted(piece) for piece in pieces)


def test_casefold_keeps_taint():
    untrusted = UntrustedString("ABC", ["web"])
    assert isinstance(untrusted.casefold(), UntrustedString)


def test_safe_format_of_clean_inputs_is_not_tainted():
    result = safe_format("hello {}", "world")
    assert not isinstance(result, UntrustedString)
    assert is_untainted(result)


# ------------------------------------------------------------------ BUG-17 masking


def test_nested_call_argument_counting_uses_the_outer_frame():
    controller = _controller()
    vocab = {0: "foo", 1: "(", 2: ")", 3: ",", 4: "1", 5: "2", 6: "3", 7: "\n", 8: " ", 19: "def", 20: ":"}
    controller = VexxCodeController(vocabulary=vocab)
    width = max(vocab) + 1
    prefix = "def foo(a, b):\n    return a\nfoo(1, 2"
    masked = controller.mask_logits(np.zeros(width), prefix_text=prefix, vocab_map=vocab)
    assert np.isfinite(masked[2]), "foo(1, 2) must be allowed"
    too_many = controller.mask_logits(
        np.zeros(width), prefix_text="def foo(a, b):\n    return a\nfoo(1, 2, 3", vocab_map=vocab
    )
    assert not np.isfinite(too_many[2]), "foo(1, 2, 3) must be masked"


def test_tokenizer_hash_is_used_for_the_vocab_plan():
    controller = _controller()
    session = CodeSession()
    first = controller.mask_logits(
        np.zeros(len(CODE_VOCAB)), session=session, prefix_text="x = 1", vocab_map=CODE_VOCAB,
        tokenizer_hash="abc123",
    )
    second = controller.mask_logits(
        np.zeros(len(CODE_VOCAB)), session=session, prefix_text="x = 1", vocab_map=CODE_VOCAB,
        tokenizer_hash="abc123",
    )
    assert np.array_equal(first, second)


# ---------------------------------------------------------------- BUG-18 f-strings


def test_nested_fstring_with_mismatched_delimiter_is_invalid():
    state = advance_lexer_state(LexerState(), "x = f\"{f'{(1 + 2]}'}\"")
    assert state.valid is False


def test_nested_fstring_is_valid():
    assert advance_lexer_state(LexerState(), "x = f\"{f'{1}'}\"").valid is True


def test_raw_string_rejects_a_literal_newline():
    assert advance_lexer_state(LexerState(), "x = r'a\nb'").valid is False


def test_raw_string_escaped_quote_stays_open():
    assert advance_lexer_state(LexerState(), "x = r'it\\'s'").valid is True


# --------------------------------------------------------------------- BUG-19 CAS


def test_intermediate_exponentiation_blowup_is_rejected():
    from vexx.cas import safe_sympy

    with pytest.raises(ValueError):
        safe_sympy("(2 ** 10000) ** 10000", {})


# ------------------------------------------------------------- BUG-20 signatures


def test_tampered_signature_returns_false():
    certificate = build_certificate(point, {"x": 0.25, "y": 0.75})
    certificate.metadata["signature"] = "not-valid-hex!!"
    certificate.metadata["signature_algorithm"] = "Ed25519"
    public_key, _private = ProofCertificate.generate_keypair()
    assert certificate.verify_signature(public_key) is False


# ---------------------------------------------------------------------- BUG-21 SQL


def test_parenthesised_tenant_predicate_is_found():
    verifier = SqlVerifier(
        SqlCatalog(tables={"orders": ["id", "tenant_id"]}, require_tenant=True)
    )
    result = verifier.verify("SELECT id FROM orders WHERE (tenant_id = 1) AND id > 0")
    assert result.ok, result.errors
    assert result.has_tenant_filter


def test_single_tenant_predicate_still_verifies_a_join():
    catalog = SqlCatalog(
        tables={"users": ["id", "tenant_id"], "orders": ["id", "tenant_id", "user_id"]},
        require_tenant=True,
    )
    result = SqlVerifier(catalog).verify(
        "SELECT u.id FROM users u JOIN orders o ON u.id = o.user_id WHERE u.tenant_id = 1"
    )
    assert result.ok, result.errors


def test_opt_in_join_hardening_requires_every_table():
    catalog = SqlCatalog(
        tables={"users": ["id", "tenant_id"], "orders": ["id", "tenant_id", "user_id"]},
        require_tenant=True,
        require_tenant_on_joins=True,
    )
    result = SqlVerifier(catalog).verify(
        "SELECT u.id FROM users u JOIN orders o ON u.id = o.user_id WHERE u.tenant_id = 1"
    )
    assert not result.ok


def test_legacy_tenant_predicate_helper_uses_the_ast():
    verifier = SqlVerifier({"orders": ["id", "tenant_id"]})
    assert verifier.tenant_isolation_ok("WHERE tenant_id = x")
    assert not verifier.tenant_isolation_ok("WHERE note = 'tenant_id'")


# ----------------------------------------------------------------- BUG-22 control


def test_crane_phase_uses_the_marker_state_machine():
    from vexx.code_engine.protocol import ProtocolConfig

    config = ProtocolConfig()
    assert VexxCodeController.crane_phase("") == "STRICT_VERIFIED"
    assert VexxCodeController.crane_phase(config.think_start + "plot") == "UNCONSTRAINED_THINK"
    assert VexxCodeController.crane_phase(config.experiment_start + "x = 1") == "EXPERIMENT_DESIGN"


def test_controller_policy_delegates_to_the_policy_module():
    with pytest.raises(PermissionError):
        VexxCodeController._check_ast_policy('f = "{0.__class__}".format\nf(1)')
