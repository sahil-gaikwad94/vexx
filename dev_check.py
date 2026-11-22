"""Temporary harness: exercise modules written so far without the package __init__."""

from __future__ import annotations

import importlib.util
import sys

MODULES = {
    "vexx.code_engine.config": "python/vexx/code_engine/config.py",
    "vexx.code_engine.contracts": "python/vexx/code_engine/contracts.py",
    "vexx.code_engine.lexer": "python/vexx/code_engine/lexer.py",
    "vexx.code_engine.policy": "python/vexx/code_engine/policy.py",
    "vexx.code_engine.protocol": "python/vexx/code_engine/protocol.py",
}


def load():
    for name, path in MODULES.items():
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules


def main() -> None:
    mods = load()
    ProtocolTracker = mods["vexx.code_engine.protocol"].ProtocolTracker
    ProtocolPhase = mods["vexx.code_engine.protocol"].ProtocolPhase
    CodePolicy = mods["vexx.code_engine.policy"].CodePolicy
    check_ast_policy = mods["vexx.code_engine.policy"].check_ast_policy
    CodePolicyError = mods["vexx.code_engine.policy"].CodePolicyError
    CodeEngineConfig = mods["vexx.code_engine.config"].CodeEngineConfig

    # ---- protocol
    cfg = mods["vexx.code_engine.protocol"].ProtocolConfig()
    tracker = ProtocolTracker(cfg, authorized_starts=frozenset(cfg.markers))
    settled, transitions = tracker.feed("<think>plan the ")
    print("phase after think:", tracker.phase, "settled:", repr(settled), "transitions:", len(transitions))
    settled, transitions = tracker.feed("solution")
    print("settled in reasoning:", repr(settled))
    settled, _ = tracker.feed("</think>def f():")
    print("phase:", tracker.phase, "settled:", repr(settled))

    # marker split across every boundary
    text = "<think>a</think><emit>b</emit>"
    for cut in range(1, len(text)):
        t = ProtocolTracker(cfg, authorized_starts=frozenset(cfg.markers))
        t.feed(text[:cut])
        t.feed(text[cut:])
        assert t.phase is ProtocolPhase.STRICT_EMIT, (cut, t.phase)
        assert not t.errors, (cut, t.errors)
    print("marker split at every boundary: OK")

    # chunk invariance of settled text
    whole = ProtocolTracker(cfg, authorized_starts=frozenset(cfg.markers))
    whole_settled, _ = whole.feed(text)
    for cut in range(1, len(text)):
        t = ProtocolTracker(cfg, authorized_starts=frozenset(cfg.markers))
        out = t.feed(text[:cut])[0] + t.feed(text[cut:])[0] + t.flush()
        assert out == whole_settled, (cut, out, whole_settled)
    print("settled text chunk-invariant: OK")

    # nested experiment: a *different* span opened inside an open one
    t = ProtocolTracker(cfg, authorized_starts=frozenset(cfg.markers))
    t.feed("<experiment>a<think>b")
    print("nested span errors:", t.errors)
    assert any("nested" in e for e in t.errors)

    # duplicate open of the same marker
    t = ProtocolTracker(cfg, authorized_starts=frozenset(cfg.markers))
    t.feed("<experiment>a<experiment>b")
    print("duplicate open errors:", t.errors)
    assert any("duplicate" in e for e in t.errors)

    # empty cell is legal (the body is simply empty)
    t = ProtocolTracker(cfg, authorized_starts=frozenset(cfg.markers))
    t.feed("<experiment></experiment>")
    print("empty cell errors:", t.errors, "body:", repr(t.experiment))
    assert not t.errors

    # unterminated cell is *incomplete*, not an error
    t = ProtocolTracker(cfg, authorized_starts=frozenset(cfg.markers))
    t.feed("<experiment>x = 1")
    print("unterminated cell errors:", t.errors, "phase:", t.phase)
    assert not t.errors and t.phase is ProtocolPhase.EXPERIMENT

    # a prompt-provided open marker must not be re-detected as model output
    t = ProtocolTracker(cfg, authorized_starts=frozenset(cfg.markers))
    settled, _ = t.feed("<think>already open")
    print("prompt-provided open, settled:", repr(settled), "phase:", t.phase)
    assert t.phase is ProtocolPhase.REASONING

    # close before open
    t = ProtocolTracker(cfg, authorized_starts=frozenset(cfg.markers))
    t.feed("</think>")
    print("close-before-open errors:", t.errors)
    assert t.errors

    # duplicate open
    t = ProtocolTracker(cfg, authorized_starts=frozenset(cfg.markers))
    t.feed("<think>a<think>b")
    print("duplicate open errors:", t.errors)
    assert t.errors

    # unauthorized
    t = ProtocolTracker(cfg)
    t.feed("<experiment>x")
    print("unauthorized errors:", t.errors)
    assert t.errors

    # experiment phase
    t = ProtocolTracker(cfg, authorized_starts=frozenset(cfg.markers))
    t.feed("<experiment>RESULT = [i*i for i in range(3)]</experiment>")
    print("experiment phase:", t.phase, "experiment text:", repr(t.experiment))

    # ---- policy
    check_ast_policy("import math\ndef f(x):\n    return math.sqrt(x)\n", CodePolicy())
    print("allowed import: OK")
    for bad, label in [
        ("import os\n", "import os"),
        ("from os import path\n", "from os import"),
        ("from . import x\n", "relative"),
        ("import ._p\n", "private relative"),
        ("def f():\n    global x\n", "global"),
        ("x = open('f')\n", "open"),
        ("x = ().__class__\n", "dunder"),
        ("def f():\n    x = 1\n", "ok-but-boring"),
    ]:
        try:
            check_ast_policy(bad, CodePolicy())
            print(f"  {label}: ACCEPTED")
        except CodePolicyError as exc:
            print(f"  {label}: blocked -> {exc}")
        except SyntaxError as exc:
            print(f"  {label}: syntax error -> {exc}")

    # nonlocal allowed
    check_ast_policy("def f():\n    x = 0\n    def g():\n        nonlocal x\n        x += 1\n", CodePolicy())
    print("nonlocal allowed: OK")

    # ---- config validation
    for kwargs, label in [
        ({"mode": "ruby"}, "bad mode"),
        ({"unknown_call_policy": "nope"}, "bad policy"),
        ({"max_exponent": 0}, "zero exponent"),
        ({"sandbox_wall_timeout_s": -1}, "negative timeout"),
        ({"allowed_imports": ("_private",)}, "private import"),
        ({"allowed_imports": "math"}, "string imports"),
        ({"max_candidate_count": True}, "bool as int"),
    ]:
        try:
            CodeEngineConfig(**kwargs)
            print(f"  config {label}: ACCEPTED (BAD)")
        except (ValueError, TypeError) as exc:
            print(f"  config {label}: rejected -> {type(exc).__name__}")
    print("config defaults:", CodeEngineConfig().describe_limits())


if __name__ == "__main__":
    main()