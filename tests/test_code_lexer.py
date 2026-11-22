"""Tier-1 lexer: chunk-boundary invariance and lexical fail-closed behaviour."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from vexx.code_engine.lexer import LexerState, advance_lexer_state

PREFIX = (
    "def f(n):\n"
    "    acc = []\n"
    "    for i in range(n):\n"
    "        acc.append(str(i))\n"
    '    text = f"value={n} and None"\n'
    "    return text, acc\n"
)

PIECES = [
    "x", " ", "\n", "(", ")", "[", "]", "{", "}", ",", ":", "=",
    '"', '\'"\'"\'"', 'f"', "'", "#c", "name", ".", "1", "_", "{",
]


def _relex(text: str) -> LexerState:
    return advance_lexer_state(LexerState(), text)


@pytest.mark.parametrize("piece", PIECES)
def test_incremental_matches_full_relex(piece: str):
    """Feeding a token at a time must equal re-lexing the whole prefix."""
    base = _relex(PREFIX)
    incremental = advance_lexer_state(base, piece)
    full = _relex(PREFIX + piece)
    assert incremental.valid == full.valid, piece
    if full.valid:
        assert incremental.stack == full.stack, piece
        assert incremental.in_str == full.in_str, piece
        assert incremental.fstring_stack == full.fstring_stack, piece


def test_triple_quote_opener_split_across_chunks():
    state = _relex("x = f")
    state = advance_lexer_state(state, '"')
    state = advance_lexer_state(state, '""')
    assert state.in_str == '"""'
    assert state.f_mode


def test_single_quote_then_triple_quote_stays_consistent():
    """``f"``, then ``""`` must become a triple-quoted f-string, not two strings."""
    whole = _relex('x = f"""a = {1}"""')
    split = advance_lexer_state(advance_lexer_state(_relex("x = f"), '"'), '""')
    split = advance_lexer_state(split, 'a = {1}"""')
    assert split.valid == whole.valid
    assert split.in_str == whole.in_str


@pytest.mark.parametrize(
    "text",
    [
        "x = (1 + 2",
        "x = [1, 2",
        "x = {1: 2",
        'x = f"{1',
        "x = 'unterminated",
    ],
)
def test_incomplete_constructs_stay_open(text: str):
    """An unterminated construct is *incomplete*, not invalid.

    The lexer must keep the frame/string open so generation can continue, and must not
    report a hard error — otherwise every partial line would dead-end the decode.
    """
    state = _relex(text)
    assert state.valid, text
    assert state.reason is None, text
    assert state.stack or state.in_str is not None or state.fstring_stack, text


@pytest.mark.parametrize(
    "text,reason",
    [
        ("x = 1)", "unmatched_close"),
        ("x = ]", "unmatched_close"),
        ("x = }", "unmatched_close"),
        ("x = (1]", "mismatched_close"),
        ('x = f"}"', "unmatched_close"),
    ],
)
def test_invalid_code_fails_closed(text: str, reason: str):
    state = _relex(text)
    assert not state.valid, text
    assert state.reason == reason, text


def test_balanced_delimiters_are_valid():
    state = _relex("x = [1, (2, 3), {4: 5}]")
    assert state.valid
    assert state.stack == []


def test_call_frame_records_callee_and_comma_count():
    state = _relex("res = compute(a, b")
    assert state.stack == [("(", "compute", 1)]
    state = advance_lexer_state(state, ", c")
    assert state.stack == [("(", "compute", 2)]


def test_closed_frame_reports_callee_on_close():
    state = advance_lexer_state(_relex("res = compute(1, 2"), ")")
    assert state.closed_frames == [("(", "compute", 1)]


def test_keyword_callee_is_not_recorded():
    """``if (x)`` is a keyword, not a call, so it must not become a contract check."""
    state = _relex("if (x")
    assert state.stack == [("(", None, 0)]


def test_comment_hides_delimiters():
    state = _relex("x = 1  # ( unbalanced\n")
    assert state.valid
    assert state.stack == []


def test_fstring_expression_accepts_nested_quotes():
    """PEP 701: a nested string literal inside a replacement field is valid.

    The Maya reference rejected this outright (``quote_in_fstring_expression``).
    """
    for text in (
        'x = f"{format_value(\'x\')}"',
        'x = f"{d["key"]}"',
        'x = f"{(lambda y: y) (\'a\')}"',
        'x = f"{func("a", \'b\')}"',
    ):
        state = _relex(text)
        assert state.valid, text
        assert state.reason is None, text


def test_fstring_nested_braces_and_dicts():
    state = _relex('x = f"{ {"a": 1} }"')
    assert state.valid
    state = _relex('x = f"{[i for i in range(3)]}"')
    assert state.valid


def test_fstring_stray_close_brace_fails_closed():
    # A close brace with no matching open at the top level is invalid.
    assert not _relex("x = }").valid
    # A close brace that does not match the innermost delimiter is invalid.
    assert not _relex('x = f"{1}" }').valid


def test_fstring_escaped_braces_are_literal():
    state = _relex('x = f"{{literal}}"')
    assert state.valid
    assert state.stack == []


def test_fstring_state_is_chunk_invariant():
    text = 'x = f"a{func(1, \'q\')}b{2}c"'
    whole = _relex(text)
    for cut in range(1, len(text)):
        incremental = advance_lexer_state(_relex(text[:cut]), text[cut:])
        assert incremental.valid == whole.valid, cut
        assert incremental.in_str == whole.in_str, cut


def test_raw_and_bytes_strings():
    assert _relex(r"x = r'\d+'").valid
    assert _relex("x = b'\\x00'").valid
    assert _relex("x = rb'\\d'").valid


def test_clone_drops_closed_frames():
    state = advance_lexer_state(_relex("f(1"), ")")
    assert state.closed_frames
    assert state.clone().closed_frames == []


def test_to_dict_is_serializable():
    payload = _relex("x = [1, 2").to_dict()
    assert payload["stack"] == [("[", None, 1)]
    assert payload["valid"] is True


def test_empty_chunk_is_a_noop():
    state = _relex("x = (1")
    assert advance_lexer_state(state, "").stack == state.stack