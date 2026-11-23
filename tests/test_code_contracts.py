"""Tier-1 callable contracts: calling conventions, partial headers, call checking."""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from vexx.code_engine.contracts import (
    KEYWORD_ONLY,
    POS_ONLY,
    POS_OR_KEYWORD,
    UNKNOWN_DYNAMIC_DISPATCH,
    CallableContract,
    ParameterContract,
    build_contracts,
    check_call,
    contract_from_source,
    header_contract_from_text,
    signature_table_from_source,
)

SOURCE = textwrap.dedent(
    '''
    def plain(a, b):
        return a + b

    def star(a, *args):
        return a

    def double_star(a, **kwargs):
        return a

    def defaults(a, b=1, c=2):
        return a

    def kwonly(a, *, key, opt=1):
        return a

    def posonly(a, /, b):
        return a

    async def coro(a, timeout=1.0, *args):
        return a

    class C:
        def method(self, x):
            return x

        @staticmethod
        def helper(y):
            return y

        @classmethod
        def build(cls, z):
            return z
    '''
)


def _contract(name: str) -> CallableContract:
    table = build_contracts(SOURCE, "<test>")
    return table[name][-1]


# --------------------------------------------------------------- parameter kinds


def test_parameter_kinds():
    contract = _contract("kwonly")
    kinds = {p.name: p.kind for p in contract.parameters}
    assert kinds["a"] == POS_OR_KEYWORD
    assert kinds["key"] == KEYWORD_ONLY
    assert kinds["opt"] == KEYWORD_ONLY
    assert contract.required_keyword_names() == ["key"]


def test_positional_only_is_not_keyword():
    contract = _contract("posonly")
    assert contract.parameters[0].kind == POS_ONLY
    assert not contract.accepts_keyword("a")
    assert contract.accepts_keyword("b")


def test_var_positional_and_var_keyword_are_distinct():
    star = _contract("star")
    assert star.has_var_positional
    assert not star.has_var_keyword
    assert star.positional_bounds() == (1, None)

    double = _contract("double_star")
    assert double.has_var_keyword
    assert not double.has_var_positional
    # The v2 bug: **kwargs must not permit extra *positional* arguments.
    assert double.positional_bounds() == (1, 1)


def test_defaults_and_nested_annotations():
    table = build_contracts("def f(a: Dict[str, List[int]] = {}) -> None:\n    pass\n")
    contract = table["f"][-1]
    assert contract.parameters[0].annotation == "Dict[str, List[int]]"
    assert contract.return_annotation == "None"
    assert contract.parameters[0].required is False


def test_multiline_signature_parses():
    table = build_contracts("def f(\n    a,\n    b=2,\n):\n    return a\n")
    assert [p.name for p in table["f"][-1].parameters] == ["a", "b"]


def test_binding_and_self_removal():
    instance = _contract("C.method")
    assert instance.binding == "instance"
    assert [p.name for p in instance.parameters] == ["x"]

    static = _contract("C.helper")
    assert static.binding == "static"
    assert [p.name for p in static.parameters] == ["y"]

    cls = _contract("C.build")
    assert cls.binding == "class"
    assert [p.name for p in cls.parameters] == ["z"]

    free = _contract("plain")
    assert free.binding == "function"


def test_overloads_are_preserved_and_implementation_selected():
    source = textwrap.dedent(
        """
        from typing import overload

        @overload
        def f(x: int) -> int: ...

        @overload
        def f(x: str) -> str: ...

        def f(x):
            return x
        """
    )
    contracts = build_contracts(source)["f"]
    assert len(contracts) == 3
    assert sum(1 for c in contracts if c.overload_group is not None) == 2
    chosen = contract_from_source(source, "f")
    assert chosen is not None
    assert chosen.overload_group is None


def test_same_method_name_does_not_collapse():
    source = textwrap.dedent(
        """
        class A:
            def run(self, x):
                return x

        class B:
            def run(self, x, y=0):
                return x
        """
    )
    table = build_contracts(source)
    assert table["A.run"][-1].positional_bounds() == (1, 1)
    assert table["B.run"][-1].positional_bounds() == (1, 2)
    # The merged short-name view takes the widest bounds instead of one false signature.
    merged = signature_table_from_source(source)["run"]
    assert merged["min_arity"] == 1
    assert merged["max_arity"] == 2


def test_signature_table_uses_widest_bounds():
    table = signature_table_from_source(SOURCE)
    assert table["plain"]["max_arity"] == 2
    assert table["star"]["max_arity"] is None
    assert table["double_star"]["max_arity"] == 1
    assert table["double_star"]["has_var_keyword"] is True


# ------------------------------------------------------------------ call checking


def test_too_few_and_too_many_positional():
    contract = _contract("plain")
    assert check_call(contract, 1).status == "too_few_positional"
    assert check_call(contract, 2).passed
    assert check_call(contract, 3).status == "too_many_positional"


def test_keyword_only_requirements_are_separate_from_arity():
    contract = _contract("kwonly")
    # Arity is satisfied by ``a`` alone, but ``key`` is still required.
    missing = check_call(contract, 1)
    assert missing.status == "missing_keyword"
    assert missing.kind == "keyword"
    assert missing.missing_keyword == ("key",)
    assert check_call(contract, 1, ["key"]).passed


def test_star_args_allow_extra_positional():
    contract = _contract("star")
    assert check_call(contract, 9).passed
    assert check_call(contract, 0).status == "too_few_positional"


def test_double_star_is_not_positional_varargs():
    contract = _contract("double_star")
    assert check_call(contract, 3).status == "too_many_positional"
    assert check_call(contract, 1, ["anything", "else"]).passed


def test_defaults_make_positional_optional():
    contract = _contract("defaults")
    assert check_call(contract, 0).status == "too_few_positional"
    assert check_call(contract, 1).passed
    assert check_call(contract, 3).passed
    assert check_call(contract, 4).status == "too_many_positional"


def test_duplicate_argument_detected():
    duplicate = check_call(_contract("plain"), 2, ["a"])
    assert duplicate.status == "duplicate_argument"
    assert duplicate.kind == "keyword"
    assert duplicate.duplicate == ("a",)


def test_call_site_kwargs_suppresses_false_rejection():
    contract = _contract("plain")
    assert check_call(contract, 0, [], has_kwargs=True).passed
    assert check_call(contract, 5, [], has_starargs=True).passed


def test_unexpected_keyword_reported():
    result = check_call(_contract("plain"), 2, ["nope"])
    assert result.status == "unexpected_keyword"
    assert result.unexpected_keyword == ("nope",)


def test_unknown_contract_is_not_a_verdict():
    result = check_call(CallableContract.unknown("m::f", "dynamic"), 3)
    assert result.status == UNKNOWN_DYNAMIC_DISPATCH
    assert result.kind == "unknown"
    assert not result.passed


def test_required_keyword_only_not_satisfied_by_positional():
    """A required keyword-only argument cannot be supplied positionally."""
    contract = _contract("kwonly")
    assert check_call(contract, 2).status == "too_many_positional"
    assert check_call(contract, 1).status == "missing_keyword"


# ----------------------------------------------------------------- in-flight headers


def test_complete_header_is_partial_ast():
    contract = header_contract_from_text("def f(a: int, b: Dict[str, List[int]] = {}) -> None:\n    ")
    assert contract is not None
    assert contract.name == "f"
    assert contract.confidence == "partial_ast"
    assert [p.name for p in contract.parameters] == ["a", "b"]
    assert contract.positional_bounds() == (1, 2)


def test_multiline_incomplete_header():
    contract = header_contract_from_text('def g(\n    a: int,\n    b: "Tuple[int, ...]" = (),\n')
    assert contract is not None
    assert [p.name for p in contract.parameters] == ["a", "b"]


def test_unbalanced_header_recovers_partially_never_exactly():
    contract = header_contract_from_text("def broken(a, b: List[int")
    assert contract is not None
    assert contract.confidence == "partial_ast"
    assert [p.name for p in contract.parameters] == ["a", "b"]
    assert contract.positional_bounds() == (2, 2)


def test_no_signature_returns_none():
    assert header_contract_from_text("x = 1\ny = 2\n") is None


def test_partial_header_keeps_kwargs_semantics():
    contract = header_contract_from_text("def f(a, **kw):\n    pass\n")
    assert contract is not None
    assert contract.positional_bounds() == (1, 1)
    assert contract.has_var_keyword
    assert not contract.has_var_positional


# ------------------------------------------------------------------- validation


def test_parameter_kind_is_validated():
    with pytest.raises(ValueError):
        ParameterContract(name="a", kind="nonsense", required=True)


def test_binding_and_confidence_are_validated():
    with pytest.raises(ValueError):
        CallableContract(symbol_id="m::f", binding="nonsense")
    with pytest.raises(ValueError):
        CallableContract(symbol_id="m::f", confidence="nonsense")


def test_contract_roundtrips_through_dict():
    contract = _contract("kwonly")
    restored = CallableContract.from_dict(contract.to_dict())
    assert restored == contract


def test_symbol_table_entry_shape():
    entry = _contract("kwonly").to_symbol_table_entry()
    assert entry["min_arity"] == 1
    assert entry["max_arity"] == 1
    assert entry["required_keywords"] == ["key"]
    assert entry["has_var_keyword"] is False


def test_nested_function_is_visible():
    table = build_contracts("def outer():\n    def inner(a):\n        return a\n")
    assert "inner" in table