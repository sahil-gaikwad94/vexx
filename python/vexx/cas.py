"""Safe symbolic CAS helpers for Vexx 2.1.

SymPy is an *optional* dependency: ``import vexx`` works without it, and CAS
features raise :class:`OptionalDependencyError` when the extra is missing.
"""

from __future__ import annotations

import ast
import math
from fractions import Fraction
from typing import Any

_MAX_EXPR_CHARS = 512
_MAX_EXPR_NODES = 200
_MAX_EXPONENT = 10_000
_MAX_INT_BITS = 4096
_MAX_FACTORIAL_ARG = 5000
_MAX_BINOMIAL_N = 100_000


class OptionalDependencyError(ImportError):
    def __init__(self, extra: str = "cas") -> None:
        super().__init__(f"this feature requires the '{extra}' extra: pip install 'vexx-ai[{extra}]'")


def _sympy():
    try:
        import sympy as sp
    except ImportError as exc:
        raise OptionalDependencyError("cas") from exc
    return sp


def _bounded_factorial(value: Any) -> Any:
    sp = _sympy()
    number = sp.sympify(value)
    if number.is_number:
        try:
            iv = int(number)
        except (TypeError, ValueError):
            pass
        else:
            if abs(iv) > _MAX_FACTORIAL_ARG:
                raise ValueError(f"factorial argument above {_MAX_FACTORIAL_ARG}")
    return sp.factorial(number)


def _bounded_binomial(n: Any, k: Any) -> Any:
    sp = _sympy()
    nn, kk = sp.sympify(n), sp.sympify(k)
    for value in (nn, kk):
        if value.is_number:
            try:
                iv = int(value)
            except (TypeError, ValueError):
                continue
            if abs(iv) > _MAX_BINOMIAL_N:
                raise ValueError(f"binomial argument above {_MAX_BINOMIAL_N}")
    return sp.binomial(nn, kk)


_ALLOWED_FUNCS = None  # built lazily


def _allowed_funcs() -> dict[str, Any]:
    global _ALLOWED_FUNCS
    if _ALLOWED_FUNCS is not None:
        return _ALLOWED_FUNCS
    sp = _sympy()
    _ALLOWED_FUNCS = {
        "sqrt": sp.sqrt, "exp": sp.exp, "log": sp.log, "sin": sp.sin,
        "cos": sp.cos, "tan": sp.tan, "abs": sp.Abs, "Abs": sp.Abs,
        "gcd": sp.gcd, "lcm": sp.lcm,
        "binomial": _bounded_binomial, "comb": _bounded_binomial,
        "factorial": _bounded_factorial, "isprime": sp.isprime,
        "totient": sp.totient, "Mod": sp.Mod, "floor": sp.floor,
        "ceil": sp.ceiling, "ceiling": sp.ceiling, "Rational": sp.Rational,
        "sign": sp.sign, "Min": sp.Min, "Max": sp.Max,
    }
    return _ALLOWED_FUNCS


_ALLOWED_BINOPS = None


def _allowed_binops() -> dict[type, Any]:
    global _ALLOWED_BINOPS
    if _ALLOWED_BINOPS is not None:
        return _ALLOWED_BINOPS
    sp = _sympy()
    _ALLOWED_BINOPS = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b,
        ast.Pow: lambda a, b: a ** b,
        ast.Mod: lambda a, b: a % b,
        ast.FloorDiv: lambda a, b: a // b,
    }
    return _ALLOWED_BINOPS


def safe_sympy(expr: str, env: dict[str, Any] | None = None) -> Any:
    """Parse an arithmetic expression into SymPy via a restricted AST builder."""
    sp = _sympy()
    if not isinstance(expr, str) or not expr.strip():
        raise ValueError("empty expression")
    expr = expr.strip()
    if len(expr) > _MAX_EXPR_CHARS:
        raise ValueError(f"expression longer than {_MAX_EXPR_CHARS} characters")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"invalid expression syntax: {exc.msg}") from exc
    if sum(1 for _ in ast.walk(tree)) > _MAX_EXPR_NODES:
        raise ValueError("expression has too many AST nodes")

    env = dict(env or {})
    # Validate env keys/values.
    validated_env: dict[str, Any] = {}
    for key, value in env.items():
        if not isinstance(key, str) or not key.isidentifier() or key.startswith("_"):
            raise ValueError(f"invalid environment key {key!r}")
        if isinstance(value, sp.Basic):
            validated_env[key] = value
        elif isinstance(value, bool):
            raise ValueError("bool values are not allowed in CAS environment")
        elif isinstance(value, int):
            if abs(value).bit_length() > _MAX_INT_BITS:
                raise ValueError("environment integer too large")
            validated_env[key] = sp.Integer(value)
        elif isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError("non-finite float in environment")
            validated_env[key] = sp.Rational(str(value))
        elif isinstance(value, Fraction):
            validated_env[key] = sp.Rational(value.numerator, value.denominator)
        elif isinstance(value, str):
            validated_env[key] = safe_sympy(value, {})
        else:
            raise ValueError(f"unsupported environment value for {key!r}")

    funcs = _allowed_funcs()
    binops = _allowed_binops()

    def build(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return build(node.body)
        if isinstance(node, ast.Constant):
            value = node.value
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"unsupported literal {value!r}")
            if isinstance(value, int):
                if abs(value).bit_length() > _MAX_INT_BITS:
                    raise ValueError("integer literal too large")
                return sp.Integer(value)
            if not math.isfinite(value):
                raise ValueError("non-finite float literal")
            return sp.Rational(str(value))
        if isinstance(node, ast.Name):
            if node.id.startswith("_") or not node.id.isidentifier():
                raise ValueError(f"unsupported name {node.id!r}")
            if node.id in validated_env:
                return validated_env[node.id]
            if node.id == "pi":
                return sp.pi
            if node.id == "E":
                return sp.E
            return sp.Symbol(node.id)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = build(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and type(node.op) in binops:
            left, right = build(node.left), build(node.right)
            if isinstance(node.op, ast.Pow):
                if right.is_number:
                    try:
                        rv = float(right)
                    except (TypeError, ValueError):
                        rv = None
                    if rv is not None and abs(rv) > _MAX_EXPONENT:
                        raise ValueError("exponent too large")
                if isinstance(right, sp.Integer) and abs(int(right)) > _MAX_EXPONENT:
                    raise ValueError("exponent too large")
            result = binops[type(node.op)](left, right)
            # Reject zoo/oo/nan numerical results.
            if result.is_number and (result == sp.zoo or result == sp.oo or result == -sp.oo or result == sp.nan):
                raise ValueError("non-finite result")
            return result
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in funcs:
                raise ValueError("unsupported function call")
            if node.keywords:
                raise ValueError("keyword arguments are not supported")
            # Bound variadic arg counts.
            if node.func.id in ("Min", "Max", "gcd", "lcm"):
                if len(node.args) > 64:
                    raise ValueError("too many arguments")
            args = [build(a) for a in node.args]
            return funcs[node.func.id](*args)
        raise ValueError(f"disallowed syntax: {type(node).__name__}")

    result = build(tree)
    if isinstance(result, bool):
        return sp.true if result else sp.false
    return result


def crane_cas_evaluate(expr_str: str, env: dict[str, Any] | None = None) -> str:
    """Evaluate an expression exactly; return a string result or a typed status."""
    sp = _sympy()
    env = dict(env or {})
    clean = (
        (expr_str or "").strip()
        .replace("^", "**")
        .replace("×", "*")
        .replace("÷", "/")
        .replace("−", "-")
    )
    try:
        local_syms: dict[str, Any] = {}
        for key, value in env.items():
            if isinstance(value, sp.Basic):
                local_syms[key] = value
            elif isinstance(value, (int, float, Fraction)):
                if isinstance(value, bool):
                    raise ValueError("bool not allowed")
                local_syms[key] = sp.Rational(str(value)) if not isinstance(value, int) else sp.Integer(value)
            elif isinstance(value, str):
                local_syms[key] = safe_sympy(value, {})
            else:
                raise ValueError(f"unsupported environment value for {key!r}")
        parsed = safe_sympy(clean, local_syms)
        if parsed.has(sp.zoo, sp.oo, -sp.oo, sp.nan):
            return "DIV_BY_ZERO"
        simplified = sp.simplify(parsed)
        if simplified.has(sp.zoo, sp.oo, -sp.oo, sp.nan):
            return "DIV_BY_ZERO"
        if isinstance(simplified, bool) or simplified in (sp.true, sp.false):
            return "True" if simplified == sp.true else "False"
        if simplified.is_Rational:
            frac = Fraction(int(simplified.p), int(simplified.q))
            return str(frac.numerator) if frac.denominator == 1 else f"{frac.numerator}/{frac.denominator}"
        return str(simplified)
    except ZeroDivisionError:
        return "DIV_BY_ZERO"
    except OptionalDependencyError:
        raise
    except Exception as exc:
        msg = str(exc)
        if "division" in msg.lower() or "zero" in msg.lower():
            return "DIV_BY_ZERO"
        return f"ERROR: {msg}"


def solve_linear_system(eqs: list[str], vars_list: list[str]) -> dict[str, Any]:
    sp = _sympy()
    if not isinstance(eqs, list) or not isinstance(vars_list, list):
        raise TypeError("eqs and vars_list must be lists of strings")
    if not (0 < len(eqs) <= 8) or not (0 < len(vars_list) <= 8):
        raise ValueError("solve_linear_system supports 1-8 equations and variables")
    if any(not isinstance(v, str) or not v.isidentifier() or v.startswith("_") for v in vars_list):
        raise ValueError("variable names must be valid public identifiers")
    if len(set(vars_list)) != len(vars_list):
        raise ValueError("variable names must be unique")

    syms = [sp.Symbol(v) for v in vars_list]
    declared = set(syms)
    env = {name: sym for name, sym in zip(vars_list, syms)}
    expressions = []
    for equation in eqs:
        if not isinstance(equation, str) or equation.count("=") != 1:
            raise ValueError(f"Equation {equation!r} must contain exactly one '='")
        lhs, rhs = equation.replace("−", "-").split("=", 1)
        expr = sp.expand(safe_sympy(lhs, env) - safe_sympy(rhs, env))
        undeclared = expr.free_symbols - declared
        if undeclared:
            raise ValueError("equation contains undeclared variable(s): " + ", ".join(sorted(str(x) for x in undeclared)))
        try:
            poly = sp.Poly(expr, *syms)
        except sp.PolynomialError as exc:
            raise ValueError("equations must be polynomial in declared variables") from exc
        if poly.total_degree() > 1:
            raise ValueError("nonlinear equation supplied to linear-system solver")
        expressions.append(expr)

    coeff_rows = [[sp.expand(expr).coeff(symbol) for symbol in syms] for expr in expressions]
    constants = [sp.expand(expr).subs({symbol: 0 for symbol in syms}) for expr in expressions]
    coeff = sp.Matrix(coeff_rows)
    augmented = coeff.row_join(sp.Matrix([-c for c in constants]))
    rank_coeff = int(coeff.rank())
    rank_augmented = int(augmented.rank())
    solution_set = sp.linsolve(expressions, *syms)

    if solution_set is sp.EmptySet or rank_coeff != rank_augmented:
        return {
            "status": "inconsistent",
            "variables": list(vars_list),
            "solution": None,
            "free_variables": [],
            "rank_coeff": rank_coeff,
            "rank_augmented": rank_augmented,
            "residual_check": {},
        }

    solution_tuple = next(iter(solution_set))
    solution_exprs = [sp.simplify(value) for value in solution_tuple]
    solution = {name: str(value) for name, value in zip(vars_list, solution_exprs)}
    free_symbols = set().union(*(value.free_symbols for value in solution_exprs)) & declared
    free_variables = [name for name in vars_list if sp.Symbol(name) in free_symbols]
    residuals = {str(expr): str(sp.simplify(expr.subs(dict(zip(syms, solution_exprs))))) for expr in expressions}
    if any(sp.simplify(expr.subs(dict(zip(syms, solution_exprs)))) != 0 for expr in expressions):
        raise ArithmeticError("internal solver residual check failed")

    status = "underdetermined" if (free_variables or rank_coeff < len(syms)) else "unique"
    return {
        "status": status,
        "variables": list(vars_list),
        "solution": solution,
        "free_variables": free_variables,
        "rank_coeff": rank_coeff,
        "rank_augmented": rank_augmented,
        "residual_check": residuals,
    }


def integrate_definite(expr_str: str, var_name: str, a: Any, b: Any) -> str:
    sp = _sympy()
    if not isinstance(var_name, str) or not var_name.isidentifier() or var_name.startswith("_"):
        raise ValueError("variable name must be a valid public identifier")
    x = sp.Symbol(var_name)
    expr = safe_sympy(expr_str.replace("^", "**"), {var_name: x})
    lo = safe_sympy(str(a), {})
    hi = safe_sympy(str(b), {})
    val = sp.integrate(expr, (x, lo, hi))
    return str(sp.simplify(val))


def modular_pow(a: int, b: int, m: int) -> str:
    if not isinstance(a, int) or not isinstance(b, int) or not isinstance(m, int):
        raise TypeError("modular_pow requires integers")
    if isinstance(a, bool) or isinstance(b, bool) or isinstance(m, bool):
        raise TypeError("bool is not allowed")
    if m == 0:
        raise ValueError("modulus must be non-zero")
    if b < 0:
        raise ValueError("negative exponent not supported")
    if abs(a).bit_length() > 4096 or abs(b).bit_length() > 4096 or abs(m).bit_length() > 4096:
        raise ValueError("integer too large")
    return str(pow(int(a), int(b), int(m)))