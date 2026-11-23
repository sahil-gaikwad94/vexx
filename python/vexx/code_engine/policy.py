"""Static AST policy for generated code.

This is the *first* gate, not the isolation boundary.  A restricted-Python AST check
cannot make hostile code safe; only the out-of-process sandbox in
:mod:`vexx.code_engine.sandbox` can contain it.  The policy exists to reject the obvious
escapes early, with a clear error, before a child process is spawned.

The policy is configurable but never silently widened: a caller that relaxes it must say
so explicitly, and the resulting profile is recorded in the candidate report.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable
from dataclasses import dataclass

from .config import DEFAULT_ALLOWED_IMPORTS

__all__ = [
    "CodePolicyError",
    "CodePolicy",
    "DEFAULT_FORBIDDEN_IDENTIFIERS",
    "DEFAULT_FORBIDDEN_NODES",
    "check_ast_policy",
    "parse_and_check",
    "iter_calls",
]

#: Identifiers that must never be referenced by generated code.
DEFAULT_FORBIDDEN_IDENTIFIERS: tuple[str, ...] = (
    "os", "sys", "subprocess", "socket", "shutil", "pathlib", "open",
    "eval", "exec", "compile", "__import__", "globals", "locals",
    "getattr", "setattr", "delattr", "vars", "input", "breakpoint",
)

#: AST node names rejected outright.  ``Global`` is rejected because a module-level
#: ``global`` write is a side effect the sandbox cannot roll back.  ``Nonlocal`` is
#: deliberately *not* rejected: it only rebinds names in an enclosing function frame and
#: has no sandbox-escape surface, while blocking it rejects textbook Tarjan SCC.
DEFAULT_FORBIDDEN_NODES: tuple[str, ...] = ("Global",)

#: Attribute names that would let code walk into the object graph.
_FORBIDDEN_ATTRS = frozenset({
    "__globals__", "__builtins__", "__subclasses__", "__code__", "__class__",
    "__dict__", "__mro__", "__bases__", "__loader__", "__spec__",
    "__init__", "__new__", "__del__", "__class_getitem__",
    "__reduce__", "__reduce_ex__", "__getstate__", "__setstate__",
    "__getattribute__", "__setattr__", "__delattr__", "__getattr__",
    "__instancecheck__", "__subclasscheck__", "__mro_entries__", "__set_name__",
    "__get__", "__set__", "__delete__", "__objclass__",
    "__func__", "__self__", "__closure__", "__defaults__",
    "__kwdefaults__", "__annotations__", "__wrapped__", "__module__", "__doc__",
    "__name__", "__qualname__", "__type_params__",
    "__weakref__", "__slots__", "__prepare__",
    "__getnewargs__", "__getnewargs_ex__",
    "__sizeof__", "__dir__",
})

#: Attribute names that expose frame/code introspection — must never be reachable.
_FRAME_ATTRS = frozenset({
    "gi_frame", "cr_frame", "ag_frame", "tb_frame",
    "f_globals", "f_builtins", "f_locals", "f_back", "f_code",
    "gi_code", "gi_yieldfrom", "cr_code", "cr_origin",
    "ag_code", "ag_arg", "ag_locals", "ag_kwargs",
    "tb_next", "tb_lasti", "tb_lineno",
    "gi_suspended", "cr_running", "cr_await",
})

#: Methods that must never be called on untrusted strings/templates.
_FORBIDDEN_STRING_METHODS = frozenset({"format", "format_map", "vformat"})

#: Attribute names that must never be referenced by generated code.
_FORBIDDEN_IDENTIFIERS_EXT = frozenset({
    "os", "sys", "subprocess", "socket", "shutil", "pathlib", "open",
    "eval", "exec", "compile", "__import__", "globals", "locals",
    "getattr", "setattr", "delattr", "vars", "input", "breakpoint",
    "ctypes", "signal", "resource", "fcntl", "posix", "nt", "pwd",
    "grp", "termios", "tty", "pty", "select", "selectors", "asyncio",
    "threading", "multiprocessing", "concurrent", "queue", "socketserver",
    "http", "urllib", "ftplib", "smtplib", "telnetlib", "xmlrpc",
    "pickle", "shelve", "marshal", "dbm", "sqlite3",
    "importlib", "pkgutil", "modulefinder", "runpy", "pdb",
    "gc", "weakref", "weakrefproxy", "atexit", "copyreg", "reprlib", "enum", "graphlib", "heapq",
})


class CodePolicyError(PermissionError):
    """Raised when source violates the static code policy."""

    def __init__(self, message: str, *, node: str = "", line: int = 0):
        super().__init__(message)
        self.node = node
        self.line = int(line)

    def as_dict(self) -> dict:
        return {"error": str(self), "node": self.node, "line": self.line}


@dataclass(frozen=True)
class CodePolicy:
    """Static policy configuration.

    ``allowed_imports`` is an allow-list, so an import is permitted only when its root
    package is listed.  Relative and private imports are always rejected.
    """

    allowed_imports: tuple[str, ...] = DEFAULT_ALLOWED_IMPORTS
    forbidden_identifiers: tuple[str, ...] = DEFAULT_FORBIDDEN_IDENTIFIERS
    forbidden_nodes: tuple[str, ...] = DEFAULT_FORBIDDEN_NODES
    #: Permit module-level ``global`` writes (off by default; a side effect cannot be
    #: rolled back between candidate tests).
    allow_global: bool = False
    #: Permit ``nonlocal`` (on by default; needed by valid nested recursion).
    allow_nonlocal: bool = True
    #: Permit ``import`` of the listed modules only; when false all imports are rejected.
    allow_imports: bool = True
    max_source_chars: int = 24_000
    max_ast_nodes: int = 20_000

    def __post_init__(self) -> None:
        if not isinstance(self.allowed_imports, tuple):
            raise TypeError("allowed_imports must be a tuple")
        for module in self.allowed_imports:
            if not isinstance(module, str) or not module.isidentifier():
                raise ValueError(f"invalid allowed import: {module!r}")
        for name in self.forbidden_identifiers:
            if not isinstance(name, str) or not name:
                raise ValueError(f"invalid forbidden identifier: {name!r}")
        for name in self.forbidden_nodes:
            if not isinstance(name, str) or not name:
                raise ValueError(f"invalid forbidden node: {name!r}")
        if self.max_source_chars < 1 or self.max_ast_nodes < 1:
            raise ValueError("policy size limits must be positive")

    def to_dict(self) -> dict:
        return {
            "allowed_imports": list(self.allowed_imports),
            "forbidden_identifiers": list(self.forbidden_identifiers),
            "forbidden_nodes": list(self.forbidden_nodes),
            "allow_global": self.allow_global,
            "allow_nonlocal": self.allow_nonlocal,
            "allow_imports": self.allow_imports,
        }


def _node_name(node: ast.AST) -> str:
    return type(node).__name__


def _check_import(node: ast.Import | ast.ImportFrom, policy: CodePolicy) -> None:
    if not policy.allow_imports:
        raise CodePolicyError("imports are forbidden by policy", node=_node_name(node), line=node.lineno)
    if isinstance(node, ast.ImportFrom):
        if int(node.level or 0) != 0:
            raise CodePolicyError("relative imports are forbidden", node="ImportFrom", line=node.lineno)
        modules = [node.module or ""]
        names = [alias.name for alias in node.names]
    else:
        modules = [alias.name for alias in node.names]
        names = []
    for module in modules:
        root = str(module).split(".")[0]
        if not root:
            raise CodePolicyError(f"invalid import: {module!r}", node="Import", line=node.lineno)
        if root not in policy.allowed_imports:
            raise CodePolicyError(f"forbidden import: {module}", node="Import", line=node.lineno)
        if any(part.startswith("_") for part in str(module).split(".")):
            raise CodePolicyError(f"forbidden private import: {module}", node="Import", line=node.lineno)
    for name in names:
        if str(name).split(".")[0].startswith("_"):
            raise CodePolicyError(f"forbidden private import: {name}", node="ImportFrom", line=node.lineno)
    for alias in node.names:
        if alias.asname and str(alias.asname).startswith("_"):
            raise CodePolicyError(f"forbidden private import alias: {alias.asname}", node="Import", line=node.lineno)


def check_ast_policy(source: str, policy: CodePolicy | None = None) -> ast.Module:
    """Parse ``source`` and enforce ``policy``.

    Returns the parsed module so callers can reuse the tree.  Raises
    :class:`SyntaxError` for invalid syntax and :class:`CodePolicyError` for a policy
    violation.
    """
    policy = policy or CodePolicy()
    if not isinstance(source, str):
        raise TypeError("source must be a string")
    if len(source) > policy.max_source_chars:
        raise CodePolicyError(
            f"source exceeds {policy.max_source_chars} characters ({len(source)})",
            node="Module",
        )
    tree = ast.parse(source)

    nodes = 0
    for node in ast.walk(tree):
        nodes += 1
        if nodes > policy.max_ast_nodes:
            raise CodePolicyError(
                f"source has more than {policy.max_ast_nodes} AST nodes",
                node="Module",
            )
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            _check_import(node, policy)
            continue
        if isinstance(node, ast.Global) and not policy.allow_global:
            raise CodePolicyError("global statements are forbidden", node="Global", line=node.lineno)
        if isinstance(node, ast.Nonlocal) and not policy.allow_nonlocal:
            raise CodePolicyError("nonlocal statements are forbidden", node="Nonlocal", line=node.lineno)
        name = _node_name(node)
        if name in policy.forbidden_nodes:
            raise CodePolicyError(f"forbidden AST node: {name}", node=name, line=getattr(node, "lineno", 0))
        if isinstance(node, ast.Name) and node.id in policy.forbidden_identifiers:
            raise CodePolicyError(f"forbidden identifier: {node.id}", node="Name", line=node.lineno)
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__") or node.attr in _FORBIDDEN_ATTRS:
                raise CodePolicyError(
                    f"forbidden dunder attribute: {node.attr}",
                    node="Attribute",
                    line=node.lineno,
                )
            if node.attr in _FRAME_ATTRS:
                raise CodePolicyError(
                    f"forbidden frame/introspection attribute: {node.attr}",
                    node="Attribute",
                    line=node.lineno,
                )
        if isinstance(node, ast.Call):
            # Block str.format / str.format_map on untrusted strings/templates.
            func = node.func
            if isinstance(func, ast.Attribute):
                if func.attr in _FORBIDDEN_STRING_METHODS:
                    raise CodePolicyError(
                        f"forbidden string method: {func.attr}",
                        node="Call",
                        line=node.lineno,
                    )
            elif isinstance(func, ast.Name) and func.id in _FORBIDDEN_IDENTIFIERS_EXT:
                raise CodePolicyError(
                    f"forbidden identifier: {func.id}",
                    node="Name",
                    line=node.lineno,
                )
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_IDENTIFIERS_EXT:
            raise CodePolicyError(
                f"forbidden identifier: {node.id}",
                node="Name",
                line=node.lineno,
            )
    return tree


def parse_and_check(source: str, policy: CodePolicy | None = None) -> ast.Module:
    """Alias for :func:`check_ast_policy` with a clearer name at call sites."""
    return check_ast_policy(source, policy)


def iter_calls(tree: ast.AST) -> Iterable[ast.Call]:
    """Yield every call node in ``tree`` (used by the masker and tests)."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            yield node


def dotted_name(node: ast.AST) -> str | None:
    """Render ``a.b.c`` for a Name/Attribute chain, else ``None``."""
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
        return ".".join(reversed(parts))
    return None


def call_argument_shape(call: ast.Call) -> dict:
    """Summarise a call site for :func:`~vexx.code_engine.contracts.check_call`.

    ``has_starargs`` / ``has_kwargs`` mean the real argument count is not statically
    known, so the checker must not reject the call on arity grounds.
    """
    keywords: list[str] = []
    has_kwargs = False
    for keyword in call.keywords:
        if keyword.arg is None:
            has_kwargs = True
        else:
            keywords.append(keyword.arg)
    has_starargs = any(isinstance(arg, ast.Starred) for arg in call.args)
    positional = sum(1 for arg in call.args if not isinstance(arg, ast.Starred))
    return {
        "n_positional": positional,
        "keyword_names": keywords,
        "has_starargs": has_starargs,
        "has_kwargs": has_kwargs,
    }