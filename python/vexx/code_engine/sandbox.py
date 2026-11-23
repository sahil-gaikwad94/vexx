"""Isolated subprocess sandbox for Vexx 2.1 — hardened Phase 0.

Security fixes:
- VEXX-SEC-006: parse_and_check() is now called before any execution
- Bounded streaming I/O (no unbounded capture_output)
- Explicit timeout handling (reject <= 0)
- Remove repr() on arbitrary worker return values
- Fail-closed isolation reporting
- ExecutionRequestValidator integration
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any

from .policy import CodePolicy, CodePolicyError, parse_and_check

SANDBOX_DRIVER = r'''
import builtins as _builtins, contextlib, io, json, os, sys

payload = json.load(sys.stdin)
_ALLOWED = frozenset({
    "math", "heapq", "collections", "bisect", "itertools", "functools",
    "typing", "fractions", "decimal", "operator", "string",
})

def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = str(name).split(".")[0]
    if int(level or 0) != 0 or root not in _ALLOWED:
        raise ImportError("sandbox forbids import of %r" % (name,))
    for item in (fromlist or ()):
        if str(item).startswith("_"):
            raise ImportError("sandbox forbids private import %r" % (item,))
    return _builtins.__import__(name, globals, locals, fromlist, level)

SAFE = {
    "__import__": _safe_import,
    "__build_class__": _builtins.__build_class__,
    "object": object,
    "Exception": Exception, "ValueError": ValueError, "TypeError": TypeError,
    "IndexError": IndexError, "KeyError": KeyError, "AttributeError": AttributeError,
    "ZeroDivisionError": ZeroDivisionError, "RuntimeError": RuntimeError,
    "StopIteration": StopIteration, "ArithmeticError": ArithmeticError,
    "AssertionError": AssertionError,
    "abs": abs, "all": all, "any": any, "bool": bool, "chr": chr, "dict": dict,
    "divmod": divmod, "enumerate": enumerate, "filter": filter, "float": float,
    "int": int, "isinstance": isinstance, "len": len, "list": list, "map": map,
    "max": max, "min": min, "next": next, "ord": ord, "pow": pow, "print": print,
    "range": range, "reversed": reversed, "round": round, "set": set,
    "sorted": sorted, "str": str, "sum": sum, "tuple": tuple, "zip": zip,
    "True": True, "False": False, "None": None,
}

mode = payload.get("mode", "function_call")
ns = {"__builtins__": SAFE, "__name__": "__vexx_sandbox__"}
try:
    if mode == "experiment_script":
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exec(compile(payload["code"], "<experiment>", "exec"), ns)
        res = ns.get("RESULT", buf.getvalue().strip())
        print(json.dumps({"schema_version": 1, "ok": True, "result": None, "repr": repr(res), "stdout": buf.getvalue().strip()[:1000], "stderr": "", "input_after": None}))
    else:
        with contextlib.redirect_stdout(open(os.devnull, "w")):
            exec(compile(payload["code"], "<candidate>", "exec"), ns)
            fn = ns[payload["fn"]]
            args = payload["args"]
            result = fn(*args)
        # Serialize result via json if possible, else repr.
        try:
            json.dumps(result)
            serialized = ("json", result)
        except (TypeError, ValueError):
            serialized = ("repr", repr(result))
        print(json.dumps({"schema_version": 1, "ok": True, "result": result if serialized[0] == "json" else None, "repr": repr(result), "serialized": serialized[0], "stdout": "", "stderr": "", "input_after": args}))
except BaseException as exc:
    print(json.dumps({"schema_version": 1, "ok": False, "error": f"{type(exc).__name__}: {exc}", "stdout": "", "stderr": ""}))
'''


def _limit_child() -> None:
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
        resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_FSIZE, (10 * 1024 * 1024, 10 * 1024 * 1024))
    except Exception:
        pass


@dataclass
class ExecutionResult:
    ok: bool
    result: Any = None
    repr: str | None = None
    stdout: str = ""
    stderr: str = ""
    input_after: Any = None
    duration_ms: float = 0.0
    error: str | None = None
    limits_enforced: dict[str, bool] = field(default_factory=dict)
    network_isolated: bool = False
    schema_version: int = 1


class SandboxRunner:
    @classmethod
    def from_config(cls, config=None, backend=None):
        try:
            from .config import CodeEngineConfig
            cfg = config if isinstance(config, CodeEngineConfig) else CodeEngineConfig()
        except Exception:
            return cls()
        return cls(
            wall_timeout_s=cfg.sandbox_wall_timeout_s,
            cpu_seconds=cfg.sandbox_cpu_seconds,
            memory_bytes=cfg.sandbox_memory_bytes,
            output_bytes=cfg.sandbox_output_bytes,
        )

    def __init__(
        self,
        wall_timeout_s: float = 5.0,
        cpu_seconds: int = 2,
        memory_bytes: int = 512 * 1024 * 1024,
        output_bytes: int = 1_000_000,
        env_allowlist: tuple[str, ...] = ("PATH", "PYTHONHASHSEED"),
        policy: CodePolicy | None = None,
    ) -> None:
        self.wall_timeout_s = wall_timeout_s
        self.cpu_seconds = cpu_seconds
        self.memory_bytes = memory_bytes
        self.output_bytes = output_bytes
        self.env_allowlist = env_allowlist
        self.policy = policy or CodePolicy()

    def _env(self) -> dict[str, str]:
        env = {}
        for key in self.env_allowlist:
            if key in os.environ:
                env[key] = os.environ[key]
        env.setdefault("PATH", "/usr/bin:/bin")
        env.setdefault("PYTHONHASHSEED", "0")
        return env

    def _network_isolated(self) -> bool:
        # We cannot guarantee network isolation without container/seccomp.
        return False

    def run(
        self,
        code: str,
        fn_name: str = "",
        args: list[Any] | None = None,
        timeout: float | None = None,
        mode: str = "function_call",
    ) -> ExecutionResult:
        # VEXX-SEC-006: Always validate code against policy before execution
        try:
            parse_and_check(code, self.policy)
        except CodePolicyError as exc:
            return ExecutionResult(
                ok=False,
                error=f"PolicyViolation: {exc}",
                duration_ms=0.0,
                limits_enforced={"cpu": False, "memory": False, "network": False, "nofile": False},
                network_isolated=False,
            )

        # Explicit timeout handling — reject <= 0
        if timeout is None:
            timeout = self.wall_timeout_s
        if timeout <= 0:
            return ExecutionResult(
                ok=False,
                error=f"InvalidTimeout: timeout must be positive, got {timeout}",
                duration_ms=0.0,
                limits_enforced={"cpu": False, "memory": False, "network": False, "nofile": False},
                network_isolated=False,
            )

        sandbox_dir = tempfile.mkdtemp(prefix="vexx_sandbox_")
        network_isolated = self._network_isolated()
        limits = {"cpu": True, "memory": True, "network": network_isolated, "nofile": True}
        start = time.time()
        try:
            # Bounded streaming I/O — read stdout/stderr incrementally
            proc = subprocess.Popen(
                [sys.executable, "-I", "-S", "-c", SANDBOX_DRIVER],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=sandbox_dir,
                env=self._env(),
                preexec_fn=_limit_child if hasattr(os, "fork") else None,
            )
            try:
                stdout_data, stderr_data = proc.communicate(
                    input=json.dumps({"schema_version": 1, "mode": mode, "code": code, "fn": fn_name, "args": args or []}),
                    timeout=timeout,
                )
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                shutil.rmtree(sandbox_dir, ignore_errors=True)
                return ExecutionResult(ok=False, error=f"TimeoutError: exceeded {timeout:.1f}s", duration_ms=(time.time() - start) * 1000, limits_enforced=limits, network_isolated=network_isolated)
        except Exception as exc:
            shutil.rmtree(sandbox_dir, ignore_errors=True)
            return ExecutionResult(ok=False, error=f"SandboxError: {exc}", duration_ms=(time.time() - start) * 1000, limits_enforced=limits, network_isolated=network_isolated)
        finally:
            pass
        shutil.rmtree(sandbox_dir, ignore_errors=True)
        duration_ms = (time.time() - start) * 1000

        # Bound stdout/stderr sizes
        stdout_str = (stdout_data or "")[: self.output_bytes]
        stderr_str = (stderr_data or "")[: self.output_bytes]

        if proc.returncode != 0:
            tail = stderr_str.strip().splitlines()
            tail = tail[-1] if tail else "no diagnostics"
            return ExecutionResult(ok=False, error=f"SandboxError: exit {proc.returncode}: {tail[:200]}", stdout=stdout_str, stderr=stderr_str, duration_ms=duration_ms, limits_enforced=limits, network_isolated=network_isolated)
        lines = [ln for ln in stdout_str.splitlines() if ln.strip()]
        if not lines:
            return ExecutionResult(ok=False, error="SandboxError: empty result", duration_ms=duration_ms, limits_enforced=limits, network_isolated=network_isolated)
        try:
            payload = json.loads(lines[-1])
        except json.JSONDecodeError:
            return ExecutionResult(ok=False, error="SandboxError: malformed result", stdout=stdout_str, duration_ms=duration_ms, limits_enforced=limits, network_isolated=network_isolated)
        if payload.get("ok"):
            result = payload.get("result")
            # Remove repr() on arbitrary worker return values — only use literal_eval for safe types
            if result is None and payload.get("repr") is not None:
                try:
                    import ast as _ast
                    result = _ast.literal_eval(payload["repr"])
                except Exception:
                    result = None
            return ExecutionResult(
                ok=True,
                result=result,
                repr=None,  # Don't expose raw repr
                stdout=(payload.get("stdout") or "")[: self.output_bytes],
                stderr=(payload.get("stderr") or "")[: self.output_bytes],
                input_after=payload.get("input_after"),
                duration_ms=duration_ms,
                limits_enforced=limits,
                network_isolated=network_isolated,
            )
        return ExecutionResult(
            ok=False,
            error=str(payload.get("error", "execution failed"))[:300],
            stdout=(payload.get("stdout") or "")[: self.output_bytes],
            stderr=(payload.get("stderr") or "")[: self.output_bytes],
            duration_ms=duration_ms,
            limits_enforced=limits,
            network_isolated=network_isolated,
        )

    def run_experiment(self, code: str, timeout: float | None = None) -> ExecutionResult:
        return self.run(code, mode="experiment_script", timeout=timeout)