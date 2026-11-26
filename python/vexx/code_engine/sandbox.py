"""Isolated subprocess sandbox for Vexx 2.1 — hardened.

Guarantees, in order of importance:

* **The policy runs first.** ``parse_and_check`` rejects obvious escapes before a child is
  ever spawned.
* **Results are typed, not ``repr``-only.** ``ExecutionResult.serialized_kind`` says whether
  ``result`` is JSON-native, recovered from a literal ``repr``, or an opaque display string.
  Setting ``repr=None`` (an earlier hardening pass) silently turned every return value into
  ``None``, which broke ``safe_execute_python``, differential fuzzing and experiment cells.
* **Resource limits are the configured ones and are *verified*.** CPU, address space and
  file-descriptor limits come from :class:`~vexx.code_engine.config.CodeEngineConfig`, are
  applied in a ``preexec`` hook, and are read back in the child with ``getrlimit``. The
  envelope reports the observed booleans, so the host never claims a limit it did not get.
* **I/O is bounded while streaming.** Output is drained through ``selectors`` and the child
  is killed the moment it exceeds the byte budget or the deadline; nothing unbounded is
  buffered in host RAM first.
* **Optional kernel isolation.** On Linux the child calls ``unshare(CLONE_NEWNET | ...)``
  before ``exec`` when the platform allows it; the child reports whether its network
  namespace actually differs from the parent's, so ``network_isolated`` is measured rather
  than assumed.
* **Batched execution.** ``run_batch`` executes every test case in *one* child process with a
  per-case ``SIGALRM`` budget, which removes the per-probe interpreter start-up cost without
  letting one case consume the whole budget.
"""

from __future__ import annotations

import ast
import json
import os
import selectors
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any

from .policy import CodePolicy, CodePolicyError, parse_and_check

#: Hard cap on the display string handed back to the host.
MAX_DISPLAY_CHARS = 4096

SANDBOX_DRIVER = r'''
import ast as _ast, builtins as _builtins, contextlib, io, json, os, sys, types

_ALLOWED = frozenset(__ALLOWED__)
_MAX_OUT = 1000000


class _OutputLimit(Exception):
    pass


class _BoundedBuffer(io.TextIOBase):
    """Captured stdout with a hard character cap.

    ``io.StringIO`` was unbounded: a ``print`` loop inside the child grew the buffer until
    the address-space rlimit killed the interpreter, which surfaced as an opaque exit
    code instead of a clear output-limit verdict.
    """

    def __init__(self, limit):
        self._parts = []
        self._size = 0
        self._limit = int(limit)

    def write(self, text):
        text = str(text)
        self._size += len(text)
        if self._size > self._limit:
            raise _OutputLimit("stdout exceeds %d characters" % self._limit)
        self._parts.append(text)
        return len(text)

    def getvalue(self):
        return "".join(self._parts)

    def flush(self):
        return None

def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = str(name).split(".")[0]
    if int(level or 0) != 0 or root not in _ALLOWED:
        raise ImportError("sandbox forbids import of %r" % (name,))
    for item in (fromlist or ()):
        if str(item).startswith("_"):
            raise ImportError("sandbox forbids private import %r" % (item,))
    module = _builtins.__import__(name, globals, locals, fromlist, level)
    if root == "string":
        # ``string.Formatter`` / ``string.Template`` walk attribute paths given as
        # strings; the sandbox exposes only the literal constants.
        safe = types.ModuleType("string")
        for const in ("ascii_letters", "ascii_lowercase", "ascii_uppercase", "digits",
                      "hexdigits", "octdigits", "punctuation", "printable", "whitespace"):
            if hasattr(module, const):
                setattr(safe, const, getattr(module, const))
        return safe
    return module

SAFE = {
    "__import__": _safe_import,
    "__build_class__": _builtins.__build_class__,
    "object": object, "type": type, "super": super, "property": property,
    "staticmethod": staticmethod, "classmethod": classmethod,
    "Exception": Exception, "ValueError": ValueError, "TypeError": TypeError,
    "IndexError": IndexError, "KeyError": KeyError, "AttributeError": AttributeError,
    "ZeroDivisionError": ZeroDivisionError, "RuntimeError": RuntimeError,
    "StopIteration": StopIteration, "ArithmeticError": ArithmeticError,
    "AssertionError": AssertionError, "KeyError": KeyError, "OverflowError": OverflowError,
    "NotImplementedError": NotImplementedError, "LookupError": LookupError,
    "abs": abs, "all": all, "any": any, "bool": bool, "chr": chr, "dict": dict,
    "divmod": divmod, "enumerate": enumerate, "filter": filter, "float": float,
    "int": int, "isinstance": isinstance, "issubclass": issubclass, "len": len,
    "list": list, "map": map, "max": max, "min": min, "next": next, "ord": ord,
    "pow": pow, "print": print, "range": range, "reversed": reversed, "round": round,
    "set": set, "sorted": sorted, "str": str, "sum": sum, "tuple": tuple, "zip": zip,
    "frozenset": frozenset, "slice": slice, "hash": hash, "repr": repr, "format": format,
    "True": True, "False": False, "None": None,
}

def _safe_repr(value):
    try:
        text = repr(value)
    except BaseException:
        text = "<unrepresentable %s>" % type(value).__name__
    return text[:4096]

def _encode(value):
    try:
        json.dumps(value, allow_nan=False)
        return {"serialized": "json", "result": value, "repr": _safe_repr(value),
                "type": type(value).__name__}
    except BaseException:
        pass
    text = _safe_repr(value)
    kind = "opaque_repr"
    try:
        _ast.literal_eval(text)
        kind = "literal"
    except BaseException:
        kind = "opaque_repr"
    return {"serialized": kind, "result": None, "repr": text, "type": type(value).__name__}

def _limits(expected):
    try:
        import resource
    except BaseException:
        return {"cpu": False, "memory": False, "nofile": False}
    out = {}
    for key, rname in (("cpu", "RLIMIT_CPU"), ("memory", "RLIMIT_AS"), ("nofile", "RLIMIT_NOFILE")):
        want = expected.get(key)
        if want is None or not hasattr(resource, rname):
            out[key] = False
            continue
        try:
            soft, _hard = resource.getrlimit(getattr(resource, rname))
        except BaseException:
            out[key] = False
            continue
        try:
            out[key] = soft != resource.RLIM_INFINITY and int(soft) <= int(want)
        except BaseException:
            out[key] = False
    return out

def _network_isolated(parent_inode):
    if parent_inode is None:
        return False
    try:
        inode = os.stat("/proc/self/ns/net").st_ino
    except BaseException:
        return False
    try:
        return int(inode) != int(parent_inode)
    except BaseException:
        return False

def _run_once(code, fn, args, mode, limit):
    ns = {"__builtins__": SAFE, "__name__": "__vexx_sandbox__"}
    buf = _BoundedBuffer(limit)
    if mode == "experiment_script":
        with contextlib.redirect_stdout(buf):
            exec(compile(code, "<experiment>", "exec"), ns)
        result = ns.get("RESULT", buf.getvalue().strip())
        encoded = _encode(result)
        return {"ok": True, "stdout": buf.getvalue()[-4096:], "stderr": "",
                "input_after": None, **encoded}
    with contextlib.redirect_stdout(_BoundedBuffer(limit)):
        exec(compile(code, "<candidate>", "exec"), ns)
        fn_obj = ns[fn]
        result = fn_obj(*args)
    encoded = _encode(result)
    return {"ok": True, "stdout": "", "stderr": "", "input_after": args, **encoded}

payload = json.load(sys.stdin)
mode = payload.get("mode", "function_call")
_MAX_OUT = int((payload.get("limits") or {}).get("output") or _MAX_OUT)
limits = _limits(payload.get("limits") or {})
network = _network_isolated(payload.get("parent_netns"))
base = {"schema_version": 2, "limits": limits, "network_isolated": network}

try:
    if mode == "batch":
        import signal

        class _CaseTimeout(Exception):
            pass

        def _on_alarm(signum, frame):
            raise _CaseTimeout("case exceeded its time budget")

        try:
            signal.signal(signal.SIGALRM, _on_alarm)
        except BaseException:
            pass
        cases = payload.get("cases") or []
        per_case = float(payload.get("per_case_seconds") or 0.0)
        results = []
        for case in cases:
            if per_case > 0:
                try:
                    signal.setitimer(signal.ITIMER_REAL, per_case)
                except BaseException:
                    pass
            try:
                results.append(_run_once(payload["code"], payload["fn"], case, "function_call", _MAX_OUT))
            except BaseException as exc:
                results.append({"ok": False, "error": "%s: %s" % (type(exc).__name__, exc),
                                "stdout": "", "stderr": ""})
            finally:
                try:
                    signal.setitimer(signal.ITIMER_REAL, 0.0)
                except BaseException:
                    pass
        print(json.dumps({**base, "ok": True, "cases": results}, default=str))
    else:
        out = _run_once(payload["code"], payload.get("fn", ""), payload["args"], mode, _MAX_OUT)
        print(json.dumps({**base, **out}, default=str))
except BaseException as exc:
    print(json.dumps({**base, "ok": False, "error": "%s: %s" % (type(exc).__name__, exc),
                      "stdout": "", "stderr": ""}, default=str))
'''


@dataclass
class ExecutionResult:
    ok: bool
    result: Any = None
    repr: str | None = None
    #: ``"json"`` | ``"literal"`` | ``"opaque_repr"`` | ``"none"``.
    serialized_kind: str = "none"
    stdout: str = ""
    stderr: str = ""
    input_after: Any = None
    duration_ms: float = 0.0
    error: str | None = None
    limits_enforced: dict[str, bool] = field(default_factory=dict)
    network_isolated: bool = False
    type_name: str = ""
    schema_version: int = 2


def _decode_payload(payload: dict[str, Any]) -> tuple[Any, str, str | None, str]:
    """Return ``(result, kind, display_repr, type_name)`` from a child envelope."""
    kind = str(payload.get("serialized") or "none")
    text = payload.get("repr")
    text = text[:MAX_DISPLAY_CHARS] if isinstance(text, str) else None
    type_name = str(payload.get("type") or "")
    if kind == "json":
        return payload.get("result"), "json", text, type_name
    if kind == "literal" and isinstance(text, str):
        try:
            return ast.literal_eval(text), "literal", text, type_name
        except (ValueError, SyntaxError):
            return None, "opaque_repr", text, type_name
    if kind == "opaque_repr":
        return None, "opaque_repr", text, type_name
    return payload.get("result"), "none", text, type_name


def _communicate_bounded(
    proc: subprocess.Popen,
    stdin_data: str,
    timeout: float,
    max_bytes: int,
) -> tuple[str, str, bool, bool]:
    """Stream stdout/stderr with hard memory and wall-time caps.

    Returns ``(stdout, stderr, overflowed, timed_out)``.  The child is killed as soon as the combined
    output exceeds ``2 * max_bytes`` or the deadline passes, so an infinite ``print`` loop
    cannot exhaust host memory before the timeout fires.
    """
    if proc.stdin is not None:
        try:
            proc.stdin.write(stdin_data)
            proc.stdin.flush()
        except (BrokenPipeError, OSError):
            pass
        try:
            proc.stdin.close()
        except OSError:
            pass
    sel = selectors.DefaultSelector()
    if proc.stdout is not None:
        sel.register(proc.stdout, selectors.EVENT_READ, "out")
    if proc.stderr is not None:
        sel.register(proc.stderr, selectors.EVENT_READ, "err")
    out_chunks: list[str] = []
    err_chunks: list[str] = []
    out_len = err_len = 0
    deadline = time.monotonic() + timeout
    overflowed = False
    try:
        while sel.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                proc.kill()
                proc.wait()
                return "".join(out_chunks)[:max_bytes], "".join(err_chunks)[:max_bytes], False, True
            for key, _ in sel.select(timeout=min(0.05, remaining)):
                chunk = key.fileobj.read(4096)
                if not chunk:
                    sel.unregister(key.fileobj)
                    continue
                if key.data == "out":
                    out_len += len(chunk)
                    out_chunks.append(chunk)
                else:
                    err_len += len(chunk)
                    err_chunks.append(chunk)
                if out_len + err_len > max_bytes * 2:
                    overflowed = True
                    proc.kill()
                    proc.wait()
                    return "".join(out_chunks)[:max_bytes], "".join(err_chunks)[:max_bytes], True, False
    finally:
        sel.close()
    try:
        proc.wait(timeout=1.0)
    except subprocess.TimeoutExpired:  # pragma: no cover - defensive
        proc.kill()
        proc.wait()
    return "".join(out_chunks)[:max_bytes], "".join(err_chunks)[:max_bytes], overflowed, False


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
            max_files=cfg.sandbox_max_files,
            max_processes=cfg.sandbox_max_processes,
            policy=CodePolicy(allowed_imports=cfg.allowed_imports),
            network_isolation=cfg.network_isolated,
        )

    def __init__(
        self,
        wall_timeout_s: float = 5.0,
        cpu_seconds: int = 2,
        memory_bytes: int = 512 * 1024 * 1024,
        output_bytes: int = 1_000_000,
        env_allowlist: tuple[str, ...] = ("PATH", "PYTHONHASHSEED"),
        policy: CodePolicy | None = None,
        max_files: int = 64,
        max_processes: int = 64,
        network_isolation: bool = True,
    ) -> None:
        if wall_timeout_s <= 0:
            raise ValueError("wall_timeout_s must be positive")
        self.wall_timeout_s = float(wall_timeout_s)
        self.cpu_seconds = int(cpu_seconds)
        self.memory_bytes = int(memory_bytes)
        self.output_bytes = int(output_bytes)
        self.env_allowlist = env_allowlist
        self.policy = policy or CodePolicy()
        self.max_files = int(max_files)
        self.max_processes = int(max_processes)
        #: Attempt kernel network namespace isolation (reported honestly when unavailable).
        self.network_isolation = bool(network_isolation)
        self._driver = SANDBOX_DRIVER.replace(
            "__ALLOWED__", repr(tuple(self.policy.allowed_imports))
        )

    # ------------------------------------------------------------------ helpers

    def _env(self) -> dict[str, str]:
        env = {}
        for key in self.env_allowlist:
            if key in os.environ:
                env[key] = os.environ[key]
        env.setdefault("PATH", "/usr/bin:/bin")
        env.setdefault("PYTHONHASHSEED", "0")
        return env

    def _network_isolated(self) -> bool:
        """Static capability answer; the per-run truth comes from the child envelope."""
        return bool(self.network_isolation and hasattr(os, "unshare"))

    def _preexec(self):
        if not hasattr(os, "fork"):
            return None
        cpu = self.cpu_seconds
        memory = self.memory_bytes
        max_files = self.max_files
        max_processes = self.max_processes
        output_bytes = self.output_bytes
        isolate = self.network_isolation

        def _apply_limits() -> None:
            try:
                import resource
            except Exception:  # pragma: no cover - non-POSIX
                return
            for name, value in (
                ("RLIMIT_CPU", (cpu, cpu + 1)),
                ("RLIMIT_AS", (memory, memory)),
                ("RLIMIT_NOFILE", (max_files, max_files)),
                ("RLIMIT_NPROC", (max_processes, max_processes)),
                ("RLIMIT_CORE", (0, 0)),
                ("RLIMIT_FSIZE", (output_bytes, output_bytes)),
            ):
                try:
                    resource.setrlimit(getattr(resource, name), value)
                except Exception:
                    pass
            if isolate and hasattr(os, "unshare"):
                flags = 0
                for flag in ("CLONE_NEWNET", "CLONE_NEWIPC", "CLONE_NEWUTS"):
                    flags |= getattr(os, flag, 0)
                try:
                    if flags:
                        os.unshare(flags)
                except Exception:
                    # Unprivileged user namespaces may be disabled; the child reports the
                    # namespace it actually got, so this stays honest.
                    pass

        return _apply_limits

    def _parent_netns(self) -> int | None:
        try:
            return int(os.stat("/proc/self/ns/net").st_ino)
        except Exception:
            return None

    def _limits_payload(self) -> dict[str, int]:
        return {
            "cpu": self.cpu_seconds,
            "memory": self.memory_bytes,
            "nofile": self.max_files,
            "output": self.output_bytes,
        }

    def _spawn(self, payload: dict[str, Any], sandbox_dir: str, timeout: float):
        proc = subprocess.Popen(
            [sys.executable, "-I", "-S", "-c", self._driver],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=sandbox_dir,
            env=self._env(),
            preexec_fn=self._preexec(),
        )
        stdout_data, stderr_data, overflowed, timed_out = _communicate_bounded(
            proc,
            json.dumps(payload, default=str),
            timeout,
            self.output_bytes,
        )
        return proc, stdout_data, stderr_data, overflowed, timed_out

    # ---------------------------------------------------------------------- run

    def run(
        self,
        code: str,
        fn_name: str = "",
        args: list[Any] | None = None,
        timeout: float | None = None,
        mode: str = "function_call",
    ) -> ExecutionResult:
        # VEXX-SEC-006: Always validate code against policy before execution.
        try:
            parse_and_check(code, self.policy)
        except CodePolicyError as exc:
            return ExecutionResult(
                ok=False,
                error=f"PolicyViolation: {exc}",
                limits_enforced={"cpu": False, "memory": False, "nofile": False, "network": False},
                network_isolated=False,
            )
        except SyntaxError as exc:
            return ExecutionResult(
                ok=False,
                error=f"SyntaxError: {exc}",
                limits_enforced={"cpu": False, "memory": False, "nofile": False, "network": False},
                network_isolated=False,
            )

        if timeout is None:
            timeout = self.wall_timeout_s
        if timeout <= 0:
            return ExecutionResult(
                ok=False,
                error=f"InvalidTimeout: timeout must be positive, got {timeout}",
                limits_enforced={"cpu": False, "memory": False, "nofile": False, "network": False},
                network_isolated=False,
            )

        sandbox_dir = tempfile.mkdtemp(prefix="vexx_sandbox_")
        start = time.monotonic()
        try:
            payload = {
                "schema_version": 2,
                "mode": mode,
                "code": code,
                "fn": fn_name,
                "args": args or [],
                "limits": self._limits_payload(),
                "parent_netns": self._parent_netns(),
            }
            proc, stdout_data, stderr_data, overflowed, timed_out = self._spawn(payload, sandbox_dir, timeout)
        except subprocess.TimeoutExpired:
            shutil.rmtree(sandbox_dir, ignore_errors=True)
            return ExecutionResult(
                ok=False,
                error=f"TimeoutError: exceeded {timeout:.1f}s",
                duration_ms=(time.monotonic() - start) * 1000,
                limits_enforced={"cpu": False, "memory": False, "nofile": False, "network": False},
                network_isolated=False,
            )
        except Exception as exc:
            shutil.rmtree(sandbox_dir, ignore_errors=True)
            return ExecutionResult(
                ok=False,
                error=f"SandboxError: {exc}",
                duration_ms=(time.monotonic() - start) * 1000,
                limits_enforced={"cpu": False, "memory": False, "nofile": False, "network": False},
                network_isolated=False,
            )
        finally:
            shutil.rmtree(sandbox_dir, ignore_errors=True)

        duration_ms = (time.monotonic() - start) * 1000
        if timed_out:
            return ExecutionResult(
                ok=False,
                error=f"TimeoutError: exceeded {timeout:.1f}s",
                stdout=stdout_data[: self.output_bytes],
                stderr=stderr_data[: self.output_bytes],
                duration_ms=duration_ms,
                limits_enforced={"cpu": False, "memory": False, "nofile": False, "network": False},
            )
        return self._interpret(proc, stdout_data, stderr_data, overflowed, timeout, duration_ms)

    def _interpret(self, proc, stdout_data, stderr_data, overflowed, timeout, duration_ms) -> ExecutionResult:
        empty_limits = {"cpu": False, "memory": False, "nofile": False, "network": False}
        if overflowed:
            return ExecutionResult(
                ok=False,
                error=f"OutputLimit: child produced more than {self.output_bytes} bytes",
                stdout=stdout_data[: self.output_bytes],
                stderr=stderr_data[: self.output_bytes],
                duration_ms=duration_ms,
                limits_enforced=empty_limits,
            )
        lines = [ln for ln in (stdout_data or "").splitlines() if ln.strip()]
        payload: dict[str, Any] | None = None
        if lines:
            try:
                candidate = json.loads(lines[-1])
                if isinstance(candidate, dict) and "schema_version" in candidate:
                    payload = candidate
            except json.JSONDecodeError:
                payload = None
        if payload is None:
            if proc.returncode != 0:
                tail = (stderr_data or "").strip().splitlines()
                tail_text = tail[-1][:200] if tail else "no diagnostics"
                return ExecutionResult(
                    ok=False,
                    error=f"SandboxError: exit {proc.returncode}: {tail_text}",
                    stdout=stdout_data[: self.output_bytes],
                    stderr=stderr_data[: self.output_bytes],
                    duration_ms=duration_ms,
                    limits_enforced=empty_limits,
                )
            return ExecutionResult(
                ok=False,
                error="SandboxError: empty result",
                stdout=stdout_data[: self.output_bytes],
                stderr=stderr_data[: self.output_bytes],
                duration_ms=duration_ms,
                limits_enforced=empty_limits,
            )

        observed = payload.get("limits") or {}
        limits = {
            "cpu": bool(observed.get("cpu")),
            "memory": bool(observed.get("memory")),
            "nofile": bool(observed.get("nofile")),
            "network": bool(payload.get("network_isolated")),
        }
        network = bool(payload.get("network_isolated"))
        if payload.get("ok"):
            result, kind, text, type_name = _decode_payload(payload)
            return ExecutionResult(
                ok=True,
                result=result,
                repr=text,
                serialized_kind=kind,
                type_name=type_name,
                stdout=(payload.get("stdout") or "")[: self.output_bytes],
                stderr=(payload.get("stderr") or "")[: self.output_bytes],
                input_after=payload.get("input_after"),
                duration_ms=duration_ms,
                limits_enforced=limits,
                network_isolated=network,
            )
        return ExecutionResult(
            ok=False,
            error=str(payload.get("error", "execution failed"))[:300],
            stdout=(payload.get("stdout") or "")[: self.output_bytes],
            stderr=(payload.get("stderr") or "")[: self.output_bytes],
            duration_ms=duration_ms,
            limits_enforced=limits,
            network_isolated=network,
        )

    def run_batch(
        self,
        code: str,
        fn_name: str,
        cases: list[list[Any]],
        timeout: float | None = None,
        per_case_timeout: float | None = None,
    ) -> list[ExecutionResult]:
        """Execute every case in one child process, with a per-case time budget.

        This removes the interpreter start-up cost from every probe (the dominant cost of
        ``verify_candidate`` on small tests) while still isolating cases from one another:
        each case re-executes the module in a fresh namespace.
        """
        try:
            parse_and_check(code, self.policy)
        except (CodePolicyError, SyntaxError) as exc:
            return [ExecutionResult(ok=False, error=f"PolicyViolation: {exc}") for _ in cases]
        if not cases:
            return []
        timeout = timeout or self.wall_timeout_s
        per_case = float(per_case_timeout or min(2.0, timeout))
        sandbox_dir = tempfile.mkdtemp(prefix="vexx_sandbox_")
        start = time.monotonic()
        try:
            payload = {
                "schema_version": 2,
                "mode": "batch",
                "code": code,
                "fn": fn_name,
                "cases": cases,
                "per_case_seconds": per_case,
                "limits": self._limits_payload(),
                "parent_netns": self._parent_netns(),
            }
            proc, stdout_data, stderr_data, overflowed, timed_out = self._spawn(payload, sandbox_dir, timeout)
        except subprocess.TimeoutExpired:
            return [
                ExecutionResult(ok=False, error=f"TimeoutError: exceeded {timeout:.1f}s")
                for _ in cases
            ]
        except Exception as exc:
            return [ExecutionResult(ok=False, error=f"SandboxError: {exc}") for _ in cases]
        finally:
            shutil.rmtree(sandbox_dir, ignore_errors=True)

        duration_ms = (time.monotonic() - start) * 1000
        if timed_out:
            return [
                ExecutionResult(ok=False, error=f"TimeoutError: exceeded {timeout:.1f}s")
                for _ in cases
            ]
        if overflowed:
            return [
                ExecutionResult(ok=False, error=f"OutputLimit: > {self.output_bytes} bytes")
                for _ in cases
            ]
        lines = [ln for ln in (stdout_data or "").splitlines() if ln.strip()]
        payload: dict[str, Any] | None = None
        if lines:
            try:
                candidate = json.loads(lines[-1])
                if isinstance(candidate, dict) and "cases" in candidate:
                    payload = candidate
            except json.JSONDecodeError:
                payload = None
        if payload is None or "cases" not in payload:
            tail = (stderr_data or "").strip().splitlines()
            tail_text = tail[-1][:200] if tail else "no diagnostics"
            return [
                ExecutionResult(ok=False, error=f"SandboxError: {tail_text}", duration_ms=duration_ms)
                for _ in cases
            ]
        observed = payload.get("limits") or {}
        limits = {
            "cpu": bool(observed.get("cpu")),
            "memory": bool(observed.get("memory")),
            "nofile": bool(observed.get("nofile")),
            "network": bool(payload.get("network_isolated")),
        }
        network = bool(payload.get("network_isolated"))
        results: list[ExecutionResult] = []
        for entry in payload.get("cases") or []:
            if not isinstance(entry, dict):
                results.append(ExecutionResult(ok=False, error="SandboxError: malformed case"))
                continue
            if entry.get("ok"):
                result, kind, text, type_name = _decode_payload(entry)
                results.append(
                    ExecutionResult(
                        ok=True,
                        result=result,
                        repr=text,
                        serialized_kind=kind,
                        type_name=type_name,
                        input_after=entry.get("input_after"),
                        duration_ms=duration_ms / max(1, len(cases)),
                        limits_enforced=limits,
                        network_isolated=network,
                    )
                )
            else:
                results.append(
                    ExecutionResult(
                        ok=False,
                        error=str(entry.get("error", "execution failed"))[:300],
                        duration_ms=duration_ms / max(1, len(cases)),
                        limits_enforced=limits,
                        network_isolated=network,
                    )
                )
        while len(results) < len(cases):
            results.append(ExecutionResult(ok=False, error="SandboxError: missing case result"))
        return results[: len(cases)]

    def run_experiment(self, code: str, timeout: float | None = None) -> ExecutionResult:
        return self.run(code, mode="experiment_script", timeout=timeout)
