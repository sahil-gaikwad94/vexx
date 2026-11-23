"""Candidate verification, consensus selection, differential fuzzing."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from .config import CodeEngineConfig
from .policy import check_ast_policy
from .sandbox import SandboxRunner


@dataclass
class CandidateReport:
    source: str
    source_hash: str
    syntax_ok: bool
    policy_ok: bool
    contract_ok: bool
    passed: int
    total: int
    errors: list[str]
    mutation_free: bool
    sandbox_profile: dict[str, Any]
    runtime_ms: float
    warnings: list[str] = field(default_factory=list)


def _hash_source(src: str) -> str:
    return hashlib.sha256(src.encode("utf-8")).hexdigest()


def _contract_ok(code: str, fn_name: str, test_cases: list[dict[str, Any]]) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    node = next(
        (c for c in ast.walk(tree) if isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef)) and c.name == fn_name),
        None,
    )
    if node is None:
        return False
    args = node.args
    positional = list(args.posonlyargs) + list(args.args)
    if positional and positional[0].arg in {"self", "cls"}:
        positional = positional[1:]
    required = max(0, len(positional) - len(args.defaults))
    strict_max = None if args.vararg is not None else len(positional)
    for tc in test_cases:
        n_args = len(tc.get("input") or [])
        if n_args < required:
            return False
        if strict_max is not None and n_args > strict_max:
            return False
    return True


def _deep_copy_json(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _canonical(value: Any) -> Any:
    """Canonical JSON-compatible representation for comparison."""
    if isinstance(value, tuple):
        return [_canonical(v) for v in value]
    if isinstance(value, set):
        return sorted(_canonical(v) for v in value)
    if isinstance(value, dict):
        return {str(k): _canonical(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, float):
        if value != value:  # nan
            return None
        return value
    return value


class CandidateVerifier:
    def __init__(self, config: CodeEngineConfig, runner: SandboxRunner | None = None) -> None:
        self.config = config
        self.runner = runner or SandboxRunner(
            wall_timeout_s=config.sandbox_wall_timeout_s,
            cpu_seconds=config.sandbox_cpu_seconds,
            memory_bytes=config.sandbox_memory_bytes,
            output_bytes=config.sandbox_output_bytes,
        )

    def verify_candidate(
        self,
        source: str,
        function_name: str,
        test_cases: list[dict[str, Any]],
    ) -> CandidateReport:
        # Validate test_cases schema.
        for idx, tc in enumerate(test_cases):
            if not isinstance(tc, dict) or "input" not in tc:
                raise ValueError(f"test case {idx} missing 'input'")
            if not isinstance(tc["input"], list):
                raise ValueError(f"test case {idx} 'input' must be a list")
            has_expected = "expected" in tc
            has_one_of = "expected_one_of" in tc
            if has_expected == has_one_of:
                raise ValueError(f"test case {idx} must have exactly one of 'expected'/'expected_one_of'")

        syntax_ok = True
        policy_ok = True
        try:
            check_ast_policy(source, None)
        except SyntaxError:
            syntax_ok = False
            policy_ok = False
        except Exception:
            policy_ok = False

        contract_ok = _contract_ok(source, function_name, test_cases) if syntax_ok else False

        passed = 0
        errors: list[str] = []
        mutation_free = True
        import time
        t0 = time.time()
        if syntax_ok and policy_ok:
            for idx, tc in enumerate(test_cases):
                original = _deep_copy_json(tc["input"])
                res = self.runner.run(source, function_name, copy.deepcopy(original))
                if not res.ok:
                    errors.append(f"Test {idx} raised {res.error}")
                    continue
                try:
                    result = res.result
                    if result is None and res.repr is not None:
                        import ast as _ast
                        result = _ast.literal_eval(res.repr)
                except Exception:
                    errors.append(f"Test {idx} returned a non-literal: {res.repr!r}")
                    continue
                expected_ok = False
                if "expected_one_of" in tc:
                    expected_ok = any(_canonical(result) == _canonical(exp) for exp in tc["expected_one_of"])
                else:
                    expected_ok = _canonical(result) == _canonical(tc.get("expected"))
                after = res.input_after
                try:
                    after_val = after if not isinstance(after, str) else __import__("ast").literal_eval(after)
                except Exception:
                    after_val = after
                mutation_ok = _canonical(after_val) == _canonical(original)
                if expected_ok and mutation_ok:
                    passed += 1
                elif not mutation_ok:
                    errors.append(f"Test {idx} mutated its input")
                    mutation_free = False
                else:
                    errors.append(f"Test {idx} failed: got {result!r}, expected {tc.get('expected', tc.get('expected_one_of'))!r}")
        runtime_ms = (time.time() - t0) * 1000

        if not test_cases:
            # zero tests => not a pass
            pass

        report = CandidateReport(
            source=source,
            source_hash=_hash_source(source),
            syntax_ok=syntax_ok,
            policy_ok=policy_ok,
            contract_ok=contract_ok,
            passed=passed,
            total=len(test_cases),
            errors=errors,
            mutation_free=mutation_free,
            sandbox_profile={"network_isolated": self.runner._network_isolated() if hasattr(self.runner, "_network_isolated") else False},
            runtime_ms=runtime_ms,
        )
        return report

    def select_verified(
        self,
        candidates: list[str],
        function_name: str,
        test_cases: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        scored: list[dict[str, Any]] = []
        for code in candidates:
            rep = self.verify_candidate(code, function_name, test_cases)
            scored.append({
                "source": code,
                "source_hash": rep.source_hash,
                "syntax_ok": rep.syntax_ok,
                "policy_ok": rep.policy_ok,
                "contract_ok": rep.contract_ok,
                "passed": rep.passed,
                "total": rep.total,
                "errors": rep.errors,
                "mutation_free": rep.mutation_free,
                "runtime_ms": rep.runtime_ms,
            })
        # Append preference if exactly one full pass and at least one executed loser.
        winners = [s for s in scored if s["total"] > 0 and s["passed"] == s["total"]]
        losers = [s for s in scored if s["total"] > 0 and s["passed"] < s["total"]]
        if len(winners) == 1 and losers:
            # preference record stored by caller; here just annotate
            for s in scored:
                s["preference_winner"] = winners[0]["source_hash"]
        return scored

    def select_candidate(
        self,
        candidates: list[str],
        function_name: str,
        inputs: list[list[Any]],
        timeout: float | None = None,
    ) -> dict[str, Any]:
        if not inputs:
            raise ValueError("MBR selection needs at least one probe input")
        timeout = timeout or self.config.sandbox_wall_timeout_s
        scoreboard: list[dict[str, Any]] = []
        for index, code in enumerate(candidates):
            policy_ok = True
            try:
                check_ast_policy(code, None)
            except Exception:
                policy_ok = False
            contract_ok = policy_ok and _contract_ok(code, function_name, [{"input": list(inp)} for inp in inputs])
            outputs: list[str] = []
            mutation_free = True
            executed = contract_ok
            error: str | None = None
            if executed:
                for inp in inputs:
                    res = self.runner.run(code, function_name, copy.deepcopy(inp), timeout=timeout)
                    if not res.ok:
                        executed = False
                        error = res.error
                        outputs = []
                        break
                    outputs.append(json.dumps(_canonical(res.result), sort_keys=True))
                    after = res.input_after
                    try:
                        after_val = after if not isinstance(after, str) else __import__("ast").literal_eval(after)
                    except Exception:
                        after_val = after
                    if _canonical(after_val) != _canonical(inp):
                        mutation_free = False
            # ast depth
            try:
                tree = ast.parse(code)
                depth = 0
                stack = [(tree, 1)]
                while stack:
                    node, d = stack.pop()
                    depth = max(depth, d)
                    for ch in ast.iter_child_nodes(node):
                        stack.append((ch, d + 1))
            except SyntaxError:
                depth = 1 << 20
            scoreboard.append({
                "index": index,
                "code": code,
                "policy_ok": policy_ok,
                "contract_ok": contract_ok,
                "executed": executed,
                "outputs": outputs,
                "mutation_free": mutation_free,
                "ast_depth": depth,
                "error": error,
            })

        survivors = [r for r in scoreboard if r["executed"] and r["outputs"]]
        eligible = [r for r in survivors if r["mutation_free"]] or survivors
        counts: dict[tuple[str, ...], int] = {}
        for r in eligible:
            key = tuple(r["outputs"])
            counts[key] = counts.get(key, 0) + 1
        best_votes = max(counts.values()) if counts else 0
        tied = {k for k, v in counts.items() if v == best_votes}
        pool = [r for r in eligible if tuple(r["outputs"]) in tied]
        pool.sort(key=lambda r: (r["ast_depth"], len(r["code"]), r["index"]))
        winner = pool[0] if pool else None
        best_outputs = tuple(winner["outputs"]) if winner else ()
        votes = counts.get(best_outputs, 0)
        agreement = (votes / len(eligible)) if eligible else 0.0
        consensus = bool(winner is not None and votes >= 2 and votes * 2 > len(eligible))
        return {
            "winner": winner["code"] if winner else None,
            "winner_index": winner["index"] if winner else None,
            "outputs": list(best_outputs),
            "votes": votes,
            "survivors": len(eligible),
            "agreement": agreement,
            "consensus": consensus,
            "scoreboard": scoreboard,
        }

    def fuzz_against_oracle(
        self,
        candidate: str,
        reference: str,
        function_name: str,
        inputs: list[list[Any]] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        probes = list(inputs) if inputs is not None else [
            [], [0], [-1, -2, -1], [3, 3, 3], [1, 2, 3, 4, 5], [2, 1],
        ]
        # Normalize: each probe is an arg list.
        probe_inputs = [list(p) for p in probes]
        if not probe_inputs:
            raise ValueError("property fuzzing needs at least one probe input")
        timeout = timeout or self.config.sandbox_wall_timeout_s
        mismatches: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        for probe in probe_inputs:
            ref_res = self.runner.run(reference, function_name, copy.deepcopy(probe), timeout=timeout)
            if not ref_res.ok:
                errors.append({"input": probe, "side": "reference", "error": ref_res.error})
                continue
            cand_res = self.runner.run(candidate, function_name, copy.deepcopy(probe), timeout=timeout)
            if not cand_res.ok:
                errors.append({"input": probe, "side": "candidate", "error": cand_res.error})
                continue
            try:
                exp = ref_res.result if ref_res.result is not None else __import__("ast").literal_eval(ref_res.repr)
                act = cand_res.result if cand_res.result is not None else __import__("ast").literal_eval(cand_res.repr)
                agrees = _canonical(act) == _canonical(exp)
            except Exception:
                agrees = ref_res.repr == cand_res.repr
            if not agrees:
                mismatches.append({"input": probe, "expected": ref_res.repr, "actual": cand_res.repr})
        return {
            "fn": function_name,
            "checked": len(probe_inputs),
            "agrees": not mismatches and not errors,
            "mismatches": mismatches,
            "errors": errors,
            "probes": probe_inputs,
            "label": "differentially_agrees_on_probes",
        }