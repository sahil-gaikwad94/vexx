"""Bounded multi-cell experiment execution for Vexx 2.1."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

from .config import CodeEngineConfig
from .policy import CodePolicy, check_ast_policy
from .sandbox import SandboxRunner


@dataclass
class ExperimentResult:
    ok: bool
    output: str | None = None
    error: str | None = None
    retained: list[str] = field(default_factory=list)


class ExperimentEpisode:
    def __init__(self, config: CodeEngineConfig, policy: CodePolicy | None = None, runner: SandboxRunner | None = None) -> None:
        self.config = config
        self.policy = policy or CodePolicy()
        self.runner = runner or SandboxRunner(
            wall_timeout_s=config.sandbox_wall_timeout_s,
            cpu_seconds=config.sandbox_cpu_seconds,
            memory_bytes=config.sandbox_memory_bytes,
            output_bytes=config.sandbox_output_bytes,
        )
        self.preamble: list[str] = []
        self.cell_count = 0
        self._episode_id = 0

    def reset(self) -> int:
        self.preamble.clear()
        self.cell_count = 0
        self._episode_id += 1
        return self._episode_id

    @property
    def episode_id(self) -> int:
        return self._episode_id

    def _allowed_retained(self, node: ast.AST) -> list[str]:
        kept: list[str] = []
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            kept.append(ast.unparse(node))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            kept.append(ast.unparse(node))
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id == "RESULT" for t in targets):
                return kept
            # Only simple immutable constants.
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                kept.append(ast.unparse(node))
        return kept

    def run_cell(self, script: str) -> ExperimentResult:
        if len(script) > self.config.max_experiment_source_chars:
            return ExperimentResult(ok=False, error="source too large")
        if self.cell_count >= self.config.max_experiment_cells:
            return ExperimentResult(ok=False, error="cell limit reached")
        try:
            check_ast_policy(script, self.policy)
        except Exception as exc:
            return ExperimentResult(ok=False, error=f"policy: {exc}")
        combined_parts = list(self.preamble) + [script]
        combined = "\n".join(combined_parts)
        try:
            check_ast_policy(combined, self.policy)
        except Exception as exc:
            return ExperimentResult(ok=False, error=f"policy: {exc}")
        if sum(len(c) for c in combined_parts) > self.config.max_experiment_source_chars:
            return ExperimentResult(ok=False, error="preamble+source too large")

        res = self.runner.run(combined, mode="experiment_script")
        self.cell_count += 1
        if not res.ok:
            return ExperimentResult(ok=False, error=res.error)

        # Absorb definitions from the cell only.
        try:
            tree = ast.parse(script)
        except SyntaxError:
            tree = None
        retained: list[str] = []
        if tree is not None:
            for node in tree.body:
                retained.extend(self._allowed_retained(node))
        # Enforce preamble size limits.
        self.preamble.extend(retained)
        while len(self.preamble) > self.config.max_experiment_cells:
            self.preamble.pop(0)
        while sum(len(c) for c in self.preamble) > self.config.max_experiment_source_chars:
            if not self.preamble:
                break
            self.preamble.pop(0)

        return ExperimentResult(ok=True, output=res.repr or res.stdout, retained=retained)