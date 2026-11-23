"""RLVR rewards and DPO export for Vexx 2.1."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from typing import Any

from ..code_engine.candidate import CandidateVerifier

REWARD_LADDER = {
    "syntax_or_policy": -0.5,
    "contract_violation": -0.2,
    "input_mutation": -0.2,
    "no_tests_passed": 0.0,
    "partial_pass": None,  # 0.3 * pass_rate
    "full_pass": 1.0,
}


def execution_reward(item: dict[str, Any]) -> tuple[float, str]:
    total = int(item.get("total") or 0)
    passed = int(item.get("passed") or 0)
    if not bool(item.get("syntax_ok", True)):
        return -0.5, "syntax_or_policy"
    if not bool(item.get("contract_ok", True)):
        return -0.2, "contract_violation"
    if not bool(item.get("mutation_free", True)):
        return -0.2, "input_mutation"
    if total <= 0 or passed <= 0:
        return 0.0, "no_tests_passed"
    if passed >= total:
        return 1.0, "full_pass"
    return 0.3 * (float(passed) / float(total)), "partial_pass"


def compute_code_rewards(
    candidates: list[str],
    function_name: str,
    test_cases: list[dict[str, Any]],
    verifier: CandidateVerifier | None = None,
    config: Any = None,
) -> dict[str, Any]:
    if verifier is None:
        from ..code_engine.config import CodeEngineConfig
        verifier = CandidateVerifier(config or CodeEngineConfig())
    details = verifier.select_verified(candidates, function_name, test_cases)
    raw: list[float] = []
    for item in details:
        reward, reason = execution_reward(item)
        item["reward"] = reward
        item["reward_reason"] = reason
        raw.append(reward)

    n = len(raw)
    if n == 0:
        return {
            "rewards": [],
            "advantages": [],
            "mean_reward": 0.0,
            "std_reward": 0.0,
            "zero_variance_group": True,
            "details": details,
        }
    mean_r = sum(raw) / n
    var = sum((r - mean_r) ** 2 for r in raw) / n  # population std
    std_r = math.sqrt(var)
    if std_r < 1e-8:
        advantages = [0.0 for _ in raw]
        zero_var = True
    else:
        advantages = [(r - mean_r) / (std_r + 1e-6) for r in raw]
        zero_var = False
    for item, adv in zip(details, advantages):
        item["advantage"] = adv
    return {
        "rewards": raw,
        "advantages": advantages,
        "mean_reward": mean_r,
        "std_reward": std_r,
        "zero_variance_group": zero_var,
        "details": details,
    }


def export_code_preferences(
    path: str,
    rows: list[dict[str, Any]],
    schema_version: int = 1,
    redact_source: bool = False,
) -> dict[str, Any]:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    # Validate rows.
    validated: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("row must be a dict")
        r = dict(row)
        r.setdefault("schema_version", schema_version)
        # redact source if requested
        if redact_source and "winner" in r and isinstance(r["winner"], str):
            r["winner_hash"] = hashlib.sha256(r["winner"].encode()).hexdigest()
            del r["winner"]
        if redact_source and "losers" in r and isinstance(r["losers"], list):
            r["loser_hashes"] = [hashlib.sha256(str(x).encode()).hexdigest() for x in r["losers"]]
            del r["losers"]
        validated.append(r)

    # Atomic write.
    tmp_fd, tmp_path = tempfile.mkstemp(prefix=".dpo_", dir=os.path.dirname(os.path.abspath(path)) or ".")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            for row in validated:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    # Manifest.
    manifest_path = path + ".manifest.json"
    manifest = {
        "schema_version": schema_version,
        "count": len(validated),
        "sha256": _file_sha256(path),
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    return manifest


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()