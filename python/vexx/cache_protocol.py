"""Backend-neutral cache snapshot/rewind protocol for Vexx 2.1."""

from __future__ import annotations

import copy
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class RewindableCache(Protocol):
    def clone_snapshot(self) -> Any: ...
    def restore_snapshot(self, snapshot: Any) -> Any: ...
    def can_rewind_to(self, token_index: int) -> bool: ...


def clone_cache_generic(cache: Any) -> Any:
    if hasattr(cache, "clone"):
        try:
            return cache.clone()
        except Exception:
            pass
    if isinstance(cache, (list, tuple)):
        return [clone_cache_generic(item) for item in cache]
    if isinstance(cache, dict):
        return {k: clone_cache_generic(v) for k, v in cache.items()}
    try:
        import numpy as np
        if isinstance(cache, np.ndarray):
            return cache.copy()
    except Exception:
        pass
    try:
        import torch
        if hasattr(cache, "detach"):
            return cache.clone()
    except Exception:
        pass
    return copy.deepcopy(cache)


def _contains_recurrent_state(node: Any) -> bool:
    if isinstance(node, dict):
        if "state" in node or "conv" in node:
            return True
        return any(_contains_recurrent_state(v) for v in node.values())
    if isinstance(node, (list, tuple)):
        return any(_contains_recurrent_state(item) for item in node)
    return False


def create_empty_hybrid_cache_v2(
    num_loops: int,
    prelude_layers: int,
    core_layers: int,
    coda_layers: int,
    batch_size: int = 1,
    config_hash: str = "",
) -> dict:
    """Create an empty Schema-2 structured multi-loop hybrid cache bundle."""
    return {
        "schema_version": 2,
        "num_loops": int(num_loops),
        "position": 0,
        "batch_size": int(batch_size),
        "config_hash": str(config_hash),
        "prelude": [None for _ in range(int(prelude_layers))],
        "core_loops": [
            [None for _ in range(int(core_layers))]
            for _ in range(int(num_loops))
        ],
        "coda": [None for _ in range(int(coda_layers))],
    }


def truncate_cache_at_token(cache: Any, tok_idx: int) -> Any:
    tok_idx = int(tok_idx)
    if isinstance(cache, (list, tuple)):
        for item in cache:
            truncate_cache_at_token(item, tok_idx)
        return cache
    if isinstance(cache, dict):
        position = cache.get("position")
        if position is not None and int(position) > tok_idx and _contains_recurrent_state(cache):
            raise ValueError(
                f"cannot truncate a hybrid cache at token {tok_idx}: it is at "
                f"position {int(position)} and carries recurrent state with no "
                "sequence axis; restore an exact snapshot or replay the prefix"
            )
        if "k" in cache and "v" in cache and hasattr(cache["k"], "shape"):
            if cache["k"].shape[2] > tok_idx:
                cache["k"] = cache["k"][:, :, :tok_idx, :]
                cache["v"] = cache["v"][:, :, :tok_idx, :]
        else:
            for key in ("prelude", "core_loops", "coda"):
                if key in cache:
                    truncate_cache_at_token(cache[key], tok_idx)
            if "position" in cache:
                cache["position"] = min(int(cache["position"]), tok_idx)
        return cache
    return cache