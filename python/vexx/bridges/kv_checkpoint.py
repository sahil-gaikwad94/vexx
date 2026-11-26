"""KV-cache checkpointing for live inference servers (vLLM V1 and SGLang).

Vexx's jump-back search rewinds the model to an earlier prefix and re-decodes from there.  On a
CPU test harness that is free; on a GPU server it means the KV cache holds blocks for tokens
that no longer exist.  These adapters snapshot the *cache-side* state (block tables for vLLM,
radix-cache prefixes for SGLang) and hand it back to the host so a rewind releases exactly the
blocks the server allocated and keeps the ones a cached prefix can reuse.

Both adapters are duck-typed: hosts are third-party objects and their private layouts change
between releases, so every hook is resolved defensively and a missing hook raises
:class:`UnsupportedHost` with the exact attribute that was expected instead of silently doing
nothing.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "KVCheckpoint",
    "SGLangCheckpointAdapter",
    "UnsupportedHost",
    "VLLMCheckpointAdapter",
    "adapter_for",
]


class UnsupportedHost(TypeError):
    """The host object does not expose the hook this adapter needs."""


@dataclass(frozen=True)
class KVCheckpoint:
    """A rewind point: the prompt/decoded tokens and the blocks backing them."""

    seq_id: str
    num_tokens: int
    token_ids: tuple[int, ...]
    block_ids: tuple[int, ...] = ()
    prefix_len: int | None = None
    format: str = "vexx.kv.v1"

    def rewind_length(self, new_len: int) -> int:
        if new_len < 0 or new_len > self.num_tokens:
            raise ValueError(
                f"cannot rewind {self.seq_id!r} to {new_len} tokens; checkpoint holds "
                f"{self.num_tokens}"
            )
        return new_len


def _attr_chain(obj: Any, *names: str) -> Any:
    for name in names:
        if obj is None:
            return None
        obj = getattr(obj, name, None)
    return obj


class _CheckpointStore:
    def __init__(self, capacity: int = 8) -> None:
        self.capacity = max(1, int(capacity))
        self._items: dict[str, KVCheckpoint] = {}

    def put(self, checkpoint: KVCheckpoint) -> KVCheckpoint:
        self._items.pop(checkpoint.seq_id, None)
        self._items[checkpoint.seq_id] = checkpoint
        while len(self._items) > self.capacity:
            oldest = next(iter(self._items))
            self._items.pop(oldest, None)
        return checkpoint

    def get(self, seq_id: str) -> KVCheckpoint:
        try:
            return self._items[str(seq_id)]
        except KeyError as error:
            raise KeyError(f"no KV checkpoint for {seq_id!r}") from error

    def drop(self, seq_id: str) -> None:
        self._items.pop(str(seq_id), None)


class VLLMCheckpointAdapter:
    """Snapshot/rewind vLLM V1 block tables.

    The host may be a ``BlockManager``/``KVCacheManager`` itself or anything holding one
    (``scheduler.block_manager``, ``engine_core.scheduler.block_manager``).  Blocks are read
    from ``block_tables[seq_id]`` (dict or mapping-like) and released through the first of
    ``free``, ``free_block`` or ``block_pool.free_block`` that exists.
    """

    def __init__(self, host: Any, *, capacity: int = 8) -> None:
        self.host = host
        self.store = _CheckpointStore(capacity)
        manager = _attr_chain(host, "block_manager")
        if manager is None:
            manager = _attr_chain(host, "scheduler", "block_manager")
        self.block_manager = host if hasattr(host, "block_tables") else manager
        if self.block_manager is None:
            raise UnsupportedHost(
                "vLLM host exposes no block_tables; expected host.block_manager or "
                "host.scheduler.block_manager"
            )

    # ------------------------------------------------------------------ reading

    def block_table(self, seq_id: Any) -> list[int]:
        tables = getattr(self.block_manager, "block_tables", None)
        if tables is None:
            raise UnsupportedHost("vLLM block manager has no block_tables mapping")
        try:
            blocks = tables[str(seq_id)]
        except (KeyError, TypeError):
            try:
                blocks = tables[seq_id]
            except (KeyError, TypeError) as error:
                raise KeyError(f"no block table for sequence {seq_id!r}") from error
        out: list[int] = []
        for block in blocks:
            out.append(int(getattr(block, "block_id", block)))
        return out

    # ------------------------------------------------------------------ writing

    def _release(self, block_id: int) -> None:
        manager = self.block_manager
        for name in ("free", "free_block"):
            method = getattr(manager, name, None)
            if callable(method):
                method(block_id)
                return
        pool = getattr(manager, "block_pool", None)
        method = getattr(pool, "free_block", None)
        if callable(method):
            method(block_id)
            return
        raise UnsupportedHost("vLLM block manager exposes no free/free_block/block_pool")

    def capture(self, seq_id: Any, token_ids: Sequence[int]) -> KVCheckpoint:
        checkpoint = KVCheckpoint(
            seq_id=str(seq_id),
            num_tokens=len(token_ids),
            token_ids=tuple(int(token) for token in token_ids),
            block_ids=tuple(self.block_table(seq_id)),
        )
        return self.store.put(checkpoint)

    def rewind(self, seq_id: Any, new_len: int, *, block_size: int | None = None) -> KVCheckpoint:
        """Release the blocks past ``new_len`` tokens and return the trimmed checkpoint."""
        checkpoint = self.store.get(str(seq_id))
        checkpoint.rewind_length(new_len)
        size = int(block_size or getattr(self.block_manager, "block_size", 1) or 1)
        keep_blocks = (new_len + size - 1) // size
        current = self.block_table(seq_id)
        for block_id in current[keep_blocks:]:
            self._release(block_id)
        trimmed = KVCheckpoint(
            seq_id=checkpoint.seq_id,
            num_tokens=new_len,
            token_ids=checkpoint.token_ids[:new_len],
            block_ids=tuple(current[:keep_blocks]),
            prefix_len=checkpoint.prefix_len,
        )
        return self.store.put(trimmed)

    def release(self, seq_id: Any) -> None:
        self.store.drop(str(seq_id))


class SGLangCheckpointAdapter:
    """Jump-forward prefill and checkpointing against SGLang's ``RadixCache``.

    ``match_prefix`` is content addressed, so a rewind is just "forget the tokens past the
    prefix"; the blocks stay in the radix tree and are reused on the next pass.  ``capture``
    records how much of a token prefix the cache already holds so a retry can skip prefill.
    """

    def __init__(self, host: Any, *, capacity: int = 8) -> None:
        self.host = host
        self.store = _CheckpointStore(capacity)
        cache = _attr_chain(host, "radix_cache") or _attr_chain(host, "tree_cache")
        if cache is None:
            cache = _attr_chain(host, "scheduler", "radix_cache")
        self.radix_cache = host if hasattr(host, "match_prefix") else cache
        if self.radix_cache is None:
            raise UnsupportedHost(
                "SGLang host exposes no RadixCache; expected host.radix_cache or "
                "host.tree_cache"
            )

    def match_prefix(self, token_ids: Sequence[int]) -> int:
        """Number of cached leading tokens ``token_ids`` already shares with the radix tree."""
        matcher = getattr(self.radix_cache, "match_prefix", None)
        if not callable(matcher):
            raise UnsupportedHost("SGLang radix cache has no match_prefix")
        result = matcher([int(token) for token in token_ids])
        # Real hosts return ``MatchResult(device_indices=..., last_device_node=...)``.
        indices = getattr(result, "device_indices", None)
        if indices is None:
            indices = getattr(result, "indices", None)
        if indices is None:
            indices = result
        try:
            return len(indices)
        except TypeError:
            return int(indices)

    def capture(self, seq_id: Any, token_ids: Sequence[int]) -> KVCheckpoint:
        tokens = tuple(int(token) for token in token_ids)
        prefix_len = self.match_prefix(tokens)
        checkpoint = KVCheckpoint(
            seq_id=str(seq_id),
            num_tokens=len(tokens),
            token_ids=tokens,
            prefix_len=prefix_len,
        )
        return self.store.put(checkpoint)

    def rewind(self, seq_id: Any, new_len: int) -> KVCheckpoint:
        checkpoint = self.store.get(str(seq_id))
        checkpoint.rewind_length(new_len)
        trimmed = KVCheckpoint(
            seq_id=checkpoint.seq_id,
            num_tokens=new_len,
            token_ids=checkpoint.token_ids[:new_len],
            prefix_len=checkpoint.prefix_len,
        )
        return self.store.put(trimmed)

    def jump_forward(self, token_ids: Sequence[int]) -> int:
        """Tokens whose KV state the cache already holds (safe to skip during prefill)."""
        return self.match_prefix(token_ids)

    def release(self, seq_id: Any) -> None:
        self.store.drop(str(seq_id))


def adapter_for(host: Any, **kwargs: Any) -> VLLMCheckpointAdapter | SGLangCheckpointAdapter:
    """Pick the adapter that matches the host's available hooks."""
    if hasattr(host, "match_prefix") or hasattr(host, "radix_cache") or hasattr(host, "tree_cache"):
        return SGLangCheckpointAdapter(host, **kwargs)
    return VLLMCheckpointAdapter(host, **kwargs)
