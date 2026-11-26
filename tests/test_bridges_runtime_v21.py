"""Runtime-bridge tests: DLPack hand-off and KV-cache checkpoint adapters.

Torch, vLLM and SGLang are not installed in this environment, so the container types and
server hosts here are faithful fakes.  They pin the *contract* (which hook is used, what is
released, what is reused) rather than the third-party implementation.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from vexx.bridges._dlpack import is_torch_tensor, to_torch_like
from vexx.bridges.kv_checkpoint import (
    SGLangCheckpointAdapter,
    UnsupportedHost,
    VLLMCheckpointAdapter,
    adapter_for,
)


class _FakeDevice:
    def __init__(self, kind: str = "cpu"):
        self.type = kind


class _FakeTensor:
    def __init__(self, dtype=np.float32, device="cpu"):
        self.dtype = dtype
        self.device = _FakeDevice(device)

    def detach(self):
        return self

    def to(self, *args, **kwargs):
        return self


class _FakeTorch(types.SimpleNamespace):
    def __init__(self):
        super().__init__()
        self.calls: list[tuple] = []

    def from_dlpack(self, array):
        self.calls.append(("from_dlpack", getattr(array, "dtype", None)))
        return ("dlpack-view", array.shape)

    def as_tensor(self, array, **kwargs):
        self.calls.append(("as_tensor", kwargs))
        return ("copied", array.shape)


@pytest.fixture()
def fake_torch(monkeypatch):
    module = _FakeTorch()
    monkeypatch.setitem(sys.modules, "torch", module)
    return module


def test_detects_torch_like_containers():
    assert is_torch_tensor(_FakeTensor())
    assert not is_torch_tensor(np.zeros(2))


def test_cpu_mask_is_handed_off_zero_copy(fake_torch):
    mask = np.zeros((2, 5), dtype=np.float32)
    result = to_torch_like(mask, _FakeTensor(dtype=np.float32, device="cpu"))
    assert result == ("dlpack-view", mask.shape)
    assert fake_torch.calls == [("from_dlpack", np.dtype("float32"))]


def test_device_or_dtype_change_copies_once(fake_torch):
    mask = np.zeros((1, 3), dtype=np.float32)
    result = to_torch_like(mask, _FakeTensor(dtype=np.float64, device="cpu"))
    assert result == ("copied", mask.shape)
    assert fake_torch.calls[0][0] == "as_tensor"
    assert fake_torch.calls[0][1]["dtype"] is np.float64

    fake_torch.calls.clear()
    to_torch_like(mask, _FakeTensor(dtype=np.float32, device="cuda"))
    assert fake_torch.calls[0][0] == "as_tensor"


def test_without_torch_the_array_is_returned_unchanged(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    mask = np.zeros((1, 3), dtype=np.float32)
    assert to_torch_like(mask, _FakeTensor()) is mask


# --------------------------------------------------------------------------- KV adapters


class _Block:
    def __init__(self, block_id: int):
        self.block_id = block_id


class _FakeBlockManager:
    block_size = 2

    def __init__(self, tables):
        self.block_tables = {seq: [_Block(i) for i in blocks] for seq, blocks in tables.items()}
        self.freed: list[int] = []

    def free(self, block_id):
        self.freed.append(block_id)


def test_vllm_checkpoint_releases_only_the_blocks_past_the_rewind_point():
    manager = _FakeBlockManager({"s0": [10, 11, 12]})
    adapter = VLLMCheckpointAdapter(manager)
    checkpoint = adapter.capture("s0", list(range(5)))
    assert checkpoint.block_ids == (10, 11, 12)
    assert checkpoint.num_tokens == 5

    trimmed = adapter.rewind("s0", 3)
    assert trimmed.num_tokens == 3
    assert trimmed.token_ids == (0, 1, 2)
    assert manager.freed == [12]
    with pytest.raises(ValueError):
        adapter.rewind("s0", 9)


class _FakeMatch:
    def __init__(self, length: int):
        self.device_indices = list(range(length))


class _FakeRadixCache:
    def __init__(self, cached: int):
        self.cached = cached

    def match_prefix(self, token_ids):
        return _FakeMatch(min(self.cached, len(token_ids)))


def test_sglang_jump_forward_reports_the_reusable_prefix():
    cache = _FakeRadixCache(cached=4)
    adapter = SGLangCheckpointAdapter(types.SimpleNamespace(radix_cache=cache))
    assert adapter.jump_forward([1, 2, 3, 4, 5]) == 4
    checkpoint = adapter.capture("s1", [1, 2, 3, 4, 5])
    assert checkpoint.prefix_len == 4
    assert adapter.rewind("s1", 2).token_ids == (1, 2)


def test_hosts_without_the_hooks_are_rejected_loudly():
    with pytest.raises(UnsupportedHost):
        VLLMCheckpointAdapter(types.SimpleNamespace())
    with pytest.raises(UnsupportedHost):
        SGLangCheckpointAdapter(types.SimpleNamespace())


def test_adapter_for_picks_by_available_hooks():
    assert isinstance(
        adapter_for(types.SimpleNamespace(radix_cache=_FakeRadixCache(1))), SGLangCheckpointAdapter
    )
    assert isinstance(
        adapter_for(_FakeBlockManager({"s": [0]})), VLLMCheckpointAdapter
    )
