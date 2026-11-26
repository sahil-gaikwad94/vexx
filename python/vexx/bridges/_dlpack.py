"""Tensor hand-off helpers: keep the caller's container type without a needless copy.

``VexxCodeLogitsProcessor`` computes a NumPy mask, but Hugging Face, vLLM and SGLang all hand
it a ``torch.Tensor``.  Returning a NumPy array crashes ``generate()``; returning a freshly
allocated CPU tensor works but throws away the mask copy we already made.  On CPU with a
matching dtype, DLPack lets both sides share the same buffer (``torch.from_dlpack``), and only
when a device transfer or dtype conversion is genuinely required do we copy.

Nothing here imports ``torch`` at module load time: the package must keep working on machines
that only have NumPy.
"""

from __future__ import annotations

from typing import Any

__all__ = ["is_torch_tensor", "to_torch_like"]


def is_torch_tensor(value: Any) -> bool:
    """Duck-typed torch detection (avoids importing torch just to answer the question)."""
    return hasattr(value, "detach") and hasattr(value, "to") and hasattr(value, "dtype")


def _dtype_name(dtype: Any) -> str:
    # ``np.float32`` is a *type* (``str`` gives ``<class 'numpy.float32'>``) while a torch
    # dtype stringifies as ``torch.float32``; both reduce to the bare name here.
    return getattr(dtype, "__name__", str(dtype)).rsplit(".", 1)[-1]


def to_torch_like(array: Any, reference: Any) -> Any:
    """Return ``array`` as a tensor matching ``reference``'s dtype/device.

    * No ``reference`` (or no torch installed) -> the array is returned unchanged.
    * CPU reference with a matching dtype -> DLPack zero-copy view over the same buffer.
    * Otherwise -> one explicit ``as_tensor`` copy onto the target device/dtype.
    """
    if reference is None or not is_torch_tensor(reference):
        return array
    try:
        import torch
    except ImportError:  # pragma: no cover - depends on the host environment
        return array

    device = getattr(reference, "device", None)
    dtype = getattr(reference, "dtype", None)
    array_dtype = getattr(array, "dtype", None)
    same_dtype = (
        dtype is not None
        and array_dtype is not None
        and _dtype_name(dtype) == str(array_dtype)
    )
    on_cpu = getattr(device, "type", "cpu") == "cpu"

    if on_cpu and same_dtype and hasattr(array, "__dlpack__"):
        try:
            return torch.from_dlpack(array)
        except Exception:
            # DLPack is an optimisation, never a correctness requirement.
            pass
    kwargs: dict[str, Any] = {}
    if dtype is not None:
        kwargs["dtype"] = dtype
    if device is not None:
        kwargs["device"] = device
    return torch.as_tensor(array, **kwargs)
