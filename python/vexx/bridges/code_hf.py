"""Hugging Face code-generation logits processor for Vexx 2.1."""

from __future__ import annotations

import hashlib
from typing import Any

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is a CAS extra
    np = None

from ..code_engine.controller import VexxCodeController
from ..code_engine.masking import CodeSession, DeadEnd
from ._dlpack import to_torch_like


class VexxCodeLogitsProcessor:
    """HF-style logits processor applying Vexx code masks per batch row.

    ``scores`` keeps its container type: a processor that returns a NumPy array crashes
    ``model.generate()`` on both CPU and GPU tensors.  Only :class:`DeadEnd` is swallowed
    (a row that cannot legally continue is left untouched and counted); every other error
    propagates so a genuinely broken masker is never silently fail-open.
    """

    def __init__(
        self,
        controller: VexxCodeController,
        vocabulary: dict[int, str],
        prompt_length: int = 0,
        mode: str = "python",
    ) -> None:
        self.controller = controller
        self.vocabulary = vocabulary
        self.prompt_length = prompt_length
        self.mode = mode
        self._sessions: dict[int, CodeSession] = {}
        self.dead_ends = 0
        self.tokenizer_hash = self._hash_vocabulary(vocabulary)

    @staticmethod
    def _hash_vocabulary(vocabulary: dict[int, str]) -> str:
        digest = hashlib.sha256()
        for tid in sorted(vocabulary):
            digest.update(str(int(tid)).encode())
            digest.update(b"\0")
            digest.update(str(vocabulary[tid]).encode("utf-8", "replace"))
            digest.update(b"\n")
        return digest.hexdigest()

    def _session_for(self, row: int, prefix: str) -> CodeSession:
        if row not in self._sessions:
            self._sessions[row] = self.controller.new_session(prefix="", mode=self.mode)
        return self._sessions[row]

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        # scores: (batch, vocab) — torch.Tensor in HF, but NumPy is accepted for tests.
        is_torch = hasattr(scores, "detach") and hasattr(scores, "to")
        if is_torch:
            arr = scores.detach().to("cpu").numpy()
        else:
            arr = np.asarray(scores)
        if arr.ndim != 2:
            raise ValueError("scores must be 2-D (batch, vocab)")
        batch, _vocab = arr.shape
        out = arr.copy()
        # Reused across rows: the vocabulary classification is per-tokenizer, not per-row.
        vocab_map = self.vocabulary
        for b in range(batch):
            ids = list(input_ids[b]) if hasattr(input_ids[b], "__iter__") else [input_ids[b]]
            gen_ids = [int(t) for t in ids[self.prompt_length:]]
            text = "".join(vocab_map.get(t, "") for t in gen_ids)
            session = self._session_for(b, text)
            try:
                masked = self.controller.mask_logits(
                    out[b],
                    session=session,
                    prefix_text=text,
                    vocab_map=vocab_map,
                    mode=self.mode,
                    tokenizer_hash=self.tokenizer_hash,
                )
                out[b] = masked
            except DeadEnd:
                # Nothing legal can follow: leave the row alone so the model can still emit
                # its EOS, and make the dead end observable instead of silent.
                self.dead_ends += 1
        if is_torch:
            # Zero-copy when the mask can stay on the host's device (DLPack); otherwise the
            # helper performs exactly one device/dtype copy.
            return to_torch_like(out, scores)
        return out

    def verify(self, text: str, function_name: str | None = None, test_cases: list | None = None) -> Any:
        if function_name is None or not test_cases:
            return None
        return self.controller.verify_candidate(text, function_name, test_cases)
