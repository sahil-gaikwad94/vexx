"""Hugging Face code-generation logits processor for Vexx 2.1."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..code_engine.config import CodeEngineConfig
from ..code_engine.controller import VexxCodeController
from ..code_engine.masking import CodeSession


class VexxCodeLogitsProcessor:
    """HF-style logits processor applying Vexx code masks per batch row."""

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

    def _session_for(self, row: int, prefix: str) -> CodeSession:
        if row not in self._sessions:
            self._sessions[row] = self.controller.new_session(prefix="", mode=self.mode)
        sess = self._sessions[row]
        # Keep session prefix in sync is handled by caller; here we trust accumulated.
        return sess

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        # scores: (batch, vocab)
        arr = np.asarray(scores)
        if arr.ndim != 2:
            raise ValueError("scores must be 2-D (batch, vocab)")
        batch, vocab = arr.shape
        out = arr.copy()
        for b in range(batch):
            # Decode only the generated suffix: input_ids length beyond prompt_length.
            ids = list(input_ids[b]) if hasattr(input_ids[b], "__iter__") else [input_ids[b]]
            gen_ids = ids[self.prompt_length:]
            # Reconstruct text from vocab.
            text = "".join(self.vocabulary.get(int(t), "") for t in gen_ids)
            sess = self._session_for(b, text)
            try:
                masked = self.controller.mask_logits(
                    out[b], session=sess, prefix_text=text, vocab_map=self.vocabulary,
                    mode=self.mode,
                )
                out[b] = masked
            except Exception:
                # On dead-end, leave row unchanged but record; do not crash the batch.
                pass
        return out

    def verify(self, text: str, function_name: str | None = None, test_cases: list | None = None) -> Any:
        if function_name is None or not test_cases:
            return None
        return self.controller.verify_candidate(text, function_name, test_cases)