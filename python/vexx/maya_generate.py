"""Generation helpers for Vexx (ported from Maya-250M, model-agnostic)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

try:
    import numpy as np
except ImportError:
    np = None

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

PLAIN = "plain"
PYTHON = "python"
JSON = "json"
THINK = "think"
VALID_MODES = {PLAIN, PYTHON, JSON, THINK}


@dataclass
class GenerationResult:
    prompt: str
    continuation: str
    full_text: str
    finish_reason: str
    tokens_generated: int
    truncated: bool
    provenance: list[str] = field(default_factory=list)
    num_loops: int = 0

    @property
    def text(self) -> str:
        return self.full_text


def top_p_filter(logits, top_p: float):
    if top_p is None or top_p >= 1.0:
        return logits
    if torch is None or isinstance(logits, np.ndarray):
        return logits
    squeeze = logits.dim() == 1
    work = logits.unsqueeze(0) if squeeze else logits
    sorted_logits, sorted_indices = torch.sort(work, descending=True, dim=-1)
    cumulative = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    remove = cumulative > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    filtered = work.clone()
    filtered.scatter_(1, sorted_indices, torch.where(remove, float("-inf"), sorted_logits))
    return filtered.squeeze(0) if squeeze else filtered


def sample_next(logits, temperature: float | None = 0.0, top_p: float | None = None, generator=None):
    if torch is None or isinstance(logits, np.ndarray):
        arr = np.asarray(logits)
        return np.argmax(arr, axis=-1, keepdims=True)
    if temperature is None or temperature <= 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    scaled = logits / temperature
    if top_p is not None and top_p < 1.0:
        scaled = top_p_filter(scaled, top_p)
    probs = torch.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator)


def generate_maya(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 64,
    temperature: float | None = 0.0,
    top_p: float | None = None,
    vexx: Any = None,
    add_bos: bool = True,
    seed: int | None = None,
    mode: str = PLAIN,
    provenance: list[str] | None = None,
    context_mode: str = "rolling_window",
    num_loops: int | None = None,
) -> GenerationResult:
    if mode not in VALID_MODES:
        raise ValueError(f"unknown decoding mode {mode!r}; valid: {sorted(VALID_MODES)}")
    if context_mode not in {"rolling_window", "strict"}:
        raise ValueError(f"unknown context_mode {context_mode!r}")

    model.eval() if hasattr(model, "eval") else None
    origins: list[str] = ["neural_generation"]

    encode = getattr(tokenizer, "encode", None)
    ids = encode(prompt, add_bos=add_bos, add_eos=False) if encode else []
    if len(ids) == 0:
        ids = [tokenizer.token_to_id["<bos>"] if hasattr(tokenizer, "token_to_id") and "<bos>" in getattr(tokenizer, "token_to_id", {}) else 0]
    truncated = False
    if len(ids) > getattr(model, "max_ctx", len(ids) + 1):
        ids = ids[-model.max_ctx:]
        truncated = True
    if torch is not None:
        input_ids = torch.tensor([ids], dtype=torch.long)
    else:
        input_ids = np.asarray([ids], dtype=np.int64)

    prefill = model.prefill(input_ids, num_loops=num_loops, tokenizer_hash=getattr(tokenizer, "tokenizer_hash", lambda: None)() if hasattr(tokenizer, "tokenizer_hash") else None)
    logits, cache, truncation_info = prefill if isinstance(prefill, tuple) and len(prefill) == 3 else (prefill, None, {"truncated": False, "tokens_dropped": 0, "effective_len": len(ids), "num_loops": 0})
    loops_used = int((truncation_info or {}).get("num_loops", 0))
    seq_start = int(input_ids.shape[1]) - int((truncation_info or {}).get("tokens_dropped", 0))

    if context_mode == "strict" and (truncated or (truncation_info or {}).get("truncated")):
        note = "context_limit_reached"
        if provenance is not None:
            provenance.extend(origins + [note])
        return GenerationResult(prompt=prompt, continuation="", full_text=prompt + "\n[CONTEXT_LIMIT_REACHED] " + note, finish_reason="context_limit_reached", tokens_generated=0, truncated=True, provenance=origins if provenance is None else provenance, num_loops=loops_used)

    if truncated or (truncation_info or {}).get("truncated"):
        origins.append("prompt_truncated")

    generated: list[int] = []
    eos_id = getattr(tokenizer, "token_to_id", {}).get("<eos>")
    exp_end_id = getattr(tokenizer, "token_to_id", {}).get("</experiment>")

    generator = None
    if seed is not None and torch is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    cur_text = prompt
    finish_reason = "length"
    decode = getattr(model, "decode_step", None)
    id_to_token = getattr(tokenizer, "id_to_token", None)
    if id_to_token is None:
        id_to_token = {v: k for k, v in getattr(tokenizer, "token_to_id", {}).items()}

    for step_idx in range(max_new_tokens):
        ndim = getattr(logits, "ndim", getattr(logits, "dim", lambda: 1)())
        step_logits = logits[0] if ndim == 2 else logits[0, -1]
        if vexx is not None:
            masked = vexx.apply_in_flight_mask(
                step_logits.detach().cpu().numpy() if hasattr(step_logits, "detach") else np.asarray(step_logits),
                cur_text,
                getattr(tokenizer, "id_to_token", id_to_token),
                mode=mode,
                remaining_token_budget=max_new_tokens - step_idx,
            )
            if torch is not None and hasattr(step_logits, "device"):
                step_logits = torch.tensor(masked, dtype=step_logits.dtype, device=step_logits.device)
            else:
                step_logits = np.asarray(masked, dtype=np.float32)
            is_fin = bool(torch.isfinite(step_logits).any()) if (torch is not None and hasattr(step_logits, "device")) else bool(np.isfinite(step_logits).any())
            if not is_fin:
                raise RuntimeError("Vexx masked every token; refusing arbitrary argmax")

        _unsqueezed = step_logits.unsqueeze(0) if hasattr(step_logits, "unsqueeze") else np.expand_dims(step_logits, 0)
        next_id = int(sample_next(_unsqueezed, temperature, top_p, generator).item() if hasattr(sample_next(_unsqueezed, temperature, top_p, generator), "item") else sample_next(_unsqueezed, temperature, top_p, generator))
        if eos_id is not None and next_id == eos_id:
            finish_reason = "eos"
            break
        generated.append(next_id)
        decode_fn = getattr(tokenizer, "decode", None)
        cur_text = prompt + (decode_fn(generated, preserve_protocol=True) if decode_fn else "")

        if (
            vexx is not None
            and cur_text.endswith("=")
            and "<<" in cur_text
            and getattr(vexx, "crane_phase", lambda x: "")(cur_text) == "UNCONSTRAINED_THINK"
        ):
            m_cas = re.search(r"<<([^<>\n]+)=$", cur_text)
            if m_cas:
                value = vexx.crane_cas_evaluate(m_cas.group(1).strip()) + ">>"
                if not (isinstance(value, str) and value.startswith("ERROR")):
                    origins.append("deterministic_calculation")
                    if len(generated) >= max_new_tokens or (context_mode == "strict" and seq_start >= getattr(model, "max_ctx", seq_start + 1)):
                        break
                    if decode is not None:
                        t = torch.tensor([[next_id]], dtype=torch.long) if torch is not None else np.asarray([[next_id]], dtype=np.int64)
                        logits, cache = decode(t, cache, seq_start)
                        seq_start += 1
                    for cid in (tokenizer.encode(value, add_bos=False, add_eos=False) if encode else []):
                        if len(generated) >= max_new_tokens or (context_mode == "strict" and seq_start >= getattr(model, "max_ctx", seq_start + 1)):
                            break
                        if decode is not None:
                            t = torch.tensor([[cid]], dtype=torch.long) if torch is not None else np.asarray([[cid]], dtype=np.int64)
                            logits, cache = decode(t, cache, seq_start)
                            seq_start += 1
                        generated.append(cid)
                    cur_text = prompt + (decode_fn(generated, preserve_protocol=True) if decode_fn else "")
                    continue

        if vexx is not None and exp_end_id is not None and next_id == exp_end_id:
            last_open = cur_text.rfind("<experiment>")
            last_close = cur_text.rfind("</experiment>")
            if last_open != -1 and last_close > last_open:
                script_body = cur_text[last_open + len("<experiment>") : last_close].strip()
                obs = f" => [OBS: {vexx.execute_experiment_block(script_body)}] "
                origins.append("experiment_execution")
                if len(generated) >= max_new_tokens or (context_mode == "strict" and seq_start >= getattr(model, "max_ctx", seq_start + 1)):
                    break
                if decode is not None:
                    t = torch.tensor([[next_id]], dtype=torch.long) if torch is not None else np.asarray([[next_id]], dtype=np.int64)
                    logits, cache = decode(t, cache, seq_start)
                    seq_start += 1
                for cid in (tokenizer.encode(obs, add_bos=False, add_eos=False) if encode else []):
                    if len(generated) >= max_new_tokens or (context_mode == "strict" and seq_start >= getattr(model, "max_ctx", seq_start + 1)):
                        break
                    if decode is not None:
                        t = torch.tensor([[cid]], dtype=torch.long) if torch is not None else np.asarray([[cid]], dtype=np.int64)
                        logits, cache = decode(t, cache, seq_start)
                        seq_start += 1
                    generated.append(cid)
                cur_text = prompt + (decode_fn(generated, preserve_protocol=True) if decode_fn else "")
                continue

        if decode is not None:
            t = torch.tensor([[next_id]], dtype=torch.long) if torch is not None else np.asarray([[next_id]], dtype=np.int64)
            logits, cache = decode(t, cache, seq_start)
            seq_start += 1

    continuation = (tokenizer.decode(generated, preserve_protocol=True) if hasattr(tokenizer, "decode") else "")
    full_text = prompt + continuation
    if provenance is not None:
        provenance.extend(origins)
    return GenerationResult(
        prompt=prompt,
        continuation=continuation,
        full_text=full_text,
        finish_reason=finish_reason,
        tokens_generated=len(generated),
        truncated=truncated,
        provenance=origins if provenance is None else provenance,
    num_loops=loops_used,
)