# Vexx 2.1 Architecture

Vexx 2.1 combines the existing structured-output/verification engine (VIR,
JSON PDA, CDCL, CRANE, cartridges, bridges, Rust/PyO3/WASM) with a new
modular code-generation / research-verification subsystem.

## Module layout

- `python/vexx/code_engine/` — lexer, contracts, protocol, config, masking,
  controller, policy, sandbox, experiments, candidate, conflict.
- `python/vexx/cas.py` — safe symbolic CAS (optional `.[cas]`).
- `python/vexx/calibration.py` — statistical/dimensional/epistemic diagnostics.
- `python/vexx/cache_protocol.py` — backend-neutral cache rewind.
- `python/vexx/rl/code_rewards.py` — RLVR rewards and DPO export.
- `python/vexx/bridges/code_hf.py` — Hugging Face code logits processor.

## Layered execution

0. CRANE protocol phases (reasoning / experiment / strict emit).
1. Incremental lexer + callable contracts.
2. Safe symbolic CAS.
3. Rational conflict contraction + generic cache rewind.
4. Statistical / dimensional / epistemic diagnostics.
5. Isolated execution + candidate verification.
6. Selection / RLVR / DPO export.

No single tier proves correctness; evidence levels stay distinct.