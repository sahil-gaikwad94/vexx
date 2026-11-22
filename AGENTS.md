# Project notes

- Vexx 2.1: standalone release; package and Rust versions aligned at 2.1.0.
- Code engine lives in `python/vexx/code_engine/`; CAS in `python/vexx/cas.py`; calibration in `python/vexx/calibration.py`.
- Base `import vexx` works without optional extras; SymPy/NumPy are lazy-loaded.
- Run tests: `PYTHONPATH=python python -m pytest -q`.
- Rust/WASM claims must be revalidated in CI (no cargo in audit env).