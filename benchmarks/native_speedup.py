"""Native-vs-pure Python masking benchmark for Vexx.

This changes between the two runs is ``VEXX_NO_NATIVE``. That makes the comparison honest:
the pure-Python engine is always measured, and the native extension is only used when the
kill switch and the availability probe both allow it.

    python benchmarks/native_speedup.py            # comparison table
    python benchmarks/native_speedup.py --json     # machine-readable
    VEXX_NO_NATIVE=1 python benchmarks/native_speedup.py --worker   # one side only
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _vocabulary() -> list[str]:
    vocab = [
        "{\"a\": ",
        "}",
        "\"a\"",
        ": ",
        "1",
        "null",
        "]",
        "x",
        "",
    ]
    for i in range(600):
        vocab.append(f'"k{i}": {i}')
    return vocab


def _prefixes(vocabulary: list[str]) -> list[str]:
    prefixes = [""]
    for token in vocabulary[:40]:
        prefixes.append(token)
    return prefixes


def _time_block(block, count: int) -> dict:
    samples = []
    for _ in range(max(1, count)):
        start = time.perf_counter()
        for _ in range(20):
            block()
        samples.append((time.perf_counter() - start) * 1e6)
    samples.sort()
    return {
        "p50_us": round(samples[len(samples) // 2], 3),
        "p95_us": round(samples[min(len(samples) - 1, int(0.95 * len(samples)))], 3),
        "calls_per_s": int(1e6 / samples[len(samples) // 2]) if samples[len(samples) // 2] else 0,
    }


def measure(iterations: int, repeats: int) -> dict:
    import vexx
    from vexx import native
    from vexx.json_pda import JSONPushdownAutomaton, TokenMasker
    from vexx.interval import Interval

    vocabulary = _vocabulary()
    eos = vocabulary.index("")
    automaton = JSONPushdownAutomaton()
    mask_block = lambda: TokenMasker(vocabulary, eos_token=eos).valid_mask('{"a": 1')
    numeric_block = lambda: vexx.Interval(0.0, 1.0).from_numeric_prefix("0.3", Interval(0.0, 1.0))

    return {
        "mode": "native" if native.available() else "pure",
        "native_available": native.available(),
        "reason": native.load_error(),
        "python": platform.python_version(),
        "vocabulary": len(vocabulary),
        "mask": _time_block(mask_block, iterations),
        "numeric_prefix": _time_block(numeric_block, iterations),
    }


def repeat_warmup(iterations: int) -> int:
    return max(1, iterations // 10)


def _worker(args) -> int:
    print(json.dumps(measure(args.iterations, args.repeats)))
    return 0


def _child(iterations: int, repeats: int, disabled: bool) -> dict:
    env = dict(os.environ)
    if disabled:
        env["VEXX_NO_NATIVE"] = "1"
    else:
        env.pop("VEXX_NO_NATIVE", None)
    env["PYTHONPATH"] = os.pathsep.join(
        [os.path.join(REPO_ROOT, "python"), env.get("PYTHONPATH", "")]
    ).strip(os.pathsep)
    command = [
        sys.executable,
        os.path.abspath(__file__),
        "--worker",
        "--iterations",
        str(iterations),
        "--repeats",
        str(repeats),
    ]
    result = subprocess.run(command, capture_output=True, text=True, env=env, check=False)
    if result.returncode != 0:
        raise SystemExit(f"worker failed ({result.returncode}):\n{result.stderr.strip()}")
    return json.loads(result.stdout.strip().splitlines()[-1])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.worker:
        return _worker(args)
    if args.iterations < 1 or args.repeats < 1:
        parser.error("iterations and repeats must be positive")

    native_side = _child(args.iterations, args.repeats, disabled=False)
    pure_side = _child(args.iterations, args.repeats, disabled=True)

    native_available = bool(native_side.get("native_available"))
    comparison_status = "native_available" if native_available else "native_unavailable"
    report = {
        "python": native_side["python"],
        "vocabulary": native_side["vocabulary"],
        "native_available": native_available,
        "reason": native_side["reason"],
        "comparison_status": comparison_status,
        "workloads": {},
    }
    for workload in ("mask", "numeric_prefix"):
        pure = pure_side[workload]
        native = native_side[workload]
        speedup = pure["p50_us"] / native["p50_us"] if (native_available and native["p50_us"]) else None
        report["workloads"][workload] = {
            "pure": pure,
            "native": native,
            "speedup": round(speedup, 2) if speedup else None,
        }

    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    if not native_available:
        print("native extension not available:", native_side["reason"])
        print("build it with: maturin build --release -m crates/vexx-py/Cargo.toml -o dist")
        print("               pip install --no-deps dist/*.whl")
        print("the pure-Python engine is unaffected; nothing to compare.")
        return 0

    print(f"python {report['python']}, vocabulary {report['vocabulary']} tokens")
    print(f"{'workload':<18}{'pure µs/call':>14}{'native µs/call':>16}{'speedup':>10}")
    for workload, numbers in report["workloads"].items():
        print(
            f"{workload:<18}{numbers['pure']['p50_us']:>14.3f}"
            f"{numbers['native']['p50_us']:>16.3f}{str(numbers['speedup']) + 'x':>10}"
        )
    print("\np50 over repeated blocks of the same call sites; VEXX_NO_NATIVE pins the pure path.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
