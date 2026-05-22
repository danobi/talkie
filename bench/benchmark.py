"""Benchmark suite for Talkie 13B inference performance.

Measures prefill throughput, decode throughput, time-to-first-token,
end-to-end latency, peak GPU memory, and batch generation performance.

Usage:
    uv run python bench/benchmark.py --model talkie-1930-13b-base
    uv run python bench/benchmark.py --model talkie-1930-13b-base --scenarios short medium long batch

    # Save results as a baseline:
    uv run python bench/benchmark.py --save bench/baselines/main.json

    # Run and diff against a saved baseline:
    uv run python bench/benchmark.py --baseline bench/baselines/main.json

    # Diff two saved runs without re-running:
    uv run python bench/benchmark.py --diff bench/baselines/main.json bench/baselines/latest.json
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import torch

# Add project root to path so we can import talkie.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from talkie.config import MODELS
from talkie.generate import GenerationConfig, Talkie


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    scenario: str
    prompt_tokens: int
    generated_tokens: int
    prefill_ms: float
    prefill_tok_per_sec: float
    decode_ms: float
    decode_tok_per_sec: float
    ttft_ms: float
    total_ms: float
    peak_memory_gb: float
    trials: int
    # Per-trial raw data for computing std.
    trial_details: list[dict] = field(default_factory=list)


@dataclass
class BenchmarkSuite:
    model_name: str
    device: str
    dtype: str
    load_time_s: float = 0.0
    results: list[BenchmarkResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def _sync():
    """Synchronize CUDA if available, ensuring accurate timing."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _reset_peak_memory():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _peak_memory_gb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024**3)
    return 0.0


# ---------------------------------------------------------------------------
# Prompt generation
# ---------------------------------------------------------------------------

# A fixed prompt that tokenizes to a predictable length. We repeat/truncate
# to hit the target token count.
_SEED_TEXT = (
    "The history of civilization is a long and winding road, filled with "
    "unexpected turns and remarkable discoveries. From the earliest days of "
    "human settlement along the great rivers of Mesopotamia and Egypt, through "
    "the rise and fall of empires in Greece and Rome, to the transformative "
    "periods of the Renaissance and Enlightenment, humanity has continuously "
    "sought to understand and reshape the world. The industrial revolution "
    "brought about changes that would have been unimaginable to our ancestors, "
    "fundamentally altering the relationship between people and their labor. "
    "Today we stand at yet another crossroads, contemplating the implications "
    "of new technologies and scientific breakthroughs that promise to reshape "
    "society once more. The question before us is not whether change will come, "
    "but how we shall meet it, and what wisdom from the past we might carry "
    "forward into an uncertain future. "
)


def make_prompt(tokenizer, target_tokens: int) -> tuple[str, list[int]]:
    """Build a prompt string that tokenizes to approximately *target_tokens*."""
    # Encode seed, then repeat until we have enough tokens.
    seed_ids = tokenizer.encode(_SEED_TEXT)
    if len(seed_ids) >= target_tokens:
        token_ids = seed_ids[:target_tokens]
    else:
        reps = math.ceil(target_tokens / len(seed_ids))
        token_ids = (seed_ids * reps)[:target_tokens]
    prompt = tokenizer.decode(token_ids)
    # Re-encode to get exact count (decode/encode can drift slightly).
    final_ids = tokenizer.encode(prompt)
    return prompt, final_ids


# ---------------------------------------------------------------------------
# Individual benchmarks
# ---------------------------------------------------------------------------


def bench_prefill(
    model: Talkie,
    token_ids: list[int],
    warmup: int = 2,
    trials: int = 5,
) -> list[dict]:
    """Benchmark prefill (forward pass on prompt) in isolation."""
    device = model.device
    x = torch.tensor([token_ids], dtype=torch.long, device=device)

    # Warmup.
    with torch.no_grad(), model._autocast:
        for _ in range(warmup):
            _sync()
            model.model.forward(x)
            _sync()

    results = []
    with torch.no_grad(), model._autocast:
        for _ in range(trials):
            _reset_peak_memory()
            _sync()
            t0 = time.perf_counter()
            model.model.forward(x)
            _sync()
            t1 = time.perf_counter()
            results.append(
                {
                    "elapsed_ms": (t1 - t0) * 1000,
                    "peak_memory_gb": _peak_memory_gb(),
                }
            )
    return results


def bench_decode(
    model: Talkie,
    token_ids: list[int],
    max_tokens: int = 128,
    warmup: int = 1,
    trials: int = 3,
) -> list[dict]:
    """Benchmark full autoregressive decode (prefill + generation).

    Returns per-trial timing that separates prefill from decode.
    """
    device = model.device

    # Warmup with a short generation.
    for _ in range(warmup):
        x = torch.tensor([token_ids], dtype=torch.long, device=device)
        with torch.no_grad(), model._autocast:
            _sync()
            # Prefill.
            model.model.forward(x)
            _sync()
            # Generate a few tokens.
            for _ in range(min(8, max_tokens)):
                next_tok = model.model.sample_batch(x, t=1.0)
                x = torch.cat([x, next_tok.unsqueeze(1)], dim=1)
            _sync()

    results = []
    for _ in range(trials):
        x = torch.tensor([token_ids], dtype=torch.long, device=device)
        _reset_peak_memory()

        with torch.no_grad(), model._autocast:
            # -- Prefill --
            _sync()
            t_start = time.perf_counter()
            logits = model.model.forward(x)
            _sync()
            t_prefill = time.perf_counter()

            # Sample first token from prefill logits.
            first_logits = logits / 1.0  # temperature=1 for benchmarking
            first_logits = first_logits + torch.zeros_like(
                first_logits
            )  # no gumbel for determinism
            first_tok = torch.argmax(first_logits, dim=-1)
            x = torch.cat([x, first_tok.unsqueeze(1)], dim=1)
            _sync()
            t_first_token = time.perf_counter()

            # -- Decode remaining tokens --
            generated = 1
            for _ in range(max_tokens - 1):
                next_tok = model.model.sample_batch(x, t=1.0)
                x = torch.cat([x, next_tok.unsqueeze(1)], dim=1)
                generated += 1
                tok_id = int(next_tok[0])
                if tok_id in model._stop_ids:
                    break
            _sync()
            t_end = time.perf_counter()

        results.append(
            {
                "prefill_ms": (t_prefill - t_start) * 1000,
                "ttft_ms": (t_first_token - t_start) * 1000,
                "decode_ms": (t_end - t_first_token) * 1000,
                "total_ms": (t_end - t_start) * 1000,
                "generated_tokens": generated,
                "peak_memory_gb": _peak_memory_gb(),
            }
        )
    return results


def bench_batch(
    model: Talkie,
    token_ids: list[int],
    batch_size: int = 4,
    max_tokens: int = 64,
    warmup: int = 1,
    trials: int = 3,
) -> list[dict]:
    """Benchmark batch generation."""
    prompt, _ = make_prompt(model.tokenizer, len(token_ids))
    configs = [
        GenerationConfig(temperature=1.0, max_tokens=max_tokens)
        for _ in range(batch_size)
    ]

    # Warmup.
    for _ in range(warmup):
        model.batch_generate(prompt, configs)

    results = []
    for _ in range(trials):
        _reset_peak_memory()
        _sync()
        t0 = time.perf_counter()
        gen_results = model.batch_generate(prompt, configs)
        _sync()
        t1 = time.perf_counter()

        total_tokens = sum(r.token_count for r in gen_results)
        results.append(
            {
                "total_ms": (t1 - t0) * 1000,
                "total_tokens": total_tokens,
                "peak_memory_gb": _peak_memory_gb(),
            }
        )
    return results


# ---------------------------------------------------------------------------
# Scenario runner
# ---------------------------------------------------------------------------

SCENARIOS = {
    "short": {
        "prompt_tokens": 32,
        "max_tokens": 128,
        "description": "Short prompt (32 tok) → 128 tok generation",
    },
    "medium": {
        "prompt_tokens": 256,
        "max_tokens": 128,
        "description": "Medium prompt (256 tok) → 128 tok generation",
    },
    "long": {
        "prompt_tokens": 1024,
        "max_tokens": 64,
        "description": "Long prompt (1024 tok) → 64 tok generation",
    },
    "batch": {
        "prompt_tokens": 128,
        "max_tokens": 64,
        "batch_size": 4,
        "description": "Batch (4x) from 128 tok prompt → 64 tok each",
    },
}


def run_scenario(
    model: Talkie,
    name: str,
    scenario: dict,
    warmup: int,
    trials: int,
) -> BenchmarkResult:
    prompt_tokens = scenario["prompt_tokens"]
    max_tokens = scenario["max_tokens"]
    prompt, token_ids = make_prompt(model.tokenizer, prompt_tokens)
    actual_prompt_tokens = len(token_ids)

    print(f"\n{'=' * 60}")
    print(f"  Scenario: {name} — {scenario['description']}")
    print(f"  Prompt tokens: {actual_prompt_tokens}, Max gen tokens: {max_tokens}")
    print(f"{'=' * 60}")

    if "batch_size" in scenario:
        # Batch benchmark.
        batch_size = scenario["batch_size"]
        print(f"  Batch size: {batch_size}")
        trial_data = bench_batch(
            model,
            token_ids,
            batch_size=batch_size,
            max_tokens=max_tokens,
            warmup=warmup,
            trials=trials,
        )
        avg_total_ms = sum(t["total_ms"] for t in trial_data) / len(trial_data)
        avg_total_tokens = sum(t["total_tokens"] for t in trial_data) / len(trial_data)
        avg_peak_mem = sum(t["peak_memory_gb"] for t in trial_data) / len(trial_data)
        avg_tok_per_sec = (
            avg_total_tokens / (avg_total_ms / 1000) if avg_total_ms > 0 else 0
        )

        result = BenchmarkResult(
            scenario=name,
            prompt_tokens=actual_prompt_tokens,
            generated_tokens=int(avg_total_tokens),
            prefill_ms=0,
            prefill_tok_per_sec=0,
            decode_ms=avg_total_ms,
            decode_tok_per_sec=avg_tok_per_sec,
            ttft_ms=0,
            total_ms=avg_total_ms,
            peak_memory_gb=avg_peak_mem,
            trials=trials,
            trial_details=trial_data,
        )
    else:
        # Prefill-only benchmark.
        print("  Running prefill benchmark...")
        prefill_data = bench_prefill(
            model,
            token_ids,
            warmup=warmup,
            trials=trials,
        )
        avg_prefill_ms = sum(t["elapsed_ms"] for t in prefill_data) / len(prefill_data)
        prefill_tok_per_sec = (
            actual_prompt_tokens / (avg_prefill_ms / 1000) if avg_prefill_ms > 0 else 0
        )

        # Decode benchmark.
        print("  Running decode benchmark...")
        decode_data = bench_decode(
            model,
            token_ids,
            max_tokens=max_tokens,
            warmup=warmup,
            trials=trials,
        )
        avg_decode_ms = sum(t["decode_ms"] for t in decode_data) / len(decode_data)
        avg_ttft_ms = sum(t["ttft_ms"] for t in decode_data) / len(decode_data)
        avg_total_ms = sum(t["total_ms"] for t in decode_data) / len(decode_data)
        avg_gen_tokens = sum(t["generated_tokens"] for t in decode_data) / len(
            decode_data
        )
        avg_peak_mem = max(
            max(t["peak_memory_gb"] for t in prefill_data),
            max(t["peak_memory_gb"] for t in decode_data),
        )
        decode_tok_per_sec = (
            avg_gen_tokens / (avg_decode_ms / 1000) if avg_decode_ms > 0 else 0
        )

        result = BenchmarkResult(
            scenario=name,
            prompt_tokens=actual_prompt_tokens,
            generated_tokens=int(avg_gen_tokens),
            prefill_ms=avg_prefill_ms,
            prefill_tok_per_sec=prefill_tok_per_sec,
            decode_ms=avg_decode_ms,
            decode_tok_per_sec=decode_tok_per_sec,
            ttft_ms=avg_ttft_ms,
            total_ms=avg_total_ms,
            peak_memory_gb=avg_peak_mem,
            trials=trials,
            trial_details=prefill_data + decode_data,
        )

    _print_result(result)
    return result


def _print_result(r: BenchmarkResult):
    print(f"\n  Results ({r.trials} trials):")
    if r.prefill_ms > 0:
        print(
            f"    Prefill:       {r.prefill_ms:8.1f} ms  ({r.prefill_tok_per_sec:,.0f} tok/s)"
        )
    if r.ttft_ms > 0:
        print(f"    TTFT:          {r.ttft_ms:8.1f} ms")
    if r.decode_ms > 0:
        print(
            f"    Decode:        {r.decode_ms:8.1f} ms  ({r.decode_tok_per_sec:,.1f} tok/s)"
        )
    print(f"    Total:         {r.total_ms:8.1f} ms")
    print(f"    Generated:     {r.generated_tokens} tokens")
    print(f"    Peak memory:   {r.peak_memory_gb:.2f} GB")


# ---------------------------------------------------------------------------
# Serialization & diffing
# ---------------------------------------------------------------------------

SCHEMA_VERSION = 1


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def save_suite(suite: BenchmarkSuite, path: str | Path) -> Path:
    """Serialize a benchmark suite to JSON on disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "suite": asdict(suite),
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def load_suite(path: str | Path) -> tuple[BenchmarkSuite, dict]:
    """Load a benchmark suite from JSON.

    Returns the suite and a metadata dict (timestamp, git_commit, schema_version).
    """
    path = Path(path)
    payload = json.loads(path.read_text())
    schema = payload.get("schema_version")
    if schema != SCHEMA_VERSION:
        print(f"Warning: {path} has schema_version={schema}, expected {SCHEMA_VERSION}")
    suite_data = dict(payload["suite"])
    results = [BenchmarkResult(**r) for r in suite_data.pop("results", [])]
    suite = BenchmarkSuite(results=results, **suite_data)
    metadata = {k: v for k, v in payload.items() if k != "suite"}
    return suite, metadata


# (label, attr, unit, higher_is_better) — metrics shown in the diff table.
_DIFF_METRICS: list[tuple[str, str, str, bool]] = [
    ("Prefill tok/s", "prefill_tok_per_sec", "tok/s", True),
    ("Decode tok/s", "decode_tok_per_sec", "tok/s", True),
    ("TTFT", "ttft_ms", "ms", False),
    ("Total", "total_ms", "ms", False),
    ("Peak memory", "peak_memory_gb", "GB", False),
]


def _diff_row(
    label: str,
    baseline_val: float,
    current_val: float,
    unit: str,
    higher_is_better: bool,
    significance_pct: float = 1.0,
) -> str:
    delta = current_val - baseline_val
    if baseline_val == 0:
        pct_str = "    n/a"
        verdict = ""
    else:
        pct = delta / baseline_val * 100
        pct_str = f"{pct:+6.1f}%"
        if abs(pct) < significance_pct:
            verdict = ""
        elif (delta > 0) == higher_is_better:
            verdict = "  (better)"
        else:
            verdict = "  (worse)"
    return (
        f"    {label:<16} {baseline_val:>10.2f} -> {current_val:>10.2f} "
        f"{unit:<5}  {pct_str}{verdict}"
    )


def diff_suites(
    baseline: BenchmarkSuite,
    current: BenchmarkSuite,
    baseline_meta: dict | None = None,
    current_meta: dict | None = None,
) -> None:
    """Print a side-by-side comparison of two benchmark suites."""
    print(f"\n{'=' * 78}")
    print("  DIFF: baseline -> current")
    print(f"{'=' * 78}")
    print(f"  Model:   {baseline.model_name} -> {current.model_name}")
    print(f"  Device:  {baseline.device} -> {current.device}")
    print(f"  Dtype:   {baseline.dtype} -> {current.dtype}")
    for label, meta in (("Baseline", baseline_meta), ("Current ", current_meta)):
        if not meta:
            continue
        commit = (meta.get("git_commit") or "?")[:12]
        ts = meta.get("timestamp", "?")
        print(f"  {label}: {ts}  @ {commit}")

    print("\n  Load time:")
    print(_diff_row("seconds", baseline.load_time_s, current.load_time_s, "s", False))

    baseline_by_name = {r.scenario: r for r in baseline.results}
    current_by_name = {r.scenario: r for r in current.results}
    ordered: list[str] = list(baseline_by_name.keys())
    for s in current_by_name:
        if s not in baseline_by_name:
            ordered.append(s)

    for name in ordered:
        b = baseline_by_name.get(name)
        c = current_by_name.get(name)
        print(f"\n  [{name}]")
        if b is None:
            print("    (only in current)")
            continue
        if c is None:
            print("    (only in baseline)")
            continue
        for label, attr, unit, higher_is_better in _DIFF_METRICS:
            bv = getattr(b, attr)
            cv = getattr(c, attr)
            if bv == 0 and cv == 0:
                continue
            print(_diff_row(label, bv, cv, unit, higher_is_better))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def profile_load(
    model_name: str,
    trace_path: str = "load_trace.json",
    device: str | None = None,
    cache_dir: str | None = None,
) -> None:
    """Profile model loading and export a Chrome trace for Perfetto."""
    from torch.profiler import ProfilerActivity, profile, record_function

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    print(f"Profiling model load: {model_name}...")
    with profile(
        activities=activities,
        record_shapes=True,
        with_stack=True,
        profile_memory=True,
    ) as prof:
        with record_function("model_load"):
            Talkie(model_name, device=device, cache_dir=cache_dir)
            _sync()

    prof.export_chrome_trace(trace_path)
    print(f"Trace saved to {trace_path}")
    print("Open in Perfetto: https://ui.perfetto.dev/")

    # Also print a summary table to the terminal.
    print(f"\n{prof.key_averages().table(sort_by='self_cpu_time_total', row_limit=30)}")


def run_benchmarks(
    model_name: str,
    scenarios: list[str] | None = None,
    warmup: int = 2,
    trials: int = 5,
    device: str | None = None,
    cache_dir: str | None = None,
) -> BenchmarkSuite:
    """Run the full benchmark suite and return results."""
    if scenarios is None:
        scenarios = list(SCENARIOS.keys())

    print(f"Loading model: {model_name}...")
    _reset_peak_memory()
    t0 = time.perf_counter()
    model = Talkie(model_name, device=device, cache_dir=cache_dir)
    _sync()
    t1 = time.perf_counter()
    load_time_s = t1 - t0
    load_memory_gb = _peak_memory_gb()
    print(
        f"Model loaded in {load_time_s:.1f}s on {model.device} ({load_memory_gb:.2f} GB)"
    )

    suite = BenchmarkSuite(
        model_name=model_name,
        device=str(model.device),
        dtype="bfloat16",
        load_time_s=load_time_s,
    )

    for name in scenarios:
        if name not in SCENARIOS:
            print(f"Warning: unknown scenario {name!r}, skipping")
            continue
        result = run_scenario(
            model, name, SCENARIOS[name], warmup=warmup, trials=trials
        )
        suite.results.append(result)

    # Print summary table.
    print(f"\n{'=' * 60}")
    print(f"  SUMMARY — {model_name}")
    print(f"{'=' * 60}")
    print(f"  Model load:  {suite.load_time_s:.1f}s")
    print()
    print(
        f"  {'Scenario':<10} {'Prefill tok/s':>14} {'Decode tok/s':>14} {'TTFT ms':>10} {'Total ms':>10} {'Mem GB':>8}"
    )
    print(f"  {'-' * 66}")
    for r in suite.results:
        pfill = f"{r.prefill_tok_per_sec:,.0f}" if r.prefill_tok_per_sec > 0 else "—"
        dec = f"{r.decode_tok_per_sec:,.1f}" if r.decode_tok_per_sec > 0 else "—"
        ttft = f"{r.ttft_ms:.1f}" if r.ttft_ms > 0 else "—"
        print(
            f"  {r.scenario:<10} {pfill:>14} {dec:>14} {ttft:>10} {r.total_ms:>10.1f} {r.peak_memory_gb:>8.2f}"
        )

    return suite


def main():
    parser = argparse.ArgumentParser(description="Benchmark Talkie 13B inference")
    parser.add_argument(
        "--model",
        default="talkie-1930-13b-base",
        choices=list(MODELS.keys()),
        help="Model to benchmark (default: talkie-1930-13b-base)",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=None,
        choices=list(SCENARIOS.keys()),
        help="Scenarios to run (default: all)",
    )
    parser.add_argument("--warmup", type=int, default=2, help="Warmup iterations")
    parser.add_argument(
        "--trials", type=int, default=5, help="Timing trials per benchmark"
    )
    parser.add_argument("--device", default=None, help="PyTorch device")
    parser.add_argument("--cache-dir", default=None, help="HuggingFace cache directory")
    parser.add_argument(
        "--profile-load",
        nargs="?",
        const="load_trace.json",
        default=None,
        metavar="PATH",
        help="Profile model loading and save Chrome trace for Perfetto (default: load_trace.json)",
    )
    parser.add_argument(
        "--save",
        default=None,
        metavar="PATH",
        help="Save benchmark results to JSON at PATH (capture a baseline)",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        metavar="PATH",
        help="After running, diff results against a previously-saved baseline JSON",
    )
    parser.add_argument(
        "--diff",
        nargs=2,
        default=None,
        metavar=("BASELINE", "CURRENT"),
        help="Diff two saved benchmark JSON files without running the suite",
    )
    args = parser.parse_args()

    if args.diff:
        baseline_suite, baseline_meta = load_suite(args.diff[0])
        current_suite, current_meta = load_suite(args.diff[1])
        diff_suites(baseline_suite, current_suite, baseline_meta, current_meta)
        return

    if args.profile_load:
        profile_load(
            model_name=args.model,
            trace_path=args.profile_load,
            device=args.device,
            cache_dir=args.cache_dir,
        )
        return

    suite = run_benchmarks(
        model_name=args.model,
        scenarios=args.scenarios,
        warmup=args.warmup,
        trials=args.trials,
        device=args.device,
        cache_dir=args.cache_dir,
    )

    if args.save:
        save_path = save_suite(suite, args.save)
        print(f"\nSaved benchmark results to {save_path}")

    if args.baseline:
        baseline_suite, baseline_meta = load_suite(args.baseline)
        current_meta = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_commit(),
        }
        diff_suites(baseline_suite, suite, baseline_meta, current_meta)


if __name__ == "__main__":
    main()
