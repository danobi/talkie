"""Output-equivalence regression test for Talkie inference.

Greedy-decodes a fixed set of prompts and either records a baseline or
compares against one. Intended as a fast correctness check during
performance optimization work: any change that alters the model's logits
enough to flip an argmax will show up as a mismatch.

Determinism: we use ``temperature=1.0, top_k=1``. ``top_k=1`` sets every
non-argmax logit to ``-inf`` before the Gumbel-noise addition, so the noise
cannot change the winner. No random seed is needed.

CUDA kernels are usually deterministic for fixed shapes on the same
hardware, but small numerical drift between different GPUs, driver
versions, or kernel choices is possible. The diff report shows the
first-mismatch position so partial matches are still informative.

Usage::

    # Capture a baseline before starting optimization work
    uv run python evals/equivalence.py --save evals/baselines/base.json

    # After making changes, regression-check against the baseline
    uv run python evals/equivalence.py --baseline evals/baselines/base.json

    # Compare two saved runs without re-running the model
    uv run python evals/equivalence.py --diff a.json b.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import torch

# Add project root to path so we can import talkie when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from talkie.chat import format_prompt
from talkie.config import MODELS
from talkie.generate import Talkie
from talkie.sampling import scalar_top_k_tensor


SCHEMA_VERSION = 1
DEFAULT_CACHE_DIR = Path("/tmp/talkie-cache")


# ---------------------------------------------------------------------------
# Prompt set
# ---------------------------------------------------------------------------
#
# Five prompts of mixed length and shape. The contents are chosen to be
# in-distribution for a pre-1931 language model (no modern slang or named
# entities), so that even the base model produces something the
# instruction-tuned model would also handle sensibly. The full suite
# generates ~400 tokens total — a couple of minutes on a single GPU.


@dataclass(frozen=True)
class PromptSpec:
    name: str
    prompt: str
    max_tokens: int


PROMPTS: list[PromptSpec] = [
    PromptSpec(
        name="short",
        prompt="The autumn wind blew through the streets of London, and",
        max_tokens=64,
    ),
    PromptSpec(
        name="arithmetic",
        prompt=(
            "John has seven apples and gives three to his sister, then buys "
            "five more at the market. The number of apples John now has is"
        ),
        max_tokens=24,
    ),
    PromptSpec(
        name="medium",
        prompt=(
            "Of all the events that shaped the nineteenth century, the "
            "expansion of the railway perhaps did more than any other to "
            "alter the rhythm of daily life. Distances that once required "
            "many days of travel by stagecoach were"
        ),
        max_tokens=96,
    ),
    PromptSpec(
        name="long",
        prompt=(
            "The art of writing a good letter, though often neglected in "
            "this hurried age, remains one of the most pleasing of social "
            "accomplishments. A well-composed letter is at once a "
            "conversation and a keepsake, an expression of friendship that "
            "outlives the moment of its writing. The first rule, as every "
            "thoughtful correspondent will tell you, is to write only when "
            "one has something to say; the second is to say it plainly. "
            "Begin your letter with"
        ),
        max_tokens=96,
    ),
    PromptSpec(
        name="qa",
        prompt="What were the principal causes of the French Revolution?",
        max_tokens=128,
    ),
]


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class PromptResult:
    name: str
    prompt: str
    max_tokens: int
    token_ids: list[int]
    text: str
    token_hash: str
    elapsed_ms: float


@dataclass
class EvalSuite:
    model_name: str
    device: str
    dtype: str
    results: list[PromptResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------


def greedy_decode(model: Talkie, spec: PromptSpec) -> tuple[list[int], float]:
    """Greedy-decode up to ``spec.max_tokens`` tokens from ``spec.prompt``.

    Returns ``(token_ids, elapsed_ms)``. Stops early on a stop token.
    """
    if model.spec.style == "it":
        formatted = format_prompt(spec.prompt)
    else:
        formatted = spec.prompt

    prompt_ids = model.tokenizer.encode(formatted, allowed_special="all")
    x = torch.tensor([prompt_ids], dtype=torch.long, device=model.device)
    top_k_t = scalar_top_k_tensor(1, model.device)

    generated: list[int] = []
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad(), model._autocast:
        for _ in range(spec.max_tokens):
            next_tok = model.model.sample_batch(x, t=1.0, top_k=top_k_t)
            tok_id = int(next_tok[0])
            if tok_id in model._stop_ids:
                break
            generated.append(tok_id)
            x = torch.cat([x, next_tok.unsqueeze(1)], dim=1)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return generated, elapsed_ms


def hash_tokens(token_ids: list[int]) -> str:
    h = hashlib.sha256()
    for tid in token_ids:
        h.update(tid.to_bytes(4, "little", signed=False))
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Suite runner
# ---------------------------------------------------------------------------


def run_suite(
    model_name: str,
    device: str | None = None,
    cache_dir: str | None = None,
    quantize: str | None = None,
) -> EvalSuite:
    print(f"Loading model: {model_name}...")
    if quantize:
        print(f"  quantize={quantize}")
    t0 = time.perf_counter()
    model = Talkie(model_name, device=device, cache_dir=cache_dir, quantize=quantize)
    print(f"Loaded in {time.perf_counter() - t0:.1f}s on {model.device}")

    dtype_label = "bfloat16" if not quantize else f"bfloat16+{quantize}"
    suite = EvalSuite(
        model_name=model_name,
        device=str(model.device),
        dtype=dtype_label,
    )
    for spec in PROMPTS:
        print(f"  [{spec.name}] decoding...", end="", flush=True)
        token_ids, elapsed_ms = greedy_decode(model, spec)
        text = model.tokenizer.decode(token_ids)
        suite.results.append(
            PromptResult(
                name=spec.name,
                prompt=spec.prompt,
                max_tokens=spec.max_tokens,
                token_ids=token_ids,
                text=text,
                token_hash=hash_tokens(token_ids),
                elapsed_ms=elapsed_ms,
            )
        )
        print(f" {len(token_ids)} tok ({elapsed_ms / 1000:.2f}s)")
    return suite


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def save_suite(suite: EvalSuite, path: str | Path) -> Path:
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


def load_suite(path: str | Path) -> tuple[EvalSuite, dict]:
    payload = json.loads(Path(path).read_text())
    schema = payload.get("schema_version")
    if schema != SCHEMA_VERSION:
        print(
            f"Warning: {path} has schema_version={schema}, "
            f"expected {SCHEMA_VERSION}"
        )
    suite_data = dict(payload["suite"])
    results = [PromptResult(**r) for r in suite_data.pop("results", [])]
    suite = EvalSuite(results=results, **suite_data)
    metadata = {k: v for k, v in payload.items() if k != "suite"}
    return suite, metadata


# ---------------------------------------------------------------------------
# Diff
# ---------------------------------------------------------------------------


def _first_mismatch(a: list[int], b: list[int]) -> int | None:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    if len(a) != len(b):
        return n
    return None


def diff_suites(
    baseline: EvalSuite,
    current: EvalSuite,
    baseline_meta: dict | None = None,
    current_meta: dict | None = None,
) -> int:
    """Print a per-prompt diff. Returns the number of mismatched prompts."""
    print(f"\n{'=' * 70}")
    print("  EQUIVALENCE CHECK: baseline -> current")
    print(f"{'=' * 70}")
    print(f"  Model:   {baseline.model_name} -> {current.model_name}")
    print(f"  Device:  {baseline.device} -> {current.device}")
    print(f"  Dtype:   {baseline.dtype} -> {current.dtype}")
    for label, meta in (("Baseline", baseline_meta), ("Current ", current_meta)):
        if not meta:
            continue
        commit = (meta.get("git_commit") or "?")[:12]
        ts = meta.get("timestamp", "?")
        print(f"  {label}: {ts}  @ {commit}")
    print()

    baseline_by_name = {r.name: r for r in baseline.results}
    current_by_name = {r.name: r for r in current.results}
    names = list(baseline_by_name.keys())
    for name in current_by_name:
        if name not in baseline_by_name:
            names.append(name)

    n_fail = 0
    n_pass = 0
    for name in names:
        b = baseline_by_name.get(name)
        c = current_by_name.get(name)
        if b is None:
            print(f"  [{name:<14}] only in current — skipped")
            continue
        if c is None:
            print(f"  [{name:<14}] only in baseline — FAIL")
            n_fail += 1
            continue

        idx = _first_mismatch(b.token_ids, c.token_ids)
        if idx is None:
            n_pass += 1
            print(
                f"  [{name:<14}] PASS  ({len(c.token_ids)} tokens, "
                f"{c.elapsed_ms / 1000:.2f}s)"
            )
        else:
            n_fail += 1
            print(
                f"  [{name:<14}] FAIL  first mismatch at token {idx} "
                f"(baseline len={len(b.token_ids)}, current len={len(c.token_ids)})"
            )
            # Decode a short window around the divergence for human inspection.
            window = 12
            b_window = b.token_ids[max(0, idx - 2) : idx + window]
            c_window = c.token_ids[max(0, idx - 2) : idx + window]
            # We can't decode from c.token_ids without a tokenizer, but the
            # full text is in the result. Show the tail of the text instead.
            preview_len = 100
            preview_start = max(0, len(b.text) - preview_len)
            print(f"      baseline tail: {b.text[preview_start:]!r}")
            preview_start = max(0, len(c.text) - preview_len)
            print(f"      current  tail: {c.text[preview_start:]!r}")
            print(f"      baseline ids[{idx}:{idx + window}]: {b_window[2:]}")
            print(f"      current  ids[{idx}:{idx + window}]: {c_window[2:]}")

    total = n_pass + n_fail
    print(f"\n  {n_pass} / {total} prompts match exactly.")
    return n_fail


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Output-equivalence regression test for Talkie."
    )
    parser.add_argument(
        "--model",
        default="talkie-1930-13b-base",
        choices=list(MODELS.keys()),
        help="Model to evaluate (default: talkie-1930-13b-base)",
    )
    parser.add_argument("--device", default=None, help="PyTorch device")
    parser.add_argument(
        "--cache-dir",
        default=str(DEFAULT_CACHE_DIR),
        help=f"HuggingFace cache directory (default: {DEFAULT_CACHE_DIR})",
    )
    parser.add_argument(
        "--save",
        default=None,
        metavar="PATH",
        help="Save current outputs as a baseline at PATH.",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        metavar="PATH",
        help="After running, diff against the baseline JSON at PATH.",
    )
    parser.add_argument(
        "--diff",
        nargs=2,
        default=None,
        metavar=("BASELINE", "CURRENT"),
        help="Diff two saved JSON files without re-running the model.",
    )
    parser.add_argument(
        "--quantize",
        choices=["int4"],
        default=None,
        help="Apply weight-only quantization after loading the checkpoint.",
    )
    args = parser.parse_args()

    if args.diff:
        b, b_meta = load_suite(args.diff[0])
        c, c_meta = load_suite(args.diff[1])
        n_fail = diff_suites(b, c, b_meta, c_meta)
        return 1 if n_fail else 0

    suite = run_suite(
        args.model,
        device=args.device,
        cache_dir=args.cache_dir,
        quantize=args.quantize,
    )

    if args.save:
        path = save_suite(suite, args.save)
        print(f"\nSaved baseline to {path}")

    if args.baseline:
        baseline, baseline_meta = load_suite(args.baseline)
        current_meta = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_commit(),
        }
        n_fail = diff_suites(baseline, suite, baseline_meta, current_meta)
        return 1 if n_fail else 0

    if not args.save and not args.baseline:
        # Default: print hashes and previews so the run isn't wasted.
        print("\nNo --save or --baseline given; outputs not persisted.")
        for r in suite.results:
            print(
                f"\n  [{r.name}] hash={r.token_hash[:12]} "
                f"({len(r.token_ids)} tok, {r.elapsed_ms / 1000:.2f}s)"
            )
            print(f"    {r.text[:140]!r}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
