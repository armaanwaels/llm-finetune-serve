"""Benchmark one running server at several concurrency levels.

    python -m lfs.bench.cli --base-url http://localhost:8000 --server vllm \
        --model outputs/mistral7b/merged --concurrency 1,4,16,32 --out results/mistral7b
"""
import argparse
import asyncio
from pathlib import Path

from lfs.bench import env
from lfs.bench.loadgen import run_level
from lfs.bench.metrics import summarize
from lfs.bench.prompts import make_prompts
from lfs.bench.report import fmt, to_markdown, write_csv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--server", required=True, help="label for the results table, e.g. vllm or hf-generate")
    ap.add_argument("--model", required=True, help="model name the server was started with; also the tokenizer path")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--dataset", default="b-mc2/sql-create-context")
    ap.add_argument("--concurrency", default="1,4,16,32")
    ap.add_argument("--requests-per-level", type=int, default=0, help="0 means 4 x concurrency, at least 16")
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--input-tokens", type=int, default=256)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--no-ignore-eos", action="store_true", help="let generation stop at EOS (output length then varies)")
    ap.add_argument("--out", required=True, help="output prefix; writes <out>.csv and <out>.md")
    a = ap.parse_args()

    levels = [int(x) for x in a.concurrency.split(",")]
    n_max = max(max(4 * c, 16) for c in levels) if not a.requests_per_level else a.requests_per_level
    prompts = make_prompts(a.model, a.dataset, n=n_max, input_tokens=a.input_tokens)

    rows = []
    for c in levels:
        n = a.requests_per_level or max(4 * c, 16)
        results, wall = asyncio.run(run_level(
            a.base_url, a.model, prompts, c, n, a.warmup, a.max_tokens, not a.no_ignore_eos,
        ))
        row = {"server": a.server, **summarize(results, c, wall)}
        rows.append(row)
        print(f"{a.server} c={c}: {fmt(row['output_tok_per_s'])} out tok/s, "
              f"ttft p50 {fmt(row['ttft_p50_ms'])} ms, e2e p95 {fmt(row['e2e_p95_ms'])} ms, errors {row['errors']}", flush=True)
        errs = [r.error for r in results if r.error]
        if errs:
            print("  first error:", errs[0], flush=True)

    meta = {**env.capture(a.model, a.dtype), "input_tokens": a.input_tokens, "max_tokens": a.max_tokens,
            "ignore_eos": not a.no_ignore_eos, "warmup": a.warmup}
    out = Path(a.out)
    write_csv(rows, out.with_suffix(".csv"), meta)
    md = to_markdown(rows, meta)
    out.with_suffix(".md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
