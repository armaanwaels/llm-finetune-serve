"""Write benchmark rows to CSV and a markdown table."""
import csv
from pathlib import Path

COLUMNS = [
    "server", "concurrency", "requests", "errors", "req_per_s", "output_tok_per_s",
    "ttft_p50_ms", "ttft_p95_ms", "ttft_p99_ms", "e2e_p50_ms", "e2e_p95_ms", "e2e_p99_ms",
    "per_request_decode_tok_per_s_p50", "mean_prompt_tokens", "mean_completion_tokens", "wall_s",
]


def fmt(v) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.1f}" if abs(v) >= 10 else f"{v:.2f}"
    return str(v)


def write_csv(rows: list[dict], path: Path, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = COLUMNS + sorted(meta)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({**r, **meta})


def to_markdown(rows: list[dict], meta: dict) -> str:
    shown = ["server", "concurrency", "output_tok_per_s", "req_per_s", "ttft_p50_ms", "ttft_p95_ms",
             "ttft_p99_ms", "e2e_p50_ms", "e2e_p95_ms", "e2e_p99_ms", "errors"]
    lines = [
        "| " + " | ".join(shown) + " |",
        "|" + "---|" * len(shown),
    ]
    for r in rows:
        lines.append("| " + " | ".join(fmt(r.get(c)) for c in shown) + " |")
    lines.append("")
    lines.append("Run environment: " + ", ".join(f"{k}={v}" for k, v in sorted(meta.items())))
    return "\n".join(lines) + "\n"
