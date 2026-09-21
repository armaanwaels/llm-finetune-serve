"""Pure functions over per-request timings. No I/O, so they are easy to test."""
from dataclasses import dataclass


@dataclass
class RequestResult:
    start: float            # perf_counter when the request was sent
    first_token: float | None  # perf_counter when the first non-empty text chunk arrived
    end: float              # perf_counter when the stream closed
    prompt_tokens: int
    completion_tokens: int
    error: str | None = None

    @property
    def ttft(self) -> float | None:
        return None if self.first_token is None else self.first_token - self.start

    @property
    def e2e(self) -> float:
        return self.end - self.start

    @property
    def decode_tps(self) -> float | None:
        """Tokens/s after the first token, for one request."""
        if self.first_token is None or self.completion_tokens < 2:
            return None
        span = self.end - self.first_token
        return (self.completion_tokens - 1) / span if span > 0 else None


def percentile(values: list[float], p: float) -> float:
    """Linear interpolation between closest ranks, same as numpy's default."""
    if not values:
        raise ValueError("percentile of empty list")
    if not 0 <= p <= 100:
        raise ValueError("p must be in [0, 100]")
    xs = sorted(values)
    k = (len(xs) - 1) * p / 100
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def summarize(results: list[RequestResult], concurrency: int, wall_s: float) -> dict:
    """One row of the results table for a concurrency level.

    wall_s is the time from the first measured request being sent to the last one
    finishing, so aggregate throughput includes queueing, not just decode time.
    """
    ok = [r for r in results if r.error is None]
    ttfts = [r.ttft for r in ok if r.ttft is not None]
    e2es = [r.e2e for r in ok]
    decode = [r.decode_tps for r in ok if r.decode_tps is not None]
    out_tokens = sum(r.completion_tokens for r in ok)
    row = {
        "concurrency": concurrency,
        "requests": len(results),
        "errors": len(results) - len(ok),
        "wall_s": wall_s,
        "req_per_s": len(ok) / wall_s if wall_s > 0 else 0.0,
        "output_tok_per_s": out_tokens / wall_s if wall_s > 0 else 0.0,
        "mean_prompt_tokens": sum(r.prompt_tokens for r in ok) / len(ok) if ok else 0.0,
        "mean_completion_tokens": out_tokens / len(ok) if ok else 0.0,
        "per_request_decode_tok_per_s_p50": percentile(decode, 50) if decode else None,
    }
    for name, vals in (("ttft", ttfts), ("e2e", e2es)):
        for p in (50, 95, 99):
            row[f"{name}_p{p}_ms"] = percentile(vals, p) * 1000 if vals else None
    return row
