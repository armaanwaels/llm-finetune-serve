"""Closed-loop async load generator for an OpenAI-compatible /v1/completions server.

N workers each keep exactly one request in flight, so the server always sees
`concurrency` concurrent requests. TTFT is taken at the client, from the moment
the request is sent to the first streamed chunk carrying text.
"""
import asyncio
import json
import time

import httpx

from lfs.bench.metrics import RequestResult


async def one_request(client: httpx.AsyncClient, url: str, model: str, prompt: str, max_tokens: int, ignore_eos: bool) -> RequestResult:
    body = {
        "model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0,
        "stream": True, "stream_options": {"include_usage": True},
    }
    if ignore_eos:
        body["ignore_eos"] = True
    start = time.perf_counter()
    first = None
    usage = None
    chunks = 0
    try:
        async with client.stream("POST", url, json=body) as resp:
            if resp.status_code != 200:
                await resp.aread()
                return RequestResult(start, None, time.perf_counter(), 0, 0, f"HTTP {resp.status_code}: {resp.text[:200]}")
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                event = json.loads(data)
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices", []):
                    if choice.get("text"):
                        chunks += 1
                        if first is None:
                            first = time.perf_counter()
    except httpx.HTTPError as e:
        return RequestResult(start, first, time.perf_counter(), 0, 0, f"{type(e).__name__}: {e}")
    end = time.perf_counter()
    if usage is None:
        # Server did not report usage: fall back to chunk count, which undercounts
        # when a chunk carries several tokens. Flag it rather than hide it.
        return RequestResult(start, first, end, 0, chunks, "no usage reported")
    return RequestResult(start, first, end, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))


async def run_level(
    base_url: str, model: str, prompts: list[str], concurrency: int, n_requests: int,
    n_warmup: int, max_tokens: int, ignore_eos: bool, timeout_s: float = 600.0,
) -> tuple[list[RequestResult], float]:
    """Run warmup, then n_requests measured requests at a fixed concurrency.

    Returns the measured results and the wall time they took.
    """
    url = base_url.rstrip("/") + "/v1/completions"
    limits = httpx.Limits(max_connections=concurrency + 4, max_keepalive_connections=concurrency + 4)
    async with httpx.AsyncClient(timeout=timeout_s, limits=limits) as client:
        async def drain(count: int) -> list[RequestResult]:
            queue: asyncio.Queue[int] = asyncio.Queue()
            for i in range(count):
                queue.put_nowait(i)
            results: list[RequestResult] = []

            async def worker():
                while True:
                    try:
                        i = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        return
                    results.append(await one_request(client, url, model, prompts[i % len(prompts)], max_tokens, ignore_eos))

            await asyncio.gather(*(worker() for _ in range(min(concurrency, count))))
            return results

        if n_warmup:
            await drain(n_warmup)  # discarded: first requests pay for CUDA graphs, caches, lazy init
        t0 = time.perf_counter()
        results = await drain(n_requests)
        wall = time.perf_counter() - t0
    return results, wall
