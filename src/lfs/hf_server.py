"""Naive baseline: Hugging Face `generate`, one request at a time.

Exposes the subset of the OpenAI /v1/completions API the benchmark uses
(streaming, max_tokens, ignore_eos, stream_options.include_usage), so the same
client measures both servers. Requests queue behind a lock because plain
`generate` has no way to add a new sequence to a batch that is already decoding.
That is exactly the behaviour continuous batching replaces.

    python -m lfs.hf_server --model outputs/smoke/merged --port 8001
"""
import argparse
import asyncio
import json
import threading
import time
import uuid

import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

DTYPES = {"auto": "auto", "float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


class CountingStreamer:
    """Wraps a streamer and counts generated token ids exactly.

    generate() pushes the prompt ids first; the inner streamer skips those itself.
    """

    def __init__(self, inner):
        self.inner, self.n, self._seen_prompt = inner, 0, False

    def put(self, value):
        if self._seen_prompt:
            self.n += value.shape[-1] if value.dim() > 1 else value.numel()
        self._seen_prompt = True
        self.inner.put(value)

    def end(self):
        self.inner.end()


def build_app(model_path: str, dtype: str) -> FastAPI:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=DTYPES[dtype]).to(device).eval()
    lock = asyncio.Lock()
    app = FastAPI()

    @app.get("/v1/models")
    async def models():
        return {"data": [{"id": model_path, "object": "model"}]}

    @app.post("/v1/completions")
    async def completions(req: Request):
        body = await req.json()
        prompt, max_tokens = body["prompt"], int(body.get("max_tokens", 16))
        ignore_eos = bool(body.get("ignore_eos", False))
        include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
        rid = f"cmpl-{uuid.uuid4().hex[:12]}"
        inputs = tok(prompt, return_tensors="pt").to(device)
        n_prompt = int(inputs["input_ids"].shape[1])
        gen_kwargs = dict(**inputs, max_new_tokens=max_tokens, do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id)
        if ignore_eos:
            gen_kwargs["min_new_tokens"] = max_tokens

        if not body.get("stream"):
            async with lock:
                out = await asyncio.to_thread(model.generate, **gen_kwargs)
            new = out[0, n_prompt:]
            return JSONResponse({
                "id": rid, "object": "text_completion", "model": model_path,
                "choices": [{"index": 0, "text": tok.decode(new, skip_special_tokens=True), "finish_reason": "length"}],
                "usage": {"prompt_tokens": n_prompt, "completion_tokens": int(new.shape[0]), "total_tokens": n_prompt + int(new.shape[0])},
            })

        async def stream():
            async with lock:
                streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
                counting = CountingStreamer(streamer)
                thread = threading.Thread(target=model.generate, kwargs={**gen_kwargs, "streamer": counting})
                thread.start()
                it = iter(streamer)
                while True:
                    text = await asyncio.to_thread(next, it, None)
                    if text is None:
                        break
                    if text:
                        chunk = {"id": rid, "object": "text_completion", "created": int(time.time()), "model": model_path,
                                 "choices": [{"index": 0, "text": text, "finish_reason": None}]}
                        yield f"data: {json.dumps(chunk)}\n\n"
                thread.join()
                if include_usage:
                    usage = {"prompt_tokens": n_prompt, "completion_tokens": counting.n, "total_tokens": n_prompt + counting.n}
                    yield f"data: {json.dumps({'id': rid, 'object': 'text_completion', 'choices': [], 'usage': usage})}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dtype", default="auto", choices=list(DTYPES))
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8001)
    a = ap.parse_args()
    uvicorn.run(build_app(a.model, a.dtype), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
