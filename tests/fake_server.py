"""A tiny OpenAI-style streaming server with known timings, for testing the client."""
import asyncio
import json
import socket
import threading
import time

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse


def make_app(ttft_s: float, per_token_s: float, fail_every: int = 0) -> FastAPI:
    app = FastAPI()
    app.state.inflight = 0
    app.state.max_inflight = 0
    app.state.total = 0
    app.state.bodies = []

    @app.post("/v1/completions")
    async def completions(req: Request):
        body = await req.json()
        app.state.total += 1
        app.state.bodies.append(body)
        if fail_every and app.state.total % fail_every == 0:
            return JSONResponse({"error": "boom"}, status_code=500)
        n = body["max_tokens"]

        async def stream():
            app.state.inflight += 1
            app.state.max_inflight = max(app.state.max_inflight, app.state.inflight)
            try:
                await asyncio.sleep(ttft_s)
                for i in range(n):
                    if i:
                        await asyncio.sleep(per_token_s)
                    yield f"data: {json.dumps({'choices': [{'index': 0, 'text': 'x'}]})}\n\n"
                usage = {"prompt_tokens": 7, "completion_tokens": n, "total_tokens": 7 + n}
                yield f"data: {json.dumps({'choices': [], 'usage': usage})}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                app.state.inflight -= 1

        return StreamingResponse(stream(), media_type="text/event-stream")

    return app


class ServerThread:
    def __init__(self, app: FastAPI):
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        self.port = sock.getsockname()[1]
        sock.close()
        self.app = app
        self.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="error"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def __enter__(self):
        self.thread.start()
        while not self.server.started:
            time.sleep(0.01)
        return f"http://127.0.0.1:{self.port}"

    def __exit__(self, *exc):
        self.server.should_exit = True
        self.thread.join(timeout=5)
