import pytest

from lfs.bench.loadgen import run_level
from tests.fake_server import ServerThread, make_app

PROMPTS = ["a", "b", "c"]


async def test_ttft_and_tokens_measured_at_client():
    app = make_app(ttft_s=0.2, per_token_s=0.01)
    with ServerThread(app) as url:
        results, wall = await run_level(url, "m", PROMPTS, concurrency=1, n_requests=3, n_warmup=0, max_tokens=5, ignore_eos=True)
    assert len(results) == 3
    for r in results:
        assert r.error is None
        assert r.completion_tokens == 5 and r.prompt_tokens == 7
        assert 0.19 <= r.ttft < 0.35
        assert r.e2e >= r.ttft + 4 * 0.01 * 0.9
    assert wall >= 3 * 0.2


async def test_concurrency_is_held_and_warmup_is_excluded():
    app = make_app(ttft_s=0.1, per_token_s=0.0)
    with ServerThread(app) as url:
        results, _ = await run_level(url, "m", PROMPTS, concurrency=4, n_requests=12, n_warmup=4, max_tokens=2, ignore_eos=True)
    assert len(results) == 12          # warmup results are not returned
    assert app.state.total == 16       # but warmup requests were sent
    assert app.state.max_inflight == 4


async def test_request_body_asks_for_usage_and_ignore_eos():
    app = make_app(ttft_s=0.0, per_token_s=0.0)
    with ServerThread(app) as url:
        await run_level(url, "m", PROMPTS, concurrency=1, n_requests=1, n_warmup=0, max_tokens=3, ignore_eos=True)
    body = app.state.bodies[0]
    assert body["stream"] is True and body["ignore_eos"] is True
    assert body["stream_options"] == {"include_usage": True}
    assert body["max_tokens"] == 3 and body["temperature"] == 0.0


async def test_server_errors_are_recorded_not_raised():
    app = make_app(ttft_s=0.0, per_token_s=0.0, fail_every=2)
    with ServerThread(app) as url:
        results, _ = await run_level(url, "m", PROMPTS, concurrency=1, n_requests=4, n_warmup=0, max_tokens=2, ignore_eos=True)
    errors = [r for r in results if r.error]
    assert len(errors) == 2 and errors[0].error.startswith("HTTP 500")


@pytest.mark.parametrize("bad_url", ["http://127.0.0.1:1"])
async def test_connection_failure_is_an_error_result(bad_url):
    results, _ = await run_level(bad_url, "m", PROMPTS, concurrency=1, n_requests=1, n_warmup=0, max_tokens=2, ignore_eos=True, timeout_s=2)
    assert results[0].error is not None
