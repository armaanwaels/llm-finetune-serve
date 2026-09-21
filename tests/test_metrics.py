import math

import pytest

from lfs.bench.metrics import RequestResult, percentile, summarize


def test_percentile_matches_numpy_linear():
    xs = [1.0, 2.0, 3.0, 4.0]
    assert percentile(xs, 0) == 1.0
    assert percentile(xs, 100) == 4.0
    assert percentile(xs, 50) == 2.5
    # numpy.percentile([1,2,3,4], 95) == 3.85
    assert math.isclose(percentile(xs, 95), 3.85)
    assert percentile([5.0], 99) == 5.0


def test_percentile_is_order_independent():
    assert percentile([4.0, 1.0, 3.0, 2.0], 50) == 2.5


def test_percentile_rejects_bad_input():
    with pytest.raises(ValueError):
        percentile([], 50)
    with pytest.raises(ValueError):
        percentile([1.0], 101)


def test_request_timings():
    r = RequestResult(start=10.0, first_token=10.2, end=11.2, prompt_tokens=5, completion_tokens=11)
    assert math.isclose(r.ttft, 0.2)
    assert math.isclose(r.e2e, 1.2)
    # 10 tokens after the first, over 1.0 s
    assert math.isclose(r.decode_tps, 10.0)


def test_decode_tps_undefined_for_single_token_or_no_first_token():
    assert RequestResult(0, 0.1, 0.2, 1, 1).decode_tps is None
    assert RequestResult(0, None, 0.2, 1, 0).decode_tps is None


def test_summarize_throughput_uses_wall_time_and_skips_errors():
    rs = [
        RequestResult(0.0, 0.1, 1.0, 10, 100),
        RequestResult(0.0, 0.3, 2.0, 10, 100),
        RequestResult(0.0, None, 0.5, 0, 0, error="HTTP 500"),
    ]
    row = summarize(rs, concurrency=2, wall_s=2.0)
    assert row["requests"] == 3 and row["errors"] == 1
    assert row["output_tok_per_s"] == 100.0      # 200 tokens / 2 s
    assert row["req_per_s"] == 1.0
    assert math.isclose(row["ttft_p50_ms"], 200.0)
    assert math.isclose(row["e2e_p50_ms"], 1500.0)
    assert row["mean_completion_tokens"] == 100.0


def test_summarize_all_errors_does_not_crash():
    row = summarize([RequestResult(0, None, 1, 0, 0, error="x")], concurrency=1, wall_s=1.0)
    assert row["ttft_p50_ms"] is None and row["output_tok_per_s"] == 0.0
