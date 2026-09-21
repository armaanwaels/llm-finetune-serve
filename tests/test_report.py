import csv

from lfs.bench.report import COLUMNS, fmt, to_markdown, write_csv

ROW = {"server": "vllm", "concurrency": 4, "requests": 16, "errors": 0, "req_per_s": 2.5,
       "output_tok_per_s": 320.123, "ttft_p50_ms": 45.67, "ttft_p95_ms": None}
META = {"model": "m", "gpu": "none (CPU)", "dtype": "float32"}


def test_csv_has_all_columns_and_meta(tmp_path):
    p = tmp_path / "out" / "r.csv"
    write_csv([ROW, {**ROW, "concurrency": 8}], p, META)
    rows = list(csv.DictReader(open(p)))
    assert len(rows) == 2
    assert set(COLUMNS) <= set(rows[0])
    assert rows[1]["concurrency"] == "8" and rows[0]["gpu"] == "none (CPU)"


def test_markdown_table_shape_and_missing_values():
    md = to_markdown([ROW], META)
    lines = md.splitlines()
    assert lines[0].startswith("| server | concurrency |")
    assert lines[2].startswith("| vllm | 4 | 320.1 |")
    assert "n/a" in lines[2]
    assert "gpu=none (CPU)" in md


def test_fmt():
    assert fmt(None) == "n/a"
    assert fmt(3.14159) == "3.14"
    assert fmt(123.456) == "123.5"
    assert fmt(7) == "7"
