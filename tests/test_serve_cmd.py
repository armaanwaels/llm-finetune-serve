from lfs.serve_vllm import vllm_command


def test_vllm_command_flags():
    cmd = vllm_command("outputs/m/merged", 8000, "bfloat16", 4096, 64, 0.9)
    assert cmd[:3] == ["vllm", "serve", "outputs/m/merged"]
    joined = " ".join(cmd)
    for flag in ("--port 8000", "--dtype bfloat16", "--max-model-len 4096", "--max-num-seqs 64", "--gpu-memory-utilization 0.9"):
        assert flag in joined
