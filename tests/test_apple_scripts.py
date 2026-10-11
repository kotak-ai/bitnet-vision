"""scripts/apple_check.py and scripts/apple_bench.py: dry runs on the CPU with tiny shapes, and their helpers.

On the Mac the scripts compare MPS, the Metal kernels and MLX with the CPU; here only the CPU runs, so these tests
check that every row ends as ok, FAILED (with the error) or skipped (with a reason), that nothing crashes when MPS,
Metal or MLX is missing, and that the JSON file holds the rows and the machine details.
"""

import copy
import importlib.util
import json
import math
import os

import pytest
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HAS_MLX = importlib.util.find_spec("mlx") is not None
NO_MPS = pytest.mark.skipif(torch.backends.mps.is_available(), reason="checks the behaviour without an MPS device")


def _load_script(name):
    spec = importlib.util.spec_from_file_location(f"script_{name}", os.path.join(ROOT, "scripts", f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def check():
    return _load_script("apple_check")


@pytest.fixture(scope="module")
def bench():
    return _load_script("apple_bench")


@pytest.fixture
def keep_mlx_device():
    """The bench sets MLX's default device; put it back so later MLX tests still run on the GPU of a Mac."""
    if not HAS_MLX:
        yield
        return
    import mlx.core as mx

    device = mx.default_device()
    yield
    mx.set_default_device(device)


def _rows(result, path):
    with open(path) as f:
        saved = json.load(f)
    assert [(r["id"], r["status"]) for r in saved["rows"]] == [(r["id"], r["status"]) for r in result["rows"]]
    assert saved["summary"] == result["summary"]
    for key in ("python", "torch", "git_commit", "mps_available", "metal_kernels_available", "mlx"):
        assert key in saved["environment"]
    for row in saved["rows"]:
        assert row["status"] in ("ok", "FAILED", "skipped"), row
        if row["status"] != "ok":
            assert row["message"], row
    return saved["rows"]


@pytest.mark.slow
def test_check_dry_run(check, tmp_path):
    out = tmp_path / "check.json"
    result = check.main(["--device", "cpu", "--quick", "--out", str(out)])
    rows = _rows(result, out)
    by_id = {r["id"]: r for r in rows}
    assert {r["id"].split(".")[0] for r in rows} == {str(i) for i in range(1, 9)}
    failed = [(r["id"], r["message"]) for r in rows if r["status"] == "FAILED"]
    assert not failed
    assert by_id["6.1"]["status"] == "skipped" and "Apple GPU" in by_id["6.1"]["message"]
    for r in rows:
        if r["id"].startswith("7."):
            assert r["status"] == ("ok" if HAS_MLX else "skipped"), r
            if not HAS_MLX:
                assert "pip install mlx" in r["message"]
        elif r["id"] != "6.1":
            assert r["status"] == "ok", r
    assert "outcome" in by_id["4.4"]  # what torch.func.jvp through F.scaled_dot_product_attention did
    assert by_id["6.2"]["message"].startswith("reference ops")  # no Metal here
    assert all(c["value"] <= c["limit"] for c in by_id["8.2"]["checks"])


@pytest.mark.slow
def test_bench_dry_run(bench, tmp_path, keep_mlx_device):
    out = tmp_path / "bench.json"
    result = bench.main(["--device", "cpu", "--quick", "--out", str(out)])
    rows = _rows(result, out)
    assert {r["section"] for r in rows} == {"A", "B", "C"}
    for r in rows:
        if r["status"] == "FAILED":
            # Old PyTorch has no float16 LayerNorm on the CPU; nothing else may fail.
            assert r.get("variant") == "full precision fp16" and "Half" in r["message"], r
        elif r["status"] == "skipped":
            assert r["variant"].startswith(("MLX", "int8pack")), r
            assert not r["variant"].startswith("MLX") or not HAS_MLX
            assert not r["variant"].startswith("int8pack") or not hasattr(torch, "_weight_int8pack_mm")
        else:
            assert 0 < r["p10_ms"] <= r["ms"] <= r["p90_ms"] and r["rounds"] == 1
    ok = [r for r in rows if r["status"] == "ok"]
    for r in ok:
        if r["section"] == "A":
            assert r["tflops"] > 0 and r["vs_fp32_linear"] > 0
            if "err_vs_cpu" in r:
                assert r["err_vs_cpu"] <= 5e-2
        else:
            assert r["images_per_s"] > 0
        if r["section"] == "C":
            assert math.isfinite(r["loss"]) and r["million_images_per_day"] > 0
    fused = [r for r in ok if r["variant"].lower().startswith("fused")]
    assert fused and all(r["path"].startswith("reference ops") for r in fused)
    steps = {r["variant"] for r in ok if r["section"] == "C"}
    assert {"full precision, jvp step", "ternary, jvp step"} <= steps


def test_failures_become_rows(check, bench, tmp_path, monkeypatch, keep_mlx_device):
    def boom(self):
        raise RuntimeError("boom")

    monkeypatch.setattr(check.Checks, "torch_version", boom)
    result = check.main(["--device", "cpu", "--quick", "--only", "1", "--out", str(tmp_path / "c.json")])
    rows = _rows(result, tmp_path / "c.json")
    assert rows[0]["status"] == "FAILED" and rows[0]["message"] == "RuntimeError: boom" and "traceback" in rows[0]
    assert rows[1]["status"] == "ok"

    def broken(*args, **kwargs):
        raise RuntimeError("no fused layer")

    monkeypatch.setattr(bench, "convert_to_fused", broken)
    monkeypatch.setattr(bench, "QUICK_TOKENS", (8,))
    monkeypatch.setattr(bench.Bench, "train_case", lambda *a, **k: boom(None))
    result = bench.main(["--device", "cpu", "--quick", "--only", "A,C", "--out", str(tmp_path / "b.json")])
    rows = _rows(result, tmp_path / "b.json")
    fused = [r for r in rows if r.get("variant") == "Fused (Metal)"]
    assert len(fused) == 2 and all(r["status"] == "FAILED" and "no fused layer" in r["message"] for r in fused)
    assert all(r["status"] == "ok" for r in rows if r.get("variant") == "F.linear fp32")
    training = [r for r in rows if r["section"] == "C"]  # an error outside the variants is one FAILED row per case
    assert [r["status"] for r in training] == ["FAILED", "FAILED"] and training[0]["message"] == "RuntimeError: boom"


def test_interrupt_keeps_what_was_measured(check, tmp_path, monkeypatch):
    def interrupt(self):
        raise KeyboardInterrupt

    monkeypatch.setattr(check.Checks, "device_info", interrupt)
    result = check.main(["--device", "cpu", "--only", "1", "--out", str(tmp_path / "c.json")])
    assert result["interrupted"] and [r["id"] for r in _rows(result, tmp_path / "c.json")] == ["1.1"]


@NO_MPS
def test_mps_requested_without_mps(check, bench, tmp_path, keep_mlx_device):
    result = check.main(["--device", "mps", "--quick", "--only", "1,2,6,7", "--out", str(tmp_path / "c.json")])
    rows = {r["id"]: r for r in _rows(result, tmp_path / "c.json")}
    assert rows["1.2"]["status"] == "FAILED" and "is_available() is False" in rows["1.2"]["message"]
    for row_id, row in rows.items():
        if row_id[0] in "26":
            assert row["status"] == "skipped", row
        if row_id[0] == "7":
            assert row["status"] == "skipped" and ("Metal" in row["message"] or "pip install mlx" in row["message"])
    result = bench.main(["--device", "mps", "--quick", "--only", "B", "--out", str(tmp_path / "b.json")])
    rows = _rows(result, tmp_path / "b.json")
    assert rows and all(r["status"] == "skipped" for r in rows)


def test_arguments_are_checked(check, bench):
    with pytest.raises(SystemExit):
        check.main(["--only", "9"])
    with pytest.raises(SystemExit):
        bench.main(["--only", "D"])
    with pytest.raises(SystemExit):
        bench.main(["--repeats", "0"])


def test_environment_helpers(check, monkeypatch):
    text = "Graphics/Displays:\n\n    Apple M2 Max:\n\n      Chipset Model: Apple M2 Max\n      Type: GPU\n" \
           "      Bus: Built-In\n      Total Number of Cores: 38\n      Vendor: Apple (0x106b)\n"
    assert check.parse_gpu_info(text) == {"gpu": "Apple M2 Max", "gpu_cores": 38}
    assert check.parse_gpu_info("") == {}
    assert [check.chip_generation(c) for c in ("Apple M2 Max", "Apple M3 Ultra", "Apple M10", "Intel", None)] == [
        2, 3, 10, None, None]

    m2, m3 = {"chip": "Apple M2 Max"}, {"chip": "Apple M3 Max"}
    monkeypatch.setattr(check, "torch_version_tuple", lambda: (2, 13))
    assert "177116" in check.torch_warning(m2, "mps") and "177116" in check.torch_warning({}, "mps")
    assert check.torch_warning(m3, "mps") is None and check.torch_warning(m2, "cpu") is None
    monkeypatch.setattr(check, "torch_version_tuple", lambda: (2, 14))
    assert check.torch_warning(m2, "mps") is None
    assert check.banner("x" * 300).count("\n") >= 4  # wrapped to 100 columns


def test_verdict_and_code_entries(check):
    ok = check.verdict([check.entry("a", 1e-7, 1e-5, 1e-8), check.entry("codes", 1, 1, count=True)])
    assert ok["ok"] and ok["message"].startswith("worst: a ")  # a whole-number limit that holds is not named
    bad = check.verdict([check.entry("a", float("nan"), 1.0), check.entry("b", 0.5, 1.0)])
    assert not bad["ok"] and "OVER LIMIT: a" in bad["message"]

    pre = torch.tensor([[0.2, 1.5004, 2.7, -3.9]])  # values before rounding; 1.5004 is next to a tie
    ref = pre.round()
    near = ref.clone()
    near[0, 1] += 1  # flipped at the tie: allowed
    entries, share = check.code_entries(near, ref, pre)
    assert check.verdict(entries)["ok"] and share == 0.25
    away = ref.clone()
    away[0, 2] += 1  # flipped far from a tie: not allowed
    assert not check.verdict(check.code_entries(away, ref, pre)[0])["ok"]
    two = ref.clone()
    two[0, 1] += 2
    assert not check.verdict(check.code_entries(two, ref, pre)[0])["ok"]


def test_clear_of_weight_ties(check):
    torch.manual_seed(0)
    model = check.build_model("T/4", quick=False)  # one of its weights lies 4e-8 from a ternary rounding tie
    channel = check.BitLinear(384, 96, weight_granularity="channel")
    layers = [*check.bitlinear_modules(model), channel]
    before = [layer.weight.detach().clone() for layer in layers]
    check.clear_of_weight_ties(model)
    check.clear_of_weight_ties(channel)
    for layer, w0 in zip(layers, before):
        w = layer.weight.detach()
        a = w.double().abs()
        gamma = a.mean() if layer.weight_granularity == "tensor" else a.mean(dim=1, keepdim=True)
        assert ((a / gamma - 0.5).abs() >= 1e-5).all()
        assert (w != w0).sum() <= max(2, w.numel() // 10000)  # only weights next to a tie move
        codes32, codes64 = (check.quantize_weights(v, granularity=layer.weight_granularity)[0] for v in (w, w.double()))
        assert torch.equal(codes32.double(), codes64)


def test_explicit_attention_matches_sdpa(check):
    torch.manual_seed(0)
    model = check.build_model("T/4", quick=True, quantize=False).eval()
    explicit = check.use_explicit_attention(copy.deepcopy(model))
    x = torch.randn(3, 3, 16, 16)
    with torch.no_grad():
        torch.testing.assert_close(explicit(x), model(x), rtol=1e-5, atol=1e-6)
    attn = explicit.blocks[0].attn
    assert attn.forward.__func__ is check.explicit_attention_forward
    tokens = torch.randn(2, 5, attn.qkv.in_features)
    q, k, v = attn.qkv(tokens).reshape(2, 5, 3, attn.num_heads, -1).permute(2, 0, 3, 1, 4)
    with torch.no_grad():
        expected = attn.proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(2, 5, -1))
        torch.testing.assert_close(attn(tokens), expected, rtol=1e-5, atol=1e-6)
