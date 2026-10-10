"""End-to-end check of the command-line scripts on a tiny fake CIFAR-10 folder."""

import importlib.util
import json
import os

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_script(name):
    spec = importlib.util.spec_from_file_location(f"script_{name}", os.path.join(ROOT, "scripts", f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_cifar(root, per_file=40):
    folder = os.path.join(root, "cifar-10-batches-bin")
    os.makedirs(folder)
    g = torch.Generator().manual_seed(0)
    for name in [f"data_batch_{i}.bin" for i in range(1, 6)] + ["test_batch.bin"]:
        labels = (torch.arange(per_file) % 10).to(torch.uint8)
        pixels = torch.randint(0, 256, (per_file, 3072), dtype=torch.uint8, generator=g)
        with open(os.path.join(folder, name), "wb") as f:
            f.write(bytes(torch.cat([labels[:, None], pixels], 1).flatten().tolist()))


def test_train_eval_and_bench_scripts(tmp_path):
    data = str(tmp_path / "data")
    _fake_cifar(data)
    train = _load_script("train")
    fp_out = str(tmp_path / "fp")
    summary = train.main(["--data-dir", data, "--out", fp_out, "--fp", "--epochs", "1", "--batch-size", "50",
                          "--log-every", "0"])
    assert summary["quantized"] is False

    out = str(tmp_path / "ternary")
    summary = train.main(["--data-dir", data, "--out", out, "--epochs", "1", "--batch-size", "50", "--log-every",
                          "0", "--init", f"{fp_out}/model.pt", "--teacher", f"{fp_out}/model.pt", "--hadamard",
                          "--schedule", "restart", "--quant-ramp-epochs", "0.5",
                          "--final-hadamard-act-bits", "4", "--final-act-bits-fraction", "0.5"])
    assert summary["compression"] > 10
    assert summary["packed_test_acc"] == summary["final_test_acc"] or abs(
        summary["packed_test_acc"] - summary["final_test_acc"]) <= 0.05
    for f in ("model.pt", "packed.pt", "last.pt", "metrics.json", "summary.json", "args.json"):
        assert os.path.exists(os.path.join(out, f))
    with open(os.path.join(out, "metrics.json")) as fh:
        cfg = json.load(fh)["model_config"]
    assert cfg["hadamard"] is True and cfg["hadamard_act_bits"] == 4  # switched to 4-bit for the last steps

    evaluate = _load_script("eval")
    info = evaluate.main([os.path.join(out, "packed.pt"), "--data-dir", data])
    assert info["kind"] == "packed" and info["acc"] == summary["packed_test_acc"]
    info = evaluate.main([os.path.join(out, "model.pt"), "--data-dir", data])
    assert info["kind"] == "training" and info["acc"] == summary["final_test_acc"]

    bench = _load_script("bench")
    res = bench.main(["--batch-size", "2", "--repeats", "1"])
    assert res["bytes_fp32"] / res["bytes_packed"] > 10
