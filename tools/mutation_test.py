#!/usr/bin/env python
"""Mutation test: plant one known bug at a time in a copy of the repo and check that the tests catch it.

    python tools/mutation_test.py            # all mutants, about 30-60 minutes on a laptop CPU
    python tools/mutation_test.py qb_128     # only the named mutants

Each mutant is a (name, file, original text, buggy text) replacement. It includes every bug found in the
earlier branches of this repo (docs/AUDIT.md) and in code review of this rewrite. A mutant "survives" if
the test suite still passes with the bug in place, which means a test is missing.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W_STE = "w = ste(w, fake_quant_weights(w, self.weight_granularity), lam)"

MUTANTS = [
    # bugs of the earlier branches (docs/AUDIT.md)
    ("1bef_gamma_leak", "bitnet_vision/layers.py", W_STE,
     "g = w.abs().mean().clamp(min=1e-5); wn = w / g; w = (wn + (wn.round().clamp(-1, 1) - wn).detach()) * g"),
    ("6825_clamp_after_ste", "bitnet_vision/layers.py", W_STE,
     "g = w.abs().mean().clamp(min=1e-5).detach(); wn = w / g; "
     "w = (wn + (wn.round() - wn).detach()).clamp(-1, 1) * g"),
    ("8216_wrong_domain_ste", "bitnet_vision/layers.py", W_STE,
     "c, s = quantize_weights(w.detach()); w = (w + (c.to(w.dtype) - w).detach()) / s.to(w.dtype)"),
    ("8216_bias_scaled", "bitnet_vision/layers.py", "y = F.linear(x.to(in_dtype), w, self.bias)",
     "y = F.linear(x.to(in_dtype), w) + (self.bias * 1e-3 if self.bias is not None else 0)"),
    ("no_ste_weights", "bitnet_vision/layers.py", W_STE, "w = fake_quant_weights(w, self.weight_granularity)"),
    ("no_ste_acts", "bitnet_vision/layers.py",
     "x = ste(x, fake_quant_activations(x, self.act_bits, self.act_method), lam)",
     "x = fake_quant_activations(x, self.act_bits, self.act_method)"),
    ("qb_128", "bitnet_vision/quant.py", "scale = qmax / denom.clamp(min=eps)",
     "scale = (qmax + 1) / denom.clamp(min=eps)"),
    ("per_tensor_acts", "bitnet_vision/quant.py", "denom = xf.abs().amax(dim=-1, keepdim=True) if",
     "denom = xf.abs().amax().expand(*xf.shape[:-1], 1) if"),
    ("norm_no_fp32", "bitnet_vision/layers.py", "xf = x if x.dtype == torch.float64 else x.float()", "xf = x"),
    ("no_input_norm", "bitnet_vision/layers.py",
     "self.norm = RMSNorm(in_features, eps=norm_eps, device=device, dtype=dtype) if input_norm else None",
     "self.norm = None"),
    ("binary_weights", "bitnet_vision/quant.py", "codes = (wf * scale).round().clamp(-1, 1)",
     "codes = torch.where(wf >= 0, 1.0, -1.0).to(wf.dtype)"),
    ("quant_in_half", "bitnet_vision/quant.py",
     "return torch.float64 if x.dtype == torch.float64 else torch.float32", "return x.dtype"),
    ("act_eps_missing", "bitnet_vision/quant.py", "scale = qmax / denom.clamp(min=eps)", "scale = qmax / denom"),
    ("weight_eps_missing", "bitnet_vision/quant.py", "scale = 1.0 / gamma.clamp(min=eps)", "scale = 1.0 / gamma"),
    ("pad_then_slice_hadamard", "bitnet_vision/layers.py", "        return self.hadamard(x)\n",
     "        n = x.shape[-1]\n        m = 1 << (n - 1).bit_length()\n"
     "        return HadamardTransform(m)(F.pad(x, (0, m - n)))[..., :n]\n"),
    ("reject_4d_inputs", "bitnet_vision/layers.py", "        in_dtype = x.dtype\n",
     "        in_dtype = x.dtype\n        if x.dim() not in (2, 3):\n            raise ValueError('2D or 3D only')\n"),
    ("skip_hadamard", "bitnet_vision/layers.py", "        return self.hadamard(x)\n", "        return x\n"),
    ("absmean_4bit_wrong", "bitnet_vision/quant.py", "scale = (qmax**0.5) / denom.clamp(min=eps)",
     "scale = qmax / denom.clamp(min=eps)"),
    # numerics and options of this rewrite
    ("hadamard_unnormalized", "bitnet_vision/hadamard.py",
     "mats = [_factor_matrix(f) / math.sqrt(f) for f in factors]", "mats = [_factor_matrix(f) for f in factors]"),
    ("hadamard_bf16_under_autocast", "bitnet_vision/hadamard.py", "        with autocast_disabled(x.device.type):\n",
     "        with contextlib.nullcontext():\n"),
    ("autocast_skips_mps", "bitnet_vision/hadamard.py",
     "    try:\n        return torch.autocast(device_type=device_type, enabled=False)\n",
     "    try:\n        if device_type in (\"cpu\", \"cuda\", \"xpu\"):\n"
     "            return torch.autocast(device_type=device_type, enabled=False)\n"
     "        return contextlib.nullcontext()\n"),
    ("no_random_signs", "bitnet_vision/layers.py",
     "HadamardTransform(in_features, random_signs=random_signs)", "HadamardTransform(in_features)"),
    ("partial_strength_is_hard_switch", "bitnet_vision/layers.py", "        if lam > 0:\n            if self.act_bits",
     "        if lam >= 1:\n            if self.act_bits"),
    ("int4_range_symmetric", "bitnet_vision/quant.py", "return -(2 ** (bits - 1)), 2 ** (bits - 1) - 1",
     "return -(2 ** (bits - 1)) + 1, 2 ** (bits - 1) - 1"),
    ("quant_strength_ignored", "bitnet_vision/layers.py", "lam = self.quant_strength", "lam = 1.0"),
    ("half_inputs_not_upcast", "bitnet_vision/layers.py",
     "            x = x.float()  # norm, rotation and rounding in float32", "            pass  # "),
    # export and checkpoints
    ("pack_layout_reversed", "bitnet_vision/export.py",
     "packed |= u[i * block : (i + 1) * block] << (2 * i)",
     "packed |= u[i * block : (i + 1) * block] << (2 * (3 - i))"),
    ("export_scale_inverted", "bitnet_vision/export.py",
     "y = self._int_matmul(codes) / x_scale * self.weight_scale.float()",
     "y = self._int_matmul(codes) / x_scale / self.weight_scale.float()"),
    ("deadzone_not_folded", "bitnet_vision/export.py",
     "bias = bias + layer.deadzone_lambda * layer.deadzone_rowsum().detach()", "bias = bias"),
    ("cache_not_invalidated", "bitnet_vision/export.py",
     "self._w_cache = {}  # new weights: forget the unpacked copy", "pass"),
    ("transfer_no_rotation", "bitnet_vision/checkpoint.py", "            w = w @ r_dst.to(w.device)\n",
     "            pass\n"),
    ("transfer_no_unrotation", "bitnet_vision/checkpoint.py",
     "            w = w @ r_src.T.to(w.device)  # undo the source rotation\n", "            pass\n"),
    ("old_checkpoints_get_signs", "bitnet_vision/checkpoint.py",
     'return {"hadamard_random_signs": False, **config}', "return dict(config)"),
    # training
    ("schedule_wd_never_off", "bitnet_vision/training/schedule.py",
     "wd = self.stage2_weight_decay if stage2 else self.weight_decay", "wd = self.weight_decay"),
    ("decay_norms_too", "bitnet_vision/training/optim.py", "if p.ndim >= 2 and name not in skip:",
     "if name not in skip:"),
    ("teacher_not_detached", "bitnet_vision/training/distill.py",
     "F.log_softmax(teacher_logits.detach().float() / t, dim=-1)", "F.log_softmax(teacher_logits.float() / t, dim=-1)"),
    ("distill_no_t_squared", "bitnet_vision/training/distill.py", "    ) * (t * t)\n", "    )\n"),
    ("drop_path_no_rescale", "bitnet_vision/models/vit.py", "return x * mask / keep", "return x * mask"),
    ("act_switch_config_stale", "bitnet_vision/training/engine.py",
     "            config.hadamard_act_bits = bits\n", "            pass\n"),
    ("resume_rng_not_restored", "bitnet_vision/training/engine.py",
     'torch.set_rng_state(resume["torch_rng"])', "pass"),
    ("unfinished_ramp_allowed", "bitnet_vision/training/engine.py",
     "if quantized and quant_strength_at(total_steps - 1) < 1.0:", "if False:"),
    ("resume_settings_unchecked", "bitnet_vision/training/engine.py", "        if changed:\n            raise",
     "        if False:\n            raise"),
    ("checksum_ignored", "bitnet_vision/training/data.py", "if _md5(archive) != CIFAR10_MD5:", "if False:"),
    ("script_seed_ignored", "scripts/train.py",
     "torch.manual_seed(a.seed)  # before building the model", "pass  # before building the model"),
    # LoRA adapters
    ("lora_b_not_zero_init", "bitnet_vision/lora.py",
     "        nn.init.zeros_(b.weight)  # so a new adapter adds exactly 0\n", "        pass\n"),
    ("lora_scale_alpha_times_r", "bitnet_vision/lora.py",
     "return self.alpha / (math.sqrt(self.r) if self.use_rslora else self.r)",
     "return self.alpha * (math.sqrt(self.r) if self.use_rslora else self.r)"),
    ("lora_rslora_ignored", "bitnet_vision/lora.py",
     "return self.alpha / (math.sqrt(self.r) if self.use_rslora else self.r)", "return self.alpha / self.r"),
    ("lora_strength_ignored", "bitnet_vision/lora.py", "d = F.linear(F.linear(h, a), b) * (s.strength * s.scale)",
     "d = F.linear(F.linear(h, a), b) * s.scale"),
    ("lora_reads_normalized_input", "bitnet_vision/layers.py", "return lora_forward(self, raw, y)",
     "return lora_forward(self, x, y)"),
    ("lora_branch_under_autocast", "bitnet_vision/lora.py", "    with autocast_disabled(x.device.type):\n",
     "    with torch.autograd.profiler.record_function('lora'):\n"),
    ("lora_dropout_in_eval", "bitnet_vision/lora.py", "F.dropout(xf, s.dropout, layer.training)",
     "F.dropout(xf, s.dropout, True)"),
    ("lora_dropped_on_export", "bitnet_vision/export.py",
     "        copy_lora(layer, new)\n        return new.to(layer.weight.device)\n",
     "        return new.to(layer.weight.device)\n"),
    ("lora_dropped_by_fused_layer", "bitnet_vision/apple/mps_kernels.py",
     "return lora_forward(self, x, fused_epilogue(acc, x_scale, self.weight_scale, self.bias).to(x.dtype))",
     "return fused_epilogue(acc, x_scale, self.weight_scale, self.bias).to(x.dtype)"),
    ("lora_dropped_by_mlx", "bitnet_vision/apple/mlx_ternary.py", "                adapters=_adapters(layer),\n", ""),
    ("lora_adapters_ternarized", "bitnet_vision/layers.py",
     "            if in_lora_adapter(full):\n                continue\n", ""),
    ("lora_hook_registered_twice", "bitnet_vision/lora.py",
     "if not _native(layer) and _linear_hook not in layer._forward_hooks.values():", "if not _native(layer):"),
    ("lora_not_in_checkpoint", "bitnet_vision/checkpoint.py",
     'return {**payload, "lora": lora} if lora else payload', "return payload"),
    ("lora_file_keeps_adapter_name", "bitnet_vision/lora.py",
     'tensors[PEFT_PREFIX + _join(n, "lora_A.weight")]', 'tensors[PEFT_PREFIX + _join(n, f"lora_A.{name}.weight")]'),
    ("lora_alpha_pattern_ignored", "bitnet_vision/lora.py",
     'alpha = _pattern_value(config.get("alpha_pattern"), module, config.get("lora_alpha", r))',
     'alpha = config.get("lora_alpha", r)'),
    ("lora_merge_not_rotated", "bitnet_vision/lora.py", "            delta = rotate(delta)\n", "            pass\n"),
    ("lora_freeze_skipped", "bitnet_vision/lora.py", "        p.requires_grad_(on)\n",
     "        p.requires_grad_(True)\n"),
    ("lora_resume_unchecked", "bitnet_vision/training/engine.py",
     'if resume.get("lora", {}) != lora_config(model):', "if False:"),
]


def run(name: str, rel: str, old: str, new: str) -> tuple[str, str, str]:
    tmp = tempfile.mkdtemp(prefix=f"mutant_{name}_")
    try:
        dst = os.path.join(tmp, "repo")
        shutil.copytree(REPO, dst, ignore=shutil.ignore_patterns(".git", "__pycache__", "*_cache", "runs", "data"))
        path = os.path.join(dst, rel)
        with open(path) as f:
            src = f.read()
        if src.count(old) != 1:
            return name, "BAD MUTANT", f"pattern found {src.count(old)} times in {rel}"
        with open(path, "w") as f:
            f.write(src.replace(old, new))
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        p = subprocess.run([sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider"], cwd=dst, env=env,
                           capture_output=True, text=True, timeout=1800)
        failed = [line for line in p.stdout.splitlines() if line.startswith(("FAILED", "ERROR"))]
        return name, ("caught" if p.returncode != 0 else "SURVIVED"), (failed[0] if failed else "")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main(argv: list[str]) -> int:
    chosen = [m for m in MUTANTS if not argv or m[0] in argv]
    results = []
    for mutant in chosen:
        results.append(run(*mutant))
        print(f"{results[-1][1]:10s} {results[-1][0]:32s} {results[-1][2]}", flush=True)
    caught = sum(r[1] == "caught" for r in results)
    print(f"\n{caught} of {len(results)} mutants caught")
    return 0 if caught == len(results) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
