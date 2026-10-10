#!/usr/bin/env bash
# Reproduce the CIFAR-10 results in the README: a full-precision baseline, a ternary model trained from
# scratch, and a ternary model initialized from and distilled by the baseline.
#
#   DATA=data OUT=runs EPOCHS=15 THREADS=4 bash scripts/cifar_experiments.sh
#
# Each run resumes from <run>/last.pt if it was interrupted.
set -euo pipefail
export PYTHONUNBUFFERED=1

DATA=${DATA:-data}
OUT=${OUT:-runs}
EPOCHS=${EPOCHS:-15}
KD_EPOCHS=${KD_EPOCHS:-10}
THREADS=${THREADS:-4}
PY=${PYTHON:-python}
HERE=$(cd "$(dirname "$0")" && pwd)
COMMON=(--data-dir "$DATA" --threads "$THREADS" --resume --log-every 100)

run() {  # name, args...
  local name=$1; shift
  if [[ -f "$OUT/$name/summary.json" ]]; then echo "skip $name (done)"; return; fi
  mkdir -p "$OUT/$name"
  "$PY" "$HERE/train.py" --out "$OUT/$name" "${COMMON[@]}" "$@" 2>&1 | tee -a "$OUT/$name/log.txt"
}

# 1) full-precision twin: same architecture, quantization off
run fp --fp --epochs "$EPOCHS" --schedule cosine --lr 1e-3 --wd 0.05

# 2) ternary from scratch, BitNet b1.58 two-stage schedule (lr x2/3 and weight decay off at the midpoint)
run ternary_scratch --epochs "$EPOCHS" --schedule bitnet --lr 1e-3 --wd 0.1

# 3) ternary initialized from the fp run and distilled from it (KL only, T=1), with FTerViT's short
#    low-lr restart for the last 10% of steps
run ternary_kd --epochs "$KD_EPOCHS" --init "$OUT/fp/model.pt" --teacher "$OUT/fp/model.pt" \
  --schedule restart --lr 2e-4 --wd 0.05 --stage2-wd 0.01 --warmup-epochs 0.5
