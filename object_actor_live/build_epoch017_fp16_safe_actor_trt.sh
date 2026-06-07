#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="/home/michel/miniconda3/envs/voice_id/bin/python"
TRTEXEC="/usr/src/tensorrt/bin/trtexec"

CHECKPOINT="$REPO_DIR/epoch=017.ckpt"
OUT_DIR="$REPO_DIR/object_actor_live/exports/epoch017_actor_fp16_safeattn_prune"
ONNX="$OUT_DIR/epoch=017_b1_t16_k8_o32_224.onnx"
ENGINE="$OUT_DIR/epoch=017_b1_t16_k8_o32_224_fp16.engine"
CHECK_JSON="$ENGINE.check.json"

for path in "$PYTHON_BIN" "$TRTEXEC" "$CHECKPOINT"; do
  if [[ ! -e "$path" ]]; then
    echo "missing required path: $path" >&2
    exit 1
  fi
done

cd "$REPO_DIR"

"$PYTHON_BIN" utils/export_actor_tensorrt.py \
  --checkpoint "$CHECKPOINT" \
  --out-dir "$OUT_DIR" \
  --precision fp16 \
  --workspace-mib 512 \
  --max-aux-streams 0 \
  --builder-optimization-level 0 \
  --mask-input-dtype bool \
  --trt-safe-attention \
  --force

"$PYTHON_BIN" utils/check_actor_tensorrt.py \
  --checkpoint "$CHECKPOINT" \
  --onnx "$ONNX" \
  --engine "$ENGINE" \
  --trt-safe-attention \
  --out-json "$CHECK_JSON" \
  --max-abs-tolerance 0.08

"$TRTEXEC" \
  --loadEngine="$ENGINE" \
  --duration=5 \
  --warmUp=500 \
  --iterations=50 \
  --useCudaGraph

echo "actor TensorRT engine: $ENGINE"
echo "drift check: $CHECK_JSON"
