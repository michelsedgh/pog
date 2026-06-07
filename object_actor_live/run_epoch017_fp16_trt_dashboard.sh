#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="/home/michel/miniconda3/envs/voice_id/bin/python"

CHECKPOINT="$REPO_DIR/epoch=017.ckpt"
ACTOR_ENGINE="$REPO_DIR/object_actor_live/exports/epoch017_actor_fp16_safeattn_prune/epoch=017_b1_t16_k8_o32_224_fp16.engine"
DETECTOR_ENGINE="$REPO_DIR/object_actor_live/exports/rfdetr_nano/inference_model_fp16.engine"
LOG_DIR="$REPO_DIR/object_actor_live/logs"
LOG_FILE="$LOG_DIR/epoch017_fp16_trt_dashboard.log"
PID_FILE="$LOG_DIR/epoch017_fp16_trt_dashboard.pid"

PORT="${PORT:-7860}"
CAMERA="${CAMERA:-0}"
HOST="${HOST:-0.0.0.0}"
DET_THRESHOLD="${DET_THRESHOLD:-0.35}"
OBJECT_THRESHOLD="${OBJECT_THRESHOLD:-0.25}"
CROP_MODE="${CROP_MODE:-actor_window}"
LIVE_OBJECT_TOKENS="${LIVE_OBJECT_TOKENS:-1}"

for path in "$PYTHON_BIN" "$CHECKPOINT" "$ACTOR_ENGINE" "$DETECTOR_ENGINE"; do
  if [[ ! -e "$path" ]]; then
    echo "missing required path: $path" >&2
    exit 1
  fi
done

mkdir -p "$LOG_DIR"

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(cat "$PID_FILE")"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "dashboard already running with pid $old_pid" >&2
    exit 1
  fi
fi

cd "$REPO_DIR"

{
  echo "starting epoch017 FP16 TensorRT dashboard"
  echo "checkpoint: $CHECKPOINT"
  echo "actor_engine: $ACTOR_ENGINE"
  echo "detector_engine: $DETECTOR_ENGINE"
  echo "url: http://localhost:$PORT"
} > "$LOG_FILE"

setsid "$PYTHON_BIN" -u live_actor_dashboard.py \
  --checkpoint "$CHECKPOINT" \
  --actor-engine "$ACTOR_ENGINE" \
  --detector-engine "$DETECTOR_ENGINE" \
  --camera "$CAMERA" \
  --host "$HOST" \
  --port "$PORT" \
  --det-threshold "$DET_THRESHOLD" \
  --object-threshold "$OBJECT_THRESHOLD" \
  --crop-mode "$CROP_MODE" \
  --live-object-tokens "$LIVE_OBJECT_TOKENS" \
  >>"$LOG_FILE" 2>&1 < /dev/null &

pid="$!"
echo "$pid" > "$PID_FILE"

echo "dashboard pid: $pid"
echo "dashboard log: $LOG_FILE"
echo "dashboard url: http://localhost:$PORT"
