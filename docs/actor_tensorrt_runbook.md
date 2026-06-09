# Actor TensorRT Runbook

This runbook documents the current actor TensorRT path for the object-token
PO-GUISE actor checkpoint.

## Known-good actor engine

Use this actor engine for live tests:

```bash
object_actor_live/exports/epoch017_actor_fp16_safeattn_prune/epoch=017_b1_t16_k8_o32_224_fp16.engine
```

It was exported from:

```bash
epoch=017.ckpt
```

with:

- TensorRT FP16
- token pruning kept enabled from the checkpoint
- `trt_safe_attention=1`
- static shape `B=1, T=16, K actors=8, O objects=32, input=224`

The matching RF-DETR detector engine is:

```bash
object_actor_live/exports/rfdetr_nano/inference_model_fp16.engine
```

## Why safe attention is required

The original full TensorRT actor engine built successfully but was numerically
wrong. Bisection showed:

| Stage | Export mode | TensorRT max drift |
| --- | --- | ---: |
| prefix/object-token path | pruning disabled, unsafe attention | `3.81e-06` |
| block0 | pruning disabled, unsafe attention | `11.96` |
| block0 | pruning disabled, safe attention | `5.39e-05` |
| full actor | pruning enabled, safe attention, FP32 | `1.29e-05` |
| full actor | pruning enabled, safe attention, FP16 | `0.0463` |

So the first bad point is the first transformer attention block, not object
tokens, not heads, and not detector inputs.

The safe path exports attention as explicit:

```text
matmul -> mask -> softmax -> matmul
```

and avoids TensorRT's broken fused attention execution for this graph.

## Latency results

Measured with `trtexec --useCudaGraph` on the Orin:

| Engine | Drift | Mean latency |
| --- | ---: | ---: |
| FP32, safe attention, no pruning | `5.25e-06` | `~867 ms` |
| FP32, safe attention, pruning enabled | `1.29e-05` | `~572 ms` |
| FP16, safe attention, pruning enabled | `0.0463` | `~146 ms` |

The FP16 safe-attention engine is the practical live candidate. The FP32 safe
engines are correct but too slow.

## Build, check, and benchmark actor TensorRT

Run:

```bash
object_actor_live/build_epoch017_fp16_safe_actor_trt.sh
```

This script:

1. exports ONNX with `--trt-safe-attention`
2. builds the FP16 TensorRT actor engine
3. checks TensorRT drift against PyTorch
4. benchmarks the engine with `trtexec`

The drift report is written next to the engine:

```bash
object_actor_live/exports/epoch017_actor_fp16_safeattn_prune/epoch=017_b1_t16_k8_o32_224_fp16.engine.check.json
```

Do not use an actor TensorRT engine unless it has a passing drift report.

## Run live dashboard

Run:

```bash
object_actor_live/run_epoch017_fp16_trt_dashboard.sh
```

Default URL:

```text
http://localhost:7860
```

The script writes:

```bash
object_actor_live/logs/epoch017_fp16_trt_dashboard.log
object_actor_live/logs/epoch017_fp16_trt_dashboard.pid
```

Optional environment overrides:

```bash
PORT=7861 CAMERA=0 DET_THRESHOLD=0.35 OBJECT_THRESHOLD=0.25 \
  object_actor_live/run_epoch017_fp16_trt_dashboard.sh
```

Use `LIVE_OBJECT_TOKENS=0` only as a diagnostic to isolate actor-only behavior:

```bash
LIVE_OBJECT_TOKENS=0 object_actor_live/run_epoch017_fp16_trt_dashboard.sh
```

## Manual commands

Export/build the current known-good actor engine manually:

```bash
/home/michel/miniconda3/envs/voice_id/bin/python utils/export_actor_tensorrt.py \
  --checkpoint epoch=017.ckpt \
  --out-dir object_actor_live/exports/epoch017_actor_fp16_safeattn_prune \
  --precision fp16 \
  --workspace-mib 512 \
  --max-aux-streams 0 \
  --builder-optimization-level 0 \
  --mask-input-dtype bool \
  --trt-safe-attention \
  --force
```

Check drift:

```bash
/home/michel/miniconda3/envs/voice_id/bin/python utils/check_actor_tensorrt.py \
  --checkpoint epoch=017.ckpt \
  --onnx object_actor_live/exports/epoch017_actor_fp16_safeattn_prune/epoch=017_b1_t16_k8_o32_224.onnx \
  --engine object_actor_live/exports/epoch017_actor_fp16_safeattn_prune/epoch=017_b1_t16_k8_o32_224_fp16.engine \
  --trt-safe-attention \
  --out-json object_actor_live/exports/epoch017_actor_fp16_safeattn_prune/epoch=017_b1_t16_k8_o32_224_fp16.engine.check.json \
  --max-abs-tolerance 0.08
```

Benchmark:

```bash
/usr/src/tensorrt/bin/trtexec \
  --loadEngine=object_actor_live/exports/epoch017_actor_fp16_safeattn_prune/epoch=017_b1_t16_k8_o32_224_fp16.engine \
  --duration=5 \
  --warmUp=500 \
  --iterations=50 \
  --useCudaGraph
```

## Do not use

Do not use these for live behavior:

- old full actor TensorRT engines without `trt_safe_attention`
- FP32 safe-attention engines unless you only need a correctness diagnostic
- no-prune engines for live, because they are correct but too slow
- any engine that has not passed `utils/check_actor_tensorrt.py`

## Next validation gate

The dummy drift check proves the engine is not semantically broken like the old
TensorRT export. It does not prove every action label remains stable.

Before trusting live behavior, run saved-video PyTorch-vs-TensorRT parity on
the same videos/classes that matter for deployment.
