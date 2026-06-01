# Object Actor Live Deployment

This folder is for the clean Actor-Slot PO-GUISE+ object-interaction model.

Use this path for checkpoints trained with:

- `object_prompt=1`
- RF-DETR object candidates at inference
- actor slots
- transformer object tokens

Do not use the old `live_actor_dashboard.py` for object-interaction checkpoints. The old dashboard only feeds person boxes. This dashboard feeds:

- `video`
- `boxes`
- `valid`
- `object_boxes`
- `object_cls`
- `object_conf`
- `object_valid`

## 1. Activate Environment

```bash
cd /home/michel/Documents/poguise
conda activate voice_id
```

## 2. Put the Checkpoint in Place

Example for the epoch 4 checkpoint:

```bash
mkdir -p checkpoints/object_actor
ls -lh checkpoints/object_actor/epoch=004.ckpt
```

If your uploaded file has another name, pass that path to `--checkpoint`.

## 3. Export Actor ONNX / TensorRT

```bash
python object_actor_live/export_object_actor_tensorrt.py \
  --checkpoint checkpoints/object_actor/epoch=004.ckpt \
  --out-dir object_actor_live/exports/epoch004 \
  --precision fp16 \
  --workspace-mib 1024 \
  --benchmark \
  --force
```

This script delegates to `utils/export_actor_tensorrt.py`, verifies that the checkpoint is an object-prompt checkpoint, builds a fixed-shape ONNX file, builds a TensorRT engine with `trtexec`, and smoke-tests the object inputs.

For object-token checkpoints, validate TensorRT numerics against ONNX/PyTorch before trusting the engine. On Orin, FP16 TensorRT may build successfully while changing action logits for this architecture. If the TensorRT actor engine is not numerically faithful, run the actor through ONNX Runtime CUDA with `--onnx` and keep RF-DETR on TensorRT.

## 4. Export RF-DETR TensorRT

The dashboard can run RF-DETR through PyTorch or direct TensorRT. The TensorRT path follows RF-DETR's official ONNX export flow and NVIDIA's `trtexec --onnx ... --saveEngine ...` engine build flow.

```bash
python object_actor_live/export_rfdetr_tensorrt.py \
  --model-size nano \
  --out-dir object_actor_live/exports/rfdetr_nano \
  --precision fp16 \
  --workspace-mib 1024 \
  --benchmark \
  --force
```

This writes:

```text
object_actor_live/exports/rfdetr_nano/inference_model.onnx
object_actor_live/exports/rfdetr_nano/inference_model_fp16.engine
```

## 5. Run Live Dashboard

Use the actor backend that passed numeric validation. ONNX Runtime CUDA is the safer actor backend for object-token checkpoints:

```bash
python object_actor_live/live_object_actor_dashboard.py \
  --onnx object_actor_live/exports/epoch004/epoch=004_b1_t16_k8_m24_224.onnx \
  --detector-backend tensorrt \
  --detector-engine object_actor_live/exports/rfdetr_nano/inference_model_fp16.engine \
  --camera 0 \
  --host 0.0.0.0 \
  --port 7861 \
  --detector-model-size nano \
  --detector-device cuda \
  --detector-optimize 1 \
  --detector-compile 1 \
  --person-threshold 0.35 \
  --object-threshold book=0.35,laptop=0.35,phone=0.45,tv_monitor=0.45,remote=0.35,cup=0.35,bottle=0.35,glass=0.35 \
  --detect-every 3 \
  --action-every 5
```

Use `--engine ...fp16.engine` only after confirming that TensorRT matches ONNX/PyTorch on saved live inputs.

Open the printed LAN URL from another device on the same network.

For a smoke test without a camera:

```bash
python object_actor_live/live_object_actor_dashboard.py \
  --onnx object_actor_live/exports/epoch004/epoch=004_b1_t16_k8_m24_224.onnx \
  --smoke
```

## 6. Live Tensor Sensitivity Test

Before judging a new object-token checkpoint by the dashboard, test the exact
saved live tensor in PyTorch. This avoids TensorRT/ONNX/runtime differences and
answers whether the checkpoint actually changes action logits when the laptop
object is removed, isolated, moved, or relabeled.

Use an input written by the dashboard with `--debug-save-latest-input`:

```bash
python object_actor_live/analyze_live_object_sensitivity.py \
  --checkpoint checkpoints/object_actor/epoch=000.ckpt \
  --input-pt object_actor_live/latest_epoch001_actor_input.pt \
  --device cuda \
  --dtype auto
```

For low-memory Orin runs:

```bash
python object_actor_live/analyze_live_object_sensitivity.py \
  --checkpoint checkpoints/object_actor/epoch=000.ckpt \
  --input-pt object_actor_live/latest_epoch001_actor_input.pt \
  --device cuda \
  --dtype auto \
  --skip-gradient
```

The important rows are `objects_on`, `objects_off`, `laptop_only`,
`positive_erased_laptop`, `laptop_class_changed_to_book`, and
`laptop_box_moved_away`. A useful checkpoint should move the `Uselaptop` logit
when those object facts change. If the script says the tensor contains no
laptop/keyboard object, save a new live tensor while RF-DETR is actually drawing
the laptop.

## Notes

- TensorRT engines are hardware/version specific. Build the engine on the Orin Nano that will run it.
- Keep `batch_size=1`, `clip_frames=16`, `max_actors=8`, and `max_objects=24` unless the checkpoint was trained differently.
- The live dashboard uses RF-DETR detections from the current video stream. It does not use the Toyota JSONL cache.
- `--debug-save-latest-input path.pt` writes the exact actor tensors and object packing from the latest action step. Use it to compare PyTorch, ONNX Runtime, and TensorRT on the same live input.
- If the live camera is slow, increase `--detect-every`. If object behavior looks stale, lower `--detect-every`.
