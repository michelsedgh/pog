# Object Actor Live Deployment

This folder is for the clean Actor-Slot PO-GUISE+ object-interaction model.

Use this path for checkpoints trained with:

- `object_prompt=1`
- RF-DETR object candidates at inference
- actor slots
- object feature fusion

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

## 3. Export TensorRT

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

## 4. Run Live Dashboard

Use the engine printed by the export step:

```bash
python object_actor_live/live_object_actor_dashboard.py \
  --engine object_actor_live/exports/epoch004/epoch=004_b1_t16_k8_m24_224_fp16.engine \
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

Open the printed LAN URL from another device on the same network.

For a smoke test without a camera:

```bash
python object_actor_live/live_object_actor_dashboard.py \
  --engine object_actor_live/exports/epoch004/epoch=004_b1_t16_k8_m24_224_fp16.engine \
  --smoke
```

## Notes

- TensorRT engines are hardware/version specific. Build the engine on the Orin Nano that will run it.
- Keep `batch_size=1`, `clip_frames=16`, `max_actors=8`, and `max_objects=24` unless the checkpoint was trained differently.
- The live dashboard uses RF-DETR detections from the current video stream. It does not use the Toyota JSONL cache.
- If the live camera is slow, increase `--detect-every`. If object behavior looks stale, lower `--detect-every`.
