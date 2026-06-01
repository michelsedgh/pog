#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.actor_tensorrt import TensorRTActorEngine
from utils.export_actor_tensorrt import checkpoint_payload


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export an object-prompt Actor-Slot PO-GUISE+ checkpoint to TensorRT."
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/object_actor/epoch=004.ckpt",
        help="Object-interaction checkpoint path.",
    )
    parser.add_argument("--out-dir", default="object_actor_live/exports")
    parser.add_argument("--onnx-out", default=None)
    parser.add_argument("--engine-out", default=None)
    parser.add_argument("--trtexec", default=None)
    parser.add_argument("--export-device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--workspace-mib", type=int, default=1024)
    parser.add_argument("--max-aux-streams", type=int, default=0)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--smoke", action="store_true", default=True)
    parser.add_argument("--no-smoke", dest="smoke", action="store_false")
    parser.add_argument("--numeric-atol", type=float, default=0.5)
    return parser.parse_args()


def run_command(command):
    print("$ " + " ".join(str(part) for part in command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def inspect_object_checkpoint(checkpoint):
    payload = checkpoint_payload(checkpoint)
    hparams = payload["hparams"]
    if not int(hparams.get("object_prompt", 0)):
        raise RuntimeError(
            "This checkpoint is not an object-prompt checkpoint. "
            "Use the old actor export path for actor-only checkpoints."
        )
    if int(hparams.get("num_object_tokens", 0)) <= 0:
        raise RuntimeError("Object-prompt checkpoint has no object tokens.")
    if int(hparams.get("num_object_classes", 0)) != 19:
        raise RuntimeError(
            "Expected 19 object classes for the clean object-interaction model, "
            f"got {hparams.get('num_object_classes')}."
        )
    return payload


def _smoke_inputs(engine):
    import torch

    generator = torch.Generator(device="cpu")
    generator.manual_seed(1234)
    video = torch.randn(engine.shapes["video"], generator=generator).to(
        device="cuda",
        dtype=torch.float32,
    )
    boxes = torch.zeros(engine.shapes["boxes"], dtype=torch.float32, device="cuda")
    valid = torch.zeros(engine.shapes["valid"], dtype=torch.bool, device="cuda")
    boxes[0, 0] = torch.tensor([0.20, 0.12, 0.82, 0.95], device="cuda")
    valid[0, 0] = True

    object_boxes = torch.zeros(
        engine.shapes["object_boxes"], dtype=torch.float32, device="cuda"
    )
    object_cls = torch.full(
        engine.shapes["object_cls"], 19, dtype=torch.long, device="cuda"
    )
    object_conf = torch.zeros(
        engine.shapes["object_conf"], dtype=torch.float32, device="cuda"
    )
    object_valid = torch.zeros(
        engine.shapes["object_valid"], dtype=torch.bool, device="cuda"
    )
    object_boxes[0, 0] = torch.tensor([0.30, 0.45, 0.72, 0.80], device="cuda")
    object_cls[0, 0] = 1
    object_conf[0, 0] = 0.90
    object_valid[0, 0] = True
    object_boxes[0, 1] = torch.tensor([0.38, 0.38, 0.58, 0.58], device="cuda")
    object_cls[0, 1] = 0
    object_conf[0, 1] = 0.72
    object_valid[0, 1] = True
    return video, boxes, valid, object_boxes, object_cls, object_conf, object_valid


def smoke_engine(engine_path, onnx_path, numeric_atol):
    import numpy as np
    import onnxruntime as ort
    import torch

    engine = TensorRTActorEngine(engine_path)
    if not engine.object_prompt:
        raise RuntimeError("Exported TensorRT engine does not expose object inputs.")

    (
        video,
        boxes,
        valid,
        object_boxes,
        object_cls,
        object_conf,
        object_valid,
    ) = _smoke_inputs(engine)

    logits, presence = engine(
        video,
        boxes,
        valid,
        object_boxes=object_boxes,
        object_cls=object_cls,
        object_conf=object_conf,
        object_valid=object_valid,
    )
    print(
        "TensorRT object engine smoke ok:",
        {
            "engine": str(engine_path),
            "inputs": engine.input_names,
            "logits": tuple(logits.shape),
            "presence": tuple(presence.shape),
        },
        flush=True,
    )

    feed = {
        "video": video.detach().cpu().numpy().astype(np.float32, copy=False),
        "boxes": boxes.detach().cpu().numpy().astype(np.float32, copy=False),
        "valid": valid.detach().cpu().numpy().astype(np.bool_, copy=False),
        "object_boxes": object_boxes.detach().cpu().numpy().astype(np.float32, copy=False),
        "object_cls": object_cls.detach().cpu().numpy().astype(np.int32, copy=False),
        "object_conf": object_conf.detach().cpu().numpy().astype(np.float32, copy=False),
        "object_valid": object_valid.detach().cpu().numpy().astype(np.bool_, copy=False),
    }
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ref_logits, ref_presence = session.run(["logits", "presence"], feed)
    ref_logits = torch.from_numpy(ref_logits)
    ref_presence = torch.from_numpy(ref_presence)
    logit_diff = (logits.detach().cpu().float() - ref_logits.float()).abs().max().item()
    presence_diff = (
        presence.detach().cpu().float() - ref_presence.float()
    ).abs().max().item()
    engine_top1 = logits.detach().cpu().float().argmax(dim=-1)
    ref_top1 = ref_logits.float().argmax(dim=-1)
    top1_match = bool(torch.equal(engine_top1, ref_top1))
    print(
        "TensorRT numeric check:",
        {
            "max_logit_abs_diff": round(logit_diff, 6),
            "max_presence_abs_diff": round(presence_diff, 6),
            "top1_match": top1_match,
            "atol": float(numeric_atol),
        },
        flush=True,
    )
    if logit_diff > float(numeric_atol) or not top1_match:
        raise RuntimeError(
            "TensorRT actor engine is not numerically faithful to ONNX. "
            f"max_logit_abs_diff={logit_diff:.6f}, top1_match={top1_match}. "
            "Use the dashboard --onnx actor backend or rebuild with a safer TensorRT config."
        )


def main():
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    payload = inspect_object_checkpoint(str(checkpoint))
    hparams = payload["hparams"]
    print("Object checkpoint:", json.dumps(hparams, indent=2), flush=True)

    command = [
        sys.executable,
        "utils/export_actor_tensorrt.py",
        "--checkpoint",
        str(checkpoint),
        "--out-dir",
        args.out_dir,
        "--batch-size",
        "1",
        "--clip-frames",
        str(int(hparams.get("n_frames", 16))),
        "--input-size",
        "224",
        "--max-actors",
        str(int(hparams.get("num_actor_tokens", 8))),
        "--max-objects",
        str(int(hparams.get("num_object_tokens", 24))),
        "--opset",
        str(args.opset),
        "--precision",
        args.precision,
        "--workspace-mib",
        str(args.workspace_mib),
        "--max-aux-streams",
        str(args.max_aux_streams),
        "--export-device",
        args.export_device,
    ]
    if args.onnx_out:
        command.extend(["--onnx-out", args.onnx_out])
    if args.engine_out:
        command.extend(["--engine-out", args.engine_out])
    if args.trtexec:
        command.extend(["--trtexec", args.trtexec])
    if args.benchmark:
        command.append("--benchmark")
    if args.force:
        command.append("--force")

    run_command(command)

    engine_path = None
    if args.engine_out:
        engine_path = Path(args.engine_out)
    else:
        metadata_files = sorted(
            Path(args.out_dir).glob("*.engine.json"),
            key=lambda path: path.stat().st_mtime,
        )
        if metadata_files:
            metadata = json.loads(metadata_files[-1].read_text())
            engine_path = Path(metadata["engine"])

    if engine_path is None or not engine_path.is_file():
        raise RuntimeError("Could not resolve exported engine path.")
    if args.smoke:
        smoke_engine(engine_path, onnx_path, args.numeric_atol)

    print("\nExport complete.")
    print("Engine:", engine_path)
    print(
        "Dashboard command:",
        "python object_actor_live/live_object_actor_dashboard.py "
        f"--engine {engine_path} --camera 0 --port 7861",
    )


if __name__ == "__main__":
    main()
