#!/usr/bin/env python3
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Export the live actor checkpoint and RF-DETR Nano to TensorRT, "
            "verify ONNX/TensorRT drift, and optionally launch the live dashboard."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", default="object_actor_live/exports/live_tensorrt")
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--trtexec", default=None)
    parser.add_argument("--workspace-mib", type=int, default=2048)
    parser.add_argument("--max-aux-streams", type=int, default=0)
    parser.add_argument("--builder-optimization-level", type=int, default=None)
    parser.add_argument("--no-tf32", action="store_true")
    parser.add_argument("--mask-input-dtype", choices=["bool", "int32"], default="bool")
    parser.add_argument("--disable-token-pruning", action="store_true")
    parser.add_argument("--trt-safe-attention", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--rfdetr-model-size", default="nano")
    parser.add_argument(
        "--rfdetr-shape",
        type=int,
        default=0,
        help="Static RF-DETR square input size. 0 keeps the RF-DETR exporter default.",
    )
    parser.add_argument("--actor-drift-tolerance", type=float, default=None)
    parser.add_argument("--rfdetr-drift-tolerance", type=float, default=None)
    parser.add_argument("--launch-dashboard", action="store_true")
    parser.add_argument("--camera", default="0")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--det-threshold", type=float, default=0.35)
    parser.add_argument("--object-threshold", type=float, default=0.25)
    parser.add_argument(
        "--crop-mode",
        choices=("actor", "actor_window", "center"),
        default="actor_window",
    )
    parser.add_argument("--live-object-tokens", type=int, choices=(0, 1), default=1)
    return parser.parse_args()


def resolve_trtexec(path):
    if path:
        trtexec = Path(path)
        if not trtexec.is_file():
            raise FileNotFoundError(f"trtexec not found: {trtexec}")
        return str(trtexec)
    jetson_path = Path("/usr/src/tensorrt/bin/trtexec")
    if jetson_path.is_file():
        return str(jetson_path)
    found = shutil.which("trtexec")
    if found:
        return found
    raise FileNotFoundError("trtexec not found. Pass --trtexec /path/to/trtexec.")


def run(command):
    print("\n" + "=" * 100, flush=True)
    print("$ " + " ".join(str(part) for part in command), flush=True)
    print("=" * 100, flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def tolerance(value, precision, fp32_value, fp16_value):
    if value is not None:
        return float(value)
    return fp32_value if precision == "fp32" else fp16_value


def main():
    args = parse_args()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    trtexec = resolve_trtexec(args.trtexec)
    out_dir = (REPO_ROOT / args.out_dir).resolve()
    actor_dir = out_dir / "actor"
    rfdetr_dir = out_dir / "rfdetr_nano"
    actor_dir.mkdir(parents=True, exist_ok=True)
    rfdetr_dir.mkdir(parents=True, exist_ok=True)

    actor_onnx = actor_dir / "actor.onnx"
    actor_engine = actor_dir / f"actor_{args.precision}.engine"
    actor_check = actor_engine.with_suffix(actor_engine.suffix + ".check.json")
    rfdetr_onnx = rfdetr_dir / "inference_model.onnx"
    rfdetr_engine = rfdetr_dir / f"inference_model_{args.precision}.engine"
    rfdetr_check = rfdetr_engine.with_suffix(rfdetr_engine.suffix + ".check.json")
    manifest = out_dir / f"live_tensorrt_{args.precision}.json"

    started = time.time()
    actor_export = [
        sys.executable,
        "utils/export_actor_tensorrt.py",
        "--checkpoint",
        str(checkpoint),
        "--onnx-out",
        str(actor_onnx),
        "--engine-out",
        str(actor_engine),
        "--precision",
        args.precision,
        "--trtexec",
        trtexec,
        "--workspace-mib",
        str(args.workspace_mib),
        "--max-aux-streams",
        str(args.max_aux_streams),
        "--mask-input-dtype",
        args.mask_input_dtype,
        "--export-device",
        "cpu",
    ]
    if args.builder_optimization_level is not None:
        actor_export.extend(
            [
                "--builder-optimization-level",
                str(args.builder_optimization_level),
            ]
        )
    if args.no_tf32:
        actor_export.append("--no-tf32")
    if args.disable_token_pruning:
        actor_export.append("--disable-token-pruning")
    if args.trt_safe_attention:
        actor_export.append("--trt-safe-attention")
    if args.force:
        actor_export.append("--force")
    if args.benchmark:
        actor_export.append("--benchmark")
    run(actor_export)

    actor_tol = tolerance(args.actor_drift_tolerance, args.precision, 1e-3, 8e-2)
    actor_check_command = [
        sys.executable,
        "utils/check_actor_tensorrt.py",
        "--checkpoint",
        str(checkpoint),
        "--onnx",
        str(actor_onnx),
        "--engine",
        str(actor_engine),
        "--out-json",
        str(actor_check),
        "--max-abs-tolerance",
        str(actor_tol),
    ]
    if args.disable_token_pruning:
        actor_check_command.append("--disable-token-pruning")
    if args.trt_safe_attention:
        actor_check_command.append("--trt-safe-attention")
    run(actor_check_command)

    rfdetr_export = [
        sys.executable,
        "utils/export_rfdetr_tensorrt.py",
        "--model-size",
        str(args.rfdetr_model_size),
        "--out-dir",
        str(rfdetr_dir),
        "--precision",
        args.precision,
        "--trtexec",
        trtexec,
        "--workspace-mib",
        str(args.workspace_mib),
        "--max-aux-streams",
        str(args.max_aux_streams),
    ]
    if args.rfdetr_shape:
        rfdetr_export.extend(["--shape", str(args.rfdetr_shape)])
    if args.builder_optimization_level is not None:
        rfdetr_export.extend(
            [
                "--builder-optimization-level",
                str(args.builder_optimization_level),
            ]
        )
    if args.no_tf32:
        rfdetr_export.append("--no-tf32")
    if args.force:
        rfdetr_export.append("--force")
    if args.benchmark:
        rfdetr_export.append("--benchmark")
    run(rfdetr_export)

    rfdetr_tol = tolerance(args.rfdetr_drift_tolerance, args.precision, 1e-3, 1e-1)
    run(
        [
            sys.executable,
            "utils/check_rfdetr_tensorrt.py",
            "--onnx",
            str(rfdetr_onnx),
            "--engine",
            str(rfdetr_engine),
            "--out-json",
            str(rfdetr_check),
            "--max-abs-tolerance",
            str(rfdetr_tol),
        ]
    )

    dashboard_command = [
        sys.executable,
        "-u",
        "live_actor_dashboard.py",
        "--checkpoint",
        str(checkpoint),
        "--actor-engine",
        str(actor_engine),
        "--detector-engine",
        str(rfdetr_engine),
        "--camera",
        str(args.camera),
        "--host",
        str(args.host),
        "--port",
        str(args.port),
        "--det-threshold",
        str(args.det_threshold),
        "--object-threshold",
        str(args.object_threshold),
        "--crop-mode",
        str(args.crop_mode),
        "--live-object-tokens",
        str(args.live_object_tokens),
    ]

    payload = {
        "checkpoint": str(checkpoint),
        "precision": args.precision,
        "no_tf32": bool(args.no_tf32),
        "disable_token_pruning": bool(args.disable_token_pruning),
        "trt_safe_attention": bool(args.trt_safe_attention),
        "mask_input_dtype": args.mask_input_dtype,
        "actor_onnx": str(actor_onnx),
        "actor_engine": str(actor_engine),
        "actor_check": str(actor_check),
        "rfdetr_onnx": str(rfdetr_onnx),
        "rfdetr_engine": str(rfdetr_engine),
        "rfdetr_check": str(rfdetr_check),
        "trtexec": trtexec,
        "dashboard_command": dashboard_command,
        "elapsed_sec": round(time.time() - started, 3),
    }
    manifest.write_text(json.dumps(payload, indent=2) + "\n")
    print("\nTensorRT live export complete.", flush=True)
    print(f"Manifest: {manifest}", flush=True)
    print(f"Actor engine: {actor_engine}", flush=True)
    print(f"RF-DETR engine: {rfdetr_engine}", flush=True)
    print("Dashboard command:", " ".join(dashboard_command), flush=True)

    if args.launch_dashboard:
        run(dashboard_command)


if __name__ == "__main__":
    main()
