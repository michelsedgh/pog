#!/usr/bin/env python3
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export RF-DETR to ONNX and TensorRT for teacher-cache generation or visualization."
    )
    parser.add_argument(
        "--model-size",
        default="nano",
        choices=["nano", "small", "medium", "base", "large", "xlarge", "2xlarge"],
    )
    parser.add_argument("--weights", default=None)
    parser.add_argument("--out-dir", default="exports/rfdetr_nano")
    parser.add_argument(
        "--shape",
        type=int,
        default=0,
        help="Static square input size. 0 keeps the RF-DETR default.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--trtexec", default=None)
    parser.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--workspace-mib", type=int, default=1024)
    parser.add_argument("--max-aux-streams", type=int, default=0)
    parser.add_argument("--builder-optimization-level", type=int, default=None)
    parser.add_argument("--no-tf32", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--force", action="store_true")
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


def load_model(model_size, weights):
    from rfdetr import RFDETRBase, RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall

    model_classes = {
        "nano": RFDETRNano,
        "small": RFDETRSmall,
        "medium": RFDETRMedium,
        "base": RFDETRBase,
        "large": RFDETRLarge,
    }
    if model_size in {"xlarge", "2xlarge"}:
        try:
            from rfdetr_plus import RFDETR2XLarge, RFDETRXLarge
        except ImportError as exc:
            raise RuntimeError(f"RF-DETR {model_size} requires rfdetr_plus.") from exc
        model_classes.update({"xlarge": RFDETRXLarge, "2xlarge": RFDETR2XLarge})

    kwargs = {}
    if weights:
        kwargs["pretrain_weights"] = weights
    return model_classes[model_size](**kwargs)


def export_onnx(args, out_dir):
    model = load_model(args.model_size, args.weights)
    export_kwargs = {
        "output_dir": str(out_dir),
        "batch_size": int(args.batch_size),
        "opset_version": int(args.opset),
    }
    if args.shape:
        export_kwargs["shape"] = (int(args.shape), int(args.shape))
    print("RF-DETR export kwargs:", export_kwargs, flush=True)
    model.export(**export_kwargs)

    onnx_path = out_dir / "inference_model.onnx"
    if not onnx_path.is_file():
        candidates = sorted(out_dir.glob("*.onnx"), key=lambda p: p.stat().st_mtime)
        if not candidates:
            raise FileNotFoundError(f"No ONNX file produced under {out_dir}")
        onnx_path = candidates[-1]
    return onnx_path


def run_command(command):
    print("$ " + " ".join(str(part) for part in command), flush=True)
    subprocess.run(command, check=True)


def build_engine(trtexec, onnx_path, engine_path, args):
    command = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--memPoolSize=workspace:{int(args.workspace_mib)}M",
        f"--maxAuxStreams={int(args.max_aux_streams)}",
        "--skipInference",
    ]
    if args.precision == "fp16":
        command.append("--fp16")
    if args.precision == "fp32" and args.no_tf32:
        command.append("--noTF32")
    if args.builder_optimization_level is not None:
        command.append(f"--builderOptimizationLevel={int(args.builder_optimization_level)}")
    run_command(command)
    return command


def benchmark_engine(trtexec, engine_path, args):
    command = [
        trtexec,
        f"--loadEngine={engine_path}",
        "--warmUp=500",
        "--duration=10",
        "--avgRuns=20",
        f"--maxAuxStreams={int(args.max_aux_streams)}",
    ]
    run_command(command)
    return command


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "inference_model.onnx"
    engine_path = out_dir / f"inference_model_{args.precision}.engine"
    metadata_path = engine_path.with_suffix(engine_path.suffix + ".json")
    if not args.force:
        for path in [onnx_path, engine_path, metadata_path]:
            if path.exists():
                raise FileExistsError(f"{path} exists. Pass --force to overwrite it.")

    trtexec = resolve_trtexec(args.trtexec)
    started = time.time()
    onnx_path = export_onnx(args, out_dir)
    build_command = build_engine(trtexec, onnx_path, engine_path, args)
    benchmark_command = benchmark_engine(trtexec, engine_path, args) if args.benchmark else None

    metadata = {
        "model_size": args.model_size,
        "weights": args.weights,
        "onnx": str(onnx_path),
        "engine": str(engine_path),
        "precision": args.precision,
        "no_tf32": bool(args.no_tf32),
        "batch_size": int(args.batch_size),
        "shape": int(args.shape) if args.shape else None,
        "trtexec": trtexec,
        "build_command": build_command,
        "benchmark_command": benchmark_command,
        "elapsed_sec": round(time.time() - started, 3),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print("\nRF-DETR TensorRT export complete.")
    print("ONNX:", onnx_path)
    print("Engine:", engine_path)
    print("Metadata:", metadata_path)


if __name__ == "__main__":
    main()
