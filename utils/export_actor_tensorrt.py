#!/usr/bin/env python3
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a PO-GUISE actor checkpoint to fixed-shape ONNX and TensorRT."
    )
    parser.add_argument("--checkpoint", required=True, help="Actor checkpoint .ckpt path.")
    parser.add_argument("--out-dir", default="exports", help="Output directory.")
    parser.add_argument("--onnx-out", default=None, help="ONNX output path.")
    parser.add_argument("--engine-out", default=None, help="TensorRT engine output path.")
    parser.add_argument("--trtexec", default=None, help="Path to trtexec.")
    parser.add_argument("--export-device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--clip-frames", type=int, default=None)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--max-actors", type=int, default=None)
    parser.add_argument("--max-objects", type=int, default=None)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--mask-input-dtype", choices=["bool", "int32"], default="bool")
    parser.add_argument(
        "--disable-token-pruning",
        action="store_true",
        help=(
            "Export a TensorRT-friendlier actor graph by forcing keep_rate=1.0 "
            "and keep_rate_merge=1.0. This changes the inference graph and must "
            "be drift-checked before live use."
        ),
    )
    parser.add_argument(
        "--trt-safe-attention",
        action="store_true",
        help=(
            "Export attention as explicit matmul/mask/softmax/matmul and avoid "
            "tensor .any() branches in attention. Must be drift-checked before live use."
        ),
    )
    parser.add_argument("--workspace-mib", type=int, default=1024)
    parser.add_argument("--max-aux-streams", type=int, default=0)
    parser.add_argument("--builder-optimization-level", type=int, default=None)
    parser.add_argument("--no-tf32", action="store_true")
    parser.add_argument("--force", action="store_true", help="Overwrite output files.")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--warmup-ms", type=int, default=500)
    parser.add_argument("--duration-sec", type=int, default=10)
    parser.add_argument("--avg-runs", type=int, default=20)
    parser.add_argument("--internal-mode", choices=["inspect", "export"], default=None, help=argparse.SUPPRESS)
    parser.add_argument("--internal-json-out", default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def resolve_trtexec(path):
    if path:
        resolved = Path(path)
        if not resolved.is_file():
            raise FileNotFoundError(f"trtexec not found: {resolved}")
        return str(resolved)

    jetson_path = Path("/usr/src/tensorrt/bin/trtexec")
    if jetson_path.is_file():
        return str(jetson_path)

    found = shutil.which("trtexec")
    if found:
        return found
    raise FileNotFoundError("trtexec not found. Pass --trtexec /path/to/trtexec.")


def default_output_paths(args, hparams):
    checkpoint = Path(args.checkpoint)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = checkpoint.name
    for suffix in [".ckpt", ".pth", ".pt"]:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break

    clip_frames = int(args.clip_frames or hparams.get("n_frames", 16))
    max_actors = int(args.max_actors or hparams.get("num_actor_tokens", 0))
    max_objects = int(
        args.max_objects
        if args.max_objects is not None
        else hparams.get("num_scene_object_tokens", 0)
    )
    object_suffix = f"_o{max_objects}" if int(hparams.get("scene_object_tokens", 0)) else ""
    fixed = (
        f"{stem}_b{args.batch_size}_t{clip_frames}_k{max_actors}"
        f"{object_suffix}_{args.input_size}"
    )
    onnx_out = Path(args.onnx_out) if args.onnx_out else out_dir / f"{fixed}.onnx"
    engine_out = (
        Path(args.engine_out)
        if args.engine_out
        else out_dir / f"{fixed}_{args.precision}.engine"
    )
    metadata_out = engine_out.with_suffix(engine_out.suffix + ".json")
    return onnx_out, engine_out, metadata_out, clip_frames, max_actors


def require_writable(path, force):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        raise FileExistsError(f"{path} exists. Pass --force to overwrite it.")


def make_dummy_inputs(
    batch_size,
    clip_frames,
    input_size,
    max_actors,
    max_objects,
    num_object_classes,
    scene_object_tokens,
    device,
    mask_input_dtype="bool",
):
    import torch

    video = torch.zeros(
        (batch_size, clip_frames, 3, input_size, input_size),
        dtype=torch.float32,
        device=device,
    )
    boxes = torch.zeros((batch_size, max_actors, 4), dtype=torch.float32, device=device)
    boxes[:, 0] = torch.tensor(
        [0.25, 0.15, 0.75, 0.95],
        dtype=torch.float32,
        device=device,
    )
    mask_dtype = torch.bool if mask_input_dtype == "bool" else torch.int32
    valid = torch.zeros((batch_size, max_actors), dtype=mask_dtype, device=device)
    valid[:, 0] = 1
    if not scene_object_tokens:
        return (video, boxes, valid), ["video", "boxes", "valid"]

    object_boxes = torch.zeros(
        (batch_size, max_objects, 4),
        dtype=torch.float32,
        device=device,
    )
    object_classes = torch.full(
        (batch_size, max_objects),
        int(num_object_classes),
        dtype=torch.int32,
        device=device,
    )
    object_confs = torch.zeros(
        (batch_size, max_objects),
        dtype=torch.float32,
        device=device,
    )
    object_valid = torch.zeros(
        (batch_size, max_objects),
        dtype=mask_dtype,
        device=device,
    )
    if max_objects > 0:
        object_boxes[:, 0] = torch.tensor(
            [0.30, 0.35, 0.70, 0.75],
            dtype=torch.float32,
            device=device,
        )
        object_classes[:, 0] = 1
        object_confs[:, 0] = 0.95
        object_valid[:, 0] = 1
    return (
        video,
        boxes,
        valid,
        object_boxes,
        object_classes,
        object_confs,
        object_valid,
    ), [
        "video",
        "boxes",
        "valid",
        "object_boxes",
        "object_classes",
        "object_confs",
        "object_valid",
    ]


def run_command(command):
    print("$ " + " ".join(str(part) for part in command), flush=True)
    subprocess.run(command, check=True)


def export_onnx(model, onnx_out, args, clip_frames, max_actors, max_objects, hparams, device):
    import torch

    scene_object_tokens = bool(hparams.get("scene_object_tokens", 0))
    num_object_classes = int(hparams.get("num_object_classes", 19))

    class ActorExport(torch.nn.Module):
        def __init__(self, actor_model):
            super().__init__()
            self.actor_model = actor_model

        def forward(
            self,
            video,
            boxes,
            valid,
            object_boxes=None,
            object_classes=None,
            object_confs=None,
            object_valid=None,
        ):
            kwargs = {
                "boxes": boxes,
                "valid": valid,
            }
            if scene_object_tokens:
                kwargs.update(
                    {
                        "object_boxes": object_boxes,
                        "object_classes": object_classes.to(dtype=torch.long),
                        "object_confs": object_confs,
                        "object_valid": object_valid,
                    }
                )
            output = self.actor_model(
                video,
                **kwargs,
            )
            if not isinstance(output, (tuple, list)) or len(output) < 3:
                raise RuntimeError(
                    "Actor TensorRT export requires presence-head checkpoints."
                )
            logits = output[0]
            presence = output[2]
            if presence is None:
                raise RuntimeError(
                    "Actor TensorRT export requires presence-head checkpoints."
                )
            if scene_object_tokens:
                if len(output) != 4:
                    raise RuntimeError(
                        "Object-token actor export requires object_selection output."
                    )
                object_selection = output[3]
                return logits, presence, object_selection
            return logits, presence

    wrapped = ActorExport(model).to(device).eval()
    dummy_inputs, input_names = make_dummy_inputs(
        args.batch_size,
        clip_frames,
        args.input_size,
        max_actors,
        max_objects,
        num_object_classes,
        scene_object_tokens,
        device,
        args.mask_input_dtype,
    )
    with torch.inference_mode():
        outputs = wrapped(*dummy_inputs)
    logits = outputs[0]
    presence = outputs[1]
    output_names = ["logits", "presence"]
    output_shapes = {
        "logits": tuple(logits.shape),
        "presence": tuple(presence.shape),
    }
    if scene_object_tokens:
        output_names.append("object_selection")
        output_shapes["object_selection"] = tuple(outputs[2].shape)
    print(
        "PyTorch check:",
        {
            name: tuple(tensor.shape)
            for name, tensor in zip(input_names, dummy_inputs)
        }
        | output_shapes,
        flush=True,
    )

    torch.onnx.export(
        wrapped,
        dummy_inputs,
        str(onnx_out),
        input_names=input_names,
        output_names=output_names,
        opset_version=args.opset,
        do_constant_folding=False,
    )


def build_engine(trtexec, onnx_out, engine_out, args):
    layer_info_out = Path(str(engine_out) + ".layer_info.json")
    command = [
        trtexec,
        f"--onnx={onnx_out}",
        f"--saveEngine={engine_out}",
        f"--memPoolSize=workspace:{args.workspace_mib}M",
        f"--maxAuxStreams={args.max_aux_streams}",
        "--profilingVerbosity=detailed",
        f"--exportLayerInfo={layer_info_out}",
        "--skipInference",
    ]
    if args.precision == "fp16":
        command.append("--fp16")
    if args.precision == "fp32" and args.no_tf32:
        command.append("--noTF32")
    if args.builder_optimization_level is not None:
        command.append(f"--builderOptimizationLevel={int(args.builder_optimization_level)}")
    run_command(command)
    return command, layer_info_out


def benchmark_engine(trtexec, engine_out, args):
    command = [
        trtexec,
        f"--loadEngine={engine_out}",
        f"--warmUp={args.warmup_ms}",
        f"--duration={args.duration_sec}",
        f"--avgRuns={args.avg_runs}",
        f"--maxAuxStreams={args.max_aux_streams}",
    ]
    run_command(command)
    return command


def export_hparam_overrides(args):
    overrides = {}
    if args.disable_token_pruning:
        overrides.update(
            {
                "keep_rate": 1.0,
                "keep_rate_merge": 1.0,
            }
        )
    if args.trt_safe_attention:
        overrides["trt_safe_attention"] = 1
    return overrides


def checkpoint_payload(checkpoint_path, hparam_overrides=None):
    from train import _load_checkpoint

    checkpoint = _load_checkpoint(checkpoint_path)
    hparams = {}
    hparams.update(checkpoint.get("hyper_parameters", {}))
    hparams.update(checkpoint.get("datamodule_hyper_parameters", {}))
    if not hparams:
        raise RuntimeError(f"No hyperparameters found in checkpoint: {checkpoint_path}")
    if not hparams.get("actor_prompt", 0):
        raise RuntimeError("Checkpoint is not an actor-prompt checkpoint.")
    original_hparams = dict(hparams)
    if hparam_overrides:
        hparams.update(hparam_overrides)
    export_hparams = {
        "actor_prompt": int(hparams.get("actor_prompt", 0)),
        "actor_presence_head": int(hparams.get("actor_presence_head", 0)),
        "actor_interaction_heatmaps": int(
            hparams.get("actor_interaction_heatmaps", 0)
        ),
        "interaction_heatmap_channels": "per_actor_interacted_object",
        "scene_object_tokens": int(hparams.get("scene_object_tokens", 0)),
        "num_scene_object_tokens": int(hparams.get("num_scene_object_tokens", 0)),
        "num_object_classes": int(hparams.get("num_object_classes", 19)),
        "actor_object_logit_residual": 0,
        "actor_object_conditioned_action": int(
            bool(hparams.get("scene_object_tokens", 0))
        ),
        "n_frames": int(hparams.get("n_frames", 16)),
        "num_actor_tokens": int(hparams.get("num_actor_tokens", 0)),
        "num_classes": int(hparams.get("num_classes", 31)),
        "keep_rate": float(hparams.get("keep_rate", 1.0)),
        "keep_rate_merge": float(hparams.get("keep_rate_merge", 1.0)),
        "merge_type": str(hparams.get("merge_type", "")),
        "trt_safe_attention": int(hparams.get("trt_safe_attention", 0)),
    }
    return {
        "hparams": export_hparams,
        "hparam_overrides": dict(hparam_overrides or {}),
        "original_hparams": {
            "keep_rate": float(original_hparams.get("keep_rate", 1.0)),
            "keep_rate_merge": float(original_hparams.get("keep_rate_merge", 1.0)),
            "merge_type": str(original_hparams.get("merge_type", "")),
            "trt_safe_attention": int(original_hparams.get("trt_safe_attention", 0)),
        },
        "checkpoint_metadata": {
            "epoch": checkpoint.get("epoch"),
            "global_step": checkpoint.get("global_step"),
        },
    }


def internal_inspect(args):
    if not args.internal_json_out:
        raise ValueError("--internal-json-out is required for inspect mode.")
    payload = checkpoint_payload(args.checkpoint, export_hparam_overrides(args))
    Path(args.internal_json_out).write_text(json.dumps(payload) + "\n")


def internal_export(args):
    import torch

    from utils.actor_model import load_actor_model

    if not args.onnx_out:
        raise ValueError("--onnx-out is required for export mode.")
    if args.clip_frames is None:
        raise ValueError("--clip-frames is required for export mode.")
    if args.max_actors is None:
        raise ValueError("--max-actors is required for export mode.")

    device = torch.device(args.export_device)
    model, hparams = load_actor_model(
        args.checkpoint,
        device,
        hparam_overrides=export_hparam_overrides(args),
    )
    checkpoint_actors = int(hparams.get("num_actor_tokens", 0))
    if args.max_actors != checkpoint_actors:
        raise ValueError(
            f"--max-actors={args.max_actors} does not match checkpoint "
            f"num_actor_tokens={checkpoint_actors}."
        )
    if bool(hparams.get("scene_object_tokens", 0)):
        checkpoint_objects = int(hparams.get("num_scene_object_tokens", 0))
        if args.max_objects is not None and args.max_objects != checkpoint_objects:
            raise ValueError(
                f"--max-objects={args.max_objects} does not match checkpoint "
                f"num_scene_object_tokens={checkpoint_objects}."
            )
        max_objects = checkpoint_objects
    else:
        max_objects = 0
    export_onnx(
        model,
        Path(args.onnx_out),
        args,
        int(args.clip_frames),
        int(args.max_actors),
        int(max_objects),
        hparams,
        device,
    )


def run_inspect_child(args):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as handle:
        inspect_out = Path(handle.name)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--internal-mode",
        "inspect",
        "--checkpoint",
        args.checkpoint,
        "--internal-json-out",
        str(inspect_out),
    ]
    if args.disable_token_pruning:
        command.append("--disable-token-pruning")
    if args.trt_safe_attention:
        command.append("--trt-safe-attention")
    run_command(command)
    payload = json.loads(inspect_out.read_text())
    inspect_out.unlink(missing_ok=True)
    return payload


def run_export_child(args, onnx_out, clip_frames, max_actors, max_objects):
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--internal-mode",
        "export",
        "--checkpoint",
        args.checkpoint,
        "--onnx-out",
        str(onnx_out),
        "--export-device",
        args.export_device,
        "--batch-size",
        str(args.batch_size),
        "--clip-frames",
        str(clip_frames),
        "--input-size",
        str(args.input_size),
        "--max-actors",
        str(max_actors),
        "--max-objects",
        str(max_objects),
        "--opset",
        str(args.opset),
        "--mask-input-dtype",
        args.mask_input_dtype,
    ]
    if args.disable_token_pruning:
        command.append("--disable-token-pruning")
    if args.trt_safe_attention:
        command.append("--trt-safe-attention")
    run_command(command)
    return command


def main():
    args = parse_args()
    if args.internal_mode == "inspect":
        internal_inspect(args)
        return
    if args.internal_mode == "export":
        internal_export(args)
        return

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    payload = run_inspect_child(args)
    hparams = payload["hparams"]
    hparam_overrides = payload.get("hparam_overrides", {})
    original_hparams = payload.get("original_hparams", {})
    checkpoint_metadata = payload["checkpoint_metadata"]
    (
        onnx_out,
        engine_out,
        metadata_out,
        clip_frames,
        max_actors,
    ) = default_output_paths(args, hparams)
    checkpoint_actors = int(hparams.get("num_actor_tokens", 0))
    if max_actors != checkpoint_actors:
        raise ValueError(
            f"--max-actors={max_actors} does not match checkpoint "
            f"num_actor_tokens={checkpoint_actors}."
        )
    checkpoint_objects = int(hparams.get("num_scene_object_tokens", 0))
    scene_object_tokens = bool(hparams.get("scene_object_tokens", 0))
    max_objects = int(args.max_objects if args.max_objects is not None else checkpoint_objects)
    if scene_object_tokens and max_objects != checkpoint_objects:
        raise ValueError(
            f"--max-objects={max_objects} does not match checkpoint "
            f"num_scene_object_tokens={checkpoint_objects}."
        )
    if not scene_object_tokens:
        max_objects = 0
    require_writable(onnx_out, args.force)
    require_writable(engine_out, args.force)
    require_writable(metadata_out, args.force)

    trtexec = resolve_trtexec(args.trtexec)
    started = time.time()
    export_command = run_export_child(
        args,
        onnx_out,
        clip_frames,
        max_actors,
        max_objects,
    )
    print(f"Wrote ONNX: {onnx_out} ({onnx_out.stat().st_size / 1e6:.1f} MB)", flush=True)

    build_command, layer_info_out = build_engine(trtexec, onnx_out, engine_out, args)
    print(
        f"Wrote engine: {engine_out} ({engine_out.stat().st_size / 1e6:.1f} MB)",
        flush=True,
    )

    benchmark_command = None
    if args.benchmark:
        benchmark_command = benchmark_engine(trtexec, engine_out, args)

    metadata = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint_metadata.get("epoch"),
        "checkpoint_global_step": checkpoint_metadata.get("global_step"),
        "onnx": str(onnx_out),
        "engine": str(engine_out),
        "precision": args.precision,
        "no_tf32": bool(args.no_tf32),
        "disable_token_pruning": bool(args.disable_token_pruning),
        "hparam_overrides": hparam_overrides,
        "original_hparams": original_hparams,
        "export_hparams": {
            "keep_rate": float(hparams.get("keep_rate", 1.0)),
            "keep_rate_merge": float(hparams.get("keep_rate_merge", 1.0)),
            "merge_type": str(hparams.get("merge_type", "")),
            "trt_safe_attention": int(hparams.get("trt_safe_attention", 0)),
        },
        "input_shapes": {
            "video": [args.batch_size, clip_frames, 3, args.input_size, args.input_size],
            "boxes": [args.batch_size, max_actors, 4],
            "valid": [args.batch_size, max_actors],
        },
        "outputs": {
            "logits": [args.batch_size, max_actors, int(hparams.get("num_classes", 31))],
            "presence": [args.batch_size, max_actors],
        },
        "trtexec": trtexec,
        "mask_input_dtype": args.mask_input_dtype,
        "export_command": export_command,
        "build_command": build_command,
        "layer_info": str(layer_info_out),
        "benchmark_command": benchmark_command,
        "elapsed_sec": round(time.time() - started, 3),
    }
    if scene_object_tokens:
        metadata["input_shapes"].update(
            {
                "object_boxes": [args.batch_size, max_objects, 4],
                "object_classes": [args.batch_size, max_objects],
                "object_confs": [args.batch_size, max_objects],
                "object_valid": [args.batch_size, max_objects],
            }
        )
        metadata["outputs"]["object_selection"] = [
            args.batch_size,
            max_actors,
            max_objects + 1,
        ]
    metadata_out.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Wrote metadata: {metadata_out}", flush=True)


if __name__ == "__main__":
    main()
