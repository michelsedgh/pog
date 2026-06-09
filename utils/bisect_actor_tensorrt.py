#!/usr/bin/env python3
import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.actor_model import load_actor_model
from utils.actor_tensorrt import _torch_dtype
from utils.export_actor_tensorrt import make_dummy_inputs


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Export and compare an intermediate actor stage in PyTorch, ONNX, "
            "and TensorRT. Stages: patch, prefix, block0..block11, final_tokens, heads."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--out-dir", default="object_actor_live/exports/actor_trt_bisect")
    parser.add_argument("--trtexec", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--clip-frames", type=int, default=None)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--max-actors", type=int, default=None)
    parser.add_argument("--max-objects", type=int, default=None)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--workspace-mib", type=int, default=512)
    parser.add_argument("--builder-optimization-level", type=int, default=0)
    parser.add_argument("--no-tf32", action="store_true", default=True)
    parser.add_argument(
        "--allow-weight-streaming",
        action="store_true",
        help="Build the stage engine with TensorRT weight streaming enabled.",
    )
    parser.add_argument("--disable-token-pruning", action="store_true")
    parser.add_argument("--trt-safe-attention", action="store_true")
    parser.add_argument("--all-valid", action="store_true")
    parser.add_argument(
        "--mode",
        choices=["all", "export", "build", "check"],
        default="all",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-abs-tolerance", type=float, default=1e-3)
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


def parse_stage(stage):
    if stage in {"patch", "prefix", "final_tokens", "heads"}:
        return stage, None
    match = re.fullmatch(r"block(\d+)", stage)
    if match:
        index = int(match.group(1))
        if index < 0:
            raise ValueError("Block stage index must be non-negative.")
        return "block", index
    raise ValueError(
        f"Unsupported stage {stage!r}. Use patch, prefix, block0..blockN, "
        "final_tokens, or heads."
    )


def hparam_overrides(args):
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


class ActorStageExport(torch.nn.Module):
    def __init__(self, actor_model, stage):
        super().__init__()
        self.actor_model = actor_model
        self.net = actor_model.net
        self.stage_name, self.block_index = parse_stage(stage)
        if self.stage_name == "block" and self.block_index >= self.net.depth:
            raise ValueError(
                f"{stage} exceeds network depth {self.net.depth}."
            )
        self.scene_object_tokens = bool(actor_model.scene_object_tokens)

    def _token_prefix(
        self,
        x,
        boxes,
        valid,
        object_boxes,
        object_classes,
        object_confs,
        object_valid,
    ):
        net = self.net
        x = net.patch_embed(x)
        window_size = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)
        batch_size, num_video_tokens, _ = x.shape

        if net.pos_embed is not None:
            x = (
                x
                + net.pos_embed.expand(batch_size, -1, -1)
                .type_as(x)
                .to(x.device)
                .clone()
                .detach()
            )
        x = net.pos_drop(x)
        if self.stage_name == "patch":
            return x, None, None, None, window_size

        bbox_token_prior = None
        token_key_padding_mask = None
        prefix_tokens = []
        prefix_key_masks = []

        if net.n_actor_tokens > 0:
            boxes = boxes.to(device=x.device, dtype=x.dtype)
            valid = valid.to(device=x.device, dtype=torch.bool)
            actor_tokens = (
                net.actor_token.expand(batch_size, net.n_actor_tokens, -1)
                + net.actor_slot_embed.expand(batch_size, -1, -1)
                + net.bbox_mlp(boxes)
                + net.valid_embed(valid.long()).to(dtype=x.dtype)
            )
            bbox_token_prior = net._bbox_token_prior(boxes, valid, window_size)
            prefix_tokens.append(actor_tokens)
            prefix_key_masks.append(
                torch.zeros(
                    batch_size,
                    net.n_actor_tokens,
                    dtype=torch.bool,
                    device=x.device,
                )
            )

        if net.n_object_tokens > 0:
            object_boxes = object_boxes.to(device=x.device, dtype=x.dtype).clamp(0.0, 1.0)
            object_classes = object_classes.to(device=x.device, dtype=torch.long)
            object_confs = object_confs.to(device=x.device, dtype=x.dtype).clamp(0.0, 1.0)
            object_valid = object_valid.to(device=x.device, dtype=torch.bool)
            safe_object_classes = object_classes.clamp(0, net.none_object_id)
            safe_object_classes = torch.where(
                object_valid,
                safe_object_classes,
                torch.full_like(safe_object_classes, net.none_object_id),
            )
            object_visual_feat = net._pool_box_features(
                x,
                object_boxes,
                object_valid,
                window_size,
            )
            object_tokens = (
                net.object_slot_embed.expand(batch_size, -1, -1)
                + net.object_cls_embed(safe_object_classes).to(dtype=x.dtype)
                + net.object_bbox_mlp(object_boxes)
                + net.object_conf_mlp(object_confs.unsqueeze(-1))
                + net.object_visual_proj(object_visual_feat)
                + net.object_valid_embed(object_valid.long()).to(dtype=x.dtype)
            )
            object_tokens = object_tokens * object_valid.to(dtype=x.dtype).unsqueeze(-1)
            prefix_tokens.append(object_tokens)
            prefix_key_masks.append(~object_valid)

        if net.n_registers > 0:
            prefix_tokens.append(net.register_tokens.expand(batch_size, -1, -1))
            prefix_key_masks.append(
                torch.zeros(
                    batch_size,
                    net.n_registers,
                    dtype=torch.bool,
                    device=x.device,
                )
            )

        if net.n_heatmap_out_channels > 0:
            prefix_tokens.append(net.heatmap_tokens.expand(batch_size, -1, -1))
            prefix_key_masks.append(
                torch.zeros(
                    batch_size,
                    net.n_heatmap_tokens,
                    dtype=torch.bool,
                    device=x.device,
                )
            )

        if prefix_tokens:
            x = torch.cat([*prefix_tokens, x], dim=1)
            prefix_mask = torch.cat(prefix_key_masks, dim=1)
            video_mask = torch.zeros(
                batch_size,
                num_video_tokens,
                dtype=torch.bool,
                device=x.device,
            )
            token_key_padding_mask = torch.cat([prefix_mask, video_mask], dim=1)

        x = torch.cat([net.class_token.expand(batch_size, -1, -1), x], dim=1)
        if token_key_padding_mask is not None:
            class_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=x.device)
            token_key_padding_mask = torch.cat(
                [class_mask, token_key_padding_mask],
                dim=1,
            )
        idx = torch.arange(0, num_video_tokens, device=x.device).unsqueeze(0).repeat(
            batch_size,
            1,
        )
        return x, idx, token_key_padding_mask, bbox_token_prior, window_size

    def _run_blocks(self, x, idx, token_key_padding_mask, bbox_token_prior, window_size):
        stop = self.net.depth - 1
        if self.stage_name == "block":
            stop = self.block_index
        for index in range(stop + 1):
            x, idx, token_key_padding_mask = self.net.blocks[index](
                x,
                idx,
                window_size,
                bbox_token_prior=bbox_token_prior,
                key_padding_mask=token_key_padding_mask,
            )
        return x

    def _final_tokens(self, x):
        net = self.net
        x_class = x[:, 0, :]
        x_actor = x[:, 1 : 1 + net.n_actor_tokens, :]
        x_object = None
        if net.n_object_tokens > 0:
            object_start = 1 + net.n_actor_tokens
            object_end = object_start + net.n_object_tokens
            x_object = x[:, object_start:object_end, :]

        if net.fc_norm is not None:
            x_class = net.fc_norm(x_class)
            x_actor = net.fc_norm(x_actor)
            if x_object is not None:
                x_object = net.fc_norm(x_object)
        else:
            x_class = net.norm(x_class)
            x_actor = net.norm(x_actor)
            if x_object is not None:
                x_object = net.norm(x_object)
        x_class = net.head_dropout(x_class).unsqueeze(1)
        x_actor = net.head_dropout(x_actor)
        if x_object is None:
            return torch.cat([x_class, x_actor], dim=1)
        x_object = net.head_dropout(x_object)
        return torch.cat([x_class, x_actor, x_object], dim=1)

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
        video = video.permute(0, 2, 1, 3, 4)
        x, idx, token_key_padding_mask, bbox_token_prior, window_size = self._token_prefix(
            video,
            boxes,
            valid,
            object_boxes,
            object_classes,
            object_confs,
            object_valid,
        )
        if self.stage_name in {"patch", "prefix"}:
            return (x,)

        x = self._run_blocks(
            x,
            idx,
            token_key_padding_mask,
            bbox_token_prior,
            window_size,
        )
        if self.stage_name == "block":
            return (x,)

        final_tokens = self._final_tokens(x)
        if self.stage_name == "final_tokens":
            return (final_tokens,)

        x_actor = final_tokens[:, 1 : 1 + self.net.n_actor_tokens, :]
        x_object = None
        if self.net.n_object_tokens > 0:
            object_start = 1 + self.net.n_actor_tokens
            object_end = object_start + self.net.n_object_tokens
            x_object = final_tokens[:, object_start:object_end, :]
        object_selection_logits = None
        if self.actor_model.object_selection_head is not None:
            object_selection_logits = self.actor_model.object_selection_head(
                x_actor,
                x_object,
                object_valid.to(device=x_actor.device, dtype=torch.bool),
            )
        if (
            x_object is not None
            and object_selection_logits is not None
            and hasattr(self.actor_model.actor_head, "action_head")
        ):
            action_logits = self.actor_model.actor_head(
                x_actor,
                x_object,
                object_selection_logits,
                object_valid.to(device=x_actor.device, dtype=torch.bool),
            )
        else:
            action_logits = self.actor_model.actor_head(x_actor)
        presence = self.actor_model.presence_head(x_actor).squeeze(-1)
        if object_selection_logits is None:
            return action_logits, presence
        return action_logits, presence, object_selection_logits


class GenericTensorRTEngine:
    def __init__(self, engine_path):
        import tensorrt as trt

        if not torch.cuda.is_available():
            raise RuntimeError("TensorRT engine requires CUDA.")
        engine_path = Path(engine_path)
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")
        streamable_weights = int(getattr(self.engine, "streamable_weights_size", 0))
        if streamable_weights > 0 and hasattr(self.engine, "weight_streaming_budget_v2"):
            self.engine.weight_streaming_budget_v2 = streamable_weights
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"Failed to create TensorRT context: {engine_path}")
        self.device = torch.device("cuda")
        self.stream = torch.cuda.Stream()
        self.input_names = []
        self.output_names = []
        self.shapes = {}
        self.dtypes = {}
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            mode = self.engine.get_tensor_mode(name)
            self.shapes[name] = tuple(
                int(dim) for dim in self.engine.get_tensor_shape(name)
            )
            self.dtypes[name] = _torch_dtype(self.engine.get_tensor_dtype(name))
            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            elif mode == trt.TensorIOMode.OUTPUT:
                self.output_names.append(name)
            else:
                raise RuntimeError(f"Unknown TensorRT tensor mode: {mode}")

    def _prepare(self, name, tensor):
        shape = self.shapes[name]
        dtype = self.dtypes[name]
        if tuple(tensor.shape) != shape:
            raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
        if tensor.dtype != dtype:
            tensor = tensor.to(dtype=dtype)
        if tensor.device.type != "cuda":
            tensor = tensor.to(device=self.device)
        return tensor.contiguous()

    def __call__(self, inputs):
        tensors = {
            name: self._prepare(name, inputs[name])
            for name in self.input_names
        }
        outputs = {
            name: torch.empty(
                self.shapes[name],
                dtype=self.dtypes[name],
                device=self.device,
            )
            for name in self.output_names
        }
        tensors.update(outputs)
        current_stream = torch.cuda.current_stream()
        self.stream.wait_stream(current_stream)
        with torch.cuda.stream(self.stream):
            for name, tensor in tensors.items():
                self.context.set_tensor_address(name, tensor.data_ptr())
            ok = self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
        if not ok:
            raise RuntimeError("TensorRT execution failed.")
        self.stream.synchronize()
        current_stream.wait_stream(self.stream)
        return {name: tensor.detach().cpu() for name, tensor in outputs.items()}


def run_command(command):
    print("$ " + " ".join(str(part) for part in command), flush=True)
    subprocess.run(command, check=True)


def run_onnx(onnx_path, inputs):
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    dtype_by_type = {
        "tensor(float16)": np.float16,
        "tensor(float)": np.float32,
        "tensor(double)": np.float64,
        "tensor(int32)": np.int32,
        "tensor(int64)": np.int64,
        "tensor(bool)": np.bool_,
    }
    feed = {}
    for item in session.get_inputs():
        array = inputs[item.name].detach().cpu().numpy()
        if item.type in dtype_by_type and array.dtype != dtype_by_type[item.type]:
            array = array.astype(dtype_by_type[item.type], copy=False)
        feed[item.name] = array
    values = session.run(None, feed)
    return {
        item.name: torch.from_numpy(np.asarray(value))
        for item, value in zip(session.get_outputs(), values)
    }


def compare(ref, value):
    ref = ref.to(dtype=torch.float32)
    value = value.to(dtype=torch.float32)
    diff = (ref - value).abs()
    return {
        "max_abs": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs": float(diff.mean().item()) if diff.numel() else 0.0,
        "ref_shape": list(ref.shape),
        "candidate_shape": list(value.shape),
    }


def max_abs(report):
    return max(item["max_abs"] for item in report.values())


def output_paths(args):
    out_dir = Path(args.out_dir)
    suffix = []
    if args.disable_token_pruning:
        suffix.append("noprune")
    if args.trt_safe_attention:
        suffix.append("safeattn")
    if args.all_valid:
        suffix.append("allvalid")
    suffix_text = "_" + "_".join(suffix) if suffix else ""
    stage_dir = out_dir / f"{args.stage}{suffix_text}"
    stage_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = stage_dir / f"{args.stage}{suffix_text}.onnx"
    engine_path = stage_dir / f"{args.stage}{suffix_text}.engine"
    report_path = stage_dir / f"{args.stage}{suffix_text}.check.json"
    return onnx_path, engine_path, report_path


def main():
    args = parse_args()
    parse_stage(args.stage)
    onnx_path, engine_path, report_path = output_paths(args)
    if args.mode == "all" and not args.force:
        for path in [onnx_path, engine_path, report_path]:
            if path.exists():
                raise FileExistsError(f"{path} exists. Pass --force to overwrite.")
    elif args.mode == "export" and onnx_path.exists() and not args.force:
        raise FileExistsError(f"{onnx_path} exists. Pass --force to overwrite.")
    elif args.mode == "build" and engine_path.exists() and not args.force:
        raise FileExistsError(f"{engine_path} exists. Pass --force to overwrite.")
    elif args.mode == "check" and report_path.exists() and not args.force:
        raise FileExistsError(f"{report_path} exists. Pass --force to overwrite.")

    model, hparams, metadata = load_actor_model(
        args.checkpoint,
        torch.device("cpu"),
        return_metadata=True,
        hparam_overrides=hparam_overrides(args),
    )
    scene_object_tokens = bool(hparams.get("scene_object_tokens", 0))
    clip_frames = int(args.clip_frames or hparams.get("n_frames", 16))
    max_actors = int(args.max_actors or hparams.get("num_actor_tokens", 0))
    max_objects = int(
        args.max_objects
        if args.max_objects is not None
        else hparams.get("num_scene_object_tokens", 0)
    )
    if not scene_object_tokens:
        max_objects = 0
    wrapped = ActorStageExport(model, args.stage).eval()
    dummy_inputs, input_names = make_dummy_inputs(
        batch_size=args.batch_size,
        clip_frames=clip_frames,
        input_size=args.input_size,
        max_actors=max_actors,
        max_objects=max_objects,
        num_object_classes=int(hparams.get("num_object_classes", 19)),
        scene_object_tokens=scene_object_tokens,
        device=torch.device("cpu"),
        mask_input_dtype="bool",
    )
    inputs = {name: tensor for name, tensor in zip(input_names, dummy_inputs)}
    if args.all_valid:
        inputs["valid"].fill_(True)
        if scene_object_tokens:
            inputs["object_valid"].fill_(True)
    dummy_inputs = tuple(inputs[name] for name in input_names)

    with torch.inference_mode():
        torch_outputs = wrapped(*dummy_inputs)
    output_names = [f"output_{index}" for index in range(len(torch_outputs))]
    if args.mode in {"all", "export"}:
        torch.onnx.export(
            wrapped,
            dummy_inputs,
            str(onnx_path),
            input_names=input_names,
            output_names=output_names,
            opset_version=args.opset,
            do_constant_folding=False,
        )
        print(f"Wrote ONNX: {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")
    elif not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX not found for {args.mode} mode: {onnx_path}")
    if args.mode == "export":
        return

    trtexec = resolve_trtexec(args.trtexec)
    layer_info_path = Path(str(engine_path) + ".layer_info.json")
    build_command = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--memPoolSize=workspace:{args.workspace_mib}M",
        "--maxAuxStreams=0",
        "--profilingVerbosity=detailed",
        f"--exportLayerInfo={layer_info_path}",
        "--skipInference",
    ]
    if args.no_tf32:
        build_command.append("--noTF32")
    if args.builder_optimization_level is not None:
        build_command.append(
            f"--builderOptimizationLevel={int(args.builder_optimization_level)}"
        )
    if args.allow_weight_streaming:
        build_command.extend(["--allowWeightStreaming", "--stronglyTyped"])
    if args.mode in {"all", "build"}:
        run_command(build_command)
        print(f"Wrote engine: {engine_path} ({engine_path.stat().st_size / 1e6:.1f} MB)")
    elif not engine_path.is_file():
        raise FileNotFoundError(f"Engine not found for {args.mode} mode: {engine_path}")
    if args.mode == "build":
        return

    onnx_outputs = run_onnx(onnx_path, inputs)
    trt_outputs = GenericTensorRTEngine(engine_path)(inputs)
    reference = {
        name: value.detach().cpu()
        for name, value in zip(output_names, torch_outputs)
    }
    report = {
        "checkpoint": str(Path(args.checkpoint)),
        "checkpoint_epoch": metadata.get("epoch"),
        "stage": args.stage,
        "disable_token_pruning": bool(args.disable_token_pruning),
        "trt_safe_attention": bool(args.trt_safe_attention),
        "all_valid": bool(args.all_valid),
        "hparam_overrides": hparam_overrides(args),
        "onnx": str(onnx_path),
        "engine": str(engine_path),
        "pytorch_vs_onnx": {
            name: compare(reference[name], onnx_outputs[name])
            for name in output_names
        },
        "pytorch_vs_tensorrt": {
            name: compare(reference[name], trt_outputs[name])
            for name in output_names
        },
        "build_command": build_command,
        "layer_info": str(layer_info_path),
        "max_abs_tolerance": float(args.max_abs_tolerance),
    }
    report["pytorch_vs_onnx_max_abs"] = max_abs(report["pytorch_vs_onnx"])
    report["pytorch_vs_tensorrt_max_abs"] = max_abs(report["pytorch_vs_tensorrt"])
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if report["pytorch_vs_tensorrt_max_abs"] > float(args.max_abs_tolerance):
        raise SystemExit(
            "Stage TensorRT drift exceeded tolerance: "
            f"{report['pytorch_vs_tensorrt_max_abs']:.6g} > "
            f"{args.max_abs_tolerance:.6g}"
        )


if __name__ == "__main__":
    main()
