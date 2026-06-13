#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.actor_model import load_actor_model
from utils.actor_tensorrt import TensorRTActorEngine
from utils.export_actor_tensorrt import make_dummy_inputs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check actor ONNX/TensorRT drift against the PyTorch checkpoint."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--max-abs-tolerance", type=float, default=1e-3)
    parser.add_argument(
        "--disable-token-pruning",
        action="store_true",
        help=(
            "Compare against a PyTorch model constructed with keep_rate=1.0 "
            "and keep_rate_merge=1.0. Normally this is read from the engine "
            "metadata when available."
        ),
    )
    parser.add_argument(
        "--trt-safe-attention",
        action="store_true",
        help=(
            "Compare against a PyTorch model constructed with explicit "
            "TRT-safe attention. Normally this is read from engine metadata."
        ),
    )
    return parser.parse_args()


def metadata_hparam_overrides(engine_path):
    path = Path(str(engine_path) + ".json")
    if not path.is_file():
        return {}
    data = json.loads(path.read_text())
    overrides = data.get("hparam_overrides", {})
    if not isinstance(overrides, dict):
        raise RuntimeError(f"Invalid hparam_overrides in {path}")
    return overrides


def cli_hparam_overrides(args):
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


class ActorExport(torch.nn.Module):
    def __init__(self, actor_model, uses_object_proposals):
        super().__init__()
        self.actor_model = actor_model
        self.uses_object_proposals = bool(uses_object_proposals)

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
        kwargs = {"boxes": boxes, "valid": valid}
        if self.uses_object_proposals:
            kwargs.update(
                {
                    "object_boxes": object_boxes,
                    "object_classes": object_classes.to(dtype=torch.long),
                    "object_confs": object_confs,
                    "object_valid": object_valid,
                }
            )
        output = self.actor_model(video, **kwargs)
        logits = output[0]
        if len(output) >= 3:
            presence = output[2]
        else:
            presence = valid.to(dtype=logits.dtype)
        return logits, presence


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
        tensor = inputs[item.name]
        array = tensor.detach().cpu().numpy()
        if item.type in dtype_by_type and array.dtype != dtype_by_type[item.type]:
            array = array.astype(dtype_by_type[item.type], copy=False)
        feed[item.name] = array
    outputs = session.run(None, feed)
    return {
        item.name: torch.from_numpy(np.asarray(value))
        for item, value in zip(session.get_outputs(), outputs)
    }


def tensor_dict(dummy_inputs, input_names):
    return {name: tensor for name, tensor in zip(input_names, dummy_inputs)}


def compare_tensor(ref, value, mask=None):
    value = value.to(dtype=torch.float32)
    ref = ref.to(dtype=torch.float32)
    if mask is not None:
        mask = mask.to(dtype=torch.bool)
        while mask.ndim < ref.ndim:
            mask = mask.unsqueeze(-1)
        mask = mask.expand_as(ref)
        ref = ref[mask]
        value = value[mask]
    diff = (ref - value).abs()
    return {
        "max_abs": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs": float(diff.mean().item()) if diff.numel() else 0.0,
        "ref_shape": list(ref.shape),
        "candidate_shape": list(value.shape),
    }


def compare_outputs(reference, candidate, valid, object_valid=None):
    result = {}
    for name, ref in reference.items():
        if name in {"logits", "presence"}:
            result[name] = compare_tensor(ref, candidate[name], valid)
        else:
            result[name] = compare_tensor(ref, candidate[name])
    return result


def max_abs(comparisons):
    return max(item["max_abs"] for item in comparisons.values())


def main():
    args = parse_args()
    engine = TensorRTActorEngine(args.engine)
    hparam_overrides = metadata_hparam_overrides(args.engine)
    hparam_overrides.update(cli_hparam_overrides(args))
    model, hparams, metadata = load_actor_model(
        args.checkpoint,
        torch.device("cpu"),
        return_metadata=True,
        hparam_overrides=hparam_overrides,
    )
    if bool(hparams.get("scene_object_tokens", 0)):
        raise RuntimeError(
            "scene_object_tokens checkpoints use the removed object-selection path. "
            "Train/export with actor_object_prompt_tokens instead."
        )
    if bool(hparams.get("actor_object_slot_head", 0)):
        raise RuntimeError(
            "actor_object_slot_head checkpoints are no longer supported. "
            "Train/export with actor_object_prompt_tokens instead."
        )
    if bool(hparams.get("actor_object_factorized_head", 0)):
        raise RuntimeError(
            "actor_object_factorized_head checkpoints are no longer supported. "
            "Train/export with actor_object_prompt_tokens instead."
        )
    actor_object_prompt_tokens = bool(hparams.get("actor_object_prompt_tokens", 0))
    uses_object_proposals = actor_object_prompt_tokens
    if uses_object_proposals != bool(engine.uses_object_proposals):
        raise RuntimeError(
            "Checkpoint/engine object-proposal input mismatch: "
            f"checkpoint={uses_object_proposals}, engine={engine.uses_object_proposals}"
        )
    wrapped = ActorExport(model, uses_object_proposals).eval()

    dummy_inputs, input_names = make_dummy_inputs(
        batch_size=engine.batch_size,
        clip_frames=engine.clip_frames,
        input_size=engine.input_size,
        max_actors=engine.num_actor_tokens,
        max_objects=engine.num_scene_object_tokens,
        num_object_classes=int(hparams.get("num_object_classes", 19)),
        uses_object_proposals=uses_object_proposals,
        device=torch.device("cpu"),
        mask_input_dtype="int32" if engine.dtypes["valid"] == torch.int32 else "bool",
    )
    inputs = tensor_dict(dummy_inputs, input_names)

    with torch.inference_mode():
        torch_outputs = wrapped(*dummy_inputs)
    output_names = ["logits", "presence"]
    reference = {
        name: value.detach().cpu()
        for name, value in zip(output_names, torch_outputs)
    }
    onnx_outputs = run_onnx(args.onnx, inputs)

    object_inputs = None
    if uses_object_proposals:
        object_inputs = {
            "object_boxes": inputs["object_boxes"],
            "object_classes": inputs["object_classes"],
            "object_confs": inputs["object_confs"],
            "object_valid": inputs["object_valid"],
        }
    trt_logits, trt_presence = engine(
        inputs["video"],
        inputs["boxes"],
        inputs["valid"],
        object_inputs,
    )
    trt_outputs = {
        "logits": trt_logits.detach().cpu(),
        "presence": trt_presence.detach().cpu(),
    }
    report = {
        "checkpoint": str(Path(args.checkpoint)),
        "checkpoint_epoch": metadata.get("epoch"),
        "onnx": str(Path(args.onnx)),
        "engine": str(Path(args.engine)),
        "actor_object_prompt_tokens": actor_object_prompt_tokens,
        "uses_object_proposals": uses_object_proposals,
        "hparam_overrides": hparam_overrides,
        "num_actor_tokens": int(engine.num_actor_tokens),
        "num_scene_object_tokens": int(engine.num_scene_object_tokens),
        "pytorch_vs_onnx": compare_outputs(
            reference,
            onnx_outputs,
            inputs["valid"],
            inputs.get("object_valid"),
        ),
        "pytorch_vs_tensorrt": compare_outputs(
            reference,
            trt_outputs,
            inputs["valid"],
            inputs.get("object_valid"),
        ),
        "max_abs_tolerance": float(args.max_abs_tolerance),
    }
    report["pytorch_vs_onnx_max_abs"] = max_abs(report["pytorch_vs_onnx"])
    report["pytorch_vs_tensorrt_max_abs"] = max_abs(report["pytorch_vs_tensorrt"])

    print(json.dumps(report, indent=2), flush=True)
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(report, indent=2) + "\n")

    if report["pytorch_vs_tensorrt_max_abs"] > float(args.max_abs_tolerance):
        raise SystemExit(
            "Actor TensorRT drift exceeded tolerance: "
            f"{report['pytorch_vs_tensorrt_max_abs']:.6g} > {args.max_abs_tolerance:.6g}"
        )


if __name__ == "__main__":
    main()
