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

from utils.rfdetr_tensorrt import TensorRTRFDETRNano


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check RF-DETR TensorRT raw-output drift against ONNX Runtime."
    )
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--max-abs-tolerance", type=float, default=1e-3)
    return parser.parse_args()


def synthetic_image():
    yy, xx = np.mgrid[0:480, 0:640]
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[..., 0] = (xx % 256).astype(np.uint8)
    image[..., 1] = (yy % 256).astype(np.uint8)
    image[..., 2] = ((xx // 2 + yy // 3) % 256).astype(np.uint8)
    image[120:360, 180:460, :] = np.array([180, 180, 210], dtype=np.uint8)
    return image


def run_onnx(onnx_path, input_name, tensor):
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1:
        raise RuntimeError(f"RF-DETR ONNX must have one input, got {[item.name for item in inputs]}")
    if inputs[0].name != input_name:
        raise RuntimeError(
            f"RF-DETR ONNX/TensorRT input name mismatch: {inputs[0].name} vs {input_name}"
        )
    result = session.run(None, {input_name: tensor.detach().cpu().numpy()})
    return {
        item.name: torch.from_numpy(np.asarray(value))
        for item, value in zip(outputs, result)
    }


def compare_outputs(reference, candidate):
    result = {}
    for name, ref in reference.items():
        value = candidate[name].to(dtype=torch.float32)
        ref = ref.to(dtype=torch.float32)
        diff = (ref - value).abs()
        result[name] = {
            "max_abs": float(diff.max().item()),
            "mean_abs": float(diff.mean().item()),
            "ref_shape": list(ref.shape),
            "candidate_shape": list(value.shape),
        }
    return result


def main():
    args = parse_args()
    detector = TensorRTRFDETRNano(args.engine)
    tensor, _orig_size = detector._prepare_image(synthetic_image())
    onnx_outputs = run_onnx(args.onnx, detector.input_name, tensor)
    trt_dets, trt_labels = detector._run_raw(tensor)
    trt_outputs = {
        "dets": trt_dets.detach().cpu(),
        "labels": trt_labels.detach().cpu(),
    }
    comparisons = compare_outputs(onnx_outputs, trt_outputs)
    max_abs_value = max(item["max_abs"] for item in comparisons.values())
    report = {
        "onnx": str(Path(args.onnx)),
        "engine": str(Path(args.engine)),
        "resolution": int(detector.resolution),
        "onnx_vs_tensorrt": comparisons,
        "onnx_vs_tensorrt_max_abs": float(max_abs_value),
        "max_abs_tolerance": float(args.max_abs_tolerance),
    }
    print(json.dumps(report, indent=2), flush=True)
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(report, indent=2) + "\n")
    if max_abs_value > float(args.max_abs_tolerance):
        raise SystemExit(
            "RF-DETR TensorRT drift exceeded tolerance: "
            f"{max_abs_value:.6g} > {args.max_abs_tolerance:.6g}"
        )


if __name__ == "__main__":
    main()
