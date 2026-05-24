import argparse
import os
import sys
import tempfile
import traceback

import numpy as np
import torch

from datasets.toyotasm import ToyotaSMDataset


class SmokeReport:
    def __init__(self):
        self.passed = 0
        self.failed = 0

    def check(self, condition, message, details=None):
        if bool(condition):
            self.passed += 1
            print(f"[PASS] {message}")
            return
        self.failed += 1
        print(f"[FAIL] {message}")
        if details is not None:
            print(f"       {details}")

    def finish(self):
        print(f"\nSummary: {self.passed} passed, {self.failed} failed")
        return self.failed == 0


def build_parser():
    parser = argparse.ArgumentParser(
        description="Smoke-test Toyota actor-prompt dataset outputs."
    )
    parser.add_argument("--data_dir", default=os.getenv("DATA_DIR", "."))
    parser.add_argument("--toyota_mp4_zip", default=os.getenv("MP4_ZIP"))
    parser.add_argument("--toyota_skeleton_zip", default=os.getenv("SKELETON_ZIP"))
    parser.add_argument(
        "--toyota_video_cache_dir",
        default=os.getenv(
            "VIDEO_CACHE_DIR",
            os.path.join(tempfile.gettempdir(), "poguise_toyota_mp4_cache"),
        ),
    )
    parser.add_argument(
        "--toyota_frame_count_cache", default=os.getenv("FRAME_COUNT_CACHE")
    )
    parser.add_argument("--toyota_max_samples", type=int, default=8)
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--n_frames", type=int, default=16)
    parser.add_argument("--n_landmarks", type=int, default=13)
    parser.add_argument("--num_actor_tokens", type=int, default=8)
    parser.add_argument("--num_classes", type=int, default=31)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip_single_actor",
        action="store_true",
        help="Only run the forced synthetic two-actor dataset check.",
    )
    return parser


def dataset_kwargs(args, synthetic_two_actor):
    return {
        "data_dir": args.data_dir,
        "set_type": "train",
        "task_type": "CS",
        "n_frames": args.n_frames,
        "n_frames_stride": 1,
        "n_landmarks": args.n_landmarks,
        "heatmap_agg": 1,
        "jitter_scales_min": 256,
        "jitter_scales_max": 320,
        "actor_prompt": 1,
        "num_actor_tokens": args.num_actor_tokens,
        "toyota_frame_source": "mp4_zip",
        "toyota_mp4_zip": args.toyota_mp4_zip,
        "toyota_skeleton_zip": args.toyota_skeleton_zip,
        "toyota_video_cache_dir": args.toyota_video_cache_dir,
        "toyota_frame_count_cache": args.toyota_frame_count_cache,
        "toyota_split_source": "auto",
        "toyota_max_samples": args.toyota_max_samples,
        "toyota_synthetic_warmup_epochs": 0 if synthetic_two_actor else 99,
        "toyota_synthetic_two_actor_prob": 1.0 if synthetic_two_actor else 0.0,
        "toyota_synthetic_three_actor_prob": 0.0,
        "toyota_synthetic_same_class_prob": 1.0,
    }


def build_dataset(args, synthetic_two_actor):
    ds = ToyotaSMDataset(**dataset_kwargs(args, synthetic_two_actor))
    ds.setup()
    return ds


def validate_sample(name, frames, target, expected_valid, args, report):
    print(f"\n{name}")
    print(f"frames: {frames.shape} {frames.dtype}")
    print(f"boxes: {target['boxes'].shape} {target['boxes']}")
    print(
        "valid:",
        target["valid"],
        "num valid:",
        int(target["valid"].sum().item()),
    )
    print(f"actions: {target['actions']}")
    print(f"heatmap: {target['heatmap'].shape} {target['heatmap'].dtype}")
    print(f"kp_vis: {target['kp_vis'].shape} {target['kp_vis'].dtype}")

    boxes = target["boxes"]
    valid = target["valid"]
    actions = target["actions"]
    heatmap = target["heatmap"]
    kp_vis = target["kp_vis"]

    report.check(
        frames.shape == (args.n_frames, 3, 224, 224),
        f"{name}: frames shape is [T, C, 224, 224]",
        frames.shape,
    )
    report.check(frames.dtype == torch.float32, f"{name}: frames are float32")
    report.check(torch.isfinite(frames).all(), f"{name}: frames are finite")

    report.check(
        boxes.shape == (args.num_actor_tokens, 4),
        f"{name}: boxes shape is [K, 4]",
        boxes.shape,
    )
    report.check(boxes.dtype == torch.float32, f"{name}: boxes are float32")
    report.check(torch.isfinite(boxes).all(), f"{name}: boxes are finite")
    report.check(
        bool(((boxes >= 0.0) & (boxes <= 1.0)).all()),
        f"{name}: boxes are normalized to [0, 1]",
    )

    report.check(
        valid.shape == (args.num_actor_tokens,),
        f"{name}: valid mask shape is [K]",
        valid.shape,
    )
    report.check(valid.dtype == torch.bool, f"{name}: valid mask is bool")
    report.check(
        int(valid.sum().item()) == expected_valid,
        f"{name}: expected valid actor count is {expected_valid}",
        int(valid.sum().item()),
    )

    valid_boxes = boxes[valid]
    if valid_boxes.numel() > 0:
        positive_area = (valid_boxes[:, 2] > valid_boxes[:, 0]) & (
            valid_boxes[:, 3] > valid_boxes[:, 1]
        )
        report.check(
            bool(positive_area.all()),
            f"{name}: valid boxes have positive area",
            valid_boxes,
        )

    report.check(
        actions.shape == (args.num_actor_tokens,),
        f"{name}: actions shape is [K]",
        actions.shape,
    )
    report.check(actions.dtype == torch.long, f"{name}: actions are long labels")
    report.check(
        bool((actions[~valid] == -100).all()),
        f"{name}: invalid slots use -100 labels",
        actions,
    )
    report.check(
        bool(((actions[valid] >= 0) & (actions[valid] < args.num_classes)).all()),
        f"{name}: valid labels are zero-based class ids",
        actions[valid],
    )

    report.check(
        heatmap.shape == (args.n_landmarks, 56, 56),
        f"{name}: heatmap shape is [n_landmarks, 56, 56]",
        heatmap.shape,
    )
    report.check(heatmap.dtype == torch.float32, f"{name}: heatmap is float32")
    report.check(torch.isfinite(heatmap).all(), f"{name}: heatmap is finite")
    report.check(
        kp_vis.shape == (args.n_landmarks, 56, 56),
        f"{name}: kp_vis shape is [n_landmarks, 56, 56]",
        kp_vis.shape,
    )
    report.check(kp_vis.dtype == torch.float32, f"{name}: kp_vis is float32")
    report.check(torch.isfinite(kp_vis).all(), f"{name}: kp_vis is finite")

    if expected_valid == 2:
        report.check(
            len(torch.unique(actions[valid])) == 1,
            f"{name}: forced same-class synthetic pair has matching labels",
            actions[valid],
        )


def run_smoke(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    report = SmokeReport()

    if args.toyota_mp4_zip is None:
        report.check(False, "toyota_mp4_zip is set")
    elif not os.path.exists(args.toyota_mp4_zip):
        report.check(False, "toyota_mp4_zip exists", args.toyota_mp4_zip)
    else:
        report.check(True, "toyota_mp4_zip exists")

    if args.toyota_skeleton_zip is None:
        report.check(False, "toyota_skeleton_zip is set")
    elif not os.path.exists(args.toyota_skeleton_zip):
        report.check(False, "toyota_skeleton_zip exists", args.toyota_skeleton_zip)
    else:
        report.check(True, "toyota_skeleton_zip exists")

    if report.failed:
        return report.finish()

    if not args.skip_single_actor:
        single_ds = build_dataset(args, synthetic_two_actor=False)
        frames, target = single_ds[args.sample_index]
        validate_sample("single_actor_sample", frames, target, 1, args, report)

    synthetic_ds = build_dataset(args, synthetic_two_actor=True)
    frames, target = synthetic_ds[args.sample_index]
    validate_sample("synthetic_two_actor_sample", frames, target, 2, args, report)

    return report.finish()


def main():
    args = build_parser().parse_args()
    try:
        ok = run_smoke(args)
    except Exception:
        traceback.print_exc()
        return 1
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
