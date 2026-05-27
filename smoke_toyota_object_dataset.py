import argparse
import os
import sys
import tempfile
import traceback

import numpy as np
import torch

from datasets.object_vocab import (
    NONE_OBJECT_ID,
    NUM_OBJECT_CLASSES,
    OBJECT_CLASSES,
    OBJECT_TO_ID,
)
from datasets.toyotasm import CS_DICT, ToyotaSMDataset


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
        description="Smoke-test Toyota object-prompt dataset outputs."
    )
    parser.add_argument("--data_dir", default=os.getenv("DATA_DIR", "."))
    parser.add_argument("--toyota_mp4_zip", default=os.getenv("MP4_ZIP"))
    parser.add_argument("--toyota_skeleton_zip", default=os.getenv("SKELETON_ZIP"))
    parser.add_argument("--object_detector_cache", required=True)
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
    parser.add_argument("--toyota_max_samples", type=int, default=64)
    parser.add_argument("--scan_samples", type=int, default=32)
    parser.add_argument("--sample_index", type=int, default=None)
    parser.add_argument("--n_frames", type=int, default=16)
    parser.add_argument("--n_landmarks", type=int, default=13)
    parser.add_argument("--num_actor_tokens", type=int, default=8)
    parser.add_argument("--num_object_tokens", type=int, default=24)
    parser.add_argument("--num_object_classes", type=int, default=NUM_OBJECT_CLASSES)
    parser.add_argument("--num_classes", type=int, default=31)
    parser.add_argument("--object_conf_threshold", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow_empty_object_sample",
        action="store_true",
        help="Do not fail if no cached objects are found in the scanned samples.",
    )
    return parser


def dataset_kwargs(args, synthetic_two_actor=False):
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
        "object_prompt": 1,
        "object_detector_cache": args.object_detector_cache,
        "num_object_tokens": args.num_object_tokens,
        "num_object_classes": args.num_object_classes,
        "object_conf_threshold": args.object_conf_threshold,
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
        "toyota_synthetic_same_class_prob": 0.0,
    }


def build_dataset(args, synthetic_two_actor=False):
    ds = ToyotaSMDataset(**dataset_kwargs(args, synthetic_two_actor))
    ds.setup()
    return ds


def action_name(label):
    action_id = int(label) + 1
    for name, idx in CS_DICT.items():
        if int(idx) == action_id:
            return name
    return f"class_{int(label)}"


def print_target_summary(name, target):
    print(f"\n{name}")
    for key in (
        "object_boxes",
        "object_cls",
        "object_conf",
        "object_valid",
        "object_heatmap",
        "object_heatmap_valid",
        "object_heatmap_weight",
        "interaction_cls",
        "interaction_valid",
    ):
        value = target[key]
        print(f"{key}: {tuple(value.shape)} {value.dtype}")
    valid_actions = target["actions"][target["valid"]]
    action_text = [action_name(label) for label in valid_actions.tolist()]
    object_ids = target["object_cls"][target["object_valid"]].tolist()
    object_text = [OBJECT_CLASSES[int(cls_id)] for cls_id in object_ids]
    print("actions:", action_text)
    print("objects:", object_text)
    print("interaction_cls:", target["interaction_cls"])
    print("interaction_valid:", target["interaction_valid"])


def validate_object_target(name, frames, target, args, report):
    print_target_summary(name, target)

    report.check(
        frames.shape == (args.n_frames, 3, 224, 224),
        f"{name}: frames shape is [T, C, 224, 224]",
        frames.shape,
    )
    report.check(torch.isfinite(frames).all(), f"{name}: frames are finite")

    object_boxes = target["object_boxes"]
    object_cls = target["object_cls"]
    object_conf = target["object_conf"]
    object_valid = target["object_valid"].bool()
    object_heatmap = target["object_heatmap"]
    object_heatmap_valid = target["object_heatmap_valid"].bool()
    object_heatmap_weight = target["object_heatmap_weight"]
    interaction_cls = target["interaction_cls"]
    interaction_valid = target["interaction_valid"].bool()

    report.check(
        object_boxes.shape == (args.num_object_tokens, 4),
        f"{name}: object_boxes shape is [M, 4]",
        object_boxes.shape,
    )
    report.check(
        object_cls.shape == (args.num_object_tokens,),
        f"{name}: object_cls shape is [M]",
        object_cls.shape,
    )
    report.check(
        object_conf.shape == (args.num_object_tokens,),
        f"{name}: object_conf shape is [M]",
        object_conf.shape,
    )
    report.check(
        object_valid.shape == (args.num_object_tokens,),
        f"{name}: object_valid shape is [M]",
        object_valid.shape,
    )
    report.check(object_valid.dtype == torch.bool, f"{name}: object_valid is bool")
    report.check(torch.isfinite(object_boxes).all(), f"{name}: object boxes finite")
    report.check(torch.isfinite(object_conf).all(), f"{name}: object conf finite")
    report.check(
        bool(((object_boxes >= 0.0) & (object_boxes <= 1.0)).all()),
        f"{name}: object boxes normalized to [0, 1]",
    )
    if object_valid.any():
        valid_boxes = object_boxes[object_valid]
        positive_area = (valid_boxes[:, 2] > valid_boxes[:, 0]) & (
            valid_boxes[:, 3] > valid_boxes[:, 1]
        )
        report.check(
            bool(positive_area.all()),
            f"{name}: valid object boxes have positive area",
            valid_boxes,
        )
        report.check(
            bool(
                (
                    (object_cls[object_valid] >= 0)
                    & (object_cls[object_valid] < args.num_object_classes)
                ).all()
            ),
            f"{name}: valid object classes are in [0, C_obj)",
            object_cls[object_valid],
        )
        report.check(
            bool((object_conf[object_valid] > 0).all()),
            f"{name}: valid object confidence is positive",
            object_conf[object_valid],
        )
    report.check(
        bool((object_cls[~object_valid] == NONE_OBJECT_ID).all()),
        f"{name}: invalid object class ids use NONE pad",
        object_cls,
    )

    report.check(
        object_heatmap.shape == (args.num_object_classes, 56, 56),
        f"{name}: object_heatmap shape is [C_obj, 56, 56]",
        object_heatmap.shape,
    )
    report.check(
        object_heatmap_valid.shape == (args.num_object_classes,),
        f"{name}: object_heatmap_valid shape is [C_obj]",
        object_heatmap_valid.shape,
    )
    report.check(
        object_heatmap_valid.dtype == torch.bool,
        f"{name}: object_heatmap_valid is bool",
    )
    report.check(
        object_heatmap_weight.shape == (args.num_object_classes, 56, 56),
        f"{name}: object_heatmap_weight shape is [C_obj, 56, 56]",
        object_heatmap_weight.shape,
    )
    report.check(torch.isfinite(object_heatmap).all(), f"{name}: object heatmap finite")
    report.check(
        torch.isfinite(object_heatmap_weight).all(),
        f"{name}: object heatmap weights finite",
    )
    report.check(
        bool(((object_heatmap >= 0.0) & (object_heatmap <= 1.0)).all()),
        f"{name}: object heatmap is in [0, 1]",
    )
    report.check(
        bool(((object_heatmap_weight >= 0.0) & (object_heatmap_weight <= 1.0)).all()),
        f"{name}: object heatmap weights are in [0, 1]",
    )
    if object_valid.any():
        report.check(
            bool(object_heatmap_valid.any()),
            f"{name}: object heatmap has valid channels when objects exist",
        )
    if object_heatmap_valid.any():
        visible_max = object_heatmap[object_heatmap_valid].flatten(1).max(dim=1).values
        report.check(
            bool((visible_max > 0).all()),
            f"{name}: valid object heatmap channels contain blobs",
            visible_max,
        )

    report.check(
        interaction_cls.shape == (args.num_actor_tokens,),
        f"{name}: interaction_cls shape is [K]",
        interaction_cls.shape,
    )
    report.check(
        interaction_valid.shape == (args.num_actor_tokens,),
        f"{name}: interaction_valid shape is [K]",
        interaction_valid.shape,
    )
    report.check(interaction_valid.dtype == torch.bool, f"{name}: interaction_valid is bool")
    report.check(
        bool(((interaction_cls >= 0) & (interaction_cls <= NONE_OBJECT_ID)).all()),
        f"{name}: interaction classes are object ids or NONE",
        interaction_cls,
    )
    report.check(
        bool((interaction_cls[~interaction_valid] == NONE_OBJECT_ID).all()),
        f"{name}: uncertain interaction labels are NONE with invalid mask",
        interaction_cls,
    )

    present_objects = set(int(v) for v in object_cls[object_valid].tolist())
    actions = target["actions"].long()
    valid_actor_slots = torch.nonzero(target["valid"].bool(), as_tuple=False).flatten()
    for slot in valid_actor_slots.tolist():
        name_for_slot = action_name(actions[slot])
        if name_for_slot == "Uselaptop" and OBJECT_TO_ID["laptop"] in present_objects:
            report.check(
                bool(
                    interaction_valid[slot]
                    and int(interaction_cls[slot]) == OBJECT_TO_ID["laptop"]
                ),
                f"{name}: Uselaptop gets laptop interaction when laptop is detected",
            )
        if name_for_slot == "Readbook" and OBJECT_TO_ID["book"] in present_objects:
            report.check(
                bool(
                    interaction_valid[slot]
                    and int(interaction_cls[slot]) == OBJECT_TO_ID["book"]
                ),
                f"{name}: Readbook gets book interaction when book is detected",
            )
        if name_for_slot == "WatchTV" and bool(interaction_valid[slot]):
            report.check(
                int(interaction_cls[slot])
                in {
                    OBJECT_TO_ID["tv_monitor"],
                    OBJECT_TO_ID["remote"],
                    OBJECT_TO_ID["couch"],
                },
                f"{name}: WatchTV interaction is TV/remote/couch when valid",
                interaction_cls[slot],
            )
        if name_for_slot == "WatchTV" and not (
            {
                OBJECT_TO_ID["tv_monitor"],
                OBJECT_TO_ID["remote"],
                OBJECT_TO_ID["couch"],
            }
            & present_objects
        ):
            report.check(
                not bool(interaction_valid[slot]),
                f"{name}: WatchTV interaction is ignored when expected context is absent",
            )
        if name_for_slot == "Usetablet":
            report.check(
                not bool(interaction_valid[slot]),
                f"{name}: Usetablet does not force a COCO phone/tablet surrogate",
            )
        if name_for_slot == "Drink.Fromcan":
            report.check(
                not bool(interaction_valid[slot]),
                f"{name}: Drink.Fromcan has no forced COCO object target",
            )


def pick_sample(ds, args, report):
    if len(ds) == 0:
        report.check(False, "dataset has at least one sample")
        return None

    if args.sample_index is not None:
        frames, target = ds[int(args.sample_index)]
        return int(args.sample_index), frames, target

    max_scan = min(len(ds), max(1, int(args.scan_samples)))
    first = None
    for idx in range(max_scan):
        frames, target = ds[idx]
        if first is None:
            first = (idx, frames, target)
        if target["object_valid"].bool().any():
            return idx, frames, target

    if args.allow_empty_object_sample:
        return first

    report.check(
        False,
        f"found at least one sample with object_valid=True in first {max_scan} samples",
    )
    return first


def run_smoke(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    report = SmokeReport()

    for name, path in (
        ("toyota_mp4_zip", args.toyota_mp4_zip),
        ("toyota_skeleton_zip", args.toyota_skeleton_zip),
        ("object_detector_cache", args.object_detector_cache),
    ):
        report.check(path is not None, f"{name} is set")
        if path is not None:
            report.check(os.path.exists(path), f"{name} exists", path)
    report.check(
        args.num_object_classes == NUM_OBJECT_CLASSES,
        f"num_object_classes is {NUM_OBJECT_CLASSES}",
        args.num_object_classes,
    )
    if report.failed:
        return report.finish()

    ds = build_dataset(args, synthetic_two_actor=False)
    sample = pick_sample(ds, args, report)
    if sample is None:
        return report.finish()
    idx, frames, target = sample
    validate_object_target(f"object_sample_{idx}", frames, target, args, report)

    synthetic_ds = build_dataset(args, synthetic_two_actor=True)
    sample = pick_sample(synthetic_ds, args, report)
    if sample is None:
        return report.finish()
    idx, frames, target = sample
    validate_object_target(f"synthetic_object_sample_{idx}", frames, target, args, report)

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
