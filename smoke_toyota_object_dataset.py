#!/usr/bin/env python3
import argparse
import os
import sys
import tempfile
import traceback

import numpy as np
import torch

from datasets.object_vocab import (
    NONE_OBJECT_ID,
    OBJECT_CLASSES,
    OBJECTLESS_ACTIONS,
    OBJECT_ACTIONS_WITHOUT_RELIABLE_TEACHER,
    OBJECT_TO_ID,
    STRONG_ACTION_OBJECTS,
)
from datasets.toyota_action_taxonomy import (
    TOYOTA_ACTION_TAXONOMIES,
    toyota_action_names,
    toyota_num_classes,
)
from datasets.toyotasm import ToyotaSMDataset


class SmokeReport:
    def __init__(self):
        self.passed = 0
        self.failed = 0

    def check(self, condition, message, details=None):
        if bool(condition):
            self.passed += 1
            print(f"[PASS] {message}", flush=True)
            return
        self.failed += 1
        print(f"[FAIL] {message}", flush=True)
        if details is not None:
            print(f"       {details}", flush=True)

    def finish(self):
        print(f"\nSummary: {self.passed} passed, {self.failed} failed", flush=True)
        return self.failed == 0


def build_parser():
    parser = argparse.ArgumentParser(
        description="Smoke-test Toyota actor-conditioned interaction heatmap targets."
    )
    parser.add_argument("--data_dir", default=os.getenv("DATA_DIR", "."))
    parser.add_argument("--toyota_frame_source", default="mp4_zip", choices=["mp4_zip", "frames"])
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
    parser.add_argument("--scan_samples", type=int, default=64)
    parser.add_argument("--sample_index", type=int, default=None)
    parser.add_argument("--n_frames", type=int, default=16)
    parser.add_argument("--n_landmarks", type=int, default=13)
    parser.add_argument("--num_actor_tokens", type=int, default=8)
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument(
        "--toyota_action_taxonomy",
        default="toyota_31",
        choices=TOYOTA_ACTION_TAXONOMIES,
    )
    parser.add_argument("--object_conf_threshold", type=float, default=0.25)
    parser.add_argument("--object_camera_allowlist", default=None)
    parser.add_argument("--object_ignore_regions", default=None)
    parser.add_argument("--object_track_iou_threshold", type=float, default=0.2)
    parser.add_argument("--interaction_guided_sampling", type=int, default=1)
    parser.add_argument("--interaction_min_sampled_object_frames", type=int, default=1)
    parser.add_argument("--interaction_repair_radius_frames", type=int, default=8)
    parser.add_argument("--interaction_quality_min_actor_score", type=float, default=1.0)
    parser.add_argument("--interaction_quality_min_track_frames", type=int, default=1)
    parser.add_argument(
        "--interaction_quality_min_track_coverage", type=float, default=0.0
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow_no_interaction_sample",
        action="store_true",
        help="Do not fail if the scan finds no strong-action interaction target.",
    )
    return parser


def dataset_kwargs(args, synthetic_two_actor=False):
    kwargs = {
        "data_dir": args.data_dir,
        "set_type": "train",
        "task_type": "CS",
        "toyota_action_taxonomy": args.toyota_action_taxonomy,
        "n_frames": args.n_frames,
        "n_frames_stride": 1,
        "n_landmarks": args.n_landmarks,
        "num_classes": args.num_classes
        if args.num_classes is not None
        else toyota_num_classes("CS", args.toyota_action_taxonomy),
        "heatmap_agg": 1,
        "jitter_scales_min": 256,
        "jitter_scales_max": 320,
        "actor_prompt": 1,
        "num_actor_tokens": args.num_actor_tokens,
        "actor_interaction_heatmaps": 1,
        "object_detector_cache": args.object_detector_cache,
        "object_camera_allowlist": args.object_camera_allowlist,
        "object_ignore_regions": args.object_ignore_regions,
        "object_conf_threshold": args.object_conf_threshold,
        "object_track_iou_threshold": args.object_track_iou_threshold,
        "interaction_guided_sampling": args.interaction_guided_sampling,
        "interaction_min_sampled_object_frames": args.interaction_min_sampled_object_frames,
        "interaction_repair_radius_frames": args.interaction_repair_radius_frames,
        "interaction_quality_min_actor_score": args.interaction_quality_min_actor_score,
        "interaction_quality_min_track_frames": args.interaction_quality_min_track_frames,
        "interaction_quality_min_track_coverage": (
            args.interaction_quality_min_track_coverage
        ),
        "toyota_frame_source": args.toyota_frame_source,
        "toyota_skeleton_zip": args.toyota_skeleton_zip,
        "toyota_frame_count_cache": args.toyota_frame_count_cache,
        "toyota_split_source": "auto",
        "toyota_max_samples": args.toyota_max_samples,
        "toyota_synthetic_warmup_epochs": 0 if synthetic_two_actor else 99,
        "toyota_synthetic_two_actor_prob": 1.0 if synthetic_two_actor else 0.0,
        "toyota_synthetic_three_actor_prob": 0.0,
        "toyota_synthetic_same_class_prob": 0.0,
    }
    if args.toyota_frame_source == "mp4_zip":
        kwargs["toyota_mp4_zip"] = args.toyota_mp4_zip
        kwargs["toyota_video_cache_dir"] = args.toyota_video_cache_dir
    return kwargs


def build_dataset(args, synthetic_two_actor=False):
    ds = ToyotaSMDataset(**dataset_kwargs(args, synthetic_two_actor))
    ds.setup()
    return ds


def action_name(label):
    names = toyota_action_names("CS", getattr(action_name, "taxonomy", "toyota_31"))
    label = int(label)
    if 0 <= label < len(names):
        return names[label]
    return f"class_{int(label)}"


def print_target_summary(name, target):
    print(f"\n{name}", flush=True)
    for key in (
        "boxes",
        "valid",
        "actions",
        "interaction_cls",
        "interaction_valid",
        "interaction_heatmap",
        "interaction_heatmap_valid",
        "interaction_heatmap_positive_valid",
    ):
        value = target[key]
        print(f"{key}: {tuple(value.shape)} {value.dtype}", flush=True)
    valid_actions = target["actions"][target["valid"]]
    action_text = [action_name(label) for label in valid_actions.tolist()]
    interactions = []
    for slot in torch.nonzero(target["interaction_valid"], as_tuple=False).flatten():
        cls_id = int(target["interaction_cls"][slot])
        cls_name = "NONE" if cls_id == NONE_OBJECT_ID else OBJECT_CLASSES[cls_id]
        interactions.append(f"slot{int(slot)}:{cls_name}")
    print("actions:", action_text, flush=True)
    print("interactions:", interactions or "none", flush=True)


def validate_interaction_target(name, frames, target, args, report):
    print_target_summary(name, target)

    report.check(
        frames.shape == (args.n_frames, 3, 224, 224),
        f"{name}: frames shape is [T, C, 224, 224]",
        frames.shape,
    )
    report.check(torch.isfinite(frames).all(), f"{name}: frames are finite")

    actor_valid = target["valid"].bool()
    interaction_cls = target["interaction_cls"]
    interaction_valid = target["interaction_valid"].bool()
    interaction_heatmap = target["interaction_heatmap"]
    interaction_heatmap_valid = target["interaction_heatmap_valid"].bool()
    interaction_heatmap_positive_valid = target[
        "interaction_heatmap_positive_valid"
    ].bool()
    interaction_object_index = target["interaction_object_index"]
    interaction_object_index_valid = target["interaction_object_index_valid"].bool()

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
        interaction_heatmap.shape == (args.num_actor_tokens, 56, 56),
        f"{name}: interaction_heatmap shape is [K, 56, 56]",
        interaction_heatmap.shape,
    )
    report.check(
        interaction_heatmap_valid.shape == (args.num_actor_tokens,),
        f"{name}: interaction_heatmap_valid shape is [K]",
        interaction_heatmap_valid.shape,
    )
    report.check(
        interaction_heatmap_valid.dtype == torch.bool,
        f"{name}: interaction_heatmap_valid is bool",
    )
    report.check(
        interaction_heatmap_positive_valid.shape == (args.num_actor_tokens,),
        f"{name}: interaction_heatmap_positive_valid shape is [K]",
        interaction_heatmap_positive_valid.shape,
    )
    report.check(
        interaction_heatmap_positive_valid.dtype == torch.bool,
        f"{name}: interaction_heatmap_positive_valid is bool",
    )
    report.check(
        torch.isfinite(interaction_heatmap).all(),
        f"{name}: interaction_heatmap finite",
    )
    report.check(
        bool(((interaction_heatmap >= 0.0) & (interaction_heatmap <= 1.0)).all()),
        f"{name}: interaction_heatmap is in [0, 1]",
    )
    report.check(
        bool((interaction_valid <= actor_valid).all()),
        f"{name}: interaction targets only exist on valid actor slots",
    )
    report.check(
        bool((interaction_heatmap_valid <= interaction_valid).all()),
        f"{name}: heatmap-valid slots are supervised interaction slots",
    )
    report.check(
        bool((interaction_heatmap_positive_valid <= interaction_heatmap_valid).all()),
        f"{name}: positive heatmap-valid channels are loss-valid channels",
    )
    report.check(
        bool(((interaction_cls >= 0) & (interaction_cls <= NONE_OBJECT_ID)).all()),
        f"{name}: interaction classes are object ids or NONE",
        interaction_cls,
    )
    report.check(
        bool((interaction_cls[~interaction_valid] == NONE_OBJECT_ID).all()),
        f"{name}: ignored interactions use NONE class",
        interaction_cls,
    )

    if interaction_heatmap_positive_valid.any():
        valid_max = interaction_heatmap[interaction_heatmap_positive_valid].flatten(
            1
        ).max(dim=1).values
        report.check(
            bool((valid_max > 0).all()),
            f"{name}: positive interaction heatmaps contain visible blobs",
            valid_max,
        )

    for slot in torch.nonzero(actor_valid, as_tuple=False).flatten().tolist():
        name_for_slot = action_name(target["actions"][slot])
        if name_for_slot in OBJECTLESS_ACTIONS:
            report.check(
                bool(interaction_object_index_valid[slot]),
                f"{name}: true objectless action has a NONE object label",
            )
            report.check(
                int(interaction_object_index[slot]) == 0,
                f"{name}: true objectless action label is NONE",
                interaction_object_index[slot],
            )
        if name_for_slot in OBJECT_ACTIONS_WITHOUT_RELIABLE_TEACHER:
            report.check(
                not bool(interaction_object_index_valid[slot]),
                f"{name}: untrusted object action does not get a fake NONE label",
            )
            report.check(
                not bool(interaction_valid[slot]),
                f"{name}: untrusted object action has no forced interaction target",
            )
        if name_for_slot in STRONG_ACTION_OBJECTS and bool(interaction_valid[slot]):
            expected_ids = {
                OBJECT_TO_ID[object_name]
                for object_name in STRONG_ACTION_OBJECTS[name_for_slot]
            }
            report.check(
                int(interaction_cls[slot]) in expected_ids,
                (
                    f"{name}: {name_for_slot} supervised by declared "
                    "strong object target"
                ),
                interaction_cls[slot],
            )
            report.check(
                bool(interaction_heatmap_positive_valid[slot]),
                f"{name}: supervised actor slot has an interaction heatmap",
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
        positive_valid = target.get(
            "interaction_heatmap_positive_valid",
            target["interaction_heatmap_valid"],
        ).bool()
        if positive_valid.any():
            return idx, frames, target

    if args.allow_no_interaction_sample:
        return first

    report.check(
        False,
        f"found at least one strong-action interaction target in first {max_scan} samples",
    )
    return first


def run_smoke(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    report = SmokeReport()

    required_paths = [
        ("toyota_skeleton_zip", args.toyota_skeleton_zip),
        ("object_detector_cache", args.object_detector_cache),
    ]
    if args.toyota_frame_source == "mp4_zip":
        required_paths.append(("toyota_mp4_zip", args.toyota_mp4_zip))
    for name, path in required_paths:
        report.check(path is not None, f"{name} is set")
        if path is not None:
            report.check(os.path.exists(path), f"{name} exists", path)
    if report.failed:
        return report.finish()

    ds = build_dataset(args, synthetic_two_actor=False)
    sample = pick_sample(ds, args, report)
    if sample is None:
        return report.finish()
    idx, frames, target = sample
    validate_interaction_target(
        f"interaction_sample_{idx}", frames, target, args, report
    )

    synthetic_ds = build_dataset(args, synthetic_two_actor=True)
    sample = pick_sample(synthetic_ds, args, report)
    if sample is None:
        return report.finish()
    idx, frames, target = sample
    validate_interaction_target(
        f"synthetic_interaction_sample_{idx}", frames, target, args, report
    )

    return report.finish()


def main():
    args = build_parser().parse_args()
    action_name.taxonomy = args.toyota_action_taxonomy
    try:
        ok = run_smoke(args)
    except Exception:
        traceback.print_exc()
        return 1
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
