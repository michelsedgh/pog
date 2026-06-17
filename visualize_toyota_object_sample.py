#!/usr/bin/env python3
import argparse
import os
import sys
import tempfile
import traceback

import numpy as np
import torch
from PIL import Image, ImageDraw

from datasets.object_vocab import NONE_OBJECT_ID, OBJECT_CLASSES
from datasets.toyota_action_taxonomy import (
    TOYOTA_ACTION_TAXONOMIES,
    toyota_action_names,
    toyota_num_classes,
)
from datasets.toyotasm import ToyotaSMDataset


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Render Toyota actor interaction heatmap teacher overlays."
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
    parser.add_argument("--output_dir", default="toyota_interaction_visualizations")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--toyota_max_samples", type=int, default=256)
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
    parser.add_argument("--interaction_quality_min_actor_score", type=float, default=1.0)
    parser.add_argument("--interaction_quality_min_track_frames", type=int, default=1)
    parser.add_argument(
        "--interaction_quality_min_track_coverage", type=float, default=0.0
    )
    parser.add_argument(
        "--actions",
        default=None,
        help="Comma-separated Toyota action names to visualize, e.g. Uselaptop,Readbook.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--contact_sheet", type=int, default=1)
    parser.add_argument("--contact_cols", type=int, default=4)
    parser.add_argument(
        "--only_with_interactions",
        action="store_true",
        help="Skip samples without a strong-action interaction heatmap target.",
    )
    return parser


def dataset_kwargs(args):
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
    }
    if args.toyota_frame_source == "mp4_zip":
        kwargs["toyota_mp4_zip"] = args.toyota_mp4_zip
        kwargs["toyota_video_cache_dir"] = args.toyota_video_cache_dir
    return kwargs


def action_name(label):
    names = toyota_action_names("CS", getattr(action_name, "taxonomy", "toyota_31"))
    label = int(label)
    if 0 <= label < len(names):
        return names[label]
    return f"class_{int(label)}"


def denormalize_frame(frames, index):
    frame = frames[index : index + 1].detach().cpu() * STD + MEAN
    frame = frame.squeeze(0).permute(1, 2, 0).numpy()
    frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(frame)


def draw_normalized_box(draw, box, color, width=3):
    x1, y1, x2, y2 = [float(v) for v in box]
    draw.rectangle((x1 * 224, y1 * 224, x2 * 224, y2 * 224), outline=color, width=width)


def heatmap_overlay(image, heatmap, valid=None):
    if heatmap.ndim == 2:
        heatmap = heatmap.unsqueeze(0)
    if valid is None:
        valid = heatmap.flatten(-2).amax(dim=-1) > 0
    if not valid.any():
        return image
    hm = heatmap[valid].reshape(-1, heatmap.shape[-2], heatmap.shape[-1]).max(
        dim=0
    ).values
    hm = hm.detach().cpu().numpy()
    if float(hm.max()) <= 0:
        return image
    hm = (hm / max(float(hm.max()), 1e-6) * 180.0).astype(np.uint8)
    heat = Image.fromarray(hm, mode="L").resize(image.size, resample=Image.BILINEAR)
    red = Image.new("RGBA", image.size, (255, 40, 20, 0))
    red.putalpha(heat)
    return Image.alpha_composite(image.convert("RGBA"), red).convert("RGB")


def draw_overlay(frames, target, output_path, sample_idx):
    middle = frames.shape[0] // 2
    image = denormalize_frame(frames, middle)
    positive_valid = target.get(
        "interaction_heatmap_positive_valid",
        target["interaction_heatmap_valid"],
    ).bool()
    image = heatmap_overlay(
        image,
        target["interaction_heatmap"],
        positive_valid,
    )
    draw = ImageDraw.Draw(image)

    valid_actor_slots = torch.nonzero(target["valid"].bool(), as_tuple=False).flatten()
    for slot in valid_actor_slots.tolist():
        box = target["boxes"][slot].tolist()
        draw_normalized_box(draw, box, color=(40, 255, 80), width=3)
        label = action_name(int(target["actions"][slot]))
        x1, y1 = box[0] * 224, box[1] * 224
        suffix = ""
        if bool(target["interaction_valid"][slot]):
            cls_id = int(target["interaction_cls"][slot])
            cls_name = "NONE" if cls_id == NONE_OBJECT_ID else OBJECT_CLASSES[cls_id]
            suffix = f" -> {cls_name}"
        draw.text((x1 + 3, y1 + 3), f"actor{slot} {label}{suffix}", fill=(255, 255, 0))

    draw.rectangle((0, 0, 224, 30), fill=(0, 0, 0))
    draw.text((6, 6), f"sample {sample_idx}", fill=(255, 255, 0))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image.save(output_path)


def build_contact_sheet(paths, output_path, cols):
    if not paths:
        return
    thumb_w, thumb_h = 224, 224
    label_h = 18
    rows = (len(paths) + cols - 1) // cols
    sheet = Image.new(
        "RGB",
        (cols * thumb_w, rows * (thumb_h + label_h)),
        (20, 20, 20),
    )
    draw = ImageDraw.Draw(sheet)
    for idx, path in enumerate(paths):
        image = Image.open(path).convert("RGB")
        image.thumbnail((thumb_w, thumb_h))
        x = (idx % cols) * thumb_w
        y = (idx // cols) * (thumb_h + label_h)
        sheet.paste(image, (x, y + label_h))
        draw.text((x + 4, y + 3), os.path.basename(path), fill=(255, 255, 0))
    sheet.save(output_path, quality=95)


def main():
    args = build_parser().parse_args()
    action_name.taxonomy = args.toyota_action_taxonomy
    try:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        ds = ToyotaSMDataset(**dataset_kwargs(args))
        ds.setup()
        requested_actions = None
        if args.actions is not None and str(args.actions).strip():
            requested_actions = {
                item.strip()
                for item in str(args.actions).split(",")
                if item.strip()
            }

        written = 0
        output_paths = []
        idx = int(args.start_index)
        while idx < len(ds) and written < args.count:
            frames, target = ds[idx]
            action_names = [
                action_name(int(label))
                for label in target["actions"][target["valid"].bool()].tolist()
            ]
            if requested_actions is not None and not any(
                name in requested_actions for name in action_names
            ):
                idx += 1
                continue
            positive_valid = target.get(
                "interaction_heatmap_positive_valid",
                target["interaction_heatmap_valid"],
            ).bool()
            if args.only_with_interactions and not bool(positive_valid.any()):
                idx += 1
                continue
            output_path = os.path.join(args.output_dir, f"sample_{idx:05d}.jpg")
            draw_overlay(frames, target, output_path, idx)
            output_paths.append(output_path)
            print(output_path, flush=True)
            written += 1
            idx += 1
        if written == 0:
            raise RuntimeError("No overlays were written.")
        if args.contact_sheet:
            sheet_path = os.path.join(args.output_dir, "contact_sheet.jpg")
            build_contact_sheet(output_paths, sheet_path, max(1, int(args.contact_cols)))
            print(sheet_path, flush=True)
        return 0
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
