import argparse
import os
import sys
import tempfile
import traceback

import numpy as np
import torch
from PIL import Image, ImageDraw

from datasets.object_vocab import NONE_OBJECT_ID, NUM_OBJECT_CLASSES, OBJECT_CLASSES
from datasets.toyotasm import CS_DICT, ToyotaSMDataset


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Render Toyota object-prompt dataset overlays."
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
    parser.add_argument("--output_dir", default="toyota_object_visualizations")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--toyota_max_samples", type=int, default=256)
    parser.add_argument("--n_frames", type=int, default=16)
    parser.add_argument("--n_landmarks", type=int, default=13)
    parser.add_argument("--num_actor_tokens", type=int, default=8)
    parser.add_argument("--num_object_tokens", type=int, default=24)
    parser.add_argument("--num_object_classes", type=int, default=NUM_OBJECT_CLASSES)
    parser.add_argument("--object_conf_threshold", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--synthetic_two_actor", action="store_true")
    parser.add_argument("--contact_sheet", type=int, default=1)
    parser.add_argument("--contact_cols", type=int, default=4)
    parser.add_argument(
        "--only_with_objects",
        action="store_true",
        help="Skip samples without valid object tokens.",
    )
    return parser


def dataset_kwargs(args):
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
        "toyota_synthetic_warmup_epochs": 0 if args.synthetic_two_actor else 99,
        "toyota_synthetic_two_actor_prob": 1.0 if args.synthetic_two_actor else 0.0,
        "toyota_synthetic_three_actor_prob": 0.0,
        "toyota_synthetic_same_class_prob": 0.0,
    }


def action_name(label):
    action_id = int(label) + 1
    for name, idx in CS_DICT.items():
        if int(idx) == action_id:
            return name
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
    if valid is None:
        valid = heatmap.flatten(1).amax(dim=1) > 0
    if not valid.any():
        return image
    hm = heatmap[valid].max(dim=0).values
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
    if "interaction_heatmap" in target:
        image = heatmap_overlay(
            image,
            target["interaction_heatmap"],
            target.get("interaction_heatmap_valid", None).bool()
            if "interaction_heatmap_valid" in target
            else None,
        )
    draw = ImageDraw.Draw(image)

    valid_actor_slots = torch.nonzero(target["valid"].bool(), as_tuple=False).flatten()
    for slot in valid_actor_slots.tolist():
        box = target["boxes"][slot].tolist()
        draw_normalized_box(draw, box, color=(40, 255, 80), width=3)
        label = action_name(int(target["actions"][slot]))
        x1, y1 = box[0] * 224, box[1] * 224
        draw.text((x1 + 3, y1 + 3), f"actor{slot} {label}", fill=(255, 255, 0))

    valid_object_slots = torch.nonzero(
        target["object_valid"].bool(), as_tuple=False
    ).flatten()
    for slot in valid_object_slots.tolist():
        box = target["object_boxes"][slot].tolist()
        cls_id = int(target["object_cls"][slot])
        cls_name = OBJECT_CLASSES.get(cls_id, f"obj{cls_id}")
        conf = float(target["object_conf"][slot])
        draw_normalized_box(draw, box, color=(80, 160, 255), width=2)
        x1, y1 = box[0] * 224, box[1] * 224
        draw.text((x1 + 3, y1 + 14), f"{cls_name} {conf:.2f}", fill=(255, 255, 255))

    y = 6
    draw.rectangle((0, 0, 224, 42), fill=(0, 0, 0))
    draw.text((6, y), f"sample {sample_idx}", fill=(255, 255, 0))
    y += 12
    interactions = []
    for slot in valid_actor_slots.tolist():
        if bool(target["interaction_valid"][slot]):
            cls_id = int(target["interaction_cls"][slot])
            cls_name = "NONE" if cls_id == NONE_OBJECT_ID else OBJECT_CLASSES[cls_id]
            interactions.append(f"a{slot}:{cls_name}")
    draw.text(
        (6, y),
        "interaction " + (", ".join(interactions) if interactions else "ignored"),
        fill=(255, 255, 0),
    )

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
    try:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        ds = ToyotaSMDataset(**dataset_kwargs(args))
        ds.setup()

        written = 0
        output_paths = []
        idx = int(args.start_index)
        while idx < len(ds) and written < args.count:
            frames, target = ds[idx]
            if args.only_with_objects and not bool(target["object_valid"].any()):
                idx += 1
                continue
            output_path = os.path.join(args.output_dir, f"sample_{idx:05d}.jpg")
            draw_overlay(frames, target, output_path, idx)
            output_paths.append(output_path)
            print(output_path)
            written += 1
            idx += 1
        if written == 0:
            raise RuntimeError("No overlays were written.")
        if args.contact_sheet:
            sheet_path = os.path.join(args.output_dir, "contact_sheet.jpg")
            build_contact_sheet(output_paths, sheet_path, max(1, int(args.contact_cols)))
            print(sheet_path)
        return 0
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
