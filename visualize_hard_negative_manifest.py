import argparse
import json
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import cv2
from PIL import Image, ImageDraw

from datasets.object_vocab import OBJECT_CLASSES


def build_parser():
    parser = argparse.ArgumentParser(
        description="Render full-frame overlays for mined hard-negative manifest entries."
    )
    parser.add_argument("--manifest_json", required=True)
    parser.add_argument("--object_detector_cache", required=True)
    parser.add_argument("--data_dir", default=".")
    parser.add_argument("--toyota_mp4_zip", default="toyota_smarthome_mp4.zip")
    parser.add_argument("--toyota_video_cache_dir", default=None)
    parser.add_argument("--output_dir", default="hard_negative_visualizations")
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--bucket", action="append", default=None)
    parser.add_argument("--action", action="append", default=None)
    parser.add_argument("--max_objects_draw", type=int, default=32)
    parser.add_argument("--contact_sheet", type=int, default=1)
    parser.add_argument("--contact_cols", type=int, default=4)
    return parser


def mp4_zip_index(zip_path):
    if not zip_path or not os.path.exists(zip_path):
        return {}
    with zipfile.ZipFile(zip_path) as zf:
        return {
            os.path.splitext(os.path.basename(name))[0]: name
            for name in zf.namelist()
            if name.lower().endswith(".mp4")
        }


def video_path_for_file_id(file_id, args, zip_index):
    mp4_path = os.path.join(args.data_dir, "mp4", file_id + ".mp4")
    if os.path.exists(mp4_path):
        return mp4_path

    zip_name = zip_index.get(file_id)
    if zip_name is None:
        raise FileNotFoundError(f"No mp4 found for {file_id}")

    cache_dir = args.toyota_video_cache_dir or os.path.join(
        tempfile.gettempdir(),
        "poguise_toyota_mp4_cache",
    )
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, file_id + ".mp4")
    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
        return cache_path

    tmp_path = f"{cache_path}.{os.getpid()}.tmp"
    try:
        with zipfile.ZipFile(args.toyota_mp4_zip) as zf:
            with zf.open(zip_name) as src, open(tmp_path, "wb") as dst:
                shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
        os.replace(tmp_path, cache_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    return cache_path


def read_frame(video_path, frame_idx):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame_bgr = cap.read()
        if not ok:
            raise RuntimeError(f"Could not read frame {frame_idx} from {video_path}")
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)
    finally:
        cap.release()


def draw_label(draw, xy, text, fill):
    x, y = xy
    bbox = draw.textbbox((x, y), text)
    pad = 2
    draw.rectangle(
        (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad),
        fill=(0, 0, 0),
    )
    draw.text((x, y), text, fill=fill)


def draw_overlay(image, record, entry, max_objects_draw):
    draw = ImageDraw.Draw(image)
    title = (
        f"{entry['action']} | distractors={','.join(entry['distractors'])} | "
        f"q={float(entry['quality']):.3f}"
    )
    draw_label(draw, (6, 6), title, (255, 255, 0))
    draw_label(
        draw,
        (6, 24),
        f"{record['file_id']} frame {int(record['frame_idx'])}",
        (255, 255, 0),
    )

    distractors = set(entry["distractors"])
    objects = sorted(
        record.get("objects", []),
        key=lambda item: (
            str(item.get("cls")) not in distractors,
            -float(item.get("conf", 0.0)),
        ),
    )[:max_objects_draw]
    for obj in objects:
        x1, y1, x2, y2 = [float(value) for value in obj["xyxy"]]
        cls_id = int(obj.get("cls_id", -1))
        cls_name = OBJECT_CLASSES.get(cls_id, obj.get("cls", f"obj{cls_id}"))
        detector_cls = obj.get("detector_cls", cls_name)
        conf = float(obj.get("conf", 0.0))
        is_distractor = cls_name in distractors
        color = (255, 60, 60) if is_distractor else (80, 180, 255)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=4 if is_distractor else 2)
        label = f"{cls_name} {conf:.2f}"
        if detector_cls != cls_name:
            label += f" [{detector_cls}]"
        draw_label(draw, (x1 + 3, max(42, y1 + 3)), label, (255, 255, 255))
    return image


def build_contact_sheet(paths, output_path, cols):
    if not paths:
        return
    thumb_w, thumb_h = 320, 240
    label_h = 20
    rows = (len(paths) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + label_h)), (20, 20, 20))
    draw = ImageDraw.Draw(sheet)
    for idx, path in enumerate(paths):
        image = Image.open(path).convert("RGB")
        image.thumbnail((thumb_w, thumb_h))
        x = (idx % cols) * thumb_w
        y = (idx // cols) * (thumb_h + label_h)
        sheet.paste(image, (x, y + label_h))
        draw.text((x + 4, y + 3), os.path.basename(path), fill=(255, 255, 0))
    sheet.save(output_path, quality=95)


def select_entries(args):
    entries = json.loads(Path(args.manifest_json).read_text())
    selected = []
    bucket_filter = set(args.bucket or [])
    action_filter = set(args.action or [])
    for entry in entries:
        if action_filter and entry["action"] not in action_filter:
            continue
        if bucket_filter:
            buckets = {f"{entry['action']}__{name}" for name in entry["distractors"]}
            if not buckets.intersection(bucket_filter):
                continue
        selected.append(entry)
        if len(selected) >= args.count:
            break
    return selected


def load_selected_records(cache_path, entries):
    wanted = {
        (entry["file_id"], int(entry["best_frame_idx"])): entry
        for entry in entries
    }
    records = {}
    with open(cache_path) as fh:
        for raw in fh:
            rec = json.loads(raw)
            key = (rec["file_id"], int(rec["frame_idx"]))
            if key in wanted:
                records[key] = rec
                if len(records) == len(wanted):
                    break
    missing = [key for key in wanted if key not in records]
    if missing:
        raise RuntimeError(f"Missing selected cache records: {missing[:10]}")
    return records


def main():
    args = build_parser().parse_args()
    entries = select_entries(args)
    if not entries:
        raise RuntimeError("No manifest entries matched the requested filters.")
    records = load_selected_records(args.object_detector_cache, entries)
    zip_index = mp4_zip_index(args.toyota_mp4_zip)
    os.makedirs(args.output_dir, exist_ok=True)

    paths = []
    for idx, entry in enumerate(entries):
        key = (entry["file_id"], int(entry["best_frame_idx"]))
        record = records[key]
        video_path = video_path_for_file_id(entry["file_id"], args, zip_index)
        image = read_frame(video_path, int(record["frame_idx"]))
        image = draw_overlay(image, record, entry, args.max_objects_draw)
        output_path = os.path.join(
            args.output_dir,
            f"{idx:03d}_{entry['action']}_{entry['file_id']}_f{int(record['frame_idx']):06d}.jpg",
        )
        image.save(output_path, quality=95)
        print(output_path)
        paths.append(output_path)

    if args.contact_sheet:
        sheet_path = os.path.join(args.output_dir, "contact_sheet.jpg")
        build_contact_sheet(paths, sheet_path, args.contact_cols)
        print(sheet_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
