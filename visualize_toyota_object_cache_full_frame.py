import argparse
import json
import os
import shutil
import sys
import tempfile
import zipfile
from collections import defaultdict

import cv2
from PIL import Image, ImageDraw

from datasets.object_vocab import OBJECT_CLASSES


def build_parser():
    parser = argparse.ArgumentParser(
        description="Render full-frame Toyota object detector cache overlays."
    )
    parser.add_argument("--data_dir", default=".")
    parser.add_argument("--toyota_mp4_zip", default="toyota_smarthome_mp4.zip")
    parser.add_argument("--toyota_video_cache_dir", default=None)
    parser.add_argument("--object_detector_cache", required=True)
    parser.add_argument("--output_dir", default="toyota_object_cache_full_frame")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument(
        "--select",
        default="max_objects",
        choices=["max_objects", "middle", "first"],
        help="Which cached frame to draw for each clip.",
    )
    parser.add_argument("--max_objects_draw", type=int, default=32)
    parser.add_argument("--contact_sheet", type=int, default=1)
    parser.add_argument("--contact_cols", type=int, default=3)
    return parser


def load_cache_records(path):
    by_file = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            by_file[record["file_id"]].append(record)
    for records in by_file.values():
        records.sort(key=lambda item: int(item["frame_idx"]))
    return dict(by_file)


def choose_record(records, mode):
    if mode == "first":
        return records[0]
    if mode == "middle":
        return records[len(records) // 2]
    if mode == "max_objects":
        return max(records, key=lambda item: len(item.get("objects", [])))
    raise ValueError(f"Unsupported select mode: {mode}")


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
        tempfile.gettempdir(), "poguise_toyota_mp4_cache"
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


def draw_overlay(image, record, max_objects_draw):
    draw = ImageDraw.Draw(image)
    frame_text = f"{record['file_id']} frame {int(record['frame_idx'])}"
    draw_label(draw, (6, 6), frame_text, (255, 255, 0))

    objects = sorted(
        record.get("objects", []),
        key=lambda item: float(item.get("conf", 0.0)),
        reverse=True,
    )[:max_objects_draw]
    for obj in objects:
        x1, y1, x2, y2 = [float(v) for v in obj["xyxy"]]
        cls_id = int(obj.get("cls_id", -1))
        cls_name = OBJECT_CLASSES.get(cls_id, obj.get("cls", f"obj{cls_id}"))
        detector_cls = obj.get("detector_cls", cls_name)
        conf = float(obj.get("conf", 0.0))
        draw.rectangle((x1, y1, x2, y2), outline=(80, 180, 255), width=3)
        label = f"{cls_name} {conf:.2f}"
        if detector_cls != cls_name:
            label += f" [{detector_cls}]"
        draw_label(draw, (x1 + 3, max(18, y1 + 3)), label, (255, 255, 255))
    return image


def build_contact_sheet(paths, output_path, cols):
    if not paths:
        return
    thumb_w, thumb_h = 320, 240
    label_h = 18
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
    sheet.save(output_path)


def main():
    args = build_parser().parse_args()
    try:
        by_file = load_cache_records(args.object_detector_cache)
        if not by_file:
            raise RuntimeError("Object detector cache is empty.")
        zip_index = mp4_zip_index(args.toyota_mp4_zip)
        os.makedirs(args.output_dir, exist_ok=True)

        selected = []
        file_ids = sorted(by_file.keys())
        for file_id in file_ids[int(args.start_index) :]:
            if len(selected) >= args.count:
                break
            record = choose_record(by_file[file_id], args.select)
            video_path = video_path_for_file_id(file_id, args, zip_index)
            image = read_frame(video_path, int(record["frame_idx"]))
            image = draw_overlay(image, record, args.max_objects_draw)
            output_path = os.path.join(
                args.output_dir,
                f"{file_id}_f{int(record['frame_idx']):06d}.jpg",
            )
            image.save(output_path, quality=95)
            selected.append(output_path)
            print(output_path)

        if args.contact_sheet:
            sheet_path = os.path.join(args.output_dir, "contact_sheet.jpg")
            build_contact_sheet(selected, sheet_path, args.contact_cols)
            print(sheet_path)
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
