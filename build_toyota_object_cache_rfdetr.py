import argparse
import json
import os
import shutil
import sys
import tempfile
import zipfile

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import cv2
import numpy as np

from datasets.object_vocab import (
    DEFAULT_OBJECT_CLASS_THRESHOLDS,
    DETECTOR_TO_OBJECT,
    OBJECT_TO_ID,
    object_box_ignored_for_file_id,
    object_allowed_for_file_id,
    parse_object_camera_allowlist,
    parse_object_ignore_regions,
)
from datasets.toyotasm import ToyotaSMDataset


def build_parser():
    parser = argparse.ArgumentParser(
        description="Build a Toyota object detector JSONL cache with RF-DETR."
    )
    parser.add_argument("--data_dir", default=".")
    parser.add_argument(
        "--set_type",
        default="train",
        choices=["train", "val", "test", "all"],
        help="Toyota split to cache. Use all for train+val, plus test when test fraction > 0.",
    )
    parser.add_argument("--task_type", default="CS")
    parser.add_argument(
        "--toyota_frame_source",
        default="auto",
        choices=["auto", "frames", "mp4", "mp4_zip"],
        help="Toyota input source. Use frames to match the extracted-frame training path.",
    )
    parser.add_argument("--toyota_mp4_zip", default="toyota_smarthome_mp4.zip")
    parser.add_argument("--toyota_video_cache_dir", default=None)
    parser.add_argument("--toyota_frame_count_cache", default=None)
    parser.add_argument(
        "--toyota_split_source",
        default="auto",
        choices=["auto", "files"],
    )
    parser.add_argument("--toyota_val_fraction", type=float, default=0.15)
    parser.add_argument("--toyota_test_fraction", type=float, default=0.20)
    parser.add_argument("--toyota_max_samples", type=int, default=64)
    parser.add_argument("--toyota_seed", type=int, default=42)
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--n_frames", type=int, default=16)
    parser.add_argument("--file_id", action="append", default=None)
    parser.add_argument("--file_ids_txt", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--model_size",
        default="large",
        choices=["nano", "small", "medium", "base", "large", "xlarge", "2xlarge"],
    )
    parser.add_argument("--weights", default=None)
    parser.add_argument("--device", default=None, choices=[None, "cpu", "cuda", "mps"])
    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=0.25,
        help="Raw RF-DETR threshold before PO-GUISE object-class filtering.",
    )
    parser.add_argument(
        "--object_class_thresholds",
        default=None,
        help=(
            "Comma-separated PO-GUISE object thresholds, e.g. "
            "tv_monitor=0.75,phone=0.70. Defaults are conservative."
        ),
    )
    parser.add_argument(
        "--object_camera_allowlist",
        default=None,
        help=(
            "Semicolon-separated class camera allowlist, e.g. "
            "tv_monitor=c05,c06. Default keeps tv_monitor only in c05/c06. "
            "Use 'none' to disable view filtering."
        ),
    )
    parser.add_argument(
        "--object_ignore_regions",
        default=None,
        help=(
            "Semicolon-separated normalized camera ignore regions, e.g. "
            "c03=0,0,0.26,0.42. Default masks static c03 recording hardware. "
            "Use 'none' to disable region filtering."
        ),
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--frame_stride",
        type=int,
        default=1,
        help="Run detector every N frames. Use 1 for training caches.",
    )
    parser.add_argument(
        "--sampled_frames_only",
        type=int,
        default=0,
        help="Only cache the Toyota clip frame indices sampled by the dataset.",
    )
    parser.add_argument("--max_frames_per_clip", type=int, default=0)
    parser.add_argument("--resume", type=int, default=1)
    parser.add_argument("--limit_clips", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument(
        "--optimize_for_inference",
        type=int,
        default=1,
        help="Call RF-DETR optimize_for_inference() when the installed model exposes it.",
    )
    return parser


def selected_set_types(args):
    if args.set_type != "all":
        return [args.set_type]
    set_types = ["train", "val"]
    if float(args.toyota_test_fraction) > 0:
        set_types.append("test")
    return set_types


def load_file_ids(args):
    if args.file_id:
        return list(dict.fromkeys(args.file_id))

    if args.file_ids_txt:
        with open(args.file_ids_txt) as f:
            return [line.strip() for line in f if line.strip()]

    file_ids = []
    for set_type in selected_set_types(args):
        ds = ToyotaSMDataset(
            data_dir=args.data_dir,
            set_type=set_type,
            task_type=args.task_type,
            n_frames=16,
            n_frames_stride=1,
            n_landmarks=0,
            heatmap_agg=1,
            jitter_scales_min=256,
            jitter_scales_max=320,
            actor_prompt=0,
            object_prompt=0,
            toyota_frame_source=args.toyota_frame_source,
            toyota_mp4_zip=args.toyota_mp4_zip,
            toyota_video_cache_dir=args.toyota_video_cache_dir,
            toyota_frame_count_cache=args.toyota_frame_count_cache,
            toyota_split_source=args.toyota_split_source,
            toyota_val_fraction=args.toyota_val_fraction,
            toyota_test_fraction=args.toyota_test_fraction,
            toyota_max_samples=args.toyota_max_samples,
            toyota_seed=args.toyota_seed,
        )
        file_ids.extend(ds.data_df.file_id.tolist())
    return list(dict.fromkeys(file_ids))


def build_sampling_dataset(args, set_type):
    ds = ToyotaSMDataset(
        data_dir=args.data_dir,
        set_type=set_type,
        task_type=args.task_type,
        n_frames=args.n_frames,
        n_frames_stride=1,
        n_landmarks=0,
        heatmap_agg=1,
        jitter_scales_min=256,
        jitter_scales_max=320,
        actor_prompt=1,
        num_actor_tokens=8,
        object_prompt=0,
        toyota_frame_source=args.toyota_frame_source,
        toyota_mp4_zip=args.toyota_mp4_zip,
        toyota_video_cache_dir=args.toyota_video_cache_dir,
        toyota_frame_count_cache=args.toyota_frame_count_cache,
        toyota_split_source=args.toyota_split_source,
        toyota_val_fraction=args.toyota_val_fraction,
        toyota_test_fraction=args.toyota_test_fraction,
        toyota_max_samples=args.toyota_max_samples,
        toyota_seed=args.toyota_seed,
        toyota_synthetic_warmup_epochs=99,
        toyota_synthetic_two_actor_prob=0.0,
        toyota_synthetic_three_actor_prob=0.0,
    )
    ds.setup()
    return ds


def sampled_frame_indices(ds, idx):
    file_id = ds.data_df.iloc[idx].file_id
    n_frames = ds._num_frames(file_id)
    video_height, video_width = ds._video_size(file_id)
    pose_available = None
    if ds.actor_prompt and ds.needs_skeleton and ds.toyota_pose_guided_sampling:
        pose_available = ds._pose_available_by_frame(
            idx,
            n_frames,
            video_height,
            video_width,
        )

    if ds.set_type == "test":
        start_frame = ds.data_df.iloc[idx].start
        end_frame = ds.data_df.iloc[idx].end
        if end_frame < 0 or end_frame >= n_frames:
            end_frame = n_frames - 1
    elif n_frames > 128:
        max_start = max(0, n_frames - 129)
        if ds.set_type == "train":
            start_frame = None
            if pose_available is not None:
                start_frame = ds._sample_pose_guided_start(
                    0,
                    max_start,
                    pose_available,
                )
            if start_frame is None:
                start_frame = np.random.randint(0, n_frames - 128)
            end_frame = min(start_frame + 128, n_frames - 1)
        else:
            start_frame = None
            if pose_available is not None:
                start_frame = ds._sample_pose_guided_start(
                    0,
                    max_start,
                    pose_available,
                )
            if start_frame is None:
                start_frame = n_frames // 2 - 64
                end_frame = n_frames // 2 + 64
            else:
                end_frame = min(start_frame + 128, n_frames - 1)
    else:
        start_frame = 0
        end_frame = n_frames - 1

    frames_idx = ds._sample_frame_indices(
        start_frame,
        end_frame,
        pose_available=pose_available,
    )
    if len(frames_idx) < ds.n_frames:
        frames_idx = np.pad(frames_idx, (0, ds.n_frames - len(frames_idx)), "edge")
    return sorted(set(int(v) for v in frames_idx.tolist()))


def sampled_frame_map(args):
    np.random.seed(args.sample_seed)
    frame_map = {}
    for set_type in selected_set_types(args):
        ds = build_sampling_dataset(args, set_type)
        for idx, file_id in enumerate(ds.data_df.file_id.tolist()):
            frame_map[file_id] = set(sampled_frame_indices(ds, idx))
    return frame_map


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


def build_model(args):
    _ensure_transformers_backbone_api()
    from rfdetr import (
        RFDETRBase,
        RFDETRLarge,
        RFDETRMedium,
        RFDETRNano,
        RFDETRSmall,
    )

    model_classes = {
        "nano": RFDETRNano,
        "small": RFDETRSmall,
        "medium": RFDETRMedium,
        "base": RFDETRBase,
        "large": RFDETRLarge,
    }
    if args.model_size in {"xlarge", "2xlarge"}:
        try:
            from rfdetr_plus import RFDETR2XLarge, RFDETRXLarge
        except ImportError as exc:
            raise RuntimeError(
                f"RF-DETR {args.model_size} requires rfdetr_plus in a compatible env."
            ) from exc
        model_classes.update({"xlarge": RFDETRXLarge, "2xlarge": RFDETR2XLarge})

    kwargs = {}
    if args.weights:
        kwargs["pretrain_weights"] = args.weights
    if args.device:
        kwargs["device"] = args.device
    model = model_classes[args.model_size](**kwargs)
    if args.optimize_for_inference and hasattr(model, "optimize_for_inference"):
        optimized = model.optimize_for_inference(batch_size=args.batch_size)
        if optimized is not None:
            model = optimized
    return model


def _ensure_transformers_backbone_api():
    try:
        import transformers
    except ImportError:
        return

    if hasattr(transformers, "BackboneConfigMixin") and hasattr(
        transformers,
        "BackboneMixin",
    ):
        return

    try:
        from transformers.utils.backbone_utils import (
            BackboneConfigMixin,
            BackboneMixin,
        )
    except ImportError:
        return

    transformers.BackboneConfigMixin = BackboneConfigMixin
    transformers.BackboneMixin = BackboneMixin


def coco_classes():
    from rfdetr.util.coco_classes import COCO_CLASSES

    return {int(k): str(v) for k, v in COCO_CLASSES.items()}


def mapped_object_name(detector_name):
    detector_name = str(detector_name).strip().lower()
    return DETECTOR_TO_OBJECT.get(detector_name, detector_name)


def parse_object_class_thresholds(threshold_text):
    thresholds = dict(DEFAULT_OBJECT_CLASS_THRESHOLDS)
    if threshold_text is None or str(threshold_text).strip() == "":
        return thresholds

    for item in str(threshold_text).split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                f"Invalid object threshold '{item}'. Expected name=value."
            )
        name, value = [part.strip() for part in item.split("=", 1)]
        if name not in OBJECT_TO_ID:
            raise ValueError(f"Unknown object class in threshold override: {name}")
        threshold = float(value)
        if not 0 <= threshold <= 1:
            raise ValueError(f"Threshold for {name} must be in [0, 1]")
        thresholds[name] = threshold
    return thresholds


def detections_to_objects(
    detections,
    class_names,
    object_thresholds,
    object_camera_allowlist,
    object_ignore_regions,
    file_id,
    width,
    height,
):
    objects = []
    xyxy = getattr(detections, "xyxy", np.zeros((0, 4), dtype=np.float32))
    conf = getattr(detections, "confidence", None)
    class_id = getattr(detections, "class_id", None)
    if conf is None or class_id is None:
        return objects

    for box, score, det_cls_id in zip(xyxy, conf, class_id):
        detector_cls_id = int(det_cls_id)
        detector_name = class_names.get(detector_cls_id)
        if detector_name is None:
            continue
        object_name = mapped_object_name(detector_name)
        object_cls_id = OBJECT_TO_ID.get(object_name)
        if object_cls_id is None:
            continue
        if not object_allowed_for_file_id(
            object_name,
            file_id,
            object_camera_allowlist,
        ):
            continue
        if float(score) < object_thresholds[object_name]:
            continue
        x1, y1, x2, y2 = [float(v) for v in box.tolist()]
        x1 = max(0.0, min(float(width), x1))
        y1 = max(0.0, min(float(height), y1))
        x2 = max(0.0, min(float(width), x2))
        y2 = max(0.0, min(float(height), y2))
        if x2 <= x1 or y2 <= y1:
            continue
        if object_box_ignored_for_file_id(
            (x1, y1, x2, y2),
            file_id,
            width,
            height,
            object_ignore_regions,
        ):
            continue
        objects.append(
            {
                "cls": object_name,
                "cls_id": int(object_cls_id),
                "detector_cls": detector_name,
                "detector_cls_id": detector_cls_id,
                "conf": float(score),
                "xyxy": [x1, y1, x2, y2],
            }
        )
    return objects


def read_existing_keys(path):
    if not path or not os.path.exists(path):
        return set()
    keys = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            keys.add((record["file_id"], int(record["frame_idx"])))
    return keys


def flush_batch(
    model,
    class_names,
    object_thresholds,
    object_camera_allowlist,
    object_ignore_regions,
    batch,
    writer,
    args,
):
    if not batch:
        return 0, 0
    original_batch_len = len(batch)
    images = [item["image"] for item in batch]
    if args.optimize_for_inference and original_batch_len < args.batch_size:
        images.extend([images[-1]] * (args.batch_size - original_batch_len))
    detections = model.predict(images, threshold=args.conf_threshold)
    if not isinstance(detections, list):
        detections = [detections]
    detections = detections[:original_batch_len]

    written = 0
    objects_total = 0
    for item, det in zip(batch, detections):
        objects = detections_to_objects(
            det,
            class_names,
            object_thresholds,
            object_camera_allowlist,
            object_ignore_regions,
            item["file_id"],
            item["width"],
            item["height"],
        )
        objects_total += len(objects)
        writer.write(
            json.dumps(
                {
                    "file_id": item["file_id"],
                    "frame_idx": int(item["frame_idx"]),
                    "width": int(item["width"]),
                    "height": int(item["height"]),
                    "detector": "rfdetr",
                    "model": args.model_size,
                    "objects": objects,
                },
                separators=(",", ":"),
            )
            + "\n"
        )
        written += 1
    return written, objects_total


def iter_video_frames(video_path, frame_stride, max_frames):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_idx = 0
        yielded = 0
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if frame_idx % frame_stride == 0:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                yield frame_idx, frame_rgb, width, height
                yielded += 1
                if max_frames > 0 and yielded >= max_frames:
                    break
            frame_idx += 1
    finally:
        cap.release()


def iter_image_frames(frame_folder, frame_stride, max_frames):
    frame_files = sorted(
        name
        for name in os.listdir(frame_folder)
        if name.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if not frame_files:
        raise RuntimeError(f"No image frames found in {frame_folder}")

    yielded = 0
    for frame_idx, frame_name in enumerate(frame_files):
        if frame_idx % frame_stride != 0:
            continue
        frame_path = os.path.join(frame_folder, frame_name)
        frame_bgr = cv2.imread(frame_path, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            raise RuntimeError(f"Could not read frame image: {frame_path}")
        height, width = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        yield frame_idx, frame_rgb, width, height
        yielded += 1
        if max_frames > 0 and yielded >= max_frames:
            break


def iter_clip_frames(file_id, args, zip_index):
    frame_folder = os.path.join(args.data_dir, "frames", file_id)
    if args.toyota_frame_source in {"auto", "frames"} and os.path.isdir(
        frame_folder
    ):
        return iter_image_frames(
            frame_folder,
            frame_stride=args.frame_stride,
            max_frames=args.max_frames_per_clip,
        )
    if args.toyota_frame_source == "frames":
        raise FileNotFoundError(f"No extracted frame folder found for {file_id}")

    mp4_path = os.path.join(args.data_dir, "mp4", file_id + ".mp4")
    if args.toyota_frame_source in {"auto", "mp4"} and os.path.exists(mp4_path):
        return iter_video_frames(
            mp4_path,
            frame_stride=args.frame_stride,
            max_frames=args.max_frames_per_clip,
        )
    if args.toyota_frame_source == "mp4":
        raise FileNotFoundError(f"No mp4 found for {file_id}")

    if args.toyota_frame_source in {"auto", "mp4_zip"}:
        video_path = video_path_for_file_id(file_id, args, zip_index)
        return iter_video_frames(
            video_path,
            frame_stride=args.frame_stride,
            max_frames=args.max_frames_per_clip,
        )

    raise FileNotFoundError(f"No supported frame source found for {file_id}")


def build_cache(args):
    if args.frame_stride <= 0:
        raise ValueError("frame_stride must be positive")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if args.sampled_frames_only and args.frame_stride != 1:
        raise ValueError("sampled_frames_only requires frame_stride=1")
    if args.num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")

    selected_frames = sampled_frame_map(args) if args.sampled_frames_only else {}
    file_ids = list(selected_frames.keys()) if selected_frames else load_file_ids(args)
    file_ids = sorted(file_ids)
    if args.limit_clips > 0:
        file_ids = file_ids[: args.limit_clips]
    if args.num_shards > 1:
        file_ids = [
            file_id
            for pos, file_id in enumerate(file_ids)
            if pos % args.num_shards == args.shard_index
        ]
    if not file_ids:
        raise RuntimeError("No Toyota clips selected for object cache generation.")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    done = read_existing_keys(args.output) if args.resume else set()
    mode = "a" if args.resume else "w"
    zip_index = mp4_zip_index(args.toyota_mp4_zip)
    model = build_model(args)
    class_names = coco_classes()
    object_thresholds = parse_object_class_thresholds(args.object_class_thresholds)
    object_camera_allowlist = parse_object_camera_allowlist(
        args.object_camera_allowlist
    )
    object_ignore_regions = parse_object_ignore_regions(args.object_ignore_regions)
    print("Object class thresholds:", object_thresholds)
    print("Object camera allowlist:", object_camera_allowlist)
    print("Object ignore regions:", object_ignore_regions)
    print(
        f"Selected {len(file_ids)} clips for shard "
        f"{args.shard_index + 1}/{args.num_shards}."
    )

    total_frames = 0
    total_objects = 0
    with open(args.output, mode) as writer:
        for clip_pos, file_id in enumerate(file_ids, start=1):
            batch = []
            clip_frames = 0
            clip_objects = 0
            for frame_idx, image, width, height in iter_clip_frames(
                file_id,
                args,
                zip_index,
            ):
                if selected_frames and int(frame_idx) not in selected_frames[file_id]:
                    continue
                if (file_id, int(frame_idx)) in done:
                    continue
                batch.append(
                    {
                        "file_id": file_id,
                        "frame_idx": int(frame_idx),
                        "image": image,
                        "width": width,
                        "height": height,
                    }
                )
                if len(batch) >= args.batch_size:
                    written, objects = flush_batch(
                        model,
                        class_names,
                        object_thresholds,
                        object_camera_allowlist,
                        object_ignore_regions,
                        batch,
                        writer,
                        args,
                    )
                    clip_frames += written
                    clip_objects += objects
                    batch.clear()
            if batch:
                written, objects = flush_batch(
                    model,
                    class_names,
                    object_thresholds,
                    object_camera_allowlist,
                    object_ignore_regions,
                    batch,
                    writer,
                    args,
                )
                clip_frames += written
                clip_objects += objects

            total_frames += clip_frames
            total_objects += clip_objects
            print(
                f"[{clip_pos}/{len(file_ids)}] {file_id}: "
                f"wrote {clip_frames} frames, {clip_objects} mapped objects"
            )

    print(
        f"Done. Wrote {total_frames} frame records and {total_objects} mapped objects "
        f"to {args.output}"
    )


def main():
    args = build_parser().parse_args()
    try:
        build_cache(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
