import concurrent.futures
import json
import os
import shutil
import sys
import threading
import time
from collections import defaultdict

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import cv2
import numpy as np

import build_toyota_object_cache_rfdetr as cache_base

_THREAD_STATE = threading.local()
_MODEL_INIT_LOCK = threading.Lock()


def build_parser():
    parser = cache_base.build_parser()
    parser.description = (
        "Build/resume a sharded Toyota RF-DETR object cache with a single "
        "ONNX Runtime TensorRT GPU worker and threaded CPU JPEG preprocessing."
    )
    parser.add_argument(
        "--shard_output_dir",
        required=True,
        help="Directory containing/writing part-XXX-of-NNN.jsonl shard files.",
    )
    parser.add_argument(
        "--decode_workers",
        type=int,
        default=max(1, (os.cpu_count() or 4) - 2),
        help="CPU threads for JPEG decode and resize/normalize preprocessing.",
    )
    parser.add_argument(
        "--prefetch_batches",
        type=int,
        default=4,
        help="Maximum queued preprocessed batches before GPU inference catches up.",
    )
    parser.add_argument(
        "--inference_workers",
        type=int,
        default=4,
        help="TensorRT/ONNX Runtime sessions to run concurrently.",
    )
    parser.add_argument(
        "--inference_queue_batches",
        type=int,
        default=2,
        help="Maximum queued GPU batches per inference worker.",
    )
    parser.add_argument(
        "--log_interval_frames",
        type=int,
        default=5000,
        help="Print progress after this many newly written frames.",
    )
    parser.add_argument(
        "--merge_when_done",
        type=int,
        default=1,
        help="Merge shards into --output after all missing frames are written.",
    )
    return parser


def onnx_input_hw(onnx_model_path):
    import onnx

    model = onnx.load(onnx_model_path)
    if not model.graph.input:
        raise ValueError(f"ONNX model has no inputs: {onnx_model_path}")
    input_type = model.graph.input[0].type.tensor_type
    dims = [dim.dim_value for dim in input_type.shape.dim]
    if len(dims) != 4 or dims[2] <= 0 or dims[3] <= 0:
        raise ValueError(
            "ONNX model must have static [B,C,H,W] input shape, "
            f"got {dims} from {onnx_model_path}"
        )
    return int(dims[2]), int(dims[3])


def get_thread_model(args):
    model = getattr(_THREAD_STATE, "model", None)
    if model is not None:
        return model
    with _MODEL_INIT_LOCK:
        model = cache_base.build_model(args)
    if not hasattr(model, "predict_preprocessed"):
        raise RuntimeError("Selected detector does not support predict_preprocessed().")
    _THREAD_STATE.model = model
    return model


def shard_path(shard_output_dir, shard_idx, num_shards):
    return os.path.join(
        shard_output_dir,
        f"part-{shard_idx:03d}-of-{num_shards:03d}.jsonl",
    )


def load_frame_count_cache(path):
    if not path or not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    return {str(k): int(v) for k, v in data.items()}


def count_frame_files(frame_folder):
    return sum(
        1
        for name in os.listdir(frame_folder)
        if name.lower().endswith((".jpg", ".jpeg", ".png"))
    )


def selected_frame_count(file_id, args, frame_counts):
    n_frames = frame_counts.get(file_id)
    if n_frames is None:
        frame_folder = os.path.join(args.data_dir, "frames", file_id)
        n_frames = count_frame_files(frame_folder)
    count = (int(n_frames) + int(args.frame_stride) - 1) // int(args.frame_stride)
    if args.max_frames_per_clip > 0:
        count = min(count, int(args.max_frames_per_clip))
    return count


def format_duration(seconds):
    if seconds is None or not np.isfinite(seconds) or seconds < 0:
        return "unknown"
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def validate_args(args):
    if args.inference_backend != "onnxruntime_trt":
        raise ValueError(
            "The producer/consumer pipeline is intentionally TensorRT-only. "
            "Pass --inference_backend onnxruntime_trt."
        )
    if args.toyota_frame_source != "frames":
        raise ValueError(
            "The producer/consumer pipeline is intentionally frame-folder-only. "
            "Pass --toyota_frame_source frames."
        )
    if args.sampled_frames_only:
        raise ValueError("This pipeline is for full-frame caches; sampled_frames_only must be 0.")
    if args.shard_index != 0:
        raise ValueError(
            "This pipeline owns all shards in one process. Leave --shard_index 0."
        )
    if args.num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if args.frame_stride <= 0:
        raise ValueError("frame_stride must be positive")
    if args.decode_workers <= 0:
        raise ValueError("decode_workers must be positive")
    if args.prefetch_batches <= 0:
        raise ValueError("prefetch_batches must be positive")
    if args.inference_workers <= 0:
        raise ValueError("inference_workers must be positive")
    if args.inference_queue_batches <= 0:
        raise ValueError("inference_queue_batches must be positive")


def select_file_ids(args):
    file_ids = sorted(cache_base.load_file_ids(args))
    if args.limit_clips > 0:
        file_ids = file_ids[: args.limit_clips]
    if not file_ids:
        raise RuntimeError("No Toyota clips selected for object cache generation.")
    return file_ids


def shard_file_ids(file_ids, num_shards):
    by_shard = defaultdict(list)
    for pos, file_id in enumerate(file_ids):
        by_shard[pos % num_shards].append(file_id)
    return by_shard


def read_done_by_shard(args):
    os.makedirs(args.shard_output_dir, exist_ok=True)
    done_by_shard = {}
    for shard_idx in range(args.num_shards):
        path = shard_path(args.shard_output_dir, shard_idx, args.num_shards)
        done_by_shard[shard_idx] = (
            cache_base.read_existing_keys(path) if args.resume else set()
        )
    return done_by_shard


def iter_frame_paths_for_clip(file_id, args):
    frame_folder = os.path.join(args.data_dir, "frames", file_id)
    if not os.path.isdir(frame_folder):
        raise FileNotFoundError(f"No extracted frame folder found for {file_id}")
    frame_files = sorted(
        name
        for name in os.listdir(frame_folder)
        if name.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if not frame_files:
        raise RuntimeError(f"No image frames found in {frame_folder}")

    yielded = 0
    for frame_idx, frame_name in enumerate(frame_files):
        if frame_idx % args.frame_stride != 0:
            continue
        if args.max_frames_per_clip > 0 and yielded >= args.max_frames_per_clip:
            break
        yielded += 1
        yield frame_idx, os.path.join(frame_folder, frame_name)


def iter_missing_frame_jobs(args, file_ids, done_by_shard):
    for pos, file_id in enumerate(file_ids):
        shard_idx = pos % args.num_shards
        done = done_by_shard[shard_idx]
        for frame_idx, frame_path in iter_frame_paths_for_clip(file_id, args):
            key = (file_id, int(frame_idx))
            if key in done:
                continue
            yield {
                "shard_idx": shard_idx,
                "file_id": file_id,
                "frame_idx": int(frame_idx),
                "frame_path": frame_path,
            }


def preprocess_frame(job, target_width, target_height, mean, std):
    frame_bgr = cv2.imread(job["frame_path"], cv2.IMREAD_COLOR)
    if frame_bgr is None:
        raise RuntimeError(f"Could not read frame image: {job['frame_path']}")
    height, width = frame_bgr.shape[:2]
    resized = cv2.resize(
        frame_bgr,
        (target_width, target_height),
        interpolation=cv2.INTER_LINEAR,
    )
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    array = rgb.astype(np.float32) / 255.0
    array = (array - mean) / std
    chw = np.transpose(array, (2, 0, 1)).astype(np.float32, copy=False)
    return {
        "shard_idx": job["shard_idx"],
        "file_id": job["file_id"],
        "frame_idx": job["frame_idx"],
        "width": int(width),
        "height": int(height),
        "input": chw,
    }


def detection_record_line(item, objects, args):
    return (
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


def infer_write_preprocessed_batch(
    args,
    class_names,
    object_thresholds,
    object_camera_allowlist,
    object_ignore_regions,
    writers,
    writer_locks,
    batch_items,
):
    if not batch_items:
        return 0, 0, {}
    model = get_thread_model(args)
    model_inputs = np.stack([item["input"] for item in batch_items]).astype(
        np.float32,
        copy=False,
    )
    image_shapes = [(item["height"], item["width"]) for item in batch_items]
    detections = model.predict_preprocessed(
        model_inputs,
        image_shapes,
        threshold=args.conf_threshold,
    )
    written = 0
    objects_total = 0
    lines_by_shard = defaultdict(list)
    keys_by_shard = defaultdict(list)
    for item, det in zip(batch_items, detections):
        objects = cache_base.detections_to_objects(
            det,
            class_names,
            object_thresholds,
            object_camera_allowlist,
            object_ignore_regions,
            item["file_id"],
            item["width"],
            item["height"],
        )
        shard_idx = item["shard_idx"]
        lines_by_shard[shard_idx].append(detection_record_line(item, objects, args))
        keys_by_shard[shard_idx].append((item["file_id"], int(item["frame_idx"])))
        objects_total += len(objects)
        written += 1

    for shard_idx, lines in lines_by_shard.items():
        with writer_locks[shard_idx]:
            writers[shard_idx].writelines(lines)
    return written, objects_total, dict(keys_by_shard)


def print_progress(written, total_missing, objects_total, started_at, prefix="pipeline"):
    elapsed = max(time.time() - started_at, 1e-6)
    fps = written / elapsed
    remaining = max(total_missing - written, 0)
    eta = remaining / fps if fps > 0 else None
    pct = 100.0 * written / max(total_missing, 1)
    print(
        f"{prefix}: {written:,}/{total_missing:,} new frames ({pct:.2f}%), "
        f"{fps:.2f} fps, {objects_total:,} objects, ETA {format_duration(eta)}",
        flush=True,
    )


def merge_shards(args):
    tmp_path = args.output + f".{os.getpid()}.tmp"
    with open(tmp_path, "w") as out_fh:
        for shard_idx in range(args.num_shards):
            path = shard_path(args.shard_output_dir, shard_idx, args.num_shards)
            if not os.path.exists(path) or os.path.getsize(path) == 0:
                raise RuntimeError(f"Missing or empty shard: {path}")
            with open(path) as in_fh:
                shutil.copyfileobj(in_fh, out_fh, length=8 * 1024 * 1024)
    os.replace(tmp_path, args.output)
    print(f"Merged object cache: {args.output}", flush=True)


def build_cache_pipeline(args):
    validate_args(args)
    cv2.setNumThreads(1)

    file_ids = select_file_ids(args)
    by_shard = shard_file_ids(file_ids, args.num_shards)
    done_by_shard = read_done_by_shard(args)
    frame_counts = load_frame_count_cache(args.toyota_frame_count_cache)
    total_selected = sum(
        selected_frame_count(file_id, args, frame_counts) for file_id in file_ids
    )
    total_done = sum(len(done_by_shard[idx]) for idx in range(args.num_shards))
    total_missing = max(total_selected - total_done, 0)

    print(f"Selected {len(file_ids)} clips across {args.num_shards} shards.")
    for shard_idx in range(args.num_shards):
        print(
            f"shard {shard_idx:03d}: {len(by_shard[shard_idx])} clips, "
            f"{len(done_by_shard[shard_idx]):,} existing frame records"
        )
    print(
        f"Frame records: {total_done:,} existing / {total_selected:,} selected; "
        f"{total_missing:,} missing."
    )
    if total_missing == 0:
        print("No missing frames to process.")
        if args.merge_when_done:
            merge_shards(args)
        return

    input_height, input_width = onnx_input_hw(args.onnx_model_path)
    class_names = cache_base.coco_classes()
    object_thresholds = cache_base.parse_object_class_thresholds(
        args.object_class_thresholds
    )
    object_camera_allowlist = cache_base.parse_object_camera_allowlist(
        args.object_camera_allowlist
    )
    object_ignore_regions = cache_base.parse_object_ignore_regions(
        args.object_ignore_regions
    )
    print("Object class thresholds:", object_thresholds)
    print("Object camera allowlist:", object_camera_allowlist)
    print("Object ignore regions:", object_ignore_regions)
    print(
        f"Pipeline workers: decode={args.decode_workers}, "
        f"inference={args.inference_workers}, batch={args.batch_size}, "
        f"input={input_height}x{input_width}"
    )

    writers = {}
    writer_locks = {}
    try:
        for shard_idx in range(args.num_shards):
            path = shard_path(args.shard_output_dir, shard_idx, args.num_shards)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            writers[shard_idx] = open(path, "a" if args.resume else "w", buffering=1)
            writer_locks[shard_idx] = threading.Lock()

        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        max_inflight = max(args.batch_size, args.batch_size * args.prefetch_batches)
        max_inflight_inference = max(
            args.inference_workers,
            args.inference_workers * args.inference_queue_batches,
        )
        job_iter = iter_missing_frame_jobs(args, file_ids, done_by_shard)
        decode_futures = {}
        inference_futures = set()
        batch_items = []
        written_total = 0
        objects_total = 0
        next_log = int(args.log_interval_frames)
        started_at = time.time()

        def submit_until_full(executor):
            while len(decode_futures) < max_inflight:
                try:
                    job = next(job_iter)
                except StopIteration:
                    return
                future = executor.submit(
                    preprocess_frame,
                    job,
                    input_width,
                    input_height,
                    mean,
                    std,
                )
                decode_futures[future] = job

        def collect_inference(wait_for_one):
            nonlocal written_total, objects_total, next_log
            if not inference_futures:
                return
            if wait_for_one:
                done_futures, _ = concurrent.futures.wait(
                    list(inference_futures),
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
            else:
                done_futures = {
                    future for future in inference_futures if future.done()
                }
            for future in done_futures:
                inference_futures.remove(future)
                written, objects, keys_by_shard = future.result()
                written_total += written
                objects_total += objects
                for shard_idx, keys in keys_by_shard.items():
                    done_by_shard[shard_idx].update(keys)
                if written_total >= next_log:
                    print_progress(
                        written_total,
                        total_missing,
                        objects_total,
                        started_at,
                    )
                    next_log += int(args.log_interval_frames)

        def submit_inference(executor, items):
            while len(inference_futures) >= max_inflight_inference:
                collect_inference(wait_for_one=True)
            inference_futures.add(
                executor.submit(
                    infer_write_preprocessed_batch,
                    args,
                    class_names,
                    object_thresholds,
                    object_camera_allowlist,
                    object_ignore_regions,
                    writers,
                    writer_locks,
                    items,
                )
            )

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.decode_workers
        ) as decode_executor, concurrent.futures.ThreadPoolExecutor(
            max_workers=args.inference_workers
        ) as inference_executor:
            submit_until_full(decode_executor)
            while decode_futures:
                done_futures, _ = concurrent.futures.wait(
                    list(decode_futures),
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for future in done_futures:
                    decode_futures.pop(future)
                    batch_items.append(future.result())
                    if len(batch_items) >= args.batch_size:
                        submit_inference(inference_executor, batch_items)
                        batch_items = []
                collect_inference(wait_for_one=False)
                submit_until_full(decode_executor)

            if batch_items:
                submit_inference(inference_executor, batch_items)
                batch_items = []

            while inference_futures:
                collect_inference(wait_for_one=True)

        for writer in writers.values():
            writer.flush()
        print_progress(written_total, total_missing, objects_total, started_at)
    finally:
        for writer in writers.values():
            writer.close()

    if args.merge_when_done:
        merge_shards(args)


def main():
    args = build_parser().parse_args()
    try:
        build_cache_pipeline(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
