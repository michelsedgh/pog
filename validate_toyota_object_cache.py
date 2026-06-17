import argparse
import json
import math
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

from build_toyota_object_cache_rfdetr import mapped_object_name
from datasets.object_vocab import (
    DEFAULT_OBJECT_CLASS_THRESHOLDS,
    NONE_OBJECT_ID,
    OBJECT_CLASSES,
    object_allowed_for_file_id,
    object_box_ignored_for_file_id,
    parse_object_camera_allowlist,
    parse_object_ignore_regions,
)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Validate a Toyota RF-DETR object JSONL cache against frame tar indexes."
    )
    parser.add_argument("--cache", required=True)
    parser.add_argument("--tar_index", action="append", required=True)
    parser.add_argument(
        "--object_camera_allowlist",
        default=None,
        help=(
            "Optional class camera allowlist, e.g. tv_monitor=c05,c06. "
            "Default disables view filtering."
        ),
    )
    parser.add_argument(
        "--object_ignore_regions",
        default=None,
        help=(
            "Optional normalized camera ignore regions, e.g. "
            "c03=0,0,0.26,0.42. Default disables region filtering."
        ),
    )
    parser.add_argument("--progress_seconds", type=float, default=30.0)
    parser.add_argument("--max_examples", type=int, default=20)
    return parser


def load_expected_counts(tar_indexes):
    expected_counts = {}
    part_counts = []
    for db_path in tar_indexes:
        con = sqlite3.connect(str(db_path))
        try:
            rows = con.execute(
                """
                select substr(path, 9) as file_id, count(*)
                from files
                where isgenerated = 0 and lower(name) like '%.jpg'
                group by path
                """
            )
            part_frames = 0
            part_clips = 0
            for file_id, count in rows:
                if file_id in expected_counts:
                    raise ValueError(f"duplicate clip across tar indexes: {file_id}")
                count = int(count)
                expected_counts[str(file_id)] = count
                part_frames += count
                part_clips += 1
            part_counts.append((db_path.name, part_clips, part_frames))
        finally:
            con.close()
    return expected_counts, part_counts


def fail_examples(counter_or_list, limit):
    if hasattr(counter_or_list, "most_common"):
        return counter_or_list.most_common(limit)
    return list(counter_or_list[:limit])


def validate(args):
    cache_path = Path(args.cache)
    tar_indexes = [Path(path) for path in args.tar_index]
    if not cache_path.exists():
        raise FileNotFoundError(cache_path)
    for path in tar_indexes:
        if not path.exists():
            raise FileNotFoundError(path)

    allowlist = parse_object_camera_allowlist(args.object_camera_allowlist)
    ignore_regions = parse_object_ignore_regions(args.object_ignore_regions)
    thresholds = dict(DEFAULT_OBJECT_CLASS_THRESHOLDS)

    print("cache:", cache_path, f"{cache_path.stat().st_size / 1024 ** 3:.2f} GiB")
    print("building expected frame counts from ratarmount sqlite indexes...")
    expected_counts, part_counts = load_expected_counts(tar_indexes)
    expected_frames = sum(expected_counts.values())
    print("parts:", part_counts)
    print(f"expected clips={len(expected_counts):,} expected frames={expected_frames:,}")

    seen = {file_id: bytearray(count) for file_id, count in expected_counts.items()}
    seen_count_by_file = Counter()
    line_count = 0
    object_count = 0
    bad_json = 0
    unexpected_file = Counter()
    out_of_range = []
    duplicates = 0
    schema_errors = []
    class_mismatch = Counter()
    invalid_cls_id = Counter()
    invalid_box = 0
    nonfinite = 0
    threshold_violations = Counter()
    allowlist_violations = Counter()
    ignore_region_violations = Counter()
    objects_by_class = Counter()
    objects_by_detector = Counter()
    objects_by_camera = Counter()
    records_by_camera = Counter()
    width_height = Counter()
    empty_records = 0
    start = time.time()
    last_print = start

    with cache_path.open() as fh:
        for line_no, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                schema_errors.append((line_no, "blank line"))
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                bad_json += 1
                if bad_json <= args.max_examples:
                    schema_errors.append((line_no, f"json decode: {exc}"))
                continue
            line_count += 1
            try:
                file_id = rec["file_id"]
                frame_idx = int(rec["frame_idx"])
                width = int(rec["width"])
                height = int(rec["height"])
                objects = rec.get("objects", [])
            except Exception as exc:
                if len(schema_errors) < args.max_examples:
                    schema_errors.append(
                        (line_no, f"missing/bad top-level field: {exc}")
                    )
                continue

            camera = file_id.rsplit("_", 1)[-1] if "_" in file_id else "unknown"
            records_by_camera[camera] += 1
            width_height[(width, height)] += 1
            if not objects:
                empty_records += 1

            expected_count = expected_counts.get(file_id)
            if expected_count is None:
                unexpected_file[file_id] += 1
            elif frame_idx < 0 or frame_idx >= expected_count:
                if len(out_of_range) < args.max_examples:
                    out_of_range.append((line_no, file_id, frame_idx, expected_count))
            else:
                marks = seen[file_id]
                if marks[frame_idx]:
                    duplicates += 1
                else:
                    marks[frame_idx] = 1
                    seen_count_by_file[file_id] += 1

            if not isinstance(objects, list):
                if len(schema_errors) < args.max_examples:
                    schema_errors.append(
                        (line_no, f"objects is {type(objects).__name__}, not list")
                    )
                continue

            for obj in objects:
                object_count += 1
                try:
                    cls_id = int(obj["cls_id"])
                    cls_name = str(obj["cls"])
                    conf = float(obj["conf"])
                    x1, y1, x2, y2 = [float(value) for value in obj["xyxy"]]
                except Exception as exc:
                    if len(schema_errors) < args.max_examples:
                        schema_errors.append((line_no, f"bad object field: {exc}"))
                    continue

                if not (0 <= cls_id < NONE_OBJECT_ID):
                    invalid_cls_id[cls_id] += 1
                    continue
                expected_cls_name = OBJECT_CLASSES[cls_id]
                if cls_name != expected_cls_name:
                    class_mismatch[(cls_id, cls_name, expected_cls_name)] += 1
                detector_cls = obj.get("detector_cls")
                if detector_cls is not None:
                    mapped = mapped_object_name(detector_cls)
                    if mapped != cls_name:
                        class_mismatch[
                            (cls_id, cls_name, f"detector maps to {mapped}")
                        ] += 1
                    objects_by_detector[str(detector_cls)] += 1
                if not all(math.isfinite(value) for value in (conf, x1, y1, x2, y2)):
                    nonfinite += 1
                if not 0 <= conf <= 1.000001:
                    nonfinite += 1
                if (
                    x2 <= x1
                    or y2 <= y1
                    or x1 < -1e-4
                    or y1 < -1e-4
                    or x2 > width + 1e-4
                    or y2 > height + 1e-4
                ):
                    invalid_box += 1
                if conf + 1e-8 < thresholds[cls_name]:
                    threshold_violations[cls_name] += 1
                if not object_allowed_for_file_id(cls_name, file_id, allowlist):
                    allowlist_violations[(camera, cls_name)] += 1
                if object_box_ignored_for_file_id(
                    (x1, y1, x2, y2),
                    file_id,
                    width,
                    height,
                    ignore_regions,
                ):
                    ignore_region_violations[(camera, cls_name)] += 1
                objects_by_class[cls_name] += 1
                objects_by_camera[(camera, cls_name)] += 1

            now = time.time()
            if args.progress_seconds > 0 and now - last_print > args.progress_seconds:
                rate = line_count / max(now - start, 1e-6)
                print(
                    f"parsed {line_count:,}/{expected_frames:,} records "
                    f"at {rate:,.0f} rec/s",
                    flush=True,
                )
                last_print = now

    missing_by_file = []
    unique_expected_seen = 0
    for file_id, count in expected_counts.items():
        seen_count = int(sum(seen[file_id]))
        unique_expected_seen += seen_count
        if seen_count != count:
            missing_by_file.append((file_id, count, seen_count, count - seen_count))

    elapsed = time.time() - start
    issues = {
        "bad_json": bad_json,
        "schema_errors": len(schema_errors),
        "unexpected_file_ids": len(unexpected_file),
        "out_of_range_records": len(out_of_range),
        "duplicate_keys": duplicates,
        "missing_file_ids_or_frames": len(missing_by_file),
        "invalid_cls_ids": sum(invalid_cls_id.values()),
        "class_mismatches": sum(class_mismatch.values()),
        "invalid_boxes": invalid_box,
        "nonfinite_values": nonfinite,
        "threshold_violations": sum(threshold_violations.values()),
        "camera_allowlist_violations": sum(allowlist_violations.values()),
        "ignore_region_violations": sum(ignore_region_violations.values()),
    }

    print("\nVALIDATION SUMMARY")
    print(f"parsed records: {line_count:,}")
    print(f"expected frames: {expected_frames:,}")
    print(f"unique expected keys seen: {unique_expected_seen:,}")
    print(f"elapsed: {elapsed:.1f}s, parse rate: {line_count / max(elapsed, 1):,.0f} rec/s")
    print(f"objects: {object_count:,}")
    print(f"empty frame records: {empty_records:,} ({100 * empty_records / max(line_count, 1):.2f}%)")
    print("records_by_camera:", dict(sorted(records_by_camera.items())))
    print("top width/height:", width_height.most_common(10))
    print("objects_by_class:", dict(objects_by_class.most_common()))
    print(
        "tv_monitor_by_camera:",
        {
            camera: count
            for (camera, cls), count in sorted(objects_by_camera.items())
            if cls == "tv_monitor"
        },
    )
    print("top detector classes:", objects_by_detector.most_common(30))
    print("issues:", issues)

    examples = {
        "schema error examples": schema_errors,
        "unexpected file examples": unexpected_file,
        "out-of-range examples": out_of_range,
        "missing examples": missing_by_file,
        "invalid cls id examples": invalid_cls_id,
        "class mismatch examples": class_mismatch,
        "threshold violations": threshold_violations,
        "allowlist violations": allowlist_violations,
        "ignore-region violations": ignore_region_violations,
    }
    for title, values in examples.items():
        if values:
            print(f"{title}:", fail_examples(values, args.max_examples))

    ok = (
        line_count == expected_frames
        and unique_expected_seen == expected_frames
        and all(value == 0 for value in issues.values())
    )
    print("PASS" if ok else "FAIL")
    return 0 if ok else 2


def main():
    args = build_parser().parse_args()
    return validate(args)


if __name__ == "__main__":
    sys.exit(main())
