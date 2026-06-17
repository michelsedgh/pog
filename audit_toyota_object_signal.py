#!/usr/bin/env python3
import argparse
import csv
import json
import os
import random
import re
from collections import Counter, defaultdict

from datasets.object_vocab import (
    OBJECT_TO_ID,
    STRONG_ACTION_OBJECTS,
    object_allowed_for_file_id,
    object_box_ignored_for_file_id,
    parse_object_camera_allowlist,
    parse_object_ignore_regions,
)


CS_DICT = {
    "Cook.Cleandishes": 1,
    "Cook.Cleanup": 2,
    "Cook.Cut": 3,
    "Cook.Stir": 4,
    "Cook.Usestove": 5,
    "Cutbread": 6,
    "Drink.Frombottle": 7,
    "Drink.Fromcan": 8,
    "Drink.Fromcup": 9,
    "Drink.Fromglass": 10,
    "Eat.Attable": 11,
    "Eat.Snack": 12,
    "Enter": 13,
    "Getup": 14,
    "Laydown": 15,
    "Leave": 16,
    "Makecoffee.Pourgrains": 17,
    "Makecoffee.Pourwater": 18,
    "Maketea.Boilwater": 19,
    "Maketea.Insertteabag": 20,
    "Pour.Frombottle": 21,
    "Pour.Fromcan": 22,
    "Pour.Fromkettle": 23,
    "Readbook": 24,
    "Sitdown": 25,
    "Takepills": 26,
    "Uselaptop": 27,
    "Usetablet": 28,
    "Usetelephone": 29,
    "Walk": 30,
    "WatchTV": 31,
}


ID_TO_ACTION = {idx - 1: name for name, idx in CS_DICT.items()}
ACTION_TO_ID = {name: idx - 1 for name, idx in CS_DICT.items()}

ACTION_DISTRACTORS = {
    "Uselaptop": ["book", "tv_monitor", "remote"],
    "Readbook": ["laptop", "keyboard_mouse", "tv_monitor", "remote"],
    "Usetelephone": ["tv_monitor", "remote", "laptop", "book"],
    "Drink.Frombottle": ["cup", "glass"],
    "Drink.Fromcup": ["bottle", "glass"],
    "Drink.Fromglass": ["cup", "bottle"],
}

ACTION_OBJECTS = {
    action: {
        "positive": list(objects),
        "distractor": ACTION_DISTRACTORS.get(action, []),
    }
    for action, objects in STRONG_ACTION_OBJECTS.items()
}

GROUPS = {
    "laptop_book_tv": ["Uselaptop", "Readbook", "WatchTV"],
    "phone_tv": ["Usetelephone", "WatchTV"],
    "drink": ["Drink.Frombottle", "Drink.Fromcup", "Drink.Fromglass"],
}


def subject_id(file_id):
    match = re.search(r"_p(\d+)_", str(file_id))
    return None if match is None else int(match.group(1))


def action_name(file_id):
    return str(file_id).split("_", 1)[0]


def normalize_label(value):
    text = str(value)
    if text in ACTION_TO_ID:
        return ACTION_TO_ID[text]
    return int(float(text))


def read_cache_file_ids(path):
    file_ids = set()
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            file_id = str(entry["file_id"])
            if action_name(file_id) in CS_DICT and subject_id(file_id) is not None:
                file_ids.add(file_id)
    return sorted(file_ids)


def auto_split(file_ids, split, val_fraction, test_fraction, seed):
    rows = []
    for file_id in file_ids:
        subj = subject_id(file_id)
        name = action_name(file_id)
        if subj is None or name not in CS_DICT:
            continue
        rows.append((file_id, ACTION_TO_ID[name], subj))
    subjects = sorted({row[2] for row in rows})
    rng = random.Random(seed)
    rng.shuffle(subjects)
    n_subjects = len(subjects)
    n_test = round(n_subjects * test_fraction)
    n_val = round((n_subjects - n_test) * val_fraction)
    if n_subjects >= 3:
        if test_fraction > 0:
            n_test = max(1, n_test)
        if val_fraction > 0:
            n_val = max(1, n_val)
    test_subjects = set(subjects[:n_test])
    val_subjects = set(subjects[n_test : n_test + n_val])
    if split == "test":
        keep_subjects = test_subjects
    elif split == "val":
        keep_subjects = val_subjects
    elif split == "train":
        keep_subjects = set(subjects) - test_subjects - val_subjects
    else:
        raise ValueError(f"unknown split: {split}")
    return [(file_id, label) for file_id, label, subj in rows if subj in keep_subjects]


def load_base_predictions(path):
    if path is None:
        return {}
    preds = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if "file_id" not in reader.fieldnames:
            raise ValueError("base prediction CSV must contain file_id")
        for row in reader:
            if "label_idx" in row:
                label = normalize_label(row["label_idx"])
            elif "label" in row:
                label = normalize_label(row["label"])
            else:
                raise ValueError("base prediction CSV must contain label or label_idx")
            if "pred_idx" in row:
                pred = normalize_label(row["pred_idx"])
            elif "pred" in row:
                pred = normalize_label(row["pred"])
            else:
                raise ValueError("base prediction CSV must contain pred or pred_idx")
            preds[str(row["file_id"])] = {"label": label, "pred": pred}
    return preds


def object_name(obj):
    if "cls" in obj:
        return str(obj["cls"])
    cls_id = int(obj["cls_id"])
    for name, idx in OBJECT_TO_ID.items():
        if idx == cls_id:
            return name
    raise ValueError(f"unknown object cls_id: {cls_id}")


def build_object_features(path, file_ids, conf_threshold, camera_allowlist, ignore_regions):
    wanted = set(file_ids)
    features = {
        file_id: {
            "max_conf": defaultdict(float),
            "frame_count": 0,
            "detected_frames": defaultdict(int),
            "area_sum": defaultdict(float),
            "area_count": defaultdict(int),
        }
        for file_id in wanted
    }
    with open(path) as f:
        for line_idx, line in enumerate(f, 1):
            if not line.strip():
                continue
            entry = json.loads(line)
            file_id = str(entry["file_id"])
            if file_id not in wanted:
                continue
            width = float(entry.get("width") or 1.0)
            height = float(entry.get("height") or 1.0)
            features[file_id]["frame_count"] += 1
            seen_this_frame = set()
            for obj in entry.get("objects", []):
                name = object_name(obj)
                if name not in OBJECT_TO_ID:
                    continue
                conf = float(obj.get("conf", 0.0))
                if conf < conf_threshold:
                    continue
                if not object_allowed_for_file_id(name, file_id, camera_allowlist):
                    continue
                box = obj.get("xyxy")
                if box is None or len(box) != 4:
                    continue
                if object_box_ignored_for_file_id(
                    box,
                    file_id,
                    width,
                    height,
                    ignore_regions,
                ):
                    continue
                x1, y1, x2, y2 = [float(value) for value in box]
                area = max(0.0, x2 - x1) * max(0.0, y2 - y1) / max(1.0, width * height)
                feat = features[file_id]
                feat["max_conf"][name] = max(feat["max_conf"][name], conf)
                feat["area_sum"][name] += area
                feat["area_count"][name] += 1
                seen_this_frame.add(name)
            for name in seen_this_frame:
                features[file_id]["detected_frames"][name] += 1
    return features


def has_any(feat, names):
    return any(feat["max_conf"].get(name, 0.0) > 0.0 for name in names)


def top_confusions(counter, limit=5):
    if not counter:
        return "-"
    return ";".join(f"{ID_TO_ACTION.get(label, label)}:{count}" for label, count in counter.most_common(limit))


def print_table(title, rows, headers):
    print(f"\n{title}")
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, header in enumerate(headers):
            widths[idx] = max(widths[idx], len(str(row.get(header, ""))))
    print("  ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)))
    print("  ".join("-" * widths[idx] for idx in range(len(headers))))
    for row in rows:
        print("  ".join(str(row.get(header, "")).ljust(widths[idx]) for idx, header in enumerate(headers)))


def main():
    parser = argparse.ArgumentParser(
        description="Audit Toyota object-cache signal for object-sensitive actions."
    )
    parser.add_argument("--object_cache", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--test_fraction", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--object_conf_threshold", type=float, default=0.25)
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
    parser.add_argument("--base_predictions_csv", default=None)
    parser.add_argument("--output_csv", default=None)
    args = parser.parse_args()

    camera_allowlist = parse_object_camera_allowlist(args.object_camera_allowlist)
    ignore_regions = parse_object_ignore_regions(args.object_ignore_regions)
    all_file_ids = read_cache_file_ids(args.object_cache)
    split_rows = auto_split(
        all_file_ids,
        args.split,
        args.val_fraction,
        args.test_fraction,
        args.seed,
    )
    split_file_ids = [file_id for file_id, _ in split_rows]
    labels_by_file = dict(split_rows)
    features = build_object_features(
        args.object_cache,
        split_file_ids,
        args.object_conf_threshold,
        camera_allowlist,
        ignore_regions,
    )
    base_preds = load_base_predictions(args.base_predictions_csv)

    action_rows = []
    detailed_rows = []
    for action, spec in ACTION_OBJECTS.items():
        action_idx = ACTION_TO_ID[action]
        file_ids = [file_id for file_id, label in split_rows if label == action_idx]
        total = len(file_ids)
        pos = spec["positive"]
        dist = spec["distractor"]
        pos_count = sum(has_any(features[file_id], pos) for file_id in file_ids)
        dist_count = sum(has_any(features[file_id], dist) for file_id in file_ids)
        both_count = sum(
            has_any(features[file_id], pos) and has_any(features[file_id], dist)
            for file_id in file_ids
        )
        wrong_ids = []
        confusion = Counter()
        for file_id in file_ids:
            pred = base_preds.get(file_id)
            if pred is None:
                continue
            if pred["label"] != action_idx:
                raise ValueError(f"label mismatch for {file_id}")
            if pred["pred"] != action_idx:
                wrong_ids.append(file_id)
                confusion[pred["pred"]] += 1
        wrong_total = len(wrong_ids) if base_preds else ""
        wrong_pos = (
            sum(has_any(features[file_id], pos) for file_id in wrong_ids)
            if base_preds
            else ""
        )
        wrong_dist = (
            sum(has_any(features[file_id], dist) for file_id in wrong_ids)
            if base_preds
            else ""
        )
        row = {
            "action": action,
            "total": total,
            "positive_detected": pos_count,
            "positive_rate": f"{pos_count / total:.3f}" if total else "nan",
            "distractor_detected": dist_count,
            "distractor_rate": f"{dist_count / total:.3f}" if total else "nan",
            "positive_and_distractor": both_count,
            "base_wrong": wrong_total,
            "base_wrong_positive_detected": wrong_pos,
            "base_wrong_distractor_detected": wrong_dist,
            "top_base_confusions": top_confusions(confusion),
        }
        action_rows.append(row)
        for file_id in file_ids:
            feat = features[file_id]
            detailed_rows.append(
                {
                    "file_id": file_id,
                    "action": action,
                    "label_idx": action_idx,
                    "positive_detected": int(has_any(feat, pos)),
                    "distractor_detected": int(has_any(feat, dist)),
                    **{
                        f"conf_{name}": f"{feat['max_conf'].get(name, 0.0):.6f}"
                        for name in sorted(OBJECT_TO_ID)
                    },
                }
            )

    group_rows = []
    for group_name, actions in GROUPS.items():
        action_set = set(actions)
        rows = [row for row in action_rows if row["action"] in action_set]
        total = sum(int(row["total"]) for row in rows)
        pos_total = sum(int(row["positive_detected"]) for row in rows)
        dist_total = sum(int(row["distractor_detected"]) for row in rows)
        wrong_values = [
            int(row["base_wrong"]) for row in rows if row["base_wrong"] != ""
        ]
        wrong_total = sum(wrong_values) if wrong_values else ""
        group_rows.append(
            {
                "group": group_name,
                "total": total,
                "positive_detected": pos_total,
                "positive_rate": f"{pos_total / total:.3f}" if total else "nan",
                "distractor_detected": dist_total,
                "distractor_rate": f"{dist_total / total:.3f}" if total else "nan",
                "base_wrong": wrong_total,
            }
        )

    print(f"object_cache: {args.object_cache}")
    print(f"split: {args.split}")
    print(f"all clips from object cache: {len(all_file_ids)}")
    print(f"split clips: {len(split_rows)}")
    print(f"base predictions: {'yes' if base_preds else 'no'}")

    print_table(
        "Object-Sensitive Group Coverage",
        group_rows,
        [
            "group",
            "total",
            "positive_detected",
            "positive_rate",
            "distractor_detected",
            "distractor_rate",
            "base_wrong",
        ],
    )
    print_table(
        "Action Coverage / Base Error Audit",
        action_rows,
        [
            "action",
            "total",
            "positive_detected",
            "positive_rate",
            "distractor_detected",
            "distractor_rate",
            "positive_and_distractor",
            "base_wrong",
            "base_wrong_positive_detected",
            "base_wrong_distractor_detected",
            "top_base_confusions",
        ],
    )

    if args.output_csv:
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(detailed_rows[0].keys()))
            writer.writeheader()
            writer.writerows(detailed_rows)
        print(f"\nwrote detailed object features: {args.output_csv}")


if __name__ == "__main__":
    main()
