import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

from datasets.object_vocab import OBJECT_CLASSES
from datasets.toyotasm import CS_DICT


HARD_NEGATIVE_RULES = {
    "WatchTV": ["laptop", "book", "phone", "keyboard_mouse"],
    "Uselaptop": ["book", "tv_monitor", "phone"],
    "Readbook": ["laptop", "tv_monitor", "phone"],
    "Usetelephone": ["book", "laptop", "tv_monitor"],
    "Eat.Attable": ["cup", "bottle"],
    "Eat.Snack": ["cup", "bottle"],
    "Takepills": ["food_snack", "bowl", "utensil"],
    "Sitdown": ["phone", "laptop", "book"],
    "Getup": ["phone", "laptop", "book"],
}


EXPECTED_CONFUSIONS = {
    "WatchTV": ["Uselaptop", "Readbook", "Usetelephone"],
    "Uselaptop": ["WatchTV", "Readbook", "Usetelephone"],
    "Readbook": ["Uselaptop", "WatchTV", "Usetelephone"],
    "Usetelephone": ["Readbook", "Uselaptop", "WatchTV"],
    "Eat.Attable": ["Drink.Fromcup", "Drink.Frombottle", "Takepills"],
    "Eat.Snack": ["Drink.Fromcup", "Drink.Frombottle", "Takepills"],
    "Takepills": ["Eat.Attable", "Eat.Snack"],
    "Sitdown": ["Usetelephone", "Uselaptop", "Readbook"],
    "Getup": ["Usetelephone", "Uselaptop", "Readbook"],
}


def build_parser():
    parser = argparse.ArgumentParser(
        description="Audit Toyota object-cache hard negatives by action/object co-occurrence."
    )
    parser.add_argument("--object_detector_cache", required=True)
    parser.add_argument("--summary_json", default="hard_negative_audit_summary.json")
    parser.add_argument("--manifest_json", default="hard_negatives.json")
    parser.add_argument("--min_conf", type=float, default=0.35)
    parser.add_argument("--min_frames", type=int, default=5)
    parser.add_argument("--min_frame_fraction", type=float, default=0.01)
    parser.add_argument("--min_area", type=float, default=0.0005)
    parser.add_argument("--top_per_bucket", type=int, default=50)
    parser.add_argument("--max_manifest", type=int, default=2000)
    parser.add_argument("--progress_seconds", type=float, default=30.0)
    return parser


def action_from_file_id(file_id):
    for action_name in sorted(CS_DICT.keys(), key=len, reverse=True):
        if file_id.startswith(action_name + "_p"):
            return action_name
    return None


def empty_stats():
    return {
        "frames_seen": 0,
        "max_conf": 0.0,
        "max_area": 0.0,
        "best_frame_idx": -1,
        "best_frame_score": 0.0,
    }


def update_object_stats(stats, frame_idx, conf, area):
    stats["frames_seen"] += 1
    if conf > stats["max_conf"]:
        stats["max_conf"] = float(conf)
    if area > stats["max_area"]:
        stats["max_area"] = float(area)
    frame_score = float(conf) * math.sqrt(max(float(area), 1e-9))
    if frame_score > stats["best_frame_score"]:
        stats["best_frame_score"] = frame_score
        stats["best_frame_idx"] = int(frame_idx)


def visible_object_stats(stats, total_frames, args):
    if total_frames <= 0:
        return None
    frame_fraction = stats["frames_seen"] / float(total_frames)
    if stats["frames_seen"] < args.min_frames:
        return None
    if frame_fraction < args.min_frame_fraction:
        return None
    if stats["max_conf"] < args.min_conf:
        return None
    if stats["max_area"] < args.min_area:
        return None
    output = dict(stats)
    output["frame_fraction"] = frame_fraction
    return output


def quality_score(stats):
    conf = float(stats["max_conf"])
    fraction_score = min(float(stats["frame_fraction"]) / 0.05, 1.0)
    area_score = min(float(stats["max_area"]) / 0.02, 1.0)
    return conf * math.sqrt(max(fraction_score, 0.0)) * math.sqrt(max(area_score, 0.0))


def parse_cache(path, args):
    clip_frame_counts = Counter()
    clip_action = {}
    clip_object_stats = defaultdict(lambda: defaultdict(empty_stats))
    bad_actions = Counter()
    line_count = 0
    object_count = 0

    import time

    start = time.time()
    last_print = start
    with open(path) as fh:
        for line_no, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            rec = json.loads(line)
            line_count += 1
            file_id = rec["file_id"]
            frame_idx = int(rec["frame_idx"])
            width = float(rec["width"])
            height = float(rec["height"])
            action = clip_action.get(file_id)
            if action is None:
                action = action_from_file_id(file_id)
                if action is None:
                    bad_actions[file_id] += 1
                clip_action[file_id] = action
            clip_frame_counts[file_id] += 1

            per_frame_best = {}
            for obj in rec.get("objects", []):
                cls_id = int(obj["cls_id"])
                cls_name = OBJECT_CLASSES.get(cls_id)
                if cls_name is None:
                    continue
                conf = float(obj["conf"])
                x1, y1, x2, y2 = [float(value) for value in obj["xyxy"]]
                area = max(0.0, (x2 - x1) * (y2 - y1)) / max(width * height, 1.0)
                current = per_frame_best.get(cls_name)
                if current is None or conf > current[0]:
                    per_frame_best[cls_name] = (conf, area)
                object_count += 1

            for cls_name, (conf, area) in per_frame_best.items():
                update_object_stats(
                    clip_object_stats[file_id][cls_name],
                    frame_idx,
                    conf,
                    area,
                )

            now = time.time()
            if args.progress_seconds > 0 and now - last_print >= args.progress_seconds:
                rate = line_count / max(now - start, 1e-6)
                print(
                    f"parsed {line_count:,} records, {len(clip_frame_counts):,} clips, "
                    f"{object_count:,} objects at {rate:,.0f} rec/s",
                    flush=True,
                )
                last_print = now

    return clip_frame_counts, clip_action, clip_object_stats, bad_actions, line_count, object_count


def build_audit(args):
    (
        clip_frame_counts,
        clip_action,
        clip_object_stats,
        bad_actions,
        line_count,
        object_count,
    ) = parse_cache(args.object_detector_cache, args)

    action_totals = Counter()
    presence_by_action = defaultdict(Counter)
    hard_counts = defaultdict(Counter)
    hard_any_counts = Counter()
    bucket_examples = defaultdict(list)
    manifest = []

    for file_id, total_frames in clip_frame_counts.items():
        action = clip_action.get(file_id)
        if action is None:
            continue
        action_totals[action] += 1
        visible = {}
        for cls_name, stats in clip_object_stats[file_id].items():
            object_stats = visible_object_stats(stats, total_frames, args)
            if object_stats is not None:
                visible[cls_name] = object_stats
                presence_by_action[action][cls_name] += 1

        distractor_names = HARD_NEGATIVE_RULES.get(action, [])
        distractors = [
            name for name in distractor_names
            if name in visible
        ]
        if not distractors:
            continue

        hard_any_counts[action] += 1
        for name in distractors:
            hard_counts[action][name] += 1

        distractor_stats = {
            name: visible[name]
            for name in distractors
        }
        best_quality = max(quality_score(stats) for stats in distractor_stats.values())
        best_frame_idx = max(
            (
                int(stats["best_frame_idx"])
                for stats in distractor_stats.values()
            ),
            default=-1,
        )
        entry = {
            "file_id": file_id,
            "action": action,
            "distractors": distractors,
            "expected_confusion": EXPECTED_CONFUSIONS.get(action, []),
            "quality": best_quality,
            "best_frame_idx": best_frame_idx,
            "total_frames": int(total_frames),
            "distractor_stats": distractor_stats,
        }
        manifest.append(entry)
        for name in distractors:
            bucket_examples[(action, name)].append(entry)

    for examples in bucket_examples.values():
        examples.sort(key=lambda item: item["quality"], reverse=True)
        del examples[args.top_per_bucket:]

    manifest.sort(key=lambda item: item["quality"], reverse=True)
    if args.max_manifest > 0:
        manifest = manifest[: args.max_manifest]

    summary = {
        "cache": str(args.object_detector_cache),
        "records": line_count,
        "objects": object_count,
        "clips": len(clip_frame_counts),
        "filters": {
            "min_conf": args.min_conf,
            "min_frames": args.min_frames,
            "min_frame_fraction": args.min_frame_fraction,
            "min_area": args.min_area,
        },
        "bad_action_file_ids": dict(bad_actions.most_common()),
        "action_totals": dict(sorted(action_totals.items())),
        "presence_by_action": {
            action: dict(counter.most_common())
            for action, counter in sorted(presence_by_action.items())
        },
        "hard_negative_any_counts": dict(sorted(hard_any_counts.items())),
        "hard_negative_counts": {
            action: dict(counter.most_common())
            for action, counter in sorted(hard_counts.items())
        },
        "hard_negative_examples": {
            f"{action}__{distractor}": examples
            for (action, distractor), examples in sorted(bucket_examples.items())
        },
    }
    return summary, manifest


def print_report(summary, manifest):
    print("\nHARD NEGATIVE AUDIT")
    print(f"clips: {summary['clips']:,}")
    print(f"records: {summary['records']:,}")
    print(f"objects: {summary['objects']:,}")
    print("filters:", summary["filters"])
    print("\nAction totals and hard-negative coverage:")
    action_totals = summary["action_totals"]
    hard_any = summary["hard_negative_any_counts"]
    hard_counts = summary["hard_negative_counts"]
    for action in sorted(HARD_NEGATIVE_RULES):
        total = int(action_totals.get(action, 0))
        hard_total = int(hard_any.get(action, 0))
        pct = 100.0 * hard_total / max(total, 1)
        print(
            f"  {action:16s} total={total:4d} hard_any={hard_total:4d} "
            f"({pct:5.1f}%) distractors={hard_counts.get(action, {})}"
        )

    print("\nTop manifest examples:")
    for entry in manifest[:20]:
        print(
            f"  {entry['quality']:.3f} {entry['action']:16s} "
            f"{entry['file_id']} distractors={entry['distractors']} "
            f"frame={entry['best_frame_idx']}"
        )


def main():
    args = build_parser().parse_args()
    summary, manifest = build_audit(args)
    Path(args.summary_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    Path(args.manifest_json).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print_report(summary, manifest)
    print(f"\nwrote {args.summary_json}")
    print(f"wrote {args.manifest_json} ({len(manifest)} entries)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
