#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd


CORE_COLUMNS = [
    "epoch",
    "val_loss",
    "val_acc_macro",
    "val_f1",
    "val_loss_interaction_heatmap",
    "val_loss_heatmap_log",
    "val_loss_heatmap_frobenius",
    "val_loss_pose_heatmap_frobenius",
    "val_loss_interaction_heatmap_frobenius",
    "train_nash_weight_action",
    "train_nash_weight_heatmap",
    "val_interaction_teacher_slot_rate",
    "val_interaction_teacher_slot_count",
    "val_interaction_heatmap_iou",
    "val_interaction_heatmap_soft_iou",
    "val_interaction_heatmap_positive_mean",
    "val_interaction_heatmap_pred_max",
    "val_interaction_heatmap_target_max",
    "val_interaction_heatmap_center_l2",
    "interaction_score",
]

GROUPS = [
    "laptop_book_tv",
    "phone_tv",
    "drink_cup_bottle_glass",
    "drink",
]

ACTIONS = [
    "Uselaptop",
    "Readbook",
    "WatchTV",
    "Usetelephone",
    "Drink_Fromcup",
    "Drink_Frombottle",
    "Drink_Fromglass",
    "Pour_Frombottle",
    "Cutbread",
    "Cook_Cut",
    "Cook_Stir",
    "Cook_Cleandishes",
    "Cook_Usestove",
]

OBJECTS = [
    "laptop",
    "book",
    "phone",
    "cup",
    "bottle",
    "utensil",
    "bowl",
    "sink",
    "cooking_appliance",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize actor interaction heatmap training metrics."
    )
    parser.add_argument(
        "--root",
        default="/mnt/local-scratch/poguise_data/checkpoints",
        help="Checkpoint root used when --run is not supplied.",
    )
    parser.add_argument(
        "--pattern",
        default="actor_object_poguiseplus_final_*",
        help="Run glob under --root used when --run is not supplied.",
    )
    parser.add_argument("--run", default=None, help="Specific run directory.")
    parser.add_argument("--metrics", default=None, help="Specific metrics.csv path.")
    return parser.parse_args()


def resolve_metrics(args):
    if args.metrics:
        metrics = Path(args.metrics)
        if not metrics.is_file():
            raise SystemExit(f"metrics.csv not found: {metrics}")
        return metrics.parent.parent, metrics

    if args.run:
        run = Path(args.run)
    else:
        root = Path(args.root)
        runs = sorted(root.glob(args.pattern), key=lambda p: p.stat().st_mtime)
        if not runs:
            raise SystemExit(f"No runs found matching {root / args.pattern}")
        run = runs[-1]

    metrics_files = sorted(
        run.glob("version_*/metrics.csv"), key=lambda p: p.stat().st_mtime
    )
    if not metrics_files:
        raise SystemExit(f"No metrics.csv found under {run}")
    return run, metrics_files[-1]


def last_nonnull(series):
    series = series.dropna()
    return series.iloc[-1] if len(series) else float("nan")


def metric(row, name):
    for candidate in (name, f"{name}_epoch", f"{name}_step"):
        if candidate in row.index:
            return row[candidate]
    return float("nan")


def df_metric(df, name, default):
    for candidate in (name, f"{name}_epoch", f"{name}_step"):
        if candidate in df.columns:
            return df[candidate].fillna(default)
    return default


def compute_interaction_score(df):
    iou = df_metric(df, "val_interaction_heatmap_iou", 0.0)
    soft_iou = df_metric(df, "val_interaction_heatmap_soft_iou", 0.0)
    pos = df_metric(df, "val_interaction_heatmap_positive_mean", 0.0)
    center = df_metric(df, "val_interaction_heatmap_center_l2", 56.0)
    action = df_metric(df, "val_f1", 0.0)
    return iou + soft_iou + pos - 0.02 * center + 0.25 * action


def print_row(title, row):
    print(f"\n{title}: epoch {int(row['epoch'])}")
    print(f"val_loss: {metric(row, 'val_loss'):.4f}")
    print(
        "macro/f1: "
        f"{metric(row, 'val_acc_macro'):.4f} / {metric(row, 'val_f1'):.4f}"
    )
    print(
        "interaction heatmap: "
        f"loss {metric(row, 'val_loss_interaction_heatmap'):.6f}, "
        f"iou {metric(row, 'val_interaction_heatmap_iou'):.4f}, "
        f"soft_iou {metric(row, 'val_interaction_heatmap_soft_iou'):.4f}, "
        f"positive {metric(row, 'val_interaction_heatmap_positive_mean'):.4f}, "
        f"pred_max {metric(row, 'val_interaction_heatmap_pred_max'):.4f}, "
        f"target_max {metric(row, 'val_interaction_heatmap_target_max'):.4f}, "
        f"center_l2 {metric(row, 'val_interaction_heatmap_center_l2'):.2f}"
    )
    print(
        "trusted teacher slots: "
        f"rate {metric(row, 'val_interaction_teacher_slot_rate'):.4f}, "
        f"count {metric(row, 'val_interaction_teacher_slot_count'):.1f}"
    )
    print(
        "poguise+ heatmap loss: "
        f"log {metric(row, 'val_loss_heatmap_log'):.4f}, "
        f"fro {metric(row, 'val_loss_heatmap_frobenius'):.4f}, "
        f"pose_fro {metric(row, 'val_loss_pose_heatmap_frobenius'):.4f}, "
        f"interaction_fro {metric(row, 'val_loss_interaction_heatmap_frobenius'):.4f}"
    )
    print(
        "nash weights: "
        f"action {metric(row, 'train_nash_weight_action'):.4f}, "
        f"heatmap {metric(row, 'train_nash_weight_heatmap'):.4f}"
    )
    print(f"interaction_score: {metric(row, 'interaction_score'):.4f}")

    print("\nTARGET GROUPS:")
    for group in GROUPS:
        col = f"val_group_{group}_acc"
        if col in row.index and pd.notna(row[col]):
            print(f"{group}: {float(row[col]):.4f}")

    print("\nTARGET ACTIONS:")
    for action in ACTIONS:
        col = f"val_action_{action}_acc"
        if col in row.index and pd.notna(row[col]):
            teacher_col = f"val_action_{action}_interaction_teacher_rate"
            teacher = metric(row, teacher_col)
            if pd.notna(teacher):
                print(f"{action}: acc {float(row[col]):.4f}, teacher {teacher:.4f}")
            else:
                print(f"{action}: acc {float(row[col]):.4f}")

    object_rows = []
    for object_name in OBJECTS:
        pos = metric(row, f"val_interaction_heatmap_{object_name}_positive_mean")
        iou = metric(row, f"val_interaction_heatmap_{object_name}_iou")
        count = metric(row, f"val_interaction_teacher_{object_name}_slot_count")
        if pd.notna(pos) or pd.notna(iou) or pd.notna(count):
            object_rows.append((object_name, count, pos, iou))
    if object_rows:
        print("\nOBJECT HEATMAP CHANNELS:")
        for object_name, count, pos, iou in object_rows:
            count_text = "nan" if pd.isna(count) else f"{float(count):.0f}"
            pos_text = "nan" if pd.isna(pos) else f"{float(pos):.4f}"
            iou_text = "nan" if pd.isna(iou) else f"{float(iou):.4f}"
            print(
                f"{object_name}: teacher_count {count_text}, "
                f"positive {pos_text}, iou {iou_text}"
            )


def main():
    args = parse_args()
    run, metrics = resolve_metrics(args)
    df = pd.read_csv(metrics)
    if "epoch" not in df.columns:
        raise SystemExit(f"{metrics} has no epoch column")

    epoch_df = df.groupby("epoch", as_index=False).agg(
        {col: last_nonnull for col in df.columns if col != "epoch"}
    )
    validation_cols = [
        col
        for col in (
            "val_loss",
            "val_f1",
            "val_interaction_heatmap_iou",
            "val_loss_interaction_heatmap",
        )
        if col in epoch_df.columns
    ]
    if validation_cols:
        epoch_df = epoch_df.dropna(how="all", subset=validation_cols).reset_index(
            drop=True
        )
    if epoch_df.empty:
        raise SystemExit(f"{metrics} has no completed validation epochs")
    epoch_df["interaction_score"] = compute_interaction_score(epoch_df)

    display_cols = [col for col in CORE_COLUMNS if col in epoch_df.columns]
    print("run:", run)
    print("metrics:", metrics)
    print("\nEPOCH SUMMARY:\n")
    print(epoch_df[display_cols].to_string(index=False))

    latest = epoch_df.iloc[-1]
    valid_score = epoch_df["interaction_score"].notna()
    if valid_score.any():
        best = epoch_df.loc[epoch_df.loc[valid_score, "interaction_score"].idxmax()]
    else:
        best = latest

    print_row("LATEST", latest)
    print_row("BEST_BY_INTERACTION_SCORE", best)

    print("\nREAD THIS:")
    print("- Interaction heatmap IoU/positive response/center error show whether the model is learning object-region supervision.")
    print("- Target group/action accuracy shows whether the actor classifier still handles object-confusable classes.")
    print("- Object heatmap channels are class-specific actor-object teacher labels.")
    print("- RF-DETR boxes are teacher labels here; they are not runtime model inputs.")


if __name__ == "__main__":
    main()
