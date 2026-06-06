#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd


CORE_COLUMNS = [
    "epoch",
    "val_loss",
    "val_acc_macro",
    "val_f1",
    "val_actor_all_slot_acc",
    "val_actor_slot_consistency",
    "val_actor_pair_acc",
    "val_actor_pair_swap_acc",
    "val_actor_pair_same_acc",
    "val_actor_pair_diff_acc",
    "val_actor_presence_acc",
    "val_actor_presence_bg_acc",
    "val_loss_interaction_heatmap",
    "val_loss_heatmap_log",
    "val_loss_heatmap_frobenius",
    "val_loss_pose_heatmap_frobenius",
    "val_loss_interaction_heatmap_raw_frobenius",
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
    "val_object_selection_loss",
    "val_object_selection_acc",
    "val_object_selection_none_acc",
    "val_object_selection_object_acc",
    "val_object_selection_true_prob",
    "val_object_selection_teacher_count",
    "val_object_selection_none_teacher_count",
    "val_object_selection_object_teacher_count",
    "val_object_counterfactual_selected_logit_drop",
    "val_object_counterfactual_selected_prob_drop",
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


def fmt(value, digits=4):
    if pd.isna(value):
        return "nan"
    return f"{float(value):.{digits}f}"


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
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print full heatmap/channel/group diagnostics instead of the compact object-use summary.",
    )
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


def df_has_metric(df, name):
    return any(
        candidate in df.columns
        for candidate in (name, f"{name}_epoch", f"{name}_step")
    )


def print_table(title, rows, columns=None):
    if not rows:
        return
    print(f"\n{title}:\n")
    table = pd.DataFrame(rows)
    if columns is not None:
        columns = [col for col in columns if col in table.columns]
        table = table[columns]
    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        220,
        "display.max_colwidth",
        80,
    ):
        print(table.to_string(index=False))


def print_compact_object_use_summary(epoch_df):
    cols = [
        "epoch",
        "val_acc_macro",
        "val_f1",
        "val_actor_all_slot_acc",
        "val_actor_pair_acc",
        "val_actor_pair_swap_acc",
        "val_actor_pair_same_acc",
        "val_actor_pair_diff_acc",
        "val_actor_presence_acc",
        "val_object_selection_acc",
        "val_object_selection_none_acc",
        "val_object_selection_object_acc",
        "val_object_selection_true_prob",
        "val_object_selection_none_teacher_count",
        "val_object_selection_object_teacher_count",
        "val_object_counterfactual_selected_logit_drop",
        "val_object_counterfactual_selected_prob_drop",
        "val_action_Uselaptop_acc",
        "val_action_Readbook_acc",
        "val_action_Usetelephone_acc",
        "val_action_Drink_Fromcup_acc",
        "val_action_Drink_Frombottle_acc",
        "val_interaction_heatmap_laptop_positive_mean",
        "val_interaction_heatmap_laptop_iou",
        "val_interaction_heatmap_book_positive_mean",
        "val_interaction_heatmap_phone_positive_mean",
        "val_interaction_heatmap_cup_positive_mean",
        "val_interaction_heatmap_bottle_positive_mean",
    ]
    display_cols = [col for col in cols if col in epoch_df.columns]
    if len(display_cols) > 1:
        print("\nOBJECT-ACTION USE SUMMARY:\n")
        with pd.option_context("display.max_columns", None, "display.width", 220):
            print(epoch_df[display_cols].to_string(index=False))


def print_compact_target_actions(epoch_df):
    rows = action_progress_rows(epoch_df)
    if not rows:
        return
    keep = {
        "Uselaptop",
        "Readbook",
        "Usetelephone",
        "Drink_Fromcup",
        "Drink_Frombottle",
        "Pour_Frombottle",
        "Cutbread",
        "Cook_Cut",
    }
    rows = [row for row in rows if row["action"] in keep]
    print_table(
        "KEY TARGET ACTIONS",
        rows,
        [
            "action",
            "teacher",
            "e0_acc",
            "latest_acc",
            "delta",
            "best_epoch",
            "best_acc",
            "verdict",
        ],
    )


def print_compact_best(epoch_df):
    metrics = [
        "val_f1",
        "val_object_selection_acc",
        "val_object_selection_none_acc",
        "val_object_selection_object_acc",
        "val_object_selection_true_prob",
        "val_object_selection_none_teacher_count",
        "val_object_selection_object_teacher_count",
        "val_object_counterfactual_selected_logit_drop",
        "val_object_counterfactual_selected_prob_drop",
        "val_action_Uselaptop_acc",
        "val_action_Readbook_acc",
        "val_action_Usetelephone_acc",
    ]
    rows = []
    for name in metrics:
        if not df_has_metric(epoch_df, name):
            continue
        series = df_metric(epoch_df, name, float("nan"))
        valid = series.dropna()
        if not len(valid):
            continue
        best_idx = valid.idxmax()
        rows.append(
            {
                "metric": name,
                "best_epoch": epoch_df.loc[best_idx, "epoch"],
                "best_value": series.loc[best_idx],
            }
        )
    print_table("BEST COMPACT SIGNALS", rows, ["metric", "best_epoch", "best_value"])


def print_object_use_epoch_table(epoch_df):
    object_cols = [
        "val_interaction_heatmap_positive_mean",
        "val_interaction_heatmap_pred_max",
        "val_interaction_heatmap_soft_iou",
        "val_interaction_heatmap_center_l2",
        "val_object_selection_acc",
        "val_object_selection_none_acc",
        "val_object_selection_object_acc",
        "val_object_selection_true_prob",
        "val_object_selection_none_teacher_count",
        "val_object_selection_object_teacher_count",
        "val_object_counterfactual_selected_logit_drop",
        "val_object_counterfactual_selected_prob_drop",
    ]
    if not any(df_has_metric(epoch_df, col) for col in object_cols):
        return
    cols = [
        "epoch",
        "val_acc_macro",
        "val_f1",
        "val_interaction_heatmap_positive_mean",
        "val_interaction_heatmap_pred_max",
        "val_interaction_heatmap_soft_iou",
        "val_interaction_heatmap_center_l2",
        "val_object_selection_acc",
        "val_object_selection_none_acc",
        "val_object_selection_object_acc",
        "val_object_selection_true_prob",
        "val_object_counterfactual_selected_logit_drop",
        "val_object_counterfactual_selected_prob_drop",
    ]
    display_cols = [col for col in cols if col in epoch_df.columns]
    if len(display_cols) <= 1:
        return
    print("\nEPOCH OBJECT-USE QUALITY:\n")
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(epoch_df[display_cols].to_string(index=False))


def print_object_channel_progress(epoch_df):
    if epoch_df.empty:
        return
    base = epoch_df.iloc[0]
    latest = epoch_df.iloc[-1]
    rows = []
    for object_name in OBJECTS:
        pos_col = f"val_interaction_heatmap_{object_name}_positive_mean"
        iou_col = f"val_interaction_heatmap_{object_name}_iou"
        count_col = f"val_interaction_teacher_{object_name}_slot_count"
        if not (
            df_has_metric(epoch_df, pos_col)
            or df_has_metric(epoch_df, iou_col)
            or df_has_metric(epoch_df, count_col)
        ):
            continue
        pos_series = df_metric(epoch_df, pos_col, float("nan"))
        valid_pos = pos_series.dropna()
        best_epoch = float("nan")
        best_pos = float("nan")
        if len(valid_pos):
            best_idx = valid_pos.idxmax()
            best_epoch = epoch_df.loc[best_idx, "epoch"]
            best_pos = pos_series.loc[best_idx]
        rows.append(
            {
                "object": object_name,
                "teachers": metric(latest, count_col),
                "e0_pos": metric(base, pos_col),
                "latest_pos": metric(latest, pos_col),
                "delta": metric(latest, pos_col) - metric(base, pos_col),
                "latest_iou": metric(latest, iou_col),
                "best_epoch": best_epoch,
                "best_pos": best_pos,
            }
        )
    print_table(
        "OBJECT CHANNEL PROGRESS",
        rows,
        [
            "object",
            "teachers",
            "e0_pos",
            "latest_pos",
            "delta",
            "latest_iou",
            "best_epoch",
            "best_pos",
        ],
    )


def action_verdict(delta):
    if pd.isna(delta):
        return "unknown"
    if delta <= -0.03:
        return "hurt badly"
    if delta < -0.005:
        return "hurt"
    if delta >= 0.02:
        return "improved"
    return "stable"


def action_progress_rows(epoch_df):
    if epoch_df.empty:
        return []
    base = epoch_df.iloc[0]
    latest = epoch_df.iloc[-1]
    rows = []
    for action in ACTIONS:
        acc_col = f"val_action_{action}_acc"
        teacher_col = f"val_action_{action}_interaction_teacher_rate"
        if not df_has_metric(epoch_df, acc_col):
            continue
        acc_series = df_metric(epoch_df, acc_col, float("nan"))
        valid_acc = acc_series.dropna()
        best_epoch = float("nan")
        best_acc = float("nan")
        if len(valid_acc):
            best_idx = valid_acc.idxmax()
            best_epoch = epoch_df.loc[best_idx, "epoch"]
            best_acc = acc_series.loc[best_idx]
        delta = metric(latest, acc_col) - metric(base, acc_col)
        rows.append(
            {
                "action": action,
                "teacher": metric(latest, teacher_col),
                "e0_acc": metric(base, acc_col),
                "latest_acc": metric(latest, acc_col),
                "delta": delta,
                "best_epoch": best_epoch,
                "best_acc": best_acc,
                "verdict": action_verdict(delta),
            }
        )
    return rows


def print_action_progress(epoch_df):
    rows = action_progress_rows(epoch_df)
    print_table(
        "TARGET ACTION PROGRESS",
        rows,
        [
            "action",
            "teacher",
            "e0_acc",
            "latest_acc",
            "delta",
            "best_epoch",
            "best_acc",
            "verdict",
        ],
    )

    hurt = [
        row
        for row in rows
        if pd.notna(row["delta"]) and float(row["delta"]) < -0.005
    ]
    improved = [
        row
        for row in rows
        if pd.notna(row["delta"]) and float(row["delta"]) >= 0.02
    ]
    print_table(
        "ACTIONS HURT VS EPOCH 0",
        hurt,
        [
            "action",
            "teacher",
            "e0_acc",
            "latest_acc",
            "delta",
            "best_epoch",
            "best_acc",
            "verdict",
        ],
    )
    print_table(
        "ACTIONS IMPROVED VS EPOCH 0",
        improved,
        [
            "action",
            "teacher",
            "e0_acc",
            "latest_acc",
            "delta",
            "best_epoch",
            "best_acc",
            "verdict",
        ],
    )


def print_group_progress(epoch_df):
    if epoch_df.empty:
        return
    base = epoch_df.iloc[0]
    latest = epoch_df.iloc[-1]
    rows = []
    for group in GROUPS:
        col = f"val_group_{group}_acc"
        if not df_has_metric(epoch_df, col):
            continue
        series = df_metric(epoch_df, col, float("nan"))
        valid = series.dropna()
        best_epoch = float("nan")
        best_acc = float("nan")
        if len(valid):
            best_idx = valid.idxmax()
            best_epoch = epoch_df.loc[best_idx, "epoch"]
            best_acc = series.loc[best_idx]
        rows.append(
            {
                "group": group,
                "e0_acc": metric(base, col),
                "latest_acc": metric(latest, col),
                "delta": metric(latest, col) - metric(base, col),
                "best_epoch": best_epoch,
                "best_acc": best_acc,
            }
        )
    print_table(
        "GROUP PROGRESS",
        rows,
        ["group", "e0_acc", "latest_acc", "delta", "best_epoch", "best_acc"],
    )


def print_best_epochs(epoch_df):
    metrics = [
        "val_f1",
        "val_acc_macro",
        "val_interaction_heatmap_soft_iou",
        "val_interaction_heatmap_positive_mean",
        "val_interaction_heatmap_laptop_positive_mean",
        "val_interaction_heatmap_laptop_iou",
        "val_actor_all_slot_acc",
        "val_actor_pair_acc",
        "val_object_selection_acc",
        "val_object_selection_true_prob",
        "val_object_counterfactual_selected_logit_drop",
    ]
    rows = []
    for name in metrics:
        if not df_has_metric(epoch_df, name):
            continue
        series = df_metric(epoch_df, name, float("nan"))
        valid = series.dropna()
        if not len(valid):
            continue
        best_idx = valid.idxmax()
        rows.append(
            {
                "metric": name,
                "best_epoch": epoch_df.loc[best_idx, "epoch"],
                "best_value": series.loc[best_idx],
            }
        )
    print_table("BEST EPOCHS", rows, ["metric", "best_epoch", "best_value"])


def print_decision(epoch_df):
    if epoch_df.empty:
        return
    base = epoch_df.iloc[0]
    latest = epoch_df.iloc[-1]
    f1_base = metric(base, "val_f1")
    f1_latest = metric(latest, "val_f1")
    f1_delta = f1_latest - f1_base
    latest_epoch = int(latest["epoch"])
    pos = metric(latest, "val_interaction_heatmap_positive_mean")
    pred_max = metric(latest, "val_interaction_heatmap_pred_max")
    soft_iou = metric(latest, "val_interaction_heatmap_soft_iou")
    center_l2 = metric(latest, "val_interaction_heatmap_center_l2")
    laptop_pos = metric(latest, "val_interaction_heatmap_laptop_positive_mean")
    laptop_iou = metric(latest, "val_interaction_heatmap_laptop_iou")
    selection_acc = metric(latest, "val_object_selection_acc")
    selection_prob = metric(latest, "val_object_selection_true_prob")
    selection_none_count = metric(latest, "val_object_selection_none_teacher_count")
    selection_object_count = metric(latest, "val_object_selection_object_teacher_count")
    actor_all_slot = metric(latest, "val_actor_all_slot_acc")
    actor_pair = metric(latest, "val_actor_pair_acc")
    actor_pair_swap = metric(latest, "val_actor_pair_swap_acc")
    actor_presence = metric(latest, "val_actor_presence_acc")
    cf_logit = metric(latest, "val_object_counterfactual_selected_logit_drop")
    cf_prob = metric(latest, "val_object_counterfactual_selected_prob_drop")

    print("\nDECISION:\n")
    print(f"latest epoch: {latest_epoch}")
    print(f"F1 latest/base: {fmt(f1_latest)} / {fmt(f1_base)} delta {fmt(f1_delta)}")
    print(
        "heatmap: "
        f"positive {fmt(pos)}, pred_max {fmt(pred_max)}, "
        f"soft_iou {fmt(soft_iou)}, center_l2 {fmt(center_l2)}"
    )
    if pd.notna(laptop_pos) or pd.notna(laptop_iou):
        print(f"laptop heatmap: positive {fmt(laptop_pos)}, iou {fmt(laptop_iou)}")
    if pd.notna(selection_acc) or pd.notna(selection_prob):
        print(
            "object selection: "
            f"acc {fmt(selection_acc)}, true_prob {fmt(selection_prob)}, "
            f"none_teachers {fmt(selection_none_count, 0)}, "
            f"object_teachers {fmt(selection_object_count, 0)}"
        )
    if (
        pd.notna(actor_all_slot)
        or pd.notna(actor_pair)
        or pd.notna(actor_pair_swap)
        or pd.notna(actor_presence)
    ):
        print(
            "actor slots: "
            f"all_slot {fmt(actor_all_slot)}, pair {fmt(actor_pair)}, "
            f"swap {fmt(actor_pair_swap)}, presence {fmt(actor_presence)}"
        )
    if pd.notna(cf_logit) or pd.notna(cf_prob):
        print(
            "selected-object removal: "
            f"logit_drop {fmt(cf_logit)}, prob_drop {fmt(cf_prob)}"
        )

    if pd.notna(f1_delta) and f1_delta < -0.01:
        print("STOP/ROLL BACK: action F1 dropped more than 0.01 from epoch 0.")
        return
    if pd.notna(cf_logit):
        if cf_logit > 0.02 and pd.notna(selection_acc) and selection_acc > 0.20:
            print(
                "GOOD SUPPORTING SIGN: selected-object removal affects the "
                "true action logit."
            )
        elif cf_logit <= 0.0:
            print(
                "WARNING: selected-object removal is not lowering the true "
                "action logit yet."
            )
        else:
            print("CONTINUE/COMPARE: object path is partly learning; action dependence is still weak.")
        return
    if pd.notna(pos) and pd.notna(pred_max):
        if pos > 0.05 and pred_max > 0.10:
            print(
                "GOOD HEATMAP SIGN: actor-object heatmaps are responding; "
                "use object-token metrics for final proof."
            )
        else:
            print("CONTINUE: actor-object heatmaps are not strong yet.")
        return
    print("INSUFFICIENT SIGNAL: no object-use metrics were found in this run.")


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
    if pd.notna(metric(row, "val_object_selection_acc")):
        print(
            "object selection: "
            f"loss {metric(row, 'val_object_selection_loss'):.4f}, "
            f"acc {metric(row, 'val_object_selection_acc'):.4f}, "
            f"none_acc {metric(row, 'val_object_selection_none_acc'):.4f}, "
            f"object_acc {metric(row, 'val_object_selection_object_acc'):.4f}, "
            f"true_prob {metric(row, 'val_object_selection_true_prob'):.4f}, "
            f"teachers {metric(row, 'val_object_selection_teacher_count'):.1f}, "
            f"none_teachers {metric(row, 'val_object_selection_none_teacher_count'):.0f}, "
            f"object_teachers {metric(row, 'val_object_selection_object_teacher_count'):.0f}"
        )
    if pd.notna(metric(row, "val_object_counterfactual_selected_logit_drop")):
        print(
            "selected-object counterfactual: "
            f"logit_drop {metric(row, 'val_object_counterfactual_selected_logit_drop'):.4f}, "
            f"prob_drop {metric(row, 'val_object_counterfactual_selected_prob_drop'):.4f}"
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
        print("\nHEATMAP BY SELECTED OBJECT CLASS:")
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
    display_cols = [col for col in CORE_COLUMNS if col in epoch_df.columns]
    print("run:", run)
    print("metrics:", metrics)

    latest = epoch_df.iloc[-1]

    if args.verbose:
        print("\nEPOCH SUMMARY:\n")
        print(epoch_df[display_cols].to_string(index=False))
        print_object_use_epoch_table(epoch_df)
        print_object_channel_progress(epoch_df)
        print_action_progress(epoch_df)
        print_group_progress(epoch_df)
        print_best_epochs(epoch_df)
        print_row("LATEST", latest)
    else:
        print_compact_object_use_summary(epoch_df)
        print_compact_target_actions(epoch_df)
        print_compact_best(epoch_df)

    print_decision(epoch_df)

    print("\nREAD THIS:")
    print("- Main proof: val_f1/per-action accuracy stay healthy while object selection and heatmaps improve.")
    print("- NONE selection should be strong for objectless actions, so detector misses do not dominate.")
    print("- Counterfactual selected-object removal is supporting evidence, not the checkpoint target by itself.")
    print("- Guardrail: val_f1/per-action target accuracy should not collapse while object-use metrics rise.")
    print("- Heatmap/object-channel metrics are secondary; use --verbose when debugging teacher quality.")
    print("- With --scene_object_tokens 1, RF-DETR boxes are runtime model inputs as well as teacher labels.")


if __name__ == "__main__":
    main()
