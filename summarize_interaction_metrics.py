#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd

from datasets.object_vocab import OBJECT_CLASSES


GROUPS = [
    "laptop_book_tv",
    "phone_tablet",
    "phone_tv",
    "drink_cup_bottle_glass",
    "drink",
    "eat_pills",
    "eat",
    "cook_eat_kitchen",
    "object_mapped",
    "objectless",
]

ACTIONS = [
    "Cook_Cleandishes",
    "Cook_Cleanup",
    "Cut",
    "Cook_Cut",
    "Cook_Stir",
    "Cook_Usestove",
    "Cutbread",
    "Drink",
    "Drink_Frombottle",
    "Drink_Fromcan",
    "Drink_Fromcup",
    "Drink_Fromglass",
    "Eat_Attable",
    "Eat_Snack",
    "Enter",
    "Getup",
    "Laydown",
    "Leave",
    "Makecoffee_Pourgrains",
    "Makecoffee_Pourwater",
    "Maketea_Boilwater",
    "Maketea_Insertteabag",
    "Pour_Frombottle",
    "Pour_Fromcan",
    "Pour_Fromkettle",
    "Readbook",
    "Sitdown",
    "Takepills",
    "Uselaptop",
    "Usetablet",
    "Usetelephone",
    "Walk",
    "WatchTV",
]

KEY_ACTIONS = [
    "Uselaptop",
    "Readbook",
    "WatchTV",
    "Usetelephone",
    "Drink_Fromcup",
    "Drink",
    "Drink_Frombottle",
    "Drink_Fromglass",
    "Pour_Frombottle",
    "Cutbread",
    "Cook_Cut",
    "Cook_Stir",
    "Cook_Cleandishes",
]

OBJECTS = [OBJECT_CLASSES[index] for index in sorted(OBJECT_CLASSES)]

CORE_COLUMNS = [
    "epoch",
    "val_loss",
    "val_acc_macro",
    "val_f1",
    "val_deploy_score",
    "val_group_object_mapped_acc",
    "val_group_objectless_acc",
    "val_group_laptop_book_tv_acc",
    "val_group_phone_tv_acc",
    "val_loss_heatmap_aux",
    "val_object_dropout_action_acc",
    "val_object_dropout_action_Uselaptop_acc",
    "val_actor_object_prompt_token_count",
    "val_token_selection_actor_box_keep_rate",
    "val_token_selection_visible_object_box_keep_rate",
    "val_interaction_heatmap_iou",
    "val_interaction_heatmap_soft_iou",
    "val_interaction_heatmap_positive_mean",
]


def fmt(value, digits=4):
    if pd.isna(value):
        return "nan"
    return f"{float(value):.{digits}f}"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize actor-object relation and PO-GUISE+ metrics."
    )
    parser.add_argument(
        "--root",
        default="/mnt/local-scratch/poguise_data/checkpoints",
        help="Checkpoint root used when --run is not supplied.",
    )
    parser.add_argument(
        "--pattern",
        default="actor_object_relation_*",
        help="Run glob under --root used when --run is not supplied.",
    )
    parser.add_argument("--run", default=None, help="Specific run directory.")
    parser.add_argument("--metrics", default=None, help="Specific metrics.csv path.")
    parser.add_argument("--verbose", action="store_true")
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


def metric_prefers_lower(name):
    lower_markers = (
        "_loss",
        "_wrong_",
        "_object_action_pred_rate",
        "_center_l2",
    )
    return any(marker in name for marker in lower_markers)


def print_table(title, rows, columns=None):
    if not rows:
        return
    print(f"\n{title}:\n")
    table = pd.DataFrame(rows)
    if columns is not None:
        columns = [col for col in columns if col in table.columns]
        table = table[columns]
    empty_cols = [
        col
        for col in table.columns
        if col not in {"action", "group", "object", "metric"}
        and table[col].isna().all()
    ]
    if empty_cols:
        table = table.drop(columns=empty_cols)
    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        220,
        "display.max_colwidth",
        80,
    ):
        print(table.to_string(index=False))


def load_epoch_metrics(metrics):
    raw = pd.read_csv(metrics)
    if "epoch" not in raw.columns:
        raise SystemExit(f"{metrics} has no epoch column")
    epoch_df = raw.groupby("epoch", as_index=False).agg(
        {column: last_nonnull for column in raw.columns if column != "epoch"}
    )
    val_cols = [
        column
        for column in ("val_loss", "val_f1", "val_acc_macro", "val_deploy_score")
        if column in epoch_df.columns
    ]
    if val_cols:
        epoch_df = epoch_df.dropna(how="all", subset=val_cols)
    if epoch_df.empty:
        raise SystemExit(f"{metrics} has no completed validation epochs")
    return epoch_df


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
    base = epoch_df.iloc[0]
    latest = epoch_df.iloc[-1]
    rows = []
    for action in ACTIONS:
        acc_col = f"val_action_{action}_acc"
        if not df_has_metric(epoch_df, acc_col):
            continue
        series = df_metric(epoch_df, acc_col, float("nan"))
        valid = series.dropna()
        best_epoch = float("nan")
        best_acc = float("nan")
        if len(valid):
            best_idx = valid.idxmax()
            best_epoch = epoch_df.loc[best_idx, "epoch"]
            best_acc = series.loc[best_idx]
        delta = metric(latest, acc_col) - metric(base, acc_col)
        rows.append(
            {
                "action": action,
                "count": metric(latest, f"val_action_{action}_count"),
                "teacher": metric(
                    latest,
                    f"val_action_{action}_interaction_teacher_rate",
                ),
                "e0_acc": metric(base, acc_col),
                "latest_acc": metric(latest, acc_col),
                "delta": delta,
                "best_epoch": best_epoch,
                "best_acc": best_acc,
                "verdict": action_verdict(delta),
            }
        )
    return rows


def print_compact_epoch_summary(epoch_df):
    cols = [
        "epoch",
        "val_acc_macro",
        "val_f1",
        "val_group_object_mapped_acc",
        "val_group_objectless_acc",
        "val_loss_heatmap_aux",
        "val_object_dropout_action_acc",
        "val_object_dropout_action_Uselaptop_acc",
        "val_interaction_heatmap_positive_mean",
        "val_interaction_heatmap_soft_iou",
    ]
    display_cols = [col for col in cols if col in epoch_df.columns]
    if len(display_cols) > 1:
        print("
EPOCH SNAPSHOT:
")
        with pd.option_context("display.max_columns", None, "display.width", 220):
            print(epoch_df[display_cols].to_string(index=False))


def print_key_action_progress(epoch_df):
    rows = [row for row in action_progress_rows(epoch_df) if row["action"] in KEY_ACTIONS]
    print_table(
        "KEY ACTION PROGRESS",
        rows,
        [
            "action",
            "count",
            "teacher",
            "e0_acc",
            "latest_acc",
            "delta",
            "best_epoch",
            "best_acc",
            "verdict",
        ],
    )


def print_action_progress(epoch_df):
    print_table(
        "ALL ACTION PROGRESS",
        action_progress_rows(epoch_df),
        [
            "action",
            "count",
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
                "count": metric(latest, f"val_group_{group}_count"),
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
        ["group", "count", "e0_acc", "latest_acc", "delta", "best_epoch", "best_acc"],
    )


def print_object_channel_progress(epoch_df):
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
        series = df_metric(epoch_df, pos_col, float("nan"))
        valid = series.dropna()
        best_epoch = float("nan")
        best_pos = float("nan")
        if len(valid):
            best_idx = valid.idxmax()
            best_epoch = epoch_df.loc[best_idx, "epoch"]
            best_pos = series.loc[best_idx]
        rows.append(
            {
                "object": object_name,
                "teachers": metric(latest, count_col),
                "latest_pos": metric(latest, pos_col),
                "latest_iou": metric(latest, iou_col),
                "best_epoch": best_epoch,
                "best_pos": best_pos,
            }
        )
    print_table(
        "OBJECT CHANNEL PROGRESS",
        rows,
        ["object", "teachers", "latest_pos", "latest_iou", "best_epoch", "best_pos"],
    )


def print_compact_best(epoch_df):
    metrics = [
        "val_deploy_score",
        "val_f1",
        "val_acc_macro",
        "val_group_object_mapped_acc",
        "val_group_objectless_acc",
        "val_object_dropout_action_acc",
        "val_object_dropout_action_Uselaptop_acc",
        "val_interaction_heatmap_soft_iou",
        "val_interaction_heatmap_positive_mean",
        "val_action_Uselaptop_acc",
        "val_action_Readbook_acc",
    ]
    rows = []
    for name in metrics:
        if not df_has_metric(epoch_df, name):
            continue
        series = df_metric(epoch_df, name, float("nan"))
        valid = series.dropna()
        if not len(valid):
            continue
        best_idx = valid.idxmin() if metric_prefers_lower(name) else valid.idxmax()
        rows.append(
            {
                "metric": name,
                "best_epoch": epoch_df.loc[best_idx, "epoch"],
                "best_value": series.loc[best_idx],
            }
        )
    print_table("BEST COMPACT SIGNALS", rows, ["metric", "best_epoch", "best_value"])





def print_row(title, row):
    print(f"\n{title}: epoch {int(row['epoch'])}")
    print(f"val_loss: {metric(row, 'val_loss'):.4f}")
    print(
        "macro/f1: "
        f"{metric(row, 'val_acc_macro'):.4f} / {metric(row, 'val_f1'):.4f}"
    )
    if pd.notna(metric(row, "val_deploy_score")):
        print(
            "deploy score: "
            f"{metric(row, 'val_deploy_score'):.4f}, "
            f"key_mean {metric(row, 'val_deploy_key_action_mean'):.4f}, "
            f"key_min {metric(row, 'val_deploy_key_action_min'):.4f}"
        )
    print(
        "actor-object relation: "
        f"loss {metric(row, 'val_loss_actor_object_relation'):.4f}, "
        f"teacher_acc {metric(row, 'val_relation_exact_teacher_acc'):.4f}, "
        f"teacher_prob {metric(row, 'val_relation_exact_teacher_prob'):.4f}, "
        f"useful_exact {metric(row, 'val_relation_useful_mass_exact'):.4f}, "
        f"null_objectless {metric(row, 'val_relation_null_rate_objectless'):.4f}, "
        "null_missing "
        f"{metric(row, 'val_relation_null_rate_missing_objectful'):.4f}, "
        "logit_scale "
        f"{metric(row, 'val_relation_logit_scale'):.4f}, "
        "roi_norm "
        f"{metric(row, 'val_object_region_visual_feature_norm'):.4f}"
    )
    if pd.notna(metric(row, "val_relation_action_joint_exact_acc")):
        print(
            "relation-action joint: "
            f"balanced {metric(row, 'val_relation_action_joint_balanced_acc'):.4f}, "
            f"exact {metric(row, 'val_relation_action_joint_exact_acc'):.4f}, "
            "rel_ok_action_wrong "
            f"{metric(row, 'val_relation_correct_action_wrong_exact_rate'):.4f}, "
            "uselaptop_joint "
            f"{metric(row, 'val_relation_action_joint_Uselaptop_acc'):.4f}, "
            "uselaptop_margin "
            f"{metric(row, 'val_action_Uselaptop_object_confuser_margin'):.4f}"
        )
    if pd.notna(metric(row, "val_object_dropout_action_acc")):
        print(
            "object-dropout fallback: "
            f"action {metric(row, 'val_object_dropout_action_acc'):.4f}, "
            "uselaptop_action "
            f"{metric(row, 'val_object_dropout_action_Uselaptop_acc'):.4f}, "
            "uselaptop_joint "
            f"{metric(row, 'val_object_dropout_relation_action_joint_missing_Uselaptop_acc'):.4f}, "
            "uselaptop_margin "
            f"{metric(row, 'val_object_dropout_action_Uselaptop_missing_object_confuser_margin'):.4f}"
        )
    if pd.notna(metric(row, "val_deploy_object_present_true_prob_gain")):
        print(
            "object-present gain: "
            f"true_prob {metric(row, 'val_deploy_object_present_true_prob_gain'):.4f}, "
            "true_prob_median "
            f"{metric(row, 'val_deploy_object_present_true_prob_gain_median'):.4f}, "
            "true_prob_neg "
            f"{metric(row, 'val_deploy_object_present_true_prob_gain_negative_rate'):.4f}, "
            "true_logit "
            f"{metric(row, 'val_deploy_object_present_true_logit_gain'):.4f}, "
            "action_margin "
            f"{metric(row, 'val_deploy_object_present_action_margin_gain'):.4f}, "
            "action_margin_median "
            f"{metric(row, 'val_deploy_object_present_action_margin_gain_median'):.4f}, "
            "action_margin_neg "
            f"{metric(row, 'val_deploy_object_present_action_margin_gain_negative_rate'):.4f}, "
            "uselaptop_prob "
            f"{metric(row, 'val_deploy_object_present_Uselaptop_prob_gain'):.4f}, "
            "uselaptop_prob_median "
            f"{metric(row, 'val_deploy_object_present_Uselaptop_prob_gain_median'):.4f}, "
            "uselaptop_prob_neg "
            f"{metric(row, 'val_deploy_object_present_Uselaptop_prob_gain_negative_rate'):.4f}, "
            "uselaptop_margin "
            f"{metric(row, 'val_deploy_object_present_Uselaptop_confuser_margin_gain'):.4f}"
        )
    if pd.notna(metric(row, "val_actor_object_pair_action_score_abs")):
        print(
            "pair action scoring: "
            f"score_abs {metric(row, 'val_actor_object_pair_action_score_abs'):.4f}, "
            f"margin {metric(row, 'val_actor_object_pair_action_margin'):.4f}, "
            "win "
            f"{metric(row, 'val_actor_object_pair_action_margin_win_rate'):.4f}, "
            "uselaptop_margin "
            f"{metric(row, 'val_actor_object_pair_action_Uselaptop_margin'):.4f}, "
            "uselaptop_win "
            f"{metric(row, 'val_actor_object_pair_action_Uselaptop_margin_win_rate'):.4f}"
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
    if pd.notna(metric(row, "val_token_selection_visual_keep_rate")):
        print(
            "token-selection retention: "
            f"visual {metric(row, 'val_token_selection_visual_keep_rate'):.4f}, "
            f"actor_box {metric(row, 'val_token_selection_actor_box_keep_rate'):.4f}, "
            "visible_obj_box "
            f"{metric(row, 'val_token_selection_visible_object_box_keep_rate'):.4f}, "
            "teacher_obj_box "
            f"{metric(row, 'val_token_selection_exact_teacher_object_keep_rate'):.4f}, "
            "interaction_hm "
            f"{metric(row, 'val_token_selection_interaction_heatmap_keep_rate'):.4f}, "
            f"laptop_box {metric(row, 'val_token_selection_laptop_box_keep_rate'):.4f}, "
            "uselaptop_teacher "
            f"{metric(row, 'val_token_selection_Uselaptop_teacher_object_keep_rate'):.4f}"
        )
    if pd.notna(metric(row, "val_objectless_with_object_visible_acc")):
        print(
            "objectless hard negatives: "
            f"acc {metric(row, 'val_objectless_with_object_visible_acc'):.4f}, "
            f"count {metric(row, 'val_objectless_with_object_visible_count'):.0f}, "
            "object_action_pred_rate "
            f"{metric(row, 'val_objectless_with_object_visible_object_action_pred_rate'):.4f}"
        )
    print(
        "poguise+ heatmap loss: "
        f"log {metric(row, 'val_loss_heatmap_log'):.4f}, "
        f"aux {metric(row, 'val_loss_heatmap_aux'):.4f}, "
        f"fro {metric(row, 'val_loss_heatmap_frobenius'):.4f}, "
        f"pose_fro {metric(row, 'val_loss_pose_heatmap_frobenius'):.4f}, "
        f"interaction_fro {metric(row, 'val_loss_interaction_heatmap_frobenius'):.4f}"
    )
    nash_main = metric(row, "train_nash_weight_main_deploy")
    if pd.isna(nash_main):
        nash_main = metric(row, "train_nash_weight_action")
    nash_aux = metric(row, "train_nash_weight_heatmap_aux")
    if pd.isna(nash_aux):
        nash_aux = metric(row, "train_nash_weight_heatmap")
    print(
        "nash weights: "
        f"main_deploy {nash_main:.4f}, "
        f"heatmap_aux {nash_aux:.4f}"
    )


def main():
    args = parse_args()
    run, metrics = resolve_metrics(args)
    epoch_df = load_epoch_metrics(metrics)
    display_cols = [col for col in CORE_COLUMNS if col in epoch_df.columns]

    print("run:", run)
    print("metrics:", metrics)
    if args.verbose:
        print("\nEPOCH SUMMARY:\n")
        with pd.option_context("display.max_columns", None, "display.width", 240):
            print(epoch_df[display_cols].to_string(index=False))
        print_object_channel_progress(epoch_df)
        print_action_progress(epoch_df)
        print_group_progress(epoch_df)
        print_compact_best(epoch_df)
        print_row("LATEST", epoch_df.iloc[-1])
    else:
        print_compact_epoch_summary(epoch_df)
        print_key_action_progress(epoch_df)
        print_group_progress(epoch_df)
        print_compact_best(epoch_df)

    print("\nREAD THIS:")
    print("- Main proof: val_f1/per-action accuracy stay healthy while joint relation-action metrics improve.")
    print("- Detector-dropout metrics prove objectful actions still work when the compatible object token is hidden.")
    print("- For each actor, the object objective is one CE over NULL plus detected object slots.")
    print("- Runtime objects update actor tokens inside the transformer and feed ROI object memory into pair action scoring.")
    print("- Object-present gain must be non-negative; detected objects should not hurt the true action versus an object-hidden pass.")
    print("- The only object objective is relation CE; removed side objectives are not part of this run.")
    print("- Exact objectful cases must get both action and relation right; relation-right/action-wrong is a coupling failure.")
    print("- Exact compatible detections should select the teacher object; missing/objectless cases should route relation attention to NULL.")
    print("- Objectless hard-negative metrics remain a protection check: visible objects must not force object actions.")


if __name__ == "__main__":
    main()
