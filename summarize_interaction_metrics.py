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
    "val_deploy_key_action_mean",
    "val_deploy_key_action_min",
    "val_deploy_object_dropout_action_acc",
    "val_deploy_object_dropout_Uselaptop_acc",
    "val_deploy_object_dropout_joint_missing_acc",
    "val_deploy_object_dropout_joint_Uselaptop_acc",
    "val_deploy_object_present_true_prob_gain",
    "val_deploy_object_present_true_logit_gain",
    "val_deploy_object_present_action_margin_gain",
    "val_deploy_object_present_Uselaptop_prob_gain",
    "val_deploy_object_present_Uselaptop_confuser_margin_gain",
    "val_actor_object_fusion_delta_norm",
    "val_actor_all_slot_acc",
    "val_actor_pair_acc",
    "val_actor_pair_swap_acc",
    "val_actor_pair_same_acc",
    "val_actor_presence_acc",
    "val_loss_main_deploy",
    "val_loss_heatmap_aux",
    "val_loss_actor_object_relation",
    "val_relation_exact_teacher_acc",
    "val_relation_exact_teacher_prob",
    "val_relation_useful_mass_exact",
    "val_relation_null_prob_exact",
    "val_relation_null_rate_objectless",
    "val_relation_null_prob_objectless",
    "val_relation_useful_mass_objectless",
    "val_relation_null_rate_missing_objectful",
    "val_relation_null_prob_missing_objectful",
    "val_relation_useful_mass_missing_objectful",
    "val_relation_logit_scale",
    "val_relation_valid_object_logit_bonus",
    "val_relation_action_joint_balanced_acc",
    "val_relation_action_joint_acc",
    "val_relation_action_joint_exact_acc",
    "val_relation_action_joint_objectless_acc",
    "val_relation_action_joint_missing_objectful_acc",
    "val_relation_correct_action_wrong_exact_rate",
    "val_action_correct_relation_wrong_exact_rate",
    "val_action_acc_when_relation_exact",
    "val_relation_exact_when_action_correct",
    "val_relation_action_joint_Uselaptop_acc",
    "val_relation_correct_action_wrong_Uselaptop_rate",
    "val_relation_action_joint_missing_Uselaptop_acc",
    "val_relation_correct_action_wrong_missing_Uselaptop_rate",
    "val_action_Uselaptop_object_confuser_margin",
    "val_action_Uselaptop_object_confuser_win_rate",
    "val_action_Uselaptop_missing_object_confuser_margin",
    "val_action_Uselaptop_missing_object_confuser_win_rate",
    "val_object_dropout_actor_rate",
    "val_object_dropout_object_drop_rate",
    "val_object_dropout_action_acc",
    "val_object_dropout_action_Uselaptop_acc",
    "val_object_dropout_relation_null_rate_missing_objectful",
    "val_object_dropout_relation_action_joint_missing_objectful_acc",
    "val_object_dropout_relation_action_joint_missing_Uselaptop_acc",
    "val_object_dropout_relation_correct_action_wrong_missing_Uselaptop_rate",
    "val_object_dropout_action_Uselaptop_missing_object_confuser_margin",
    "val_object_dropout_action_Uselaptop_missing_object_confuser_win_rate",
    "val_object_dropout_object_present_true_prob_gain",
    "val_object_dropout_object_present_action_margin_gain",
    "val_object_dropout_object_present_Uselaptop_prob_gain",
    "val_object_dropout_object_present_Uselaptop_confuser_margin_gain",
    "val_actor_object_prompt_token_count",
    "val_token_selection_visual_keep_rate",
    "val_token_selection_actor_box_keep_rate",
    "val_token_selection_visible_object_box_keep_rate",
    "val_token_selection_exact_teacher_object_keep_rate",
    "val_token_selection_interaction_heatmap_keep_rate",
    "val_token_selection_laptop_box_keep_rate",
    "val_token_selection_book_box_keep_rate",
    "val_token_selection_phone_box_keep_rate",
    "val_token_selection_tv_monitor_box_keep_rate",
    "val_token_selection_Uselaptop_teacher_object_keep_rate",
    "val_interaction_teacher_slot_rate",
    "val_interaction_teacher_slot_count",
    "val_interaction_heatmap_missing_object_masked_rate",
    "val_interaction_heatmap_missing_object_masked_count",
    "val_interaction_heatmap_exact_compatible_valid_rate",
    "val_interaction_heatmap_mismatch_valid_rate",
    "val_interaction_heatmap_iou",
    "val_interaction_heatmap_soft_iou",
    "val_interaction_heatmap_positive_mean",
    "val_interaction_heatmap_pred_max",
    "val_interaction_heatmap_target_max",
    "val_interaction_heatmap_center_l2",
    "val_objectless_with_object_visible_acc",
    "val_objectless_with_object_visible_count",
    "val_objectless_with_object_visible_object_action_pred_rate",
    "val_objectless_with_laptop_visible_acc",
    "val_objectless_with_laptop_visible_count",
    "val_objectless_with_book_visible_acc",
    "val_objectless_with_book_visible_count",
    "val_objectless_with_phone_visible_acc",
    "val_objectless_with_phone_visible_count",
    "val_watchtv_fp_rate_objectless",
    "train_nash_weight_action",
    "train_nash_weight_heatmap",
    "train_nash_weight_main_deploy",
    "train_nash_weight_heatmap_aux",
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
        "val_actor_all_slot_acc",
        "val_actor_pair_acc",
        "val_group_object_mapped_acc",
        "val_group_objectless_acc",
        "val_group_laptop_book_tv_acc",
        "val_group_phone_tv_acc",
        "val_loss_actor_object_relation",
        "val_relation_exact_teacher_acc",
        "val_relation_exact_teacher_prob",
        "val_relation_useful_mass_exact",
        "val_relation_null_prob_exact",
        "val_relation_null_rate_objectless",
        "val_relation_null_prob_objectless",
        "val_relation_useful_mass_objectless",
        "val_relation_null_rate_missing_objectful",
        "val_relation_null_prob_missing_objectful",
        "val_relation_useful_mass_missing_objectful",
        "val_relation_logit_scale",
        "val_relation_valid_object_logit_bonus",
        "val_object_region_visual_feature_norm",
        "val_relation_action_joint_balanced_acc",
        "val_relation_action_joint_acc",
        "val_relation_action_joint_exact_acc",
        "val_relation_correct_action_wrong_exact_rate",
        "val_relation_action_joint_Uselaptop_acc",
        "val_relation_correct_action_wrong_Uselaptop_rate",
        "val_action_Uselaptop_object_confuser_margin",
        "val_action_Uselaptop_object_confuser_win_rate",
        "val_object_dropout_action_acc",
        "val_object_dropout_action_Uselaptop_acc",
        "val_object_dropout_relation_action_joint_missing_Uselaptop_acc",
        "val_object_dropout_action_Uselaptop_missing_object_confuser_margin",
        "val_deploy_object_present_Uselaptop_prob_gain",
        "val_deploy_object_present_Uselaptop_confuser_margin_gain",
        "val_deploy_object_present_true_logit_gain",
        "val_deploy_object_present_action_margin_gain",
        "val_actor_object_fusion_delta_norm",
        "val_actor_object_prompt_token_count",
        "val_token_selection_visual_keep_rate",
        "val_token_selection_actor_box_keep_rate",
        "val_token_selection_visible_object_box_keep_rate",
        "val_token_selection_exact_teacher_object_keep_rate",
        "val_token_selection_interaction_heatmap_keep_rate",
        "val_token_selection_laptop_box_keep_rate",
        "val_token_selection_Uselaptop_teacher_object_keep_rate",
        "val_interaction_heatmap_positive_mean",
        "val_interaction_heatmap_pred_max",
        "val_interaction_heatmap_soft_iou",
        "val_interaction_heatmap_center_l2",
    ]
    display_cols = [col for col in cols if col in epoch_df.columns]
    if len(display_cols) > 1:
        print("\nEPOCH SNAPSHOT:\n")
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
        "val_deploy_object_dropout_action_acc",
        "val_deploy_object_dropout_Uselaptop_acc",
        "val_deploy_object_dropout_joint_missing_acc",
        "val_deploy_object_dropout_joint_Uselaptop_acc",
        "val_group_object_mapped_acc",
        "val_group_objectless_acc",
        "val_group_laptop_book_tv_acc",
        "val_group_phone_tv_acc",
        "val_loss_actor_object_relation",
        "val_relation_exact_teacher_acc",
        "val_relation_exact_teacher_prob",
        "val_relation_useful_mass_exact",
        "val_relation_null_prob_exact",
        "val_relation_null_rate_objectless",
        "val_relation_null_prob_objectless",
        "val_relation_useful_mass_objectless",
        "val_relation_null_rate_missing_objectful",
        "val_relation_null_prob_missing_objectful",
        "val_relation_useful_mass_missing_objectful",
        "val_relation_logit_scale",
        "val_relation_valid_object_logit_bonus",
        "val_object_region_visual_feature_norm",
        "val_relation_action_joint_balanced_acc",
        "val_relation_action_joint_acc",
        "val_relation_action_joint_exact_acc",
        "val_relation_action_joint_objectless_acc",
        "val_relation_action_joint_missing_objectful_acc",
        "val_relation_correct_action_wrong_exact_rate",
        "val_action_correct_relation_wrong_exact_rate",
        "val_action_acc_when_relation_exact",
        "val_relation_exact_when_action_correct",
        "val_relation_action_joint_Uselaptop_acc",
        "val_relation_correct_action_wrong_Uselaptop_rate",
        "val_relation_action_joint_missing_Uselaptop_acc",
        "val_relation_correct_action_wrong_missing_Uselaptop_rate",
        "val_action_Uselaptop_object_confuser_margin",
        "val_action_Uselaptop_object_confuser_win_rate",
        "val_action_Uselaptop_missing_object_confuser_margin",
        "val_action_Uselaptop_missing_object_confuser_win_rate",
        "val_object_dropout_actor_rate",
        "val_object_dropout_object_drop_rate",
        "val_object_dropout_action_acc",
        "val_object_dropout_action_Uselaptop_acc",
        "val_object_dropout_relation_null_rate_missing_objectful",
        "val_object_dropout_relation_action_joint_missing_objectful_acc",
        "val_object_dropout_relation_action_joint_missing_Uselaptop_acc",
        "val_object_dropout_relation_correct_action_wrong_missing_Uselaptop_rate",
        "val_object_dropout_action_Uselaptop_missing_object_confuser_margin",
        "val_object_dropout_action_Uselaptop_missing_object_confuser_win_rate",
        "val_deploy_object_present_true_prob_gain",
        "val_deploy_object_present_true_logit_gain",
        "val_deploy_object_present_action_margin_gain",
        "val_deploy_object_present_Uselaptop_prob_gain",
        "val_deploy_object_present_Uselaptop_confuser_margin_gain",
        "val_actor_object_fusion_delta_norm",
        "val_token_selection_actor_box_keep_rate",
        "val_token_selection_visible_object_box_keep_rate",
        "val_token_selection_exact_teacher_object_keep_rate",
        "val_token_selection_interaction_heatmap_keep_rate",
        "val_token_selection_laptop_box_keep_rate",
        "val_token_selection_Uselaptop_teacher_object_keep_rate",
        "val_actor_object_prompt_token_count",
        "val_interaction_heatmap_soft_iou",
        "val_interaction_heatmap_positive_mean",
        "val_interaction_heatmap_laptop_positive_mean",
        "val_interaction_heatmap_laptop_iou",
        "val_action_Uselaptop_acc",
        "val_action_Readbook_acc",
        "val_action_WatchTV_acc",
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
        best_idx = valid.idxmin() if metric_prefers_lower(name) else valid.idxmax()
        rows.append(
            {
                "metric": name,
                "best_epoch": epoch_df.loc[best_idx, "epoch"],
                "best_value": series.loc[best_idx],
            }
        )
    print_table("BEST COMPACT SIGNALS", rows, ["metric", "best_epoch", "best_value"])


def print_decision(epoch_df):
    base = epoch_df.iloc[0]
    latest = epoch_df.iloc[-1]
    f1_base = metric(base, "val_f1")
    f1_latest = metric(latest, "val_f1")
    f1_delta = f1_latest - f1_base
    latest_epoch = int(latest["epoch"])

    deploy_score = metric(latest, "val_deploy_score")
    deploy_key_mean = metric(latest, "val_deploy_key_action_mean")
    deploy_key_min = metric(latest, "val_deploy_key_action_min")
    object_mapped_acc = metric(latest, "val_group_object_mapped_acc")
    objectless_acc = metric(latest, "val_group_objectless_acc")
    relation_loss = metric(latest, "val_loss_actor_object_relation")
    relation_exact_acc = metric(latest, "val_relation_exact_teacher_acc")
    relation_exact_prob = metric(latest, "val_relation_exact_teacher_prob")
    relation_useful_exact = metric(latest, "val_relation_useful_mass_exact")
    relation_null_exact = metric(latest, "val_relation_null_prob_exact")
    relation_null_objectless = metric(latest, "val_relation_null_rate_objectless")
    relation_null_prob_objectless = metric(latest, "val_relation_null_prob_objectless")
    relation_useful_objectless = metric(latest, "val_relation_useful_mass_objectless")
    relation_null_missing = metric(latest, "val_relation_null_rate_missing_objectful")
    relation_null_prob_missing = metric(
        latest,
        "val_relation_null_prob_missing_objectful",
    )
    relation_useful_missing = metric(
        latest,
        "val_relation_useful_mass_missing_objectful",
    )
    relation_valid_bonus = metric(latest, "val_relation_valid_object_logit_bonus")
    relation_logit_scale = metric(latest, "val_relation_logit_scale")
    joint_balanced = metric(latest, "val_relation_action_joint_balanced_acc")
    joint_acc = metric(latest, "val_relation_action_joint_acc")
    joint_exact = metric(latest, "val_relation_action_joint_exact_acc")
    joint_objectless = metric(latest, "val_relation_action_joint_objectless_acc")
    joint_missing = metric(latest, "val_relation_action_joint_missing_objectful_acc")
    relation_correct_action_wrong = metric(
        latest,
        "val_relation_correct_action_wrong_exact_rate",
    )
    action_correct_relation_wrong = metric(
        latest,
        "val_action_correct_relation_wrong_exact_rate",
    )
    action_when_relation_exact = metric(latest, "val_action_acc_when_relation_exact")
    relation_when_action_correct = metric(
        latest,
        "val_relation_exact_when_action_correct",
    )
    uselaptop_joint = metric(latest, "val_relation_action_joint_Uselaptop_acc")
    uselaptop_relation_action_wrong = metric(
        latest,
        "val_relation_correct_action_wrong_Uselaptop_rate",
    )
    uselaptop_missing_joint = metric(
        latest,
        "val_relation_action_joint_missing_Uselaptop_acc",
    )
    uselaptop_missing_relation_action_wrong = metric(
        latest,
        "val_relation_correct_action_wrong_missing_Uselaptop_rate",
    )
    uselaptop_confuser_margin = metric(
        latest,
        "val_action_Uselaptop_object_confuser_margin",
    )
    uselaptop_confuser_win = metric(
        latest,
        "val_action_Uselaptop_object_confuser_win_rate",
    )
    uselaptop_missing_confuser_margin = metric(
        latest,
        "val_action_Uselaptop_missing_object_confuser_margin",
    )
    uselaptop_missing_confuser_win = metric(
        latest,
        "val_action_Uselaptop_missing_object_confuser_win_rate",
    )
    dropout_action = metric(latest, "val_object_dropout_action_acc")
    dropout_uselaptop_action = metric(
        latest,
        "val_object_dropout_action_Uselaptop_acc",
    )
    dropout_null_missing = metric(
        latest,
        "val_object_dropout_relation_null_rate_missing_objectful",
    )
    dropout_joint_missing = metric(
        latest,
        "val_object_dropout_relation_action_joint_missing_objectful_acc",
    )
    dropout_uselaptop_joint = metric(
        latest,
        "val_object_dropout_relation_action_joint_missing_Uselaptop_acc",
    )
    dropout_uselaptop_relation_action_wrong = metric(
        latest,
        "val_object_dropout_relation_correct_action_wrong_missing_Uselaptop_rate",
    )
    dropout_uselaptop_margin = metric(
        latest,
        "val_object_dropout_action_Uselaptop_missing_object_confuser_margin",
    )
    dropout_uselaptop_win = metric(
        latest,
        "val_object_dropout_action_Uselaptop_missing_object_confuser_win_rate",
    )
    object_present_true_gain = metric(
        latest,
        "val_deploy_object_present_true_prob_gain",
    )
    object_present_true_logit_gain = metric(
        latest,
        "val_deploy_object_present_true_logit_gain",
    )
    object_present_action_margin_gain = metric(
        latest,
        "val_deploy_object_present_action_margin_gain",
    )
    object_present_uselaptop_gain = metric(
        latest,
        "val_deploy_object_present_Uselaptop_prob_gain",
    )
    object_present_uselaptop_margin_gain = metric(
        latest,
        "val_deploy_object_present_Uselaptop_confuser_margin_gain",
    )
    fusion_delta_norm = metric(latest, "val_actor_object_fusion_delta_norm")

    pos = metric(latest, "val_interaction_heatmap_positive_mean")
    pred_max = metric(latest, "val_interaction_heatmap_pred_max")
    soft_iou = metric(latest, "val_interaction_heatmap_soft_iou")
    center_l2 = metric(latest, "val_interaction_heatmap_center_l2")
    laptop_pos = metric(latest, "val_interaction_heatmap_laptop_positive_mean")
    laptop_iou = metric(latest, "val_interaction_heatmap_laptop_iou")

    prompt_tokens = metric(latest, "val_actor_object_prompt_token_count")
    token_keep_visual = metric(latest, "val_token_selection_visual_keep_rate")
    token_keep_actor = metric(latest, "val_token_selection_actor_box_keep_rate")
    token_keep_visible_obj = metric(
        latest,
        "val_token_selection_visible_object_box_keep_rate",
    )
    token_keep_teacher = metric(
        latest,
        "val_token_selection_exact_teacher_object_keep_rate",
    )
    token_keep_interaction = metric(
        latest,
        "val_token_selection_interaction_heatmap_keep_rate",
    )
    token_keep_laptop = metric(latest, "val_token_selection_laptop_box_keep_rate")
    token_keep_book = metric(latest, "val_token_selection_book_box_keep_rate")
    token_keep_phone = metric(latest, "val_token_selection_phone_box_keep_rate")
    token_keep_tv = metric(latest, "val_token_selection_tv_monitor_box_keep_rate")
    token_keep_uselaptop_teacher = metric(
        latest,
        "val_token_selection_Uselaptop_teacher_object_keep_rate",
    )

    heatmap_missing_mask_rate = metric(
        latest,
        "val_interaction_heatmap_missing_object_masked_rate",
    )
    heatmap_missing_mask_count = metric(
        latest,
        "val_interaction_heatmap_missing_object_masked_count",
    )
    heatmap_exact_valid = metric(
        latest,
        "val_interaction_heatmap_exact_compatible_valid_rate",
    )
    heatmap_mismatch_valid = metric(
        latest,
        "val_interaction_heatmap_mismatch_valid_rate",
    )

    hard_objectless = metric(latest, "val_objectless_with_object_visible_acc")
    hard_objectless_count = metric(latest, "val_objectless_with_object_visible_count")
    hard_object_action_rate = metric(
        latest,
        "val_objectless_with_object_visible_object_action_pred_rate",
    )
    actor_all_slot = metric(latest, "val_actor_all_slot_acc")
    actor_pair = metric(latest, "val_actor_pair_acc")
    actor_pair_swap = metric(latest, "val_actor_pair_swap_acc")
    actor_presence = metric(latest, "val_actor_presence_acc")

    print("\nDECISION:\n")
    print(f"latest epoch: {latest_epoch}")
    if pd.notna(deploy_score):
        print(
            "deploy score: "
            f"{fmt(deploy_score)}, key_mean {fmt(deploy_key_mean)}, "
            f"key_min {fmt(deploy_key_min)}"
        )
    print(f"F1 latest/base: {fmt(f1_latest)} / {fmt(f1_base)} delta {fmt(f1_delta)}")
    print(
        "heatmap: "
        f"positive {fmt(pos)}, pred_max {fmt(pred_max)}, "
        f"soft_iou {fmt(soft_iou)}, center_l2 {fmt(center_l2)}"
    )
    if pd.notna(laptop_pos) or pd.notna(laptop_iou):
        print(f"laptop heatmap: positive {fmt(laptop_pos)}, iou {fmt(laptop_iou)}")
    if pd.notna(object_mapped_acc) or pd.notna(objectless_acc):
        print(
            "action groups: "
            f"object_mapped {fmt(object_mapped_acc)}, "
            f"objectless {fmt(objectless_acc)}"
        )
    if pd.notna(relation_exact_acc) or pd.notna(relation_null_objectless):
        print(
            "actor-object relation: "
            f"loss {fmt(relation_loss)}, "
            f"teacher_acc {fmt(relation_exact_acc)}, "
            f"teacher_prob {fmt(relation_exact_prob)}, "
            f"useful_exact {fmt(relation_useful_exact)}, "
            f"null_exact {fmt(relation_null_exact)}, "
            f"null_objectless {fmt(relation_null_objectless)}, "
            f"null_prob_objectless {fmt(relation_null_prob_objectless)}, "
            f"useful_objectless {fmt(relation_useful_objectless)}, "
            f"null_missing {fmt(relation_null_missing)}, "
            f"null_prob_missing {fmt(relation_null_prob_missing)}, "
            f"useful_missing {fmt(relation_useful_missing)}, "
            f"logit_scale {fmt(relation_logit_scale)}, "
            f"valid_bonus {fmt(relation_valid_bonus)}, "
            f"tokens {fmt(prompt_tokens, 0)}"
        )
    if pd.notna(joint_acc) or pd.notna(joint_exact):
        print(
            "relation-action joint: "
            f"balanced {fmt(joint_balanced)}, "
            f"all {fmt(joint_acc)}, exact {fmt(joint_exact)}, "
            f"objectless {fmt(joint_objectless)}, "
            f"missing {fmt(joint_missing)}, "
            f"rel_ok_action_wrong {fmt(relation_correct_action_wrong)}, "
            f"action_ok_rel_wrong {fmt(action_correct_relation_wrong)}, "
            f"action_when_rel_ok {fmt(action_when_relation_exact)}, "
            f"rel_when_action_ok {fmt(relation_when_action_correct)}"
        )
    if pd.notna(uselaptop_joint) or pd.notna(uselaptop_confuser_margin):
        print(
            "uselaptop joint: "
            f"joint {fmt(uselaptop_joint)}, "
            f"rel_ok_action_wrong {fmt(uselaptop_relation_action_wrong)}, "
            f"missing_joint {fmt(uselaptop_missing_joint)}, "
            f"missing_rel_ok_action_wrong {fmt(uselaptop_missing_relation_action_wrong)}, "
            f"confuser_margin {fmt(uselaptop_confuser_margin)}, "
            f"confuser_win {fmt(uselaptop_confuser_win)}, "
            f"missing_margin {fmt(uselaptop_missing_confuser_margin)}, "
            f"missing_win {fmt(uselaptop_missing_confuser_win)}"
        )
    if pd.notna(dropout_action) or pd.notna(dropout_uselaptop_joint):
        print(
            "object-dropout fallback: "
            f"action {fmt(dropout_action)}, "
            f"uselaptop_action {fmt(dropout_uselaptop_action)}, "
            f"null_missing {fmt(dropout_null_missing)}, "
            f"joint_missing {fmt(dropout_joint_missing)}, "
            f"uselaptop_joint {fmt(dropout_uselaptop_joint)}, "
            "uselaptop_rel_ok_action_wrong "
            f"{fmt(dropout_uselaptop_relation_action_wrong)}, "
            f"uselaptop_margin {fmt(dropout_uselaptop_margin)}, "
            f"uselaptop_win {fmt(dropout_uselaptop_win)}"
        )
    if pd.notna(object_present_true_gain) or pd.notna(object_present_uselaptop_gain):
        print(
            "object-present gain: "
            f"true_prob {fmt(object_present_true_gain)}, "
            f"true_logit {fmt(object_present_true_logit_gain)}, "
            f"action_margin {fmt(object_present_action_margin_gain)}, "
            f"uselaptop_prob {fmt(object_present_uselaptop_gain)}, "
            f"uselaptop_margin {fmt(object_present_uselaptop_margin_gain)}"
        )
    if pd.notna(fusion_delta_norm):
        print(f"learned object fusion: delta_norm {fmt(fusion_delta_norm)}")
    if pd.notna(token_keep_visual):
        print(
            "token-selection retention: "
            f"visual {fmt(token_keep_visual)}, "
            f"actor_box {fmt(token_keep_actor)}, "
            f"visible_obj_box {fmt(token_keep_visible_obj)}, "
            f"teacher_obj_box {fmt(token_keep_teacher)}, "
            f"interaction_hm {fmt(token_keep_interaction)}"
        )
    if (
        pd.notna(token_keep_laptop)
        or pd.notna(token_keep_uselaptop_teacher)
        or pd.notna(token_keep_book)
    ):
        print(
            "token-selection object retention: "
            f"laptop_box {fmt(token_keep_laptop)}, "
            f"book_box {fmt(token_keep_book)}, "
            f"phone_box {fmt(token_keep_phone)}, "
            f"tv_box {fmt(token_keep_tv)}, "
            f"uselaptop_teacher {fmt(token_keep_uselaptop_teacher)}"
        )
    if pd.notna(heatmap_missing_mask_rate):
        print(
            "missing-aware heatmap supervision: "
            f"masked_rate {fmt(heatmap_missing_mask_rate)}, "
            f"masked_count {fmt(heatmap_missing_mask_count, 0)}, "
            f"exact_valid {fmt(heatmap_exact_valid)}, "
            f"mismatch_valid {fmt(heatmap_mismatch_valid)}"
        )
    if pd.notna(hard_objectless):
        print(
            "objectless hard negatives: "
            f"acc {fmt(hard_objectless)}, count {fmt(hard_objectless_count, 0)}, "
            f"object_action_pred_rate {fmt(hard_object_action_rate)}"
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

    if pd.notna(f1_delta) and f1_delta < -0.01:
        print("STOP/ROLL BACK: action F1 dropped more than 0.01 from epoch 0.")
        return
    if pd.notna(dropout_uselaptop_action) and dropout_uselaptop_action < 0.70:
        print(
            "FALLBACK FAIL: Uselaptop is weak when the compatible object is "
            "dropped from detector inputs."
        )
        return
    if pd.notna(dropout_uselaptop_joint) and dropout_uselaptop_joint < 0.70:
        print(
            "FALLBACK FAIL: Uselaptop action and NULL relation do not agree under "
            "object dropout."
        )
        return
    if pd.notna(object_present_uselaptop_gain) and object_present_uselaptop_gain < -0.02:
        print(
            "OBJECT PATH FAIL: detected object evidence is hurting Uselaptop "
            "relative to the object-hidden pass."
        )
        return
    if (
        pd.notna(object_present_action_margin_gain)
        and object_present_action_margin_gain < -0.02
    ):
        print(
            "OBJECT PATH FAIL: detected object evidence is hurting the correct "
            "action margin relative to the object-hidden pass."
        )
        return
    if pd.notna(fusion_delta_norm) and latest_epoch >= 2 and fusion_delta_norm < 0.05:
        print(
            "OBJECT FUSION WARNING: learned object-context fusion is nearly inactive."
        )
        return
    if pd.notna(dropout_joint_missing) and dropout_joint_missing < 0.65:
        print(
            "CONTINUE: detector-miss fallback is improving but not strong enough yet."
        )
        return
    if pd.notna(joint_exact):
        if (
            pd.notna(relation_correct_action_wrong)
            and relation_correct_action_wrong > 0.15
        ):
            print(
                "ACTION COUPLING FAIL: relation is often correct while action is "
                "wrong on exact objectful cases."
            )
            return
        if joint_exact < 0.60:
            print(
                "NOT PROVEN: exact objectful relation and action are not jointly "
                "correct often enough yet."
            )
            return
        if joint_exact < 0.70:
            print(
                "CONTINUE: joint relation-action evidence is improving but is not "
                "strong enough yet."
            )
            return
        if (
            joint_exact >= 0.70
            and pd.notna(relation_exact_acc)
            and relation_exact_acc >= 0.65
        ):
            print("GOOD JOINT SIGN: exact object binding and action agree.")
            return
    if (
        pd.notna(relation_exact_acc)
        and relation_exact_acc >= 0.65
        and pd.notna(relation_null_objectless)
        and relation_null_objectless >= 0.90
    ):
        print("GOOD RELATION SIGN: actor-object slot assignment is learning.")
        return
    if pd.notna(pos) and pd.notna(pred_max):
        if pos > 0.05 and pred_max > 0.10:
            print("GOOD HEATMAP SIGN: actor-object heatmaps are responding.")
        else:
            print("CONTINUE: relation/heatmap evidence is not strong yet.")
        return
    print("INSUFFICIENT SIGNAL: no relation or heatmap metrics were found in this run.")


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
        "valid_bonus "
        f"{metric(row, 'val_relation_valid_object_logit_bonus'):.4f}, "
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
            "true_logit "
            f"{metric(row, 'val_deploy_object_present_true_logit_gain'):.4f}, "
            "action_margin "
            f"{metric(row, 'val_deploy_object_present_action_margin_gain'):.4f}, "
            "uselaptop_prob "
            f"{metric(row, 'val_deploy_object_present_Uselaptop_prob_gain'):.4f}, "
            "uselaptop_margin "
            f"{metric(row, 'val_deploy_object_present_Uselaptop_confuser_margin_gain'):.4f}"
        )
    if pd.notna(metric(row, "val_actor_object_fusion_delta_norm")):
        print(
            "learned object fusion: "
            f"delta_norm {metric(row, 'val_actor_object_fusion_delta_norm'):.4f}"
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

    print_decision(epoch_df)

    print("\nREAD THIS:")
    print("- Main proof: val_f1/per-action accuracy stay healthy while joint relation-action metrics improve.")
    print("- Detector-dropout metrics prove objectful actions still work when the compatible object token is hidden.")
    print("- For each actor, the object objective is one CE over NULL plus detected object slots.")
    print("- Runtime objects update actor tokens inside the transformer and feed selected object memory into actor_head.")
    print("- Object-present gain must be non-negative; detected objects should not hurt the true action versus an object-hidden pass.")
    print("- The only object objective is relation CE; removed side objectives are not part of this run.")
    print("- Exact objectful cases must get both action and relation right; relation-right/action-wrong is a coupling failure.")
    print("- Exact compatible detections should select the teacher object; missing/objectless cases should route relation attention to NULL.")
    print("- Objectless hard-negative metrics remain a protection check: visible objects must not force object actions.")


if __name__ == "__main__":
    main()
