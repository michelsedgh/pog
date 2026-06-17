#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd

from datasets.object_vocab import OBJECT_CLASSES


CORE_COLUMNS = [
    "epoch",
    "val_loss",
    "val_acc_macro",
    "val_f1",
    "val_deploy_score",
    "val_deploy_key_action_mean",
    "val_deploy_key_action_min",
    "val_deploy_objectless_with_object_visible_acc",
    "val_deploy_objectless_with_object_visible_object_action_pred_rate",
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
    "val_loss_heatmap_optimized",
    "val_loss_heatmap_frobenius",
    "val_loss_pose_heatmap_frobenius",
    "val_loss_pose_heatmap_mse_scaled",
    "val_loss_interaction_heatmap_raw_frobenius",
    "val_loss_interaction_heatmap_frobenius",
    "val_loss_interaction_heatmap_mse_scaled",
    "val_loss_interaction_heatmap_pos_balanced",
    "val_loss_interaction_heatmap_center",
    "val_loss_main_deploy",
    "val_loss_grounding_aux",
    "val_loss_actor_object_relation",
    "val_loss_actor_object_engagement",
    "val_actor_object_engagement_acc",
    "val_actor_object_engagement_none_acc",
    "val_actor_object_engagement_laptop_acc",
    "val_actor_object_engagement_book_acc",
    "val_actor_object_engagement_phone_tablet_acc",
    "val_actor_object_engagement_tv_monitor_acc",
    "val_actor_object_engagement_laptop_book_tv_acc",
    "val_loss_actor_object_binding_state",
    "val_loss_actor_object_binding_action",
    "val_actor_object_binding_state_margin",
    "val_actor_object_binding_action_margin",
    "val_actor_object_binding_state_pass_rate",
    "val_actor_object_binding_action_pass_rate",
    "val_actor_object_binding_count",
    "val_relation_exact_teacher_acc",
    "val_relation_exact_teacher_prob",
    "val_relation_null_rate_objectless",
    "val_relation_null_rate_missing_objectful",
    "val_relation_useful_mass_exact",
    "val_relation_useful_mass_objectless",
    "val_relation_useful_mass_missing_objectful",
    "val_relation_null_prob_exact",
    "val_relation_null_prob_objectless",
    "val_relation_null_prob_missing_objectful",
    "val_loss_objectless_object_action_suppression",
    "val_loss_object_prompt_grounding",
    "val_object_prompt_grounding_acc",
    "val_object_prompt_grounding_true_prob",
    "val_object_prompt_exact_teacher_valid_rate_1based",
    "val_object_prompt_exact_compatible_rate_1based",
    "val_object_prompt_any_compatible_proposal_rate",
    "val_object_prompt_exact_compatible_count",
    "val_object_prompt_exact_correct_object_rate",
    "val_object_prompt_exact_correct_object_prob",
    "val_object_prompt_attention_exact_teacher_mean",
    "val_object_prompt_attention_objectless_visible_mean",
    "val_object_prompt_attention_objectless_visible_max",
    "val_object_prompt_attention_objectless_visible_entropy",
    "val_object_prompt_drop_objectless_pred_match",
    "val_object_prompt_drop_objectless_true_prob_delta",
    "val_object_prompt_drop_objectless_kl",
    "val_object_prompt_drop_objectless_acc",
    "val_object_prompt_drop_objectless_object_action_pred_rate",
    "val_object_prompt_distractor_objectless_pred_match",
    "val_object_prompt_distractor_objectless_kl",
    "val_object_prompt_distractor_objectless_acc",
    "val_object_prompt_distractor_objectless_object_action_pred_rate",
    "val_object_prompt_drop_exact_true_logit_drop",
    "val_object_prompt_drop_exact_true_prob_drop",
    "val_object_prompt_drop_exact_pred_match",
    "val_object_prompt_drop_exact_acc",
    "val_object_prompt_drop_Uselaptop_true_logit_drop",
    "val_object_prompt_drop_Uselaptop_true_prob_drop",
    "val_object_prompt_drop_Uselaptop_pred_match",
    "val_object_prompt_drop_Readbook_true_logit_drop",
    "val_object_prompt_drop_WatchTV_true_logit_drop",
    "val_object_prompt_drop_Usetelephone_true_logit_drop",
    "val_actor_object_prompt_token_count",
    "val_token_selection_visual_keep_rate",
    "val_token_selection_visual_keep_count",
    "val_token_selection_actor_box_keep_rate",
    "val_token_selection_visible_object_box_keep_rate",
    "val_token_selection_exact_teacher_object_keep_rate",
    "val_token_selection_interaction_heatmap_keep_rate",
    "val_token_selection_laptop_box_keep_rate",
    "val_token_selection_book_box_keep_rate",
    "val_token_selection_phone_box_keep_rate",
    "val_token_selection_tv_monitor_box_keep_rate",
    "val_token_selection_Uselaptop_teacher_object_keep_rate",
    "val_token_selection_Readbook_teacher_object_keep_rate",
    "val_token_selection_WatchTV_teacher_object_keep_rate",
    "val_token_selection_Usetelephone_teacher_object_keep_rate",
    "val_loss_motion_aux",
    "val_motion_aux_acc",
    "train_nash_weight_action",
    "train_nash_weight_heatmap",
    "train_nash_weight_main_deploy",
    "train_nash_weight_grounding_aux",
    "train_actor_object_missing_view_count",
    "train_actor_object_missing_view_Uselaptop_count",
    "train_actor_object_missing_view_Readbook_count",
    "train_actor_object_missing_view_WatchTV_count",
    "train_actor_object_missing_view_Drink_count",
    "train_loss_actor_object_missing_view_action",
    "train_actor_object_missing_view_action_acc",
    "train_loss_actor_object_missing_view_engagement",
    "train_actor_object_missing_view_engagement_acc",
    "train_loss_actor_object_missing_view_relation_null",
    "train_actor_object_missing_view_relation_null_prob",
    "train_actor_object_missing_view_relation_null_acc",
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
    "val_object_counterfactual_teacher_logit_drop",
    "val_object_counterfactual_teacher_prob_drop",
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
]

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
        "val_group_drink_cup_bottle_glass_acc",
        "val_group_drink_acc",
        "val_loss_actor_object_relation",
        "val_loss_actor_object_engagement",
        "val_actor_object_engagement_acc",
        "val_actor_object_engagement_laptop_acc",
        "val_actor_object_engagement_book_acc",
        "val_actor_object_engagement_phone_tablet_acc",
        "val_actor_object_engagement_tv_monitor_acc",
        "val_actor_object_engagement_laptop_book_tv_acc",
        "val_loss_actor_object_binding_state",
        "val_loss_actor_object_binding_action",
        "val_actor_object_binding_state_margin",
        "val_actor_object_binding_action_margin",
        "val_actor_object_binding_state_pass_rate",
        "val_actor_object_binding_action_pass_rate",
        "val_actor_object_binding_count",
        "train_actor_object_missing_view_count",
        "train_loss_actor_object_missing_view_action",
        "train_actor_object_missing_view_action_acc",
        "train_loss_actor_object_missing_view_engagement",
        "train_actor_object_missing_view_engagement_acc",
        "train_loss_actor_object_missing_view_relation_null",
        "train_actor_object_missing_view_relation_null_prob",
        "train_actor_object_missing_view_relation_null_acc",
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
        "val_object_counterfactual_teacher_logit_drop",
        "val_object_prompt_grounding_acc",
        "val_object_prompt_grounding_true_prob",
        "val_object_prompt_exact_correct_object_rate",
        "val_object_prompt_exact_correct_object_prob",
        "val_object_prompt_attention_exact_teacher_mean",
        "val_object_prompt_attention_objectless_visible_max",
        "val_object_prompt_drop_objectless_pred_match",
        "val_object_prompt_drop_objectless_kl",
        "val_object_prompt_distractor_objectless_object_action_pred_rate",
        "val_object_prompt_drop_exact_true_logit_drop",
        "val_object_prompt_drop_Uselaptop_true_logit_drop",
        "val_object_prompt_drop_Uselaptop_true_prob_drop",
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
    rows = action_progress_rows(epoch_df)
    if not rows:
        return
    rows = [row for row in rows if row["action"] in KEY_ACTIONS]
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


def print_compact_best(epoch_df):
    metrics = [
        "val_deploy_score",
        "val_f1",
        "val_acc_macro",
        "val_group_object_mapped_acc",
        "val_group_objectless_acc",
        "val_group_laptop_book_tv_acc",
        "val_group_phone_tv_acc",
        "val_loss_actor_object_relation",
        "val_loss_actor_object_engagement",
        "val_actor_object_engagement_acc",
        "val_actor_object_engagement_laptop_acc",
        "val_actor_object_engagement_book_acc",
        "val_actor_object_engagement_phone_tablet_acc",
        "val_actor_object_engagement_tv_monitor_acc",
        "val_actor_object_engagement_laptop_book_tv_acc",
        "val_loss_actor_object_binding_state",
        "val_loss_actor_object_binding_action",
        "val_actor_object_binding_state_margin",
        "val_actor_object_binding_action_margin",
        "val_actor_object_binding_state_pass_rate",
        "val_actor_object_binding_action_pass_rate",
        "val_actor_object_binding_count",
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
        "val_object_counterfactual_teacher_logit_drop",
        "val_object_prompt_grounding_acc",
        "val_object_prompt_grounding_true_prob",
        "val_object_prompt_exact_correct_object_rate",
        "val_object_prompt_exact_correct_object_prob",
        "val_object_prompt_attention_exact_teacher_mean",
        "val_object_prompt_attention_objectless_visible_max",
        "val_object_prompt_drop_objectless_pred_match",
        "val_object_prompt_drop_objectless_kl",
        "val_object_prompt_distractor_objectless_object_action_pred_rate",
        "val_object_prompt_drop_exact_true_logit_drop",
        "val_object_prompt_drop_exact_true_prob_drop",
        "val_object_prompt_drop_Uselaptop_true_logit_drop",
        "val_object_prompt_drop_Uselaptop_true_prob_drop",
        "val_object_prompt_drop_Uselaptop_pred_match",
        "val_token_selection_actor_box_keep_rate",
        "val_token_selection_visible_object_box_keep_rate",
        "val_token_selection_exact_teacher_object_keep_rate",
        "val_token_selection_interaction_heatmap_keep_rate",
        "val_token_selection_laptop_box_keep_rate",
        "val_token_selection_Uselaptop_teacher_object_keep_rate",
        "val_actor_object_prompt_token_count",
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
        "val_interaction_heatmap_missing_object_masked_rate",
        "val_object_counterfactual_teacher_logit_drop",
        "val_object_counterfactual_teacher_prob_drop",
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
        "val_interaction_heatmap_missing_object_masked_rate",
        "val_interaction_heatmap_exact_compatible_valid_rate",
        "val_interaction_heatmap_mismatch_valid_rate",
        "val_object_counterfactual_teacher_logit_drop",
        "val_object_counterfactual_teacher_prob_drop",
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
        count_col = f"val_action_{action}_count"
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
                "count": metric(latest, count_col),
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
        "ALL ACTION PROGRESS",
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

    hurt = [
        row
        for row in rows
        if pd.notna(row["delta"]) and float(row["delta"]) < -0.005
    ]
    hurt = sorted(hurt, key=lambda row: float(row["delta"]))
    improved = [
        row
        for row in rows
        if pd.notna(row["delta"]) and float(row["delta"]) >= 0.02
    ]
    improved = sorted(improved, key=lambda row: float(row["delta"]), reverse=True)
    print_table(
        "ACTIONS HURT VS EPOCH 0",
        hurt,
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
    print_table(
        "ACTIONS IMPROVED VS EPOCH 0",
        improved,
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
    if epoch_df.empty:
        return
    base = epoch_df.iloc[0]
    latest = epoch_df.iloc[-1]
    rows = []
    for group in GROUPS:
        col = f"val_group_{group}_acc"
        count_col = f"val_group_{group}_count"
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
                "count": metric(latest, count_col),
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


def print_best_epochs(epoch_df):
    metrics = [
        "val_deploy_score",
        "val_f1",
        "val_acc_macro",
        "val_interaction_heatmap_soft_iou",
        "val_interaction_heatmap_positive_mean",
        "val_interaction_heatmap_laptop_positive_mean",
        "val_interaction_heatmap_laptop_iou",
        "val_actor_all_slot_acc",
        "val_actor_pair_acc",
        "val_objectless_with_object_visible_acc",
        "val_objectless_with_laptop_visible_acc",
        "val_object_counterfactual_teacher_logit_drop",
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
    engagement_loss = metric(latest, "val_loss_actor_object_engagement")
    engagement_acc = metric(latest, "val_actor_object_engagement_acc")
    engagement_laptop_acc = metric(
        latest,
        "val_actor_object_engagement_laptop_acc",
    )
    engagement_book_acc = metric(latest, "val_actor_object_engagement_book_acc")
    engagement_phone_acc = metric(
        latest,
        "val_actor_object_engagement_phone_tablet_acc",
    )
    engagement_tv_acc = metric(
        latest,
        "val_actor_object_engagement_tv_monitor_acc",
    )
    engagement_lbt_acc = metric(
        latest,
        "val_actor_object_engagement_laptop_book_tv_acc",
    )
    pos = metric(latest, "val_interaction_heatmap_positive_mean")
    pred_max = metric(latest, "val_interaction_heatmap_pred_max")
    soft_iou = metric(latest, "val_interaction_heatmap_soft_iou")
    center_l2 = metric(latest, "val_interaction_heatmap_center_l2")
    laptop_pos = metric(latest, "val_interaction_heatmap_laptop_positive_mean")
    laptop_iou = metric(latest, "val_interaction_heatmap_laptop_iou")

    prompt_loss = metric(latest, "val_loss_object_prompt_grounding")
    prompt_acc = metric(latest, "val_object_prompt_grounding_acc")
    prompt_true_prob = metric(latest, "val_object_prompt_grounding_true_prob")
    prompt_teacher_valid = metric(
        latest,
        "val_object_prompt_exact_teacher_valid_rate_1based",
    )
    prompt_teacher_compat = metric(
        latest,
        "val_object_prompt_exact_compatible_rate_1based",
    )
    prompt_any_compat = metric(
        latest,
        "val_object_prompt_any_compatible_proposal_rate",
    )
    prompt_exact_correct = metric(
        latest,
        "val_object_prompt_exact_correct_object_rate",
    )
    prompt_exact_prob = metric(
        latest,
        "val_object_prompt_exact_correct_object_prob",
    )
    prompt_exact_teacher_attention = metric(
        latest,
        "val_object_prompt_attention_exact_teacher_mean",
    )
    prompt_objectless_attention_mean = metric(
        latest,
        "val_object_prompt_attention_objectless_visible_mean",
    )
    prompt_objectless_attention_max = metric(
        latest,
        "val_object_prompt_attention_objectless_visible_max",
    )
    prompt_objectless_attention_entropy = metric(
        latest,
        "val_object_prompt_attention_objectless_visible_entropy",
    )
    prompt_drop_objectless_match = metric(
        latest,
        "val_object_prompt_drop_objectless_pred_match",
    )
    prompt_drop_objectless_prob_delta = metric(
        latest,
        "val_object_prompt_drop_objectless_true_prob_delta",
    )
    prompt_drop_objectless_kl = metric(
        latest,
        "val_object_prompt_drop_objectless_kl",
    )
    prompt_drop_objectless_acc = metric(
        latest,
        "val_object_prompt_drop_objectless_acc",
    )
    prompt_drop_objectless_object_rate = metric(
        latest,
        "val_object_prompt_drop_objectless_object_action_pred_rate",
    )
    prompt_distractor_objectless_match = metric(
        latest,
        "val_object_prompt_distractor_objectless_pred_match",
    )
    prompt_distractor_objectless_kl = metric(
        latest,
        "val_object_prompt_distractor_objectless_kl",
    )
    prompt_distractor_objectless_acc = metric(
        latest,
        "val_object_prompt_distractor_objectless_acc",
    )
    prompt_distractor_objectless_object_rate = metric(
        latest,
        "val_object_prompt_distractor_objectless_object_action_pred_rate",
    )
    prompt_drop_exact_logit = metric(
        latest,
        "val_object_prompt_drop_exact_true_logit_drop",
    )
    prompt_drop_exact_prob = metric(
        latest,
        "val_object_prompt_drop_exact_true_prob_drop",
    )
    prompt_drop_exact_match = metric(
        latest,
        "val_object_prompt_drop_exact_pred_match",
    )
    prompt_drop_exact_acc = metric(
        latest,
        "val_object_prompt_drop_exact_acc",
    )
    prompt_drop_uselaptop_logit = metric(
        latest,
        "val_object_prompt_drop_Uselaptop_true_logit_drop",
    )
    prompt_drop_uselaptop_prob = metric(
        latest,
        "val_object_prompt_drop_Uselaptop_true_prob_drop",
    )
    prompt_drop_uselaptop_match = metric(
        latest,
        "val_object_prompt_drop_Uselaptop_pred_match",
    )
    prompt_drop_readbook_logit = metric(
        latest,
        "val_object_prompt_drop_Readbook_true_logit_drop",
    )
    prompt_drop_watchtv_logit = metric(
        latest,
        "val_object_prompt_drop_WatchTV_true_logit_drop",
    )
    prompt_drop_phone_logit = metric(
        latest,
        "val_object_prompt_drop_Usetelephone_true_logit_drop",
    )
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
    motion_aux_loss = metric(latest, "val_loss_motion_aux")
    motion_aux_acc = metric(latest, "val_motion_aux_acc")
    actor_all_slot = metric(latest, "val_actor_all_slot_acc")
    actor_pair = metric(latest, "val_actor_pair_acc")
    actor_pair_swap = metric(latest, "val_actor_pair_swap_acc")
    actor_presence = metric(latest, "val_actor_presence_acc")
    cf_logit = metric(latest, "val_object_counterfactual_teacher_logit_drop")
    cf_prob = metric(latest, "val_object_counterfactual_teacher_prob_drop")
    binding_state_loss = metric(
        latest,
        "val_loss_actor_object_binding_state",
    )
    binding_action_loss = metric(latest, "val_loss_actor_object_binding_action")
    binding_state_margin = metric(
        latest,
        "val_actor_object_binding_state_margin",
    )
    binding_action_margin = metric(
        latest,
        "val_actor_object_binding_action_margin",
    )
    binding_state_pass = metric(
        latest,
        "val_actor_object_binding_state_pass_rate",
    )
    binding_action_pass = metric(
        latest,
        "val_actor_object_binding_action_pass_rate",
    )
    binding_count = metric(latest, "val_actor_object_binding_count")
    missing_view_count = metric(latest, "train_actor_object_missing_view_count")
    missing_view_action_loss = metric(
        latest,
        "train_loss_actor_object_missing_view_action",
    )
    missing_view_action_acc = metric(
        latest,
        "train_actor_object_missing_view_action_acc",
    )
    missing_view_engagement_loss = metric(
        latest,
        "train_loss_actor_object_missing_view_engagement",
    )
    missing_view_engagement_acc = metric(
        latest,
        "train_actor_object_missing_view_engagement_acc",
    )
    missing_view_relation_loss = metric(
        latest,
        "train_loss_actor_object_missing_view_relation_null",
    )
    missing_view_relation_null_prob = metric(
        latest,
        "train_actor_object_missing_view_relation_null_prob",
    )
    missing_view_relation_null_acc = metric(
        latest,
        "train_actor_object_missing_view_relation_null_acc",
    )

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
            f"useful_missing {fmt(relation_useful_missing)}"
        )
    if pd.notna(engagement_acc) or pd.notna(engagement_loss):
        print(
            "actor-object engagement: "
            f"loss {fmt(engagement_loss)}, acc {fmt(engagement_acc)}, "
            f"laptop {fmt(engagement_laptop_acc)}, "
            f"book {fmt(engagement_book_acc)}, "
            f"phone_tablet {fmt(engagement_phone_acc)}, "
            f"tv_monitor {fmt(engagement_tv_acc)}, "
            f"laptop_book_tv {fmt(engagement_lbt_acc)}"
        )
    if pd.notna(binding_state_margin) or pd.notna(binding_action_margin):
        print(
            "object-action binding margins: "
            f"count {fmt(binding_count, 0)}, "
            f"state_loss {fmt(binding_state_loss)}, "
            f"state_margin {fmt(binding_state_margin)}, "
            f"state_pass {fmt(binding_state_pass)}, "
            f"action_loss {fmt(binding_action_loss)}, "
            f"action_margin {fmt(binding_action_margin)}, "
            f"action_pass {fmt(binding_action_pass)}"
        )
    if pd.notna(missing_view_count):
        print(
            "missing-object view training: "
            f"count {fmt(missing_view_count, 0)}, "
            f"action_loss {fmt(missing_view_action_loss)}, "
            f"action_acc {fmt(missing_view_action_acc)}, "
            f"engagement_loss {fmt(missing_view_engagement_loss)}, "
            f"engagement_acc {fmt(missing_view_engagement_acc)}, "
            f"relation_null_loss {fmt(missing_view_relation_loss)}, "
            f"null_prob {fmt(missing_view_relation_null_prob)}, "
            f"null_acc {fmt(missing_view_relation_null_acc)}"
        )
    if pd.notna(prompt_acc) or pd.notna(prompt_loss):
        print(
            "object prompt grounding: "
            f"loss {fmt(prompt_loss)}, acc {fmt(prompt_acc)}, "
            f"true_prob {fmt(prompt_true_prob)}, "
            f"teacher_valid {fmt(prompt_teacher_valid)}, "
            f"teacher_compat {fmt(prompt_teacher_compat)}, "
            f"any_compat {fmt(prompt_any_compat)}, "
            f"exact_correct {fmt(prompt_exact_correct)}, "
            f"exact_prob {fmt(prompt_exact_prob)}, "
            f"tokens {fmt(prompt_tokens, 0)}"
        )
    if (
        pd.notna(prompt_exact_teacher_attention)
        or pd.notna(prompt_objectless_attention_max)
    ):
        print(
            "object prompt attention: "
            f"exact_teacher {fmt(prompt_exact_teacher_attention)}, "
            f"objectless_mean {fmt(prompt_objectless_attention_mean)}, "
            f"objectless_max {fmt(prompt_objectless_attention_max)}, "
            f"objectless_entropy {fmt(prompt_objectless_attention_entropy)}"
        )
    if (
        pd.notna(prompt_drop_objectless_match)
        or pd.notna(prompt_drop_exact_logit)
    ):
        print(
            "object-prompt drop: "
            f"objectless_match {fmt(prompt_drop_objectless_match)}, "
            f"objectless_kl {fmt(prompt_drop_objectless_kl)}, "
            f"objectless_prob_delta {fmt(prompt_drop_objectless_prob_delta)}, "
            f"objectless_acc {fmt(prompt_drop_objectless_acc)}, "
            f"objectless_obj_rate {fmt(prompt_drop_objectless_object_rate)}, "
            f"distractor_match {fmt(prompt_distractor_objectless_match)}, "
            f"distractor_kl {fmt(prompt_distractor_objectless_kl)}, "
            f"distractor_acc {fmt(prompt_distractor_objectless_acc)}, "
            f"distractor_obj_rate {fmt(prompt_distractor_objectless_object_rate)}, "
            f"exact_logit_drop {fmt(prompt_drop_exact_logit)}, "
            f"exact_prob_drop {fmt(prompt_drop_exact_prob)}, "
            f"exact_match {fmt(prompt_drop_exact_match)}, "
            f"exact_acc {fmt(prompt_drop_exact_acc)}"
        )
    if (
        pd.notna(prompt_drop_uselaptop_logit)
        or pd.notna(prompt_drop_readbook_logit)
        or pd.notna(prompt_drop_watchtv_logit)
    ):
        print(
            "object-prompt action causality: "
            f"uselaptop_logit_drop {fmt(prompt_drop_uselaptop_logit)}, "
            f"uselaptop_prob_drop {fmt(prompt_drop_uselaptop_prob)}, "
            f"uselaptop_match {fmt(prompt_drop_uselaptop_match)}, "
            f"readbook_logit_drop {fmt(prompt_drop_readbook_logit)}, "
            f"watchtv_logit_drop {fmt(prompt_drop_watchtv_logit)}, "
            f"phone_logit_drop {fmt(prompt_drop_phone_logit)}"
        )
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
    if pd.notna(motion_aux_loss):
        print(
            "aux safeguards: "
            f"motion_loss {fmt(motion_aux_loss)}, motion_acc {fmt(motion_aux_acc)}"
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
            "teacher-object removal: "
            f"logit_drop {fmt(cf_logit)}, prob_drop {fmt(cf_prob)}"
        )

    if pd.notna(f1_delta) and f1_delta < -0.01:
        print("STOP/ROLL BACK: action F1 dropped more than 0.01 from epoch 0.")
        return
    if pd.notna(engagement_lbt_acc) and engagement_lbt_acc >= 0.70:
        print("GOOD ENGAGEMENT SIGN: object-state semantics are being learned.")
        return
    if pd.notna(prompt_exact_correct) and prompt_exact_correct >= 0.70:
        print("GOOD PROMPT SIGN: exact compatible objects are grounding to prompt tokens.")
        return
    if pd.notna(cf_logit) and cf_logit > 0.02:
        print("GOOD SUPPORTING SIGN: removing the teacher object changes the true action logit.")
        return
    if pd.notna(pos) and pd.notna(pred_max):
        if pos > 0.05 and pred_max > 0.10:
            print("GOOD HEATMAP SIGN: actor-object heatmaps are responding.")
        else:
            print("CONTINUE: heatmap/prompt evidence is not strong yet.")
        return
    print("INSUFFICIENT SIGNAL: no object-prompt or heatmap metrics were found in this run.")


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
    if pd.notna(metric(row, "val_object_counterfactual_teacher_logit_drop")):
        print(
            "teacher-object counterfactual: "
            f"logit_drop {metric(row, 'val_object_counterfactual_teacher_logit_drop'):.4f}, "
            f"prob_drop {metric(row, 'val_object_counterfactual_teacher_prob_drop'):.4f}"
        )
    if pd.notna(metric(row, "val_object_prompt_attention_exact_teacher_mean")):
        print(
            "object prompt attention: "
            f"exact_teacher {metric(row, 'val_object_prompt_attention_exact_teacher_mean'):.4f}, "
            f"objectless_mean {metric(row, 'val_object_prompt_attention_objectless_visible_mean'):.4f}, "
            f"objectless_max {metric(row, 'val_object_prompt_attention_objectless_visible_max'):.4f}, "
            f"objectless_entropy {metric(row, 'val_object_prompt_attention_objectless_visible_entropy'):.4f}"
        )
    if pd.notna(metric(row, "val_object_prompt_drop_objectless_pred_match")):
        print(
            "object-prompt drop: "
            f"objectless_match {metric(row, 'val_object_prompt_drop_objectless_pred_match'):.4f}, "
            f"objectless_kl {metric(row, 'val_object_prompt_drop_objectless_kl'):.4f}, "
            "objectless_prob_delta "
            f"{metric(row, 'val_object_prompt_drop_objectless_true_prob_delta'):.4f}, "
            f"objectless_acc {metric(row, 'val_object_prompt_drop_objectless_acc'):.4f}, "
            "objectless_obj_rate "
            f"{metric(row, 'val_object_prompt_drop_objectless_object_action_pred_rate'):.4f}, "
            "distractor_match "
            f"{metric(row, 'val_object_prompt_distractor_objectless_pred_match'):.4f}, "
            "distractor_kl "
            f"{metric(row, 'val_object_prompt_distractor_objectless_kl'):.4f}, "
            "distractor_acc "
            f"{metric(row, 'val_object_prompt_distractor_objectless_acc'):.4f}, "
            "distractor_obj_rate "
            f"{metric(row, 'val_object_prompt_distractor_objectless_object_action_pred_rate'):.4f}, "
            f"exact_logit_drop {metric(row, 'val_object_prompt_drop_exact_true_logit_drop'):.4f}, "
            f"exact_prob_drop {metric(row, 'val_object_prompt_drop_exact_true_prob_drop'):.4f}, "
            f"exact_match {metric(row, 'val_object_prompt_drop_exact_pred_match'):.4f}, "
            f"exact_acc {metric(row, 'val_object_prompt_drop_exact_acc'):.4f}"
        )
    if pd.notna(metric(row, "val_object_prompt_drop_Uselaptop_true_logit_drop")):
        print(
            "object-prompt action causality: "
            "uselaptop_logit_drop "
            f"{metric(row, 'val_object_prompt_drop_Uselaptop_true_logit_drop'):.4f}, "
            "uselaptop_prob_drop "
            f"{metric(row, 'val_object_prompt_drop_Uselaptop_true_prob_drop'):.4f}, "
            "uselaptop_match "
            f"{metric(row, 'val_object_prompt_drop_Uselaptop_pred_match'):.4f}, "
            "readbook_logit_drop "
            f"{metric(row, 'val_object_prompt_drop_Readbook_true_logit_drop'):.4f}, "
            "watchtv_logit_drop "
            f"{metric(row, 'val_object_prompt_drop_WatchTV_true_logit_drop'):.4f}, "
            "phone_logit_drop "
            f"{metric(row, 'val_object_prompt_drop_Usetelephone_true_logit_drop'):.4f}"
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
    if pd.notna(metric(row, "val_loss_motion_aux")):
        print(
            "aux safeguards: "
            f"motion_loss {metric(row, 'val_loss_motion_aux'):.4f}, "
            f"motion_acc {metric(row, 'val_motion_aux_acc'):.4f}"
        )
    print(
        "poguise+ heatmap loss: "
        f"log {metric(row, 'val_loss_heatmap_log'):.4f}, "
        f"fro {metric(row, 'val_loss_heatmap_frobenius'):.4f}, "
        f"pose_fro {metric(row, 'val_loss_pose_heatmap_frobenius'):.4f}, "
        f"interaction_fro {metric(row, 'val_loss_interaction_heatmap_frobenius'):.4f}"
    )
    nash_main = metric(row, "train_nash_weight_main_deploy")
    if pd.isna(nash_main):
        nash_main = metric(row, "train_nash_weight_action")
    nash_aux = metric(row, "train_nash_weight_grounding_aux")
    if pd.isna(nash_aux):
        nash_aux = metric(row, "train_nash_weight_heatmap")
    print(
        "nash weights: "
        f"main_deploy {nash_main:.4f}, "
        f"grounding_aux {nash_aux:.4f}"
    )

    print("\nACTION GROUPS:")
    for group in GROUPS:
        col = f"val_group_{group}_acc"
        if col in row.index and pd.notna(row[col]):
            count = metric(row, f"val_group_{group}_count")
            if pd.notna(count):
                print(f"{group}: acc {float(row[col]):.4f}, count {count:.0f}")
            else:
                print(f"{group}: acc {float(row[col]):.4f}")

    action_lines = []
    for action in ACTIONS:
        col = f"val_action_{action}_acc"
        if col in row.index and pd.notna(row[col]):
            teacher_col = f"val_action_{action}_interaction_teacher_rate"
            teacher = metric(row, teacher_col)
            count = metric(row, f"val_action_{action}_count")
            count_text = "" if pd.isna(count) else f", count {count:.0f}"
            if pd.notna(teacher):
                action_lines.append(
                    f"{action}: acc {float(row[col]):.4f}, "
                    f"teacher {teacher:.4f}{count_text}"
                )
            else:
                action_lines.append(f"{action}: acc {float(row[col]):.4f}{count_text}")
    if action_lines:
        print("\nALL ACTIONS:")
        for line in action_lines:
            print(line)

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
        print_compact_epoch_summary(epoch_df)
        print_key_action_progress(epoch_df)
        print_group_progress(epoch_df)
        print_compact_best(epoch_df)

    print_decision(epoch_df)

    print("\nREAD THIS:")
    print("- Main proof: val_f1/per-action accuracy stay healthy while engagement, relation NULL/useful-mass, prompt grounding, and interaction heatmaps improve.")
    print("- PO-GUISE+ actor/video tokens make the decision; runtime objects update actor tokens inside the transformer before actor_head.")
    print("- Exact compatible detections should attend to the teacher object and raise relation useful mass.")
    print("- Engagement should separate laptop/book/phone/TV state; this is the key low-motion object-use signal.")
    print("- Missing compatible detections and objectless actions should push relation attention to NULL.")
    print("- Objectless classes should route relation attention to NULL; hard-negative object-action pred rate remains the protection check.")
    print("- Heatmap/object-channel metrics are secondary; use --verbose when debugging teacher quality.")
    print("- Runtime objects should affect token selection and actor tokens inside the transformer, not add late action logits.")


if __name__ == "__main__":
    main()
