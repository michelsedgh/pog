#!/usr/bin/env python3
"""Summarize Actor-Slot PO-GUISE+ object/action association metrics.

This intentionally focuses on object-sensitive diagnostics instead of only
global validation loss or global F1.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


TARGET_CLASSES = [
    "Uselaptop",
    "Readbook",
    "WatchTV",
    "Usetelephone",
    "Drink_Frombottle",
    "Drink_Fromcup",
    "Drink_Fromglass",
]

TARGET_GROUPS = [
    "laptop_book_tv",
    "phone_tv",
    "drink",
    "drink_cup_bottle_glass",
]

SUMMARY_COLUMNS = [
    "epoch",
    "val_loss",
    "val_acc_macro_objects_on",
    "val_acc_macro_objects_off",
    "val_acc_macro_objects_shuffled",
    "macro_gain_vs_off",
    "macro_gain_vs_shuf",
    "val_f1_objects_on",
    "val_f1_objects_off",
    "val_f1_objects_shuffled",
    "f1_gain_vs_off",
    "f1_gain_vs_shuf",
    "val_loss_interaction",
    "val_interaction_select_mass_object",
    "val_interaction_select_acc_object",
    "val_object_interaction_margin_gain_on_vs_positive_erased",
    "val_object_interaction_margin_gain_on_vs_shuffled",
    "val_obj_iou",
    "val_obj_recall_visible",
    "pass_gate",
    "association_score",
]


def _last_nonnull(series: pd.Series):
    series = series.dropna()
    return series.iloc[-1] if len(series) else float("nan")


def _value(row: pd.Series, name: str, default: float = float("nan")) -> float:
    if name not in row or pd.isna(row[name]):
        return default
    return float(row[name])


def _format(value: float) -> str:
    if pd.isna(value):
        return "nan"
    return f"{float(value):.4f}"


def _find_run(root: Path, pattern: str) -> Path:
    runs = sorted(root.glob(pattern), key=lambda path: path.stat().st_mtime)
    if not runs:
        raise SystemExit(f"No runs found matching {root / pattern}")
    return runs[-1]


def _find_metrics(run: Path) -> Path:
    metrics_files = sorted(
        run.glob("version_*/metrics.csv"),
        key=lambda path: path.stat().st_mtime,
    )
    if not metrics_files:
        raise SystemExit(f"No metrics.csv found under {run}")
    return metrics_files[-1]


def _series_or_zero(frame: pd.DataFrame, name: str) -> pd.Series:
    if name in frame.columns:
        return frame[name].fillna(0.0)
    return pd.Series(0.0, index=frame.index)


def _series_or_floor(frame: pd.DataFrame, name: str, floor: float) -> pd.Series:
    if name in frame.columns:
        return frame[name].fillna(floor)
    return pd.Series(floor, index=frame.index)


def _epoch_frame(metrics: Path) -> pd.DataFrame:
    df = pd.read_csv(metrics)
    if "epoch" not in df.columns:
        raise SystemExit(f"{metrics} has no epoch column")
    epoch_df = df.groupby("epoch", as_index=False).agg(
        {col: _last_nonnull for col in df.columns if col != "epoch"}
    )
    for base, off, shuffled, gain_off, gain_shuf in [
        (
            "val_acc_macro_objects_on",
            "val_acc_macro_objects_off",
            "val_acc_macro_objects_shuffled",
            "macro_gain_vs_off",
            "macro_gain_vs_shuf",
        ),
        (
            "val_f1_objects_on",
            "val_f1_objects_off",
            "val_f1_objects_shuffled",
            "f1_gain_vs_off",
            "f1_gain_vs_shuf",
        ),
    ]:
        if all(col in epoch_df.columns for col in (base, off)):
            epoch_df[gain_off] = epoch_df[base] - epoch_df[off]
        if all(col in epoch_df.columns for col in (base, shuffled)):
            epoch_df[gain_shuf] = epoch_df[base] - epoch_df[shuffled]

    pe = epoch_df.get("val_object_interaction_margin_gain_on_vs_positive_erased")
    shuf = epoch_df.get("val_object_interaction_margin_gain_on_vs_shuffled")
    mass = epoch_df.get("val_interaction_select_mass_object")
    f1_off = _series_or_floor(epoch_df, "f1_gain_vs_off", -999.0)
    f1_shuf = _series_or_floor(epoch_df, "f1_gain_vs_shuf", -999.0)
    if pe is not None and shuf is not None and mass is not None:
        epoch_df["pass_gate"] = (
            (pe > 0)
            & (shuf > 0)
            & (mass >= 0.50)
            & (mass <= 0.98)
            & (f1_off >= -0.003)
            & (f1_shuf >= -0.003)
        )
        epoch_df["association_score"] = (
            pe.fillna(0.0).clip(lower=0.0, upper=0.20) * 4.0
            + shuf.fillna(0.0).clip(lower=0.0, upper=0.20) * 4.0
            + mass.fillna(0.0).clip(lower=0.0, upper=1.0) * 0.15
            + _series_or_zero(epoch_df, "val_interaction_select_acc_object") * 0.10
            + _series_or_zero(epoch_df, "val_obj_iou") * 0.10
            + _series_or_floor(epoch_df, "f1_gain_vs_off", -0.05).clip(lower=-0.05, upper=0.05)
            + _series_or_floor(epoch_df, "f1_gain_vs_shuf", -0.05).clip(lower=-0.05, upper=0.05)
        )
    else:
        epoch_df["pass_gate"] = False
        epoch_df["association_score"] = 0.0
    return epoch_df


def _print_epoch_table(epoch_df: pd.DataFrame):
    cols = [col for col in SUMMARY_COLUMNS if col in epoch_df.columns]
    print("\nEPOCH SUMMARY:\n")
    print(epoch_df[cols].to_string(index=False))


def _row_table(row: pd.Series, names: Iterable[str], prefix: str):
    rows = []
    for name in names:
        on = _value(row, f"{prefix}_{name}_objects_on")
        off = _value(row, f"{prefix}_{name}_objects_off")
        erased = _value(row, f"{prefix}_{name}_objects_positive_erased")
        shuffled = _value(row, f"{prefix}_{name}_objects_shuffled")
        sufficient = _value(row, f"{prefix}_{name}_objects_sufficient")
        class_swapped = _value(row, f"{prefix}_{name}_objects_class_swapped")
        if all(pd.isna(v) for v in (on, off, erased, shuffled, sufficient, class_swapped)):
            continue
        rows.append(
            {
                "name": name,
                "on": on,
                "off": off,
                "erased": erased,
                "shuf": shuffled,
                "suff": sufficient,
                "swap": class_swapped,
                "on-off": on - off,
                "on-erased": on - erased,
                "on-shuf": on - shuffled,
                "suff-off": sufficient - off,
                "on-swap": on - class_swapped,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "name",
                "on",
                "off",
                "erased",
                "shuf",
                "suff",
                "swap",
                "on-off",
                "on-erased",
                "on-shuf",
                "suff-off",
                "on-swap",
            ]
        )
    return pd.DataFrame(rows)


def _print_row(label: str, row: pd.Series):
    print(f"\n{label}: epoch {int(row['epoch'])}")
    print(
        "macro on/off/shuf: "
        f"{_format(_value(row, 'val_acc_macro_objects_on'))} / "
        f"{_format(_value(row, 'val_acc_macro_objects_off'))} / "
        f"{_format(_value(row, 'val_acc_macro_objects_shuffled'))}"
    )
    print(
        "f1    on/off/shuf: "
        f"{_format(_value(row, 'val_f1_objects_on'))} / "
        f"{_format(_value(row, 'val_f1_objects_off'))} / "
        f"{_format(_value(row, 'val_f1_objects_shuffled'))}"
    )
    for name in [
        "pass_gate",
        "association_score",
        "val_interaction_select_mass_object",
        "val_interaction_select_acc_object",
        "val_object_interaction_margin_gain_on_vs_positive_erased",
        "val_object_interaction_margin_gain_on_vs_shuffled",
        "val_obj_iou",
        "val_obj_recall_visible",
    ]:
        if name in row:
            print(f"{name}: {row[name] if name == 'pass_gate' else _format(row[name])}")

    class_df = _row_table(row, TARGET_CLASSES, "val_action")
    if len(class_df):
        print("\nPER-CLASS OBJECT ABLATIONS:")
        print(class_df.to_csv(index=False, float_format="%.4f").strip())

    group_df = _row_table(row, TARGET_GROUPS, "val")
    if len(group_df):
        print("\nGROUP OBJECT ABLATIONS:")
        print(group_df.to_csv(index=False, float_format="%.4f").strip())

    _print_warnings(class_df, group_df)


def _print_warnings(class_df: pd.DataFrame, group_df: pd.DataFrame):
    warnings = []
    for _, item in pd.concat([class_df, group_df], ignore_index=True).iterrows():
        name = item["name"]
        if not pd.isna(item["on-shuf"]) and item["on-shuf"] < -0.02:
            warnings.append(f"{name}: shuffled beats objects_on by {-item['on-shuf']:.4f}")
        if not pd.isna(item["on-erased"]) and item["on-erased"] < -0.01:
            warnings.append(f"{name}: positive-erased beats objects_on by {-item['on-erased']:.4f}")
        if "suff-off" in item and not pd.isna(item["suff-off"]) and item["suff-off"] < -0.02:
            warnings.append(
                f"{name}: object-sufficient view is below objects_off by {-item['suff-off']:.4f}"
            )
        if "on-swap" in item and not pd.isna(item["on-swap"]) and item["on-swap"] < -0.01:
            warnings.append(
                f"{name}: class-swapped view beats objects_on by {-item['on-swap']:.4f}"
            )
    if warnings:
        print("\nRED FLAGS:")
        for warning in warnings:
            print(f"- {warning}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=None)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/mnt/local-scratch/poguise_data/checkpoints"),
    )
    parser.add_argument("--pattern", type=str, default="actor_object_poguiseplus_*")
    args = parser.parse_args()

    run = args.run if args.run is not None else _find_run(args.root, args.pattern)
    metrics = _find_metrics(run)
    epoch_df = _epoch_frame(metrics)

    print(f"run: {run}")
    print(f"metrics: {metrics}")
    _print_epoch_table(epoch_df)

    latest = epoch_df.iloc[-1]
    _print_row("LATEST", latest)

    passing = epoch_df[epoch_df["pass_gate"].astype(bool)]
    if len(passing):
        best = passing.sort_values(
            ["association_score", "val_f1_objects_on"],
            ascending=[False, False],
        ).iloc[0]
        _print_row("BEST_PASSING_BY_OBJECT_ASSOCIATION", best)
    else:
        best = epoch_df.sort_values(
            ["association_score", "val_f1_objects_on"],
            ascending=[False, False],
        ).iloc[0]
        _print_row("BEST_NONPASSING_BY_OBJECT_ASSOCIATION", best)

    print("\nREAD THIS:")
    print("- Positive interaction margins > 0 mean objects are affecting true-action support.")
    print("- on-erased > 0 on target classes means the specific interacted object helped.")
    print("- on-shuf > 0 means real object layout helped more than shuffled objects.")
    print("- Uselaptop may stay 1.0 on Toyota; use the live tensor sensitivity test for laptop.")


if __name__ == "__main__":
    main()
