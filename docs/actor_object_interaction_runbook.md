# Actor Object Interaction Runbook

This runbook is for the PO-GUISE+ style actor-object training path, not the old frozen object-specialist reranker diagnostic.

The model is trained and evaluated with the same object interface used at inference:

- RF-DETR object boxes
- object class ids
- object confidences
- object valid mask

Training feeds all detected relevant objects. The action label only creates a weak positive mask for strong interacted-object actions. Distractors stay visible.

## What This Trains

For a `Uselaptop` clip with laptop, book, TV, couch, and cup visible:

- all objects are passed as candidates
- laptop/keyboard tokens are weak positives
- book/TV/cup/couch remain distractors
- the selection loss encourages the actor-object selector to put mass on laptop/keyboard tokens
- the interaction heatmap target covers the laptop/keyboard boxes
- the positive-erased counterfactual removes laptop/keyboard tokens but leaves distractors visible
- the shuffled counterfactual replaces objects with label-mismatched objects from another clip

For objectless actions such as `Walk`, `Enter`, `Leave`, `Laydown`, and `Getup`, the model is trained to keep predictions stable when objects are disabled.

Strong interaction supervision is used only for:

- `Uselaptop -> laptop, keyboard_mouse`
- `Readbook -> book`
- `Usetelephone -> phone`
- `Drink.Fromcup -> cup`
- `Drink.Frombottle -> bottle`
- `Drink.Fromglass -> glass`

Do not force strong positives for `WatchTV`, `Sitdown`, `Eat`, `Takepills`, or `Usetablet`.

## Cell 1: Pull Latest Code

Run this after the normal Colab setup cell.

```python
import shlex
import subprocess

REPO_DIR = "/content/pog"

def run(cmd, cwd=REPO_DIR):
    print("$", " ".join(shlex.quote(str(x)) for x in cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)

run(["git", "pull", "--ff-only", "origin", "main"])
run(["git", "log", "-1", "--oneline"])
```

## Cell 2: Frozen-Backbone PO-GUISE+ Object Warmup

This is the clean Actor-Slot PO-GUISE+ warmup. It does not use `object_relation_only`, `object_specialist_heads`, `specialist_sampler`, or any logit residual mask. The backbone and base actor path stay frozen; only the actor-object feature fusion modules and the heatmap head train.

```python
import os
import sys
import shlex
import subprocess
import time
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

REPO_DIR = "/content/pog"
SCRATCH_ROOT = "/mnt/local-scratch" if os.path.isdir("/mnt/local-scratch") else "/content"
DATA_DIR = f"{SCRATCH_ROOT}/poguise_data"

START_CKPT = f"{DATA_DIR}/object_v2_resume_epoch.ckpt"
SKELETON_ZIP = f"{DATA_DIR}/toyota_smarthome_skeleton_v1.2.zip"
OBJECT_DETECTOR_CACHE = f"{DATA_DIR}/toyota_rfdetr_2xlarge_coco19_full.jsonl"
HARD_NEGATIVE_MANIFEST = f"{DATA_DIR}/hard_negatives.json"
FRAME_COUNT_CACHE = f"{DATA_DIR}/toyota_frame_counts.json"

PREPROC_CACHE_DIR = f"{DATA_DIR}/toyota_preprocessed_cache"
OBJECT_PREPROC_CACHE_DIR = f"{PREPROC_CACHE_DIR}/objects"
LANDMARK_PREPROC_CACHE_DIR = f"{PREPROC_CACHE_DIR}/landmarks"
os.makedirs(OBJECT_PREPROC_CACHE_DIR, exist_ok=True)
os.makedirs(LANDMARK_PREPROC_CACHE_DIR, exist_ok=True)

STAMP = time.strftime("%Y%m%d_%H%M%S")
MODEL_NAME = f"actor_object_poguiseplus_clean_actorfrozen_warmup_from_actor_slot_{STAMP}"
EPOCH_DIR = f"{DATA_DIR}/checkpoints/{MODEL_NAME}/epoch_checkpoints"

def require_file(path, name):
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        raise FileNotFoundError(f"{name} missing/empty: {p}")
    print(f"{name}: {p} ({p.stat().st_size / (1024 ** 2):.1f} MB)", flush=True)

def run_stream(cmd, cwd=None):
    print("\n" + "=" * 100, flush=True)
    print("$ " + " ".join(shlex.quote(str(x)) for x in cmd), flush=True)
    print("=" * 100, flush=True)
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in proc.stdout:
        print(line, end="", flush=True)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}")

require_file(START_CKPT, "Actor-slot checkpoint")
require_file(SKELETON_ZIP, "Skeleton zip")
require_file(OBJECT_DETECTOR_CACHE, "RF-DETR object cache")
require_file(HARD_NEGATIVE_MANIFEST, "Hard-negative manifest")

run_stream(["git", "pull", "--ff-only", "origin", "main"], cwd=REPO_DIR)

cmd = [
    sys.executable, "-u", "train.py",
    "--model_file", START_CKPT,
    "--strict_load", "0",
    "--dataset", "toyotasm",
    "--dataset_artifact", "toyotasm",
    "--data_dir", DATA_DIR,
    "--toyota_frame_source", "frames",
    "--toyota_skeleton_zip", SKELETON_ZIP,
    "--toyota_frame_count_cache", FRAME_COUNT_CACHE,
    "--toyota_object_cache_dir", OBJECT_PREPROC_CACHE_DIR,
    "--toyota_landmark_cache_dir", LANDMARK_PREPROC_CACHE_DIR,
    "--toyota_split_source", "auto",
    "--toyota_val_fraction", "0.15",
    "--toyota_test_fraction", "0.0",
    "--toyota_max_samples", "0",
    "--pretrained", "none",
    "--net_size", "b",
    "--num_classes", "31",
    "--n_landmarks", "13",
    "--hw_out_conv", "8",
    "--use_register_tokens", "0",
    "--actor_prompt", "1",
    "--num_actor_tokens", "8",
    "--actor_presence_head", "1",
    "--presence_loss_weight", "0.05",
    "--actor_bbox_prior_weight", "0.1",
    "--actor_bbox_prior_expand", "1.75",
    "--actor_val_diagnostics", "1",
    "--actor_val_diagnostic_max_pairs", "32",
    "--object_prompt", "1",
    "--num_object_tokens", "24",
    "--num_object_classes", "19",
    "--object_detector_cache", OBJECT_DETECTOR_CACHE,
    "--object_camera_allowlist", "tv_monitor=c05,c06",
    "--object_ignore_regions", "c03=0,0,0.26,0.42",
    "--object_conf_threshold", "0.25",
    "--object_heatmap_size", "56",
    "--object_heatmap_negative_weight", "0.05",
    "--object_none_target_prob", "0.5",
    "--object_track_iou_threshold", "0.2",
    "--object_bbox_prior_weight", "0.05",
    "--object_bbox_prior_expand", "1.25",
    "--object_heatmap_weight", "50",
    "--object_warmup_freeze_actor_path", "1",
    "--object_relation_hidden_dim", "512",
    "--object_relation_dropout", "0.05",
    "--object_fusion_gate_init", "0.04",
    "--object_delta_scale", "0.5",
    "--object_interaction_loss_weight", "0.10",
    "--object_interaction_heatmap_weight", "50",
    "--object_residual_l2_weight", "0.02",
    "--object_counterfactual_margin_weight", "0.10",
    "--object_counterfactual_margin", "0.05",
    "--object_counterfactual_branch_grad", "0",
    "--object_objectless_consistency_weight", "0.03",
    "--object_dropout_prob", "0.05",
    "--object_token_dropout_prob", "0.02",
    "--class_balanced_sampler", "1",
    "--hard_negative_sampler", "1",
    "--hard_negative_manifest", HARD_NEGATIVE_MANIFEST,
    "--hard_negative_prob", "0.15",
    "--keep_rate", "0.6",
    "--keep_rate_merge", "0.3",
    "--merge_type", "tome",
    "--merge_mode", "0",
    "--sim_metric", "0",
    "--topk_type", "1",
    "--freeze_backbone", "1",
    "--freeze_stages", "-1",
    "--linear_probe", "0",
    "--mixup", "0",
    "--grad_weights", "0",
    "--target_kp_loss_weight", "0",
    "--log_kp_loss_weight", "0",
    "--kp_loss_weight", "1000",
    "--kp_only", "0",
    "--deepspeed_optim", "0",
    "--toyota_pose_guided_sampling", "1",
    "--toyota_min_pose_frames", "1",
    "--toyota_synthetic_warmup_epochs", "0",
    "--toyota_synthetic_two_actor_prob", "0",
    "--toyota_synthetic_three_actor_prob", "0",
    "--toyota_synthetic_same_class_prob", "0",
    "--toyota_actor_background_box_prob", "0",
    "--batch_size", "64",
    "--accum_grad_batches", "1",
    "--max_epochs", "4",
    "--t_max_scheduler", "4",
    "--lr", "0",
    "--lr_head", "1e-4",
    "--lr_head_hm", "5e-5",
    "--weight_decay", "0.04",
    "--weight_decay_head", "0.01",
    "--weight_decay_head_hm", "0.01",
    "--label_smoothing", "0.1",
    "--gradient_clip_val", "1.5",
    "--num_workers", "12",
    "--persistent_workers", "1",
    "--prefetch_factor", "2",
    "--precision", "bf16-mixed",
    "--accelerator", "gpu",
    "--gpus", "1",
    "--limit_val_batches", "1.0",
    "--check_val_every_n_epoch", "1",
    "--num_sanity_val_steps", "0",
    "--log_every_n_steps", "50",
    "--checkpoint_monitor", "val_f1_objects_on",
    "--checkpoint_mode", "max",
    "--checkpoint_filename", "{epoch:03d}-{val_f1_objects_on:.4f}-{val_acc_macro_objects_on:.4f}-{val_loss:.4f}",
    "--save_top_k", "3",
    "--save_every_epoch_checkpoints", "1",
    "--epoch_checkpoint_dir", EPOCH_DIR,
    "--epoch_checkpoint_filename", "{epoch:03d}",
    "--default_root_dir", f"{DATA_DIR}/checkpoints",
    "--model_name", MODEL_NAME,
]

run_stream(cmd, cwd=REPO_DIR)

print("\nDONE")
print("Run:", f"{DATA_DIR}/checkpoints/{MODEL_NAME}")
print("Epoch checkpoints:", EPOCH_DIR)
```

## Cell 3: Check Metrics

```python
from pathlib import Path
import pandas as pd
import math

DATA_DIR = "/mnt/local-scratch/poguise_data"
pattern = "actor_object_poguiseplus_clean_actorfrozen_warmup_from_actor_slot_*"

runs = sorted((Path(DATA_DIR) / "checkpoints").glob(pattern), key=lambda p: p.stat().st_mtime)
if not runs:
    raise SystemExit(f"No runs found matching {pattern}")

root = runs[-1]
metrics = sorted(root.glob("version_*/metrics.csv"), key=lambda p: p.stat().st_mtime)[-1]
df = pd.read_csv(metrics)

def last_nonnull(s):
    s = s.dropna()
    return s.iloc[-1] if len(s) else float("nan")

epoch_df = df.groupby("epoch", as_index=False).agg({c: last_nonnull for c in df.columns if c != "epoch"})

cols = [
    "epoch",
    "val_loss",
    "val_obj_heatmap_loss",
    "val_loss_interaction",
    "val_loss_interaction_heatmap",
    "val_acc_macro",
    "val_f1",
    "val_acc_macro_objects_on",
    "val_acc_macro_objects_off",
    "val_acc_macro_objects_shuffled",
    "val_f1_objects_on",
    "val_f1_objects_off",
    "val_f1_objects_shuffled",
    "val_interaction_select_mass_object",
    "val_interaction_select_acc_object",
    "val_object_true_logit_gain_on_vs_positive_erased",
    "val_object_margin_gain_on_vs_positive_erased",
    "val_object_interaction_true_logit_gain_on_vs_positive_erased",
    "val_object_interaction_margin_gain_on_vs_positive_erased",
    "val_object_true_logit_gain_on_vs_shuffled",
    "val_object_margin_gain_on_vs_shuffled",
    "val_object_interaction_true_logit_gain_on_vs_shuffled",
    "val_object_interaction_margin_gain_on_vs_shuffled",
    "val_obj_iou",
    "val_obj_recall_visible",
    "val_laptop_book_tv_objects_on",
    "val_laptop_book_tv_objects_positive_erased",
    "val_laptop_book_tv_objects_shuffled",
    "val_drink_cup_bottle_glass_objects_on",
    "val_drink_cup_bottle_glass_objects_positive_erased",
    "val_drink_cup_bottle_glass_objects_shuffled",
    "val_action_Uselaptop_objects_on",
    "val_action_Uselaptop_objects_positive_erased",
    "val_action_Uselaptop_objects_off",
    "val_action_Uselaptop_objects_shuffled",
    "val_action_Readbook_objects_on",
    "val_action_Readbook_objects_positive_erased",
    "val_action_Readbook_objects_off",
    "val_action_Readbook_objects_shuffled",
    "val_action_WatchTV_objects_on",
    "val_action_WatchTV_objects_positive_erased",
    "val_action_WatchTV_objects_off",
    "val_action_WatchTV_objects_shuffled",
    "val_action_Usetelephone_objects_on",
    "val_action_Usetelephone_objects_positive_erased",
    "val_action_Usetelephone_objects_off",
    "val_action_Usetelephone_objects_shuffled",
    "val_action_Drink_Frombottle_objects_on",
    "val_action_Drink_Frombottle_objects_positive_erased",
    "val_action_Drink_Frombottle_objects_off",
    "val_action_Drink_Frombottle_objects_shuffled",
    "val_action_Drink_Fromcup_objects_on",
    "val_action_Drink_Fromcup_objects_positive_erased",
    "val_action_Drink_Fromcup_objects_off",
    "val_action_Drink_Fromcup_objects_shuffled",
    "val_action_Drink_Fromglass_objects_on",
    "val_action_Drink_Fromglass_objects_positive_erased",
    "val_action_Drink_Fromglass_objects_off",
    "val_action_Drink_Fromglass_objects_shuffled",
]
cols = [c for c in cols if c in epoch_df.columns]
val = epoch_df[df.columns.intersection(["epoch"]).tolist() + [c for c in cols if c != "epoch"]].copy()
val = val[val[[c for c in cols if c != "epoch"]].notna().any(axis=1)]

print("run:", root)
print("metrics:", metrics)
print("\nvalidation rows:")
print(val[cols].tail(10).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

ready = val.dropna(subset=[
    "val_acc_macro_objects_on",
    "val_acc_macro_objects_off",
    "val_acc_macro_objects_shuffled",
    "val_f1_objects_on",
    "val_f1_objects_off",
    "val_f1_objects_shuffled",
], how="any").copy()

if ready.empty:
    raise SystemExit("No complete object ablation validation rows yet.")

ready["global_macro_gain_vs_off"] = ready["val_acc_macro_objects_on"] - ready["val_acc_macro_objects_off"]
ready["global_macro_gain_vs_shuffled"] = ready["val_acc_macro_objects_on"] - ready["val_acc_macro_objects_shuffled"]
ready["global_f1_gain_vs_off"] = ready["val_f1_objects_on"] - ready["val_f1_objects_off"]
ready["global_f1_gain_vs_shuffled"] = ready["val_f1_objects_on"] - ready["val_f1_objects_shuffled"]

score_cols = [
    "global_macro_gain_vs_shuffled",
    "global_f1_gain_vs_shuffled",
    "val_object_margin_gain_on_vs_positive_erased",
    "val_object_interaction_margin_gain_on_vs_positive_erased",
    "val_interaction_select_mass_object",
]
score_cols = [c for c in score_cols if c in ready.columns]
best = ready.sort_values(score_cols, ascending=False).iloc[0] if score_cols else ready.iloc[-1]
latest = ready.iloc[-1]

def show_row(name, row):
    print(f"\n{name}: epoch {int(row['epoch'])}")
    print(f"macro on/off/shuf: {row['val_acc_macro_objects_on']:.4f} / {row['val_acc_macro_objects_off']:.4f} / {row['val_acc_macro_objects_shuffled']:.4f}")
    print(f"f1    on/off/shuf: {row['val_f1_objects_on']:.4f} / {row['val_f1_objects_off']:.4f} / {row['val_f1_objects_shuffled']:.4f}")
    print(f"macro gains off/shuf: {row['global_macro_gain_vs_off']:.4f} / {row['global_macro_gain_vs_shuffled']:.4f}")
    print(f"f1 gains off/shuf:    {row['global_f1_gain_vs_off']:.4f} / {row['global_f1_gain_vs_shuffled']:.4f}")
    for c in [
        "val_obj_heatmap_loss",
        "val_loss_interaction",
        "val_loss_interaction_heatmap",
        "val_interaction_select_mass_object",
        "val_interaction_select_acc_object",
        "val_object_true_logit_gain_on_vs_positive_erased",
        "val_object_margin_gain_on_vs_positive_erased",
        "val_object_interaction_true_logit_gain_on_vs_positive_erased",
        "val_object_interaction_margin_gain_on_vs_positive_erased",
        "val_object_margin_gain_on_vs_shuffled",
        "val_object_interaction_margin_gain_on_vs_shuffled",
    ]:
        if c in row and not math.isnan(row[c]):
            print(f"{c}: {row[c]:.4f}")

show_row("LATEST", latest)
show_row("BEST_CAUSAL_OBJECT_SIGNAL", best)

print("\nDECISION")
print("latest real objects beat shuffled macro:", bool(latest["global_macro_gain_vs_shuffled"] > 0.003))
print("latest F1 not worse than off by >.003:", bool(latest["global_f1_gain_vs_off"] >= -0.003))
if "val_object_margin_gain_on_vs_positive_erased" in latest:
    print("latest positive-erased margin gain > 0:", bool(latest["val_object_margin_gain_on_vs_positive_erased"] > 0))
if "val_object_interaction_margin_gain_on_vs_positive_erased" in latest:
    print("latest strong-object positive-erased margin gain > 0:", bool(latest["val_object_interaction_margin_gain_on_vs_positive_erased"] > 0))
if "val_interaction_select_mass_object" in latest:
    print("latest object selection mass sane:", bool(latest["val_interaction_select_mass_object"] >= 0.50))
```

## Cell 4: Short Full Fine-Tune

Only run this after the warmup shows positive-erased causal gain and real objects beat shuffled without F1 collapse. Start from the best warmup epoch checkpoint and change only:

- `--model_file` to the selected warmup checkpoint
- `--freeze_backbone 0`
- `--max_epochs 6`
- `--t_max_scheduler 6`
- `--lr 1e-6`
- `--lr_head 5e-5`
- `--lr_head_hm 5e-5`

Keep the object losses and RF-DETR object-cache arguments the same.
