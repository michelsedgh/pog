# Actor Object Interaction Runbook

This runbook is for the PO-GUISE+ style actor-object training path.

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

## Cell 1: Colab Setup

Use your full setup cell here. It must install dependencies, pull `/content/pog`, download or reuse the Drive files, mount the frame tar archives, build `/mnt/local-scratch/poguise_data/frames`, and write `/content/poguise_colab_env.sh`.

The next cells intentionally read `/content/poguise_colab_env.sh` instead of redefining paths. That keeps the notebook consistent with the setup cell and lets later starts reuse:

- downloaded skeleton/object/checkpoint/manifest files
- mounted tar index files
- `/mnt/local-scratch/poguise_data/toyota_preprocessed_cache/objects`
- `/mnt/local-scratch/poguise_data/toyota_preprocessed_cache/landmarks`

```python
# Paste/run your setup cell unchanged.
#
# Required final line from that cell:
#   /content/poguise_colab_env.sh exists and exports REPO_DIR, DATA_DIR,
#   SKELETON_ZIP, RESUME_CKPT_PATH, OBJECT_DETECTOR_CACHE,
#   HARD_NEGATIVE_MANIFEST, FRAME_COUNT_CACHE, and TOYOTA_FRAMES_DIR.
```

## Cell 2: Frozen-Backbone PO-GUISE+ Object Warmup

This is the clean Actor-Object Token PO-GUISE+ warmup. The backbone and base actor path stay frozen; only the object token embeddings, actor-object selection head, and heatmap head train.

```python
import os
import sys
import shlex
import subprocess
import time
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

ENV_FILE = "/content/poguise_colab_env.sh"
REQUIRED_COMMIT = "243a083"

def load_colab_env(path=ENV_FILE):
    env = {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run Cell 1 first; later cells read the cache paths from it."
        )
    for raw_line in p.read_text().splitlines():
        line = raw_line.strip()
        if not line.startswith("export ") or "=" not in line:
            continue
        key, value = line[len("export "):].split("=", 1)
        value = shlex.split(value)[0] if value else ""
        env[key] = value
        os.environ[key] = value
    return env

ENV = load_colab_env()

REPO_DIR = ENV["REPO_DIR"]
DATA_DIR = ENV["DATA_DIR"]
START_CKPT = ENV.get("RESUME_CKPT_PATH") or ENV["CKPT_PATH"]
SKELETON_ZIP = ENV["SKELETON_ZIP"]
OBJECT_DETECTOR_CACHE = ENV["OBJECT_DETECTOR_CACHE"]
HARD_NEGATIVE_MANIFEST = ENV["HARD_NEGATIVE_MANIFEST"]
FRAME_COUNT_CACHE = ENV["FRAME_COUNT_CACHE"]

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
run_stream(["git", "log", "-1", "--oneline"], cwd=REPO_DIR)
run_stream(["git", "merge-base", "--is-ancestor", REQUIRED_COMMIT, "HEAD"], cwd=REPO_DIR)

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
    "--object_interaction_hidden_dim", "512",
    "--object_interaction_dropout", "0.05",
    "--object_interaction_loss_weight", "0.10",
    "--object_interaction_heatmap_weight", "50",
    "--object_counterfactual_margin_weight", "0.10",
    "--object_counterfactual_margin", "0.05",
    "--object_counterfactual_branch_grad", "0",
    "--object_objectless_consistency_weight", "0.02",
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
    "--lr_head", "5e-5",
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

## Cell 3: Check Warmup Metrics

This cell reads the same setup env file as Cell 2 and only searches the clean warmup run pattern. If you want a specific run, set `RUN_DIR_OVERRIDE` to its full checkpoint directory.

```python
import os
import shlex
from pathlib import Path
import pandas as pd
import math

ENV_FILE = "/content/poguise_colab_env.sh"
RUN_PATTERN = "actor_object_poguiseplus_clean_actorfrozen_warmup_from_actor_slot_*"
RUN_DIR_OVERRIDE = ""

def load_colab_env(path=ENV_FILE):
    env = {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing {path}. Run Cell 1 first.")
    for raw_line in p.read_text().splitlines():
        line = raw_line.strip()
        if not line.startswith("export ") or "=" not in line:
            continue
        key, value = line[len("export "):].split("=", 1)
        value = shlex.split(value)[0] if value else ""
        env[key] = value
        os.environ[key] = value
    return env

ENV = load_colab_env()
DATA_DIR = ENV["DATA_DIR"]
ROOT = Path(DATA_DIR) / "checkpoints"

if RUN_DIR_OVERRIDE:
    root = Path(RUN_DIR_OVERRIDE)
    if not root.exists():
        raise SystemExit(f"RUN_DIR_OVERRIDE does not exist: {root}")
else:
    runs = sorted(ROOT.glob(RUN_PATTERN), key=lambda p: p.stat().st_mtime)
    if not runs:
        raise SystemExit(f"No runs found matching {ROOT / RUN_PATTERN}")
    print("matching runs:")
    for run_path in runs[-5:]:
        print(" ", run_path)
    root = runs[-1]

metrics_files = sorted(root.glob("version_*/metrics.csv"), key=lambda p: p.stat().st_mtime)
if not metrics_files:
    raise SystemExit(f"No metrics.csv found under {root}")
metrics = metrics_files[-1]
df = pd.read_csv(metrics)

def last_nonnull(s):
    s = s.dropna()
    return s.iloc[-1] if len(s) else float("nan")

epoch_df = df.groupby("epoch", as_index=False).agg({
    c: last_nonnull for c in df.columns if c != "epoch"
})

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
val = epoch_df[cols].copy()
metric_cols = [c for c in cols if c != "epoch"]
val = val[val[metric_cols].notna().any(axis=1)]

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
print("latest F1 not worse than shuffled by >.003:", bool(latest["global_f1_gain_vs_shuffled"] >= -0.003))
if "val_object_margin_gain_on_vs_positive_erased" in latest:
    print("latest positive-erased margin gain > 0:", bool(latest["val_object_margin_gain_on_vs_positive_erased"] > 0))
if "val_object_interaction_margin_gain_on_vs_positive_erased" in latest:
    print("latest strong-object positive-erased margin gain > 0:", bool(latest["val_object_interaction_margin_gain_on_vs_positive_erased"] > 0))
if "val_interaction_select_mass_object" in latest:
    print("latest object selection mass sane:", bool(latest["val_interaction_select_mass_object"] >= 0.50))
```

## Cell 4: Short Full Fine-Tune

Only run this after the warmup shows positive-erased causal gain and no F1 collapse. The cell refuses to start if no warmup epoch passes the gate.

It starts from the best passing warmup epoch checkpoint, unfreezes the backbone, and uses a lower-memory batch shape (`batch_size=32`, `accum_grad_batches=2`) so the effective batch remains 64 without relying on the warmup memory profile.

```python
import os
import sys
import shlex
import subprocess
import time
from pathlib import Path

import pandas as pd

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

ENV_FILE = "/content/poguise_colab_env.sh"
REQUIRED_COMMIT = "243a083"
WARMUP_PATTERN = "actor_object_poguiseplus_clean_actorfrozen_warmup_from_actor_slot_*"
WARMUP_RUN_DIR_OVERRIDE = ""
PASS_REQUIRED = True

def load_colab_env(path=ENV_FILE):
    env = {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing {path}. Run Cell 1 first.")
    for raw_line in p.read_text().splitlines():
        line = raw_line.strip()
        if not line.startswith("export ") or "=" not in line:
            continue
        key, value = line[len("export "):].split("=", 1)
        value = shlex.split(value)[0] if value else ""
        env[key] = value
        os.environ[key] = value
    return env

ENV = load_colab_env()

REPO_DIR = ENV["REPO_DIR"]
DATA_DIR = ENV["DATA_DIR"]
SKELETON_ZIP = ENV["SKELETON_ZIP"]
OBJECT_DETECTOR_CACHE = ENV["OBJECT_DETECTOR_CACHE"]
HARD_NEGATIVE_MANIFEST = ENV["HARD_NEGATIVE_MANIFEST"]
FRAME_COUNT_CACHE = ENV["FRAME_COUNT_CACHE"]

PREPROC_CACHE_DIR = f"{DATA_DIR}/toyota_preprocessed_cache"
OBJECT_PREPROC_CACHE_DIR = f"{PREPROC_CACHE_DIR}/objects"
LANDMARK_PREPROC_CACHE_DIR = f"{PREPROC_CACHE_DIR}/landmarks"
os.makedirs(OBJECT_PREPROC_CACHE_DIR, exist_ok=True)
os.makedirs(LANDMARK_PREPROC_CACHE_DIR, exist_ok=True)

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

def last_nonnull(s):
    s = s.dropna()
    return s.iloc[-1] if len(s) else float("nan")

def find_warmup_run():
    if WARMUP_RUN_DIR_OVERRIDE:
        run = Path(WARMUP_RUN_DIR_OVERRIDE)
        if not run.exists():
            raise SystemExit(f"WARMUP_RUN_DIR_OVERRIDE does not exist: {run}")
        return run

    root = Path(DATA_DIR) / "checkpoints"
    runs = sorted(root.glob(WARMUP_PATTERN), key=lambda p: p.stat().st_mtime)
    if not runs:
        raise SystemExit(f"No warmup runs found matching {root / WARMUP_PATTERN}")
    print("matching warmup runs:")
    for run_path in runs[-5:]:
        print(" ", run_path)
    return runs[-1]

def load_epoch_metrics(run):
    metrics_files = sorted(run.glob("version_*/metrics.csv"), key=lambda p: p.stat().st_mtime)
    if not metrics_files:
        raise SystemExit(f"No metrics.csv found under {run}")
    metrics = metrics_files[-1]
    df = pd.read_csv(metrics)
    epoch_df = df.groupby("epoch", as_index=False).agg({
        c: last_nonnull for c in df.columns if c != "epoch"
    })
    return metrics, epoch_df

def choose_warmup_checkpoint(run, epoch_df):
    required = [
        "val_f1_objects_on",
        "val_f1_objects_off",
        "val_f1_objects_shuffled",
        "val_acc_macro_objects_on",
    ]
    missing = [c for c in required if c not in epoch_df.columns]
    if missing:
        raise SystemExit(f"Warmup metrics missing required columns: {missing}")

    ready = epoch_df.dropna(subset=required).copy()
    if ready.empty:
        raise SystemExit("No complete warmup validation epochs yet.")

    erased_margin_cols = [
        c for c in [
            "val_object_margin_gain_on_vs_positive_erased",
            "val_object_interaction_margin_gain_on_vs_positive_erased",
        ]
        if c in ready.columns
    ]
    shuffled_margin_cols = [
        c for c in [
            "val_object_margin_gain_on_vs_shuffled",
            "val_object_interaction_margin_gain_on_vs_shuffled",
        ]
        if c in ready.columns
    ]
    if not erased_margin_cols:
        raise SystemExit("Warmup metrics missing positive-erased margin columns.")
    if not shuffled_margin_cols:
        raise SystemExit("Warmup metrics missing shuffled margin columns.")

    ready["f1_safe_vs_off"] = ready["val_f1_objects_on"] >= ready["val_f1_objects_off"] - 0.003
    ready["f1_safe_vs_shuffled"] = ready["val_f1_objects_on"] >= ready["val_f1_objects_shuffled"] - 0.003
    ready["erased_margin_positive"] = ready[erased_margin_cols].fillna(0).max(axis=1) > 0
    ready["shuffled_margin_positive"] = ready[shuffled_margin_cols].fillna(0).max(axis=1) > 0

    passing = ready[
        ready["f1_safe_vs_off"]
        & ready["f1_safe_vs_shuffled"]
        & ready["erased_margin_positive"]
        & ready["shuffled_margin_positive"]
    ].copy()

    print("\nwarmup epoch gate table:")
    table_cols = [
        "epoch",
        "val_f1_objects_on",
        "val_f1_objects_off",
        "val_f1_objects_shuffled",
        "val_acc_macro_objects_on",
        "f1_safe_vs_off",
        "f1_safe_vs_shuffled",
        "erased_margin_positive",
        "shuffled_margin_positive",
    ]
    print(ready[table_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    if passing.empty:
        message = (
            "No warmup epoch passed the full fine-tune gate. "
            "Do not full fine-tune this run unless this is a debugging experiment."
        )
        if PASS_REQUIRED:
            raise SystemExit(message)
        print("WARNING:", message)
        passing = ready

    best = passing.sort_values(
        ["val_f1_objects_on", "val_acc_macro_objects_on"],
        ascending=False,
    ).iloc[0]
    epoch = int(best["epoch"])

    candidates = [
        run / "epoch_checkpoints" / f"{epoch:03d}.ckpt",
        run / "epoch_checkpoints" / f"{epoch:03d}",
    ]
    candidates.extend(sorted(run.glob(f"**/{epoch:03d}-*.ckpt"), key=lambda p: p.stat().st_mtime))
    candidates.extend(sorted(run.glob(f"**/epoch={epoch}*.ckpt"), key=lambda p: p.stat().st_mtime))

    for ckpt in candidates:
        if ckpt.exists() and ckpt.stat().st_size > 0:
            return epoch, ckpt

    raise SystemExit(f"Could not find checkpoint for warmup epoch {epoch} under {run}")

require_file(SKELETON_ZIP, "Skeleton zip")
require_file(OBJECT_DETECTOR_CACHE, "RF-DETR object cache")
require_file(HARD_NEGATIVE_MANIFEST, "Hard-negative manifest")

run_stream(["git", "pull", "--ff-only", "origin", "main"], cwd=REPO_DIR)
run_stream(["git", "log", "-1", "--oneline"], cwd=REPO_DIR)
run_stream(["git", "merge-base", "--is-ancestor", REQUIRED_COMMIT, "HEAD"], cwd=REPO_DIR)

warmup_run = find_warmup_run()
metrics_path, epoch_df = load_epoch_metrics(warmup_run)
best_epoch, START_CKPT = choose_warmup_checkpoint(warmup_run, epoch_df)
require_file(START_CKPT, f"Warmup checkpoint epoch {best_epoch}")

STAMP = time.strftime("%Y%m%d_%H%M%S")
MODEL_NAME = f"actor_object_poguiseplus_clean_fullft_from_warmup_e{best_epoch:03d}_{STAMP}"
EPOCH_DIR = f"{DATA_DIR}/checkpoints/{MODEL_NAME}/epoch_checkpoints"

print("\nwarmup run:", warmup_run)
print("warmup metrics:", metrics_path)
print("selected warmup checkpoint:", START_CKPT)
print("full fine-tune model:", MODEL_NAME)

cmd = [
    sys.executable, "-u", "train.py",
    "--model_file", str(START_CKPT),
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
    "--object_warmup_freeze_actor_path", "0",
    "--object_interaction_hidden_dim", "512",
    "--object_interaction_dropout", "0.05",
    "--object_interaction_loss_weight", "0.10",
    "--object_interaction_heatmap_weight", "50",
    "--object_counterfactual_margin_weight", "0.10",
    "--object_counterfactual_margin", "0.05",
    "--object_counterfactual_branch_grad", "0",
    "--object_objectless_consistency_weight", "0.02",
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
    "--freeze_backbone", "0",
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
    "--batch_size", "32",
    "--accum_grad_batches", "2",
    "--max_epochs", "4",
    "--t_max_scheduler", "4",
    "--lr", "1e-6",
    "--lr_head", "5e-5",
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

## Cell 5: Check Full Fine-Tune Metrics

Run Cell 3 again with this one line changed:

```python
RUN_PATTERN = "actor_object_poguiseplus_clean_fullft_from_warmup_e*"
```

Use `RUN_DIR_OVERRIDE` if you want to inspect one exact full fine-tune directory instead of the newest matching run.
