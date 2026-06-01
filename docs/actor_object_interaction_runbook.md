# Actor Object Interaction Runbook

This runbook is for the PO-GUISE+ style actor-object training path.

Current decision: the architecture is ready for a controlled no-new-data test.
Do not add another residual, specialist, class-prior, or relation-only path before
this run. The remaining question is whether Toyota plus RF-DETR pseudo-object
labels contain enough signal for the live laptop failure.

The only approved flow here is:

1. Run the Colab setup cell and reuse its cached paths.
2. Run the architecture preflight smoke.
3. Run a two-epoch frozen object-token warmup from the clean actor-slot checkpoint.
4. Pick the earliest passing checkpoint, not the lowest validation loss.
5. Only if warmup passes, run a two-epoch short unfreeze.
6. Test the resulting checkpoint on the saved live laptop tensor.

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

Use the full setup cell from the notebook. It must install dependencies, pull `/content/pog`, download or reuse the Drive files, mount the frame tar archives, build `/mnt/local-scratch/poguise_data/frames`, and write `/content/poguise_colab_env.sh`.

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

## Cell 2: Architecture Preflight Smoke

Run this once after Cell 1 and after every `git pull` that changes the object path. This cell does not load Toyota frames or start training. It verifies that the checked-out code is the clean object-token path and that invalid object candidates are truly neutral.

```python
import os
import shlex
import subprocess
from pathlib import Path

import torch

ENV_FILE = "/content/poguise_colab_env.sh"
REQUIRED_COMMIT = os.environ.get("POGUISE_REQUIRED_COMMIT", "")

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

def run(cmd, cwd=None, check=True):
    print("$ " + " ".join(shlex.quote(str(x)) for x in cmd), flush=True)
    result = subprocess.run(
        cmd,
        cwd=cwd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.stdout:
        print(result.stdout, end="", flush=True)
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}")
    return result

ENV = load_colab_env()
REPO_DIR = ENV["REPO_DIR"]

run(["git", "pull", "--ff-only", "origin", "main"], cwd=REPO_DIR)
run(["git", "log", "-1", "--oneline"], cwd=REPO_DIR)
if REQUIRED_COMMIT:
    run(["git", "merge-base", "--is-ancestor", REQUIRED_COMMIT, "HEAD"], cwd=REPO_DIR)

forbidden_terms = [
    "none" + "_token",
    "object" + "_specialist",
    "object" + "_relation_only",
    "object" + "_action_gate",
    "object" + "_delta",
    "logit" + " residual",
    "actor_logits" + " \\+",
]
forbidden_pattern = "|".join(forbidden_terms)
result = run(
    [
        "git",
        "grep",
        "-n",
        "-E",
        forbidden_pattern,
        "--",
        "models",
        "modules",
        "train.py",
        "docs",
        "blocks",
    ],
    cwd=REPO_DIR,
    check=False,
)
if result.returncode == 0:
    raise RuntimeError("Stale object path text/code matched the forbidden search above.")
if result.returncode != 1:
    raise RuntimeError(f"git grep failed with exit code {result.returncode}")
print("Stale object path search: clean", flush=True)

import sys
sys.path.insert(0, REPO_DIR)
from models.poguise import POGUISE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16 if device.type == "cuda" else torch.float32
torch.manual_seed(0)

model = POGUISE(
    net_size="b",
    pretrained="none",
    mode="train",
    dataset="toyotasm",
    num_classes=31,
    n_landmarks=13,
    hw_out_conv=8,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    drop_path_rate=0.0,
    head_drop_rate=0.0,
    keep_rate=1.0,
    keep_rate_merge=1.0,
    enhanced_weight_class=1.0,
    enhanced_weight_heatmap=1.0,
    sim_metric=0,
    topk_type=1,
    merge_mode=0,
    merge_type="tome",
    use_register_tokens=0,
    n_registers=0,
    actor_prompt=1,
    num_actor_tokens=8,
    actor_presence_head=1,
    actor_bbox_prior_weight=0.1,
    actor_bbox_prior_expand=1.75,
    object_prompt=1,
    num_object_tokens=24,
    num_object_classes=19,
    object_bbox_prior_weight=0.05,
    object_bbox_prior_expand=1.25,
    object_pool_expand=1.2,
    object_pair_pool_expand=1.1,
    object_interaction_hidden_dim=128,
    object_interaction_dropout=0.0,
    object_heatmap_size=56,
    freeze_backbone=0,
    lr_head_hm=0.0,
    ret_feat=0,
    linear_probe=0,
)
model.eval().to(device=device, dtype=dtype)

B, T, C, H, W = 1, 16, 3, 224, 224
K, M = 8, 24
x = torch.randn(B, T, C, H, W, device=device, dtype=dtype)
boxes = torch.zeros(B, K, 4, device=device, dtype=dtype)
boxes[0, 0] = torch.tensor([0.20, 0.20, 0.80, 0.95], device=device, dtype=dtype)
valid = torch.zeros(B, K, device=device, dtype=torch.bool)
valid[0, 0] = True
object_boxes = torch.zeros(B, M, 4, device=device, dtype=dtype)
object_boxes[0, 0] = torch.tensor([0.35, 0.45, 0.60, 0.62], device=device, dtype=dtype)
object_boxes[0, 1] = torch.tensor([0.05, 0.10, 0.30, 0.60], device=device, dtype=dtype)
object_cls = torch.full((B, M), 19, device=device, dtype=torch.long)
object_cls[0, 0] = 1
object_cls[0, 1] = 13
object_conf = torch.zeros(B, M, device=device, dtype=dtype)
object_conf[0, 0] = 0.9
object_conf[0, 1] = 0.6
object_valid = torch.zeros(B, M, device=device, dtype=torch.bool)
object_valid[0, :2] = True

with torch.inference_mode():
    net_out = model.net(
        x.permute(0, 2, 1, 3, 4),
        boxes=boxes,
        valid=valid,
        object_boxes=object_boxes,
        object_cls=object_cls,
        object_conf=object_conf,
        object_valid=object_valid,
    )
    pair_visual = net_out[-1]
    out = model(
        x,
        boxes=boxes,
        valid=valid,
        object_boxes=object_boxes,
        object_cls=object_cls,
        object_conf=object_conf,
        object_valid=object_valid,
    )
    action_logits, heatmaps, presence, selection_logits, interaction_heatmap = out

    all_invalid = torch.zeros_like(object_valid)
    off = model(
        x,
        boxes=boxes,
        valid=valid,
        object_boxes=object_boxes,
        object_cls=object_cls,
        object_conf=object_conf,
        object_valid=all_invalid,
    )
    off_selection = off[3].float()
    off_probs = torch.softmax(off_selection, dim=-1)
    off_heatmap = off[4]

print("device:", device, "dtype:", dtype)
print("shape action_logits:", tuple(action_logits.shape))
print("shape heatmaps:", tuple(heatmaps.shape))
print("shape presence:", tuple(presence.shape))
print("shape selection:", tuple(selection_logits.shape))
print("shape interaction:", tuple(interaction_heatmap.shape))
print("shape pair_visual:", tuple(pair_visual.shape))
print("all_invalid real_logit_max:", off_selection[..., :-1].max().item())
print("all_invalid none_prob_min:", off_probs[..., -1].min().item())
print("all_invalid real_prob_max:", off_probs[..., :-1].max().item())
print("all_invalid heatmap_abs_max:", off_heatmap.abs().max().item())

assert tuple(action_logits.shape) == (1, 8, 31)
assert tuple(heatmaps.shape) == (1, 32, 56, 56)
assert tuple(presence.shape) == (1, 8)
assert tuple(selection_logits.shape) == (1, 8, 25)
assert tuple(interaction_heatmap.shape) == (1, 8, 56, 56)
assert tuple(pair_visual.shape[:3]) == (1, 8, 24)
assert pair_visual.shape[-1] == model.net.num_features
assert all(torch.isfinite(t).all().item() for t in out)
assert torch.isfinite(off_selection[..., -1]).all()
assert off_probs[..., -1].min().item() > 0.999
assert off_probs[..., :-1].max().item() < 1e-6
assert off_heatmap.abs().max().item() < 1e-6

print("Architecture preflight passed.", flush=True)
```

## Cell 3: Frozen-Backbone PO-GUISE+ Object Warmup

This is the clean Actor-Object Token PO-GUISE+ warmup. The backbone and base actor path stay frozen; only the object token embeddings/projections, visual object grounding, actor-object selection head, and heatmap head train.

The command prints trainable parameter names before dataset setup. During this
warmup, you should see object token/projection parameters, `object_interaction.*`,
and heatmap head parameters. You should not see full transformer blocks or
`model.head`, `actor_head`, or `presence_head` trainable.

Also check this line before the first epoch starts:

```text
Object class embedding init: real_abs_max=... real_std=... padding_abs_max=0.000000e+00
```

`real_abs_max` and `real_std` must be nonzero. If they are zero, laptop/book/phone
classes are indistinguishable at object-token initialization and the run is invalid.

```python
import os
import sys
import shlex
import subprocess
import time
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

ENV_FILE = "/content/poguise_colab_env.sh"
REQUIRED_COMMIT = os.environ.get("POGUISE_REQUIRED_COMMIT", "")

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
if REQUIRED_COMMIT:
    run_stream(["git", "merge-base", "--is-ancestor", REQUIRED_COMMIT, "HEAD"], cwd=REPO_DIR)
else:
    print("POGUISE_REQUIRED_COMMIT not set; using current checked-out code.", flush=True)

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
    "--object_pool_expand", "1.2",
    "--object_pair_pool_expand", "1.1",
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
    "--max_epochs", "2",
    "--t_max_scheduler", "2",
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
    "--print_trainable_params", "1",
]

run_stream(cmd, cwd=REPO_DIR)

print("\nDONE")
print("Run:", f"{DATA_DIR}/checkpoints/{MODEL_NAME}")
print("Epoch checkpoints:", EPOCH_DIR)
```

## Cell 4: Check Warmup Metrics

This cell reads the same setup env file as Cell 3 and only searches the clean warmup run pattern. If you want a specific run, set `RUN_DIR_OVERRIDE` to its full checkpoint directory.

```bash
cd /content/pog

python3 - <<'PY'
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

def last_nonnull(s):
    s = s.dropna()
    return s.iloc[-1] if len(s) else float("nan")

def safe(row, col):
    return row[col] if col in row and not pd.isna(row[col]) else float("nan")

ENV = load_colab_env()
ROOT = Path(ENV["DATA_DIR"]) / "checkpoints"

if RUN_DIR_OVERRIDE:
    run = Path(RUN_DIR_OVERRIDE)
else:
    runs = sorted(ROOT.glob(RUN_PATTERN), key=lambda p: p.stat().st_mtime)
    if not runs:
        raise SystemExit(f"No runs found matching {ROOT / RUN_PATTERN}")
    run = runs[-1]

metrics_files = sorted(run.glob("version_*/metrics.csv"), key=lambda p: p.stat().st_mtime)
if not metrics_files:
    raise SystemExit(f"No metrics.csv found under {run}")

metrics = metrics_files[-1]
df = pd.read_csv(metrics)
epoch_df = df.groupby("epoch", as_index=False).agg({
    c: last_nonnull for c in df.columns if c != "epoch"
})

needed = [
    "val_acc_macro_objects_on",
    "val_acc_macro_objects_off",
    "val_acc_macro_objects_shuffled",
    "val_f1_objects_on",
    "val_f1_objects_off",
    "val_f1_objects_shuffled",
]
ready = epoch_df.dropna(subset=[c for c in needed if c in epoch_df.columns]).copy()
if ready.empty:
    print("run:", run)
    print("metrics:", metrics)
    raise SystemExit("No complete validation rows yet.")

ready["macro_gain_vs_off"] = ready["val_acc_macro_objects_on"] - ready["val_acc_macro_objects_off"]
ready["macro_gain_vs_shuf"] = ready["val_acc_macro_objects_on"] - ready["val_acc_macro_objects_shuffled"]
ready["f1_gain_vs_off"] = ready["val_f1_objects_on"] - ready["val_f1_objects_off"]
ready["f1_gain_vs_shuf"] = ready["val_f1_objects_on"] - ready["val_f1_objects_shuffled"]

erased_col = "val_object_interaction_margin_gain_on_vs_positive_erased"
shuf_col = "val_object_interaction_margin_gain_on_vs_shuffled"
mass_col = "val_interaction_select_mass_object"

ready["pass_gate"] = (
    (ready["f1_gain_vs_off"] >= -0.003)
    & (ready["f1_gain_vs_shuf"] >= -0.003)
    & (ready.get(erased_col, 0) > 0)
    & (ready.get(shuf_col, 0) > 0)
    & (ready.get(mass_col, 0) >= 0.50)
)

summary_cols = [
    "epoch", "val_loss",
    "val_acc_macro_objects_on", "val_acc_macro_objects_off", "val_acc_macro_objects_shuffled",
    "macro_gain_vs_off", "macro_gain_vs_shuf",
    "val_f1_objects_on", "val_f1_objects_off", "val_f1_objects_shuffled",
    "f1_gain_vs_off", "f1_gain_vs_shuf",
    "val_loss_interaction",
    "val_interaction_select_mass_object",
    "val_interaction_select_acc_object",
    erased_col,
    shuf_col,
    "val_obj_iou",
    "val_obj_recall_visible",
    "pass_gate",
]
summary_cols = [c for c in summary_cols if c in ready.columns]

print("run:", run)
print("metrics:", metrics)
print("\nEPOCH SUMMARY:\n")
print(ready[summary_cols].tail(10).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

latest = ready.iloc[-1]
passing = ready[ready["pass_gate"]]
best = passing.sort_values(["val_f1_objects_on", "val_acc_macro_objects_on"], ascending=False).iloc[0] if len(passing) else ready.sort_values(["val_f1_objects_on", "val_acc_macro_objects_on"], ascending=False).iloc[0]

def show(name, row):
    print(f"\n{name}: epoch {int(row['epoch'])}")
    print(f"macro on/off/shuf: {row['val_acc_macro_objects_on']:.4f} / {row['val_acc_macro_objects_off']:.4f} / {row['val_acc_macro_objects_shuffled']:.4f}")
    print(f"f1    on/off/shuf: {row['val_f1_objects_on']:.4f} / {row['val_f1_objects_off']:.4f} / {row['val_f1_objects_shuffled']:.4f}")
    print(f"macro gains off/shuf: {row['macro_gain_vs_off']:.4f} / {row['macro_gain_vs_shuf']:.4f}")
    print(f"f1 gains off/shuf:    {row['f1_gain_vs_off']:.4f} / {row['f1_gain_vs_shuf']:.4f}")
    print("pass_gate:", bool(row["pass_gate"]))
    for c in [mass_col, "val_interaction_select_acc_object", erased_col, shuf_col, "val_obj_iou", "val_obj_recall_visible"]:
        if c in row and not pd.isna(row[c]):
            print(f"{c}: {row[c]:.4f}")

show("LATEST", latest)
show("BEST_PASSING" if len(passing) else "BEST_NONPASSING", best)

print("\nIMPORTANT:")
print("- pass_gate=True means worth considering.")
print("- Do not trust a checkpoint just because val_loss drops.")
print("- Best warmup is usually the earliest passing epoch, not necessarily the last.")
PY
```

## Cell 5: Short Full Fine-Tune

Only run this after the warmup shows positive-erased causal gain and no severe F1 collapse. If you leave `START_CKPT_OVERRIDE` empty, the cell auto-selects the best passing warmup epoch and refuses to start if no epoch passes. If you are making a deliberate borderline run, paste an explicit warmup checkpoint path into `START_CKPT_OVERRIDE`.

It unfreezes the backbone and uses a lower-memory batch shape (`batch_size=32`, `accum_grad_batches=2`) so the effective batch remains 64 without relying on the warmup memory profile. Default is 4 epochs; use 6 only as a deliberate longer run.

```python
import os
import sys
import shlex
import subprocess
import time
import re
from pathlib import Path

import pandas as pd

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

ENV_FILE = "/content/poguise_colab_env.sh"
REQUIRED_COMMIT = os.environ.get("POGUISE_REQUIRED_COMMIT", "")
WARMUP_PATTERN = "actor_object_poguiseplus_clean_actorfrozen_warmup_from_actor_slot_*"
WARMUP_RUN_DIR_OVERRIDE = ""
PASS_REQUIRED = True

# Paste a specific warmup checkpoint here to bypass automatic gate selection.
# Example:
# START_CKPT_OVERRIDE = "/mnt/local-scratch/poguise_data/checkpoints/actor_object_poguiseplus_clean_actorfrozen_warmup_from_actor_slot_20260601_083153/epoch_checkpoints/epoch=002.ckpt"
START_CKPT_OVERRIDE = ""

# 4 is the recommended full fine-tune length. 6 is allowed but more likely to drift.
FULLFT_EPOCHS = 4

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

def infer_epoch_from_checkpoint(path):
    match = re.search(r"epoch[=/-](\d+)", str(path))
    if match:
        return int(match.group(1))
    match = re.search(r"(?:^|/)(\d{3})(?:$|\\.)", str(path))
    if match:
        return int(match.group(1))
    return -1

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
        run / "epoch_checkpoints" / f"epoch={epoch:03d}.ckpt",
        run / "epoch_checkpoints" / f"epoch={epoch}.ckpt",
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
if REQUIRED_COMMIT:
    run_stream(["git", "merge-base", "--is-ancestor", REQUIRED_COMMIT, "HEAD"], cwd=REPO_DIR)
else:
    print("POGUISE_REQUIRED_COMMIT not set; using current checked-out code.", flush=True)

if FULLFT_EPOCHS < 1 or FULLFT_EPOCHS > 6:
    raise SystemExit("FULLFT_EPOCHS must be between 1 and 6. Recommended: 4.")

if START_CKPT_OVERRIDE:
    START_CKPT = Path(START_CKPT_OVERRIDE)
    best_epoch = infer_epoch_from_checkpoint(START_CKPT)
    warmup_run = START_CKPT.parents[1] if START_CKPT.parent.name == "epoch_checkpoints" else START_CKPT.parent
    metrics_path = None
    require_file(START_CKPT, f"Manual warmup checkpoint epoch {best_epoch if best_epoch >= 0 else 'unknown'}")
    print("\nMANUAL CHECKPOINT OVERRIDE ENABLED")
    print("This bypasses the automatic pass gate. Use only for deliberate borderline runs.")
else:
    warmup_run = find_warmup_run()
    metrics_path, epoch_df = load_epoch_metrics(warmup_run)
    best_epoch, START_CKPT = choose_warmup_checkpoint(warmup_run, epoch_df)
    require_file(START_CKPT, f"Warmup checkpoint epoch {best_epoch}")

STAMP = time.strftime("%Y%m%d_%H%M%S")
epoch_tag = f"e{best_epoch:03d}" if best_epoch >= 0 else "manual"
MODEL_NAME = f"actor_object_poguiseplus_clean_fullft_from_warmup_{epoch_tag}_{STAMP}"
EPOCH_DIR = f"{DATA_DIR}/checkpoints/{MODEL_NAME}/epoch_checkpoints"

print("\nwarmup run:", warmup_run)
print("warmup metrics:", metrics_path)
print("selected warmup checkpoint:", START_CKPT)
print("full fine-tune model:", MODEL_NAME)
print("full fine-tune epochs:", FULLFT_EPOCHS)

cmd = [
    sys.executable, "-u", "train.py",
    "--model_file", str(START_CKPT),
    "--strict_load", "1",
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
    "--object_pool_expand", "1.2",
    "--object_pair_pool_expand", "1.1",
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
    "--max_epochs", str(FULLFT_EPOCHS),
    "--t_max_scheduler", str(FULLFT_EPOCHS),
    "--lr", "5e-7",
    "--lr_head", "3e-5",
    "--lr_head_hm", "3e-5",
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
    "--print_trainable_params", "1",
]

run_stream(cmd, cwd=REPO_DIR)

print("\nDONE")
print("Run:", f"{DATA_DIR}/checkpoints/{MODEL_NAME}")
print("Epoch checkpoints:", EPOCH_DIR)
```

## Cell 6: Check Full Fine-Tune Metrics

Run Cell 4 again with this one line changed:

```python
RUN_PATTERN = "actor_object_poguiseplus_clean_fullft_from_warmup_e*"
```

Use `RUN_DIR_OVERRIDE` if you want to inspect one exact full fine-tune directory instead of the newest matching run.

## Cell 7: Live Tensor A/B Test

Toyota validation is not enough for the laptop failure because `Uselaptop` can
be saturated on the Toyota split. After choosing the earliest passing warmup or
short-unfreeze checkpoint, copy that checkpoint to the Orin and run the saved
live tensor diagnostic in PyTorch before exporting TensorRT.

On the Orin:

```bash
cd /home/michel/Documents/poguise
conda activate voice_id

python object_actor_live/analyze_live_object_sensitivity.py \
  --checkpoint /path/to/best_object_token_checkpoint.ckpt \
  --input-pt object_actor_live/latest_epoch001_actor_input.pt \
  --device cuda \
  --dtype auto
```

If the Orin is low on memory, skip the gradient audit:

```bash
python object_actor_live/analyze_live_object_sensitivity.py \
  --checkpoint /path/to/best_object_token_checkpoint.ckpt \
  --input-pt object_actor_live/latest_epoch001_actor_input.pt \
  --device cuda \
  --dtype auto \
  --skip-gradient
```

Use the checkpoint only if the live tensor shows object sensitivity in the right
direction:

- `laptop_only` should raise `Uselaptop` relative to `objects_off`, or at least
  move it meaningfully.
- `positive_erased_laptop` should lower the laptop-supported action evidence.
- `laptop_class_changed_to_book` or `laptop_box_moved_away` should change the
  relevant logits.
- If the script says no laptop/keyboard object exists in the tensor, save a new
  tensor while RF-DETR is actually detecting the laptop.

If the live tensor still shows `laptop selected but Uselaptop barely moves`,
the next blocker is distribution gap, not another object architecture patch.
