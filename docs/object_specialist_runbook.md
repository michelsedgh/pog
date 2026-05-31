# Object Specialist Runbook

Deprecated: do not use this runbook for current training. The active object path is the clean Actor-Slot PO-GUISE+ feature-fusion path in `docs/actor_object_interaction_runbook.md`. Specialist rerankers, relation-only training, and specialist sampling now raise errors in the main training code.

This runbook starts after the normal Colab setup cell has finished downloading data, mounting frames, installing dependencies, and writing `/content/poguise_colab_env.sh`.

The required implementation commit is:

```text
1490baa Train object specialists with focused relation objective
```

Your latest `git log -1` may show a newer documentation commit. That is fine. The required check is that `1490baa` is an ancestor of `HEAD`.

The setup cell writes `/content/poguise_colab_env.sh`, but the cells below also define the needed paths directly so they can be pasted into Colab without sourcing that shell file.

The first run is **relation-only**, not full fine-tune. The goal is to prove that scoped object specialists improve the specific object-confusion groups while the base actor model is frozen.

In this diagnostic, relation-only training uses a specialist-only objective. It does not train the object specialist with global 31-class CE. Samples inside a specialist group train group reranking; samples outside those groups train no-boost/preservation losses.

## What This Architecture Trains

The base actor model still predicts all 31 actions.

Object specialists are allowed to modify only these groups:

- `laptop_book_tv`: `Uselaptop`, `Readbook`, `WatchTV`
- `phone_tv`: `Usetelephone`, `WatchTV`
- `drink_cup_bottle_glass`: `Drink.Frombottle`, `Drink.Fromcup`, `Drink.Fromglass`

Skipped by design:

- `Usetablet`: no reliable COCO tablet class.
- `Drink.Fromcan`: no reliable COCO can class.
- `Takepills` and eat/pills: weak COCO evidence, too noisy for this first object-specialist pass.

The specialist branch consumes only the relevant object subsets for each group. It cannot modify unrelated classes like `Walk`, `Getup`, `Sitdown`, `Leave`, etc.

## Cell 1: Pull Latest Code

```python
import subprocess
import shlex

REPO_DIR = "/content/pog"
REQUIRED_COMMIT = "1490baa"

def run(cmd, cwd=REPO_DIR):
    print("$", " ".join(shlex.quote(str(x)) for x in cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)

run(["git", "pull", "--ff-only", "origin", "main"])
run(["git", "log", "-1", "--oneline"])
run(["git", "merge-base", "--is-ancestor", REQUIRED_COMMIT, "HEAD"])
```

## Cell 2: Relation-Only Specialist Training

```python
import os
import sys
import shlex
import subprocess
import time
from pathlib import Path

import pandas as pd

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
MODEL_NAME = f"actor_object_specialist_relonly_from_actor_slot_{STAMP}"
EPOCH_DIR = f"{DATA_DIR}/checkpoints/{MODEL_NAME}/epoch_checkpoints"

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

def require_file(path, name):
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        raise FileNotFoundError(f"{name} missing/empty: {p}")
    print(f"{name}: {p} ({p.stat().st_size / (1024**2):.1f} MB)", flush=True)

require_file(START_CKPT, "Clean actor-slot checkpoint")
require_file(SKELETON_ZIP, "Skeleton zip")
require_file(OBJECT_DETECTOR_CACHE, "Object detector cache")
require_file(HARD_NEGATIVE_MANIFEST, "Hard-negative manifest")

run_stream(["git", "pull", "--ff-only", "origin", "main"], cwd=REPO_DIR)
run_stream(["git", "merge-base", "--is-ancestor", "1490baa", "HEAD"], cwd=REPO_DIR)


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
    "--presence_loss_weight", "0",
    "--actor_bbox_prior_weight", "0.1",
    "--actor_bbox_prior_expand", "1.75",
    "--actor_val_diagnostics", "1",
    "--actor_val_diagnostic_max_pairs", "32",
    "--object_prompt", "1",
    "--object_specialist_heads", "1",
    "--num_object_tokens", "24",
    "--num_object_classes", "19",
    "--object_detector_cache", OBJECT_DETECTOR_CACHE,
    "--object_camera_allowlist", "tv_monitor=c05,c06",
    "--object_ignore_regions", "c03=0,0,0.26,0.42",
    "--object_conf_threshold", "0.25",
    "--object_heatmap_size", "56",
    "--object_heatmap_negative_weight", "0.05",
    "--object_none_target_prob", "0.3",
    "--object_track_iou_threshold", "0.2",
    "--object_bbox_prior_weight", "0",
    "--object_heatmap_weight", "0",
    "--object_relation_only", "1",
    "--object_relation_hidden_dim", "512",
    "--object_relation_dropout", "0.05",
    "--object_action_gate_init", "0.20",
    "--object_delta_scale", "1.0",
    "--object_interaction_loss_weight", "0.10",
    "--object_interaction_heatmap_weight", "10",
    "--object_residual_l2_weight", "0.005",
    "--object_counterfactual_margin_weight", "0.05",
    "--object_counterfactual_margin", "0.05",
    "--object_objectless_consistency_weight", "0.03",
    "--object_specialist_group_loss_weight", "1.0",
    "--object_specialist_no_boost_weight", "0.05",
    "--object_specialist_no_boost_margin", "0.01",
    "--object_dropout_prob", "0.05",
    "--object_token_dropout_prob", "0.02",
    "--class_balanced_sampler", "0",
    "--specialist_sampler", "1",
    "--specialist_positive_prob", "0.55",
    "--hard_negative_sampler", "1",
    "--hard_negative_manifest", HARD_NEGATIVE_MANIFEST,
    "--hard_negative_prob", "0.25",
    "--normal_anchor_prob", "0.20",
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
    "--kp_loss_weight", "0",
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
    "--max_epochs", "5",
    "--t_max_scheduler", "5",
    "--lr", "0",
    "--lr_head", "3e-4",
    "--lr_head_hm", "0",
    "--weight_decay", "0",
    "--weight_decay_head", "0.01",
    "--weight_decay_head_hm", "0",
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
    "--checkpoint_monitor", "val_acc_macro_objects_on",
    "--checkpoint_mode", "max",
    "--checkpoint_filename", "{epoch:03d}-{val_acc_macro_objects_on:.4f}-{val_f1_objects_on:.4f}-{val_loss:.4f}",
    "--save_top_k", "5",
    "--save_every_epoch_checkpoints", "1",
    "--epoch_checkpoint_dir", EPOCH_DIR,
    "--epoch_checkpoint_filename", "{epoch:03d}",
    "--default_root_dir", f"{DATA_DIR}/checkpoints",
    "--model_name", MODEL_NAME,
]

run_stream(cmd, cwd=REPO_DIR)
print("Run:", Path(DATA_DIR) / "checkpoints" / MODEL_NAME)
print("Epoch checkpoints:", EPOCH_DIR)
```

## Cell 3: Check Results From Terminal

Run this in a Colab terminal. It does not interrupt training.

```bash
cd /content/pog
python3 - <<'PY'
from pathlib import Path
import pandas as pd

DATA_DIR = "/mnt/local-scratch/poguise_data"
pattern = "actor_object_specialist_relonly_from_actor_slot_*"
runs = sorted((Path(DATA_DIR) / "checkpoints").glob(pattern), key=lambda p: p.stat().st_mtime)
if not runs:
    raise SystemExit(f"No run found matching {pattern}")
root = runs[-1]
p = sorted(root.glob("version_*/metrics.csv"), key=lambda x: x.stat().st_mtime)[-1]
df = pd.read_csv(p)

cols = [
    "epoch",
    "val_loss",
    "val_acc_macro_objects_on",
    "val_acc_macro_objects_off",
    "val_acc_macro_objects_shuffled",
    "val_f1_objects_on",
    "val_f1_objects_off",
    "val_f1_objects_shuffled",
    "val_laptop_book_tv_objects_on",
    "val_laptop_book_tv_objects_off",
    "val_laptop_book_tv_objects_shuffled",
    "val_phone_tv_objects_on",
    "val_phone_tv_objects_off",
    "val_phone_tv_objects_shuffled",
    "val_drink_cup_bottle_glass_objects_on",
    "val_drink_cup_bottle_glass_objects_off",
    "val_drink_cup_bottle_glass_objects_shuffled",
    "val_object_relation_gate",
    "val_object_relation_gate_max",
    "val_interaction_select_mass_object",
    "val_interaction_select_acc_object",
    "val_loss_object_specialist_group",
    "val_loss_object_specialist_no_boost",
    "val_laptop_book_tv_objects_on_group_margin",
    "val_laptop_book_tv_margin_gain_on_vs_off",
    "val_laptop_book_tv_margin_gain_on_vs_shuffled",
    "val_laptop_book_tv_pred_changed_on_vs_off",
    "val_laptop_book_tv_correct_changes_on_vs_off",
    "val_laptop_book_tv_wrong_changes_on_vs_off",
    "val_phone_tv_objects_on_group_margin",
    "val_phone_tv_margin_gain_on_vs_off",
    "val_phone_tv_margin_gain_on_vs_shuffled",
    "val_phone_tv_pred_changed_on_vs_off",
    "val_phone_tv_correct_changes_on_vs_off",
    "val_phone_tv_wrong_changes_on_vs_off",
    "val_drink_objects_on_group_margin",
    "val_drink_margin_gain_on_vs_off",
    "val_drink_margin_gain_on_vs_shuffled",
    "val_drink_pred_changed_on_vs_off",
    "val_drink_correct_changes_on_vs_off",
    "val_drink_wrong_changes_on_vs_off",
]
cols = [c for c in cols if c in df.columns]
val = df[df[[c for c in cols if c != "epoch"]].notna().any(axis=1)].copy()
ready = val.dropna(subset=[
    "val_acc_macro_objects_on",
    "val_acc_macro_objects_off",
    "val_acc_macro_objects_shuffled",
    "val_f1_objects_on",
    "val_f1_objects_off",
    "val_f1_objects_shuffled",
], how="any").copy()

print("run:", root)
print("metrics:", p)
print("\nvalidation rows:")
print(val[cols].tail(10).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

if ready.empty:
    raise SystemExit("\nNo complete validation row yet.")

ready["macro_gain_vs_off"] = ready["val_acc_macro_objects_on"] - ready["val_acc_macro_objects_off"]
ready["macro_gain_vs_shuffled"] = ready["val_acc_macro_objects_on"] - ready["val_acc_macro_objects_shuffled"]
ready["f1_gain_vs_off"] = ready["val_f1_objects_on"] - ready["val_f1_objects_off"]
ready["f1_gain_vs_shuffled"] = ready["val_f1_objects_on"] - ready["val_f1_objects_shuffled"]
passing = ready[
    (ready["macro_gain_vs_off"] >= 0.003)
    & (ready["macro_gain_vs_shuffled"] >= 0.003)
    & (ready["f1_gain_vs_off"] >= -0.003)
    & (ready["f1_gain_vs_shuffled"] >= -0.003)
]
latest = ready.iloc[-1]
best_pool = passing if not passing.empty else ready
best = best_pool.sort_values(
    ["macro_gain_vs_shuffled", "macro_gain_vs_off", "val_acc_macro_objects_on"],
    ascending=False,
).iloc[0]

def show(name, row):
    print(f"\n{name}: epoch {int(row['epoch'])}")
    print(f"macro on/off/shuf: {row['val_acc_macro_objects_on']:.4f} / {row['val_acc_macro_objects_off']:.4f} / {row['val_acc_macro_objects_shuffled']:.4f}")
    print(f"f1    on/off/shuf: {row['val_f1_objects_on']:.4f} / {row['val_f1_objects_off']:.4f} / {row['val_f1_objects_shuffled']:.4f}")
    print(f"macro gains off/shuf: {row['macro_gain_vs_off']:.4f} / {row['macro_gain_vs_shuffled']:.4f}")
    print(f"f1 gains off/shuf:    {row['f1_gain_vs_off']:.4f} / {row['f1_gain_vs_shuffled']:.4f}")

show("LATEST", latest)
show("BEST_OBJECT_USEFULNESS", best)
print("\npassing epochs:", [int(x) for x in passing["epoch"].tolist()])
PY
```

## Decision Gate Before Full Fine-Tune

Do not full fine-tune unless the relation-only run shows specialist signal in the target groups and global metrics stay close to the actor baseline.

Minimum signal pass:

```text
laptop_book_tv objects_on > max(off, shuffled), with positive margin gain
drink_cup_bottle_glass objects_on > max(off, shuffled), with positive margin gain
phone_tv improves or is disabled before full fine-tune
global F1 is not worse than off/shuffled by more than about 0.003
```

Correct-change counts are useful, but they can stay near zero early if the specialist is improving group margins without flipping many final top-1 predictions yet. Treat positive group accuracy and positive group-margin gains as the first diagnostic signal. Treat correct-change > wrong-change as the stronger follow-up signal.

If target-group accuracy and margins do not improve, do not hide it with full fine-tuning. The specialist objective or data signal still needs work.

## Cell 4: Conditional Full Fine-Tune

Run this only after Cell 3 reports at least one passing relation-only epoch.

This cell selects the best passing relation-only checkpoint and fine-tunes from it. It keeps the specialist heads enabled, keeps the residual small and bounded, and enables the visible object heatmap loss so the model continues to learn PO-GUISE+ style object localization guidance.

```python
import os
import sys
import shlex
import subprocess
import time
from pathlib import Path

import pandas as pd

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

REPO_DIR = "/content/pog"
SCRATCH_ROOT = "/mnt/local-scratch" if os.path.isdir("/mnt/local-scratch") else "/content"
DATA_DIR = f"{SCRATCH_ROOT}/poguise_data"

SKELETON_ZIP = f"{DATA_DIR}/toyota_smarthome_skeleton_v1.2.zip"
OBJECT_DETECTOR_CACHE = f"{DATA_DIR}/toyota_rfdetr_2xlarge_coco19_full.jsonl"
HARD_NEGATIVE_MANIFEST = f"{DATA_DIR}/hard_negatives.json"
FRAME_COUNT_CACHE = f"{DATA_DIR}/toyota_frame_counts.json"

PREPROC_CACHE_DIR = f"{DATA_DIR}/toyota_preprocessed_cache"
OBJECT_PREPROC_CACHE_DIR = f"{PREPROC_CACHE_DIR}/objects"
LANDMARK_PREPROC_CACHE_DIR = f"{PREPROC_CACHE_DIR}/landmarks"
os.makedirs(OBJECT_PREPROC_CACHE_DIR, exist_ok=True)
os.makedirs(LANDMARK_PREPROC_CACHE_DIR, exist_ok=True)

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

def require_file(path, name):
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        raise FileNotFoundError(f"{name} missing/empty: {p}")
    print(f"{name}: {p} ({p.stat().st_size / (1024**2):.1f} MB)", flush=True)

run_stream(["git", "pull", "--ff-only", "origin", "main"], cwd=REPO_DIR)
run_stream(["git", "merge-base", "--is-ancestor", "1490baa", "HEAD"], cwd=REPO_DIR)

relation_runs = sorted(
    (Path(DATA_DIR) / "checkpoints").glob("actor_object_specialist_relonly_from_actor_slot_*"),
    key=lambda p: p.stat().st_mtime,
)
if not relation_runs:
    raise RuntimeError("No specialist relation-only run found.")

relation_root = relation_runs[-1]
metrics_csv = sorted(relation_root.glob("version_*/metrics.csv"), key=lambda p: p.stat().st_mtime)[-1]
df = pd.read_csv(metrics_csv)
ready = df.dropna(subset=[
    "val_acc_macro_objects_on",
    "val_acc_macro_objects_off",
    "val_acc_macro_objects_shuffled",
    "val_f1_objects_on",
    "val_f1_objects_off",
    "val_f1_objects_shuffled",
], how="any").copy()
if ready.empty:
    raise RuntimeError(f"No complete validation row in {metrics_csv}")

ready["macro_gain_vs_off"] = ready["val_acc_macro_objects_on"] - ready["val_acc_macro_objects_off"]
ready["macro_gain_vs_shuffled"] = ready["val_acc_macro_objects_on"] - ready["val_acc_macro_objects_shuffled"]
ready["f1_gain_vs_off"] = ready["val_f1_objects_on"] - ready["val_f1_objects_off"]
ready["f1_gain_vs_shuffled"] = ready["val_f1_objects_on"] - ready["val_f1_objects_shuffled"]
passing = ready[
    (ready["macro_gain_vs_off"] >= 0.003)
    & (ready["macro_gain_vs_shuffled"] >= 0.003)
    & (ready["f1_gain_vs_off"] >= -0.003)
    & (ready["f1_gain_vs_shuffled"] >= -0.003)
].copy()
if passing.empty:
    raise RuntimeError("Relation-only run has no passing epoch. Do not full fine-tune yet.")

best = passing.sort_values(
    ["macro_gain_vs_shuffled", "macro_gain_vs_off", "val_acc_macro_objects_on"],
    ascending=False,
).iloc[0]
best_epoch = int(best["epoch"])
START_CKPT = relation_root / "epoch_checkpoints" / f"epoch={best_epoch:03d}.ckpt"
require_file(START_CKPT, "Best specialist relation-only checkpoint")
require_file(SKELETON_ZIP, "Skeleton zip")
require_file(OBJECT_DETECTOR_CACHE, "Object detector cache")
require_file(HARD_NEGATIVE_MANIFEST, "Hard-negative manifest")

print("Using relation-only run:", relation_root)
print("Using relation-only epoch:", best_epoch)
print(f"macro gains off/shuf: {best['macro_gain_vs_off']:.4f} / {best['macro_gain_vs_shuffled']:.4f}")
print(f"f1 gains off/shuf:    {best['f1_gain_vs_off']:.4f} / {best['f1_gain_vs_shuffled']:.4f}")

STAMP = time.strftime("%Y%m%d_%H%M%S")
MODEL_NAME = f"actor_object_specialist_full_ft_epoch{best_epoch:03d}_{STAMP}"
EPOCH_DIR = f"{DATA_DIR}/checkpoints/{MODEL_NAME}/epoch_checkpoints"

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
    "--object_specialist_heads", "1",
    "--num_object_tokens", "24",
    "--num_object_classes", "19",
    "--object_detector_cache", OBJECT_DETECTOR_CACHE,
    "--object_camera_allowlist", "tv_monitor=c05,c06",
    "--object_ignore_regions", "c03=0,0,0.26,0.42",
    "--object_conf_threshold", "0.25",
    "--object_heatmap_size", "56",
    "--object_heatmap_negative_weight", "0.05",
    "--object_none_target_prob", "0.3",
    "--object_track_iou_threshold", "0.2",
    "--object_bbox_prior_weight", "0.02",
    "--object_heatmap_weight", "25",
    "--object_relation_only", "0",
    "--object_relation_hidden_dim", "512",
    "--object_relation_dropout", "0.05",
    "--object_action_gate_init", "0.03",
    "--object_delta_scale", "0.5",
    "--object_interaction_loss_weight", "0.02",
    "--object_interaction_heatmap_weight", "5",
    "--object_residual_l2_weight", "0.03",
    "--object_counterfactual_margin_weight", "0.10",
    "--object_counterfactual_margin", "0.05",
    "--object_objectless_consistency_weight", "0.03",
    "--object_specialist_group_loss_weight", "0.25",
    "--object_specialist_no_boost_weight", "0.10",
    "--object_specialist_no_boost_margin", "0.01",
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
    "--batch_size", "64",
    "--accum_grad_batches", "1",
    "--max_epochs", "6",
    "--t_max_scheduler", "6",
    "--lr", "1e-6",
    "--lr_head", "7.5e-5",
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
    "--checkpoint_monitor", "val_acc_macro_objects_on",
    "--checkpoint_mode", "max",
    "--checkpoint_filename", "{epoch:03d}-{val_acc_macro_objects_on:.4f}-{val_f1_objects_on:.4f}-{val_loss:.4f}",
    "--save_top_k", "5",
    "--save_every_epoch_checkpoints", "1",
    "--epoch_checkpoint_dir", EPOCH_DIR,
    "--epoch_checkpoint_filename", "{epoch:03d}",
    "--default_root_dir", f"{DATA_DIR}/checkpoints",
    "--model_name", MODEL_NAME,
]

run_stream(cmd, cwd=REPO_DIR)
print("Run:", Path(DATA_DIR) / "checkpoints" / MODEL_NAME)
print("Epoch checkpoints:", EPOCH_DIR)
```

## Cell 5: Check Full Fine-Tune Results From Terminal

```bash
cd /content/pog
python3 - <<'PY'
from pathlib import Path
import pandas as pd

DATA_DIR = "/mnt/local-scratch/poguise_data"
pattern = "actor_object_specialist_full_ft_epoch*"
runs = sorted((Path(DATA_DIR) / "checkpoints").glob(pattern), key=lambda p: p.stat().st_mtime)
if not runs:
    raise SystemExit(f"No run found matching {pattern}")
root = runs[-1]
p = sorted(root.glob("version_*/metrics.csv"), key=lambda x: x.stat().st_mtime)[-1]
df = pd.read_csv(p)

cols = [
    "epoch",
    "val_loss",
    "val_acc_macro_objects_on",
    "val_acc_macro_objects_off",
    "val_acc_macro_objects_shuffled",
    "val_f1_objects_on",
    "val_f1_objects_off",
    "val_f1_objects_shuffled",
    "val_laptop_book_tv_objects_on",
    "val_laptop_book_tv_objects_off",
    "val_laptop_book_tv_objects_shuffled",
    "val_phone_tv_objects_on",
    "val_phone_tv_objects_off",
    "val_phone_tv_objects_shuffled",
    "val_drink_cup_bottle_glass_objects_on",
    "val_drink_cup_bottle_glass_objects_off",
    "val_drink_cup_bottle_glass_objects_shuffled",
    "val_object_relation_gate",
    "val_object_relation_gate_max",
    "val_interaction_select_mass_object",
    "val_interaction_select_acc_object",
]
cols = [c for c in cols if c in df.columns]
val = df[df[[c for c in cols if c != "epoch"]].notna().any(axis=1)].copy()
ready = val.dropna(subset=[
    "val_acc_macro_objects_on",
    "val_acc_macro_objects_off",
    "val_acc_macro_objects_shuffled",
    "val_f1_objects_on",
    "val_f1_objects_off",
    "val_f1_objects_shuffled",
], how="any").copy()

print("run:", root)
print("metrics:", p)
print("\nvalidation rows:")
print(val[cols].tail(10).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

if ready.empty:
    raise SystemExit("\nNo complete validation row yet.")

ready["macro_gain_vs_off"] = ready["val_acc_macro_objects_on"] - ready["val_acc_macro_objects_off"]
ready["macro_gain_vs_shuffled"] = ready["val_acc_macro_objects_on"] - ready["val_acc_macro_objects_shuffled"]
ready["f1_gain_vs_off"] = ready["val_f1_objects_on"] - ready["val_f1_objects_off"]
ready["f1_gain_vs_shuffled"] = ready["val_f1_objects_on"] - ready["val_f1_objects_shuffled"]
latest = ready.iloc[-1]
best = ready.sort_values(
    ["macro_gain_vs_shuffled", "macro_gain_vs_off", "val_acc_macro_objects_on"],
    ascending=False,
).iloc[0]

for name, row in [("LATEST", latest), ("BEST", best)]:
    print(f"\n{name}: epoch {int(row['epoch'])}")
    print(f"macro on/off/shuf: {row['val_acc_macro_objects_on']:.4f} / {row['val_acc_macro_objects_off']:.4f} / {row['val_acc_macro_objects_shuffled']:.4f}")
    print(f"f1    on/off/shuf: {row['val_f1_objects_on']:.4f} / {row['val_f1_objects_off']:.4f} / {row['val_f1_objects_shuffled']:.4f}")
    print(f"macro gains off/shuf: {row['macro_gain_vs_off']:.4f} / {row['macro_gain_vs_shuffled']:.4f}")
    print(f"f1 gains off/shuf:    {row['f1_gain_vs_off']:.4f} / {row['f1_gain_vs_shuffled']:.4f}")
PY
```
