# Actor Semantic Object Runbook

This is the hybrid actor-slot PO-GUISE+ path for object-confusable Toyota
actions and live laptop/book/phone use.

Runtime model input:

```text
video
actor boxes
actor valid mask
RF-DETR scene object boxes/classes/confidences
```

The model also predicts class-specific actor-object heatmaps. RF-DETR is used
both as the training teacher for those heatmaps and as an optional runtime object
token source. The video path still exists, so detector misses should degrade
gracefully instead of becoming a hard failure.

## Architecture

The transformer token order is:

```text
class token
actor tokens
scene object tokens
register tokens
heatmap tokens
video tokens
```

Actor tokens carry actor box and slot identity. Scene object tokens carry object
class, box, confidence, pooled visual features from the object box, and a valid
mask. Invalid/padded object slots are zeroed and attention-masked.

The losses are:

```text
action CE
+ pose heatmap log-Frobenius loss
+ semantic actor-object heatmap log-Frobenius loss
+ actor-to-object selection CE
+ optional selected-object counterfactual margin loss
```

Validation always logs the selected-object counterfactual effect when object
tokens are enabled: removing the selected object token should lower the true
action logit.

## Supervised Object Pairs

Reliable pairs:

```text
Uselaptop        -> laptop
Readbook         -> book
Drink.Fromcup    -> cup
```

Quality-gated sparse pairs:

```text
Usetelephone     -> phone
Drink.Frombottle -> bottle
Pour.Frombottle  -> bottle
Cutbread         -> utensil
Cook.Cut         -> utensil
Cook.Stir        -> bowl
Cook.Cleandishes -> sink
Cook.Usestove    -> cooking_appliance
```

WatchTV and Drink.Fromglass are intentionally not forced as object-supervised
targets. Missing RF-DETR detections are treated as unknown labels, not blank
negative labels.

## Metrics To Trust

For heatmap learning:

```text
val_interaction_heatmap_soft_iou
val_interaction_heatmap_positive_mean
val_interaction_heatmap_center_l2
val_interaction_heatmap_laptop_positive_mean
val_interaction_heatmap_laptop_iou
```

For runtime object-token learning:

```text
val_object_selection_acc
val_object_selection_true_prob
val_object_selection_teacher_count
val_object_counterfactual_selected_logit_drop
val_object_counterfactual_selected_prob_drop
```

For action health:

```text
val_f1
val_acc_macro
val_action_Uselaptop_acc
val_action_Readbook_acc
val_action_Usetelephone_acc
val_action_Drink_Fromcup_acc
val_action_Drink_Frombottle_acc
```

The strongest proof is not global F1. It is:

```text
selected object is predicted correctly
true action logit drops when that object token is removed
target action accuracy stays stable
```

## Preflight

Run after Colab setup:

```python
import os, shlex, subprocess, sys

%cd {REPO_DIR}

def run(cmd):
    print("$", " ".join(shlex.quote(str(x)) for x in cmd), flush=True)
    subprocess.run(cmd, check=True)

run(["git", "pull", "--ff-only", "origin", "main"])
run(["git", "log", "-1", "--oneline"])
run([sys.executable, "-m", "pip", "install", "--quiet", "cvxpy", "ecos"])
run([
    sys.executable, "-m", "py_compile",
    "blocks/poguise.py",
    "models/poguise.py",
    "modules/heatmap_module.py",
    "datasets/toyotasm.py",
    "losses/interaction_heatmap_losses.py",
    "losses/poguiseplus_losses.py",
    "grad_weights/nash_mtl.py",
    "train.py",
    "smoke_toyota_object_dataset.py",
    "summarize_interaction_metrics.py",
])
```

Expected: no traceback.

## Warmup Cell

Start from the actor-slot checkpoint from Cell 1. This trains semantic heatmaps,
scene object tokens, object selection, and the last few transformer blocks while
keeping the base actor classifier path frozen.

```python
from pathlib import Path
from datetime import datetime
import os, sys

assert "run_training_with_epoch_summaries" in globals(), "Run the helper cell first."
assert Path(RESUME_CKPT_PATH).is_file(), RESUME_CKPT_PATH

%cd {REPO_DIR}

TS = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_NAME = f"actor_scene_object_warmup_{TS}"
EPOCH_DIR = str(Path(DATA_DIR) / "checkpoints" / RUN_NAME / "epoch_checkpoints")

warmup_cmd = [
    sys.executable, "-u", "train.py",
    "--model_file", RESUME_CKPT_PATH, "--strict_load", "0",
    "--dataset", "toyotasm", "--dataset_artifact", "toyotasm", "--data_dir", DATA_DIR,
    "--toyota_frame_source", "frames",
    "--toyota_skeleton_zip", SKELETON_ZIP,
    "--toyota_frame_count_cache", FRAME_COUNT_CACHE,
    "--toyota_object_cache_dir", f"{DATA_DIR}/toyota_preprocessed_cache/objects",
    "--toyota_landmark_cache_dir", f"{DATA_DIR}/toyota_preprocessed_cache/landmarks",
    "--toyota_split_source", "auto", "--toyota_val_fraction", "0.15", "--toyota_test_fraction", "0.0",
    "--pretrained", "none", "--net_size", "b",
    "--num_classes", "31", "--n_landmarks", "13", "--hw_out_conv", "8",
    "--actor_prompt", "1", "--num_actor_tokens", "8",
    "--actor_presence_head", "1", "--presence_loss_weight", "0.05",
    "--actor_interaction_heatmaps", "1", "--interaction_object_classes", "19",
    "--scene_object_tokens", "1", "--num_scene_object_tokens", "32", "--num_object_classes", "19",
    "--object_detector_cache", OBJECT_DETECTOR_CACHE,
    "--object_camera_allowlist", "tv_monitor=c05,c06",
    "--object_ignore_regions", "c03=0,0,0.26,0.42",
    "--object_conf_threshold", "0.25",
    "--object_token_dropout_prob", "0.15",
    "--object_token_class_dropout_prob", "0.10",
    "--object_token_box_jitter", "0.08",
    "--object_token_confidence_noise", "0.10",
    "--interaction_heatmap_size", "56",
    "--interaction_heatmap_sigma", "1.5",
    "--interaction_guided_sampling", "1",
    "--interaction_min_sampled_object_frames", "1",
    "--interaction_repair_radius_frames", "8",
    "--interaction_quality_min_actor_score", "1.0",
    "--interaction_quality_min_track_frames", "1",
    "--interaction_quality_min_track_coverage", "0.0",
    "--freeze_backbone", "1",
    "--interaction_warmup_freeze_actor_path", "1",
    "--interaction_unfreeze_last_blocks", "4",
    "--class_balanced_sampler", "1",
    "--hard_negative_sampler", "1",
    "--hard_negative_manifest", HARD_NEGATIVE_MANIFEST,
    "--hard_negative_prob", "0.15",
    "--keep_rate", "0.6", "--keep_rate_merge", "0.3",
    "--merge_type", "tome", "--merge_mode", "0",
    "--sim_metric", "0", "--topk_type", "1",
    "--mixup", "0",
    "--grad_weights", "1",
    "--nash_update_weights_every", "10",
    "--nash_max_norm", "2.0",
    "--poguiseplus_heatmap_loss_weight", "1.0",
    "--poguiseplus_pose_heatmap_weight", "0.5",
    "--poguiseplus_interaction_heatmap_weight", "8.0",
    "--poguiseplus_heatmap_log_eps", "1e-6",
    "--object_selection_loss_weight", "0.5",
    "--object_counterfactual_loss_weight", "0.0",
    "--object_counterfactual_eval", "1",
    "--kp_only", "0",
    "--toyota_pose_guided_sampling", "1",
    "--toyota_min_pose_frames", "1",
    "--toyota_synthetic_warmup_epochs", "0",
    "--batch_size", "64",
    "--accum_grad_batches", "1",
    "--max_epochs", "5",
    "--t_max_scheduler", "5",
    "--lr", "2e-6",
    "--lr_head", "2e-4",
    "--lr_head_hm", "7e-4",
    "--weight_decay", "0.04",
    "--weight_decay_head", "0.01",
    "--weight_decay_head_hm", "0.005",
    "--label_smoothing", "0.1",
    "--gradient_clip_val", "1.0",
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
    "--checkpoint_monitor", "val_object_counterfactual_selected_logit_drop",
    "--checkpoint_mode", "max",
    "--checkpoint_filename", "{epoch:03d}-{val_f1:.4f}-{val_object_selection_acc:.4f}-{val_object_counterfactual_selected_logit_drop:.4f}",
    "--save_top_k", "3",
    "--save_every_epoch_checkpoints", "1",
    "--epoch_checkpoint_dir", EPOCH_DIR,
    "--epoch_checkpoint_filename", "{epoch:03d}",
    "--default_root_dir", f"{DATA_DIR}/checkpoints",
    "--model_name", RUN_NAME,
]

warmup_run_dir = run_training_with_epoch_summaries(
    warmup_cmd,
    RUN_NAME,
    poll_secs=20,
)
print("Warmup run:", warmup_run_dir)
```

## Check Running Metrics

```bash
source /content/poguise_colab_env.sh
cd "$REPO_DIR"
python3 summarize_interaction_metrics.py --pattern 'actor_scene_object_warmup_*'
```

Use warmup only if object selection and counterfactual signals appear:

```text
val_object_selection_acc should rise above random
val_object_selection_true_prob should rise
val_object_counterfactual_selected_logit_drop should become positive
laptop positive/iou should keep improving
val_f1 should not collapse
```

## Full Fine Tune Cell

Use the warmup checkpoint with the best combination of object selection,
counterfactual drop, laptop heatmap, and stable F1.

```python
from pathlib import Path
from datetime import datetime
import os, sys

assert "run_training_with_epoch_summaries" in globals(), "Run the helper cell first."

%cd {REPO_DIR}

WARMUP_CKPT = "/mnt/local-scratch/poguise_data/checkpoints/PASTE_RUN/epoch_checkpoints/epoch=XXX.ckpt"
assert Path(WARMUP_CKPT).is_file(), WARMUP_CKPT

TS = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_NAME = f"actor_scene_object_fullft_{TS}"
EPOCH_DIR = str(Path(DATA_DIR) / "checkpoints" / RUN_NAME / "epoch_checkpoints")

fullft_cmd = [
    sys.executable, "-u", "train.py",
    "--model_file", WARMUP_CKPT, "--strict_load", "0",
    "--dataset", "toyotasm", "--dataset_artifact", "toyotasm", "--data_dir", DATA_DIR,
    "--toyota_frame_source", "frames",
    "--toyota_skeleton_zip", SKELETON_ZIP,
    "--toyota_frame_count_cache", FRAME_COUNT_CACHE,
    "--toyota_object_cache_dir", f"{DATA_DIR}/toyota_preprocessed_cache/objects",
    "--toyota_landmark_cache_dir", f"{DATA_DIR}/toyota_preprocessed_cache/landmarks",
    "--toyota_split_source", "auto", "--toyota_val_fraction", "0.15", "--toyota_test_fraction", "0.0",
    "--pretrained", "none", "--net_size", "b",
    "--num_classes", "31", "--n_landmarks", "13", "--hw_out_conv", "8",
    "--actor_prompt", "1", "--num_actor_tokens", "8",
    "--actor_presence_head", "1", "--presence_loss_weight", "0.05",
    "--actor_interaction_heatmaps", "1", "--interaction_object_classes", "19",
    "--scene_object_tokens", "1", "--num_scene_object_tokens", "32", "--num_object_classes", "19",
    "--object_detector_cache", OBJECT_DETECTOR_CACHE,
    "--object_camera_allowlist", "tv_monitor=c05,c06",
    "--object_ignore_regions", "c03=0,0,0.26,0.42",
    "--object_conf_threshold", "0.25",
    "--object_token_dropout_prob", "0.20",
    "--object_token_class_dropout_prob", "0.15",
    "--object_token_box_jitter", "0.10",
    "--object_token_confidence_noise", "0.15",
    "--interaction_heatmap_size", "56",
    "--interaction_heatmap_sigma", "1.5",
    "--interaction_guided_sampling", "1",
    "--interaction_min_sampled_object_frames", "1",
    "--interaction_repair_radius_frames", "8",
    "--interaction_quality_min_actor_score", "1.0",
    "--interaction_quality_min_track_frames", "1",
    "--interaction_quality_min_track_coverage", "0.0",
    "--freeze_backbone", "0",
    "--interaction_warmup_freeze_actor_path", "0",
    "--interaction_unfreeze_last_blocks", "0",
    "--class_balanced_sampler", "1",
    "--hard_negative_sampler", "1",
    "--hard_negative_manifest", HARD_NEGATIVE_MANIFEST,
    "--hard_negative_prob", "0.15",
    "--keep_rate", "0.6", "--keep_rate_merge", "0.3",
    "--merge_type", "tome", "--merge_mode", "0",
    "--sim_metric", "0", "--topk_type", "1",
    "--mixup", "0",
    "--grad_weights", "1",
    "--nash_update_weights_every", "10",
    "--nash_max_norm", "1.5",
    "--poguiseplus_heatmap_loss_weight", "1.0",
    "--poguiseplus_pose_heatmap_weight", "0.5",
    "--poguiseplus_interaction_heatmap_weight", "6.0",
    "--poguiseplus_heatmap_log_eps", "1e-6",
    "--object_selection_loss_weight", "0.5",
    "--object_counterfactual_loss_weight", "0.03",
    "--object_counterfactual_margin", "0.05",
    "--object_counterfactual_eval", "1",
    "--kp_only", "0",
    "--toyota_pose_guided_sampling", "1",
    "--toyota_min_pose_frames", "1",
    "--toyota_synthetic_warmup_epochs", "0",
    "--batch_size", "32",
    "--accum_grad_batches", "2",
    "--max_epochs", "4",
    "--t_max_scheduler", "4",
    "--lr", "1e-6",
    "--lr_head", "5e-5",
    "--lr_head_hm", "2e-4",
    "--weight_decay", "0.04",
    "--weight_decay_head", "0.01",
    "--weight_decay_head_hm", "0.005",
    "--label_smoothing", "0.1",
    "--gradient_clip_val", "1.0",
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
    "--checkpoint_monitor", "val_object_counterfactual_selected_logit_drop",
    "--checkpoint_mode", "max",
    "--checkpoint_filename", "{epoch:03d}-{val_f1:.4f}-{val_object_selection_acc:.4f}-{val_object_counterfactual_selected_logit_drop:.4f}",
    "--save_top_k", "3",
    "--save_every_epoch_checkpoints", "1",
    "--epoch_checkpoint_dir", EPOCH_DIR,
    "--epoch_checkpoint_filename", "{epoch:03d}",
    "--default_root_dir", f"{DATA_DIR}/checkpoints",
    "--model_name", RUN_NAME,
]

fullft_run_dir = run_training_with_epoch_summaries(
    fullft_cmd,
    RUN_NAME,
    poll_secs=20,
)
print("Full fine-tune run:", fullft_run_dir)
```

Check full fine-tune:

```bash
source /content/poguise_colab_env.sh
cd "$REPO_DIR"
python3 summarize_interaction_metrics.py --pattern 'actor_scene_object_fullft_*'
```
