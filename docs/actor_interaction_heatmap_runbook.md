# Actor Interaction Heatmap Runbook

This is the final actor-slot PO-GUISE+ path used for Toyota object-confusable
actions.

Runtime model input:

```text
video
actor boxes
actor valid mask
```

RF-DETR detections are training teacher labels only. They create actor-conditioned
interacted-object heatmaps for strong action/object pairs. They are not fed to the
action model at inference.

## Architecture

The model is actor-slot PO-GUISE with extra heatmap channels:

```text
pose heatmaps:                  13 channels
actor interaction heatmaps:      K channels, one per actor slot
```

The transformer sees the same protected semantic tokens as the actor-slot model:

```text
class token
actor tokens
register tokens
heatmap tokens
video tokens
```

There are no runtime object tokens, no object decoder, no object selection loss,
no shuffled-object ablation branch, and no logit residual.

## Strong Targets

The Toyota action label decides whether an interaction heatmap target is valid.
Reliable pairs are supervised whenever the expected object is detected:

```text
Uselaptop        -> laptop
Readbook         -> book
Drink.Fromcup    -> cup
```

Sparse pairs are supervised only when the selected object track passes the
quality gate. The default gate is intentionally light: the selected track must be
actor-associated, and the sampler tries to include at least one expected-object
frame when the cache has one.

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

Drink.Fromglass, WatchTV, Sitdown, Eat, Takepills, Walk, Enter, Leave, and
Laydown do not receive forced interacted-object heatmap targets. This prevents
missing detector labels and context objects from becoming false action evidence.

For each valid actor slot, the dataset selects one actor-associated track from
the matching object class. It does not merge every matching object in the scene.
The selected track is scored by proximity/overlap with the actor box and detector
confidence, then converted into a clip-level center-Gaussian motion heatmap. The
clip target is max-aggregated over sampled frames, so a clean sparse detection
still gives a strong spatial target instead of being diluted by clip length.

Training keeps a strict teacher contract:

```text
every valid actor slot:
  action CE loss

valid actor slot with a trusted object track:
  action CE loss
  + actor-conditioned interaction heatmap loss

valid actor slot with missing/noisy object teacher:
  action CE loss only
```

Missing RF-DETR detections are unknown labels, not negative labels. The dataset
does not train blank interaction heatmaps for missing phones, bottles, utensils,
or other sparse objects.

## Preflight

Run this after Cell 1 setup:

```bash
source /content/poguise_colab_env.sh
cd "$REPO_DIR"
git pull --ff-only origin main
git log -1 --oneline

python3 -m pip install --quiet cvxpy ecos

python3 -m py_compile \
  blocks/poguise.py \
  models/poguise.py \
  modules/heatmap_module.py \
  datasets/toyotasm.py \
  losses/interaction_heatmap_losses.py \
  losses/poguiseplus_losses.py \
  grad_weights/nash_mtl.py \
  train.py \
  smoke_toyota_object_dataset.py \
  visualize_toyota_object_sample.py \
  summarize_interaction_metrics.py

python3 smoke_toyota_object_dataset.py \
  --data_dir "$DATA_DIR" \
  --toyota_frame_source frames \
  --toyota_skeleton_zip "$SKELETON_ZIP" \
  --toyota_frame_count_cache "$FRAME_COUNT_CACHE" \
  --object_detector_cache "$OBJECT_DETECTOR_CACHE" \
  --object_camera_allowlist tv_monitor=c05,c06 \
  --object_ignore_regions c03=0,0,0.26,0.42
```

Expected result:

```text
Summary: ... passed, 0 failed
```

## Warmup

Train interaction heatmap channels and a small late-transformer adaptation from
the actor-slot checkpoint:

```bash
source /content/poguise_colab_env.sh
cd "$REPO_DIR"

TS=$(date +%Y%m%d_%H%M%S)
RUN_NAME="actor_interaction_heatmap_warmup_${TS}"
EPOCH_DIR="$DATA_DIR/checkpoints/$RUN_NAME/epoch_checkpoints"

python3 -u train.py \
  --model_file "$RESUME_CKPT_PATH" --strict_load 0 \
  --dataset toyotasm --dataset_artifact toyotasm --data_dir "$DATA_DIR" \
  --toyota_frame_source frames --toyota_skeleton_zip "$SKELETON_ZIP" \
  --toyota_frame_count_cache "$FRAME_COUNT_CACHE" \
  --toyota_object_cache_dir "$DATA_DIR/toyota_preprocessed_cache/objects" \
  --toyota_landmark_cache_dir "$DATA_DIR/toyota_preprocessed_cache/landmarks" \
  --toyota_split_source auto --toyota_val_fraction 0.15 --toyota_test_fraction 0.0 \
  --pretrained none --net_size b --num_classes 31 --n_landmarks 13 --hw_out_conv 8 \
  --actor_prompt 1 --num_actor_tokens 8 --actor_presence_head 1 \
  --presence_loss_weight 0.05 --actor_interaction_heatmaps 1 \
  --object_detector_cache "$OBJECT_DETECTOR_CACHE" \
  --object_camera_allowlist tv_monitor=c05,c06 \
  --object_ignore_regions c03=0,0,0.26,0.42 \
  --object_conf_threshold 0.25 --interaction_heatmap_size 56 \
  --interaction_heatmap_sigma 1.5 \
  --interaction_guided_sampling 1 --interaction_min_sampled_object_frames 1 \
  --interaction_repair_radius_frames 8 \
  --interaction_quality_min_actor_score 1.0 \
  --interaction_quality_min_track_frames 1 \
  --interaction_quality_min_track_coverage 0.0 \
  --freeze_backbone 1 --interaction_warmup_freeze_actor_path 1 \
  --interaction_unfreeze_last_blocks 2 \
  --class_balanced_sampler 1 --hard_negative_sampler 1 \
  --hard_negative_manifest "$HARD_NEGATIVE_MANIFEST" --hard_negative_prob 0.15 \
  --keep_rate 0.6 --keep_rate_merge 0.3 --merge_type tome --merge_mode 0 \
  --sim_metric 0 --topk_type 1 \
  --mixup 0 --grad_weights 1 --nash_update_weights_every 20 --nash_max_norm 1.0 \
  --poguiseplus_heatmap_loss_weight 1.0 --poguiseplus_heatmap_log_eps 1e-6 \
  --kp_only 0 \
  --toyota_pose_guided_sampling 1 --toyota_min_pose_frames 1 \
  --toyota_synthetic_warmup_epochs 0 \
  --batch_size 64 --accum_grad_batches 1 --max_epochs 3 --t_max_scheduler 3 \
  --lr 5e-7 --lr_head 0 --lr_head_hm 5e-5 \
  --weight_decay 0.04 --weight_decay_head 0.01 --weight_decay_head_hm 0.01 \
  --label_smoothing 0.1 --gradient_clip_val 1.5 \
  --num_workers 12 --persistent_workers 1 --prefetch_factor 2 \
  --precision bf16-mixed --accelerator gpu --gpus 1 \
  --limit_val_batches 1.0 --check_val_every_n_epoch 1 \
  --num_sanity_val_steps 0 --log_every_n_steps 50 \
  --checkpoint_monitor val_interaction_heatmap_iou --checkpoint_mode max \
  --checkpoint_filename '{epoch:03d}-{val_f1:.4f}-{val_interaction_heatmap_iou:.4f}' \
  --save_top_k 3 --save_every_epoch_checkpoints 1 \
  --epoch_checkpoint_dir "$EPOCH_DIR" --epoch_checkpoint_filename '{epoch:03d}' \
  --default_root_dir "$DATA_DIR/checkpoints" --model_name "$RUN_NAME"
```

Check results after each epoch:

```bash
source /content/poguise_colab_env.sh
cd "$REPO_DIR"
python3 summarize_interaction_metrics.py \
  --pattern 'actor_interaction_heatmap_warmup_*'
```

Important metrics:

```text
val_interaction_heatmap_iou
val_interaction_heatmap_positive_mean
val_interaction_heatmap_center_l2
val_interaction_teacher_slot_rate
val_interaction_teacher_slot_count
val_loss_heatmap_log
train_nash_weight_action
train_nash_weight_heatmap
val_group_laptop_book_tv_acc
val_group_phone_tv_acc
val_group_drink_cup_bottle_glass_acc
val_action_Uselaptop_acc
val_action_Readbook_acc
val_action_WatchTV_acc
val_action_Usetelephone_acc
val_action_Drink_Fromcup_acc
val_action_Drink_Frombottle_acc
val_action_Drink_Fromglass_acc
val_action_Pour_Frombottle_acc
val_action_Cutbread_acc
val_action_Cook_Cut_acc
val_action_Cook_Stir_acc
val_action_Cook_Cleandishes_acc
val_action_Cook_Usestove_acc
val_acc_macro
val_f1
```

## Fine Tune

Use the best warmup checkpoint by interaction heatmap quality while preserving
reasonable action macro/F1. Then run a short fine-tune:

```bash
source /content/poguise_colab_env.sh
cd "$REPO_DIR"

WARMUP_CKPT="/path/to/epoch=XXX.ckpt"
TS=$(date +%Y%m%d_%H%M%S)
RUN_NAME="actor_interaction_heatmap_fullft_${TS}"
EPOCH_DIR="$DATA_DIR/checkpoints/$RUN_NAME/epoch_checkpoints"

python3 -u train.py \
  --model_file "$WARMUP_CKPT" --strict_load 0 \
  --dataset toyotasm --dataset_artifact toyotasm --data_dir "$DATA_DIR" \
  --toyota_frame_source frames --toyota_skeleton_zip "$SKELETON_ZIP" \
  --toyota_frame_count_cache "$FRAME_COUNT_CACHE" \
  --toyota_object_cache_dir "$DATA_DIR/toyota_preprocessed_cache/objects" \
  --toyota_landmark_cache_dir "$DATA_DIR/toyota_preprocessed_cache/landmarks" \
  --toyota_split_source auto --toyota_val_fraction 0.15 --toyota_test_fraction 0.0 \
  --pretrained none --net_size b --num_classes 31 --n_landmarks 13 --hw_out_conv 8 \
  --actor_prompt 1 --num_actor_tokens 8 --actor_presence_head 1 \
  --presence_loss_weight 0.05 --actor_interaction_heatmaps 1 \
  --object_detector_cache "$OBJECT_DETECTOR_CACHE" \
  --object_camera_allowlist tv_monitor=c05,c06 \
  --object_ignore_regions c03=0,0,0.26,0.42 \
  --object_conf_threshold 0.25 --interaction_heatmap_size 56 \
  --interaction_heatmap_sigma 1.5 \
  --interaction_guided_sampling 1 --interaction_min_sampled_object_frames 1 \
  --interaction_repair_radius_frames 8 \
  --interaction_quality_min_actor_score 1.0 \
  --interaction_quality_min_track_frames 1 \
  --interaction_quality_min_track_coverage 0.0 \
  --freeze_backbone 0 --interaction_warmup_freeze_actor_path 0 \
  --interaction_unfreeze_last_blocks 0 \
  --class_balanced_sampler 1 --hard_negative_sampler 1 \
  --hard_negative_manifest "$HARD_NEGATIVE_MANIFEST" --hard_negative_prob 0.15 \
  --keep_rate 0.6 --keep_rate_merge 0.3 --merge_type tome --merge_mode 0 \
  --sim_metric 0 --topk_type 1 \
  --mixup 0 --grad_weights 1 --nash_update_weights_every 20 --nash_max_norm 1.0 \
  --poguiseplus_heatmap_loss_weight 1.0 --poguiseplus_heatmap_log_eps 1e-6 \
  --kp_only 0 \
  --toyota_pose_guided_sampling 1 --toyota_min_pose_frames 1 \
  --toyota_synthetic_warmup_epochs 0 \
  --batch_size 48 --accum_grad_batches 1 --max_epochs 4 --t_max_scheduler 4 \
  --lr 5e-7 --lr_head 2e-5 --lr_head_hm 2e-5 \
  --weight_decay 0.04 --weight_decay_head 0.01 --weight_decay_head_hm 0.01 \
  --label_smoothing 0.1 --gradient_clip_val 1.5 \
  --num_workers 12 --persistent_workers 1 --prefetch_factor 2 \
  --precision bf16-mixed --accelerator gpu --gpus 1 \
  --limit_val_batches 1.0 --check_val_every_n_epoch 1 \
  --num_sanity_val_steps 0 --log_every_n_steps 50 \
  --checkpoint_monitor val_interaction_heatmap_iou --checkpoint_mode max \
  --checkpoint_filename '{epoch:03d}-{val_f1:.4f}-{val_interaction_heatmap_iou:.4f}' \
  --save_top_k 3 --save_every_epoch_checkpoints 1 \
  --epoch_checkpoint_dir "$EPOCH_DIR" --epoch_checkpoint_filename '{epoch:03d}' \
  --default_root_dir "$DATA_DIR/checkpoints" --model_name "$RUN_NAME"
```

Check results:

```bash
source /content/poguise_colab_env.sh
cd "$REPO_DIR"
python3 summarize_interaction_metrics.py \
  --pattern 'actor_interaction_heatmap_fullft_*'
```

## Runtime

Use the actor dashboard/export path. Do not pass RF-DETR detections to the action
model. RF-DETR may still run for separate visualization, but it is not part of
the actor checkpoint input contract.
