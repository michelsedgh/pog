# Actor Object Interaction Runbook

This runbook is for the current Actor-Slot PO-GUISE+ object path.

The approved architecture is:

```text
video patch tokens
+ actor tokens
+ object tokens
+ heatmap tokens
through the transformer
-> actor-object decoder updates actor tokens
-> actor_head(refined_actor_tokens)
```

The object path is not a logit residual, reranker, specialist branch, or runtime
class rule. Object evidence must change the actor token that the normal
`actor_head` reads.

## What Changed

- `models/actor_object_decoder.py` owns actor/object binding.
- The decoder emits selection logits and interaction heatmaps, but also returns
  refined actor tokens.
- `models/poguise.py` runs `actor_head` only after the decoder has updated
  `x_actor`.
- `positive_erased` now removes positive object tokens and erases the positive
  object patch-grid features before the transformer.
- The training loss can include true-logit sensitivity and group-margin
  sensitivity for object-sensitive action groups.
- Invalid object tokens remain masked in transformer attention and neutral in
  token construction.

## Cell 1

Use the Colab setup cell unchanged. Later cells read:

```text
/content/poguise_colab_env.sh
```

That file must export `REPO_DIR`, `DATA_DIR`, `SKELETON_ZIP`,
`RESUME_CKPT_PATH`, `OBJECT_DETECTOR_CACHE`, `HARD_NEGATIVE_MANIFEST`,
`FRAME_COUNT_CACHE`, and `TOYOTA_FRAMES_DIR`.

## Preflight Checks

After `git pull`, verify these before training:

```bash
cd /content/pog
git log -1 --oneline
python3 -m py_compile \
  blocks/poguise.py \
  models/poguise.py \
  models/actor_object_decoder.py \
  modules/heatmap_module.py \
  losses/object_interaction_losses.py \
  train.py
python3 object_actor_live/smoke_actor_object_decoder_identity.py --device cuda
```

Also run a smoke forward. The expected output shape is:

```text
action_logits:       [1, 8, 31]
heatmaps:            [1, 32, 56, 56]
presence_logits:     [1, 8]
selection_logits:    [1, 8, 25]
interaction_heatmap: [1, 8, 56, 56]
```

If the checkpoint was trained before this decoder refactor, it is stale. Train a
new checkpoint from the clean actor-slot checkpoint with `--strict_load 0`.

## Warmup

Use a short object warmup from the clean actor-slot checkpoint.

Important flags:

```bash
--object_prompt 1
--object_warmup_freeze_actor_path 1
--freeze_backbone 1
--object_unfreeze_last_blocks 2
--lr 5e-7
--lr_head 5e-5
--lr_head_hm 5e-5
--object_decoder_update_gate_init 0.02
--object_decoder_ffn_gate_init 0.02
--object_interaction_loss_weight 0.10
--object_interaction_heatmap_weight 50
--object_counterfactual_margin_weight 0.05
--object_action_sensitivity_weight 0.10
--object_action_sensitivity_margin 0.05
--object_action_group_sensitivity_weight 0.10
--object_action_group_sensitivity_margin 0.05
--object_objectless_consistency_weight 0.02
--max_epochs 2
--t_max_scheduler 2
```

Why `--object_unfreeze_last_blocks 2`: the decoder can learn actor/object
binding while the last transformer blocks learn to carry object-token evidence
into actor tokens. Do not unfreeze the whole model for warmup.

## Fine-Tune

Only fine-tune from the best warmup checkpoint if the object diagnostics are not
obviously broken. Use a short run:

```bash
--freeze_backbone 0
--object_warmup_freeze_actor_path 0
--lr 5e-7
--lr_head 3e-5
--lr_head_hm 3e-5
--max_epochs 2
--t_max_scheduler 2
```

Keep the sensitivity losses on. Stop early if objects-on falls below both
objects-off and shuffled.

## Metrics That Matter

Do not choose by `val_loss` alone.

Primary checks:

```text
val_object_interaction_margin_gain_on_vs_positive_erased > 0
val_object_interaction_margin_gain_on_vs_shuffled > 0
val_interaction_select_mass_object is sane, not collapsed
val_interaction_select_acc_object is improving
val_obj_iou and val_obj_recall_visible are nonzero
val_f1_objects_on is not materially worse than objects_off
val_f1_objects_on is not materially worse than objects_shuffled
```

Per-class checks:

```text
Uselaptop
Readbook
WatchTV
Usetelephone
Drink.Fromcup
Drink.Frombottle
Drink.Fromglass
```

Toyota validation may still show `Uselaptop = 1.0` across modes. That does not
prove the live laptop case is fixed. The saved live tensor sensitivity test is
mandatory for that.

## Live Tensor Sensitivity Test

After training a new decoder checkpoint:

```bash
python object_actor_live/analyze_live_object_sensitivity.py \
  --checkpoint /path/to/epoch.ckpt \
  --input-pt object_actor_live/latest_epoch002_actor_input.pt \
  --device cuda \
  --dtype auto \
  --skip-gradient
```

The important rows are:

```text
objects_on
objects_off
laptop_only
positive_erased_laptop
positive_erased_laptop_visual
laptop_class_changed_to_book
laptop_box_moved_away
```

A useful Uselaptop checkpoint should make the Uselaptop logit rise for
`laptop_only` versus `objects_off`, and drop for `positive_erased_laptop_visual`
versus `objects_on`.
