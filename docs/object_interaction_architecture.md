# Object-Token PO-GUISE+ Architecture

## Problem

The actor-slot model must distinguish actions with similar pose and motion by using object evidence around the actor. The live failure is `Uselaptop` collapsing into `Readbook`, `WatchTV`, or `Usetelephone` even when RF-DETR detects the laptop.

The previous object path was cleaner than the old specialist experiments, but it still fused object information after the video transformer had already produced actor features. That made object evidence a late correction instead of part of the token-selection and actor-feature formation process.

## Final Design

There is one active object path:

1. Feed all RF-DETR object detections during training and inference.
2. Embed object boxes, classes, confidences, validity, and pooled object-region
   patch features as transformer tokens.
3. Protect class, actor, object, register, and heatmap tokens from pruning.
4. Let actor tokens attend to object tokens inside the PO-GUISE transformer blocks.
5. Use the final actor tokens directly for action classification.
6. Predict actor/object selection and actor-conditioned interaction heatmaps from
   final actor/object tokens plus pooled actor-object union visual features.
7. Train with strong-object selection targets and actor-conditioned interaction
   heatmaps. Positive-erased and shuffled object modes are diagnostics, not
   training losses.

The model no longer has a post-backbone object adapter, free action-score overrides, specialist rerankers, or relation-only modes. Object evidence must enter through transformer tokens and affect the actor token before `actor_head`.

This is the final architecture to test before collecting new live-like data. Do
not add another object bridge or training branch unless the controlled warmup,
short unfreeze, and live tensor A/B show that this path still cannot move the
target action logits.

## Runtime Inputs

Training and inference use the same object contract:

```text
video:        [B, T, 3, 224, 224]
actor_boxes:  [B, K, 4]
actor_valid:  [B, K]
object_boxes: [B, M, 4]
object_cls:   [B, M]
object_conf:  [B, M]
object_valid: [B, M]
```

The current Toyota/live configuration uses `K=8`, `M=24`, and 19 COCO-derived object classes.

## Transformer Token Order

```text
[class]
[actor tokens]
[object tokens]
[register tokens]
[heatmap tokens]
[video tokens]
```

Class, actor, object, register, and heatmap tokens are semantic tokens. They are not pruned. Video-token pruning scores visual tokens using attention from all semantic tokens when `topk_type=1`.

## Visual Grounding

Object tokens are not metadata-only tokens. Each object token includes a pooled
visual descriptor from the fixed patch grid:

```text
object_token =
  object_slot_embed
+ object_cls_embed(object_cls)
+ object_bbox_mlp(object_box)
+ object_conf_mlp(object_conf)
+ object_visual_proj(pool_patch_features_inside_object_box)
+ object_valid_embed(object_valid)
```

When adding object tokens to a clean actor-slot checkpoint, real object-class
embedding rows must keep their constructor initialization. Only the NONE/padding
row is zero. Do not zero all class rows, or `laptop`, `book`, `phone`, and other
objects become indistinguishable at warmup start.

Invalid/padded object slots are neutralized before transformer attention:

- the NONE object class id uses `padding_idx=num_object_classes`
- invalid object tokens are zeroed after token construction
- invalid object-token key positions are masked in every attention block
- invalid object-token query rows are excluded from token-pruning attention scores
- invalid object-token features are zeroed after each block

This makes `objects_off` and `positive_erased` mean that the corresponding
object evidence is genuinely unavailable to the actor tokens, not just hidden
from the later selection loss.

The pooling is fixed-grid tensor math over the patch embedding output, not
ROIAlign. This keeps the path compatible with ONNX export and avoids dynamic
crop operations.

The actor/object selection head also receives visual evidence from each
actor-object union region:

```text
selection_score(actor, object) =
  MLP(actor_token, object_token, actor_token * object_token, geometry, union_visual)
```

The NONE option is scored by a separate actor-conditioned MLP:

```text
none_score(actor) = none_mlp(actor_token)
selection_logits = concat(object_scores, none_score)
```

NONE is not represented by a fake object token. Real object logits are masked by
`object_valid`; the NONE logit is always available for objectless or weak-context
actions.

This is the intended path for learning interaction evidence such as laptop-on-lap
or book-in-hands without hard-coded hand/nearest-object rules.

## Strong Object Targets

Strong interaction supervision is only used for reliable pairs:

```text
Uselaptop        -> laptop, keyboard_mouse
Readbook         -> book
Usetelephone     -> phone
Drink.Fromcup    -> cup
Drink.Frombottle -> bottle
Drink.Fromglass  -> glass
```

All other detected objects remain visible as context/distractors. They are not treated as positive interacted-object targets.

## Losses

Use the normal action CE as the main task, plus:

- pose heatmap loss
- actor/object selection loss
- actor-conditioned interaction heatmap loss
- actor presence loss

There is no all-visible-object heatmap target. Positive-erased removes only the
positive interacted-object candidates and leaves distractors visible, but this is
used as a validation diagnostic rather than a training branch.

## Training Schedule

Start from a clean actor-slot checkpoint.

Frozen object-token warmup:

- `--object_prompt 1`
- `--freeze_backbone 1`
- `--object_warmup_freeze_actor_path 1`
- `--object_unfreeze_last_blocks 2`
- actor classifier, presence head, and global classifier are frozen
- `--object_interaction_loss_weight 0.03`
- `--object_interaction_heatmap_weight 25`
- `--kp_loss_weight 1000`
- `--lr 5e-7`
- `--lr_head 5e-5`
- `--lr_head_hm 5e-5`
- `--class_balanced_sampler 1`
- `--hard_negative_sampler 0`
- `--max_epochs 2`

`--object_unfreeze_last_blocks 2` is intentional. The actor/object decoder
learns explicit binding, while the final transformer blocks learn to carry
object-token evidence into actor tokens. Do not unfreeze the whole model during
warmup.

Only after the frozen warmup passes, run a short integrated fine-tune:

- `--freeze_backbone 0`
- `--object_warmup_freeze_actor_path 0`
- `--lr 5e-7`
- `--lr_head 3e-5`
- `--lr_head_hm 3e-5`
- `--max_epochs 2`

## Acceptance Criteria

Do not judge only raw global macro/F1. A pass requires:

- real objects beat or match objects-off and shuffled globally
- positive-erased margin gain is positive for strong object actions
- shuffled-object margin gain is positive
- selection mass and selection accuracy are sane
- interacted-object heatmap IoU/positive response are nonzero
- weak/context classes remain stable
- per-class checks for `Uselaptop`, `Readbook`, `WatchTV`, `Usetelephone`, and drink classes do not reveal a hidden regression

Toyota validation cannot fully prove the live couch-laptop case because `Uselaptop` is often already saturated there. The real live A/B remains necessary, but the model architecture must pass the object diagnostic metrics first.
