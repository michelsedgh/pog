# Object Relation Architecture

## Problem

The model must distinguish actions with similar pose and motion by using the objects near the actor. A global clip-level object summary is not enough: it can learn scene priors, but it cannot answer which object the actor is using when several objects are present.

The concrete failure mode is `Uselaptop` being predicted as `Readbook`. The useful signal is not only "a laptop exists"; it is "this actor token should attend to this detected laptop/book/phone/cup token when forming the action logits."

## Evidence

- PO-GUISE uses semantic heatmap tokens to guide token selection and preserve actor-relevant visual evidence. Its later PO-GUISE+ variant explicitly adds interacting-object prediction as a task for driver action recognition: https://arxiv.org/abs/2407.13750.
- Actor-Centric Relation Network models actor-to-context/object relations for action recognition instead of relying on global scene features: https://arxiv.org/abs/1807.10982.
- Skeleton/interacted-object work frames action recognition and interacted-object localization as mutually helpful tasks, which matches our available weak labels from detector cache plus action labels: https://arxiv.org/abs/2110.14994.
- Our object-only and frozen-logit probes showed real object features contain signal, but the previous global summary residual did not reliably beat object-off/shuffled evaluations in full training.

## Design

Use one object path:

1. Keep actor slots as the main action representation.
2. Keep detector boxes/classes/confidences as object tokens.
3. Append an explicit `NONE` object token for objectless or uncertain actions.
4. Let every actor slot query all object tokens with cross-attention.
5. Build relation features from `[actor, attended_object_context, actor * attended_object_context]`.
6. Predict:
   - an action-logit residual added to the actor logits through one scalar gate;
   - an auxiliary interacted-object class for each actor slot.
7. Evaluate object usefulness with real objects, objects-off, and objects-shuffled. A useful object path must make real objects beat both off and shuffled on macro accuracy and F1.

This is intentionally not an action-class prior. The model sees all detected objects and learns relevance from actor features, object class, object location, confidence, and the interacted-object auxiliary target.

## Training Modes

### Diagnostic relation-only

Start from a strong actor checkpoint, freeze the video/actor path, and train only the object relation modules. This tests the architecture without spending full-run time or corrupting the actor model.

Suggested settings:

- `--freeze_backbone 1`
- `--object_relation_only 1`
- `--object_relation_gate_init 0.25`
- `--object_relation_hidden_dim 512`
- `--object_relation_dropout 0.1`
- `--object_interaction_loss_weight 0.2`
- `--object_heatmap_weight 0`
- `--kp_loss_weight 0`
- `--lr_head 3e-4`
- `--max_epochs 6` to `8`

Acceptance signal:

- `val_acc_macro_objects_on > val_acc_macro_objects_off`
- `val_acc_macro_objects_on > val_acc_macro_objects_shuffled`
- `val_f1_objects_on >= val_f1_objects_off - 0.003`
- object-related groups improve without a broad F1 collapse.

### Full fine-tune

Only after the diagnostic passes, unfreeze the normal actor path and keep object heatmap/token-pruning guidance on. This trains the final integrated model rather than another blind architecture experiment.

## Rejected Path

The old object-summary residual path is removed. It used clip-level object statistics and optional hand-coded action evidence gates. That makes it too easy to encode dataset priors, and it does not directly solve actor-object association.
