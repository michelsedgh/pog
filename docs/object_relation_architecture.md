# Object Relation Architecture

## Problem

The model must distinguish actions with similar pose and motion by using the objects near the actor. A global clip-level object summary is not enough: it can learn scene priors, but it cannot answer which object the actor is using when several objects are present.

The concrete failure mode is `Uselaptop` being predicted as `Readbook`, or `WatchTV` being corrupted by a visible laptop. The useful signal is not only "a laptop exists"; it is "for this actor, which detected object region explains the action?"

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
4. Pool object visual features from the PO-GUISE heatmap-token feature map at each detected object box.
5. Score every actor/object pair. The selection tensor is `[B, K, M + 1]`, where the final object slot is `NONE`.
6. Train object selection with a positive object-token set, not a single guessed instance. For example, `Uselaptop` accepts any detected laptop or keyboard/mouse token for that actor.
7. Build the actor-conditioned interaction heatmap from the selected object distribution.
8. Supervise that interaction heatmap at the same 56x56 resolution as the PO-GUISE pose/object heatmaps. Build an 8x8 copy only for pooling from the internal heatmap-token feature grid.
9. Pool visual context through the actor-conditioned interaction heatmap.
10. Predict a bounded actor-feature adapter from:
   `[selected_object_context, interaction_visual_context, actor * selected_object_context, actor * interaction_visual_context]`.
11. Fuse at the actor feature level:
   `fused_actor = actor_feature + gamma * tanh(adapter) * real_object_mass`.
12. Reuse the normal actor action head on `fused_actor`.
13. Do not apply an object logit residual. The object path modifies representation, not final class logits directly.
14. Train real-object counterfactuals against positive-object-erased and label-mismatched objects-shuffled views. The erased view removes only the weak-positive interacted objects while leaving distractors visible, so the model must learn which visible object actually supports the action. Objectless actions are trained to stay consistent with objects-off.
15. Evaluate object usefulness with real objects, objects-off, and label-mismatched objects-shuffled. A useful object path must keep real objects competitive globally and improve the targeted object-sensitive classes.

This is intentionally not an action-class prior or object-existence shortcut. The model sees all detected objects, then learns which object token and spatial region explain each actor's action. Training and inference both pass the same RF-DETR object format: boxes, object class ids, confidences, and a valid mask.

## Training Modes

### Frozen object warmup

Start from a strong actor checkpoint, freeze the video/actor path, and train only the actor-object feature-fusion modules plus heatmap head. This tests PO-GUISE+ style object grounding without spending full-run time or corrupting the actor model.

Suggested settings:

- `--freeze_backbone 1`
- `--object_warmup_freeze_actor_path 1`
- `--object_fusion_gate_init 0.04`
- `--object_delta_scale 0.5`
- `--object_relation_hidden_dim 512`
- `--object_relation_dropout 0.05`
- `--object_heatmap_weight 50`
- `--object_interaction_loss_weight 0.10`
- `--object_interaction_heatmap_weight 50`
- `--object_residual_l2_weight 0.02`
- `--object_counterfactual_margin_weight 0.10`
- `--object_counterfactual_margin 0.05`
- `--object_objectless_consistency_weight 0.03`
- `--kp_loss_weight 1000`
- `--lr_head 1e-4`
- `--lr_head_hm 5e-5`
- `--class_balanced_sampler 1`
- `--hard_negative_sampler 1`
- `--hard_negative_prob 0.15`
- `--max_epochs 4`

Acceptance signal:

- `val_acc_macro_objects_on >= val_acc_macro_objects_off + 0.005`
- `val_acc_macro_objects_on >= val_acc_macro_objects_shuffled + 0.005`
- `val_f1_objects_on >= val_f1_objects_off - 0.003`
- `val_f1_objects_on >= val_f1_objects_shuffled - 0.003`
- `val_interaction_select_mass_object` increases for strong object actions.
- object-related groups improve without a broad F1 collapse.
- `val_object_interaction_true_logit_gain_on_vs_positive_erased` and `val_object_interaction_margin_gain_on_vs_positive_erased` are positive for strong object actions.

### Full fine-tune

Only after the frozen warmup passes, unfreeze the normal actor path and keep visible-object heatmaps/token-pruning guidance on. This trains the final integrated model rather than another blind architecture experiment.

## Rejected Path

The old object-summary residual path is removed. It used clip-level object statistics and optional hand-coded action evidence gates. That makes it too easy to encode dataset priors, and it does not directly solve actor-object association.

The old action-logit residual path is also removed. It did find object signal, but it could still push final class logits in unstable ways. The clean path uses actor-conditioned object selection plus feature-level fusion and lets the normal actor head decide the action.

The shuffled-object negative must be label-mismatched, not a simple adjacent batch roll. Toyota validation order is deterministic, so adjacent samples can be correlated by sorted file id, scene, or action family. A valid shuffled negative should break the action/object relationship instead of replacing objects with a nearby similar clip.
