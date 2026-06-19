# Training Knob Audit

Current objective: one actor-object relation path.

For every valid actor token, the model learns a single distribution over `NULL`
plus the detected object slots. Runtime detections are relation-only object
memory rather than ordinary transformer prefix tokens, so object presence cannot
bypass the relation update through generic self-attention. The normal actor
action head predicts the action from the updated actor token.

The default object run now uses two actor tokens because side-by-side actor-slot
training is active. Slot padding is still masked; the second slot is real in the
paired training examples and padded in ordinary one-person examples.

## Active Knobs

Relation:

```text
--actor_object_prompt_tokens 1
--actor_object_relation_in_transformer 1
--actor_object_relation_blocks 2,5,8
--actor_object_relation_loss_weight 1.00
--actor_object_relation_null_loss_weight 1.00
--actor_object_relation_null_logit_init 3.5
--actor_object_relation_geometry_bias_weight 1.0
--actor_object_relation_heatmap_bias_weight 2.0
--actor_object_relation_max_scale 1.5
--actor_object_relation_learned_scale 1
--actor_object_relation_layer_scale_init 0.25
--actor_relation_action_fusion 1
```

Object memory and token selection:

```text
--num_scene_object_tokens 32
--num_object_classes 19
--object_conf_threshold 0.25
--token_selection_cls_weight 0.15
--token_selection_actor_weight 0.25
--token_selection_object_weight 0.00
--token_selection_heatmap_weight 0.30
--actor_object_prompt_box_prior_weight 0.20
--actor_object_prompt_box_prior_expand 1.50
```

PO-GUISE+ heatmaps:

```text
--actor_interaction_heatmaps 1
--poguiseplus_heatmap_loss_weight 1.0
--poguiseplus_pose_heatmap_weight 0.25
--poguiseplus_interaction_heatmap_weight 3.0
--poguiseplus_normalized_heatmap_loss 1
--poguiseplus_heatmap_mse_scale 1000
```

Regular action training:

```text
--num_actor_tokens 2
--actor_pair_train_weight 0.50
--class_balanced_sampler 1
--batch_size 32
--accum_grad_batches 2
--lr 3e-5
--lr_head 5e-4
--lr_head_hm 5e-4
--grad_weights 1
```

## What Not To Reintroduce

Do not add duplicate object-state heads or separate object-action heads. The
relation path now updates actor tokens before the pruning stages and again right
before classification. The final actor classifier also receives a zero-initialized
learned fusion of actor token, selected object context, actor/object product, and
object mass, so action CE trains the real object-conditioned action path.

Do not put runtime object detections back into ordinary transformer self-attention.
Runtime detections should influence action through relation-selected object
memory and object-box pruning priors, not through global object-presence
attention shortcuts.

Do not add fake labels based on swapping object names. The dataset label remains
the action CE target; object supervision is the relation slot target.

Do not add inference rules that override action logits. They hide the model
failure and make deployment brittle.

## What To Read In A Run

A promising run should show:

- `val_f1` and key action accuracies improving or staying stable
- `val_relation_exact_teacher_acc` rising
- `val_relation_exact_teacher_prob` rising
- `val_relation_useful_mass_exact` high on compatible object examples
- `val_relation_null_rate_objectless` high
- `val_relation_null_rate_missing_objectful` high
- `val_interaction_heatmap_soft_iou` and positive response improving

A bad run can still have high relation metrics if the action head ignores the
updated actor token. In that case, do not add more side losses; inspect and
strengthen the object-context update before `actor_head`.
