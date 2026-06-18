# Training Knob Audit

Current objective: one actor-object relation path.

For every actor token, the model learns a single distribution over `NULL` plus
the detected object prompt slots. That relation update changes the actor token
inside the transformer, and the normal actor action head predicts the action.

## Active Knobs

Relation:

```text
--actor_object_prompt_tokens 1
--actor_object_relation_in_transformer 1
--actor_object_relation_blocks 2,5,8
--actor_object_relation_loss_weight 1.00
--actor_object_relation_null_loss_weight 0.50
--actor_object_relation_null_logit_init 3.0
--actor_object_relation_geometry_bias_weight 1.0
--actor_object_relation_heatmap_bias_weight 2.0
--actor_object_relation_max_scale 2.0
```

Object prompts and token selection:

```text
--num_scene_object_tokens 32
--num_object_classes 19
--object_conf_threshold 0.25
--token_selection_cls_weight 0.15
--token_selection_actor_weight 0.25
--token_selection_object_weight 0.30
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
before `actor_head`, so action CE trains the real object-conditioned actor token.

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
