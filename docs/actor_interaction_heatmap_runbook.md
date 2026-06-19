# Actor Interaction Heatmap Runbook

PO-GUISE+ heatmaps are auxiliary supervision for where the actor is interacting
with the scene. In this repo they also support relation selection by giving the
object relation update spatial evidence.

## Active Path

For each actor:

```text
actor token + relation-only object memory + interaction heatmap features
    -> relation logits over NULL plus object slots
    -> object context
    -> updated actor token inside the transformer and before the action head
    -> actor_head action logits
```

Heatmaps do not directly predict the action. They help localize the interaction
region and support object slot selection.

## Recommended Run

Use the focused launcher:

```bash
python poguise+_+objects.py --epochs 10
```

The launcher uses:

```text
--actor_interaction_heatmaps 1
--actor_object_prompt_tokens 1
--actor_object_relation_in_transformer 1
--actor_object_relation_blocks 2,5,8
--actor_object_relation_loss_weight 1.00
--poguiseplus_interaction_heatmap_weight 3.0
```

## Metrics

Watch action quality first:

```text
val_f1
val_acc_macro
val_action_Uselaptop_acc
val_action_Readbook_acc
val_action_WatchTV_acc
```

Then relation quality:

```text
val_relation_exact_teacher_acc
val_relation_exact_teacher_prob
val_relation_useful_mass_exact
val_relation_null_rate_objectless
val_relation_null_rate_missing_objectful
```

Then heatmap support:

```text
val_interaction_heatmap_positive_mean
val_interaction_heatmap_pred_max
val_interaction_heatmap_soft_iou
val_interaction_heatmap_center_l2
```

Good relation metrics without good action metrics used to mean the selected
object was not changing the final actor token strongly enough, or object tokens
were leaking through generic self-attention. The current path fixes that by
keeping detections as relation-only memory and adding final pre-head relation
fusion, not with inference overrides.
