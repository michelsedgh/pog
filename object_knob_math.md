# Object Knob Math

The current object path has one supervised object assignment:

```text
relation_logits[b, a] in R^(1 + K)
```

where:

```text
index 0      = NULL
index 1..K   = detected object memory slots
```

For actor `a` in batch item `b`:

```text
L_relation = CE(relation_logits[b, a], relation_target[b, a])
```

The target is the compatible teacher object slot when available. It is `NULL`
when the action is objectless or the expected object is missing.

The relation update also computes object attention over non-NULL slots:

```text
object_attention = softmax(relation_logits)[..., 1:]
object_context   = sum(object_attention[k] * object_token[k])
update_strength  = learned_gate * non_null_mass * actor_valid
actor_token'     = actor_token + update_strength * update(actor_token, object_context)
```

The real action prediction remains:

```text
action_logits = actor_head(actor_token')
L_action      = CE(action_logits, action_label)
```

The update uses `actor_token`, `object_context`, and their elementwise product,
so the adapter can learn interaction-specific changes instead of just adding a
generic object vector. Runtime object detections are relation-only memory, not
ordinary transformer prefix tokens, so object presence cannot bypass relation
selection through generic self-attention. The same relation module runs inside
the transformer before pruning stages and once more immediately before
`actor_head`.

So the causal path is:

```text
object memory -> relation distribution -> object context -> actor token -> action head
```

## Important Weights

`actor_object_relation_loss_weight`

Scales relation CE. Too low and the model may not select the interacted object.
Too high and the auxiliary relation task can dominate action learning.

`actor_object_relation_null_loss_weight`

Weights NULL examples in relation CE. This protects objectless actions and
missing-object cases.

`actor_object_relation_null_logit_init`

Initial NULL bias. A higher value makes early training conservative about using
objects.

`actor_object_relation_geometry_bias_weight`

Adds actor/object geometry prior into relation logits. This should help choose
nearby/intersecting objects without becoming a hard rule.

`actor_object_relation_heatmap_bias_weight`

Adds interaction heatmap evidence into relation logits. This connects PO-GUISE+
heatmap localization with object slot selection.

`actor_object_relation_max_scale`

Caps how strongly relation updates can change actor tokens. In the default
object run the actual strength is learned per relation block from
`actor_object_relation_layer_scale_init`; `max_scale` is only the upper bound.

`actor_relation_action_fusion`

Enables the final action input fusion:

```text
fuse(actor_token, selected_object_context, actor_token * selected_object_context, object_mass)
```

This keeps the same action CE target while making the selected object context
explicitly available to `actor_head`.

## Reading Metrics

Relation quality:

```text
val_relation_exact_teacher_acc
val_relation_exact_teacher_prob
val_relation_useful_mass_exact
val_relation_null_rate_objectless
val_relation_null_rate_missing_objectful
```

Action quality:

```text
val_f1
val_acc_macro
val_action_Uselaptop_acc
val_action_Readbook_acc
val_action_WatchTV_acc
val_action_Usetelephone_acc
```

Token/heatmap support:

```text
val_token_selection_exact_teacher_object_keep_rate
val_token_selection_interaction_heatmap_keep_rate
val_interaction_heatmap_soft_iou
val_interaction_heatmap_positive_mean
val_interaction_heatmap_center_l2
```

If relation metrics are good but live action remains wrong after this change, the
next suspect is data/domain coverage or detector/object proposal quality, not a
separate missing side classifier.
