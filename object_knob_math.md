# Object Knob Math

The current object path has one supervised object assignment:

```text
relation_logits[b, a] in R^(1 + K)
```

where:

```text
index 0      = NULL
index 1..K   = object prompt slots
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
actor_token'     = actor_token + gated_update(actor_token, object_context)
```

The real action prediction remains:

```text
action_logits = actor_head(actor_token')
L_action      = CE(action_logits, action_label)
```

The update uses `actor_token`, `object_context`, and their elementwise product,
so the adapter can learn interaction-specific changes instead of just adding a
generic object vector. The same relation module runs inside the transformer
before pruning stages and once more immediately before `actor_head`.

So the causal path is:

```text
object prompts -> relation distribution -> object context -> actor token -> action head
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

Caps how strongly relation updates can change actor tokens. Too low means object
context may not affect action top-1; too high can destabilize action learning.

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
