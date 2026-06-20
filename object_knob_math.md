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

Each detected object slot is represented as:

```text
object_token =
    object_slot_embedding
  + object_class_embedding
  + object_box_embedding
  + object_valid_embedding
  + ROI_pool(video_patch_tokens inside object_box)
```

The ROI component is pooled from the same spatio-temporal patch tokens used by
the PO-GUISE+ video transformer. This is the object-region visual evidence; it
is not an action-logit rule.

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
object-region memory -> relation distribution -> object context -> actor token -> action head
```

## Important Weights

`actor_object_relation_loss_weight`

Scales relation CE. Too low and the model may not select the interacted object.
Too high and the auxiliary relation task can dominate action learning.

`actor_object_relation_null_loss_weight`

Weights the NULL component in relation CE. Relation loss is component-balanced:
exact object selection is averaged separately from missing/objectless NULL
routing, then the two components are combined with this weight. This protects
objectless actions without letting numerous easy NULL samples drown out exact
object selection.

`actor_object_relation_null_logit_init`

Initial NULL bias. A higher value makes early training conservative about using
objects.

`actor_object_relation_normalize_pointers`,
`actor_object_relation_logit_scale_init`, and
`actor_object_relation_learned_logit_scale`

Use the HOTR/QPIC-style pointer idea for relation selection. The actor and
object relation projections are L2-normalized before their dot product, and a
positive learned scale controls how sharp the pointer logits are:

```text
relation_score(actor, object) =
    scale * cosine(actor_relation_query, object_relation_key)
```

This makes object selection a learned pointer problem instead of depending on
raw unnormalized feature magnitudes. The current launcher uses
`logit_scale_init=6.0` for enough cosine-logit dynamic range without replacing
the learned relation with a hard prior; `learned_logit_scale=1` lets training
soften or sharpen it.

`actor_object_relation_valid_logit_bonus`

Initial existence evidence for threshold-passing detector proposals. This is not
an action rule: it only competes with `NULL` inside the relation distribution.
The current ROI-object launcher sets this to `0.0` and disables the learned
valid bonus because object slots now carry true visual region features. Detector
thresholding still controls `object_valid`; the model learns whether that valid
visual object is the interacted object through relation CE.

With one valid object and no learned/bias evidence, the initial object mass is
approximately:

```text
sigmoid(valid_logit_bonus - null_logit_init)
```

The current launcher uses `null_logit_init=1.5` and
`valid_logit_bonus=0.0`, so a lone valid object starts around 0.18 non-NULL
mass before geometry, heatmap, visual ROI, and learned actor/object
compatibility terms. That gives the object path gradient early without
hard-coding that every valid object should beat `NULL`.

If this bonus is intentionally re-enabled for a future experiment,
`val_relation_valid_object_logit_bonus` reports where training moved it. It
should not be needed for the current ROI visual object path.

`val_relation_logit_scale` reports where training moved the normalized pointer
scale. If it collapses, actor/object compatibility is not being used. If it
explodes while object-present gains stay bad, the selector is becoming
overconfident without helping action classification.

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
The selected object context is the full selected object identity; the learned
fusion residual is multiplied by `object_mass`, so `NULL` and detector-missing
windows preserve the actor-only fallback.

`actor_relation_action_fusion_init_scale`

Initializes the final fusion residual with a small nonzero weight. A fully zero
final layer can make the action head start actor-only and delay object-action
gradients into the selected-object path. A small value keeps the pretrained actor
path dominant while proving, via `val_actor_object_fusion_delta_norm`, that the
learned object path is active.

`actor_object_present_margin_loss_weight` and
`actor_object_present_margin`

Train object-present usefulness without a hard-coded object/action table. For an
exact objectful example, the model runs both the normal object-present pass and a
compatible-object-hidden pass. It computes the true-vs-hardest-wrong action
margin in both passes:

```text
action_margin = true_action_logit - max(other_action_logits)
margin_gain = action_margin_present - stopgrad(action_margin_object_hidden)
loss = max(0, margin - margin_gain)
```

This pushes detected object evidence to improve the actual decision margin, not
just raise the true logit while also raising a confuser. The object-hidden
fallback remains protected by its own action CE.

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
first checks are `val_deploy_object_present_action_margin_gain`,
`val_deploy_object_present_*_gain`, and
`val_actor_object_fusion_delta_norm`. Only after those agree with the saved/live
probe should the next suspect be data/domain coverage or detector/object proposal
quality.
