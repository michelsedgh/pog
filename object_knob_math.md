# Detector-Guided PO-GUISE+ Knob Math

This note documents the active knobs in `poguise+_+objects.py` for the
detector-guided PO-GUISE+ actor-object architecture. The goal is to make the
6-epoch overdrive sweep interpretable instead of treating values as random.

The final action path is still:

```text
actor_tokens -> actor_head -> action logits
```

Object detections affect action only by shaping tokens inside the transformer
and by auxiliary supervision. That means the right question for each knob is:

```text
Does this knob change token retention, relation attention, actor-object state,
or the final actor_head logits?
```

## Scale Reference

For `product_v1`:

```text
action CE early scale:       log(26) ~= 3.26
engagement CE early scale:   log(9)  ~= 2.20
relation CE early scale:     log(33) ~= 3.50   # NULL + 32 object slots
prompt grounding CE scale:   log(32) ~= 3.47
```

A loss weight near `1.0` makes that auxiliary comparable to action CE. A weight
near `2.0` can dominate unless Nash-MTL downweights it.

With `--grad_weights 1`, training uses Nash-MTL on:

```text
task 0: loss_main_deploy
task 1: loss_grounding_aux
```

`loss_main_deploy` includes action CE, presence, relation loss, engagement loss,
object-action binding, missing-object view supervision, and prompt grounding. Heatmap
loss is placed in `loss_grounding_aux`. Nash-MTL can rebalance the two tasks, but
the individual weights still change the gradients inside their task.

## Sampling Cleanup

Object detections are now used as labels/prompts on the sampled clip, not as a
reason to replace sampled frames.

Removed dataset-side frame forcing includes the old object-success sampler,
objectless object-visible sampler, and nearby-frame object repair path.

Those knobs selected frames based on detector success. That is the wrong place
to solve detector-miss robustness, because it changes the temporal distribution
seen by the action model. `interaction_repair_radius_frames` did not replace
video frames, but it still repaired the object stream from neighboring frames,
which hides sampled-frame detector misses. PO-GUISE+ keeps the normal clip
sampling and uses detector pseudo-labels to supervise the semantic heatmap/object
path.

The clean training rule is:

```text
sample clip normally
if compatible object is present: train relation/binding to that object
if compatible object is missing: train relation to NULL, keep action/binding
                              supervision on the actor/video representation
```

If live testing still shows collapse when laptop detections are missing, the
next clean addition should be class-aware missing-object view training for the
under-missing classes only, not global object dropout.

## Token Selection

Code path: `KTPAttention.forward`.

For each visual token, pruning score is approximately:

```text
score =
  mean_heads(sum_semantic_queries attention(query, visual_token) * query_weight)
  + bbox_prior
```

The group weights are distributed across tokens:

```text
class per-token weight   = token_selection_cls_weight
actor per-token weight   = token_selection_actor_weight / num_actor_tokens
object per-token weight  = token_selection_object_weight / num_object_tokens
heatmap per-token weight = token_selection_heatmap_weight / num_heatmap_tokens
```

With the current shape:

```text
actor tokens  = 8
object tokens = 32
heatmap tokens = 64
```

So `token_selection_object_weight=0.60` means each object token gets
`0.60 / 32 = 0.01875`, not `0.60`.

Important: `actor_object_prompt_box_prior_weight` is different. It is an
additive score on visual tokens inside object boxes. Attention scores are often
small after averaging heads/semantic tokens, so a box prior of `0.05` can already
be meaningful. `0.50` is a real overdrive value: it should strongly force object
box patches to survive pruning if the keep budget allows it.

Pruning happens at blocks 3, 6, and 9. With:

```text
keep_rate = 0.6
keep_rate_merge = 0.3
```

the effective retained/merged visual count per pruning stage is:

```text
0.6 selected + 0.3 * 0.4 merged = 0.72
```

Across 3 pruning stages:

```text
0.72^3 ~= 0.37
```

So the final model keeps/merged about 37% of original visual tokens. Token
selection weights are deciding which regions survive into that 37%.

### Token Selection Ranges

`token_selection_object_weight`

```text
0.00-0.05  object prompts barely affect pruning
0.10-0.20  mild/moderate
0.30-0.45  strong
0.60+      overdrive; object tokens should visibly change object-box retention
```

`actor_object_prompt_box_prior_weight`

```text
0.00       no box forcing
0.03-0.08  mild but already meaningful
0.10-0.20  strong
0.30-0.60  overdrive; object box patches should be forced into top-k
>0.60      destructive; can preserve object boxes at expense of actor/context
```

`actor_object_prompt_box_prior_expand`

```text
1.0        exact detector box
1.25-1.50  reasonable context around object
1.75-2.25  strong contextual region
2.50+      destructive; object region becomes large and less specific
```

How to know if these are wrong:

```text
val_token_selection_laptop_box_keep_rate
val_token_selection_Uselaptop_teacher_object_keep_rate
val_token_selection_interaction_heatmap_keep_rate
```

If overdrive does not move these, the object-pruning path is not controlling
survival as expected. If it moves these but action causality does not change, the
bottleneck is not token selection.

## Relation Attention

Code path: `ActorObjectRelationUpdate.forward`.

For each actor/object pair:

```text
object_score =
  q(actor) dot k(object) / sqrt(relation_dim)
  + log(object_conf)
  + relation_bias
```

Then:

```text
scores = concat(NULL_logit, object_scores)
attention = softmax(scores)
```

`log(object_conf)` is negative unless confidence is 1:

```text
conf 0.25 -> -1.386
conf 0.50 -> -0.693
conf 0.90 -> -0.105
```

The object must beat NULL to be selected. NULL prior matters a lot.

If all object scores were 0 and K=32:

```text
NULL=4.0 -> p(NULL) = exp(4)/(exp(4)+32) ~= 0.63
NULL=3.0 -> p(NULL) ~= 0.39
NULL=1.5 -> p(NULL) ~= 0.12
NULL=1.0 -> p(NULL) ~= 0.08
```

In reality object confidence often subtracts 0.1-1.4 from object scores, so
`NULL=4.0` is strongly protective. `NULL=1.0-1.5` is intentionally permissive
and should make object attention/useful mass jump.

### Geometry Bias

Code path: `_actor_object_relation_geometry_bias`.

```text
dist = sqrt((dx / actor_width)^2 + (dy / actor_height)^2)
base = -dist + 0.5 * object_center_inside_expanded_actor_box
base is clamped to [-4, 2]
geometry_bias = base * geometry_bias_weight
```

Approximate effect:

```text
near/inside object:  +0.25 to +0.5 * weight
moderately far:      -1.0 * weight
very far:            down to -4.0 * weight
```

So `geometry_bias_weight=2.0` can add several logits of separation between near
and far objects. That is a strong relation prior.

### Heatmap Bias

Code path: `_actor_object_relation_heatmap_bias`.

The actor-conditioned relation heatmap gives a score in `[0, 1]` over object-box
pixels:

```text
heatmap_bias = heatmap_overlap_score * heatmap_bias_weight
```

So:

```text
weight 1.0 -> up to +1 logit
weight 2.0 -> up to +2 logits
weight 5.0 -> up to +5 logits
```

`5.0` is destructive overdrive: if the heatmap is wrong, relation attention will
be confidently wrong.

### Relation Update Scale

Actor token update:

```text
actor = actor + max_scale * gate * delta(actor, object_context)
```

`gate` is sigmoid, so it is in `[0, 1]`. The final delta layer is zero-initialized,
so the update starts as a no-op and grows during training.

Ranges:

```text
0.25-0.50  calibration strength
1.0        normal full residual update
2.0        strong
4.0-5.0    overdrive; can swamp actor representation if gate/delta grow
```

How to know if relation is too weak:

```text
val_relation_useful_mass_exact
val_relation_exact_teacher_acc
val_object_prompt_drop_Uselaptop_true_logit_drop
```

If useful mass/teacher accuracy is high but object-drop logit stays low, relation
is happening but actor_head is not using it.

## Auxiliary Losses

### Relation Loss

Code path: `_actor_object_relation_loss`.

Targets:

```text
exact compatible objectful: target = teacher object slot
objectful missing compatible object: target = NULL
objectless action: target = NULL
class mismatch: ignored unless covered by missing/objectless policy
```

Loss:

```text
mean_blocks weighted_CE(scores_over_NULL_plus_objects, target)
* actor_object_relation_loss_weight
```

`actor_object_relation_null_loss_weight` is a sample weight for NULL targets,
not a separate additive loss.

Ranges:

```text
relation_loss_weight:
  0.25-0.50  mild/moderate
  0.75-1.00  strong; comparable to action CE
  1.50-2.00  overdrive; relation can dominate main task

relation_null_loss_weight:
  0.25       weak NULL protection
  0.50       balanced protection
  1.00       strong NULL protection
  >1.00      may over-suppress useful objects
```

### Engagement Loss

Code path: `_actor_object_engagement_loss`.

The engagement head predicts 9 coarse states:

```text
none, laptop, book, phone_tablet, tv_monitor, drink, cooking, eating, other_object
```

Input:

```text
[actor_token, object_context, actor_token * object_context]
```

Targets come from action labels, not detector availability:

```text
Uselaptop -> laptop
Readbook -> book
Usetelephone/Usetablet -> phone_tablet
WatchTV -> tv_monitor
Walk/Sitdown/Getup/... -> none
```

Loss:

```text
CE(engagement_logits, state_target) * actor_object_engagement_loss_weight
```

Ranges:

```text
0.10-0.30  mild representation shaping
0.50-0.75  strong
1.00-1.50  overdrive; should visibly improve engagement metrics if path works
2.00+      destructive; can make state grouping compete with exact action CE
```

This is the most directly relevant loss for paused `Uselaptop`.

Useful metrics:

```text
val_actor_object_engagement_laptop_acc
val_actor_object_engagement_laptop_book_tv_acc
val_object_prompt_drop_Uselaptop_true_logit_drop
live paused-laptop window stability
```

### Prompt Grounding Loss

Code path: `_object_prompt_grounding_loss`.

The object grounding probe scores actor token to object prompt tokens:

```text
score(actor, object) = q(actor) dot k(object) / sqrt(dim) + log(conf)
```

Exact compatible samples train the probe to select the teacher object slot:

```text
CE(object_prompt_logits, teacher_slot) * object_prompt_grounding_loss_weight
```

This does not directly feed action logits. It teaches the object prompt geometry
and is useful as auxiliary representation pressure.

Ranges:

```text
0.10-0.25  mild
0.35-0.60  strong
1.00       overdrive; comparable to action CE scale after weighting
>1.00      likely too much unless grounding is the explicit task
```

### Removed: Objectless Suppression

The old objectless suppression path directly penalized the sum of object-mapped
action probabilities on objectless samples with visible objects. It was removed
because it was a final-action guardrail, not PO-GUISE+ style representation
learning.

Objectless behavior is now trained by normal action CE plus relation NULL
supervision, and monitored by objectless-with-visible-object diagnostics.

### Class-Aware Missing-Object View

Code path: `_actor_object_missing_view_loss`.

For object actions whose teacher object is almost always detected in Toyota, a
second training forward removes compatible object prompts for a sampled subset of
exact-teacher examples. The RGB clip, actor box, and action label stay unchanged.
The model is trained to keep the action and engagement state correct while the
relation head routes to NULL because the compatible prompt is absent.

Ranges:

```text
target missing rate 0.20-0.30  normal robustness
loss weights        0.20-0.30  mild/moderate second-view pressure
loss weights        0.50+      strong; can weaken object reliance
```

## Heatmap Losses

Interaction heatmap target is actor-conditioned:

```text
one channel per actor slot: where is this actor's interacted object?
```

Target policy:

```text
objectless: blank, valid
objectful exact compatible object: Gaussian at object center/trajectory, valid
objectful missing teacher object: invalid, not blank
```

Raw heatmap losses:

```text
heatmap_mse_loss = mean pixel MSE over valid heatmaps
heatmap_frobenius_loss = sum pixel squared error per sample
```

With `poguiseplus_normalized_heatmap_loss=1`:

```text
loss_pose_optimized = heatmap_mse_loss * poguiseplus_heatmap_mse_scale
loss_interaction_optimized = heatmap_mse_loss * poguiseplus_heatmap_mse_scale
loss_heatmap_raw =
    pose_loss * pose_weight
  + interaction_loss * interaction_weight
loss_heatmap_task = log1p(loss_heatmap_raw)
loss_grounding_aux += loss_heatmap_task * poguiseplus_heatmap_loss_weight
```

Because of `log1p`, changing interaction heatmap weight has diminishing returns
once the heatmap raw loss is large.

Ranges:

```text
poguiseplus_heatmap_loss_weight:
  0.5-1.0 normal under Nash-MTL
  >2.0 usually unnecessary because log1p + Nash already shape scale

poguiseplus_pose_heatmap_weight:
  0.10-0.50 reasonable when action is primary
  >1.0 pose may dominate auxiliary task

poguiseplus_interaction_heatmap_weight:
  1.0-3.0 reasonable
  5.0 strong
  >5.0 overdrive; may over-optimize localization without action causality

poguiseplus_heatmap_mse_scale:
  1000 is a scale-normalization constant, not a semantic weight.
  It compensates for averaged per-pixel MSE being tiny.
```

`interaction_heatmap_sigma` controls target width:

```text
1.0-1.5  sharp target
2.0-3.0  broader object-region target
>4.0     too broad; weak localization signal
```

For live transfer, a slightly broader target (`2.0-2.5`) can be reasonable
because laptop-on-lap relation is spatially approximate.

## Removed Sampling/Object-Token Hacks

The previous low-motion objectful sampler was removed. It changed the temporal
training geometry by sampling a tight low-motion span instead of the normal
live-style window. That made it a pressure hack rather than a PO-GUISE+
object-action architecture mechanism.

Nearby-frame object repair, pose-guided temporal start selection, hard-negative
oversampling, object-token box jitter, confidence noise, objectless consistency,
and synthetic actor collages were also removed. For this architecture, runtime
object prompts should be faithful detector proposals; robustness to detector
misses should come from natural missing-object windows and the explicit
class-aware missing-object view, not repaired/corrupted object prompts.

Paused laptop transfer should now be tested through actor-object binding
metrics and live/saved-video probes, not by changing the sampled temporal span.

### Object Detector Confidence Threshold

`object_conf_threshold` filters object proposals before prompt tokens/teachers.

Ranges:

```text
0.10-0.20  more recall, more noisy prompts
0.25       reasonable default
0.35-0.50  cleaner but can miss laptop/phone/book
```

For the current problem, missing laptop is not the main failure, so this should
not be the first knob to push.

## Optimizer And Training Scale

`lr=3e-5` applies to backbone/trunk. `lr_head=5e-4` applies to heads, including
actor/action and engagement. `lr_head_hm=5e-4` applies to heatmap head.

Ranges:

```text
backbone lr:
  1e-5 conservative
  3e-5 normal
  5e-5 strong
  1e-4 risky for ViT fine-tune

head lr:
  2e-4 conservative
  5e-4 normal/strong
  1e-3 overdrive
```

`gradient_clip_val=1.0` means destructive loss settings may not explode, but they
can still redirect gradients and produce wrong solutions.

## How Wild Are The Current Overdrive Tests?

The revised `transfer4` sweep is mathematically wild, not just 2x.

### `object_retention_overdrive6`

```text
object_weight = 0.60
box_prior = 0.50
box_expand = 2.25
```

This is extreme because the box prior is additive to attention-derived top-k
scores. It should visibly raise:

```text
val_token_selection_laptop_box_keep_rate
val_token_selection_Uselaptop_teacher_object_keep_rate
```

If it does not, token selection is not responding as expected.

### `binding_state_overdrive6`

```text
binding_state_loss = 1.25
```

This is extreme because engagement/binding CE starts around `log(9)=2.20`, so a
state loss above `1.0` can compete directly with action CE. It should visibly
raise:

```text
val_actor_object_binding_state_margin
val_actor_object_binding_state_pass_rate
val_actor_object_engagement_laptop_book_tv_acc
```

If binding margins rise but object-drop Uselaptop logit does not, the actor head
is still not using the binding feature.

### `relation_causality_overdrive6`

```text
NULL logit = 1.5
relation_scale = 4.0
geometry_bias = 2.0
heatmap_bias = 4.0
relation_loss = 1.5
```

This is extreme. `NULL=1.5` makes objects win by default unless the model learns
otherwise. `relation_scale=4.0` allows object context to strongly alter actor
tokens after the zero-initialized update learns.

It should visibly raise:

```text
val_relation_useful_mass_exact
val_object_prompt_drop_Uselaptop_true_logit_drop
```

If useful mass rises but Uselaptop logit drop does not, relation is not coupled
strongly enough to final action.

### `full_overdrive6`

```text
NULL logit = 1.0
relation_scale = 5.0
object_weight = 0.65
box_prior = 0.60
engagement_loss = 2.0
relation_loss = 2.0
grounding_loss = 1.0
```

This is intentionally destructive. It is not a final candidate. It asks:

```text
Can the current architecture be forced to make Uselaptop object-causal at all?
```

If `full_overdrive6` still has low:

```text
val_object_prompt_drop_Uselaptop_true_logit_drop
val_object_prompt_drop_Uselaptop_true_prob_drop
```

then the issue is not conservative values. It means object relation is not
becoming action-causal in the current architecture/training setup.

## Practical Decision Rules

After 6 epochs, use this table:

```text
Retention jumps, causality does not:
  object patches survive, but actor/action path ignores them.

Engagement jumps, causality does not:
  state auxiliary learns, but actor_head does not use it enough.

Relation useful mass jumps, causality does not:
  actor attends objects, but object context is not action-causal.

Uselaptop object-drop logit jumps, live paused laptop improves:
  knob family is relevant; back down from overdrive.

Objectless object-action pred rate explodes:
  object path works but is too strong; increase NULL/objectless safeguards or
  reduce object pressure.

Full overdrive fails:
  likely architecture/coupling issue, not just value tuning.
```

The most important metrics for the live failure are:

```text
val_object_prompt_drop_Uselaptop_true_logit_drop
val_object_prompt_drop_Uselaptop_true_prob_drop
val_actor_object_engagement_laptop_acc
val_token_selection_Uselaptop_teacher_object_keep_rate
val_relation_useful_mass_exact
val_objectless_with_object_visible_object_action_pred_rate
```
