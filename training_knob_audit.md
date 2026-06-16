# Detector-Guided PO-GUISE+ Training Knob Audit

Date: 2026-06-16

Reference clone: `reference_repos/RicardoP0_poguise_original` at commit `ae476ea`.

Current repo head audited: `476547960c006b9af54335683d1813080d7d437c`.

## Bottom Line

The detector is not the main failure. The object prompt path is learning to ground laptop detections, and the relation path often attends to the laptop in the live debug probe. The failure is that this object relation is not action-causal enough:

```text
laptop detected
relation sometimes selects laptop
actor_head still predicts Readbook in low-motion windows
```

That means the model has learned object localization/grounding, but not the semantic state:

```text
actor-laptop relation + seated/engaged posture = Uselaptop
```

Action CE alone is too weak and too shortcut-prone to teach that state. It can solve the dataset by using motion:

```text
typing/mouse motion -> Uselaptop
looking down at rectangular object -> Readbook
phone-like posture -> Usetelephone
```

The correct fix is not a late residual and not more object-logit correction. The correct fix is:

```text
PO-GUISE+ semantic-token path
+ runtime object prompts
+ in-transformer actor-object relation
+ actor-conditioned interaction heatmap
+ actor-object engagement-state auxiliary supervision
```

The engagement-state auxiliary is the missing supervision. It teaches the actor representation what the relation means, while final action still comes from `actor_head(actor_tokens)`.

## What The Original PO-GUISE Code Actually Does

The original PO-GUISE implementation is small and disciplined:

- `blocks/poguise.py` has class tokens, optional register tokens, heatmap tokens, and visual tokens.
- Token selection happens in the first three stages of ViT-base: after blocks 3, 6, and 9.
- Visual token pruning is guided by attention to semantic tokens: class and heatmap tokens.
- Discarded visual tokens are optionally merged so information is not completely dropped.
- `modules/heatmap_module.py` trains two tasks: action CE and heatmap MSE/log-MSE, optionally balanced with Nash-MTL.
- `models/poguise.py` keeps the final action path simple: processed class token/features -> classifier.

The original paper/code does not have:

- late object residuals
- base fusion
- object-action correction heads
- motion auxiliary action heads
- prompt drop losses
- objectless suppression losses
- synthetic object-class corruption
- many independent relation gates

The PO-GUISE+ paper extends the same idea with object heatmaps:

```text
semantic heatmap tokens teach where pose/object evidence is
semantic-token attention keeps useful visual tokens
main transformer representation predicts the action
```

That is the architectural standard we should preserve.

## What The Current Object Version Does Right

The hard refactor moved us into the correct architecture family:

```text
video tokens
+ actor tokens
+ heatmap tokens
+ runtime object prompt tokens
    -> transformer/token selection
    -> in-transformer actor-object relation update
    -> actor_head(actor_tokens)
```

The following parts should stay:

- Object prompt tokens inside the transformer prefix.
- Object prompt tokens included in token selection scoring.
- Category-normalized token-selection weights so 32 object tokens do not dominate by count.
- Actor-conditioned interaction heatmap, not one heatmap per object class.
- In-transformer relation updates at `[6, 9]`.
- Strong NULL option in relation attention.
- Relation loss with:
  - exact compatible objectful -> teacher object slot
  - objectless -> NULL
  - objectful missing compatible detection -> NULL
- Final action from `actor_head(actor_tokens)`.

Those choices are PO-GUISE+ consistent.

## What The Live Probe Proves

The epoch 009 live probe showed:

```text
with object tokens:
  Readbook 18 windows
  Uselaptop 6 windows
  avg Uselaptop 0.2918
  avg Readbook 0.6371

without object tokens:
  action distribution was nearly the same
```

The debug relation probe also showed windows where laptop prompt grounding or relation posterior was high, but the final action remained `Readbook`.

So the failure is not:

```text
RF-DETR cannot see laptop
```

It is:

```text
actor_head has not learned that actor-laptop relation means Uselaptop state
```

The training metrics agree:

- prompt grounding improved strongly
- relation teacher accuracy improved
- interaction heatmaps improved
- objectless NULL behavior improved
- but live object removal barely changed the laptop decision

That is a representation/action-causality gap.

## Main Code Issues Found

### 1. Relation Update Is Over-Gated

Current `ActorObjectRelationUpdate` has stacked suppression:

```text
strong NULL prior
+ confidence bias
+ relation bias
+ raw object posterior mass
+ useful_mass multiply
+ learned per-actor gate
+ global scale sigmoid(-2)
+ zero-initialized output projection
```

The specific math issue:

```python
object_context = torch.matmul(object_attn, v)
object_context = object_context * useful_mass[..., None]
```

`object_attn` is already scaled by useful mass because NULL participates in the softmax. Multiplying by `useful_mass` again squares the object contribution.

The checkpoint also showed the learned global relation scale stayed around `0.12`, so even correct laptop relation can be a tiny perturbation.

Fix:

```python
object_mass = object_attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)
object_attn_norm = object_attn / object_mass
object_context = torch.matmul(object_attn_norm, v) * object_mass
useful_mass = object_mass.squeeze(-1).clamp(0.0, 1.0)
```

Then remove one stabilizer:

- Keep NULL.
- Keep one per-actor gate or one fixed residual scale.
- Do not keep NULL + double mass + per-actor gate + low learned global scale.

Recommended clean version:

```text
NULL logit init: 3.5-4.0
relation blocks: 6,9
relation scale: fixed 0.5 or 1.0
per-actor gate: keep
double useful_mass gating: remove
zero-init final projection: keep
```

This gives no-op startup without permanently starving the object path.

### 2. Prompt Grounding Is Not Action Semantics

`object_prompt_grounding_loss` teaches:

```text
which prompt token corresponds to the teacher object
```

It does not teach:

```text
actor-laptop relation means Uselaptop
actor-book relation means Readbook
actor-phone relation means Usetelephone
```

Your metrics show grounding can be high while live action remains wrong. That is expected.

### 3. Relation Loss Is Mostly Correct But Incomplete

The relation target policy is mostly right:

```text
objectless -> NULL
exact compatible objectful -> object slot
objectful missing compatible -> NULL
```

But relation loss only teaches "which object." It does not teach "what state this actor-object relation represents."

### 4. Motion Auxiliary Is Working Against The Desired Concept

Current launcher uses:

```text
--motion_aux_loss_weight 0.35
```

This is an extra action classifier on actor tokens. It is not in PO-GUISE/PO-GUISE+. For this problem it likely reinforces the exact shortcut we dislike:

```text
motion pattern -> action
```

The main live failure is low-motion state classification. A motion auxiliary action head is the wrong pressure unless it is specifically designed to classify motion regime, not action.

Recommendation:

```text
motion_aux_loss_weight = 0.0
```

for the object-state diagnostic.

### 5. Loss Grouping Is Messy

Original PO-GUISE has two tasks:

```text
action CE
heatmap localization
```

Current training has action, relation, object prompt grounding, objectless suppression, objectless consistency, motion auxiliary, pose heatmap, interaction heatmap, positive heatmap balancing, center loss, and synthetic actor/object knobs.

The issue is not that auxiliaries are bad. The issue is that every head should answer one clear question.

Clean task grouping should be:

```text
Main semantic/action task:
  action CE
  actor-object relation CE
  actor-object engagement CE
  light objectless safeguards
  object prompt grounding

Heatmap task:
  pose heatmap MSE/log-MSE
  interaction heatmap MSE/log-MSE
```

Use Nash-MTL only between:

```text
[main semantic/action task, heatmap task]
```

Do not create separate competing Nash tasks for every auxiliary.

## The Missing Head: Actor-Object Engagement State

Add one training-only auxiliary:

```text
ActorObjectEngagementHead
```

It should predict coarse object-interaction state from the actor token and relation context:

```text
none/objectless
laptop
book
phone/tablet
tv/monitor
drink/cup/bottle
cooking/utensil/sink
eating/table/snack
other object interaction
```

Example targets:

```text
Uselaptop     -> laptop
Readbook      -> book
Usetelephone  -> phone/tablet
Usetablet     -> phone/tablet
WatchTV       -> tv/monitor
Drink         -> drink/cup/bottle
Pour_*        -> drink/cup/bottle
Cook_*        -> cooking/utensil/sink
objectless    -> none
```

Important distinction:

```text
relation target:
  NULL if compatible detector object is missing

engagement target:
  laptop for Uselaptop even if detector missed laptop
```

That is the exact missing signal. It teaches the actor representation:

```text
this looks like laptop engagement
```

even when the detector is missing or the hands are not actively typing.

This is not a late action residual. It is an auxiliary representation loss. Final inference still uses:

```python
action_logits = actor_head(actor_tokens)
```

Why this respects PO-GUISE+:

- PO-GUISE+ already uses auxiliary heatmap supervision to shape tokens for action recognition.
- The engagement head does the same for actor-object semantic state.
- It does not add logits to the action output.
- It does not make object detections a separate classifier.
- It teaches the transformer representation before the actor head.

## Heads And Roles

The clean model should have exactly these roles:

```text
actor_head:
  final action classification

heatmap head:
  where are pose landmarks and interacted object regions

relation attention:
  which detected object belongs to this actor, or NULL

engagement head:
  what object-interaction state is encoded in the actor/object relation

object prompt grounding:
  prompt-token identity alignment only
```

Everything else should be removed or disabled until the above works.

## Knob Verdict

### Keep As Architecture

```text
--actor_prompt 1
--num_actor_tokens 8
--actor_interaction_heatmaps 1
--actor_object_prompt_tokens 1
--actor_object_relation_in_transformer 1
--actor_object_relation_blocks 6,9
--num_scene_object_tokens 32
--num_object_classes 19
--keep_rate 0.6
--keep_rate_merge 0.3
--topk_type 1
--sim_metric 1
--grad_weights 1
```

### Keep But Freeze As Constants

These should not be active experiment knobs for the next run:

```text
token_selection_cls_weight      0.25
token_selection_actor_weight    0.25
token_selection_object_weight   0.10
token_selection_heatmap_weight  0.35
actor_object_prompt_box_prior_weight 0.05
actor_object_prompt_box_prior_expand 1.25
actor_object_relation_geometry_bias_weight 0.5
actor_object_relation_heatmap_bias_weight 1.0
actor_object_relation_null_logit_init 3.5-4.0
```

Reason: these are architecture constants. Tuning them now hides the actual question.

### Fix Before Next Real Run

```text
remove double useful_mass gating
return non-detached object_context from relation update
add actor-object engagement head/loss
put semantic auxiliary losses in the main task group
disable motion_aux action head
```

### Disable For The Object-State Diagnostic

```text
--motion_aux_loss_weight 0.0
--object_class_dropout_prob 0.0
--object_class_wrong_prob 0.0
--toyota_synthetic_two_actor_prob 0.0
--toyota_synthetic_three_actor_prob 0.0
--toyota_synthetic_confuser_prob 0.0
```

Reason: these are useful later, but they add noise while debugging object-state semantics.

### Keep Lightly

```text
--object_prompt_grounding_loss_weight 0.20-0.30
--actor_object_relation_loss_weight 0.50
--actor_object_relation_null_loss_weight 0.50
--objectless_prompt_consistency_loss_weight 0.10-0.20
--objectless_object_action_suppression_loss_weight 0.30-0.50
```

Objectless safeguards are useful, but they should not dominate the run. Their job is only:

```text
visible irrelevant objects should not hijack objectless actions
```

### Simplify Heatmap Loss

The paper uses CE plus log-scaled MSE. Current heatmap training also has positive-balanced and center losses.

For the clean diagnostic, use:

```text
poguiseplus_heatmap_loss_weight 1.0
poguiseplus_pose_heatmap_weight 0.25
poguiseplus_interaction_heatmap_weight 2.0-3.0
poguiseplus_normalized_heatmap_loss 1
poguiseplus_heatmap_mse_scale 1000
poguiseplus_interaction_heatmap_pos_loss_weight 0.0
poguiseplus_interaction_heatmap_center_loss_weight 0.0
```

Only re-enable positive/center losses if interaction heatmaps collapse or center L2 fails.

### Keep And Increase Slightly

```text
--objectful_low_motion_aug_prob 0.35
```

The dataset sampler already searches low landmark-motion windows for state-like object actions. Keep it and slightly increase it for the state run.

## Recommended 10-Epoch Diagnostic Setup

Start from a clean VideoMAEv2/PO-GUISE+ backbone, not from any old residual-object checkpoint. The original PO-GUISE work uses VideoMAEv2 pretrained weights; do not interpret "scratch" as random uninitialized video backbone.

Use:

```bash
--actor_object_prompt_tokens 1
--actor_object_relation_in_transformer 1
--actor_object_relation_blocks 6,9
--actor_object_relation_loss_weight 0.50
--actor_object_relation_null_loss_weight 0.50
--actor_object_engagement_loss_weight 0.30
--object_prompt_grounding_loss_weight 0.25
--objectless_prompt_consistency_loss_weight 0.15
--objectless_object_action_suppression_loss_weight 0.40
--motion_aux_loss_weight 0.0
--objectful_low_motion_aug_prob 0.35
--toyota_synthetic_two_actor_prob 0.0
--toyota_synthetic_three_actor_prob 0.0
--toyota_synthetic_confuser_prob 0.0
--object_class_dropout_prob 0.0
--object_class_wrong_prob 0.0
--max_epochs 10
--t_max_scheduler 10
```

Keep the PO-GUISE token settings:

```bash
--keep_rate 0.6
--keep_rate_merge 0.3
--topk_type 1
--sim_metric 1
--grad_weights 1
```

## Metrics That Decide If It Worked

Old residual metrics are obsolete.

Required metrics:

```text
relation_exact_teacher_acc
relation_exact_teacher_prob
relation_null_rate_objectless
relation_null_rate_missing_objectful
relation_useful_mass_exact
relation_useful_mass_objectless

engagement_acc
engagement_none_acc
engagement_laptop_acc
engagement_book_acc
engagement_phone_tablet_acc
engagement_tv_monitor_acc
engagement_low_motion_uselaptop_acc

object_prompt_grounding_acc
interaction_heatmap_soft_iou
interaction_heatmap_center_l2

object_ablation_low_motion_laptop_delta
wrong_class_laptop_state_delta
```

The key new proof is:

```text
low-motion laptop windows:
  relation attends laptop or heatmap localizes laptop region
  engagement predicts laptop state
  actor_head predicts Uselaptop more than Readbook
  removing/changing laptop prompt measurably lowers laptop-state confidence
```

## What We Should Not Do

Do not add back:

```text
late object residual
base fusion
selected-object action classifier
factorized object-action logits
large object logit correction
```

Those can improve validation by shortcuts, but they do not solve the representation problem.

Do not chase every knob. The current code has far more exposed knobs than the reference implementation. For this run, the scientific question should be one sentence:

```text
Does engagement-state supervision make in-transformer object relation action-causal?
```

## Expected Outcome

If this is the correct fix, the 10-epoch diagnostic should show:

```text
prompt grounding still good
relation NULL/object behavior still good
engagement laptop/book/phone/tv accuracy rises
low-motion Uselaptop improves
no-object live probe becomes meaningfully different from object-token probe
Readbook no longer absorbs static laptop-on-lap windows as often
```

If engagement accuracy improves but action still does not, then actor_head is not using the state feature. At that point, the next clean step is not a residual; it is to inject the engagement representation back into the actor token inside the transformer or immediately before actor_head as a representation MLP, still without adding separate action logits.

## Final Architecture Recommendation

Use this as the next clean architecture:

```text
video tokens
+ class token
+ actor slot tokens
+ object prompt tokens
+ heatmap tokens
      ↓
PO-GUISE+ token selection
      ↓
in-transformer actor-object relation update at blocks 6,9
      ↓
actor tokens encode:
  visual action evidence
  actor pose/motion
  interacted-object location
  object identity relation
  object-engagement state
      ↓
actor_head(actor_tokens)
      ↓
action logits
```

Auxiliary supervision:

```text
heatmap loss:
  where is pose/object interaction evidence?

relation loss:
  which detected object is useful, or NULL?

engagement loss:
  what object-interaction state does this actor/object relation mean?

objectless safeguards:
  visible irrelevant objects must not hijack objectless actions
```

This is the smallest correction that matches PO-GUISE+ and directly targets the live failure.
