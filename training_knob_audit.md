# Detector-Guided PO-GUISE+ Training Knob Audit

Date: 2026-06-16

Reference clone: `reference_repos/RicardoP0_poguise_original` at commit `ae476ea`.

Current repo head audited: `476547960c006b9af54335683d1813080d7d437c`.

## 2026-06-17 Update: What The Laptop-On-Lap Probe Proved

The latest saved-video probe changes the conclusion. The object path is not
failing to see the laptop.

On `live_test_clips/laptop_on_lap_20260605_135113.mp4`, the object-enabled
checkpoint often produced:

```text
relation block 9:
  top object = laptop
  laptop attention ~= 0.98-0.99
  NULL ~= 0.00-0.03

engagement/action:
  engagement top = book
  Readbook top-1
```

Many of those failing windows had no book object detected at all. This rules
out the simple explanation that the detector or relation module is selecting
the wrong object.

The current failure is:

```text
selected object identity = laptop
actor/object semantic state = book
final action = Readbook
```

That means the architecture still lacks a strong **object-class-to-action-state
binding** inside the main representation. More object attention cannot fix this
by itself; the model already attends to laptop.

### Low-Motion Augmentation Was Removed

The old low-motion sampler was a training-only pressure term. It was not part
of PO-GUISE or PO-GUISE+.

It could replace the normal Toyota/live-style temporal span with a tight
low-motion span around one center frame:

```text
standard/live-style sample:
  16 frames across roughly the full labeled/window span

low-motion augmentation:
  16 frames across a small local low-motion span
```

That was a distribution-changing augmentation, so it has been removed from
`datasets/toyotasm.py` and from the training launchers. Only reintroduce a
data-side augmentation later if it preserves the same temporal geometry used by
live inference.

### Detector-Success Frame Forcing Was Removed

The old detector-success sampling path changed which frames were sampled
for objectful clips by replacing normally sampled frames with frames where the
expected object detector succeeded. The old objectless hard-negative sampling
path did the inverse for objectless clips by forcing object-visible frames. The
old object repair path could also insert an expected object from nearby frames
when the sampled frame missed it.

That is not the PO-GUISE+ setup. PO-GUISE+ samples the clip normally and uses
pose/object pseudo-labels to supervise heatmap tokens. The detector should
label the sampled evidence; it should not choose an easier or more object-heavy
temporal view.

This matters for the live failure:

```text
Toyota Uselaptop: detector almost always sees laptop
live Uselaptop: detector can miss laptop or see it intermittently
```

If training forces objectful windows toward detector-success frames, the model
learns the clean laptop-present case but does not see enough natural
missing/intermittent evidence. The cleaned dataset now keeps the original frame
sampling geometry. Relation/grounding/binding losses still use real detected
objects when they are present; missing detections naturally train NULL relation
behavior and force the actor/video path to carry the fallback evidence.

### Corrected Missing Mechanism

The engagement head added so far is post-transformer auxiliary supervision:

```text
actor token + object_context -> engagement logits
actor token -> actor_head -> action logits
```

That helps diagnose the issue, but the probe shows it can still predict
`book` even when the relation context is a laptop. The missing mechanism is
not just an auxiliary readout. It is an in-transformer binding representation:

```text
actor token
+ selected object token/class
+ relation confidence/NULL state
    -> actor-object binding/state token or binding update
    -> later transformer blocks update actor token
    -> actor_head(actor token)
```

The binding target should directly supervise object-state confusers on the real
input:

```text
Uselaptop + selected laptop:
  laptop_state > book_state / phone_state / tv_state + margin
  Uselaptop logit > Readbook / Usetelephone / WatchTV logits + margin

Readbook + selected book:
  book_state > laptop_state / phone_state / tv_state + margin
  Readbook logit > Uselaptop / Usetelephone / WatchTV logits + margin
```

The key difference from the current confuser loss is that this acts on the
**real selected object input**, not only on a fake same-box wrong-class
counterfactual. The fake-object loss can remain a diagnostic or weak regularizer,
but it is not sufficient by itself.

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

### 4. Motion Auxiliary Was Removed

The old motion auxiliary was a second action classifier on actor tokens. It was
not part of PO-GUISE/PO-GUISE+ and could reinforce the shortcut we dislike:

```text
motion pattern -> action
```

That path has been removed. The only action classifier is now the main actor
head; object-action semantics are shaped through relation, engagement, binding
state, heatmap, and normal action CE.

### 5. Loss Grouping Is Messy

Original PO-GUISE has two tasks:

```text
action CE
heatmap localization
```

Current clean training has action CE, actor-object relation CE, actor-object engagement CE, binding-state margin, class-aware missing-object view supervision, object prompt grounding, pose heatmap, and interaction heatmap. Removed paths include motion auxiliary, objectless suppression loss, synthetic actor/object generation, detector-success frame forcing, and final-action object corrections.

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

## The Missing Mechanism: Actor-Object Binding State

The corrected implementation puts this binding state inside the
actor-object relation update:

```text
actor token + selected object context + actor*object context
    -> binding feature
    -> actor-token delta inside the transformer
    -> coarse binding-state logits for auxiliary supervision
```

It predicts coarse object-interaction state from the same representation that
updates the actor token:

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

That signal is necessary, but not sufficient. The latest probe showed cases
where the relation selected laptop and the auxiliary itself still predicted
`book`. So the semantic state should not only be decoded after the transformer;
it should be injected as a binding representation before the final actor token
is classified.

The corrected mechanism teaches the actor representation:

```text
this looks like laptop engagement
```

from both actor appearance and selected object identity.

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
remove the extra motion action head
```

### Keep Lightly

```text
--object_prompt_grounding_loss_weight 0.20-0.30
--actor_object_relation_loss_weight 0.50
--actor_object_relation_null_loss_weight 0.50
```

Objectless safeguards should come from relation NULL supervision and normal
action CE, not a separate action-probability suppression rule. The suppression
loss was removed; keep the objectless-with-visible-object metrics as diagnostics.

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

### Removed From The Clean Architecture Diagnostic

The low-motion sampler, detector-success frame forcing, objectless frame forcing,
nearby-frame object repair, object-token box jitter, and object-token confidence
noise were removed. The clean diagnostic now tests object-action binding without
changing the temporal training geometry or corrupting runtime object proposals.

Whole-clip hard-negative oversampling is also disabled for the main
object-binding diagnostic. It is not a frame corruption bug, but it adds a
distribution pressure in the opposite direction: visible objects should often
not affect the action. Bring it back only after object-action binding works and
the remaining problem is objectless false positives.

## Recommended 10-Epoch Diagnostic Setup

Start from a clean VideoMAEv2/PO-GUISE+ backbone, not from any old residual-object checkpoint. The original PO-GUISE work uses VideoMAEv2 pretrained weights; do not interpret "scratch" as random uninitialized video backbone.

Use:

```bash
--actor_object_prompt_tokens 1
--actor_object_relation_in_transformer 1
--actor_object_relation_blocks 6,9
--actor_object_relation_loss_weight 0.75
--actor_object_relation_null_loss_weight 0.50
--actor_object_engagement_loss_weight 0.50
--actor_object_binding_state_loss_weight 0.75
--actor_object_binding_margin 0.50
--actor_object_missing_view_action_loss_weight 0.25
--actor_object_missing_view_engagement_loss_weight 0.25
--actor_object_missing_view_relation_null_loss_weight 0.25
--actor_object_missing_view_target_rate 0.25
--object_prompt_grounding_loss_weight 0.35
--class_balanced_sampler 1
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

## Next Clean Addition: Class-Aware Missing-Object View

Do not add global object dropout. Toyota already has many natural missing-object
examples for phone/bottle/pour actions:

```text
Uselaptop      missing ~= 0.03
Readbook       missing ~= 0.16
WatchTV        missing ~= 0.22
Drink.Fromcup  missing ~= 0.22
Usetelephone   missing ~= 0.65
Drink bottle   missing ~= 0.56
Pour bottle    missing ~= 0.68
```

Global dropout would over-mask classes that already have plenty of missing
object supervision. Use a target missing exposure and only synthesize the
shortfall:

```text
target_missing_rate = 0.25
mask_prob(action) =
  max(0, target_missing_rate - natural_missing_rate(action))
  / max(1e-6, 1 - natural_missing_rate(action))
```

Approximate probabilities from the current audit:

```text
Uselaptop      mask_prob ~= 0.23
Readbook       mask_prob ~= 0.11
WatchTV        mask_prob ~= 0.04
Drink.Fromcup  mask_prob ~= 0.04
others above 0.25 missing: 0.00
```

Implementation should be a second training view, not a dataset mutation:

```text
normal view:
  frames unchanged
  object prompts unchanged
  relation target = exact object if present
  binding/action/engagement losses active

missing-object view:
  same frames
  same actor boxes
  remove only compatible object prompt tokens for selected actions
  keep unrelated/distractor object tokens
  relation target = NULL
  object prompt grounding ignored for removed object
  action CE still uses the true action
  engagement state CE still uses the true object state
  interaction heatmap supervision may stay valid from the original teacher
```

The missing view teaches the fallback we want:

```text
if laptop prompt is absent but the video/actor state still looks like laptop use,
actor_head should still prefer Uselaptop over Readbook/Usetelephone.
```

Keep the missing-view action/engagement weight mild:

```text
missing_view_action_ce_weight       0.20-0.30
missing_view_engagement_weight      0.20-0.30
missing_view_relation_null_weight   0.20-0.30
```

Do not apply the binding margin in missing view, because binding margin is about
the real selected object class. When the compatible object prompt is deliberately
absent, the correct relation behavior is NULL plus correct action/engagement
from actor/video evidence.

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
object_prompt_grounding_acc
interaction_heatmap_soft_iou
interaction_heatmap_center_l2

wrong_class_laptop_state_delta
```

The key new proof is:

```text
low-motion laptop windows:
  relation attends laptop or heatmap localizes laptop region
  binding/engagement predicts laptop state
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

If engagement accuracy improves but action still does not, then actor_head is
not using the state feature. The next clean step is not a residual action
branch; it is to inject the binding representation back into the actor token
inside the transformer or immediately before actor_head as a representation MLP,
still without adding separate action logits.

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
