# Actor-Object Loss Design

This repo now uses one actor-object assignment objective.

For each actor slot, the model predicts a relation distribution over:

```text
0          = NULL, no usable interacted object
1..K       = detected object memory slots
```

The relation target is:

```text
target = 1 + teacher_object_slot
```

when the dataset action is known to require an object and a compatible detected
object exists. Otherwise the target is `0` for NULL.

The loss is standard cross entropy:

```text
L_relation = CE(relation_logits[B,A,1+K], relation_target[B,A])
```

This teaches exactly one thing: for this actor, which object slot, if any, is
the interacted object.

## Architecture Contract

The clean object design is:

```text
PO-GUISE+ video/pose/interacted-object heatmap backbone
        + learned HOI-style actor-object binding
        + missing-detector robustness training
```

It is not a manual object-action rule. Runtime detector objects are candidate
memory. Each candidate contains class/box/valid metadata and a visual descriptor
pooled from the video patch tokens inside the detector box. The detector
threshold decides whether a proposal exists; the relation module learns which
valid proposal, if any, is the interacted object for each actor slot. The action
head then learns how the selected object-region context, actor
motion/pose/appearance, and interaction heatmap evidence combine into the final
action.

The current implementation follows the HOI-query idea in actor-slot form:

- the actor slot is the fixed human query
- valid detector objects are object candidates
- each object candidate contains an ORViT-style visual region descriptor pooled
  from the proposal box
- normalized actor/object relation projections produce pointer logits
- relation CE supervises the pointer over `NULL + object slots`
- the final actor action head consumes a composed interaction representation
  from actor token, selected object token, actor/object product, and relation
  mass
- relation mass gates the object-conditioned residual, so missing/NULL cases
  fall back to actor-only features

This is why there is no fixed `laptop -> Uselaptop` logit boost. The desired
behavior is still that a selected laptop strongly supports `Uselaptop`, but that
support must be learned through the selected object context and action CE. The
object-present margin loss is the guardrail that prevents a checkpoint from
looking good on relation metrics while valid object evidence hurts the true
action margin.

## Active Training Objectives

The active objectives are:

- action CE on the actor action head
- actor presence BCE when actor prompts are enabled
- side-by-side actor-pair training when `actor_pair_train_weight > 0`
- relation CE over NULL plus object slots
- detector-dropout action/relation auxiliary training for object-missing fallback
- object-present action-margin gain training for exact objectful examples
- PO-GUISE+ pose and interaction heatmap losses

The final action still comes from one actor action head. Runtime detections are
encoded as relation-only object memory, not ordinary transformer prefix tokens.
They modify actor tokens through the actor-object relation update inside the
transformer and once more immediately before classification. The classifier input
is then refined by a small-initialized learned fusion of actor token, selected
object context, actor/object product, and object mass. There is no late logit
residual and no separate object-action classifier. Object proposal checkpoints
must use this relation-plus-fusion path; passive object prompt tokens without
action coupling are intentionally rejected.

## Code Evidence

The active relation path is:

- `blocks/poguise.py::ActorObjectRelationUpdate`
  computes relation logits, object attention, object context, and a gated actor
  token update.
- `blocks/poguise.py::VisionTransformer.forward`
  pools object-region visual descriptors from the video patch tokens, keeps
  runtime detections as relation-only object memory, applies relation updates at
  the configured transformer blocks, and applies a final pre-head relation
  update keyed by the model depth, for example `12` on the base model.
- `models/poguise.py::POGUISE.forward`
  builds the final actor action input from actor tokens and the last selected
  relation context, then exposes `last_actor_object_relation_aux` and the final
  actor action logits.
- `modules/heatmap_module.py::_actor_object_relation_loss`
  creates the supervised relation target over `NULL + object slots`.
- `poguise+_+objects.py`
  launches relation training with `--actor_object_relation_loss_weight 1.00`.

The removed side objectives are not active. `train.py` intentionally rejects
old checkpoints containing the removed side-head keys.

## Loss Decision Table

| Loss | Keep? | Reason |
| --- | --- | --- |
| Action CE | Yes | It is the actual product target: classify the actor action. |
| Relation CE | Yes | It is the only direct supervision for "which object, if any, this actor is interacting with." |
| Pose/interaction heatmap losses | Yes | This is the PO-GUISE/PO-GUISE+ auxiliary task family used for token selection and localization. |
| Engagement/state CE | No | Its labels were deterministic copies of action labels, not independent annotations. It could be satisfied by a side head while `actor_head` still predicted the wrong action. |
| Prompt-grounding CE | No | It duplicated relation CE by separately teaching an actor-to-object slot distribution. |
| Object counterfactual CE with fake labels | No | Swapping/removing object tokens does not create a new ground-truth action label. Training on fake labels risks teaching artifacts instead of the dataset task. |
| Object-present margin gain | Yes | It compares the real object-present pass with the compatible-object-hidden pass for the same ground-truth action and requires detected object evidence to improve the correct-vs-hardest-wrong action margin. |
| Inference logit override | No | It hides model failure and bypasses the actor action path. |

## Why One Relation Objective

The failure mode we are trying to fix is:

```text
relation path selects laptop
final action still says Readbook or WatchTV
```

Adding more side classifiers does not fix that by itself. The clean design is
to make the selected object context part of the actor token that the real action
head sees, then train the real action head with action CE. Relation CE supplies
the actor-object assignment; action CE supplies the action label.

This repo now implements the architectural fix directly: runtime object
detections cannot leak through generic self-attention, the selected object
context updates actor tokens inside the transformer, and the final action input
explicitly receives the last selected object context. If a future run still
selects the right object but predicts the wrong action, the next suspect is
data/domain coverage, detector proposals, insufficient training, or relation
capacity, not another duplicate classifier for the same supervision.

Actor-slot learning is trained by composing two real labeled clips side by side
inside the training step. The composed sample has two valid actor slots, remapped
actor boxes, remapped object boxes, remapped one-based relation teacher indices,
and composed pose/interaction heatmaps. The labels remain the original dataset
action labels for each actor.

With the default object launcher (`batch_size=32`,
`actor_pair_train_weight=0.50`), a full one-person batch can produce 16
side-by-side composites in the same training step. That means roughly one third
of forward video clips are side-by-side composites, half of the actor labels seen
by the model come from paired clips, and the effective loss pressure from pair
training is about one third after applying the 0.50 pair-loss weight.

## Learned Relation Strength And Final Fusion

The current relation update already learns per-sample object influence:

```text
actor_update = object_mass * learned_gate * learned_layer_scale * delta
```

`object_mass` comes from the NULL/object relation distribution, and
`learned_gate` is predicted from actor/object context. `learned_layer_scale` is
a trainable per-relation-block strength initialized small and bounded by
`max_scale`, which is now a safety cap rather than the primary strength setting.

The final actor classifier consumes the selected object context explicitly:

```text
final_actor = fuse(actor_token, selected_object_context,
                   actor_token * selected_object_context, object_mass)
action_logits = actor_head(final_actor)
```

This keeps one deploy action head and one action CE target. It removes the weak
assumption that a residual relation update must store all object identity inside
the actor token before classification, while still avoiding inference-time logit
rules or duplicate object-state classifiers.

## Research Basis

The local paper notes and public paper pages point to the same architecture
shape:

- PO-GUISE is a multi-task video transformer for action classification and pose
  heatmaps. Its published description says it integrates ADL action semantics
  with human motion and uses motion-guided token pruning, not extra object-state
  classifiers. Public reference:
  https://portalcientifico.urjc.es/en/ipublic/item/10383944
- PO-GUISE+ extends that idea by adding interacting-object localization
  heatmaps. Its abstract describes a model that predicts action, pose, and the
  interacting object, and leverages object interaction plus pose for token
  selection. Public reference:
  https://arxiv.org/abs/2407.13750
- HOI transformer work such as HOTR and QPIC frames the problem as learned
  human/object/interaction binding with transformer context. HOTR predicts
  human/object pointers and action from interaction queries; QPIC predicts
  object class, subject box, object box, and verb logits from the same query
  representation. These are learned heads, not hand-coded object-to-action
  rules. Public references:
  https://github.com/kakaobrain/HOTR and https://arxiv.org/abs/2103.05399
- Object-aware video transformer work such as ORViT injects detected object
  region representations into video transformer layers so object evidence can
  shape action features directly. Our implementation now uses the same essential
  object-region idea for runtime detector candidates: each relation-bound object
  memory slot includes a visual descriptor pooled from the proposal box, then
  actor-object relation binding selects among those visual object memories.
  Public reference:
  https://arxiv.org/abs/2110.06915
- Missing-modality action-recognition work trains the model under absent inputs
  instead of assuming it will generalize when a modality disappears. That is the
  reason for detector-dropout auxiliary training. Public reference:
  https://ojs.aaai.org/index.php/AAAI/article/view/25378
- The PO-GUISE+ loss section in `po-guise+.md` defines classification CE plus
  heatmap MSE/log-MSE. It does not define an engagement/state classifier or a
  counterfactual object-action loss.
- Nash-MTL is appropriate for balancing real task losses, not for justifying an
  unlimited number of redundant side losses. The Nash-MTL paper frames MTL as
  combining task gradients to avoid one task dominating due to scale. Public
  reference:
  https://proceedings.mlr.press/v162/navon22a/navon22a.pdf
- Auxiliary-task learning literature explicitly warns about negative transfer:
  extra auxiliary tasks can reduce target-task accuracy when their gradients or
  generalization behavior conflict with the main task. Public references:
  https://openreview.net/forum?id=vZHk1QlBQW and
  https://proceedings.mlr.press/v48/leeb16.html

The conclusion is not that auxiliary losses are always bad. The conclusion is
that every auxiliary loss must have a distinct target and a direct reason to
improve the action representation. Relation CE passes that test. The removed
engagement/state and fake counterfactual losses did not.

## Metrics To Watch

Primary:

- `val_f1`
- key per-action accuracies, especially `Uselaptop`, `Readbook`, `WatchTV`
- `val_relation_exact_teacher_acc`
- `val_relation_exact_teacher_prob`
- `val_relation_useful_mass_exact`
- `val_relation_null_rate_objectless`
- `val_relation_null_rate_missing_objectful`

Supporting:

- interaction heatmap response and center error
- token-selection retention for actor boxes and teacher object boxes
- objectless hard-negative object-action prediction rate
