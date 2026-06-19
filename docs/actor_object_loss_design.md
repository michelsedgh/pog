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

## Active Training Objectives

The active objectives are:

- action CE on the actor action head
- actor presence BCE when actor prompts are enabled
- side-by-side actor-pair training when `actor_pair_train_weight > 0`
- relation CE over NULL plus object slots
- PO-GUISE+ pose and interaction heatmap losses

The final action still comes from one actor action head. Runtime detections are
encoded as relation-only object memory, not ordinary transformer prefix tokens.
They modify actor tokens through the actor-object relation update inside the
transformer and once more immediately before classification. The classifier input
is then refined by a zero-initialized learned fusion of actor token, selected
object context, actor/object product, and object mass. There is no late logit
residual and no separate object-action classifier.

## Code Evidence

The active relation path is:

- `blocks/poguise.py::ActorObjectRelationUpdate`
  computes relation logits, object attention, object context, and a gated actor
  token update.
- `blocks/poguise.py::VisionTransformer.forward`
  keeps runtime detections as relation-only object memory, applies relation
  updates at the configured transformer blocks, and applies a final pre-head
  relation update keyed by the model depth, for example `12` on the base model.
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
| Object counterfactual CE/margin | No | Swapping/removing object tokens does not create a ground-truth action label. Training on fake labels risks teaching artifacts instead of the dataset task. |
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
