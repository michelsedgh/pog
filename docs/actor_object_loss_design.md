# Actor-Object Loss Design

This repo now uses one actor-object assignment objective.

For each actor slot, the model predicts a relation distribution over:

```text
0          = NULL, no usable interacted object
1..K       = detected object prompt slots
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
- relation CE over NULL plus object slots
- PO-GUISE+ pose and interaction heatmap losses

The final action still comes from the actor action head. Runtime object prompts
modify actor tokens inside the transformer and once more immediately before that
head. There is no late object action residual and no separate object-action
classifier.

## Code Evidence

The active relation path is:

- `blocks/poguise.py::ActorObjectRelationUpdate`
  computes relation logits, object attention, object context, and a gated actor
  token update.
- `blocks/poguise.py::VisionTransformer.forward`
  applies relation updates at the configured transformer blocks and a final
  pre-head relation update keyed by the model depth, for example `12` on the
  base model.
- `models/poguise.py::POGUISE.forward`
  exposes only `last_actor_object_relation_aux` and the final actor action
  logits.
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

This repo now implements the architectural fix directly: the selected object
context is fused into the actor token that `actor_head` sees. If a future run
still selects the right object but predicts the wrong action, the next suspect is
data/domain coverage, detector proposals, or insufficient training, not another
duplicate classifier for the same supervision.

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
