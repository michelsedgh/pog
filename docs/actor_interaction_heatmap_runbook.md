# Actor Object PO-GUISE+ Training Contract

This is the intended one-way architecture for Toyota actor-slot object training.
Older experiments that used residual object decoders, shuffled object ablations,
object-selection heads, selected-object dropout, selected-object class dropout,
or checkpointing only on raw counterfactual logit drop are deprecated.

## What PO-GUISE+ Did

PO-GUISE+ trains a video transformer with:

- action cross entropy
- pose and interacting-object motion heatmaps
- `log(MSE)` heatmap scaling
- optional Nash-MTL balancing between action and heatmap objectives
- ViTPose and YOLO pseudo-labels only during training
- no detector input at inference

The object heatmap is not an object-class rule. It teaches the transformer to
keep and attend to visual tokens around the driver/object interaction.

## Our Actor-Slot Extension

Our runtime goal is different from paper-pure PO-GUISE+: we have multiple actor
slots and an RF-DETR detector available at runtime. The intended model input is:

```text
video clip
+ actor boxes / actor valid mask
+ RF-DETR object boxes / object classes / confidences / valid mask
```

The intended model path is:

```text
video tokens + actor tokens + object tokens + heatmap tokens
-> PO-GUISE+ transformer with semantic token selection
-> optional in-transformer actor-object relation updates
-> actor-token action head
-> prompt-token grounding / relation diagnostics and auxiliary losses
-> final logits = actor logits
```

The model must learn interaction, not object proximity. A laptop token alone
must not force `Uselaptop`; it is only useful when the actor visual evidence and
teacher target say that actor is interacting with the laptop.

Object prompt tokens are context, not a second classifier. They can change the
actor representation through transformer attention, but they do not directly
add class-specific action logits. True body/motion classes such as `Walk`,
`Getup`, `Sitdown`, and `Laydown` stay protected by the same actor-token action
head used for object-involving actions.

## Kept

- Actor prompt slots.
- One generic interacted-object heatmap per actor slot.
- RF-DETR object prompt tokens inside the transformer.
- Object-guided token selection with category-normalized semantic weights.
- In-transformer actor-object relation attention with a strong NULL prior.
- Plain actor-token action head.
- Prompt-token grounding loss for exact compatible teacher objects.
- Relation loss to the teacher object for exact objectful samples and to NULL
  for objectless or missing-compatible detected-object cases.
- Teacher-object removal as a validation-only supporting signal.
- PO-GUISE+ style `action CE + log(heatmap MSE)`.

## Removed

- Object-selection head.
- Selected-object-compatible object-logit residuals.
- Selected-object token dropout.
- Selected-object class dropout.
- Learnable scalar object-fusion gate.
- Pre-action actor/object token fusion.
- Configurable object-action fusion on/off and hidden-size knobs.
- Target/background weighted interacted-object heatmap loss.
- Training/checkpoint logic that treats raw teacher-object logit drop as the
  main proof of object-action learning.

Compatible teacher objects should stay visible during object-positive training
samples. Detector robustness should come from prompt-token context, objectless
hard negatives, heatmap grounding, and real detector misses, not from a separate
object-action residual classifier.

Checkpoints trained before the prompt-token cleanup are useful for diagnosis,
but they are not the final production architecture. Current training rejects old
object-specialist checkpoints instead of adapting them into the prompt-token
path.

## Teacher Target

The actor-object teacher target is built as:

```text
Toyota action label
-> expected object class from datasets/object_vocab.py
-> RF-DETR tracks of that class
-> best actor-associated track
-> teacher object index + actor-specific interacted-object heatmap
```

This is action-conditioned and actor-associated. It is not a proximity-only
rule. Object class semantics live in the RF-DETR object prompt tokens and
teacher object index, not in separate object-class heatmap channels.

The prompt-grounding teacher is about usable object-token binding:

```text
object-positive action + trusted matching object token -> teacher object token
object-positive action + no usable matching object token -> no prompt-grounding target
objectless action -> no prompt-grounding target
```

Objectless actions are supervised by action CE and objectless hard-negative
metrics, not by a NONE object-selection class. Object-involving actions without a
trusted object teacher, such as `Drink.Fromcan`, `Usetablet`, and coffee/tea
preparation variants, are left unlabeled for prompt grounding instead of being
treated as objectless.

Relation supervision is stricter because it has an explicit NULL slot:

```text
exact objectful + trusted matching object token -> teacher object token
objectful + no compatible detected object -> NULL
objectless action -> NULL
```

The heatmap teacher stays stricter: it is present only for a trusted matching
object track. Detector misses never create fake zero object heatmaps.

## Losses

The intended training objective is:

```text
action CE
+ object prompt grounding CE
+ actor-object relation CE over NULL plus object prompts
+ log(pose/object heatmap MSE)
```

With Nash-MTL enabled, the actor action objective is one task and the
pose/interacted-object heatmap objective is the second task. Runtime detections
enter the transformer as object prompt tokens. The action head remains one plain
linear head over actor tokens; there is no object-selection action head and no
detected-object residual over action logits. Object prompt grounding teaches
actor tokens to attend to compatible teacher objects when they exist, while
relation CE teaches the actor token when to use a detected object and when to
use NULL. The interacted-object heatmap teaches detector-free visual grounding.
There is intentionally no supervised object-action margin loss: object presence
alone must not force an action class. Interacted-object heatmaps use the
PO-GUISE+ normalized heatmap objective, optionally with positive-balanced and
center losses for sparse object targets.

Objectless hard-negative sampling is enabled for scene-object runs. For true
body/motion labels, the dataset tries to keep at least one sampled frame where a
mapped object is visible inside the same clip window. The action target remains
the objectless class; this teaches `object visible != object action` without
inventing a fake object-positive label.

## Actor Slots

Use real multi-actor clips when they exist. Synthetic side-by-side Toyota actor
collages were removed from the clean training path because they changed the data
distribution while debugging object-action semantics. Confuser evaluation should
come from real validation clips and object-prompt counterfactual diagnostics.

## Recommended Checkpoint Signals

Do not select checkpoints only by global F1 or only by raw counterfactual logit
drop. Use all of these:

- `val_deploy_score`
- `val_f1`
- `val_acc_macro`
- `val_group_object_mapped_acc`
- `val_group_objectless_acc`
- target action accuracy for `Uselaptop`, `Readbook`, `Usetelephone`, and drink classes
- `val_object_prompt_grounding_acc`
- `val_object_prompt_grounding_true_prob`
- `val_object_prompt_exact_correct_object_rate`
- `val_object_prompt_exact_correct_object_prob`
- `val_objectless_with_object_visible_acc`
- `val_objectless_with_object_visible_object_action_pred_rate`
- `val_object_counterfactual_teacher_logit_drop`
- live/saved probe sweeps for laptop/book/phone cases

`val_deploy_score` is the preferred checkpoint monitor for scene-object actor
runs. It is validation-only and does not affect training gradients. It rewards
macro F1, macro accuracy, object-mapped accuracy, objectless accuracy,
objectless-with-object-visible accuracy, and key action accuracy for
`Uselaptop`, `Readbook`, `Walk`, `Getup`, `Sitdown`, and `Laydown`. It penalizes
objectless-with-object-visible predictions that become object actions and any
key action below a 0.60 accuracy floor.

`val_object_counterfactual_selected_logit_drop` is validation-only supporting
evidence. It must not be used as a trainable loss or as the only checkpoint
target. The epoch 55 failure showed that object removal can look reasonable
while the laptop/book boundary is still wrong.

Validation logs every Toyota action as `val_action_<action>_acc` and
`val_action_<action>_count`. The summary compact view shows key actions and
groups; `--verbose` prints all logged per-class diagnostics. `val_f1` remains
macro F1 over all Toyota classes, not only the object-confusable subset.

## Summary Command

```bash
python3 summarize_interaction_metrics.py --pattern 'actor_object_relation_*'
```

For full heatmap/object diagnostics:

```bash
python3 summarize_interaction_metrics.py --pattern 'actor_object_relation_*' --verbose
```

## Live Probe Command

Use saved actor/object tensors to compare checkpoints before trusting live
camera impressions:

```bash
python3 object_actor_live/analyze_live_checkpoint_sweep.py \
  --device cuda \
  --input object_actor_live/latest_epoch001_actor_input.pt \
  --glob 'epoch=*.ckpt' \
  --out live_epoch_sweep.csv
```

The proof is not that the model selected `laptop`. The proof is that selecting
and keeping the laptop makes `Uselaptop` beat `Readbook` and `Usetelephone` on
the same actor clip.

## Inference Contract

For the current live dashboard:

- RF-DETR TensorRT detector only.
- PyTorch actor checkpoint.
- Detector runs every frame.
- Action runs every frame once the rolling buffer is full.
- Training-span matched clip sampling: 16 frames over the latest 128-frame span.
- Object tokens are built from the same sampled temporal window.
- No hidden alternate actor inference path.
