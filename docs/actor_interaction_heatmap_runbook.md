# Actor Object PO-GUISE+ Training Contract

This is the intended one-way architecture for Toyota actor-slot object training.
Older experiments that used residual object decoders, shuffled object ablations,
selected-object dropout, selected-object class dropout, or checkpointing only on
raw counterfactual logit drop are deprecated.

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
video tokens + actor tokens + heatmap tokens + object tokens
-> transformer
-> object selection head per actor
-> actor token fused with selected-object context when scene object tokens are enabled
-> existing actor action head
```

The model must learn interaction, not object proximity. A laptop token alone
must not force `Uselaptop`; it is only useful when the actor visual evidence and
teacher target say that actor is interacting with the laptop.

## Kept

- Actor prompt slots.
- One generic interacted-object heatmap per actor slot.
- RF-DETR scene object tokens.
- Object selection head.
- Actor/object fusion before the existing actor action head. This is not a
  separate training option: `scene_object_tokens=1` implies fusion.
- PO-GUISE+ style `action CE + log(heatmap MSE)`.
- Selected-object removal as a validation-only supporting signal.

## Removed

- Selected-object token dropout.
- Selected-object class dropout.
- Learnable scalar object-fusion gate.
- Configurable object-action fusion on/off and hidden-size knobs.
- Target/background weighted interacted-object heatmap loss.
- Training/checkpoint logic that treats raw selected-object logit drop as the
  main proof of object-action learning.

The selected object should stay visible during object-positive training samples.
Detector robustness should come from box jitter, confidence noise, non-selected
distractors, and real detector misses, not from erasing the object that the
label says is being interacted with.

Checkpoints trained before this cleanup include a deprecated fusion-gate
parameter. They are useful for diagnosis, but they are not the final production
architecture. Resume them only with `--strict_load 0`; for live deployment,
prefer checkpoints trained after this cleanup.

## Teacher Target

The actor-object teacher target is built as:

```text
Toyota action label
-> expected object class from datasets/object_vocab.py
-> RF-DETR tracks of that class
-> best actor-associated track
-> selected object index + actor-specific interacted-object heatmap
```

This is action-conditioned and actor-associated. It is not a proximity-only
rule. Object class semantics live in the RF-DETR object tokens and selected
object index, not in separate object-class heatmap channels.

The selector teacher is about usable object-token binding:

```text
object-positive action + trusted matching object token -> selected token
object-positive action + no usable matching object token -> unlabeled selector
objectless action -> NONE=0
```

`NONE` is reserved for true body/motion actions: `Enter`, `Getup`,
`Laydown`, `Leave`, `Sitdown`, and `Walk`. Object-involving actions without a
trusted object teacher, such as `Drink.Fromcan`, `Usetablet`, and coffee/tea
preparation variants, are left unlabeled for object selection instead of being
treated as objectless.

The heatmap teacher stays stricter: it is present only for a trusted matching
object track. Detector misses never create fake zero object heatmaps.

## Losses

The intended training objective is:

```text
action CE
+ balanced object/NONE selection CE at fixed weight 0.5
+ log(pose/object heatmap MSE)
```

With Nash-MTL enabled, action CE and object selection are treated as the action
task, while pose/object heatmap localization is treated as the heatmap task.
Action CE owns the action boundary. Object selection binds the actor token to a
usable object-token context, and the interacted-object heatmap teaches visual
grounding. There is intentionally no supervised object-action margin loss:
object presence alone must not force an action class. There is also no
target/background weighted heatmap variant; interacted-object heatmaps use the
same PO-GUISE+ style Frobenius/log-MSE objective as pose heatmaps.

## Synthetic Actor Slots

Synthetic side-by-side Toyota samples are optional and off by default. When
enabled, use a modest two-actor probability and bias some partners toward
object-confusable actions:

```text
--toyota_synthetic_two_actor_prob 0.20
--toyota_synthetic_three_actor_prob 0.0
--toyota_synthetic_same_class_prob 0.35
--toyota_synthetic_confuser_prob 0.50
```

This teaches slot separation without making every sample artificial. Confuser
pairing targets laptop/book/phone, drink cup/bottle/glass, and cooking-object
boundaries from `ACTION_OBJECT_CONFUSERS`.

## Recommended Checkpoint Signals

Do not select checkpoints only by global F1 or only by raw counterfactual logit
drop. Use all of these:

- `val_f1`
- `val_acc_macro`
- `val_group_object_mapped_acc`
- `val_group_objectless_acc`
- target action accuracy for `Uselaptop`, `Readbook`, `Usetelephone`, and drink classes
- `val_object_selection_acc`
- `val_object_selection_none_acc`
- `val_object_selection_object_acc`
- `val_object_selection_true_prob`
- `val_object_counterfactual_selected_logit_drop`
- live/saved probe sweeps for laptop/book/phone cases

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
python3 summarize_interaction_metrics.py --pattern 'actor_object_fused_*'
```

For full heatmap/object diagnostics:

```bash
python3 summarize_interaction_metrics.py --pattern 'actor_object_fused_*' --verbose
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
