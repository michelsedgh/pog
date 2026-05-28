from __future__ import annotations

import argparse
import math
from collections.abc import Mapping
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from datasets.object_vocab import (
    OBJECT_SUMMARY_FEATURE_DIM,
    object_summary_action_gate_prior,
    object_summary_action_object_matrix,
)


FEATURE_KEYS = (
    "features",
    "object_features",
    "object_summary",
    "object_summaries",
    "summaries",
    "x",
    "X",
)
LOGIT_KEYS = ("logits", "actor_logits", "preds", "predictions")
LABEL_KEYS = ("labels", "y", "targets", "actions")
ID_KEYS = ("file_ids", "clip_ids", "ids", "keys", "video_ids")


def _load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def _keys(obj):
    if isinstance(obj, Mapping):
        return sorted(str(k) for k in obj.keys())
    return [type(obj).__name__]


def _split_view(obj, split):
    if isinstance(obj, Mapping) and split in obj and isinstance(obj[split], Mapping):
        return obj[split]
    return obj


def _find_value(obj, split, candidates, label):
    view = _split_view(obj, split)
    if not isinstance(view, Mapping):
        raise ValueError(f"{label} cache must be a dict-like object, got {type(view)}")

    names = []
    for key in candidates:
        names.extend((f"{split}_{key}", f"{key}_{split}", key))
    for key in names:
        if key in view:
            return view[key]
    if view is not obj:
        for key in names:
            if isinstance(obj, Mapping) and key in obj:
                return obj[key]
    raise KeyError(
        f"Could not find {label} for split={split}. "
        f"Tried {names}. Top-level keys={_keys(obj)} split keys={_keys(view)}"
    )


def _to_tensor(value, dtype=None):
    if torch.is_tensor(value):
        tensor = value.detach().cpu()
    else:
        tensor = torch.as_tensor(value)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor


def _to_ids(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        value = value.detach().cpu().tolist()
    ids = []
    for item in list(value):
        if isinstance(item, bytes):
            item = item.decode("utf-8")
        if isinstance(item, (list, tuple)) and item:
            item = item[0]
        text = str(item)
        for sep in ("|", "\t"):
            if sep in text:
                text = text.split(sep, 1)[0]
        ids.append(text)
    return ids


def _maybe_ids(obj, split):
    try:
        return _to_ids(_find_value(obj, split, ID_KEYS, "ids"))
    except KeyError:
        return None


def _load_features(path, split):
    data = _load(path)
    features = _to_tensor(_find_value(data, split, FEATURE_KEYS, "features"), torch.float32)
    labels = _to_tensor(_find_value(data, split, LABEL_KEYS, "feature labels"), torch.long)
    ids = _maybe_ids(data, split)
    if labels.ndim != 1:
        labels = labels.reshape(-1)
    return features, labels, ids


def _load_logits(path, split):
    data = _load(path)
    logits = _to_tensor(_find_value(data, split, LOGIT_KEYS, "logits"), torch.float32)
    labels = _to_tensor(_find_value(data, split, LABEL_KEYS, "logit labels"), torch.long)
    ids = _maybe_ids(data, split)
    if logits.ndim == 3:
        logits = logits.reshape(-1, logits.shape[-1])
    if labels.ndim != 1:
        labels = labels.reshape(-1)
    return logits, labels, ids


def _align_features_to_logits(features, feature_labels, feature_ids, logits, labels, logit_ids):
    if feature_ids is not None and logit_ids is not None:
        feature_by_id = {}
        label_by_id = {}
        for idx, key in enumerate(feature_ids):
            feature_by_id.setdefault(key, features[idx])
            label_by_id.setdefault(key, int(feature_labels[idx]))
        aligned = []
        missing = []
        for key in logit_ids:
            if key not in feature_by_id:
                missing.append(key)
            else:
                aligned.append(feature_by_id[key])
        if missing:
            preview = ", ".join(missing[:5])
            raise ValueError(f"{len(missing)} logits had no matching object features: {preview}")
        return torch.stack(aligned, dim=0), labels

    if features.shape[0] != logits.shape[0]:
        raise ValueError(
            "Cannot align object features to actor logits without ids: "
            f"features={tuple(features.shape)} logits={tuple(logits.shape)}"
        )
    if not torch.equal(feature_labels.cpu(), labels.cpu()):
        mismatch = int((feature_labels.cpu() != labels.cpu()).sum())
        print(f"WARNING: feature/logit labels differ in {mismatch} rows; using logit labels.")
    return features, labels


def macro_metrics(logits, labels, num_classes):
    preds = logits.argmax(dim=-1)
    labels = labels.long()
    class_acc = []
    class_f1 = []
    for class_id in range(num_classes):
        target = labels == class_id
        pred = preds == class_id
        support = int(target.sum())
        if support == 0:
            continue
        tp = int((target & pred).sum())
        fp = int((~target & pred).sum())
        fn = int((target & ~pred).sum())
        class_acc.append(tp / max(support, 1))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        denom = precision + recall
        class_f1.append(0.0 if denom == 0 else 2 * precision * recall / denom)
    micro = float((preds == labels).float().mean())
    macro_acc = float(sum(class_acc) / max(len(class_acc), 1))
    macro_f1 = float(sum(class_f1) / max(len(class_f1), 1))
    return {"micro": micro, "macro": macro_acc, "f1": macro_f1}


class ObjectResidualProbe(nn.Module):
    def __init__(
        self,
        input_dim,
        num_classes,
        num_object_classes,
        hidden_dim,
        gate_init,
        task_type,
        gated,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_object_classes = num_object_classes
        self.gated = gated
        self.norm = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
        )
        self.gate_logit = nn.Parameter(torch.logit(torch.tensor(float(gate_init))))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.register_buffer(
            "action_gate_prior",
            torch.tensor(
                object_summary_action_gate_prior(task_type, num_classes),
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_object_matrix",
            torch.tensor(
                object_summary_action_object_matrix(task_type, num_classes),
                dtype=torch.float32,
            ),
        )

    def _action_evidence(self, summary):
        summary = summary.view(
            summary.shape[0],
            OBJECT_SUMMARY_FEATURE_DIM,
            self.num_object_classes,
        )
        max_conf = summary[:, 1].clamp(0.0, 1.0)
        frame_frac = summary[:, 3].clamp(0.0, 1.0)
        object_signal = max_conf * torch.sqrt(frame_frac.clamp_min(1e-6))
        return (object_signal[:, None, :] * self.action_object_matrix[None]).amax(dim=-1)

    def forward(self, actor_logits, summary):
        delta = self.mlp(self.norm(summary))
        if self.gated:
            delta = delta * self._action_evidence(summary) * self.action_gate_prior[None]
        return actor_logits + torch.sigmoid(self.gate_logit) * delta


@torch.no_grad()
def evaluate(model, actor_logits, summary, labels, num_classes, mode):
    model.eval()
    if mode == "real":
        use_summary = summary
    elif mode == "zero":
        use_summary = torch.zeros_like(summary)
    elif mode == "shuffled":
        use_summary = summary.roll(shifts=1, dims=0)
    else:
        raise ValueError(mode)
    logits = model(actor_logits, use_summary)
    return macro_metrics(logits.cpu(), labels.cpu(), num_classes)


def train_probe(name, gated, train_data, val_data, args, device):
    train_logits, train_summary, train_labels = train_data
    val_logits, val_summary, val_labels = val_data
    model = ObjectResidualProbe(
        input_dim=train_summary.shape[1],
        num_classes=args.num_classes,
        num_object_classes=args.num_object_classes,
        hidden_dim=args.hidden_dim,
        gate_init=args.gate_init,
        task_type=args.task_type,
        gated=gated,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    dataset = TensorDataset(train_logits, train_summary, train_labels)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    best = None
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_count = 0
        for actor_logits, summary, labels in loader:
            actor_logits = actor_logits.to(device)
            summary = summary.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(actor_logits, summary)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * labels.numel()
            total_count += labels.numel()

        if epoch == 0 or epoch == args.epochs - 1 or (epoch + 1) % args.print_every == 0:
            real = evaluate(model, val_logits.to(device), val_summary.to(device), val_labels, args.num_classes, "real")
            zero = evaluate(model, val_logits.to(device), val_summary.to(device), val_labels, args.num_classes, "zero")
            shuffled = evaluate(model, val_logits.to(device), val_summary.to(device), val_labels, args.num_classes, "shuffled")
            score = real["macro"] + real["f1"]
            if best is None or score > best["score"]:
                best = {
                    "epoch": epoch,
                    "score": score,
                    "real": real,
                    "zero": zero,
                    "shuffled": shuffled,
                    "gate": float(torch.sigmoid(model.gate_logit.detach().cpu())),
                }
            print(
                f"{name} epoch={epoch:03d} loss={total_loss / max(total_count, 1):.4f} "
                f"real_macro={real['macro']:.4f} real_f1={real['f1']:.4f} "
                f"zero_macro={zero['macro']:.4f} shuf_macro={shuffled['macro']:.4f} "
                f"gate={float(torch.sigmoid(model.gate_logit.detach().cpu())):.4f}",
                flush=True,
            )

    print(f"\n{name} BEST epoch={best['epoch']} gate={best['gate']:.4f}")
    for mode in ("real", "zero", "shuffled"):
        metrics = best[mode]
        print(
            f"{name}/{mode:8s} micro={metrics['micro']:.4f} "
            f"macro={metrics['macro']:.4f} f1={metrics['f1']:.4f}"
        )
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_cache", type=Path, required=True)
    parser.add_argument("--actor_logit_cache", type=Path, required=True)
    parser.add_argument("--task_type", type=str, default="CS")
    parser.add_argument("--num_classes", type=int, default=31)
    parser.add_argument("--num_object_classes", type=int, default=19)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--gate_init", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--print_every", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print("feature_cache:", args.feature_cache)
    print("actor_logit_cache:", args.actor_logit_cache)
    print("device:", device)

    split_data = {}
    for split in ("train", "val"):
        features, feature_labels, feature_ids = _load_features(args.feature_cache, split)
        logits, labels, logit_ids = _load_logits(args.actor_logit_cache, split)
        aligned_features, labels = _align_features_to_logits(
            features,
            feature_labels,
            feature_ids,
            logits,
            labels,
            logit_ids,
        )
        split_data[split] = (logits.float(), aligned_features.float(), labels.long())
        print(
            f"{split}: logits={tuple(logits.shape)} features={tuple(aligned_features.shape)} "
            f"labels={tuple(labels.shape)} ids={'yes' if logit_ids is not None else 'no'}"
        )

    train_data = split_data["train"]
    val_data = split_data["val"]
    baseline = macro_metrics(val_data[0], val_data[2], args.num_classes)
    print(
        "\nactor_off baseline "
        f"micro={baseline['micro']:.4f} macro={baseline['macro']:.4f} f1={baseline['f1']:.4f}"
    )

    print("\n" + "=" * 90)
    print("CONTROL: global residual, no action/evidence gate")
    print("=" * 90)
    global_best = train_probe("global_delta", False, train_data, val_data, args, device)

    print("\n" + "=" * 90)
    print("PROPOSED: action/evidence-gated residual")
    print("=" * 90)
    gated_best = train_probe("action_evidence_delta", True, train_data, val_data, args, device)

    print("\n" + "=" * 90)
    print("DECISION")
    print("=" * 90)
    real = gated_best["real"]
    zero = gated_best["zero"]
    shuffled = gated_best["shuffled"]
    print(f"baseline macro={baseline['macro']:.4f} f1={baseline['f1']:.4f}")
    print(f"gated real macro={real['macro']:.4f} f1={real['f1']:.4f}")
    print(f"gated zero macro={zero['macro']:.4f} f1={zero['f1']:.4f}")
    print(f"gated shuffled macro={shuffled['macro']:.4f} f1={shuffled['f1']:.4f}")
    print(f"real-zero macro gain={real['macro'] - zero['macro']:+.4f}")
    print(f"real-shuffled macro gain={real['macro'] - shuffled['macro']:+.4f}")
    print(f"real-zero f1 gain={real['f1'] - zero['f1']:+.4f}")
    passed = (
        real["macro"] > zero["macro"] + 0.005
        and real["macro"] > shuffled["macro"] + 0.005
        and real["f1"] >= zero["f1"] - 0.003
    )
    print("PASS:", bool(passed))


if __name__ == "__main__":
    main()
