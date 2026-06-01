#!/usr/bin/env python3
"""Run PyTorch object-sensitivity diagnostics on a saved live actor input.

This script is intentionally independent of TensorRT/ONNX/dashboard runtime.
It answers whether changing RF-DETR object candidates changes the action logits
for the exact live clip tensor that the dashboard packed for the actor model.
"""

import argparse
import copy
from pathlib import Path

import torch

from utils.actor_model import load_actor_model


TARGET_ACTIONS = [
    "Uselaptop",
    "Readbook",
    "WatchTV",
    "Usetelephone",
    "Drink.Fromcup",
    "Drink.Frombottle",
    "Drink.Fromglass",
]
LAPTOP_OBJECT_NAMES = {"laptop", "keyboard_mouse"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze live object sensitivity for an object-token actor checkpoint."
    )
    parser.add_argument("--checkpoint", required=True, help="Object-token actor .ckpt")
    parser.add_argument(
        "--input-pt",
        required=True,
        help="Saved dashboard input from --debug-save-latest-input",
    )
    parser.add_argument("--device", default="cuda", help="cuda, cpu, or auto")
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Model/input dtype. auto uses float16 on CUDA and float32 on CPU.",
    )
    parser.add_argument(
        "--actor-index",
        type=int,
        default=None,
        help="Actor slot to inspect. Default: highest presence among valid slots.",
    )
    parser.add_argument(
        "--target-action",
        default="Uselaptop",
        help="Action logit used for gradient audit.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=8,
        help="Number of action/object entries to print.",
    )
    parser.add_argument(
        "--allow-missing-laptop",
        action="store_true",
        help="Do not exit nonzero when the saved tensor contains no laptop/keyboard object.",
    )
    parser.add_argument(
        "--skip-gradient",
        action="store_true",
        help="Skip the backprop gradient audit. Useful on low-memory Orin runs.",
    )
    return parser.parse_args()


def resolve_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
    return torch.device(name)


def resolve_dtype(name, device):
    if name == "auto":
        return torch.float16 if device.type == "cuda" else torch.float32
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def load_payload(path):
    payload_path = Path(path)
    if not payload_path.is_file():
        raise FileNotFoundError(payload_path)
    payload = torch.load(payload_path, map_location="cpu", weights_only=False)
    required = [
        "clip",
        "boxes",
        "valid",
        "object_boxes",
        "object_cls",
        "object_conf",
        "object_valid",
        "action_classes",
        "object_classes",
    ]
    missing = [key for key in required if key not in payload]
    if missing:
        raise KeyError(f"Saved input is missing required keys: {missing}")
    return payload


def normalize_object_class_map(object_classes):
    normalized = {}
    for key, value in object_classes.items():
        normalized[int(key)] = str(value)
    return normalized


def ids_for_names(object_classes, names):
    return {
        idx
        for idx, name in object_classes.items()
        if str(name) in names
    }


def move_tensors(payload, device, dtype):
    return {
        "clip": payload["clip"].to(device=device, dtype=dtype),
        "boxes": payload["boxes"].to(device=device, dtype=dtype),
        "valid": payload["valid"].to(device=device, dtype=torch.bool),
        "object_boxes": payload["object_boxes"].to(device=device, dtype=dtype),
        "object_cls": payload["object_cls"].to(device=device).long(),
        "object_conf": payload["object_conf"].to(device=device, dtype=dtype),
        "object_valid": payload["object_valid"].to(device=device, dtype=torch.bool),
    }


def clone_inputs(inputs):
    return {
        key: value.clone()
        for key, value in inputs.items()
    }


def run_model(model, inputs):
    with torch.inference_mode():
        output = model(
            inputs["clip"],
            boxes=inputs["boxes"],
            valid=inputs["valid"],
            object_boxes=inputs["object_boxes"],
            object_cls=inputs["object_cls"],
            object_conf=inputs["object_conf"],
            object_valid=inputs["object_valid"],
        )
    if not isinstance(output, (tuple, list)) or len(output) < 3:
        raise RuntimeError("Expected actor model output with logits and presence logits.")
    if len(output) >= 5:
        logits, _heatmap, presence, selection_logits, interaction_heatmap = output[:5]
    else:
        logits, _heatmap, presence = output[:3]
        selection_logits = None
        interaction_heatmap = None
    if presence is None:
        raise RuntimeError("Model did not return actor presence logits.")
    return {
        "logits": logits.detach().float(),
        "presence": presence.detach().float(),
        "selection_logits": (
            None if selection_logits is None else selection_logits.detach().float()
        ),
        "interaction_heatmap": (
            None if interaction_heatmap is None else interaction_heatmap.detach().float()
        ),
    }


def choose_actor_index(outputs, valid, requested):
    valid = valid[0].detach().cpu().bool()
    if requested is not None:
        if requested < 0 or requested >= valid.numel():
            raise ValueError(f"--actor-index {requested} is outside [0,{valid.numel() - 1}]")
        if not bool(valid[requested]):
            print(f"WARNING: requested actor slot {requested} is not marked valid.")
        return int(requested)

    valid_indices = torch.where(valid)[0]
    if valid_indices.numel() == 0:
        raise RuntimeError("No valid actor slots in saved input.")
    presence_prob = torch.sigmoid(outputs["presence"][0]).detach().cpu()
    best = valid_indices[presence_prob[valid_indices].argmax()]
    return int(best.item())


def action_indices(action_classes):
    return {
        name: action_classes.index(name)
        for name in TARGET_ACTIONS
        if name in action_classes
    }


def top_actions(logits, action_classes, actor_idx, k):
    probs = torch.softmax(logits[0, actor_idx], dim=-1)
    values, indices = torch.topk(probs, k=min(k, probs.numel()))
    return [
        {
            "rank": rank + 1,
            "class": action_classes[int(idx)],
            "prob": float(prob),
            "logit": float(logits[0, actor_idx, int(idx)]),
        }
        for rank, (prob, idx) in enumerate(zip(values, indices))
    ]


def target_row(name, outputs, action_map, action_classes, actor_idx):
    logits = outputs["logits"][0, actor_idx]
    probs = torch.softmax(logits, dim=-1)
    row = {"mode": name}
    for action in TARGET_ACTIONS:
        idx = action_map.get(action)
        if idx is None:
            continue
        row[f"{action}_logit"] = float(logits[idx])
        row[f"{action}_prob"] = float(probs[idx])
    top = top_actions(outputs["logits"], action_classes, actor_idx, 5)
    row["top1"] = top[0]["class"] if top else "NONE"
    row["top1_prob"] = top[0]["prob"] if top else float("nan")
    return row


def object_summary(payload, object_classes):
    object_valid = payload["object_valid"][0].bool()
    object_cls = payload["object_cls"][0].long()
    object_conf = payload["object_conf"][0].float()
    object_boxes = payload["object_boxes"][0].float()
    rows = []
    for slot in range(object_valid.numel()):
        if not bool(object_valid[slot]):
            continue
        cls_id = int(object_cls[slot])
        rows.append(
            {
                "slot": slot,
                "name": object_classes.get(cls_id, f"object_{cls_id}"),
                "conf": float(object_conf[slot]),
                "box": [round(float(v), 4) for v in object_boxes[slot].tolist()],
            }
        )
    return rows


def print_object_summary(rows):
    print("\nOBJECTS IN SAVED INPUT")
    if not rows:
        print("  none")
        return
    for row in rows:
        print(
            f"  slot={row['slot']:02d} {row['name']:<16} "
            f"conf={row['conf']:.3f} box={row['box']}"
        )


def build_modes(base_inputs, object_classes):
    laptop_ids = ids_for_names(object_classes, LAPTOP_OBJECT_NAMES)
    book_ids = ids_for_names(object_classes, {"book"})
    if not book_ids:
        raise RuntimeError("Object vocabulary does not contain book.")
    book_id = sorted(book_ids)[0]

    object_cls = base_inputs["object_cls"]
    object_valid = base_inputs["object_valid"]
    laptop_mask = object_valid & torch.isin(
        object_cls,
        torch.tensor(sorted(laptop_ids), dtype=object_cls.dtype, device=object_cls.device),
    )

    modes = {}
    modes["objects_on"] = clone_inputs(base_inputs)

    off = clone_inputs(base_inputs)
    off["object_valid"] = torch.zeros_like(off["object_valid"])
    modes["objects_off"] = off

    laptop_only = clone_inputs(base_inputs)
    laptop_only["object_valid"] = laptop_mask.clone()
    modes["laptop_only"] = laptop_only

    erased = clone_inputs(base_inputs)
    erased["object_valid"] = erased["object_valid"] & ~laptop_mask
    modes["positive_erased_laptop"] = erased

    laptop_to_book = clone_inputs(base_inputs)
    laptop_to_book["object_cls"] = laptop_to_book["object_cls"].clone()
    laptop_to_book["object_cls"][laptop_mask] = int(book_id)
    modes["laptop_class_changed_to_book"] = laptop_to_book

    moved = clone_inputs(base_inputs)
    moved["object_boxes"] = moved["object_boxes"].clone()
    if laptop_mask.any():
        moved["object_boxes"][laptop_mask] = torch.tensor(
            [0.02, 0.02, 0.16, 0.16],
            dtype=moved["object_boxes"].dtype,
            device=moved["object_boxes"].device,
        )
    modes["laptop_box_moved_away"] = moved

    return modes, laptop_mask


def print_target_table(rows, action_map):
    print("\nTARGET ACTION LOGITS / PROBS")
    header = ["mode", "top1", "top1_prob"]
    for action in TARGET_ACTIONS:
        if action in action_map:
            header.extend([f"{action}_logit", f"{action}_prob"])
    print("  " + " | ".join(header))
    for row in rows:
        values = []
        for key in header:
            value = row.get(key, "")
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        print("  " + " | ".join(values))


def print_delta_table(rows, action_map):
    base = rows[0]
    print("\nLOGIT DELTAS VS objects_on")
    for row in rows[1:]:
        print(f"  {row['mode']}:")
        for action in TARGET_ACTIONS:
            key = f"{action}_logit"
            if action in action_map and key in row and key in base:
                print(f"    {action:<18} {row[key] - base[key]:+.4f}")


def print_selection(outputs_by_mode, payload, object_classes, actor_idx, laptop_mask, topk):
    print("\nOBJECT SELECTION")
    for mode_name, outputs in outputs_by_mode.items():
        selection_logits = outputs.get("selection_logits")
        if selection_logits is None:
            print(f"  {mode_name}: no selection logits")
            continue
        alpha = torch.softmax(selection_logits[0, actor_idx].float(), dim=-1).detach().cpu()
        valid = payload["object_valid"][0].bool().clone()
        if mode_name == "objects_off":
            valid.zero_()
        elif mode_name == "laptop_only":
            valid = laptop_mask[0].detach().cpu().bool()
        elif mode_name == "positive_erased_laptop":
            valid = valid & ~laptop_mask[0].detach().cpu().bool()
        object_cls = payload["object_cls"][0].long()
        object_conf = payload["object_conf"][0].float()
        laptop_mass = float(alpha[:-1][laptop_mask[0].detach().cpu().bool()].sum())
        none_mass = float(alpha[-1])
        print(f"  {mode_name}: laptop_mass={laptop_mass:.4f} none_mass={none_mass:.4f}")
        real_alpha = alpha[:-1]
        candidates = []
        for slot in range(real_alpha.numel()):
            cls_id = int(object_cls[slot])
            candidates.append(
                (
                    float(real_alpha[slot]),
                    slot,
                    object_classes.get(cls_id, f"object_{cls_id}"),
                    bool(valid[slot]),
                    float(object_conf[slot]),
                )
            )
        candidates.append((float(alpha[-1]), real_alpha.numel(), "NONE", True, 0.0))
        for score, slot, name, is_valid, conf in sorted(candidates, reverse=True)[:topk]:
            marker = "valid" if is_valid else "masked"
            print(
                f"    slot={slot:02d} {name:<16} alpha={score:.4f} "
                f"{marker:<6} conf={conf:.3f}"
            )


def gradient_audit(model, base_inputs, action_map, actor_idx, target_action):
    if target_action not in action_map:
        print(f"\nGRADIENT AUDIT skipped: missing action {target_action}")
        return
    inputs = clone_inputs(base_inputs)
    inputs["object_boxes"] = inputs["object_boxes"].detach().clone().requires_grad_(True)
    inputs["object_conf"] = inputs["object_conf"].detach().clone().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    output = model(
        inputs["clip"],
        boxes=inputs["boxes"],
        valid=inputs["valid"],
        object_boxes=inputs["object_boxes"],
        object_cls=inputs["object_cls"],
        object_conf=inputs["object_conf"],
        object_valid=inputs["object_valid"],
    )
    logits = output[0]
    target_logit = logits[0, actor_idx, action_map[target_action]]
    target_logit.backward()

    print(f"\nGRADIENT AUDIT target={target_action} actor={actor_idx}")
    box_grad = inputs["object_boxes"].grad
    conf_grad = inputs["object_conf"].grad
    print(f"  input object_boxes grad norm: {float(box_grad.norm()):.6e}")
    print(f"  input object_conf  grad norm: {float(conf_grad.norm()):.6e}")

    groups = {
        "net.object_cls_embed": "net.object_cls_embed",
        "net.object_bbox_mlp": "net.object_bbox_mlp",
        "net.object_conf_mlp": "net.object_conf_mlp",
        "net.object_visual_proj": "net.object_visual_proj",
        "object_interaction.selector": "object_interaction.selector",
        "object_interaction.pair_visual_proj": "object_interaction.pair_visual_proj",
        "object_interaction.none_mlp": "object_interaction.none_mlp",
        "actor_head": "actor_head",
    }
    named_params = list(model.named_parameters())
    for label, prefix in groups.items():
        total = 0.0
        count = 0
        for name, param in named_params:
            if name.startswith(prefix) and param.grad is not None:
                total += float(param.grad.detach().float().norm().item() ** 2)
                count += 1
        print(f"  {label:<34} grad_norm={(total ** 0.5):.6e} params_with_grad={count}")


def judge(rows, laptop_present):
    print("\nJUDGMENT")
    if not laptop_present:
        print(
            "  This saved tensor cannot diagnose the laptop failure: it contains no "
            "valid laptop/keyboard object. Save/upload a live tensor while RF-DETR "
            "is actually detecting the laptop, or provide a short video so we can "
            "pack a new tensor."
        )
        return

    by_mode = {row["mode"]: row for row in rows}
    on = by_mode.get("objects_on", {})
    off = by_mode.get("objects_off", {})
    laptop_only = by_mode.get("laptop_only", {})
    erased = by_mode.get("positive_erased_laptop", {})
    moved = by_mode.get("laptop_box_moved_away", {})
    cls_book = by_mode.get("laptop_class_changed_to_book", {})
    key = "Uselaptop_logit"
    if key not in on:
        print("  Missing Uselaptop action in checkpoint labels.")
        return
    laptop_vs_off = laptop_only.get(key, float("nan")) - off.get(key, float("nan"))
    erased_drop = erased.get(key, float("nan")) - on.get(key, float("nan"))
    moved_drop = moved.get(key, float("nan")) - on.get(key, float("nan"))
    class_change = cls_book.get(key, float("nan")) - on.get(key, float("nan"))
    print(f"  laptop_only minus objects_off Uselaptop logit: {laptop_vs_off:+.4f}")
    print(f"  positive_erased minus objects_on Uselaptop logit: {erased_drop:+.4f}")
    print(f"  laptop_moved minus objects_on Uselaptop logit: {moved_drop:+.4f}")
    print(f"  laptop_class_to_book minus objects_on Uselaptop logit: {class_change:+.4f}")

    if laptop_vs_off > 0.10 and erased_drop < -0.05:
        print("  PASS: laptop object is moving the Uselaptop logit in the right direction.")
    elif abs(laptop_vs_off) < 0.05 and abs(erased_drop) < 0.05:
        print(
            "  FAIL: object changes barely affect Uselaptop. The action path is "
            "mostly ignoring the laptop object on this live tensor."
        )
    else:
        print(
            "  MIXED: objects affect logits, but the direction/magnitude is not "
            "clean enough to trust without live A/B."
        )


def main():
    args = parse_args()
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    payload = load_payload(args.input_pt)
    object_classes = normalize_object_class_map(payload["object_classes"])
    action_classes = list(payload["action_classes"])
    action_map = action_indices(action_classes)

    print("checkpoint:", args.checkpoint)
    print("input:", args.input_pt)
    print("device:", device)
    print("dtype:", dtype)
    print("frame:", payload.get("frame"))

    object_rows = object_summary(payload, object_classes)
    print_object_summary(object_rows)

    model, hparams, metadata = load_actor_model(
        args.checkpoint,
        torch.device("cpu"),
        return_metadata=True,
    )
    model.to(device=device, dtype=dtype)
    model.eval()
    print("\nCHECKPOINT METADATA")
    print("  epoch:", metadata.get("epoch"))
    print("  global_step:", metadata.get("global_step"))
    print("  object_prompt:", hparams.get("object_prompt"))
    print("  num_object_tokens:", hparams.get("num_object_tokens"))

    base_inputs = move_tensors(payload, device, dtype)
    modes, laptop_mask = build_modes(base_inputs, object_classes)
    laptop_present = bool(laptop_mask.any().item())
    if not laptop_present:
        print("\nWARNING: no valid laptop/keyboard object in this saved input.")

    outputs_by_mode = {}
    for name, inputs in modes.items():
        outputs_by_mode[name] = run_model(model, inputs)

    actor_idx = choose_actor_index(
        outputs_by_mode["objects_on"],
        base_inputs["valid"],
        args.actor_index,
    )
    print("\nACTOR SLOT")
    print("  selected:", actor_idx)
    print(
        "  presence prob:",
        float(torch.sigmoid(outputs_by_mode["objects_on"]["presence"][0, actor_idx])),
    )

    rows = [
        target_row(name, outputs, action_map, action_classes, actor_idx)
        for name, outputs in outputs_by_mode.items()
    ]
    print_target_table(rows, action_map)
    print_delta_table(rows, action_map)
    print_selection(outputs_by_mode, payload, object_classes, actor_idx, laptop_mask, args.topk)
    if args.skip_gradient:
        print("\nGRADIENT AUDIT skipped by --skip-gradient")
    else:
        gradient_audit(model, base_inputs, action_map, actor_idx, args.target_action)
    judge(rows, laptop_present)

    if not laptop_present and not args.allow_missing_laptop:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
