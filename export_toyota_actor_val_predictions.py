#!/usr/bin/env python3
import argparse
import csv
import os
from pathlib import Path

import torch
from pytorch_lightning import seed_everything

from datamodule.base_datamod import BaseDataModule
from datasets.toyotasm import CS_DICT, ToyotaSMDataset
from models.poguise import POGUISE
from modules.heatmap_module import HeatmapModule
from train import _explicit_cli_overrides, _load_checkpoint, _merged_hparams, build_parser


ID_TO_ACTION = {idx - 1: name for name, idx in CS_DICT.items()}


def _add_export_args(parser):
    parser.add_argument("--export_predictions_file", required=True)
    parser.add_argument("--export_device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser


def _move_target(target, device):
    moved = {}
    for key, value in target.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def _unpack_model_data(data):
    if len(data) == 3:
        preds, hm_preds, presence_logits = data
    else:
        preds, hm_preds = data
    return preds


def main():
    parser = _add_export_args(build_parser())
    args = parser.parse_args()
    cli_overrides = _explicit_cli_overrides(parser, args)
    checkpoint = _load_checkpoint(args.model_file)
    hparams = _merged_hparams(args, cli_overrides, checkpoint)
    hparams.mode = "train"
    hparams.num_sanity_val_steps = 0
    hparams.reload_dataloaders_every_n_epochs = 0
    hparams.dataset = "toyotasm"
    hparams.dataset_artifact = "toyotasm"
    hparams.limit_val_batches = 1.0

    seed_everything(hparams.seed)

    if args.export_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.export_device)

    datamodule_params = vars(hparams).copy()
    datamodule_params.pop("dataset", None)
    datamodule = BaseDataModule(ToyotaSMDataset, **datamodule_params)
    datamodule.setup("validate")
    val_loader = datamodule.val_dataloader()
    val_dataset = datamodule.val_dataset
    file_ids = val_dataset.data_df.file_id.astype(str).tolist()

    module = HeatmapModule(model=POGUISE, **vars(hparams))
    state_dict = checkpoint.get("state_dict")
    if not state_dict:
        raise RuntimeError(f"Checkpoint has no state_dict: {args.model_file}")
    result = module.load_state_dict(state_dict, strict=bool(args.strict_load))
    if result.missing_keys:
        print(f"missing keys: {len(result.missing_keys)}", flush=True)
    if result.unexpected_keys:
        print(f"unexpected keys: {len(result.unexpected_keys)}", flush=True)

    model = module.model.to(device)
    model.eval()

    output_path = Path(args.export_predictions_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    offset = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            imgs, target = batch
            batch_size = imgs.shape[0]
            batch_file_ids = file_ids[offset : offset + batch_size]
            offset += batch_size

            imgs = imgs.to(device, non_blocking=True)
            target = _move_target(target, device)
            data = model(
                imgs,
                boxes=target["boxes"].float(),
                valid=target["valid"].bool(),
            )
            logits = _unpack_model_data(data).float().cpu()
            labels = target["actions"].long().cpu()
            valid = target["valid"].bool().cpu()
            probs = torch.softmax(logits, dim=-1)

            for sample_idx, file_id in enumerate(batch_file_ids):
                valid_slots = torch.nonzero(valid[sample_idx], as_tuple=False).flatten()
                if valid_slots.numel() == 0:
                    continue
                for slot in valid_slots.tolist():
                    label_idx = int(labels[sample_idx, slot].item())
                    row_logits = logits[sample_idx, slot]
                    pred_idx = int(row_logits.argmax().item())
                    row = {
                        "file_id": file_id,
                        "slot": slot,
                        "label_idx": label_idx,
                        "label": ID_TO_ACTION.get(label_idx, str(label_idx)),
                        "pred_idx": pred_idx,
                        "pred": ID_TO_ACTION.get(pred_idx, str(pred_idx)),
                        "prob_true": float(probs[sample_idx, slot, label_idx].item()),
                    }
                    for class_idx, value in enumerate(row_logits.tolist()):
                        row[f"logit_{class_idx}"] = float(value)
                    rows.append(row)

            print(
                f"exported batch {batch_idx + 1}/{len(val_loader)} rows={len(rows)}",
                flush=True,
            )

    if offset != len(file_ids):
        raise RuntimeError(
            f"Val loader/file_id length mismatch: consumed {offset}, dataset has {len(file_ids)}"
        )
    if not rows:
        raise RuntimeError("No prediction rows were exported")

    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows: {output_path}", flush=True)


if __name__ == "__main__":
    main()
