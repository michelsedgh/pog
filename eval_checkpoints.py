import re
from pathlib import Path

import pandas as pd
import torch
from pytorch_lightning import Trainer, seed_everything

from datamodule.base_datamod import BaseDataModule
from models.poguise import POGUISE
from modules.heatmap_module import HeatmapModule
from train import (
    _dataset_class,
    _explicit_cli_overrides,
    _load_checkpoint,
    _merged_hparams,
    build_parser,
)


def _add_eval_args(parser):
    parser.add_argument("--eval_checkpoint_dir", type=str, default=None)
    parser.add_argument("--eval_checkpoint_glob", type=str, default="epoch=*.ckpt")
    parser.add_argument("--eval_results_file", type=str, default=None)
    parser.add_argument("--eval_last_n_checkpoints", type=int, default=0)
    parser.add_argument("--eval_start_epoch", type=int, default=None)
    parser.add_argument("--eval_end_epoch", type=int, default=None)
    parser.add_argument("--eval_skip_existing", type=int, default=1)
    return parser


def _checkpoint_epoch(path):
    match = re.search(r"epoch[=_-](\d+)", path.name)
    if match is None:
        return None
    return int(match.group(1))


def _checkpoint_paths(hparams):
    root_dir = Path(hparams.default_root_dir) / hparams.model_name
    checkpoint_dir = Path(hparams.eval_checkpoint_dir or root_dir / "epoch_checkpoints")
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    paths = []
    for path in checkpoint_dir.glob(hparams.eval_checkpoint_glob):
        epoch = _checkpoint_epoch(path)
        if epoch is None:
            continue
        if hparams.eval_start_epoch is not None and epoch < hparams.eval_start_epoch:
            continue
        if hparams.eval_end_epoch is not None and epoch > hparams.eval_end_epoch:
            continue
        paths.append((epoch, path))
    paths.sort(key=lambda item: (item[0], str(item[1])))
    if hparams.eval_last_n_checkpoints > 0:
        paths = paths[-hparams.eval_last_n_checkpoints :]
    if not paths:
        raise RuntimeError(
            f"No checkpoints matched {checkpoint_dir}/{hparams.eval_checkpoint_glob}"
        )
    return paths


def _results_path(hparams):
    if hparams.eval_results_file:
        return Path(hparams.eval_results_file)
    return Path(hparams.default_root_dir) / hparams.model_name / "checkpoint_eval.csv"


def _existing_rows(path):
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _metric_value(value):
    if torch.is_tensor(value):
        value = value.detach().cpu()
        if value.numel() == 1:
            return float(value.item())
        return value.tolist()
    if isinstance(value, (int, float, str)):
        return value
    return str(value)


def _write_results(path, dataframe):
    if "epoch" in dataframe.columns:
        dataframe = dataframe.sort_values(["epoch", "checkpoint"]).reset_index(drop=True)
    dataframe.to_csv(path, index=False)
    return dataframe


def _load_module_checkpoint(module, ckpt_path):
    checkpoint = _load_checkpoint(ckpt_path)
    state_dict = checkpoint.get("state_dict")
    if not state_dict:
        raise RuntimeError(f"Checkpoint has no state_dict: {ckpt_path}")
    result = module.load_state_dict(state_dict, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            "Checkpoint state_dict mismatch for "
            f"{ckpt_path}. Missing={result.missing_keys}, "
            f"unexpected={result.unexpected_keys}"
        )


def main():
    parser = _add_eval_args(build_parser())
    args = parser.parse_args()
    cli_overrides = _explicit_cli_overrides(parser, args)
    checkpoint = _load_checkpoint(args.model_file)
    hparams = _merged_hparams(args, cli_overrides, checkpoint)
    hparams.mode = "train"
    hparams.num_sanity_val_steps = 0
    hparams.reload_dataloaders_every_n_epochs = 0

    seed_everything(hparams.seed)
    dataset = _dataset_class(hparams.dataset)
    datamodule_params = vars(hparams).copy()
    datamodule_params.pop("dataset", None)
    datamodule = BaseDataModule(dataset, **datamodule_params)
    datamodule.setup("validate")
    val_loader = datamodule.val_dataloader()

    accelerator = hparams.accelerator
    if accelerator == "auto":
        accelerator = "gpu" if hparams.gpus and torch.cuda.is_available() else "cpu"
    devices = hparams.gpus if accelerator == "gpu" else "auto"
    strategy = hparams.strategy
    if strategy == "auto" and accelerator == "gpu" and int(hparams.gpus) > 1:
        strategy = "ddp"

    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
        num_nodes=hparams.nodes,
        strategy=strategy,
        precision=hparams.precision,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        benchmark=True,
    )

    results_path = _results_path(hparams)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    existing = _existing_rows(results_path)
    existing_checkpoints = (
        set(existing["checkpoint"].astype(str).tolist())
        if hparams.eval_skip_existing and "checkpoint" in existing.columns
        else set()
    )

    output = existing.copy()
    wrote_count = 0
    for epoch, ckpt_path in _checkpoint_paths(hparams):
        if str(ckpt_path) in existing_checkpoints:
            print(f"Skipping already evaluated checkpoint: {ckpt_path}", flush=True)
            continue

        print(f"Evaluating epoch {epoch}: {ckpt_path}", flush=True)
        module = HeatmapModule(model=POGUISE, **vars(hparams))
        _load_module_checkpoint(module, ckpt_path)
        metrics = trainer.validate(
            module,
            dataloaders=val_loader,
            ckpt_path=None,
            verbose=False,
        )[0]
        row = {
            "epoch": epoch,
            "checkpoint": str(ckpt_path),
        }
        row.update({key: _metric_value(value) for key, value in metrics.items()})
        output = pd.concat([output, pd.DataFrame([row])], ignore_index=True)
        output = _write_results(results_path, output)
        existing_checkpoints.add(str(ckpt_path))
        wrote_count += 1

        summary_keys = [
            "val_acc_macro",
            "val_acc_micro",
            "val_actor_pair_swap_acc",
            "val_actor_all_slot_acc",
            "val_loss",
        ]
        summary = ", ".join(
            f"{key}={row[key]:.4g}"
            for key in summary_keys
            if isinstance(row.get(key), (int, float))
        )
        print(f"epoch {epoch}: {summary}", flush=True)
        print(f"Updated checkpoint eval results: {results_path}", flush=True)

    if not wrote_count:
        print(
            f"No new checkpoints to evaluate. Results file unchanged: {results_path}",
            flush=True,
        )
        return

    print(f"Wrote {wrote_count} checkpoint eval rows: {results_path}", flush=True)


if __name__ == "__main__":
    main()
