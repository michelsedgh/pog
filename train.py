import argparse
import __main__
import os
import pickle
import sys
import warnings
from argparse import ArgumentParser

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from datamodule.base_datamod import BaseDataModule
from datasets.toyotasm import ToyotaSMDataset
from models.poguise import POGUISE
from modules.heatmap_module import HeatmapModule


os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
warnings.filterwarnings(
    "ignore",
    message="The video decoding and encoding capabilities of torchvision are deprecated.*",
    category=UserWarning,
)


def to_normalized_float_tensor(vid):
    return vid.to(torch.float32) / 255


class _LegacyCheckpointPlaceholder:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __call__(self, value):
        return value


class _LegacyCheckpointUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (AttributeError, ModuleNotFoundError):
            if module == "__main__" and name == "to_normalized_float_tensor":
                return to_normalized_float_tensor
            return _LegacyCheckpointPlaceholder


class _LegacyCheckpointPickle:
    Unpickler = _LegacyCheckpointUnpickler
    Pickler = pickle.Pickler
    dump = pickle.dump
    dumps = pickle.dumps
    load = pickle.load
    loads = pickle.loads
    HIGHEST_PROTOCOL = pickle.HIGHEST_PROTOCOL


class DatasetEpochCallback(Callback):
    def _set_epoch(self, trainer, epoch):
        datamodule = trainer.datamodule
        if datamodule is None or not hasattr(datamodule, "train_dataset"):
            return
        train_dataset = datamodule.train_dataset
        if hasattr(train_dataset, "set_epoch"):
            train_dataset.set_epoch(epoch)

    def on_fit_start(self, trainer, pl_module):
        self._set_epoch(trainer, trainer.current_epoch)

    def on_train_epoch_end(self, trainer, pl_module):
        self._set_epoch(trainer, trainer.current_epoch + 1)


def _explicit_cli_overrides(parser, args):
    option_to_dest = {}
    for action in parser._actions:
        for option in action.option_strings:
            option_to_dest[option] = action.dest

    provided = set()
    for token in sys.argv[1:]:
        if not token.startswith("--"):
            continue
        option = token.split("=", 1)[0]
        dest = option_to_dest.get(option)
        if dest is not None:
            provided.add(dest)

    values = vars(args)
    return {dest: values[dest] for dest in provided}


def _load_checkpoint(path):
    if not path:
        return None
    if not hasattr(__main__, "to_normalized_float_tensor"):
        __main__.to_normalized_float_tensor = to_normalized_float_tensor
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except (AttributeError, ModuleNotFoundError) as exc:
        print(f"Retrying checkpoint load with legacy pickle compatibility: {exc}")
        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
            pickle_module=_LegacyCheckpointPickle,
        )


def _merged_hparams(args, cli_overrides, checkpoint):
    merged = vars(args).copy()
    if checkpoint is not None:
        merged.update(checkpoint.get("hyper_parameters", {}))
        merged.update(checkpoint.get("datamodule_hyper_parameters", {}))
    merged.update(cli_overrides)

    if "resume_from_checkpoint" not in cli_overrides:
        merged["resume_from_checkpoint"] = None
    elif not merged.get("resume_from_checkpoint"):
        merged["resume_from_checkpoint"] = None

    if merged.get("max_epochs") is not None:
        merged["max_nb_epochs"] = merged["max_epochs"]
    if merged.get("reload_dataloaders_every_n_epochs") is None:
        merged["reload_dataloaders_every_n_epochs"] = 0
    if merged.get("check_val_every_n_epoch") is None:
        merged["check_val_every_n_epoch"] = 1
    merged["mode"] = "train"
    merged["dataset_artifact"] = merged.get("dataset_artifact") or merged.get("dataset")
    return argparse.Namespace(**merged)


def _dataset_class(name):
    if name == "toyotasm":
        return ToyotaSMDataset
    if name == "driveact":
        from datasets.driveact import DriveActDataset

        return DriveActDataset
    raise ValueError(f"Unsupported dataset: {name}")


def _checkpoint_has_key(checkpoint, key):
    if checkpoint is None:
        return False
    return key in checkpoint.get("state_dict", {})


def _initialize_actor_prompt_from_checkpoint(module, checkpoint):
    model = module.model
    if not getattr(model, "actor_prompt", False):
        return

    initialized = []
    with torch.no_grad():
        if (
            hasattr(model, "actor_head")
            and hasattr(model, "head")
            and model.actor_head.weight.shape == model.head.weight.shape
            and not _checkpoint_has_key(checkpoint, "model.actor_head.weight")
        ):
            model.actor_head.weight.copy_(model.head.weight)
            model.actor_head.bias.copy_(model.head.bias)
            initialized.append("actor_head")

        net = getattr(model, "net", None)
        if net is None:
            return

        if (
            hasattr(net, "actor_token")
            and hasattr(net, "class_token")
            and net.actor_token.shape == net.class_token.shape
            and not _checkpoint_has_key(checkpoint, "model.net.actor_token")
        ):
            net.actor_token.copy_(net.class_token)
            initialized.append("actor_token")

        if hasattr(net, "actor_slot_embed") and not _checkpoint_has_key(
            checkpoint, "model.net.actor_slot_embed"
        ):
            net.actor_slot_embed.zero_()
            initialized.append("actor_slot_embed")

        if hasattr(net, "valid_embed") and not _checkpoint_has_key(
            checkpoint, "model.net.valid_embed.weight"
        ):
            net.valid_embed.weight.zero_()
            initialized.append("valid_embed")

        if hasattr(net, "bbox_mlp") and not _checkpoint_has_key(
            checkpoint, "model.net.bbox_mlp.2.weight"
        ):
            last = net.bbox_mlp[-1]
            if isinstance(last, torch.nn.Linear):
                torch.nn.init.zeros_(last.weight)
                torch.nn.init.zeros_(last.bias)
                initialized.append("bbox_mlp_final")

    if initialized:
        print(
            "Initialized actor-prompt modules from current class path: "
            + ", ".join(initialized)
        )


def _adapt_heatmap_final_layer_checkpoint(module, checkpoint):
    if checkpoint is None:
        return
    state_dict = checkpoint.get("state_dict", {})
    key_w = "model.net.heatmap_head.final_layer.weight"
    key_b = "model.net.heatmap_head.final_layer.bias"
    if key_w not in state_dict or key_b not in state_dict:
        return

    final_layer = getattr(module.model.net.heatmap_head, "final_layer", None)
    if final_layer is None:
        return

    target_w = final_layer.weight
    target_b = final_layer.bias
    old_w = state_dict[key_w]
    old_b = state_dict[key_b]
    if old_w.shape == target_w.shape and old_b.shape == target_b.shape:
        return

    if old_w.shape[1:] != target_w.shape[1:] or old_b.ndim != target_b.ndim:
        raise RuntimeError(
            "Cannot adapt heatmap final layer checkpoint shape: "
            f"{tuple(old_w.shape)} -> {tuple(target_w.shape)}"
        )

    new_w = target_w.detach().clone()
    new_b = target_b.detach().clone()
    new_w.zero_()
    new_b.zero_()
    copy_channels = min(old_w.shape[0], target_w.shape[0])
    new_w[:copy_channels].copy_(old_w[:copy_channels])
    new_b[:copy_channels].copy_(old_b[:copy_channels])
    state_dict[key_w] = new_w
    state_dict[key_b] = new_b
    print(
        "Adapted heatmap final layer from "
        f"{old_w.shape[0]} to {target_w.shape[0]} channels; "
        "copied existing channels and zero-initialized new channels."
    )


def _validate_no_deprecated_object_path(checkpoint):
    if checkpoint is None:
        return
    state_dict = checkpoint.get("state_dict", {})
    deprecated = [
        key
        for key in state_dict
        if any(
            needle in key
            for needle in (
                "object_interaction",
                "object_cls_embed",
                "object_slot_embed",
                "object_bbox_mlp",
                "object_conf_mlp",
                "object_visual_proj",
            )
        )
    ]
    if deprecated:
        preview = ", ".join(deprecated[:12])
        raise ValueError(
            "Deprecated object-token checkpoint detected. The active model uses "
            "RF-DETR only as an interaction-heatmap teacher and has no runtime "
            f"object-token path. First deprecated keys: {preview}"
        )


def _print_trainable_parameters(module):
    total = 0
    trainable = 0
    rows = []
    for name, param in module.named_parameters():
        numel = param.numel()
        total += numel
        if param.requires_grad:
            trainable += numel
            rows.append((name, numel))

    pct = 100.0 * float(trainable) / float(total) if total else 0.0
    print(
        f"Trainable parameters: {trainable:,} / {total:,} "
        f"({pct:.2f}%)"
    )
    for name, numel in rows:
        print(f"TRAINABLE {name} {numel:,}")


def build_parser():
    parser = ArgumentParser()
    parser = POGUISE.add_model_specific_args(parser)
    parser = ToyotaSMDataset.add_model_specific_args(parser)

    parser.add_argument("--dataset", type=str, default="toyotasm")
    parser.add_argument("--dataset_artifact", type=str, default="toyotasm")
    parser.add_argument("--model_file", type=str, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--strict_load", type=int, default=None)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--strategy", type=str, default="auto")
    parser.add_argument("--precision", type=str, default="bf16-mixed")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--persistent_workers", type=int, default=1)
    parser.add_argument("--prefetch_factor", type=int, default=None)
    parser.add_argument("--class_balanced_sampler", type=int, default=0)
    parser.add_argument("--hard_negative_sampler", type=int, default=0)
    parser.add_argument("--hard_negative_manifest", type=str, default=None)
    parser.add_argument("--hard_negative_prob", type=float, default=0.15)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--max_nb_epochs", type=int, default=200)
    parser.add_argument("--accum_grad_batches", type=int, default=2)
    parser.add_argument("--gradient_clip_val", type=float, default=1.5)
    parser.add_argument("--num_sanity_val_steps", type=int, default=2)
    parser.add_argument("--check_val_every_n_epoch", type=int, default=1)
    parser.add_argument("--limit_val_batches", type=float, default=None)
    parser.add_argument("--log_every_n_steps", type=int, default=50)

    parser.add_argument("--project_folder", type=str, default="toyotaSM")
    parser.add_argument("--model_name", type=str, default="poguise_actor_prompt")
    parser.add_argument("--default_root_dir", type=str, default="./checkpoints")
    parser.add_argument("--save_top_k", type=int, default=3)
    parser.add_argument("--checkpoint_monitor", type=str, default="val_loss")
    parser.add_argument("--checkpoint_mode", type=str, default="min")
    parser.add_argument(
        "--checkpoint_filename",
        type=str,
        default="{epoch:03d}-{val_loss:.4f}",
    )
    parser.add_argument("--save_every_epoch_checkpoints", type=int, default=0)
    parser.add_argument("--epoch_checkpoint_dir", type=str, default=None)
    parser.add_argument(
        "--epoch_checkpoint_filename",
        type=str,
        default="{epoch:03d}",
    )
    parser.add_argument("--reload_dataloaders_every_n_epochs", type=int, default=0)
    parser.add_argument("--print_trainable_params", type=int, default=0)
    parser.add_argument("--exit_after_print_trainable_params", type=int, default=0)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_head", type=float, default=6e-4)
    parser.add_argument("--lr_head_hm", type=float, default=0.0)
    parser.add_argument("--weight_decay", type=float, default=0.04)
    parser.add_argument("--weight_decay_head", type=float, default=0.01)
    parser.add_argument("--weight_decay_head_hm", type=float, default=0.01)
    parser.add_argument("--t_max_scheduler", type=int, default=10)
    parser.add_argument("--warm_restarts", type=int, default=0)

    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--mixup", type=int, default=0)
    parser.add_argument("--target_kp_loss_weight", type=int, default=0)
    parser.add_argument("--kp_loss_weight", type=float, default=1000.0)
    parser.add_argument("--interaction_warmup_freeze_actor_path", type=int, default=0)
    parser.add_argument("--log_kp_loss_weight", type=int, default=0)
    parser.add_argument("--grad_weights", type=int, default=0)
    parser.add_argument("--deepspeed_optim", type=int, default=0)
    parser.add_argument("--kp_only", type=int, default=0)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    cli_overrides = _explicit_cli_overrides(parser, args)
    checkpoint = _load_checkpoint(args.model_file)
    _validate_no_deprecated_object_path(checkpoint)
    hparams = _merged_hparams(args, cli_overrides, checkpoint)

    if hparams.actor_prompt and hparams.mixup:
        raise ValueError("actor_prompt training requires --mixup 0")
    if hparams.actor_prompt and hparams.grad_weights:
        raise ValueError("actor_prompt training requires --grad_weights 0")
    if hparams.actor_interaction_heatmaps and not hparams.actor_prompt:
        raise ValueError("actor_interaction_heatmaps requires actor_prompt")

    seed_everything(hparams.seed)
    dataset = _dataset_class(hparams.dataset)
    module = HeatmapModule(model=POGUISE, **vars(hparams))

    if checkpoint is not None:
        _adapt_heatmap_final_layer_checkpoint(module, checkpoint)
        strict = (
            bool(hparams.strict_load)
            if hparams.strict_load is not None
            else not bool(hparams.actor_prompt)
        )
        result = module.load_state_dict(checkpoint["state_dict"], strict=strict)
        if hparams.actor_prompt:
            _initialize_actor_prompt_from_checkpoint(module, checkpoint)
        if not strict:
            print("Missing keys:", result.missing_keys)
            print("Unexpected keys:", result.unexpected_keys)
    elif hparams.actor_prompt:
        _initialize_actor_prompt_from_checkpoint(module, None)

    if hparams.print_trainable_params:
        _print_trainable_parameters(module)
        if hparams.exit_after_print_trainable_params:
            return

    datamodule_params = vars(hparams).copy()
    datamodule_params.pop("dataset", None)
    datamodule = BaseDataModule(dataset, **datamodule_params)

    accelerator = hparams.accelerator
    if accelerator == "auto":
        accelerator = "gpu" if hparams.gpus and torch.cuda.is_available() else "cpu"
    devices = hparams.gpus if accelerator == "gpu" else "auto"
    strategy = hparams.strategy
    if strategy == "auto" and accelerator == "gpu" and int(hparams.gpus) > 1:
        strategy = "ddp"

    root_dir = os.path.join(hparams.default_root_dir, hparams.model_name)
    logger = CSVLogger(save_dir=hparams.default_root_dir, name=hparams.model_name)
    validation_disabled = (
        hparams.limit_val_batches is not None
        and float(hparams.limit_val_batches) == 0.0
    )
    checkpoint_monitor = hparams.checkpoint_monitor
    if checkpoint_monitor is not None and str(checkpoint_monitor).lower() == "none":
        checkpoint_monitor = None
    checkpoint_callback = ModelCheckpoint(
        monitor=None if validation_disabled else checkpoint_monitor,
        mode=hparams.checkpoint_mode,
        save_top_k=0 if validation_disabled else hparams.save_top_k,
        save_last=True,
        filename=hparams.checkpoint_filename,
    )
    callbacks = [checkpoint_callback, DatasetEpochCallback()]
    if hparams.save_every_epoch_checkpoints:
        callbacks.append(
            ModelCheckpoint(
                dirpath=hparams.epoch_checkpoint_dir
                or os.path.join(root_dir, "epoch_checkpoints"),
                filename=hparams.epoch_checkpoint_filename,
                monitor=None,
                save_top_k=-1,
                every_n_epochs=1,
                save_on_train_epoch_end=True,
                save_last=False,
            )
        )
    trainer_kwargs = {}
    if hparams.limit_val_batches is not None:
        trainer_kwargs["limit_val_batches"] = hparams.limit_val_batches
    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
        num_nodes=hparams.nodes,
        strategy=strategy,
        max_epochs=hparams.max_nb_epochs,
        precision=hparams.precision,
        default_root_dir=root_dir,
        logger=logger,
        callbacks=callbacks,
        accumulate_grad_batches=hparams.accum_grad_batches,
        gradient_clip_val=hparams.gradient_clip_val,
        num_sanity_val_steps=hparams.num_sanity_val_steps,
        check_val_every_n_epoch=hparams.check_val_every_n_epoch,
        log_every_n_steps=hparams.log_every_n_steps,
        reload_dataloaders_every_n_epochs=hparams.reload_dataloaders_every_n_epochs,
        benchmark=True,
        **trainer_kwargs,
    )
    trainer.fit(
        module,
        datamodule=datamodule,
        ckpt_path=hparams.resume_from_checkpoint,
    )


if __name__ == "__main__":
    main()
