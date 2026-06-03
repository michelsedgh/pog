import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import torchmetrics
import torch
import torch.nn.functional as F
import pandas as pd
import os
from losses.softtarget import SoftTargetCrossEntropy
from losses.heatmap_loss import KeypointMSELoss
from losses.interaction_heatmap_losses import interaction_heatmap_loss
from losses.poguiseplus_losses import heatmap_frobenius_loss
import pickle
from datasets.object_vocab import GROUPS
from datasets.toyotasm import CS_DICT, CV_DICT

try:
    from grad_weights.nash_mtl import NashMTL
except ImportError:
    NashMTL = None

try:
    from deepspeed.ops.adam import DeepSpeedCPUAdam
except ImportError:
    DeepSpeedCPUAdam = None


class HeatmapModule(pl.LightningModule):
    def __init__(self, model, hparams=None, **kwargs):
        """
        Inputs:
            model_name - Name of the model/CNN to run. Used for creating the model (see function below)
            model_hparams - Hyperparameters for the model, as dictionary.
            optimizer_name - Name of the optimizer to use. Currently supported: Adam, SGD
            optimizer_hparams - Hyperparameters for the optimizer, as dictionary. This includes learning rate, weight decay, etc.
        """
        super().__init__()
        # self.save_hyperparameters()
        self.model = model(**vars(hparams)) if hparams is not None else model(**kwargs)

        hparams = self.model.hparams
        self.actor_prompt = bool(hparams.get("actor_prompt", 0))
        self.actor_interaction_heatmaps = bool(
            hparams.get("actor_interaction_heatmaps", 0)
        )
        self.actor_poguiseplus_loss = self.actor_prompt and self.actor_interaction_heatmaps
        self.poguiseplus_heatmap_loss_weight = float(
            hparams.get("poguiseplus_heatmap_loss_weight", 1.0)
        )
        self.poguiseplus_heatmap_log_eps = float(
            hparams.get("poguiseplus_heatmap_log_eps", 1e-6)
        )
        self.num_classes = hparams.num_classes
        self.dataset_name = hparams.dataset_artifact

        # Create model
        self.lr = hparams.lr
        self.weight_decay = hparams.weight_decay
        self.lr_head = hparams.lr_head
        self.weight_decay_head = hparams.weight_decay_head
        self.t_max_scheduler = (
            hparams.t_max_scheduler if hasattr(hparams, "t_max_scheduler") else 100
        )
        # self.lr_patch_embed = hparams.lr_patch_embed
        # self.weight_decay_patch_embed = hparams.weight_decay_patch_embed
        # Create loss module
        if self.actor_prompt:
            self.train_loss = nn.CrossEntropyLoss(
                label_smoothing=float(hparams.label_smoothing or 0.0)
            )
        elif hparams.label_smoothing and hparams.mixup:
            # handled by mixup
            self.train_loss = SoftTargetCrossEntropy()
        else:
            self.train_loss = nn.CrossEntropyLoss()
        if hparams.target_kp_loss_weight and hparams.n_landmarks > 0:
            self.target_weights = [1] * hparams.n_landmarks
            self.target_weights = torch.tensor(self.target_weights)
            self.kp_loss = KeypointMSELoss(use_target_weight=True, loss_weight=1000.0)
        else:
            self.kp_loss = KeypointMSELoss()
        self.val_loss = nn.CrossEntropyLoss()
        self.val_loss_kp = self.kp_loss

        # calculate MSE and MAE for keypoint val
        self.val_mse = torchmetrics.MeanSquaredError()
        self.val_mae = torchmetrics.MeanAbsoluteError()
        self.test_mse = torchmetrics.MeanSquaredError()
        self.test_mae = torchmetrics.MeanAbsoluteError()

        # Create metrics
        # drive act dataset uses accuracy macro
        self.val_acc_micro = torchmetrics.Accuracy(
            task="multiclass", num_classes=hparams.num_classes, average="micro"
        )
        self.val_acc_macro = torchmetrics.Accuracy(
            task="multiclass", num_classes=hparams.num_classes, average="macro"
        )
        self.val_f1 = torchmetrics.F1Score(
            num_classes=hparams.num_classes, average="macro", task="multiclass"
        )
        self.best_val_acc_micro = torchmetrics.Accuracy(
            task="multiclass", num_classes=hparams.num_classes, average="micro"
        )
        self.best_val_acc_macro = torchmetrics.Accuracy(
            task="multiclass", num_classes=hparams.num_classes, average="macro"
        )
        self.best_val_f1 = torchmetrics.F1Score(
            num_classes=hparams.num_classes, average="macro", task="multiclass"
        )
        self.test_acc_micro = torchmetrics.Accuracy(
            task="multiclass", num_classes=hparams.num_classes, average="micro"
        )
        self.test_acc_macro = torchmetrics.Accuracy(
            task="multiclass", num_classes=hparams.num_classes, average="macro"
        )
        self.test_acc_topk = torchmetrics.Accuracy(
            task="multiclass", num_classes=hparams.num_classes, average="micro", top_k=5
        )
        self.test_f1 = torchmetrics.F1Score(
            num_classes=hparams.num_classes, average="macro", task="multiclass"
        )
        self.validation_step_outputs = {"preds": [], "labels": []}
        self.actor_val_diagnostics = bool(hparams.get("actor_val_diagnostics", 1))
        self.actor_val_diagnostic_max_pairs = int(
            hparams.get("actor_val_diagnostic_max_pairs", 8)
        )
        self.group_indices = self._build_group_indices()
    def load_state_dict(self, state_dict, strict=True, assign=False):
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        if self.actor_prompt and not strict:
            allowed_missing = [
                "model.net.actor_token",
                "model.net.actor_slot_embed",
                "model.net.valid_embed",
                "model.net.bbox_mlp",
                "model.actor_head",
                "model.presence_head",
            ]
            if self.model.hparams.get("use_register_tokens", 0):
                allowed_missing.append("model.net.register_tokens")
            red_flag_missing = (
                "model.net.patch_embed",
                "model.net.blocks",
                "model.net.norm",
                "model.net.fc_norm",
                "model.net.heatmap_tokens",
                "model.net.heatmap_head",
                "model.head",
            )
            unexpected_missing = [
                key
                for key in result.missing_keys
                if not key.startswith(tuple(allowed_missing))
            ]
            backbone_missing = [
                key
                for key in result.missing_keys
                if key.startswith(red_flag_missing)
            ]
            if unexpected_missing or backbone_missing:
                raise RuntimeError(
                    "Actor-prompt checkpoint load is missing non-actor weights: "
                    f"{unexpected_missing or backbone_missing}"
                )
        return result

    def forward(
        self,
        x,
        boxes=None,
        valid=None,
        action_labels=None,
    ):
        # Forward function that is run when visualizing the graph
        return self.model(
            x,
            boxes=boxes,
            valid=valid,
            action_labels=action_labels,
        )

    def _actor_prompt_param_name(self, name):
        return name.startswith(
            (
                "actor_token",
                "actor_slot_embed",
                "valid_embed",
                "bbox_mlp",
            )
        )

    def _interaction_unfrozen_backbone_params(self):
        if not self.actor_interaction_heatmaps:
            return []
        count = int(self.model.hparams.get("interaction_unfreeze_last_blocks", 0) or 0)
        if count <= 0:
            return []
        blocks = getattr(self.model.net, "blocks", None)
        if blocks is None:
            raise ValueError("interaction_unfreeze_last_blocks requires model.net.blocks")
        if count > len(blocks):
            raise ValueError(
                f"interaction_unfreeze_last_blocks={count} exceeds depth={len(blocks)}"
            )
        params = []
        for block in blocks[-count:]:
            params.extend(block.parameters())
        if getattr(self.model.net, "fc_norm", None) is not None:
            params.extend(self.model.net.fc_norm.parameters())
        if getattr(self.model.net, "norm", None) is not None:
            params.extend(self.model.net.norm.parameters())
        return params

    def _head_params(self):
        if not self.actor_prompt:
            return list(self.model.head.parameters())

        if self.actor_interaction_heatmaps and self.model.hparams.get(
            "interaction_warmup_freeze_actor_path", 0
        ):
            return []

        params = list(self.model.actor_head.parameters())
        if self.model.presence_head is not None:
            params += list(self.model.presence_head.parameters())
        for name, param in self.model.net.named_parameters():
            if self._actor_prompt_param_name(name):
                params.append(param)
        return params

    def _backbone_params(self, include_heatmap_head=True):
        params = []
        for name, param in self.model.net.named_parameters():
            if self.actor_prompt and self._actor_prompt_param_name(name):
                continue
            if not include_heatmap_head and "heatmap_head" in name:
                continue
            params.append(param)
        return params

    def _nash_shared_parameters(self):
        params = []
        for name, param in self.model.net.named_parameters():
            if not param.requires_grad:
                continue
            if "heatmap_head" in name:
                continue
            params.append(param)
        if not params:
            raise RuntimeError(
                "Nash-MTL requires trainable shared backbone parameters. "
                "Unfreeze at least one transformer block or disable --grad_weights."
            )
        return params

    def _toyota_label_dict(self):
        return CS_DICT if self.model.hparams.get("task_type", "CS") == "CS" else CV_DICT

    def _build_group_indices(self):
        label_dict = self._toyota_label_dict()
        groups = {}
        for group_name, action_names in GROUPS.items():
            indices = [
                int(label_dict[action_name]) - 1
                for action_name in action_names
                if action_name in label_dict
            ]
            if indices:
                groups[group_name] = torch.tensor(indices, dtype=torch.long)
        return groups

    def _interaction_audit_action_indices(self):
        label_dict = self._toyota_label_dict()
        action_names = (
            "Uselaptop",
            "Readbook",
            "WatchTV",
            "Usetelephone",
            "Drink.Frombottle",
            "Drink.Fromcup",
            "Drink.Fromglass",
            "Pour.Frombottle",
            "Cutbread",
            "Cook.Cut",
            "Cook.Stir",
            "Cook.Cleandishes",
            "Cook.Usestove",
        )
        indices = []
        for action_name in action_names:
            if action_name not in label_dict:
                continue
            metric_name = action_name.replace(".", "_")
            indices.append((metric_name, int(label_dict[action_name]) - 1))
        return indices

    def _pose_heatmap_pred(self, hm_preds):
        if not torch.is_tensor(hm_preds):
            return hm_preds
        n_pose = int(self.model.hparams.n_landmarks)
        if n_pose <= 0:
            return hm_preds[:, :0]
        return hm_preds[:, :n_pose]

    def _interaction_heatmap_pred(self, hm_preds):
        if not torch.is_tensor(hm_preds) or not self.actor_interaction_heatmaps:
            return None
        n_pose = int(self.model.hparams.n_landmarks)
        n_actor = int(self.model.hparams.get("num_actor_tokens", 8))
        end = n_pose + n_actor
        if hm_preds.shape[1] < end:
            raise RuntimeError(
                "Interaction heatmaps require heatmap channels "
                f"[pose={n_pose} + actors={n_actor}], got {hm_preds.shape[1]}"
            )
        return hm_preds[:, n_pose:end]

    def _log_interaction_heatmap_metrics(
        self,
        pred_heatmap,
        target_heatmap,
        heatmap_valid,
        stage,
    ):
        if pred_heatmap is None:
            return
        heatmap_valid = heatmap_valid.to(
            device=pred_heatmap.device,
            dtype=torch.bool,
        )
        if not heatmap_valid.any():
            return

        pred = pred_heatmap.float().clamp(0.0, 1.0)[heatmap_valid]
        target = target_heatmap.to(device=pred_heatmap.device).float()[heatmap_valid]
        pred_bin = pred > 0.3
        target_bin = target > 0.3
        target_visible = target_bin.flatten(1).any(dim=1)
        if not target_visible.any():
            return

        pred = pred[target_visible]
        target = target[target_visible]
        pred_bin = pred_bin[target_visible]
        target_bin = target_bin[target_visible]
        count = int(target_visible.sum().item())

        pred_max = pred.flatten(1).max(dim=1).values
        target_max = target.flatten(1).max(dim=1).values
        self._log_scalar(
            f"{stage}_interaction_heatmap_pred_max",
            pred_max.mean(),
            count,
        )
        self._log_scalar(
            f"{stage}_interaction_heatmap_target_max",
            target_max.mean(),
            count,
        )

        soft_intersection = torch.minimum(pred, target).flatten(1).sum(dim=1)
        soft_union = torch.maximum(pred, target).flatten(1).sum(dim=1)
        valid_soft_union = soft_union > 0
        if valid_soft_union.any():
            self._log_scalar(
                f"{stage}_interaction_heatmap_soft_iou",
                (
                    soft_intersection[valid_soft_union]
                    / soft_union[valid_soft_union].clamp_min(1e-6)
                ).mean(),
                count,
            )

        intersection = (pred_bin & target_bin).float().flatten(1).sum(dim=1)
        union = (pred_bin | target_bin).float().flatten(1).sum(dim=1)
        valid_union = union > 0
        if valid_union.any():
            self._log_scalar(
                f"{stage}_interaction_heatmap_iou",
                (intersection[valid_union] / union[valid_union].clamp_min(1.0)).mean(),
                count,
            )

        positive_values = pred.masked_select(target_bin)
        if positive_values.numel() > 0:
            self._log_scalar(
                f"{stage}_interaction_heatmap_positive_mean",
                positive_values.mean(),
                count,
            )

        pred_flat = pred.flatten(1)
        target_flat = target.flatten(1)
        pred_idx = pred_flat.argmax(dim=1)
        target_idx = target_flat.argmax(dim=1)
        width = pred.shape[-1]
        pred_xy = torch.stack(
            [pred_idx % width, torch.div(pred_idx, width, rounding_mode="floor")],
            dim=-1,
        ).float()
        target_xy = torch.stack(
            [
                target_idx % width,
                torch.div(target_idx, width, rounding_mode="floor"),
            ],
            dim=-1,
        ).float()
        self._log_scalar(
            f"{stage}_interaction_heatmap_center_l2",
            torch.linalg.norm(pred_xy - target_xy, dim=-1).mean(),
            count,
        )

    def _log_group_metrics(self, prefix, preds, labels):
        if not self.group_indices:
            return
        pred_labels = preds.argmax(dim=-1)
        for group_name, group_idx in self.group_indices.items():
            group_idx = group_idx.to(device=labels.device)
            mask = torch.isin(labels, group_idx)
            if not mask.any():
                continue
            metric_name = (
                prefix.format(group=group_name)
                if "{group}" in prefix
                else f"{prefix}_{group_name}_acc"
            )
            self._log_scalar(
                metric_name,
                (pred_labels[mask] == labels[mask]).float().mean(),
                mask.sum().item(),
            )

    def _log_action_metrics(self, prefix, preds, labels):
        pred_labels = preds.argmax(dim=-1)
        for metric_name, action_idx in self._interaction_audit_action_indices():
            mask = labels == int(action_idx)
            if not mask.any():
                continue
            self._log_scalar(
                prefix.format(action=metric_name),
                (pred_labels[mask] == labels[mask]).float().mean(),
                mask.sum().item(),
            )

    def _log_interaction_teacher_metrics(self, actions, valid, heatmap_valid, stage):
        valid = valid.to(device=actions.device, dtype=torch.bool)
        heatmap_valid = heatmap_valid.to(device=actions.device, dtype=torch.bool) & valid
        valid_count = int(valid.sum().item())
        if valid_count <= 0:
            return

        self._log_scalar(
            f"{stage}_interaction_teacher_slot_rate",
            heatmap_valid.float().sum() / max(valid_count, 1),
            valid_count,
        )
        self.log(
            f"{stage}_interaction_teacher_slot_count",
            heatmap_valid.float().sum(),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
            reduce_fx="sum",
            batch_size=1,
        )

        for metric_name, action_idx in self._interaction_audit_action_indices():
            mask = valid & (actions == int(action_idx))
            if not mask.any():
                continue
            self._log_scalar(
                f"{stage}_action_{metric_name}_interaction_teacher_rate",
                heatmap_valid[mask].float().mean(),
                int(mask.sum().item()),
            )

    def _append_nash_mtl_params(self, params):
        if not (
            self.model.hparams.grad_weights
            and (self.model.hparams.n_landmarks > 0 or self.actor_interaction_heatmaps)
        ):
            return
        if NashMTL is None:
            raise ImportError("cvxpy is required when grad_weights is enabled")
        self.grad_weight = NashMTL(
            n_tasks=2,
            update_weights_every=int(
                self.model.hparams.get("nash_update_weights_every", 20)
            ),
            max_norm=float(self.model.hparams.get("nash_max_norm", 1.0)),
            device=self.model.device,
        )
        nash_params = list(self.grad_weight.parameters())
        if nash_params:
            params.append(
                {
                    "params": nash_params,
                    "lr": 0.025,
                },
            )

    def configure_optimizers(self):
        # We will support Adam or SGD as optimizers.
        if self.model.hparams.freeze_backbone:
            head_params = self._head_params()
            params = []
            if head_params:
                params.append(
                    {
                        "params": head_params,
                        "lr": self.lr_head,
                        "weight_decay": self.weight_decay_head,
                    }
                )
            unfrozen_backbone_params = self._interaction_unfrozen_backbone_params()
            if unfrozen_backbone_params:
                params.append(
                    {
                        "params": unfrozen_backbone_params,
                        "lr": self.lr,
                        "weight_decay": self.weight_decay,
                    }
                )
            if (
                self.actor_interaction_heatmaps
                and getattr(self.model.hparams, "lr_head_hm", None)
                and self.model.hparams.lr_head_hm
            ):
                params.append(
                    {
                        "params": self.model.net.heatmap_head.parameters(),
                        "lr": self.model.hparams.lr_head_hm,
                        "weight_decay": self.model.hparams.weight_decay_head_hm,
                    },
                )
            if not params:
                raise ValueError(
                    "No trainable parameters selected. For interaction warmup, set "
                    "--lr_head_hm > 0 or --interaction_unfreeze_last_blocks > 0."
                )
            self._append_nash_mtl_params(params)
            optimizer = optim.AdamW(params)
        else:
            if (
                getattr(self.model.hparams, "lr_head_hm", None)
                and self.model.hparams.lr_head_hm
            ):
                # Python
                head_params = self._head_params()
                params = [
                    {
                        "params": self._backbone_params(include_heatmap_head=False),
                        "lr": self.lr,
                        "weight_decay": self.weight_decay,
                    },
                    {
                        "params": head_params,
                        "lr": self.lr_head,
                        "weight_decay": self.weight_decay_head,
                    },
                ]
                params.append(
                    {
                        "params": self.model.net.heatmap_head.parameters(),
                        "lr": self.model.hparams.lr_head_hm,
                        "weight_decay": self.model.hparams.weight_decay_head_hm,
                    },
                )
            else:
                head_params = self._head_params()
                params = [
                    {
                        "params": self._backbone_params(include_heatmap_head=True),
                        "lr": self.lr,
                        "weight_decay": self.weight_decay,
                    },
                    {
                        "params": head_params,
                        "lr": self.lr_head,
                        "weight_decay": self.weight_decay_head,
                    },
                ]
            # if lr_patch_embed exists add to params
            if getattr(self, "lr_patch_embed", None):
                params.append(
                    {
                        "params": self.model.patch_embed.parameters(),
                        "lr": self.lr_patch_embed,
                        "weight_decay": self.weight_decay_patch_embed,
                    },
                )

            self._append_nash_mtl_params(params)
            if self.model.hparams.deepspeed_optim:
                if DeepSpeedCPUAdam is None:
                    raise ImportError(
                        "deepspeed is required when deepspeed_optim is enabled"
                    )
                optimizer = DeepSpeedCPUAdam(params)
            else:
                optimizer = optim.AdamW(params)

        if self.model.hparams.warm_restarts:
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=self.t_max_scheduler
            )
        else:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.t_max_scheduler
            )

        # CosineAnnealingWarmRestarts
        return [optimizer], [scheduler]

    def backward(self, loss):
        if (
            self.model.hparams.grad_weights
            and torch.is_tensor(loss)
            and loss.ndim > 0
        ):
            if loss.numel() != 2:
                raise RuntimeError(
                    f"Nash-MTL expects two tasks [action, heatmap], got {loss.numel()}"
                )
            _, extra_outputs = self.grad_weight.backward(
                losses=loss,
                shared_parameters=self._nash_shared_parameters(),
            )
            weights = None if extra_outputs is None else extra_outputs.get("weights")
            if torch.is_tensor(weights) and weights.numel() == 2:
                self.log(
                    "train_nash_weight_action",
                    weights[0].detach(),
                    on_step=True,
                    on_epoch=True,
                    logger=True,
                    sync_dist=True,
                )
                self.log(
                    "train_nash_weight_heatmap",
                    weights[1].detach(),
                    on_step=True,
                    on_epoch=True,
                    logger=True,
                    sync_dist=True,
                )
        else:
            loss.backward()

    def _actor_step(self, imgs, target, loss_fn, stage):
        actions = target["actions"].long()
        boxes = target["boxes"].float()
        valid = target["valid"].bool()
        if not valid.any():
            raise ValueError(f"{stage} actor batch has no valid actor slots")

        data = self.model(
            imgs,
            boxes=boxes,
            valid=valid,
            action_labels=actions,
        )
        preds, hm_preds, presence_logits = self._unpack_model_data(data)

        valid_preds = preds[valid]
        valid_labels = actions[valid]
        loss_action = loss_fn(valid_preds, valid_labels)
        self.log(
            f"{stage}_loss_action",
            loss_action,
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )

        loss_action_task = loss_action
        if presence_logits is not None:
            loss_presence = F.binary_cross_entropy_with_logits(
                presence_logits, valid.float()
            )
            loss_action_task = loss_action_task + loss_presence * self.model.hparams.get(
                "presence_loss_weight", 0.05
            )
            self.log(
                f"{stage}_loss_presence",
                loss_presence,
                on_step=stage == "train",
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )

        loss_kp = None
        loss_pose_frobenius = None
        if self.model.hparams.n_landmarks > 0:
            labels_kp = target["heatmap"]
            kp_vis = target["kp_vis"]
            pose_hm_preds = self._pose_heatmap_pred(hm_preds)
            if stage == "train" and self.model.hparams.target_kp_loss_weight:
                target_weights = self.target_weights.expand(labels_kp.shape[0], -1).to(
                    labels_kp.device
                )
                loss_kp = self.kp_loss(
                    pose_hm_preds,
                    labels_kp,
                    mask=kp_vis,
                    target_weights=target_weights,
                )
            else:
                loss_kp = self.kp_loss(pose_hm_preds, labels_kp, mask=kp_vis)

            loss_pose_frobenius = heatmap_frobenius_loss(
                pose_hm_preds,
                labels_kp.to(device=pose_hm_preds.device, dtype=pose_hm_preds.dtype),
                mask=kp_vis.to(device=pose_hm_preds.device),
            )
            self.log(
                f"{stage}_loss_pose_heatmap_frobenius",
                loss_pose_frobenius,
                on_step=stage == "train",
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )

            if (
                not self.actor_poguiseplus_loss
                and stage == "train"
                and self.model.hparams.log_kp_loss_weight
            ):
                loss_kp = torch.log(loss_kp + 1e-6)

        loss_interaction_frobenius = None
        if self.actor_interaction_heatmaps:
            interaction_heatmap = self._interaction_heatmap_pred(hm_preds)
            if (
                interaction_heatmap is not None
                and "interaction_heatmap" in target
                and "interaction_heatmap_valid" in target
            ):
                heatmap_valid = target["interaction_heatmap_valid"].to(
                    device=valid.device, dtype=torch.bool
                ) & valid
                self._log_interaction_teacher_metrics(
                    actions,
                    valid,
                    heatmap_valid,
                    stage,
                )
                target_heatmap = target["interaction_heatmap"].to(
                    device=interaction_heatmap.device,
                    dtype=interaction_heatmap.dtype,
                )
                loss_interaction_heatmap = interaction_heatmap_loss(
                    interaction_heatmap,
                    target_heatmap,
                    heatmap_valid,
                )
                if loss_interaction_heatmap is None:
                    loss_interaction_heatmap = interaction_heatmap.new_zeros(())
                loss_interaction_frobenius = heatmap_frobenius_loss(
                    interaction_heatmap,
                    target_heatmap,
                    valid=heatmap_valid,
                )
                self.log(
                    f"{stage}_loss_interaction_heatmap",
                    loss_interaction_heatmap,
                    on_step=stage == "train",
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                    sync_dist=True,
                )
                self.log(
                    f"{stage}_loss_interaction_heatmap_frobenius",
                    loss_interaction_frobenius,
                    on_step=stage == "train",
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                    sync_dist=True,
                )
                if stage != "train":
                    self._log_interaction_heatmap_metrics(
                        interaction_heatmap,
                        target_heatmap,
                        heatmap_valid,
                        stage,
                    )

        if self.actor_poguiseplus_loss:
            if loss_pose_frobenius is None and loss_interaction_frobenius is None:
                raise RuntimeError(
                    "Actor PO-GUISE+ loss requires pose or interaction heatmaps"
                )
            heatmap_terms = []
            if loss_pose_frobenius is not None:
                heatmap_terms.append(loss_pose_frobenius)
            if loss_interaction_frobenius is not None:
                heatmap_terms.append(loss_interaction_frobenius)
            loss_heatmap_raw = torch.stack(heatmap_terms).sum()
            loss_heatmap_task = torch.log(
                loss_heatmap_raw + self.poguiseplus_heatmap_log_eps
            )
            self.log(
                f"{stage}_loss_heatmap_frobenius",
                loss_heatmap_raw,
                on_step=stage == "train",
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )
            self.log(
                f"{stage}_loss_heatmap_log",
                loss_heatmap_task,
                on_step=stage == "train",
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )
            if stage == "train" and self.model.hparams.grad_weights:
                loss = torch.stack([loss_action_task, loss_heatmap_task])
            elif self.model.hparams.get("kp_only", False):
                loss = loss_action_task * 1e-6 + (
                    loss_heatmap_task * self.poguiseplus_heatmap_loss_weight
                )
            else:
                loss = loss_action_task + (
                    loss_heatmap_task * self.poguiseplus_heatmap_loss_weight
                )
        else:
            loss = loss_action_task
            if loss_kp is not None:
                kp_loss_weight = float(self.model.hparams.kp_loss_weight)
                if self.model.hparams.get("kp_only", False):
                    loss = loss * 1e-6 + loss_kp * kp_loss_weight
                elif stage == "train" and self.model.hparams.grad_weights:
                    loss = torch.stack([loss, loss_kp * kp_loss_weight])
                elif kp_loss_weight > 0.0:
                    loss = loss + loss_kp * kp_loss_weight

        return (
            loss,
            valid_preds,
            valid_labels,
            hm_preds,
            loss_kp,
            preds,
            presence_logits,
        )

    def _unpack_model_data(self, data):
        if len(data) == 3:
            preds, hm_preds, presence_logits = data
        else:
            preds, hm_preds = data
            presence_logits = None
        return preds, hm_preds, presence_logits

    def _first_actor_targets(self, imgs, target):
        valid = target["valid"].bool()
        has_actor = valid.any(dim=1)
        if not has_actor.any():
            return None

        sample_idx = torch.nonzero(has_actor, as_tuple=False).flatten()
        slot_idx = valid.long().argmax(dim=1)[sample_idx]
        boxes = target["boxes"].float()[sample_idx, slot_idx]
        labels = target["actions"].long()[sample_idx, slot_idx]
        return imgs[sample_idx], boxes, labels

    def _log_scalar(self, name, value, batch_size, prog_bar=False):
        self.log(
            name,
            value,
            on_step=False,
            on_epoch=True,
            prog_bar=prog_bar,
            logger=True,
            sync_dist=True,
            batch_size=int(batch_size),
        )

    def _logger_run_name(self):
        logger = getattr(self.trainer, "logger", None)
        if logger is None:
            return "default"

        experiment = getattr(logger, "experiment", None)
        run_id = getattr(experiment, "id", None)
        if isinstance(run_id, str) and run_id:
            return run_id

        version = getattr(logger, "version", None)
        if version is not None:
            return f"version_{version}"

        name = getattr(logger, "name", None)
        if isinstance(name, str) and name:
            return name

        return "default"

    def _log_actor_presence_diagnostics(self, presence_logits, valid):
        if presence_logits is None:
            return
        probs = torch.sigmoid(presence_logits.float())
        presence_pred = probs >= 0.5
        self._log_scalar(
            "val_actor_presence_acc",
            (presence_pred == valid).float().mean(),
            valid.numel(),
        )
        if valid.any():
            self._log_scalar(
                "val_actor_presence_pos",
                probs[valid].mean(),
                valid.sum().item(),
            )
        invalid = ~valid
        if invalid.any():
            self._log_scalar(
                "val_actor_presence_empty",
                probs[invalid].mean(),
                invalid.sum().item(),
            )

    def _fixed_background_boxes(self, batch_size, num_actor_tokens, device, dtype):
        base = torch.tensor(
            [
                [0.02, 0.02, 0.34, 0.34],
                [0.66, 0.02, 0.98, 0.34],
                [0.02, 0.66, 0.34, 0.98],
                [0.66, 0.66, 0.98, 0.98],
                [0.25, 0.25, 0.75, 0.75],
                [0.00, 0.20, 0.30, 0.80],
                [0.70, 0.20, 1.00, 0.80],
                [0.20, 0.00, 0.80, 0.30],
            ],
            device=device,
            dtype=dtype,
        )
        if num_actor_tokens > base.shape[0]:
            repeats = (num_actor_tokens + base.shape[0] - 1) // base.shape[0]
            base = base.repeat(repeats, 1)
        base = base[:num_actor_tokens]
        return base.unsqueeze(0).expand(batch_size, -1, -1).clone()

    def _log_actor_background_presence(self, imgs, boxes):
        if getattr(self.model, "presence_head", None) is None:
            return
        batch_size = imgs.shape[0]
        num_actor_tokens = int(self.model.hparams.get("num_actor_tokens", 8))
        if batch_size == 0 or num_actor_tokens < 2:
            return

        diag_boxes = self._fixed_background_boxes(
            batch_size, num_actor_tokens, boxes.device, boxes.dtype
        )
        diag_valid = torch.zeros(
            batch_size, num_actor_tokens, device=boxes.device, dtype=torch.bool
        )
        diag_boxes[:, 0] = boxes
        diag_valid[:, 0] = True
        data = self.model(imgs, boxes=diag_boxes, valid=diag_valid)
        _, _, presence_logits = self._unpack_model_data(data)
        if presence_logits is None:
            return
        probs = torch.sigmoid(presence_logits.float())
        bg_probs = probs[:, 1:]
        self._log_scalar(
            "val_actor_presence_bg",
            bg_probs.mean(),
            bg_probs.numel(),
        )
        self._log_scalar(
            "val_actor_presence_bg_acc",
            (bg_probs < 0.5).float().mean(),
            bg_probs.numel(),
        )

    def _log_actor_all_slot_diagnostics(self, imgs, boxes, labels):
        batch_size = imgs.shape[0]
        num_actor_tokens = int(self.model.hparams.get("num_actor_tokens", 8))
        if batch_size == 0 or num_actor_tokens <= 0:
            return

        diag_boxes = boxes[:, None, :].expand(-1, num_actor_tokens, -1).clone()
        diag_valid = torch.ones(
            batch_size, num_actor_tokens, device=boxes.device, dtype=torch.bool
        )
        data = self.model(imgs, boxes=diag_boxes, valid=diag_valid)
        preds, _, _ = self._unpack_model_data(data)
        pred_labels = preds.argmax(dim=-1)
        expanded_labels = labels[:, None].expand(-1, num_actor_tokens)
        slot_correct = (pred_labels == expanded_labels).float()

        self._log_scalar(
            "val_actor_all_slot_acc",
            slot_correct.mean(),
            slot_correct.numel(),
            prog_bar=True,
        )
        slot_consistency = (pred_labels == pred_labels[:, :1]).float().mean()
        self._log_scalar(
            "val_actor_slot_consistency",
            slot_consistency,
            slot_correct.numel(),
        )
        for slot in range(num_actor_tokens):
            self._log_scalar(
                f"val_actor_slot{slot}_acc",
                slot_correct[:, slot].mean(),
                batch_size,
            )

    def _compose_pair_batch(self, imgs, boxes, labels, same_action=False, swap=False):
        batch_size, n_frames, channels, height, width = imgs.shape
        num_actor_tokens = int(self.model.hparams.get("num_actor_tokens", 8))
        if batch_size == 0 or num_actor_tokens < 2:
            return None

        if same_action:
            pair_count = min(batch_size, self.actor_val_diagnostic_max_pairs)
            left_idx = torch.arange(pair_count, device=imgs.device)
            right_idx = left_idx
        else:
            pair_count = min(batch_size // 2, self.actor_val_diagnostic_max_pairs)
            if pair_count == 0:
                return None
            left_idx = torch.arange(pair_count, device=imgs.device)
            right_idx = torch.arange(
                batch_size - pair_count, batch_size, device=imgs.device
            )

        split = width // 2
        left_frames = imgs[left_idx].reshape(
            pair_count * n_frames, channels, height, width
        )
        right_frames = imgs[right_idx].reshape(
            pair_count * n_frames, channels, height, width
        )
        left_panel = F.interpolate(
            left_frames,
            size=(height, split),
            mode="bilinear",
            align_corners=False,
        ).reshape(pair_count, n_frames, channels, height, split)
        right_panel = F.interpolate(
            right_frames,
            size=(height, width - split),
            mode="bilinear",
            align_corners=False,
        ).reshape(pair_count, n_frames, channels, height, width - split)

        canvas = torch.zeros(
            pair_count,
            n_frames,
            channels,
            height,
            width,
            device=imgs.device,
            dtype=imgs.dtype,
        )
        canvas[:, :, :, :, :split] = left_panel
        canvas[:, :, :, :, split:] = right_panel

        left_boxes = boxes[left_idx].clone()
        right_boxes = boxes[right_idx].clone()
        left_boxes[:, [0, 2]] *= split / float(width)
        right_boxes[:, [0, 2]] = split / float(width) + right_boxes[:, [0, 2]] * (
            (width - split) / float(width)
        )

        diag_boxes = torch.zeros(
            pair_count,
            num_actor_tokens,
            4,
            device=imgs.device,
            dtype=boxes.dtype,
        )
        diag_valid = torch.zeros(
            pair_count, num_actor_tokens, device=imgs.device, dtype=torch.bool
        )
        diag_labels = torch.stack([labels[left_idx], labels[right_idx]], dim=1)
        if swap:
            diag_boxes[:, 0] = right_boxes
            diag_boxes[:, 1] = left_boxes
            diag_labels = diag_labels.flip(dims=[1])
        else:
            diag_boxes[:, 0] = left_boxes
            diag_boxes[:, 1] = right_boxes
        diag_valid[:, :2] = True
        return canvas, diag_boxes, diag_valid, diag_labels

    def _log_actor_pair_diagnostics(self, imgs, boxes, labels):
        for name, same_action, swap in (
            ("val_actor_pair_acc", False, False),
            ("val_actor_pair_swap_acc", False, True),
            ("val_actor_pair_same_acc", True, False),
        ):
            batch = self._compose_pair_batch(
                imgs, boxes, labels, same_action=same_action, swap=swap
            )
            if batch is None:
                continue
            pair_imgs, pair_boxes, pair_valid, pair_labels = batch
            data = self.model(pair_imgs, boxes=pair_boxes, valid=pair_valid)
            preds, _, _ = self._unpack_model_data(data)
            pair_preds = preds[:, :2].argmax(dim=-1)
            correct = (pair_preds == pair_labels).float()
            self._log_scalar(
                name,
                correct.mean(),
                correct.numel(),
                prog_bar=name.endswith("acc"),
            )

            if name == "val_actor_pair_acc":
                diff_mask = pair_labels[:, 0] != pair_labels[:, 1]
                if diff_mask.any():
                    self._log_scalar(
                        "val_actor_pair_diff_acc",
                        correct[diff_mask].mean(),
                        correct[diff_mask].numel(),
                    )

    def _log_actor_val_diagnostics(self, imgs, target, full_presence_logits):
        if not self.actor_val_diagnostics:
            return
        valid = target["valid"].bool()
        self._log_actor_presence_diagnostics(full_presence_logits, valid)

        first_targets = self._first_actor_targets(imgs, target)
        if first_targets is None:
            return
        diag_imgs, diag_boxes, diag_labels = first_targets
        self._log_actor_all_slot_diagnostics(diag_imgs, diag_boxes, diag_labels)
        self._log_actor_pair_diagnostics(diag_imgs, diag_boxes, diag_labels)
        self._log_actor_background_presence(diag_imgs, diag_boxes)

    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        if isinstance(batch, torch._utils.ExceptionWrapper):
            batch.reraise()
        if self.actor_prompt:
            imgs, target = batch
            loss, _, _, _, loss_kp, _, _ = self._actor_step(
                imgs, target, self.train_loss, "train"
            )
            loss_to_log = loss.sum() if loss.ndim > 0 else loss
            self.log(
                "train_loss",
                loss_to_log,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )
            if loss_kp is not None:
                self.log(
                    "train_loss_kp",
                    loss_kp,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                    sync_dist=True,
                )
            return loss
        imgs, labels = batch
        if self.model.hparams.get("linear_probe", False):
            filename = "tmp/y_{}_train.pickle"
            # create the directory if it does not exist
            if not os.path.exists("tmp"):
                os.makedirs("tmp")
            # check if the file exists
            i = 0
            while os.path.isfile(filename.format(i)):
                i += 1
            with open(filename.format(i), "wb") as f:
                pickle.dump(labels, f)
        if self.model.hparams.n_landmarks > 0:
            labels_kp = labels[1]
            kp_vis = labels[2]
            labels = labels[0]

            if labels_kp.ndim == 5:
                labels_kp = labels_kp[:, labels_kp.shape[1] // 2, ...]
                kp_vis = kp_vis[:, kp_vis.shape[1] // 2, :]
        preds, hm_preds = self.model(imgs)
        loss = self.train_loss(preds, labels)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        if self.model.hparams.n_landmarks > 0:
            if self.model.hparams.target_kp_loss_weight:
                # expand target weights to match batch size
                target_weights = self.target_weights.expand(labels_kp.shape[0], -1).to(
                    labels_kp.device
                )
                loss_kp = self.kp_loss(
                    hm_preds, labels_kp, mask=kp_vis, target_weights=target_weights
                )
            else:
                loss_kp = self.kp_loss(hm_preds, labels_kp, mask=kp_vis)

            if self.model.hparams.log_kp_loss_weight:
                loss_kp = torch.log(loss_kp + 1e-6)
            kp_loss_weight = float(self.model.hparams.kp_loss_weight)
            if self.model.hparams.get("kp_only", False):
                loss = loss * 1e-6 + loss_kp * kp_loss_weight
            elif self.model.hparams.grad_weights:
                loss = torch.stack([loss, loss_kp * kp_loss_weight])
            elif kp_loss_weight > 0.0:
                loss = loss + loss_kp * kp_loss_weight

            self.log(
                "train_loss_kp",
                loss_kp,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )

        return loss

    def validation_step(self, batch, batch_idx):
        if isinstance(batch, torch._utils.ExceptionWrapper):
            batch.reraise()
        if self.actor_prompt:
            imgs, target = batch
            loss, preds, labels, hm, loss_kp, _, presence_logits = self._actor_step(
                imgs, target, self.val_loss, "val"
            )
            self._log_actor_val_diagnostics(imgs, target, presence_logits)
            if loss_kp is not None:
                pose_hm = self._pose_heatmap_pred(hm)
                self.val_mae(pose_hm.contiguous(), target["heatmap"].contiguous())
                self.log(
                    "val_loss_kp",
                    loss_kp,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                    sync_dist=True,
                )
                self.log(
                    "val_mae",
                    self.val_mae,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                    sync_dist=True,
                )

            self.val_acc_micro(preds, labels)
            self.val_acc_macro(preds, labels)
            self.val_f1(preds, labels)
            self._log_group_metrics("val_group_{group}_acc", preds, labels)
            self._log_action_metrics("val_action_{action}_acc", preds, labels)
            self.validation_step_outputs["preds"].append(preds.detach())
            self.validation_step_outputs["labels"].append(labels.detach())
            self.log(
                "val_loss",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )
            self.log(
                "val_acc_micro",
                self.val_acc_micro,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )
            self.log(
                "val_acc_macro",
                self.val_acc_macro,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )
            self.log(
                "val_f1",
                self.val_f1,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )
            return
        imgs, labels = batch

        if self.model.hparams.get("linear_probe", False):
            filename = "tmp/y_{}_val.pickle"
            # create the directory if it does not exist
            if not os.path.exists("tmp"):
                os.makedirs("tmp")
            # check if the file exists
            i = 0
            while os.path.isfile(filename.format(i)):
                i += 1
            with open(filename.format(i), "wb") as f:
                pickle.dump(labels, f)

        preds, hm = self.model(imgs)
        if self.model.hparams.n_landmarks > 0:
            labels_kp = labels[1]
            kp_vis = labels[2]
            labels = labels[0]
        loss = self.val_loss(preds, labels)
        if self.model.hparams.n_landmarks > 0:
            loss_kp = self.val_loss_kp(hm, labels_kp, mask=kp_vis)
            self.val_mae(hm.contiguous(), labels_kp.contiguous())
            self.log(
                "val_loss_kp",
                loss_kp,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )
            self.log(
                "val_mae",
                self.val_mae,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )
            if self.model.hparams.get("kp_only", False):
                loss = loss * 1e-6 + loss_kp

        # calculate metrics
        self.val_acc_micro(preds, labels)
        self.val_acc_macro(preds, labels)
        self.val_f1(preds, labels)

        # save preds and labels for later
        self.validation_step_outputs["preds"].append(preds.detach())
        self.validation_step_outputs["labels"].append(labels.detach())
        # Add sync_dist=True to sync logging across all GPU workers (may have performance impact)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "val_acc_micro",
            self.val_acc_micro,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "val_acc_macro",
            self.val_acc_macro,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "val_f1",
            self.val_f1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

    def on_validation_epoch_end(self) -> None:
        # log best val loss
        # TODO check if best_val_loss is correctly called on multi gpu i.e. if it is synced across all gpus
        best_val_loss = self.trainer.callback_metrics.get("best_val_loss")
        outputs = self.all_gather(self.validation_step_outputs)
        # merge the outputs
        preds = None
        for data in outputs["preds"]:
            if preds is None:
                preds = torch.Tensor(data).view(-1, self.num_classes)
            else:
                preds = torch.cat(
                    (preds, torch.Tensor(data).view(-1, self.num_classes))
                )
        labels = None
        for data in outputs["labels"]:
            if labels is None:
                labels = torch.Tensor(data).view(-1)
            else:
                labels = torch.cat((labels, torch.Tensor(data).view(-1)))

        loss = self.val_loss(preds, labels)
        # save prediction and labels for to csv

        if best_val_loss is None or loss < best_val_loss:
            self.log(
                "best_val_loss",
                loss,
                prog_bar=True,
                logger=True,
                sync_dist=False,
                rank_zero_only=True,
            )
            self.best_val_acc_micro(preds, labels)
            self.best_val_acc_macro(preds, labels)
            self.best_val_f1(preds, labels)
            # log best val acc
            self.log(
                "best_val_acc_micro",
                self.best_val_acc_micro,
                prog_bar=True,
                logger=True,
                sync_dist=False,
                rank_zero_only=True,
            )
            self.log(
                "best_val_acc_macro",
                self.best_val_acc_macro,
                prog_bar=True,
                logger=True,
                sync_dist=False,
                rank_zero_only=True,
            )
            self.log(
                "best_val_f1",
                self.best_val_f1,
                prog_bar=True,
                logger=True,
                sync_dist=False,
                rank_zero_only=True,
            )
            if self.trainer.global_rank == 0:
                run_name = self._logger_run_name()
                if not os.path.exists(self.trainer.default_root_dir):
                    os.makedirs(self.trainer.default_root_dir)
                # get class predictions
                preds = preds.argmax(dim=-1)
                pd.DataFrame(preds.cpu().numpy()).to_csv(
                    os.path.join(
                        self.trainer.default_root_dir,
                        run_name + "_preds.csv",
                    ),
                    index=False,
                )
                pd.DataFrame(labels.cpu().numpy()).to_csv(
                    os.path.join(
                        self.trainer.default_root_dir,
                        run_name + "_labels.csv",
                    ),
                    index=False,
                )
        self.validation_step_outputs.clear()
        self.validation_step_outputs = {"preds": [], "labels": []}
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return super().on_validation_epoch_end()

    def test_step(self, batch, batch_idx):
        imgs, labels = batch[0], batch[1]
        if self.actor_prompt:
            target = labels
            actions = target["actions"].long()
            boxes = target["boxes"].float()
            valid = target["valid"].bool()
            data = self.model(imgs, boxes=boxes, valid=valid)
            preds, hm, _ = self._unpack_model_data(data)
            preds = preds.float()

            if self.model.hparams.n_landmarks > 0 and "heatmap" in target:
                labels_kp = target["heatmap"]
                pose_hm = self._pose_heatmap_pred(hm)
                self.test_mae(pose_hm.contiguous(), labels_kp.contiguous())
                self.log(
                    "test_mae",
                    self.test_mae,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                    sync_dist=True,
                )

            ids = batch[2]
            chunk_nb = batch[3]
            split_nb = batch[4]
            final_result = []
            for i in range(preds.size(0)):
                valid_slots = valid[i].nonzero(as_tuple=False).flatten()
                for slot in valid_slots.tolist():
                    final_result.append(
                        {
                            "id": f"{ids[i]}:actor{slot}",
                            "preds": preds.data[i, slot].cpu().numpy().tolist(),
                            "labels": int(actions[i, slot].cpu().numpy()),
                            "chunk_nb": int(chunk_nb[i].cpu().numpy()),
                            "split_nb": int(split_nb[i].cpu().numpy()),
                            "actor_slot": int(slot),
                            "box": boxes[i, slot].cpu().numpy().tolist(),
                        }
                    )

            df = pd.DataFrame(final_result)
            dest_path = os.path.join(self.trainer.default_root_dir, self._logger_run_name())
            if not os.path.exists(dest_path):
                os.makedirs(dest_path)
            csv_file = os.path.join(dest_path, "test_results.csv")
            if not os.path.isfile(csv_file):
                print(csv_file)
                df.to_csv(csv_file, index=False)
            else:
                df.to_csv(csv_file, mode="a", header=False, index=False)
            return
        if self.model.hparams.get("linear_probe", False):
            filename = "tmp/y_{}_test.pickle"
            # create the directory if it does not exist
            if not os.path.exists("tmp"):
                os.makedirs("tmp")
            # check if the file exists
            i = 0
            while os.path.isfile(filename.format(i)):
                i += 1
            with open(filename.format(i), "wb") as f:
                pickle.dump(labels, f)
        data = self.model(imgs)  # only use class preds
        preds, hm, _ = self._unpack_model_data(data)
        # convert preds from bfloat16 to float32
        preds = preds.float()
        if len(batch) == 6:
            labels_kp = batch[5]
            self.test_mae(hm.contiguous(), labels_kp.contiguous())
            self.log(
                "test_mae",
                self.test_mae,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )

        ids = batch[2]
        chunk_nb = batch[3]
        split_nb = batch[4]
        final_result = []
        for i in range(preds.size(0)):
            row = {
                "id": ids[i],
                "preds": preds.data[i].cpu().numpy().tolist(),
                "labels": int(labels[i].cpu().numpy()),
                "chunk_nb": int(chunk_nb[i].cpu().numpy()),
                "split_nb": int(split_nb[i].cpu().numpy()),
            }
            if not isinstance(hm, int):
                row["mae"] = torchmetrics.functional.mean_absolute_error(
                    hm.contiguous(), labels_kp.contiguous()
                )
            final_result.append(row)

        df = pd.DataFrame(final_result)

        dest_path = os.path.join(self.trainer.default_root_dir, self._logger_run_name())
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)
        csv_file = os.path.join(dest_path, "test_results.csv")

        # If file does not exist, write with header
        if not os.path.isfile(csv_file):
            print(csv_file)
            df.to_csv(csv_file, index=False)
        else:  # else it exists so append without writing the header
            df.to_csv(csv_file, mode="a", header=False, index=False)
        # access lightning datamodule
