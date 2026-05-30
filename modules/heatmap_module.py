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
import pickle
from datasets.object_vocab import GROUPS, OBJECT_SPECIALIST_GROUPS, OBJECT_TO_ID
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
        self.object_prompt = bool(hparams.get("object_prompt", 0))
        self.num_object_classes = int(hparams.get("num_object_classes", 0) or 0)
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
        self.object_specialist_specs = (
            self._build_object_specialist_specs()
            if self.object_prompt
            and bool(hparams.get("object_specialist_heads", 0))
            else {}
        )
        if self.object_prompt:
            self.val_acc_macro_objects_on = torchmetrics.Accuracy(
                task="multiclass", num_classes=hparams.num_classes, average="macro"
            )
            self.val_f1_objects_on = torchmetrics.F1Score(
                num_classes=hparams.num_classes, average="macro", task="multiclass"
            )
            self.val_acc_macro_objects_off = torchmetrics.Accuracy(
                task="multiclass", num_classes=hparams.num_classes, average="macro"
            )
            self.val_f1_objects_off = torchmetrics.F1Score(
                num_classes=hparams.num_classes, average="macro", task="multiclass"
            )
            self.val_acc_macro_objects_shuffled = torchmetrics.Accuracy(
                task="multiclass", num_classes=hparams.num_classes, average="macro"
            )
            self.val_f1_objects_shuffled = torchmetrics.F1Score(
                num_classes=hparams.num_classes, average="macro", task="multiclass"
            )

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
                "model.object_cls_embed",
                "model.object_bbox_mlp",
                "model.object_conf_mlp",
                "model.object_visual_mlp",
                "model.object_valid_embed",
                "model.object_action_embed",
                "model.object_action_query",
                "model.object_action_key",
                "model.object_action_value",
                "model.object_action_geom_bias",
                "model.object_action_delta_norm",
                "model.object_action_delta",
                "model.object_specialist_base_mlp",
                "model.object_specialist_delta_norms",
                "model.object_specialist_delta_heads",
                "model.object_relation_gate_logit",
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
        object_boxes=None,
        object_cls=None,
        object_conf=None,
        object_valid=None,
    ):
        # Forward function that is run when visualizing the graph
        return self.model(
            x,
            boxes=boxes,
            valid=valid,
            action_labels=action_labels,
            object_boxes=object_boxes,
            object_cls=object_cls,
            object_conf=object_conf,
            object_valid=object_valid,
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

    def _object_relation_params(self):
        params = []
        for module in self.model._object_relation_modules():
            params.extend(module.parameters())
        params.append(self.model.object_relation_gate_logit)
        return params

    def _head_params(self):
        if not self.actor_prompt:
            return list(self.model.head.parameters())

        if self.object_prompt and self.model.hparams.get("object_relation_only", 0):
            return self._object_relation_params()

        params = list(self.model.actor_head.parameters())
        if self.model.presence_head is not None:
            params += list(self.model.presence_head.parameters())
        if self.object_prompt:
            params += self._object_relation_params()
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

    def _build_object_specialist_specs(self):
        label_dict = self._toyota_label_dict()
        specs = {}
        for group_name, group in OBJECT_SPECIALIST_GROUPS.items():
            action_indices = [
                int(label_dict[action_name]) - 1
                for action_name in group["actions"]
                if action_name in label_dict
            ]
            object_indices = [
                int(OBJECT_TO_ID[object_name])
                for object_name in group["objects"]
                if object_name in OBJECT_TO_ID
            ]
            if len(action_indices) < 2 or not object_indices:
                continue
            specs[group_name] = {
                "actions": torch.tensor(action_indices, dtype=torch.long),
                "objects": torch.tensor(object_indices, dtype=torch.long),
            }
        return specs

    def _use_specialist_only_relation_loss(self):
        return (
            self.object_prompt
            and bool(self.model.hparams.get("object_relation_only", 0))
            and bool(self.model.hparams.get("object_specialist_heads", 0))
        )

    def _object_kwargs(self, target, mode="on"):
        if not self.object_prompt:
            return {}
        required = (
            "object_boxes",
            "object_cls",
            "object_conf",
            "object_valid",
        )
        missing = [key for key in required if key not in target]
        if missing:
            raise ValueError(f"object_prompt target is missing keys: {missing}")

        kwargs = {
            "object_boxes": target["object_boxes"].float(),
            "object_cls": target["object_cls"].long(),
            "object_conf": target["object_conf"].float(),
            "object_valid": target["object_valid"].bool(),
        }
        if mode == "on":
            kwargs["object_valid"] = self.model._apply_object_dropout(
                kwargs["object_valid"]
            )
            return kwargs
        if mode == "off":
            kwargs["object_valid"] = torch.zeros_like(kwargs["object_valid"])
            return kwargs
        if mode == "positive_erased":
            if "interaction_positive_mask" not in target:
                raise ValueError(
                    "positive_erased object mode requires interaction_positive_mask"
                )
            positive_mask = target["interaction_positive_mask"].to(
                device=kwargs["object_valid"].device,
                dtype=torch.bool,
            )
            actor_valid = target["valid"].to(
                device=kwargs["object_valid"].device,
                dtype=torch.bool,
            )
            real_object_count = min(
                kwargs["object_valid"].shape[-1],
                positive_mask.shape[-1] - 1,
            )
            erase_mask = (
                positive_mask[..., :real_object_count]
                & actor_valid[..., None]
            ).any(dim=1)
            kwargs["object_valid"] = kwargs["object_valid"].clone()
            kwargs["object_valid"][..., :real_object_count] = (
                kwargs["object_valid"][..., :real_object_count]
                & ~erase_mask
            )
            return kwargs
        if mode == "shuffled":
            batch_size = kwargs["object_valid"].shape[0]
            if batch_size < 2:
                return None
            perm = self._label_mismatched_object_indices(
                target,
                batch_size,
                kwargs["object_valid"].device,
            )
            if perm is None:
                return None
            return {
                key: value.index_select(0, perm.to(device=value.device))
                for key, value in kwargs.items()
            }
        raise ValueError(f"Unsupported object mode: {mode}")

    def _primary_action_labels(self, target):
        actions = target["actions"].long()
        valid = target["valid"].bool()
        has_actor = valid.any(dim=1)
        first_slot = valid.long().argmax(dim=1)
        labels = actions[
            torch.arange(actions.shape[0], device=actions.device),
            first_slot,
        ]
        return labels.masked_fill(~has_actor, -1)

    def _label_mismatched_object_indices(self, target, batch_size, device):
        labels = self._primary_action_labels(target).to(device=device)
        if labels.shape[0] != batch_size:
            raise ValueError(
                "Object shuffle batch size does not match target labels: "
                f"{batch_size} vs {labels.shape[0]}"
            )
        indices = torch.arange(batch_size, device=device)
        valid_label = labels >= 0
        perm = torch.empty(batch_size, dtype=torch.long, device=device)
        for i in range(batch_size):
            if bool(valid_label[i]):
                candidates = indices[valid_label & (labels != labels[i])]
            else:
                candidates = indices[indices != i]
            if candidates.numel() == 0:
                return None
            label_offset = int(labels[i].item()) if bool(valid_label[i]) else 0
            choice = (i * 9973 + label_offset * 37) % int(candidates.numel())
            perm[i] = candidates[choice]
        return perm

    def _actor_logits_for_object_mode(self, imgs, target, mode):
        object_kwargs = self._object_kwargs(target, mode=mode)
        if object_kwargs is None:
            return None
        data = self.model(
            imgs,
            boxes=target["boxes"].float(),
            valid=target["valid"].bool(),
            **object_kwargs,
        )
        preds, _, _, _, _, _ = self._unpack_model_data(data)
        return preds

    def _object_counterfactual_loss(
        self,
        imgs,
        target,
        normal_logits,
        valid,
        normal_object_valid,
    ):
        valid = valid.to(device=normal_logits.device, dtype=torch.bool)
        margin_weight = float(
            self.model.hparams.get("object_counterfactual_margin_weight", 0.0)
        )
        consistency_weight = float(
            self.model.hparams.get("object_objectless_consistency_weight", 0.0)
        )
        if margin_weight <= 0.0 and consistency_weight <= 0.0:
            return normal_logits.new_zeros(())
        if (
            "interaction_valid" not in target
            or "interaction_positive_mask" not in target
        ):
            return normal_logits.new_zeros(())

        labels = target["actions"].long()
        positive_mask = target["interaction_positive_mask"].to(
            device=normal_logits.device,
            dtype=torch.bool,
        )
        normal_object_valid = normal_object_valid.to(
            device=normal_logits.device,
            dtype=torch.bool,
        )
        real_object_count = min(
            normal_object_valid.shape[-1],
            positive_mask.shape[-1] - 1,
        )
        positive_mask = positive_mask.clone()
        positive_mask[..., :real_object_count] = (
            positive_mask[..., :real_object_count]
            & normal_object_valid[:, None, :real_object_count]
        )
        inter_valid = target["interaction_valid"].to(
            device=normal_logits.device,
            dtype=torch.bool,
        ) & valid
        none_slot = positive_mask.shape[-1] - 1
        object_interaction = inter_valid & positive_mask[..., :none_slot].any(dim=-1)
        none_interaction = inter_valid & positive_mask[..., none_slot]

        normal_valid = normal_logits[valid].float()
        labels_valid = labels.to(device=normal_logits.device)[valid]
        object_mask = object_interaction[valid]
        none_mask = none_interaction[valid]

        off_logits = None
        erased_logits = None
        shuffled_logits = None
        if consistency_weight > 0.0:
            with torch.no_grad():
                off_logits = self._actor_logits_for_object_mode(imgs, target, "off")
        if margin_weight > 0.0:
            erased_logits = self._actor_logits_for_object_mode(
                imgs,
                target,
                "positive_erased",
            )
            shuffled_logits = self._actor_logits_for_object_mode(
                imgs,
                target,
                "shuffled",
            )

        weighted_loss = normal_logits.new_zeros(())
        margin_losses = []
        if margin_weight > 0.0 and object_mask.any():
            normal_object = normal_valid[object_mask]
            labels_object = labels_valid[object_mask]
            normal_true = normal_object.gather(
                1,
                labels_object[:, None],
            ).squeeze(1)
            margin = float(self.model.hparams.get("object_counterfactual_margin", 0.05))

            if erased_logits is not None:
                erased_true = erased_logits[valid].float()[object_mask].gather(
                    1,
                    labels_object[:, None],
                ).squeeze(1)
                margin_losses.append(F.relu(margin - (normal_true - erased_true)))
            if shuffled_logits is not None:
                shuffled_true = shuffled_logits[valid].float()[object_mask].gather(
                    1,
                    labels_object[:, None],
                ).squeeze(1)
                margin_losses.append(F.relu(margin - (normal_true - shuffled_true)))

        if margin_losses:
            loss_margin = torch.cat(margin_losses).mean()
            weighted_loss = weighted_loss + loss_margin * margin_weight
            self.log(
                "train_loss_object_counterfactual_margin",
                loss_margin,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )

        if consistency_weight > 0.0 and none_mask.any() and off_logits is not None:
            normal_none = normal_valid[none_mask]
            off_none = off_logits[valid].float()[none_mask]
            loss_consistency = F.kl_div(
                F.log_softmax(normal_none, dim=-1),
                F.softmax(off_none, dim=-1),
                reduction="batchmean",
            )
            weighted_loss = weighted_loss + loss_consistency * consistency_weight
            self.log(
                "train_loss_objectless_consistency",
                loss_consistency,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )

        return weighted_loss

    def _selection_logits_for_target_action(self, selection_logits, target):
        if selection_logits is None:
            return None
        if selection_logits.ndim != 4:
            raise RuntimeError(
                "Action-conditioned object selection must have shape [B, K, C, M+1]; "
                f"got {tuple(selection_logits.shape)}"
            )
        labels = target["actions"].to(
            device=selection_logits.device,
            dtype=torch.long,
        )
        labels = labels.clamp(0, selection_logits.shape[2] - 1)
        gather_index = labels[:, :, None, None].expand(
            -1,
            -1,
            1,
            selection_logits.shape[-1],
        )
        return selection_logits.gather(dim=2, index=gather_index).squeeze(2)

    def _object_specialist_loss(self, preds, object_residual, target, valid, stage):
        group_weight = float(
            self.model.hparams.get("object_specialist_group_loss_weight", 0.0)
        )
        no_boost_weight = float(
            self.model.hparams.get("object_specialist_no_boost_weight", 0.0)
        )
        if (
            object_residual is None
            or not self.object_specialist_specs
            or (group_weight <= 0.0 and no_boost_weight <= 0.0)
        ):
            return preds.new_zeros(())

        actions = target["actions"].to(device=preds.device, dtype=torch.long)
        object_cls = target["object_cls"].to(device=preds.device, dtype=torch.long)
        object_valid = target["object_valid"].to(device=preds.device, dtype=torch.bool)
        total = preds.new_zeros(())

        group_losses = []
        no_boost_losses = []
        for group_name, spec in self.object_specialist_specs.items():
            action_idx = spec["actions"].to(device=preds.device)
            object_idx = spec["objects"].to(device=preds.device)

            in_group = torch.isin(actions, action_idx) & valid
            if group_weight > 0.0 and in_group.any():
                group_logits = preds.index_select(2, action_idx)[in_group]
                group_labels = actions[in_group]
                group_targets = (
                    group_labels[:, None] == action_idx[None, :]
                ).long().argmax(dim=1)
                group_losses.append(F.cross_entropy(group_logits, group_targets))

            object_visible = (
                torch.isin(object_cls, object_idx) & object_valid
            ).any(dim=1)
            outside_group_visible = (~torch.isin(actions, action_idx)) & valid
            outside_group_visible = outside_group_visible & object_visible[:, None]
            if no_boost_weight > 0.0 and outside_group_visible.any():
                residual_group = object_residual.index_select(2, action_idx)
                max_boost = residual_group[outside_group_visible].max(dim=-1).values
                margin = float(
                    self.model.hparams.get("object_specialist_no_boost_margin", 0.02)
                )
                no_boost_losses.append(F.relu(max_boost - margin).mean())

        if group_losses:
            loss_group = torch.stack(group_losses).mean()
            total = total + loss_group * group_weight
            self.log(
                f"{stage}_loss_object_specialist_group",
                loss_group,
                on_step=stage == "train",
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )
        if no_boost_losses:
            loss_no_boost = torch.stack(no_boost_losses).mean()
            total = total + loss_no_boost * no_boost_weight
            self.log(
                f"{stage}_loss_object_specialist_no_boost",
                loss_no_boost,
                on_step=stage == "train",
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )
        return total

    def _pose_heatmap_pred(self, hm_preds):
        if not torch.is_tensor(hm_preds):
            return hm_preds
        n_pose = int(self.model.hparams.n_landmarks)
        if n_pose <= 0:
            return hm_preds[:, :0]
        return hm_preds[:, :n_pose]

    def _object_heatmap_pred(self, hm_preds):
        if not self.object_prompt or not torch.is_tensor(hm_preds):
            return None
        n_pose = int(self.model.hparams.n_landmarks)
        return hm_preds[:, n_pose : n_pose + self.num_object_classes]

    def _object_heatmap_loss(self, pred_obj, target_obj, object_weight):
        if pred_obj is None:
            return None
        weight = object_weight.to(device=pred_obj.device, dtype=pred_obj.dtype)
        if weight.ndim == 2:
            weight = weight[:, :, None, None]
        if weight.sum() <= 0:
            return pred_obj.new_zeros(())
        loss = (pred_obj - target_obj.to(dtype=pred_obj.dtype)) ** 2
        return (loss * weight).sum() / weight.sum().clamp_min(1.0)

    def _log_object_heatmap_metrics(self, pred_obj, target, stage):
        if pred_obj is None or "object_heatmap" not in target:
            return
        target_obj = target["object_heatmap"].to(device=pred_obj.device)
        if "object_heatmap_valid" in target:
            object_valid = target["object_heatmap_valid"].to(
                device=pred_obj.device, dtype=torch.bool
            )
        else:
            object_valid = target_obj.flatten(2).amax(dim=2) > 0
        if not object_valid.any():
            return

        probs = pred_obj.float().clamp(0.0, 1.0)
        pred_bin = probs > 0.3
        target_bin = target_obj.float() > 0.3
        valid_mask = object_valid[:, :, None, None]
        valid_values = probs.masked_select(valid_mask.expand_as(probs))
        if valid_values.numel() > 0:
            self._log_scalar(
                f"{stage}_obj_pred_max",
                valid_values.max(),
                object_valid.sum().item(),
            )
            self._log_scalar(
                f"{stage}_obj_pred_mean",
                valid_values.mean(),
                object_valid.sum().item(),
            )

        intersection = (pred_bin & target_bin & valid_mask).float().sum()
        union = ((pred_bin | target_bin) & valid_mask).float().sum()
        if union > 0:
            self._log_scalar(
                f"{stage}_obj_iou",
                intersection / union.clamp_min(1.0),
                object_valid.sum().item(),
            )

        visible = target_bin.flatten(2).any(dim=2) & object_valid
        if visible.any():
            positive_values = probs.masked_select(target_bin & valid_mask)
            if positive_values.numel() > 0:
                self._log_scalar(
                    f"{stage}_obj_pred_positive_mean",
                    positive_values.mean(),
                    visible.sum().item(),
                )
            pred_visible = pred_bin.flatten(2).any(dim=2) & object_valid
            self._log_scalar(
                f"{stage}_obj_recall_visible",
                (pred_visible & visible).float().sum() / visible.float().sum(),
                visible.sum().item(),
            )
            pred_flat = probs.flatten(2)
            target_flat = target_obj.float().flatten(2)
            pred_idx = pred_flat.argmax(dim=2)
            target_idx = target_flat.argmax(dim=2)
            width = pred_obj.shape[-1]
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
            center_l2 = torch.linalg.norm(pred_xy - target_xy, dim=-1)
            self._log_scalar(
                f"{stage}_obj_center_l2",
                center_l2[visible].mean(),
                visible.sum().item(),
            )

    def _log_interaction_metrics(self, selection_logits, target, valid, stage):
        if (
            selection_logits is None
            or "interaction_valid" not in target
            or "interaction_positive_mask" not in target
        ):
            return
        valid_count = valid.sum().clamp_min(1)
        inter_valid = target["interaction_valid"].to(
            device=valid.device, dtype=torch.bool
        ) & valid
        self._log_scalar(
            f"{stage}_interaction_target_frac",
            inter_valid.float().sum() / valid_count,
            int(valid_count.item()),
        )
        if not inter_valid.any():
            return
        positive_mask = target["interaction_positive_mask"].to(
            device=selection_logits.device, dtype=torch.bool
        )
        inter_valid = inter_valid & positive_mask.any(dim=-1)
        self._log_scalar(
            f"{stage}_interaction_target_usable_frac",
            inter_valid.float().sum() / valid_count,
            int(valid_count.item()),
        )
        if not inter_valid.any():
            return
        pred = selection_logits.argmax(dim=-1)
        correct = positive_mask.gather(dim=-1, index=pred.unsqueeze(-1)).squeeze(-1)
        self._log_scalar(
            f"{stage}_interaction_select_acc",
            correct.float().mean(),
            inter_valid.sum().item(),
        )
        probs = torch.softmax(selection_logits.float(), dim=-1)
        select_mass = (probs * positive_mask.float()).sum(dim=-1)
        self._log_scalar(
            f"{stage}_interaction_select_mass",
            select_mass[inter_valid].mean(),
            inter_valid.sum().item(),
        )
        none_slot = positive_mask.shape[-1] - 1
        none_mask = inter_valid & positive_mask[..., none_slot]
        object_mask = inter_valid & ~positive_mask[..., none_slot]
        if object_mask.any():
            self._log_scalar(
                f"{stage}_interaction_select_acc_object",
                correct[object_mask].float().mean(),
                object_mask.sum().item(),
            )
            self._log_scalar(
                f"{stage}_interaction_select_mass_object",
                select_mass[object_mask].mean(),
                object_mask.sum().item(),
            )
            self._log_scalar(
                f"{stage}_interaction_target_object_frac",
                object_mask.float().sum() / valid_count,
                int(valid_count.item()),
            )
        if none_mask.any():
            self._log_scalar(
                f"{stage}_interaction_select_acc_none",
                correct[none_mask].float().mean(),
                none_mask.sum().item(),
            )
            self._log_scalar(
                f"{stage}_interaction_select_mass_none",
                select_mass[none_mask].mean(),
                none_mask.sum().item(),
            )
            self._log_scalar(
                f"{stage}_interaction_target_none_frac",
                none_mask.float().sum() / valid_count,
                int(valid_count.item()),
            )

    def _log_object_relation_gate(self, stage):
        if not (
            self.object_prompt
            and hasattr(self.model, "object_relation_gate_logit")
        ):
            return
        gate = torch.sigmoid(self.model.object_relation_gate_logit.detach())
        gate_mean = gate.mean()
        self.log(
            f"{stage}_object_relation_gate",
            gate_mean,
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        if gate.ndim > 0 and stage != "train":
            self._log_scalar(
                f"{stage}_object_relation_gate_min",
                gate.min(),
                gate.numel(),
            )
            self._log_scalar(
                f"{stage}_object_relation_gate_max",
                gate.max(),
                gate.numel(),
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

    def configure_optimizers(self):
        # We will support Adam or SGD as optimizers.
        if self.model.hparams.freeze_backbone:
            head_params = self._head_params()
            params = [
                {
                    "params": head_params,
                    "lr": self.lr_head,
                    "weight_decay": self.weight_decay_head,
                },
            ]
            if (
                self.object_prompt
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

            if self.model.hparams.grad_weights and self.model.hparams.n_landmarks > 0:
                if NashMTL is None:
                    raise ImportError("cvxpy is required when grad_weights is enabled")
                self.grad_weight = NashMTL(
                    n_tasks=2,
                    update_weights_every=20,
                    device=self.model.device,
                )
                params.append(
                    {
                        "params": self.grad_weight.parameters(),
                        "lr": 0.025,
                        # "weight_decay": self.weight_decay,
                    },
                )
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
        if self.model.hparams.grad_weights and self.model.hparams.n_landmarks > 0:
            losses = [loss[0], loss[1]]
            action_head_params = (
                self.model.actor_head.parameters()
                if self.actor_prompt
                else self.model.head.parameters()
            )
            lst = [
                action_head_params,
                self.model.net.heatmap_head.parameters(),
            ]
            self.grad_weight.backward(
                losses=losses,
                shared_parameters=[
                    param
                    for name, param in self.model.net.named_parameters()
                    if "heatmap_head" not in name
                ],
                task_specific_parameters=lst,
            )
        else:
            loss.backward()

    def _actor_step(self, imgs, target, loss_fn, stage):
        actions = target["actions"].long()
        boxes = target["boxes"].float()
        valid = target["valid"].bool()
        if not valid.any():
            raise ValueError(f"{stage} actor batch has no valid actor slots")

        object_kwargs = self._object_kwargs(target, mode="on")
        data = self.model(
            imgs,
            boxes=boxes,
            valid=valid,
            action_labels=actions,
            **object_kwargs,
        )
        (
            preds,
            hm_preds,
            presence_logits,
            selection_logits,
            interaction_heatmap,
            object_residual,
        ) = self._unpack_model_data(data)
        target_action_selection_logits = self._selection_logits_for_target_action(
            selection_logits,
            target,
        )

        valid_preds = preds[valid]
        valid_labels = actions[valid]
        loss_action = loss_fn(valid_preds, valid_labels)
        if self._use_specialist_only_relation_loss():
            loss = valid_preds.sum() * 0.0
        else:
            loss = loss_action
        self.log(
            f"{stage}_loss_action",
            loss_action,
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )

        if presence_logits is not None:
            loss_presence = F.binary_cross_entropy_with_logits(
                presence_logits, valid.float()
            )
            loss = loss + loss_presence * self.model.hparams.get(
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

            if stage == "train" and self.model.hparams.log_kp_loss_weight:
                loss_kp = torch.log(loss_kp + 1e-6)
            kp_loss_weight = float(self.model.hparams.kp_loss_weight)
            if self.model.hparams.get("kp_only", False):
                loss = loss * 1e-6 + loss_kp * kp_loss_weight
            elif stage == "train" and self.model.hparams.grad_weights:
                loss = torch.stack([loss, loss_kp * kp_loss_weight])
            elif kp_loss_weight > 0.0:
                loss = loss + loss_kp * kp_loss_weight

        if self.object_prompt:
            self._log_object_relation_gate(stage)
            pred_obj = self._object_heatmap_pred(hm_preds)
            object_heatmap_weight = float(
                self.model.hparams.get("object_heatmap_weight", 200.0)
            )
            if (
                object_heatmap_weight > 0.0
                and pred_obj is not None
                and "object_heatmap" in target
            ):
                target_obj = target["object_heatmap"].to(device=pred_obj.device)
                if "object_heatmap_weight" in target:
                    object_weight = target["object_heatmap_weight"].to(
                        device=pred_obj.device
                    )
                else:
                    object_weight = target["object_heatmap_valid"].to(
                        device=pred_obj.device, dtype=torch.float32
                    )
                loss_obj = self._object_heatmap_loss(
                    pred_obj,
                    target_obj,
                    object_weight,
                )
                if loss_obj is not None:
                    loss = loss + loss_obj * object_heatmap_weight
                    self.log(
                        f"{stage}_obj_heatmap_loss",
                        loss_obj,
                        on_step=stage == "train",
                        on_epoch=True,
                        prog_bar=False,
                        logger=True,
                        sync_dist=True,
                    )
                    if stage != "train":
                        self._log_object_heatmap_metrics(pred_obj, target, stage)

            dropped_positive = None
            if (
                target_action_selection_logits is not None
                and "interaction_positive_mask" in target
            ):
                inter_valid = target["interaction_valid"].to(
                    device=valid.device, dtype=torch.bool
                ) & valid
                original_positive_mask = target["interaction_positive_mask"].to(
                    device=target_action_selection_logits.device, dtype=torch.bool
                )
                positive_mask = original_positive_mask.clone()
                effective_object_valid = object_kwargs["object_valid"].to(
                    device=target_action_selection_logits.device, dtype=torch.bool
                )
                real_object_count = min(
                    effective_object_valid.shape[-1],
                    positive_mask.shape[-1] - 1,
                )
                positive_mask[..., :real_object_count] = (
                    positive_mask[..., :real_object_count]
                    & effective_object_valid[:, None, :real_object_count]
                )
                dropped_positive = (
                    original_positive_mask[..., :real_object_count]
                    & ~effective_object_valid[:, None, :real_object_count]
                ).any(dim=-1)
                if inter_valid.any():
                    inter_valid = inter_valid & positive_mask.any(dim=-1)
                if inter_valid.any():
                    log_probs = F.log_softmax(
                        target_action_selection_logits.float(),
                        dim=-1,
                    )
                    valid_log_probs = log_probs[inter_valid]
                    valid_positive_mask = positive_mask[inter_valid]
                    mask_value = torch.finfo(valid_log_probs.dtype).min
                    selected_log_prob = torch.logsumexp(
                        valid_log_probs.masked_fill(~valid_positive_mask, mask_value),
                        dim=-1,
                    )
                    loss_interaction = -selected_log_prob.mean()
                else:
                    loss_interaction = target_action_selection_logits.new_zeros(())
                loss = loss + loss_interaction * self.model.hparams.get(
                    "object_interaction_loss_weight", 0.05
                )
                self.log(
                    f"{stage}_loss_interaction",
                    loss_interaction,
                    on_step=stage == "train",
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                    sync_dist=True,
                )
                if stage != "train":
                    self._log_interaction_metrics(
                        target_action_selection_logits,
                        target,
                        valid,
                        stage,
                    )

            if (
                interaction_heatmap is not None
                and "interaction_heatmap" in target
                and "interaction_heatmap_valid" in target
            ):
                heatmap_valid = target["interaction_heatmap_valid"].to(
                    device=valid.device, dtype=torch.bool
                ) & valid
                if dropped_positive is not None:
                    heatmap_valid = heatmap_valid & ~dropped_positive.to(
                        device=heatmap_valid.device
                    )
                if heatmap_valid.any():
                    target_heatmap = target["interaction_heatmap"].to(
                        device=interaction_heatmap.device,
                        dtype=interaction_heatmap.dtype,
                    )
                    if target_heatmap.shape[-2:] != interaction_heatmap.shape[-2:]:
                        raise RuntimeError(
                            "interaction_heatmap target/prediction size mismatch: "
                            f"{target_heatmap.shape[-2:]} vs "
                            f"{interaction_heatmap.shape[-2:]}"
                        )
                    loss_interaction_heatmap = F.mse_loss(
                        interaction_heatmap[heatmap_valid],
                        target_heatmap[heatmap_valid],
                    )
                else:
                    loss_interaction_heatmap = interaction_heatmap.new_zeros(())
                loss = loss + loss_interaction_heatmap * self.model.hparams.get(
                    "object_interaction_heatmap_weight", 0.0
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

            residual_weight = float(
                self.model.hparams.get("object_residual_l2_weight", 0.0)
            )
            if object_residual is not None and residual_weight > 0.0:
                residual_valid = object_residual[valid].float()
                loss_residual = residual_valid.square().mean()
                loss = loss + loss_residual * residual_weight
                self.log(
                    f"{stage}_loss_object_residual_l2",
                    loss_residual,
                    on_step=stage == "train",
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                    sync_dist=True,
                )

            if self.model.hparams.get("object_specialist_heads", 0):
                loss = loss + self._object_specialist_loss(
                    preds,
                    object_residual,
                    target,
                    valid,
                    stage,
                )

            if stage == "train":
                loss = loss + self._object_counterfactual_loss(
                    imgs,
                    target,
                    preds,
                    valid,
                    object_kwargs["object_valid"],
                )

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
        if len(data) == 6:
            (
                preds,
                hm_preds,
                presence_logits,
                selection_logits,
                interaction_heatmap,
                object_residual,
            ) = data
        elif len(data) == 5:
            (
                preds,
                hm_preds,
                presence_logits,
                selection_logits,
                interaction_heatmap,
            ) = data
            object_residual = None
        elif len(data) == 4:
            preds, hm_preds, presence_logits, selection_logits = data
            interaction_heatmap = None
            object_residual = None
        elif len(data) == 3:
            preds, hm_preds, presence_logits = data
            selection_logits = None
            interaction_heatmap = None
            object_residual = None
        else:
            preds, hm_preds = data
            presence_logits = None
            selection_logits = None
            interaction_heatmap = None
            object_residual = None
        return (
            preds,
            hm_preds,
            presence_logits,
            selection_logits,
            interaction_heatmap,
            object_residual,
        )

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
        _, _, presence_logits, _, _, _ = self._unpack_model_data(data)
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
        preds, _, _, _, _, _ = self._unpack_model_data(data)
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
            preds, _, _, _, _, _ = self._unpack_model_data(data)
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

    def _actor_preds_for_object_mode(self, imgs, target, mode):
        object_kwargs = self._object_kwargs(target, mode=mode)
        if object_kwargs is None:
            return None
        data = self.model(
            imgs,
            boxes=target["boxes"].float(),
            valid=target["valid"].bool(),
            **object_kwargs,
        )
        preds, _, _, _, _, _ = self._unpack_model_data(data)
        valid = target["valid"].bool()
        labels = target["actions"].long()
        return preds[valid], labels[valid]

    def _classification_margin(self, logits, labels):
        true_logits = logits.gather(1, labels[:, None]).squeeze(1)
        other_logits = logits.clone()
        other_logits.scatter_(1, labels[:, None], -torch.inf)
        return true_logits - other_logits.max(dim=1).values

    def _group_classification_margin(self, logits, labels, group_idx):
        group_logits = logits.index_select(1, group_idx)
        group_targets = (labels[:, None] == group_idx[None, :]).long().argmax(dim=1)
        true_logits = group_logits.gather(1, group_targets[:, None]).squeeze(1)
        other_logits = group_logits.clone()
        other_logits.scatter_(1, group_targets[:, None], -torch.inf)
        return true_logits - other_logits.max(dim=1).values

    def _log_specialist_ablation_metrics(self, normal_logits, labels, off, shuffled):
        if not self.object_specialist_specs or labels.numel() == 0:
            return

        comparisons = []
        if off is not None:
            comparisons.append(("off", off[0], off[1]))
        if shuffled is not None:
            comparisons.append(("shuffled", shuffled[0], shuffled[1]))
        if not comparisons:
            return

        for group_name, spec in self.object_specialist_specs.items():
            group_idx = spec["actions"].to(device=labels.device)
            group_mask = torch.isin(labels, group_idx)
            if not group_mask.any():
                continue

            normal_group = normal_logits[group_mask]
            labels_group = labels[group_mask]
            normal_margin = self._group_classification_margin(
                normal_group,
                labels_group,
                group_idx,
            )
            normal_full = normal_logits[group_mask]
            normal_pred = normal_full.argmax(dim=1)
            normal_correct = normal_pred == labels_group
            self._log_scalar(
                f"val_{group_name}_objects_on_group_margin",
                normal_margin.mean(),
                labels_group.numel(),
            )

            for mode_name, mode_logits, mode_labels in comparisons:
                if mode_logits.shape != normal_logits.shape or not torch.equal(
                    mode_labels,
                    labels,
                ):
                    raise ValueError(
                        f"Object {mode_name} specialist diagnostic returned "
                        "misaligned logits/labels"
                    )
                mode_group = mode_logits[group_mask]
                mode_margin = self._group_classification_margin(
                    mode_group,
                    labels_group,
                    group_idx,
                )
                mode_full = mode_logits[group_mask]
                mode_pred = mode_full.argmax(dim=1)
                mode_correct = mode_pred == labels_group
                changed = normal_pred != mode_pred
                correct_change = changed & normal_correct & ~mode_correct
                wrong_change = changed & ~normal_correct & mode_correct

                self._log_scalar(
                    f"val_{group_name}_margin_gain_on_vs_{mode_name}",
                    (normal_margin - mode_margin).mean(),
                    labels_group.numel(),
                )
                self._log_scalar(
                    f"val_{group_name}_pred_changed_on_vs_{mode_name}",
                    changed.float().mean(),
                    labels_group.numel(),
                )
                self._log_scalar(
                    f"val_{group_name}_correct_changes_on_vs_{mode_name}",
                    correct_change.float().mean(),
                    labels_group.numel(),
                )
                self._log_scalar(
                    f"val_{group_name}_wrong_changes_on_vs_{mode_name}",
                    wrong_change.float().mean(),
                    labels_group.numel(),
                )

    def _log_object_delta_metrics(
        self,
        normal_logits,
        labels,
        positive_erased=None,
        off=None,
        shuffled=None,
    ):
        if labels.numel() == 0:
            return

        normal_true = normal_logits.gather(1, labels[:, None]).squeeze(1)
        normal_margin = self._classification_margin(normal_logits, labels)

        for mode_name, mode_data in (
            ("positive_erased", positive_erased),
            ("off", off),
            ("shuffled", shuffled),
        ):
            if mode_data is None:
                continue
            mode_logits, mode_labels = mode_data
            if mode_logits.shape != normal_logits.shape or not torch.equal(
                mode_labels,
                labels,
            ):
                raise ValueError(
                    f"Object {mode_name} diagnostic returned misaligned logits/labels"
                )

            mode_true = mode_logits.gather(1, labels[:, None]).squeeze(1)
            mode_margin = self._classification_margin(mode_logits, labels)
            pred_changed = normal_logits.argmax(dim=1) != mode_logits.argmax(dim=1)
            self._log_scalar(
                f"val_object_delta_l1_on_vs_{mode_name}",
                (normal_logits - mode_logits).abs().mean(),
                normal_logits.numel(),
            )
            self._log_scalar(
                f"val_object_true_logit_gain_on_vs_{mode_name}",
                (normal_true - mode_true).mean(),
                labels.numel(),
            )
            self._log_scalar(
                f"val_object_margin_gain_on_vs_{mode_name}",
                (normal_margin - mode_margin).mean(),
                labels.numel(),
            )
            self._log_scalar(
                f"val_object_pred_changed_on_vs_{mode_name}",
                pred_changed.float().mean(),
                labels.numel(),
            )

    def _log_object_interaction_causal_metrics(
        self,
        normal_logits,
        labels,
        target,
        positive_erased=None,
        shuffled=None,
    ):
        if labels.numel() == 0 or "interaction_positive_mask" not in target:
            return

        valid = target["valid"].to(device=normal_logits.device, dtype=torch.bool)
        positive_mask = target["interaction_positive_mask"].to(
            device=normal_logits.device,
            dtype=torch.bool,
        )
        inter_valid = target["interaction_valid"].to(
            device=normal_logits.device,
            dtype=torch.bool,
        ) & valid
        none_slot = positive_mask.shape[-1] - 1
        object_interaction = (
            inter_valid
            & positive_mask[..., :none_slot].any(dim=-1)
        )
        object_mask = object_interaction[valid]
        if not object_mask.any():
            return

        normal_object = normal_logits[object_mask]
        labels_object = labels[object_mask]
        normal_true = normal_object.gather(
            1,
            labels_object[:, None],
        ).squeeze(1)
        normal_margin = self._classification_margin(normal_object, labels_object)

        for mode_name, mode_data in (
            ("positive_erased", positive_erased),
            ("shuffled", shuffled),
        ):
            if mode_data is None:
                continue
            mode_logits, mode_labels = mode_data
            if mode_logits.shape != normal_logits.shape or not torch.equal(
                mode_labels,
                labels,
            ):
                raise ValueError(
                    f"Object {mode_name} causal diagnostic returned misaligned "
                    "logits/labels"
                )
            mode_object = mode_logits[object_mask]
            mode_true = mode_object.gather(
                1,
                labels_object[:, None],
            ).squeeze(1)
            mode_margin = self._classification_margin(mode_object, labels_object)
            self._log_scalar(
                f"val_object_interaction_true_logit_gain_on_vs_{mode_name}",
                (normal_true - mode_true).mean(),
                labels_object.numel(),
            )
            self._log_scalar(
                f"val_object_interaction_margin_gain_on_vs_{mode_name}",
                (normal_margin - mode_margin).mean(),
                labels_object.numel(),
            )

    def _log_object_ablation_eval(self, imgs, target, normal_preds, labels):
        if not self.object_prompt:
            return

        self.val_acc_macro_objects_on(normal_preds, labels)
        self.val_f1_objects_on(normal_preds, labels)
        self.log(
            "val_acc_macro_objects_on",
            self.val_acc_macro_objects_on,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "val_f1_objects_on",
            self.val_f1_objects_on,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        self._log_group_metrics("val_{group}_objects_on", normal_preds, labels)

        positive_erased = self._actor_preds_for_object_mode(
            imgs,
            target,
            "positive_erased",
        )
        if positive_erased is not None:
            erased_preds, erased_labels = positive_erased
            self._log_group_metrics(
                "val_{group}_objects_positive_erased",
                erased_preds,
                erased_labels,
            )
        else:
            erased_preds = None
            erased_labels = None

        off = self._actor_preds_for_object_mode(imgs, target, "off")
        if off is not None:
            off_preds, off_labels = off
            self.val_acc_macro_objects_off(off_preds, off_labels)
            self.val_f1_objects_off(off_preds, off_labels)
            self.log(
                "val_acc_macro_objects_off",
                self.val_acc_macro_objects_off,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )
            self.log(
                "val_f1_objects_off",
                self.val_f1_objects_off,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )
            self._log_group_metrics("val_{group}_objects_off", off_preds, off_labels)
        else:
            off_preds = None
            off_labels = None

        shuffled = self._actor_preds_for_object_mode(imgs, target, "shuffled")
        if shuffled is not None:
            shuffled_preds, shuffled_labels = shuffled
            self.val_acc_macro_objects_shuffled(shuffled_preds, shuffled_labels)
            self.val_f1_objects_shuffled(shuffled_preds, shuffled_labels)
            self.log(
                "val_acc_macro_objects_shuffled",
                self.val_acc_macro_objects_shuffled,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )
            self.log(
                "val_f1_objects_shuffled",
                self.val_f1_objects_shuffled,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )
            self._log_group_metrics(
                "val_{group}_objects_shuffled",
                shuffled_preds,
                shuffled_labels,
            )
        else:
            shuffled_preds = None
            shuffled_labels = None

        off_data = None if off_preds is None else (off_preds, off_labels)
        erased_data = (
            None if erased_preds is None else (erased_preds, erased_labels)
        )
        shuffled_data = (
            None if shuffled_preds is None else (shuffled_preds, shuffled_labels)
        )
        self._log_object_delta_metrics(
            normal_preds,
            labels,
            positive_erased=erased_data,
            off=off_data,
            shuffled=shuffled_data,
        )
        self._log_object_interaction_causal_metrics(
            normal_preds,
            labels,
            target,
            positive_erased=erased_data,
            shuffled=shuffled_data,
        )
        self._log_specialist_ablation_metrics(
            normal_preds,
            labels,
            off_data,
            shuffled_data,
        )

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
            self._log_object_ablation_eval(imgs, target, preds, labels)
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
            object_kwargs = self._object_kwargs(target, mode="on")
            data = self.model(imgs, boxes=boxes, valid=valid, **object_kwargs)
            preds, hm, _, _, _, _ = self._unpack_model_data(data)
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
        preds, hm, _, _, _, _ = self._unpack_model_data(data)
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
