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
from losses.poguiseplus_losses import (
    heatmap_frobenius_loss,
    heatmap_mse_loss,
)
import pickle
from datasets.object_vocab import (
    NUM_OBJECT_CLASSES,
    OBJECT_CLASSES,
    OBJECT_TO_ID,
)
from datasets.toyota_action_taxonomy import (
    toyota_action_names,
    toyota_confuser_action_names,
    toyota_action_object_map,
    toyota_action_to_index,
    toyota_group_action_names,
    toyota_label_dict,
    toyota_objectless_action_names,
)
from models.actor_object_action_query_decoder import (
    ActionObjectQuerySpec,
    action_slot_target_loss,
)

try:
    from grad_weights.nash_mtl import NashMTL
except ImportError:
    NashMTL = None

try:
    from deepspeed.ops.adam import DeepSpeedCPUAdam
except ImportError:
    DeepSpeedCPUAdam = None


DEFAULT_MOTION_AUX_LOSS_WEIGHT = 0.25
DEFAULT_OBJECT_ACTION_CONFUSER_LOSS_WEIGHT = 0.5
DEPLOY_KEY_ACTIONS = (
    "Uselaptop",
    "Readbook",
    "Walk",
    "Getup",
    "Sitdown",
    "Laydown",
)


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
        if bool(hparams.get("scene_object_tokens", 0)):
            raise ValueError(
                "scene_object_tokens was removed. Use actor_object_slot_head=1."
            )
        self.actor_object_slot_head = bool(hparams.get("actor_object_slot_head", 0))
        self.uses_object_proposals = self.actor_object_slot_head
        if self.actor_object_slot_head and not self.actor_prompt:
            raise ValueError("actor_object_slot_head requires actor_prompt")
        self.actor_poguiseplus_loss = self.actor_prompt and self.actor_interaction_heatmaps
        self.poguiseplus_heatmap_loss_weight = float(
            hparams.get("poguiseplus_heatmap_loss_weight", 1.0)
        )
        self.poguiseplus_pose_heatmap_weight = float(
            hparams.get("poguiseplus_pose_heatmap_weight", 1.0)
        )
        self.poguiseplus_interaction_heatmap_weight = float(
            hparams.get("poguiseplus_interaction_heatmap_weight", 1.0)
        )
        self.poguiseplus_heatmap_log_eps = float(
            hparams.get("poguiseplus_heatmap_log_eps", 1e-6)
        )
        self.poguiseplus_normalized_heatmap_loss = bool(
            hparams.get("poguiseplus_normalized_heatmap_loss", 0)
        )
        self.poguiseplus_heatmap_mse_scale = float(
            hparams.get("poguiseplus_heatmap_mse_scale", 1000.0)
        )
        if self.poguiseplus_heatmap_mse_scale <= 0:
            raise ValueError("poguiseplus_heatmap_mse_scale must be positive")
        self.motion_aux_loss_weight = float(
            hparams.get("motion_aux_loss_weight", DEFAULT_MOTION_AUX_LOSS_WEIGHT)
        )
        if self.motion_aux_loss_weight < 0:
            raise ValueError("motion_aux_loss_weight must be >= 0")
        self.objectless_object_action_suppression_loss_weight = float(
            hparams.get("objectless_object_action_suppression_loss_weight", 0.3)
        )
        if self.objectless_object_action_suppression_loss_weight < 0:
            raise ValueError(
                "objectless_object_action_suppression_loss_weight must be >= 0"
            )
        self.object_action_confuser_loss_weight = float(
            hparams.get(
                "object_action_confuser_loss_weight",
                DEFAULT_OBJECT_ACTION_CONFUSER_LOSS_WEIGHT,
            )
        )
        if self.object_action_confuser_loss_weight < 0:
            raise ValueError("object_action_confuser_loss_weight must be >= 0")
        self.object_action_confuser_margin = float(
            hparams.get("object_action_confuser_margin", 0.75)
        )
        if self.object_action_confuser_margin < 0:
            raise ValueError("object_action_confuser_margin must be >= 0")
        self.object_slot_target_loss_weight = float(
            hparams.get("object_slot_target_loss_weight", 1.0)
        )
        if self.object_slot_target_loss_weight < 0:
            raise ValueError("object_slot_target_loss_weight must be >= 0")
        self.object_slot_ignore_missing_object = bool(
            hparams.get("object_slot_ignore_missing_object", 0)
        )
        self.object_slot_quality_loss_weight = float(
            hparams.get("object_slot_quality_loss_weight", 0.5)
        )
        if self.object_slot_quality_loss_weight < 0:
            raise ValueError("object_slot_quality_loss_weight must be >= 0")
        self.object_slot_quality_pos_weight = float(
            hparams.get("object_slot_quality_pos_weight", 1.0)
        )
        if self.object_slot_quality_pos_weight < 0:
            raise ValueError("object_slot_quality_pos_weight must be >= 0")
        self.object_slot_quality_neg_weight = float(
            hparams.get("object_slot_quality_neg_weight", 0.25)
        )
        if self.object_slot_quality_neg_weight < 0:
            raise ValueError("object_slot_quality_neg_weight must be >= 0")
        self.object_slot_quality_exact_neg_topk = int(
            hparams.get("object_slot_quality_exact_neg_topk", 4)
        )
        if self.object_slot_quality_exact_neg_topk < 0:
            raise ValueError("object_slot_quality_exact_neg_topk must be >= 0")
        self.object_slot_quality_objectless_neg_topk = int(
            hparams.get("object_slot_quality_objectless_neg_topk", 8)
        )
        if self.object_slot_quality_objectless_neg_topk < 0:
            raise ValueError("object_slot_quality_objectless_neg_topk must be >= 0")
        self.object_slot_unknown_exact_loss_weight = float(
            hparams.get("object_slot_unknown_exact_loss_weight", 0.0)
        )
        if self.object_slot_unknown_exact_loss_weight < 0:
            raise ValueError("object_slot_unknown_exact_loss_weight must be >= 0")
        self.num_classes = hparams.num_classes
        self.dataset_name = hparams.dataset_artifact
        self.is_toyota = str(self.dataset_name).lower() == "toyotasm"
        self.task_type = hparams.get("task_type", "CS")
        self.action_taxonomy = hparams.get("toyota_action_taxonomy", "toyota_31")
        if self.is_toyota:
            self.action_names = toyota_action_names(
                self.task_type,
                self.action_taxonomy,
            )
            if self.num_classes != len(self.action_names):
                raise ValueError(
                    "HeatmapModule num_classes does not match Toyota action taxonomy: "
                    f"{self.action_taxonomy} {self.task_type} expects "
                    f"{len(self.action_names)}, got {self.num_classes}."
                )
            self.action_to_index = toyota_action_to_index(
                self.task_type,
                self.action_taxonomy,
            )
            self.action_object_map = toyota_action_object_map(
                self.task_type,
                self.action_taxonomy,
            )
            self.action_object_ids_by_index = self._build_action_object_ids_by_index()
            self.action_confuser_indices = self._build_action_confuser_indices()
        else:
            self.action_names = [str(index) for index in range(self.num_classes)]
            self.action_to_index = {}
            self.action_object_map = {}
            self.action_object_ids_by_index = {}
            self.action_confuser_indices = {}

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
        self.validation_step_outputs = self._empty_validation_outputs()
        self.actor_val_diagnostics = bool(hparams.get("actor_val_diagnostics", 1))
        self.actor_val_diagnostic_max_pairs = int(
            hparams.get("actor_val_diagnostic_max_pairs", 8)
        )
        self.group_indices = self._build_group_indices()
        self.objectless_action_indices = self.group_indices.get(
            "objectless",
            torch.empty(0, dtype=torch.long),
        )
        self.actor_object_slot_spec = (
            self._build_actor_object_slot_spec()
            if self.is_toyota and self.actor_object_slot_head
            else None
        )
        self.hard_negative_object_ids = {
            int(OBJECT_TO_ID[object_name]): object_name
            for object_names in self.action_object_map.values()
            for object_name in object_names
            if object_name in OBJECT_TO_ID
        }

    def _empty_validation_outputs(self):
        return {
            "preds": [],
            "labels": [],
            "hard_objectless": [],
        }

    def load_state_dict(self, state_dict, strict=True, assign=False):
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        if self.actor_prompt and not strict:
            allowed_missing = [
                "model.net.actor_token",
                "model.net.actor_slot_embed",
                "model.net.valid_embed",
                "model.net.bbox_mlp",
                "model.actor_head",
                "model.actor_motion_head",
                "model.presence_head",
                "model.actor_object_slot_head",
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
        object_classes=None,
        object_confs=None,
        object_valid=None,
        object_heatmap_scores=None,
    ):
        # Forward function that is run when visualizing the graph
        return self.model(
            x,
            boxes=boxes,
            valid=valid,
            action_labels=action_labels,
            object_boxes=object_boxes,
            object_classes=object_classes,
            object_confs=object_confs,
            object_valid=object_valid,
            object_heatmap_scores=object_heatmap_scores,
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

    def _unique_params(self, params):
        unique = []
        seen = set()
        for param in params:
            key = id(param)
            if key in seen:
                continue
            seen.add(key)
            unique.append(param)
        return unique

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

        freeze_actor_path = self.actor_interaction_heatmaps and self.model.hparams.get(
            "interaction_warmup_freeze_actor_path", 0
        )

        params = []
        if not freeze_actor_path:
            params += list(self.model.actor_head.parameters())
            if getattr(self.model, "actor_motion_head", None) is not None:
                params += list(self.model.actor_motion_head.parameters())
            if self.model.presence_head is not None:
                params += list(self.model.presence_head.parameters())
            for name, param in self.model.net.named_parameters():
                if self._actor_prompt_param_name(name):
                    params.append(param)
        if (
            self.actor_object_slot_head
            and getattr(self.model, "actor_object_slot_head", None) is not None
        ):
            params += list(self.model.actor_object_slot_head.parameters())
        return self._unique_params(params)

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
        return toyota_label_dict(self.task_type, self.action_taxonomy)

    def _build_action_object_ids_by_index(self):
        action_object_ids = {}
        for action_name, object_names in self.action_object_map.items():
            action_idx = self.action_to_index.get(action_name)
            if action_idx is None:
                continue
            object_ids = [
                int(OBJECT_TO_ID[object_name])
                for object_name in object_names
                if object_name in OBJECT_TO_ID
            ]
            if object_ids:
                action_object_ids[int(action_idx)] = torch.tensor(
                    object_ids,
                    dtype=torch.long,
                )
        return action_object_ids

    def _build_actor_object_slot_spec(self):
        action_to_object_ids = {
            int(action_idx): tuple(int(x) for x in object_ids.tolist())
            for action_idx, object_ids in self.action_object_ids_by_index.items()
        }
        return ActionObjectQuerySpec(
            num_actions=int(self.num_classes),
            num_object_classes=int(
                self.model.hparams.get("num_object_classes", NUM_OBJECT_CLASSES)
            ),
            objectless_action_indices=tuple(
                int(x) for x in self.objectless_action_indices.tolist()
            ),
            action_to_object_ids=action_to_object_ids,
        )

    def _build_action_confuser_indices(self):
        confuser_indices = {}
        for action_name, action_idx in self.action_to_index.items():
            confusers = [
                int(self.action_to_index[confuser_name])
                for confuser_name in toyota_confuser_action_names(
                    action_name,
                    self.task_type,
                    self.action_taxonomy,
                )
                if confuser_name in self.action_to_index
            ]
            if confusers:
                confuser_indices[int(action_idx)] = torch.tensor(
                    confusers,
                    dtype=torch.long,
                )
        return confuser_indices

    def _build_group_indices(self):
        if not self.is_toyota:
            return {}
        action_to_index = self.action_to_index
        groups = {}
        for group_name, action_names in toyota_group_action_names(
            self.task_type,
            self.action_taxonomy,
        ).items():
            indices = [
                int(action_to_index[action_name])
                for action_name in action_names
                if action_name in action_to_index
            ]
            if indices:
                groups[group_name] = torch.tensor(indices, dtype=torch.long)
        object_action_indices = [
            int(action_to_index[action_name])
            for action_name in self.action_object_map
            if action_name in action_to_index
        ]
        if object_action_indices:
            groups["object_mapped"] = torch.tensor(
                object_action_indices,
                dtype=torch.long,
            )
        objectless_indices = [
            int(action_to_index[action_name])
            for action_name in toyota_objectless_action_names(
                self.task_type,
                self.action_taxonomy,
            )
            if action_name in action_to_index
        ]
        if objectless_indices:
            groups["objectless"] = torch.tensor(objectless_indices, dtype=torch.long)
        return groups

    def _interaction_audit_action_indices(self):
        if not self.is_toyota:
            return []
        indices = []
        for action_idx, action_name in enumerate(self.action_names):
            metric_name = action_name.replace(".", "_")
            indices.append((metric_name, int(action_idx)))
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
        n_interaction = n_actor
        end = n_pose + n_interaction
        if hm_preds.shape[1] < end:
            raise RuntimeError(
                "Interaction heatmaps require heatmap channels "
                f"[pose={n_pose} + actors={n_actor}], "
                f"got {hm_preds.shape[1]}"
            )
        heatmaps = hm_preds[:, n_pose:end]
        return heatmaps.reshape(heatmaps.shape[0], n_actor, *heatmaps.shape[-2:])

    def _log_interaction_heatmap_metrics(
        self,
        pred_heatmap,
        target_heatmap,
        heatmap_valid,
        stage,
        interaction_cls=None,
    ):
        if pred_heatmap is None:
            return
        heatmap_valid = heatmap_valid.to(
            device=pred_heatmap.device,
            dtype=torch.bool,
        )
        if pred_heatmap.shape != target_heatmap.shape:
            raise RuntimeError(
                "interaction heatmap metric shape mismatch: "
                f"{tuple(pred_heatmap.shape)} vs {tuple(target_heatmap.shape)}"
            )
        if heatmap_valid.shape != pred_heatmap.shape[:-2]:
            raise RuntimeError(
                "interaction heatmap metric valid shape mismatch: "
                f"{tuple(heatmap_valid.shape)} vs {tuple(pred_heatmap.shape[:-2])}"
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

        if interaction_cls is not None:
            interaction_cls = interaction_cls.to(
                device=pred_heatmap.device,
                dtype=torch.long,
            )
            if interaction_cls.shape != heatmap_valid.shape:
                raise RuntimeError(
                    "interaction class shape mismatch: "
                    f"{tuple(interaction_cls.shape)} vs {tuple(heatmap_valid.shape)}"
                )
            for cls_id, object_name in OBJECT_CLASSES.items():
                class_valid = heatmap_valid & (interaction_cls == int(cls_id))
                if not class_valid.any():
                    continue
                class_pred = pred_heatmap.float().clamp(0.0, 1.0)[class_valid]
                class_target = target_heatmap.to(
                    device=pred_heatmap.device
                ).float()[class_valid]
                class_target_bin = class_target > 0.3
                if not class_target_bin.flatten(1).any(dim=1).any():
                    continue
                class_positive = class_pred.masked_select(class_target_bin)
                class_count = int(class_valid.sum().item())
                safe_name = object_name
                if class_positive.numel() > 0:
                    self._log_scalar(
                        f"{stage}_interaction_heatmap_{safe_name}_positive_mean",
                        class_positive.mean(),
                        class_count,
                    )
                class_pred_bin = class_pred > 0.3
                class_intersection = (
                    class_pred_bin & class_target_bin
                ).float().flatten(1).sum(dim=1)
                class_union = (
                    class_pred_bin | class_target_bin
                ).float().flatten(1).sum(dim=1)
                class_valid_union = class_union > 0
                if class_valid_union.any():
                    self._log_scalar(
                        f"{stage}_interaction_heatmap_{safe_name}_iou",
                        (
                            class_intersection[class_valid_union]
                            / class_union[class_valid_union].clamp_min(1.0)
                        ).mean(),
                        class_count,
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
            self.log(
                metric_name.replace("_acc", "_count"),
                mask.float().sum(),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
                reduce_fx="sum",
                batch_size=1,
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
            self.log(
                prefix.format(action=metric_name).replace("_acc", "_count"),
                mask.float().sum(),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
                reduce_fx="sum",
                batch_size=1,
            )

    def _log_interaction_teacher_metrics(
        self,
        actions,
        valid,
        heatmap_valid,
        interaction_cls,
        stage,
    ):
        valid = valid.to(device=actions.device, dtype=torch.bool)
        heatmap_valid = heatmap_valid.to(device=actions.device, dtype=torch.bool)
        if heatmap_valid.ndim != 2:
            raise RuntimeError(
                "Actor interaction teacher mask must have shape "
                f"[batch, actors], got {tuple(heatmap_valid.shape)}"
            )
        if interaction_cls.shape != heatmap_valid.shape:
            raise RuntimeError(
                "interaction_cls must have shape "
                f"{tuple(heatmap_valid.shape)}, got {tuple(interaction_cls.shape)}"
            )
        interaction_cls = interaction_cls.to(device=actions.device, dtype=torch.long)
        slot_has_teacher = heatmap_valid & valid
        valid_count = int(valid.sum().item())
        if valid_count <= 0:
            return

        self._log_scalar(
            f"{stage}_interaction_teacher_slot_rate",
            slot_has_teacher.float().sum() / max(valid_count, 1),
            valid_count,
        )
        self.log(
            f"{stage}_interaction_teacher_slot_count",
            slot_has_teacher.float().sum(),
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
                slot_has_teacher[mask].float().mean(),
                int(mask.sum().item()),
            )

        for cls_id, object_name in OBJECT_CLASSES.items():
            cls_id = int(cls_id)
            class_teacher = slot_has_teacher & (interaction_cls == cls_id)
            if not class_teacher.any():
                continue
            self.log(
                f"{stage}_interaction_teacher_{object_name}_slot_count",
                class_teacher.float().sum(),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
                reduce_fx="sum",
                batch_size=1,
            )

    def _empty_object_inputs(self, batch_size, device, dtype=torch.float32):
        n_objects = int(self.model.hparams.get("num_scene_object_tokens", 32))
        none_id = int(self.model.hparams.get("num_object_classes", NUM_OBJECT_CLASSES))
        return {
            "object_boxes": torch.zeros(
                batch_size,
                n_objects,
                4,
                device=device,
                dtype=dtype,
            ),
            "object_classes": torch.full(
                (batch_size, n_objects),
                none_id,
                device=device,
                dtype=torch.long,
            ),
            "object_confs": torch.zeros(
                batch_size,
                n_objects,
                device=device,
                dtype=dtype,
            ),
            "object_valid": torch.zeros(
                batch_size,
                n_objects,
                device=device,
                dtype=torch.bool,
            ),
        }

    def _object_inputs_from_target(self, target, device):
        if not self.uses_object_proposals:
            return {}
        required = (
            "object_boxes",
            "object_classes",
            "object_confs",
            "object_valid",
        )
        missing = [key for key in required if key not in target]
        if missing:
            raise RuntimeError(
                "Object-proposal action heads require dataset object targets; "
                f"missing {missing}"
            )
        return {
            "object_boxes": target["object_boxes"].to(
                device=device,
                dtype=torch.float32,
            ),
            "object_classes": target["object_classes"].to(
                device=device,
                dtype=torch.long,
            ),
            "object_confs": target["object_confs"].to(
                device=device,
                dtype=torch.float32,
            ),
            "object_valid": target["object_valid"].to(
                device=device,
                dtype=torch.bool,
            ),
        }

    def _exact_teacher_object_info(self, actions, valid, target, device):
        required = (
            "object_classes",
            "object_valid",
            "interaction_object_index",
            "interaction_object_index_valid",
        )
        if any(key not in target for key in required):
            return None
        if not self.action_object_ids_by_index:
            return None

        actions = actions.to(device=device, dtype=torch.long)
        valid = valid.to(device=device, dtype=torch.bool)
        valid = valid & (actions >= 0) & (actions < int(self.num_classes))
        object_classes = target["object_classes"].to(device=device, dtype=torch.long)
        object_valid = target["object_valid"].to(device=device, dtype=torch.bool)
        selected_indices = target["interaction_object_index"].to(
            device=device,
            dtype=torch.long,
        )
        selected_valid = target["interaction_object_index_valid"].to(
            device=device,
            dtype=torch.bool,
        )
        if object_classes.ndim != 2 or object_valid.shape != object_classes.shape:
            return None

        num_objects = int(object_classes.shape[1])
        if num_objects <= 0:
            return None

        idx_1based = (selected_indices - 1).clamp(0, num_objects - 1)
        idx_0based = selected_indices.clamp(0, num_objects - 1)
        in_range_1based = (
            selected_valid
            & (selected_indices > 0)
            & ((selected_indices - 1) < num_objects)
        )
        in_range_0based = (
            selected_valid
            & (selected_indices >= 0)
            & (selected_indices < num_objects)
        )
        class_1based = object_classes.gather(1, idx_1based)
        class_0based = object_classes.gather(1, idx_0based)
        valid_1based = in_range_1based & object_valid.gather(1, idx_1based)
        valid_0based = in_range_0based & object_valid.gather(1, idx_0based)

        known_action = torch.zeros_like(valid, dtype=torch.bool)
        any_compatible = torch.zeros_like(valid, dtype=torch.bool)
        compatible_1based = torch.zeros_like(valid, dtype=torch.bool)
        compatible_0based = torch.zeros_like(valid, dtype=torch.bool)
        for action_idx, object_ids in self.action_object_ids_by_index.items():
            action_mask = valid & (actions == int(action_idx))
            if not action_mask.any():
                known_action |= action_mask
                continue
            known_action |= action_mask
            object_ids = object_ids.to(device=device, dtype=torch.long)
            object_match = (
                object_classes.unsqueeze(-1) == object_ids.view(1, 1, -1)
            ).any(dim=-1) & object_valid
            any_compatible |= action_mask & object_match.any(dim=1, keepdim=True)
            class_match_1based = (
                class_1based.unsqueeze(-1) == object_ids.view(1, 1, -1)
            ).any(dim=-1)
            class_match_0based = (
                class_0based.unsqueeze(-1) == object_ids.view(1, 1, -1)
            ).any(dim=-1)
            compatible_1based |= action_mask & valid_1based & class_match_1based
            compatible_0based |= action_mask & valid_0based & class_match_0based

        return {
            "actions": actions,
            "valid": valid,
            "object_classes": object_classes,
            "object_valid": object_valid,
            "selected_indices": selected_indices,
            "idx_1based": idx_1based,
            "idx_0based": idx_0based,
            "valid_1based": valid_1based,
            "valid_0based": valid_0based,
            "known_action": known_action,
            "any_compatible": any_compatible,
            "compatible_1based": compatible_1based,
            "compatible_0based": compatible_0based,
        }

    def _object_slot_target_loss(self, stage, actions, valid, target):
        if (
            not self.actor_object_slot_head
            or self.object_slot_target_loss_weight <= 0
            or self.actor_object_slot_spec is None
        ):
            return None
        slot_delta = getattr(self.model, "last_actor_object_slot_delta", None)
        if slot_delta is None:
            return None
        required = ("object_classes", "object_valid")
        if any(key not in target for key in required):
            return None
        loss = action_slot_target_loss(
            slot_delta=slot_delta,
            labels=actions,
            object_classes=target["object_classes"],
            object_valid=target["object_valid"],
            spec=self.actor_object_slot_spec,
            valid=valid,
            interaction_object_index=target.get("interaction_object_index"),
            interaction_object_index_valid=target.get(
                "interaction_object_index_valid"
            ),
            ignore_missing_object=self.object_slot_ignore_missing_object,
        )
        self.log(
            f"{stage}_loss_object_slot_target",
            loss,
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        return loss

    def _object_slot_quality_loss(self, stage, actions, valid, target):
        if (
            not self.actor_object_slot_head
            or self.object_slot_quality_loss_weight <= 0
        ):
            return None
        quality_logits = getattr(
            self.model,
            "last_actor_object_quality_logits",
            None,
        )
        if quality_logits is None:
            return None
        required = (
            "object_valid",
            "interaction_object_index",
            "interaction_object_index_valid",
        )
        if any(key not in target for key in required):
            return None

        device = quality_logits.device
        actions = actions.to(device=device, dtype=torch.long)
        valid = valid.to(device=device, dtype=torch.bool)
        object_valid = target["object_valid"].to(device=device, dtype=torch.bool)
        selected_indices = target["interaction_object_index"].to(
            device=device,
            dtype=torch.long,
        )
        selected_valid = target["interaction_object_index_valid"].to(
            device=device,
            dtype=torch.bool,
        )

        num_objects = int(quality_logits.shape[-1])
        if num_objects <= 0:
            return None

        objectless = self._labels_in_indices(actions, self.objectless_action_indices)
        selected_object_slots = selected_indices - 1
        selected_in_range = (
            selected_valid
            & (selected_indices > 0)
            & (selected_object_slots >= 0)
            & (selected_object_slots < num_objects)
        )
        safe_selected_slots = selected_object_slots.clamp(0, max(num_objects - 1, 0))
        selected_object_valid = selected_in_range & object_valid.gather(
            1,
            safe_selected_slots,
        )
        object_valid_slots = object_valid[:, None, :].expand_as(quality_logits)
        exact_object = valid & selected_object_valid & ~objectless
        teacher_slot_mask = torch.zeros_like(quality_logits, dtype=torch.bool)
        rows = torch.nonzero(exact_object, as_tuple=False)
        if rows.numel() > 0:
            object_slots = safe_selected_slots[exact_object]
            teacher_slot_mask[rows[:, 0], rows[:, 1], object_slots] = True

        pos_mask = teacher_slot_mask
        exact_neg_mask = (
            exact_object[:, :, None]
            & object_valid_slots
            & ~teacher_slot_mask
        )
        objectless_neg_mask = (
            (valid & objectless)[:, :, None]
            & object_valid_slots
        )

        if not (pos_mask.any() or exact_neg_mask.any() or objectless_neg_mask.any()):
            return None

        logits = quality_logits.float()

        def _topk_masked_values(values, mask, topk):
            if topk <= 0 or not mask.any():
                return values.new_empty((0,))
            k = min(int(topk), int(values.shape[-1]))
            masked = values.masked_fill(~mask, float("-inf"))
            top_values = masked.topk(k, dim=-1).values
            return top_values[torch.isfinite(top_values)]

        loss_terms = []
        if pos_mask.any() and self.object_slot_quality_pos_weight > 0:
            pos_loss = F.binary_cross_entropy_with_logits(
                logits[pos_mask],
                torch.ones_like(logits[pos_mask]),
            )
            loss_terms.append(pos_loss * self.object_slot_quality_pos_weight)
            self.log(
                f"{stage}_loss_object_slot_quality_pos",
                pos_loss,
                on_step=stage == "train",
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )
        exact_neg_values = _topk_masked_values(
            logits,
            exact_neg_mask,
            self.object_slot_quality_exact_neg_topk,
        )
        objectless_neg_values = _topk_masked_values(
            logits,
            objectless_neg_mask,
            self.object_slot_quality_objectless_neg_topk,
        )
        neg_parts = [
            values
            for values in (exact_neg_values, objectless_neg_values)
            if values.numel() > 0
        ]
        neg_values = (
            torch.cat(neg_parts)
            if neg_parts
            else logits.new_empty((0,))
        )
        if neg_values.numel() > 0 and self.object_slot_quality_neg_weight > 0:
            neg_loss = F.binary_cross_entropy_with_logits(
                neg_values,
                torch.zeros_like(neg_values),
            )
            loss_terms.append(neg_loss * self.object_slot_quality_neg_weight)
            self.log(
                f"{stage}_loss_object_slot_quality_neg",
                neg_loss,
                on_step=stage == "train",
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )
            self._log_count(
                f"{stage}_object_slot_quality_exact_neg_count",
                logits.new_tensor(float(exact_neg_values.numel())),
            )
            self._log_count(
                f"{stage}_object_slot_quality_objectless_neg_count",
                logits.new_tensor(float(objectless_neg_values.numel())),
            )
        if not loss_terms:
            return None
        loss = torch.stack(loss_terms).sum()
        self.log(
            f"{stage}_loss_object_slot_quality",
            loss,
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )

        with torch.no_grad():
            probs = torch.sigmoid(logits)
            if pos_mask.any():
                self._log_scalar(
                    f"{stage}_object_slot_quality_pos_acc",
                    (probs[pos_mask] > 0.5).float().mean(),
                    int(pos_mask.sum().item()),
                )
                self._log_scalar(
                    f"{stage}_object_slot_quality_pos_mean",
                    probs[pos_mask].mean(),
                    int(pos_mask.sum().item()),
                )
            if neg_values.numel() > 0:
                neg_probs = torch.sigmoid(neg_values)
                self._log_scalar(
                    f"{stage}_object_slot_quality_neg_acc",
                    (neg_probs < 0.5).float().mean(),
                    int(neg_values.numel()),
                )
                self._log_scalar(
                    f"{stage}_object_slot_quality_neg_mean",
                    neg_probs.mean(),
                    int(neg_values.numel()),
                )
        return loss

    def _object_slot_unknown_exact_loss(self, stage, actions, valid, target):
        if (
            not self.actor_object_slot_head
            or self.object_slot_unknown_exact_loss_weight <= 0
        ):
            return None
        slot_delta = getattr(self.model, "last_actor_object_slot_delta", None)
        if slot_delta is None:
            return None

        device = slot_delta.device
        info = self._exact_teacher_object_info(actions, valid, target, device)
        if info is None:
            return None

        actions = info["actions"]
        exact_compatible = info["known_action"] & info["compatible_1based"]
        if not exact_compatible.any():
            return None

        rows = torch.nonzero(exact_compatible, as_tuple=False)
        true_actions = actions[exact_compatible]
        true_slot_logits = slot_delta[rows[:, 0], rows[:, 1], true_actions]
        unknown_prob = torch.softmax(true_slot_logits.float(), dim=-1)[:, 1]
        loss = -torch.log1p(-unknown_prob.clamp(max=1.0 - 1.0e-6)).mean()
        self.log(
            f"{stage}_loss_object_slot_unknown_exact",
            loss,
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        if stage == "train":
            self._log_scalar(
                f"{stage}_object_slot_exact_unknown_prob",
                unknown_prob.mean(),
                int(exact_compatible.sum().item()),
            )
        return loss

    def _teacher_object_removed_inputs(
        self,
        object_inputs,
        selected_indices,
        selected_valid,
    ):
        output = {
            name: value.clone()
            for name, value in object_inputs.items()
        }
        if not selected_valid.any():
            return output
        rows = torch.nonzero(selected_valid, as_tuple=False)
        object_slots = selected_indices[selected_valid] - 1
        batch_idx = rows[:, 0]
        object_slots = object_slots.to(device=batch_idx.device, dtype=torch.long)
        none_id = int(self.model.hparams.get("num_object_classes", NUM_OBJECT_CLASSES))

        output["object_valid"][batch_idx, object_slots] = False
        output["object_boxes"][batch_idx, object_slots] = 0
        output["object_confs"][batch_idx, object_slots] = 0
        output["object_classes"][batch_idx, object_slots] = none_id
        return output

    def _log_object_counterfactual_eval(
        self,
        imgs,
        boxes,
        valid,
        actions,
        preds,
        target,
        object_inputs,
        stage,
    ):
        if not self.uses_object_proposals:
            return None
        if stage == "train":
            return None
        if (
            "interaction_object_index" not in target
            or "interaction_object_index_valid" not in target
        ):
            return None

        selected_indices = target["interaction_object_index"].to(
            device=imgs.device,
            dtype=torch.long,
        )
        selected_valid = target["interaction_object_index_valid"].to(
            device=imgs.device,
            dtype=torch.bool,
        )
        selected_valid = selected_valid & valid & (selected_indices > 0)
        if not selected_valid.any():
            return None

        counterfactual_inputs = self._teacher_object_removed_inputs(
            object_inputs,
            selected_indices,
            selected_valid,
        )
        counterfactual_data = self.model(
            imgs,
            boxes=boxes,
            valid=valid,
            action_labels=actions,
            **counterfactual_inputs,
        )
        counterfactual_preds = self._unpack_model_data(counterfactual_data)[0]

        true_labels = actions[selected_valid].long()
        base_true_logits = preds[selected_valid].gather(
            1,
            true_labels.unsqueeze(1),
        ).squeeze(1)
        counterfactual_true_logits = counterfactual_preds[selected_valid].gather(
            1,
            true_labels.unsqueeze(1),
        ).squeeze(1)
        logit_drop = base_true_logits - counterfactual_true_logits
        count = int(true_labels.numel())
        self._log_scalar(
            f"{stage}_object_counterfactual_teacher_logit_drop",
            logit_drop.detach().mean(),
            count,
        )

        base_true_probs = preds[selected_valid].float().softmax(dim=-1).gather(
            1,
            true_labels.unsqueeze(1),
        ).squeeze(1)
        counterfactual_true_probs = (
            counterfactual_preds[selected_valid]
            .float()
            .softmax(dim=-1)
            .gather(1, true_labels.unsqueeze(1))
            .squeeze(1)
        )
        self._log_scalar(
            f"{stage}_object_counterfactual_teacher_prob_drop",
            (base_true_probs - counterfactual_true_probs).mean(),
            count,
        )
        return None

    def _log_actor_object_slot_diagnostics(
        self,
        stage,
        full_preds,
        actions,
        valid,
        target,
    ):
        if stage == "train" or not self.actor_object_slot_head:
            return
        slot_delta = getattr(self.model, "last_actor_object_slot_delta", None)
        best_slot = getattr(self.model, "last_actor_object_best_slot", None)
        if slot_delta is None or best_slot is None:
            return
        if "object_classes" not in target or "object_valid" not in target:
            return

        device = slot_delta.device
        valid = valid.to(device=device, dtype=torch.bool)
        if not valid.any():
            return
        actions = actions.to(device=device, dtype=torch.long)
        valid = valid & (actions >= 0) & (actions < best_slot.shape[-1])
        if not valid.any():
            return
        safe_actions = actions.clamp(0, best_slot.shape[-1] - 1)
        object_classes = target["object_classes"].to(device=device, dtype=torch.long)
        object_valid = target["object_valid"].to(device=device, dtype=torch.bool)
        full_preds = full_preds.to(device=device)

        true_best_slot = best_slot.gather(
            dim=-1,
            index=safe_actions.unsqueeze(-1),
        ).squeeze(-1)
        object_slot = true_best_slot >= 2
        object_slot_index = (true_best_slot - 2).clamp_min(0)
        object_slot_index = object_slot_index.clamp_max(object_classes.shape[1] - 1)
        true_object_class = object_classes.gather(1, object_slot_index)
        true_object_valid = object_valid.gather(1, object_slot_index) & object_slot

        known_object_action = torch.zeros_like(valid, dtype=torch.bool)
        compatible_object_slot = torch.zeros_like(valid, dtype=torch.bool)
        for action_idx, object_ids in self.action_object_ids_by_index.items():
            action_mask = actions == int(action_idx)
            known_object_action |= action_mask
            object_ids = object_ids.to(device=device, dtype=torch.long)
            object_match = (
                true_object_class.unsqueeze(-1) == object_ids.view(1, 1, -1)
            ).any(dim=-1)
            compatible_object_slot |= action_mask & true_object_valid & object_match

        objectless = self._labels_in_indices(actions, self.objectless_action_indices)
        objectless_mask = valid & objectless
        known_object_mask = valid & known_object_action
        if known_object_mask.any():
            count = int(known_object_mask.sum().item())
            unknown = known_object_mask & (true_best_slot == 1)
            null = known_object_mask & (true_best_slot == 0)
            incompatible = known_object_mask & true_object_valid & ~compatible_object_slot
            self._log_scalar(
                f"{stage}_object_slot_true_compatible_rate",
                compatible_object_slot[known_object_mask].float().mean(),
                count,
            )
            self._log_scalar(
                f"{stage}_object_slot_true_unknown_rate",
                unknown[known_object_mask].float().mean(),
                count,
            )
            self._log_scalar(
                f"{stage}_object_slot_true_null_rate",
                null[known_object_mask].float().mean(),
                count,
            )
            self._log_scalar(
                f"{stage}_object_slot_true_incompatible_rate",
                incompatible[known_object_mask].float().mean(),
                count,
            )
            self._log_count(
                f"{stage}_object_slot_known_action_count",
                known_object_mask.float().sum(),
            )

        exact_info = self._exact_teacher_object_info(actions, valid, target, device)
        if exact_info is not None:
            exact_known = valid & exact_info["known_action"]
            if exact_known.any():
                count = int(exact_known.sum().item())
                self._log_scalar(
                    f"{stage}_object_slot_exact_teacher_valid_rate_1based",
                    exact_info["valid_1based"][exact_known].float().mean(),
                    count,
                )
                self._log_scalar(
                    f"{stage}_object_slot_exact_teacher_valid_rate_0based",
                    exact_info["valid_0based"][exact_known].float().mean(),
                    count,
                )
                self._log_scalar(
                    f"{stage}_object_slot_exact_compatible_rate_1based",
                    exact_info["compatible_1based"][exact_known].float().mean(),
                    count,
                )
                self._log_scalar(
                    f"{stage}_object_slot_exact_compatible_rate_0based",
                    exact_info["compatible_0based"][exact_known].float().mean(),
                    count,
                )
                self._log_scalar(
                    f"{stage}_object_slot_any_compatible_proposal_rate",
                    exact_info["any_compatible"][exact_known].float().mean(),
                    count,
                )
            exact_compatible = (
                valid
                & exact_info["known_action"]
                & exact_info["compatible_1based"]
            )
            teacher_missing = (
                valid
                & exact_info["known_action"]
                & ~exact_info["valid_1based"]
            )
            teacher_mismatch = (
                valid
                & exact_info["known_action"]
                & exact_info["valid_1based"]
                & ~exact_info["compatible_1based"]
            )
            if exact_compatible.any():
                exact_count = int(exact_compatible.sum().item())
                self._log_scalar(
                    f"{stage}_object_slot_unknown_rate_given_exact_compatible",
                    (true_best_slot[exact_compatible] == 1).float().mean(),
                    exact_count,
                )
                self._log_scalar(
                    f"{stage}_object_slot_object_rate_given_exact_compatible",
                    (true_best_slot[exact_compatible] >= 2).float().mean(),
                    exact_count,
                )
            if teacher_missing.any():
                missing_count = int(teacher_missing.sum().item())
                self._log_count(
                    f"{stage}_object_slot_mapped_missing_count",
                    teacher_missing.float().sum(),
                )
                self._log_scalar(
                    f"{stage}_object_slot_unknown_rate_given_missing",
                    (true_best_slot[teacher_missing] == 1).float().mean(),
                    missing_count,
                )
            if teacher_mismatch.any():
                mismatch_count = int(teacher_mismatch.sum().item())
                self._log_count(
                    f"{stage}_object_slot_mapped_mismatch_count",
                    teacher_mismatch.float().sum(),
                )
                self._log_scalar(
                    f"{stage}_object_slot_unknown_rate_given_mismatch",
                    (true_best_slot[teacher_mismatch] == 1).float().mean(),
                    mismatch_count,
                )
            if exact_compatible.any():
                expected_slot = exact_info["idx_1based"] + 2
                exact_count = int(exact_compatible.sum().item())
                self._log_count(
                    f"{stage}_object_slot_exact_compatible_count",
                    exact_compatible.float().sum(),
                )
                self._log_scalar(
                    f"{stage}_object_slot_exact_correct_object_rate",
                    (true_best_slot[exact_compatible] == expected_slot[exact_compatible])
                    .float()
                    .mean(),
                    exact_count,
                )
                rows = torch.nonzero(exact_compatible, as_tuple=False)
                exact_logits = slot_delta[
                    rows[:, 0],
                    rows[:, 1],
                    safe_actions[exact_compatible],
                ].float()
                exact_prob = exact_logits.softmax(dim=-1)
                expected_prob = exact_prob.gather(
                    1,
                    expected_slot[exact_compatible].unsqueeze(1),
                ).squeeze(1)
                self._log_scalar(
                    f"{stage}_object_slot_exact_correct_object_prob",
                    expected_prob.mean(),
                    exact_count,
                )
                self._log_scalar(
                    f"{stage}_object_slot_exact_unknown_prob",
                    exact_prob[:, 1].mean(),
                    exact_count,
                )
                quality = getattr(self.model, "last_actor_object_quality", None)
                if quality is not None:
                    quality = quality.to(device=device).float()
                    teacher_quality = quality.gather(
                        2,
                        exact_info["idx_1based"].unsqueeze(-1),
                    ).squeeze(-1)
                    self._log_scalar(
                        f"{stage}_object_slot_exact_quality_pos_mean",
                        teacher_quality[exact_compatible].mean(),
                        exact_count,
                    )
                    valid_quality = quality.masked_fill(
                        ~exact_info["object_valid"][:, None, :],
                        float("-inf"),
                    )
                    better_count = (
                        valid_quality
                        > teacher_quality.unsqueeze(-1)
                    ).sum(dim=-1)
                    teacher_rank = better_count + 1
                    self._log_scalar(
                        f"{stage}_object_slot_exact_quality_teacher_rank_mean",
                        teacher_rank[exact_compatible].float().mean(),
                        exact_count,
                    )
                    self._log_scalar(
                        f"{stage}_object_slot_exact_quality_teacher_top1_rate",
                        (teacher_rank[exact_compatible] == 1).float().mean(),
                        exact_count,
                    )
        if objectless_mask.any():
            count = int(objectless_mask.sum().item())
            self._log_scalar(
                f"{stage}_object_slot_objectless_null_rate",
                (true_best_slot[objectless_mask] == 0).float().mean(),
                count,
            )
            self._log_scalar(
                f"{stage}_object_slot_objectless_nonnull_rate",
                (true_best_slot[objectless_mask] != 0).float().mean(),
                count,
            )

        slot_logit_delta = getattr(self.model, "last_actor_object_slot_logit_delta", None)
        if slot_logit_delta is not None:
            valid_delta = slot_logit_delta.to(device=device)[valid].float()
            count = int(valid_delta.shape[0])
            self._log_scalar(
                f"{stage}_actor_object_slot_delta_abs_mean",
                valid_delta.abs().mean(),
                count,
            )
            self._log_scalar(
                f"{stage}_actor_object_slot_delta_l2_mean",
                valid_delta.norm(dim=-1).mean(),
                count,
            )
        relation_scale = getattr(
            self.model,
            "last_actor_object_slot_relation_scale",
            None,
        )
        if relation_scale is not None:
            self._log_scalar(
                f"{stage}_actor_object_slot_relation_scale",
                relation_scale.to(device=device).float(),
                int(valid.sum().item()),
            )

        quality = getattr(self.model, "last_actor_object_quality", None)
        if quality is not None:
            valid_quality = quality.to(device=device)[valid].float()
            if valid_quality.numel() > 0:
                self._log_scalar(
                    f"{stage}_object_slot_quality_mean",
                    valid_quality.mean(),
                    int(valid_quality.numel()),
                )
                self._log_scalar(
                    f"{stage}_object_slot_quality_max_mean",
                    valid_quality.amax(dim=-1).mean(),
                    int(valid_quality.shape[0]),
                )
        mismatch = getattr(self.model, "last_actor_object_mismatch", None)
        if mismatch is not None:
            valid_mismatch = mismatch.to(device=device)[valid].float()
            if valid_mismatch.numel() > 0:
                self._log_scalar(
                    f"{stage}_object_slot_mismatch_mean",
                    valid_mismatch.mean(),
                    int(valid_mismatch.numel()),
                )

        motion_logits = getattr(self.model, "last_actor_motion_logits", None)
        tracked = {
            "Uselaptop": ("laptop", ("Readbook", "Usetelephone", "WatchTV")),
            "Readbook": ("book", ("Uselaptop", "Usetelephone", "WatchTV")),
            "Usetelephone": ("phone", ("Uselaptop", "Readbook", "WatchTV")),
            "WatchTV": ("tv_monitor", ("Uselaptop", "Readbook", "Usetelephone")),
        }
        for action_name, (object_name, confuser_names) in tracked.items():
            action_idx = self._action_index(action_name)
            object_id = OBJECT_TO_ID.get(object_name)
            if action_idx is None or object_id is None:
                continue
            mask = valid & (actions == int(action_idx))
            if not mask.any():
                continue
            action_best_slot = best_slot[..., int(action_idx)]
            action_slot_is_object = action_best_slot >= 2
            action_slot_idx = (action_best_slot - 2).clamp_min(0)
            action_slot_idx = action_slot_idx.clamp_max(object_classes.shape[1] - 1)
            action_slot_class = object_classes.gather(1, action_slot_idx)
            action_slot_valid = object_valid.gather(1, action_slot_idx)
            compatible = (
                action_slot_is_object
                & action_slot_valid
                & (action_slot_class == int(object_id))
            )
            incompatible = action_slot_is_object & action_slot_valid & ~compatible
            count = int(mask.sum().item())
            self._log_scalar(
                f"{stage}_object_slot_{action_name}_{object_name}_rate",
                compatible[mask].float().mean(),
                count,
            )
            self._log_scalar(
                f"{stage}_object_slot_{action_name}_unknown_rate",
                (action_best_slot[mask] == 1).float().mean(),
                count,
            )
            self._log_scalar(
                f"{stage}_object_slot_{action_name}_null_rate",
                (action_best_slot[mask] == 0).float().mean(),
                count,
            )
            self._log_scalar(
                f"{stage}_object_slot_{action_name}_incompatible_rate",
                incompatible[mask].float().mean(),
                count,
            )
            for confuser_name in confuser_names:
                confuser_idx = self._action_index(confuser_name)
                if confuser_idx is None:
                    continue
                final_margin = (
                    full_preds[..., int(action_idx)]
                    - full_preds[..., int(confuser_idx)]
                )[mask].float()
                self._log_scalar(
                    f"{stage}_object_slot_{action_name}_minus_{confuser_name}_logit_margin",
                    final_margin.mean(),
                    count,
                )
                if motion_logits is not None:
                    motion_margin = (
                        motion_logits[..., int(action_idx)]
                        - motion_logits[..., int(confuser_idx)]
                    )[mask].float()
                    self._log_scalar(
                        f"{stage}_object_slot_motion_{action_name}_minus_{confuser_name}_logit_margin",
                        motion_margin.mean(),
                        count,
                    )
                    self._log_scalar(
                        f"{stage}_object_slot_delta_{action_name}_minus_{confuser_name}_logit_margin",
                        (final_margin - motion_margin).mean(),
                        count,
                    )

    def _object_action_confuser_loss(self, stage, preds, actions, valid, target):
        if self.object_action_confuser_loss_weight <= 0:
            return None
        if not self.is_toyota:
            return None
        if self.actor_object_slot_head:
            valid = valid.to(device=preds.device, dtype=torch.bool)
            actions = actions.to(device=preds.device, dtype=torch.long)
            if not valid.any():
                return None
            violations = []
            sample_count = 0
            for action_idx, confuser_indices in self.action_confuser_indices.items():
                confuser_indices = confuser_indices.to(device=preds.device)
                mask = valid & (actions == int(action_idx))
                if not mask.any():
                    continue
                true_logits = preds[..., int(action_idx)][mask]
                confuser_logits = preds[mask][:, confuser_indices]
                margin_error = (
                    confuser_logits
                    + float(self.object_action_confuser_margin)
                    - true_logits.unsqueeze(1)
                )
                violations.append(F.relu(margin_error).reshape(-1))
                sample_count += int(mask.sum().item())
            if not violations:
                return None
            violation_values = torch.cat(violations, dim=0)
            loss = violation_values.mean()
            self.log(
                f"{stage}_loss_object_action_confuser",
                loss,
                on_step=stage == "train",
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )
            self._log_scalar(
                f"{stage}_object_action_confuser_violation_rate",
                (violation_values > 0).float().mean(),
                max(sample_count, 1),
            )
            return loss
        return None

    def _motion_aux_loss(self, stage, actions, valid):
        if self.motion_aux_loss_weight <= 0:
            return None
        motion_logits = getattr(self.model, "last_actor_motion_logits", None)
        if motion_logits is None:
            return None
        valid = valid.to(device=motion_logits.device, dtype=torch.bool)
        if not valid.any():
            return None
        actions = actions.to(device=motion_logits.device, dtype=torch.long)
        loss = F.cross_entropy(motion_logits[valid].float(), actions[valid])
        self.log(
            f"{stage}_loss_motion_aux",
            loss,
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        pred_labels = motion_logits[valid].argmax(dim=-1)
        self._log_scalar(
            f"{stage}_motion_aux_acc",
            (pred_labels == actions[valid]).float().mean(),
            int(valid.sum().item()),
        )
        return loss

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
                    "--lr_head > 0, --lr_head_hm > 0, or "
                    "--interaction_unfreeze_last_blocks > 0."
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
                    "Nash-MTL expects two tasks [main_deploy, grounding_aux], "
                    f"got {loss.numel()}"
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
                self.log(
                    "train_nash_weight_main_deploy",
                    weights[0].detach(),
                    on_step=True,
                    on_epoch=True,
                    logger=True,
                    sync_dist=True,
                )
                self.log(
                    "train_nash_weight_grounding_aux",
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

        object_inputs = self._object_inputs_from_target(target, imgs.device)
        data = self.model(
            imgs,
            boxes=boxes,
            valid=valid,
            action_labels=actions,
            **object_inputs,
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

        loss_main_task = loss_action
        loss_motion_aux = self._motion_aux_loss(stage, actions, valid)
        if loss_motion_aux is not None:
            loss_main_task = (
                loss_main_task + loss_motion_aux * self.motion_aux_loss_weight
            )
        if presence_logits is not None:
            loss_presence = F.binary_cross_entropy_with_logits(
                presence_logits, valid.float()
            )
            loss_main_task = loss_main_task + loss_presence * self.model.hparams.get(
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

        loss_hard_objectless = self._objectless_object_action_suppression_loss(
            stage,
            preds,
            actions,
            valid,
            target,
        )
        if loss_hard_objectless is not None:
            loss_main_task = (
                loss_main_task
                + loss_hard_objectless
                * self.objectless_object_action_suppression_loss_weight
            )

        loss_confuser = self._object_action_confuser_loss(
            stage,
            preds,
            actions,
            valid,
            target,
        )
        if loss_confuser is not None:
            loss_main_task = (
                loss_main_task
                + loss_confuser * self.object_action_confuser_loss_weight
            )

        grounding_aux_terms = []
        loss_object_slot = self._object_slot_target_loss(
            stage,
            actions,
            valid,
            target,
        )
        if loss_object_slot is not None and self.object_slot_target_loss_weight > 0:
            grounding_aux_terms.append(
                loss_object_slot * self.object_slot_target_loss_weight
            )
        loss_object_slot_quality = self._object_slot_quality_loss(
            stage,
            actions,
            valid,
            target,
        )
        if (
            loss_object_slot_quality is not None
            and self.object_slot_quality_loss_weight > 0
        ):
            grounding_aux_terms.append(
                loss_object_slot_quality * self.object_slot_quality_loss_weight
            )
        loss_unknown_exact = self._object_slot_unknown_exact_loss(
            stage,
            actions,
            valid,
            target,
        )
        if (
            loss_unknown_exact is not None
            and self.object_slot_unknown_exact_loss_weight > 0
        ):
            grounding_aux_terms.append(
                loss_unknown_exact * self.object_slot_unknown_exact_loss_weight
            )

        self._log_object_counterfactual_eval(
            imgs,
            boxes,
            valid,
            actions,
            preds,
            target,
            object_inputs,
            stage,
        )
        self._log_actor_object_slot_diagnostics(
            stage,
            preds,
            actions,
            valid,
            target,
        )
        loss_kp = None
        loss_pose_frobenius = None
        loss_pose_heatmap_optimized = None
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
            if self.poguiseplus_normalized_heatmap_loss:
                loss_pose_heatmap_optimized = (
                    heatmap_mse_loss(
                        pose_hm_preds,
                        labels_kp.to(
                            device=pose_hm_preds.device,
                            dtype=pose_hm_preds.dtype,
                        ),
                        mask=kp_vis.to(device=pose_hm_preds.device),
                    )
                    * self.poguiseplus_heatmap_mse_scale
                )
                self.log(
                    f"{stage}_loss_pose_heatmap_mse_scaled",
                    loss_pose_heatmap_optimized,
                    on_step=stage == "train",
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                    sync_dist=True,
                )
            else:
                loss_pose_heatmap_optimized = loss_pose_frobenius
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
        loss_interaction_raw_frobenius = None
        loss_interaction_heatmap_optimized = None
        if self.actor_interaction_heatmaps:
            interaction_heatmap = self._interaction_heatmap_pred(hm_preds)
            if (
                interaction_heatmap is not None
                and "interaction_heatmap" in target
                and "interaction_heatmap_valid" in target
            ):
                heatmap_valid = target["interaction_heatmap_valid"].to(
                    device=valid.device, dtype=torch.bool
                )
                if heatmap_valid.ndim != 2:
                    raise RuntimeError(
                        "interaction_heatmap_valid must have shape "
                        "[batch, actors], got "
                        f"{tuple(heatmap_valid.shape)}"
                    )
                heatmap_valid = heatmap_valid & valid
                heatmap_info = self._exact_teacher_object_info(
                    actions,
                    valid,
                    target,
                    valid.device,
                )
                if heatmap_info is not None:
                    heatmap_actions = heatmap_info["actions"]
                    heatmap_objectless = self._labels_in_indices(
                        heatmap_actions,
                        self.objectless_action_indices,
                    )
                    known_object = valid & heatmap_info["known_action"]
                    teacher_missing = known_object & ~heatmap_info["valid_1based"]
                    teacher_mismatch = (
                        known_object
                        & heatmap_info["valid_1based"]
                        & ~heatmap_info["compatible_1based"]
                    )
                    exact_compatible = known_object & heatmap_info["compatible_1based"]

                    heatmap_valid = (heatmap_valid & ~teacher_missing) | (
                        valid & heatmap_objectless
                    )

                    if known_object.any():
                        known_count = int(known_object.sum().item())
                        self._log_scalar(
                            f"{stage}_interaction_heatmap_missing_object_masked_rate",
                            teacher_missing[known_object].float().mean(),
                            known_count,
                        )
                        self._log_scalar(
                            f"{stage}_interaction_heatmap_exact_compatible_valid_rate",
                            exact_compatible[known_object].float().mean(),
                            known_count,
                        )
                        self._log_scalar(
                            f"{stage}_interaction_heatmap_mismatch_valid_rate",
                            teacher_mismatch[known_object].float().mean(),
                            known_count,
                        )
                        self._log_count(
                            f"{stage}_interaction_heatmap_missing_object_masked_count",
                            teacher_missing.float().sum(),
                        )
                positive_heatmap_valid = target.get(
                    "interaction_heatmap_positive_valid",
                    target["interaction_heatmap_valid"],
                ).to(device=valid.device, dtype=torch.bool)
                if positive_heatmap_valid.shape != heatmap_valid.shape:
                    raise RuntimeError(
                        "interaction_heatmap_positive_valid must have shape "
                        f"{tuple(heatmap_valid.shape)}, got "
                        f"{tuple(positive_heatmap_valid.shape)}"
                    )
                positive_heatmap_valid = positive_heatmap_valid & heatmap_valid
                self._log_interaction_teacher_metrics(
                    actions,
                    valid,
                    positive_heatmap_valid,
                    target["interaction_cls"],
                    stage,
                )
                target_heatmap = target["interaction_heatmap"].to(
                    device=interaction_heatmap.device,
                    dtype=interaction_heatmap.dtype,
                )
                if target_heatmap.shape != interaction_heatmap.shape:
                    raise RuntimeError(
                        "Actor interaction heatmap target/prediction "
                        f"shape mismatch: {tuple(target_heatmap.shape)} vs "
                        f"{tuple(interaction_heatmap.shape)}"
                    )
                loss_interaction_heatmap = interaction_heatmap_loss(
                    interaction_heatmap,
                    target_heatmap,
                    heatmap_valid,
                )
                if loss_interaction_heatmap is None:
                    loss_interaction_heatmap = interaction_heatmap.new_zeros(())
                loss_interaction_raw_frobenius = heatmap_frobenius_loss(
                    interaction_heatmap,
                    target_heatmap,
                    valid=heatmap_valid,
                )
                loss_interaction_frobenius = loss_interaction_raw_frobenius
                if self.poguiseplus_normalized_heatmap_loss:
                    loss_interaction_heatmap_optimized = (
                        heatmap_mse_loss(
                            interaction_heatmap,
                            target_heatmap,
                            valid=heatmap_valid,
                        )
                        * self.poguiseplus_heatmap_mse_scale
                    )
                    self.log(
                        f"{stage}_loss_interaction_heatmap_mse_scaled",
                        loss_interaction_heatmap_optimized,
                        on_step=stage == "train",
                        on_epoch=True,
                        prog_bar=False,
                        logger=True,
                        sync_dist=True,
                    )
                else:
                    loss_interaction_heatmap_optimized = loss_interaction_frobenius
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
                    f"{stage}_loss_interaction_heatmap_raw_frobenius",
                    loss_interaction_raw_frobenius,
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
                        positive_heatmap_valid,
                        stage,
                        interaction_cls=target["interaction_cls"],
                    )

        if self.actor_poguiseplus_loss:
            if loss_pose_frobenius is None and loss_interaction_frobenius is None:
                raise RuntimeError(
                    "Actor PO-GUISE+ loss requires pose or interaction heatmaps"
                )
            heatmap_terms = []
            heatmap_report_terms = []
            if loss_pose_heatmap_optimized is not None:
                heatmap_terms.append(
                    loss_pose_heatmap_optimized
                    * self.poguiseplus_pose_heatmap_weight
                )
            if loss_pose_frobenius is not None:
                heatmap_report_terms.append(
                    loss_pose_frobenius * self.poguiseplus_pose_heatmap_weight
                )
            if loss_interaction_heatmap_optimized is not None:
                heatmap_terms.append(
                    loss_interaction_heatmap_optimized
                    * self.poguiseplus_interaction_heatmap_weight
                )
            if loss_interaction_frobenius is not None:
                heatmap_report_terms.append(
                    loss_interaction_frobenius
                    * self.poguiseplus_interaction_heatmap_weight
                )
            loss_heatmap_raw = torch.stack(heatmap_terms).sum()
            if self.poguiseplus_normalized_heatmap_loss:
                # Nash-MTL is more stable when task losses stay non-negative.
                loss_heatmap_task = torch.log1p(loss_heatmap_raw)
            else:
                loss_heatmap_task = torch.log(
                    loss_heatmap_raw + self.poguiseplus_heatmap_log_eps
                )
            loss_heatmap_report = torch.stack(heatmap_report_terms).sum()
            self.log(
                f"{stage}_loss_heatmap_frobenius",
                loss_heatmap_report,
                on_step=stage == "train",
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )
            self.log(
                f"{stage}_loss_heatmap_optimized",
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
            grounding_aux_terms.append(
                loss_heatmap_task * self.poguiseplus_heatmap_loss_weight
            )
            loss_aux_task = torch.stack(grounding_aux_terms).sum()
            self.log(
                f"{stage}_loss_main_deploy",
                loss_main_task,
                on_step=stage == "train",
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )
            self.log(
                f"{stage}_loss_grounding_aux",
                loss_aux_task,
                on_step=stage == "train",
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )
            if stage == "train" and self.model.hparams.grad_weights:
                loss = torch.stack([loss_main_task, loss_aux_task])
            elif self.model.hparams.get("kp_only", False):
                loss = loss_main_task * 1e-6 + loss_aux_task
            else:
                loss = loss_main_task + loss_aux_task
        else:
            loss = loss_main_task
            if grounding_aux_terms:
                loss = loss + torch.stack(grounding_aux_terms).sum()
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

    def _log_count(self, name, value):
        self.log(
            name,
            value,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
            reduce_fx="sum",
            batch_size=1,
        )

    def _labels_in_indices(self, labels, indices):
        if indices is None or int(indices.numel()) == 0:
            return torch.zeros_like(labels, dtype=torch.bool)
        indices = indices.to(device=labels.device, dtype=labels.dtype)
        return (labels.unsqueeze(-1) == indices).any(dim=-1)

    def _objectless_hard_negative_mask(
        self,
        full_preds,
        actions,
        valid,
        target,
    ):
        if not self.uses_object_proposals or "object_classes" not in target:
            return None
        if full_preds is None:
            return None
        object_classes = target["object_classes"].to(
            device=full_preds.device,
            dtype=torch.long,
        )
        object_valid = target["object_valid"].to(
            device=full_preds.device,
            dtype=torch.bool,
        )
        valid = valid.to(device=full_preds.device, dtype=torch.bool)
        actions = actions.to(device=full_preds.device, dtype=torch.long)
        hard_ids = torch.tensor(
            sorted(self.hard_negative_object_ids),
            device=full_preds.device,
            dtype=torch.long,
        )
        if hard_ids.numel() == 0:
            return None
        visible_hard_object = (
            object_valid
            & (object_classes.unsqueeze(-1) == hard_ids).any(dim=-1)
        )
        sample_has_hard_object = visible_hard_object.any(dim=1)
        objectless = self._labels_in_indices(actions, self.objectless_action_indices)
        return valid & objectless & sample_has_hard_object[:, None]

    def _objectless_object_action_suppression_loss(
        self,
        stage,
        full_preds,
        actions,
        valid,
        target,
    ):
        if self.objectless_object_action_suppression_loss_weight <= 0:
            return None
        hard_mask = self._objectless_hard_negative_mask(
            full_preds,
            actions,
            valid,
            target,
        )
        object_action_indices = self.group_indices.get("object_mapped")
        if (
            hard_mask is None
            or object_action_indices is None
            or int(object_action_indices.numel()) == 0
        ):
            return None
        if not hard_mask.any():
            return None

        object_action_indices = object_action_indices.to(
            device=full_preds.device,
            dtype=torch.long,
        )
        probs = full_preds[hard_mask].float().softmax(dim=-1)
        object_action_prob = probs[:, object_action_indices].sum(dim=-1).clamp(
            0.0,
            1.0 - 1e-6,
        )
        loss = -torch.log1p(-object_action_prob).mean()
        count = int(object_action_prob.numel())
        self.log(
            f"{stage}_loss_objectless_object_action_suppression",
            loss,
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        self._log_scalar(
            f"{stage}_objectless_with_object_visible_object_action_prob",
            object_action_prob.detach().mean(),
            count,
        )
        return loss

    def _log_objectless_hard_negative_metrics(
        self,
        stage,
        full_preds,
        actions,
        valid,
        target,
    ):
        hard_mask = self._objectless_hard_negative_mask(
            full_preds,
            actions,
            valid,
            target,
        )
        if hard_mask is None:
            return
        object_classes = target["object_classes"].to(
            device=full_preds.device,
            dtype=torch.long,
        )
        object_valid = target["object_valid"].to(
            device=full_preds.device,
            dtype=torch.bool,
        )
        valid = valid.to(device=full_preds.device, dtype=torch.bool)
        actions = actions.to(device=full_preds.device, dtype=torch.long)
        if not hard_mask.any():
            return
        objectless = self._labels_in_indices(actions, self.objectless_action_indices)

        pred_labels = full_preds.argmax(dim=-1)
        correct = pred_labels == actions
        count = int(hard_mask.sum().item())
        self._log_scalar(
            f"{stage}_objectless_with_object_visible_acc",
            correct[hard_mask].float().mean(),
            count,
        )
        self._log_count(
            f"{stage}_objectless_with_object_visible_count",
            hard_mask.float().sum(),
        )
        object_action_pred = self._labels_in_indices(
            pred_labels,
            self.group_indices.get("object_mapped"),
        )
        self._log_scalar(
            f"{stage}_objectless_with_object_visible_object_action_pred_rate",
            object_action_pred[hard_mask].float().mean(),
            count,
        )

        for object_id, object_name in sorted(self.hard_negative_object_ids.items()):
            object_mask = (
                object_valid
                & (object_classes == int(object_id))
            ).any(dim=1)
            slot_mask = valid & objectless & object_mask[:, None]
            if not slot_mask.any():
                continue
            slot_count = int(slot_mask.sum().item())
            self._log_scalar(
                f"{stage}_objectless_with_{object_name}_visible_acc",
                correct[slot_mask].float().mean(),
                slot_count,
            )
            self._log_count(
                f"{stage}_objectless_with_{object_name}_visible_count",
                slot_mask.float().sum(),
            )

    def _action_index(self, action_name):
        if action_name not in self.action_to_index:
            return None
        return int(self.action_to_index[action_name])

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
        object_inputs = (
            self._empty_object_inputs(batch_size, imgs.device)
            if self.uses_object_proposals
            else {}
        )
        data = self.model(
            imgs,
            boxes=diag_boxes,
            valid=diag_valid,
            **object_inputs,
        )
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
        object_inputs = (
            self._empty_object_inputs(batch_size, imgs.device)
            if self.uses_object_proposals
            else {}
        )
        data = self.model(
            imgs,
            boxes=diag_boxes,
            valid=diag_valid,
            **object_inputs,
        )
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
            object_inputs = (
                self._empty_object_inputs(pair_imgs.shape[0], pair_imgs.device)
                if self.uses_object_proposals
                else {}
            )
            data = self.model(
                pair_imgs,
                boxes=pair_boxes,
                valid=pair_valid,
                **object_inputs,
            )
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

    def _flatten_gathered_validation_tensor(self, outputs, name, trailing_dim=None):
        tensors = []
        for data in outputs.get(name, []):
            tensor = torch.as_tensor(data)
            if tensor.numel() == 0:
                continue
            if trailing_dim is None:
                tensor = tensor.reshape(-1)
            else:
                tensor = tensor.reshape(-1, int(trailing_dim))
            tensors.append(tensor)
        if not tensors:
            return None
        return torch.cat(tensors, dim=0)

    def _deploy_accuracy(self, pred_labels, labels, mask):
        mask = mask.to(device=labels.device, dtype=torch.bool)
        if not mask.any():
            return None
        return (pred_labels[mask] == labels[mask]).float().mean()

    def _deploy_group_accuracy(self, pred_labels, labels, group_name):
        indices = self.group_indices.get(group_name)
        if indices is None:
            return None
        mask = self._labels_in_indices(labels, indices)
        return self._deploy_accuracy(pred_labels, labels, mask)

    def _deploy_action_accuracies(self, pred_labels, labels, action_names):
        values = []
        for action_name in action_names:
            action_idx = self._action_index(action_name)
            if action_idx is None:
                continue
            value = self._deploy_accuracy(pred_labels, labels, labels == action_idx)
            if value is not None:
                values.append(value)
        return values

    def _weighted_deploy_score(self, components, penalties):
        weighted_values = []
        weights = []
        for value, weight in components:
            if value is None:
                continue
            if not torch.isfinite(value):
                continue
            weighted_values.append(value.float() * float(weight))
            weights.append(float(weight))
        if not weighted_values:
            return None
        score = torch.stack(weighted_values).sum() / max(sum(weights), 1e-6)
        for value, weight in penalties:
            if value is None:
                continue
            if torch.isfinite(value):
                score = score - value.float() * float(weight)
        return score

    def _log_deploy_checkpoint_metrics(self, preds, labels, gathered_outputs):
        if preds is None or labels is None or preds.numel() == 0 or labels.numel() == 0:
            return
        labels = labels.to(dtype=torch.long)
        pred_labels = preds.argmax(dim=-1)
        macro_acc = torchmetrics.functional.accuracy(
            preds,
            labels,
            task="multiclass",
            num_classes=self.num_classes,
            average="macro",
        )
        macro_f1 = torchmetrics.functional.f1_score(
            preds,
            labels,
            task="multiclass",
            num_classes=self.num_classes,
            average="macro",
        )
        object_mapped_acc = self._deploy_group_accuracy(
            pred_labels,
            labels,
            "object_mapped",
        )
        objectless_acc = self._deploy_group_accuracy(
            pred_labels,
            labels,
            "objectless",
        )
        hard_objectless = self._flatten_gathered_validation_tensor(
            gathered_outputs,
            "hard_objectless",
        )
        hard_objectless_acc = None
        hard_object_action_rate = None
        if hard_objectless is not None:
            hard_objectless = hard_objectless.to(
                device=labels.device,
                dtype=torch.bool,
            )
            if hard_objectless.numel() != labels.numel():
                raise RuntimeError(
                    "hard_objectless validation mask length does not match labels: "
                    f"{hard_objectless.numel()} vs {labels.numel()}"
                )
            hard_objectless_acc = self._deploy_accuracy(
                pred_labels,
                labels,
                hard_objectless,
            )
            object_action_pred = self._labels_in_indices(
                pred_labels,
                self.group_indices.get("object_mapped"),
            )
            hard_object_action_rate = (
                object_action_pred[hard_objectless].float().mean()
                if hard_objectless.any()
                else None
            )

        key_action_values = self._deploy_action_accuracies(
            pred_labels,
            labels,
            DEPLOY_KEY_ACTIONS,
        )
        key_action_mean = torch.stack(key_action_values).mean() if key_action_values else None
        key_action_min = torch.stack(key_action_values).amin() if key_action_values else None
        key_action_floor_deficit = (
            (torch.tensor(0.60, device=labels.device) - key_action_min).clamp_min(0.0)
            if key_action_min is not None
            else None
        )

        deploy_score = self._weighted_deploy_score(
            components=[
                (macro_f1, 0.25),
                (macro_acc, 0.15),
                (object_mapped_acc, 0.20),
                (objectless_acc, 0.15),
                (hard_objectless_acc, 0.15),
                (key_action_mean, 0.10),
            ],
            penalties=[
                (hard_object_action_rate, 0.20),
                (key_action_floor_deficit, 0.15),
            ],
        )
        if deploy_score is None:
            return

        for name, value in (
            ("val_deploy_score", deploy_score),
            ("val_deploy_key_action_mean", key_action_mean),
            ("val_deploy_key_action_min", key_action_min),
            (
                "val_deploy_objectless_with_object_visible_acc",
                hard_objectless_acc,
            ),
            (
                "val_deploy_objectless_with_object_visible_object_action_pred_rate",
                hard_object_action_rate,
            ),
        ):
            if value is None:
                continue
            self.log(
                name,
                value,
                prog_bar=name == "val_deploy_score",
                logger=True,
                sync_dist=False,
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
            (
                loss,
                preds,
                labels,
                hm,
                loss_kp,
                full_preds,
                presence_logits,
            ) = self._actor_step(
                imgs, target, self.val_loss, "val"
            )
            self._log_actor_val_diagnostics(imgs, target, presence_logits)
            self._log_objectless_hard_negative_metrics(
                "val",
                full_preds,
                target["actions"].long(),
                target["valid"].bool(),
                target,
            )
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
            hard_mask = self._objectless_hard_negative_mask(
                full_preds,
                target["actions"].long(),
                target["valid"].bool(),
                target,
            )
            if hard_mask is not None:
                valid_mask = target["valid"].bool().to(device=full_preds.device)
                self.validation_step_outputs["hard_objectless"].append(
                    hard_mask[valid_mask].detach()
                )
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
        preds = self._flatten_gathered_validation_tensor(
            outputs,
            "preds",
            trailing_dim=self.num_classes,
        )
        labels = self._flatten_gathered_validation_tensor(outputs, "labels")
        if preds is None or labels is None:
            self.validation_step_outputs.clear()
            self.validation_step_outputs = self._empty_validation_outputs()
            return super().on_validation_epoch_end()
        labels = labels.to(dtype=torch.long)

        loss = self.val_loss(preds, labels)
        self._log_deploy_checkpoint_metrics(preds, labels, outputs)
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
        self.validation_step_outputs = self._empty_validation_outputs()
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
            object_inputs = self._object_inputs_from_target(target, imgs.device)
            data = self.model(
                imgs,
                boxes=boxes,
                valid=valid,
                **object_inputs,
            )
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
