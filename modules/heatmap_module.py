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
    toyota_action_object_map,
    toyota_confuser_action_names,
    toyota_action_to_index,
    toyota_group_action_names,
    toyota_label_dict,
    toyota_objectless_action_names,
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
                "scene_object_tokens was removed. Use actor_object_prompt_tokens=1 "
                "for runtime object prompts."
            )
        self.actor_object_residual_head = bool(
            hparams.get("actor_object_residual_head", 0)
        )
        self.actor_object_prompt_tokens = bool(
            hparams.get("actor_object_prompt_tokens", 0)
        )
        actor_object_slot_head = bool(hparams.get("actor_object_slot_head", 0))
        if actor_object_slot_head:
            raise ValueError(
                "actor_object_slot_head was replaced by "
                "actor_object_prompt_tokens. Set --actor_object_prompt_tokens 1 "
                "and keep --actor_object_slot_head 0."
            )
        if self.actor_object_residual_head and not self.actor_object_prompt_tokens:
            raise ValueError(
                "actor_object_residual_head requires actor_object_prompt_tokens"
            )
        self.uses_object_proposals = self.actor_object_prompt_tokens
        if self.actor_object_prompt_tokens and not self.actor_prompt:
            raise ValueError("actor_object_prompt_tokens requires actor_prompt")
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
        self.poguiseplus_interaction_heatmap_pos_loss_weight = float(
            hparams.get("poguiseplus_interaction_heatmap_pos_loss_weight", 0.0)
        )
        if self.poguiseplus_interaction_heatmap_pos_loss_weight < 0:
            raise ValueError(
                "poguiseplus_interaction_heatmap_pos_loss_weight must be >= 0"
            )
        self.poguiseplus_interaction_heatmap_pos_weight = float(
            hparams.get("poguiseplus_interaction_heatmap_pos_weight", 8.0)
        )
        if self.poguiseplus_interaction_heatmap_pos_weight < 0:
            raise ValueError("poguiseplus_interaction_heatmap_pos_weight must be >= 0")
        self.poguiseplus_interaction_heatmap_center_loss_weight = float(
            hparams.get("poguiseplus_interaction_heatmap_center_loss_weight", 0.0)
        )
        if self.poguiseplus_interaction_heatmap_center_loss_weight < 0:
            raise ValueError(
                "poguiseplus_interaction_heatmap_center_loss_weight must be >= 0"
            )
        self.poguiseplus_interaction_heatmap_center_temperature = float(
            hparams.get("poguiseplus_interaction_heatmap_center_temperature", 10.0)
        )
        if self.poguiseplus_interaction_heatmap_center_temperature <= 0:
            raise ValueError(
                "poguiseplus_interaction_heatmap_center_temperature must be > 0"
            )
        self.motion_aux_loss_weight = float(
            hparams.get("motion_aux_loss_weight", DEFAULT_MOTION_AUX_LOSS_WEIGHT)
        )
        if self.motion_aux_loss_weight < 0:
            raise ValueError("motion_aux_loss_weight must be >= 0")
        self.object_residual_prompt_relation_loss_weight = float(
            hparams.get("object_residual_prompt_relation_loss_weight", 0.0)
        )
        if self.object_residual_prompt_relation_loss_weight < 0:
            raise ValueError("object_residual_prompt_relation_loss_weight must be >= 0")
        prompt_relation_final = hparams.get(
            "object_residual_prompt_relation_loss_final_weight",
            None,
        )
        self.object_residual_prompt_relation_loss_final_weight = (
            self.object_residual_prompt_relation_loss_weight
            if prompt_relation_final is None
            else float(prompt_relation_final)
        )
        if self.object_residual_prompt_relation_loss_final_weight < 0:
            raise ValueError(
                "object_residual_prompt_relation_loss_final_weight must be >= 0"
            )
        self.object_residual_relation_loss_decay_start_epoch = int(
            hparams.get("object_residual_relation_loss_decay_start_epoch", 0)
        )
        self.object_residual_relation_loss_decay_end_epoch = int(
            hparams.get("object_residual_relation_loss_decay_end_epoch", 0)
        )
        if self.object_residual_relation_loss_decay_start_epoch < 0:
            raise ValueError("object_residual_relation_loss_decay_start_epoch must be >= 0")
        if self.object_residual_relation_loss_decay_end_epoch < 0:
            raise ValueError("object_residual_relation_loss_decay_end_epoch must be >= 0")
        if (
            self.object_residual_relation_loss_decay_end_epoch
            < self.object_residual_relation_loss_decay_start_epoch
        ):
            raise ValueError(
                "object_residual_relation_loss_decay_end_epoch must be >= "
                "object_residual_relation_loss_decay_start_epoch"
            )
        self.object_residual_relation_confuser_margin = float(
            hparams.get("object_residual_relation_confuser_margin", 1.0)
        )
        if self.object_residual_relation_confuser_margin < 0:
            raise ValueError("object_residual_relation_confuser_margin must be >= 0")
        self.object_residual_null_relation_loss_weight = float(
            hparams.get(
                "object_residual_null_relation_loss_weight",
                0.25 if self.actor_object_residual_head else 0.0,
            )
        )
        if self.object_residual_null_relation_loss_weight < 0:
            raise ValueError("object_residual_null_relation_loss_weight must be >= 0")
        self.objectless_object_action_suppression_loss_weight = float(
            hparams.get("objectless_object_action_suppression_loss_weight", 0.3)
        )
        if self.objectless_object_action_suppression_loss_weight < 0:
            raise ValueError(
                "objectless_object_action_suppression_loss_weight must be >= 0"
            )
        self.object_prompt_grounding_loss_weight = float(
            hparams.get("object_prompt_grounding_loss_weight", 0.0)
        )
        if self.object_prompt_grounding_loss_weight < 0:
            raise ValueError("object_prompt_grounding_loss_weight must be >= 0")
        self.objectless_prompt_consistency_loss_weight = float(
            hparams.get("objectless_prompt_consistency_loss_weight", 0.0)
        )
        if self.objectless_prompt_consistency_loss_weight < 0:
            raise ValueError("objectless_prompt_consistency_loss_weight must be >= 0")
        self.object_prompt_wrong_class_loss_weight = float(
            hparams.get("object_prompt_wrong_class_loss_weight", 0.0)
        )
        if self.object_prompt_wrong_class_loss_weight < 0:
            raise ValueError("object_prompt_wrong_class_loss_weight must be >= 0")
        self.object_prompt_wrong_class_margin = float(
            hparams.get("object_prompt_wrong_class_margin", 0.20)
        )
        if self.object_prompt_wrong_class_margin < 0:
            raise ValueError("object_prompt_wrong_class_margin must be >= 0")
        self.object_prompt_sensitivity_loss_weight = float(
            hparams.get("object_prompt_sensitivity_loss_weight", 0.0)
        )
        if self.object_prompt_sensitivity_loss_weight < 0:
            raise ValueError("object_prompt_sensitivity_loss_weight must be >= 0")
        self.object_prompt_sensitivity_margin = float(
            hparams.get("object_prompt_sensitivity_margin", 0.20)
        )
        if self.object_prompt_sensitivity_margin < 0:
            raise ValueError("object_prompt_sensitivity_margin must be >= 0")
        self.object_prompt_sensitivity_motion_margin_threshold = float(
            hparams.get("object_prompt_sensitivity_motion_margin_threshold", 1.0)
        )
        self.object_class_dropout_prob = float(
            hparams.get("object_class_dropout_prob", 0.0)
        )
        if not 0.0 <= self.object_class_dropout_prob <= 1.0:
            raise ValueError("object_class_dropout_prob must be in [0, 1]")
        self.object_class_wrong_prob = float(
            hparams.get("object_class_wrong_prob", 0.0)
        )
        if not 0.0 <= self.object_class_wrong_prob <= 1.0:
            raise ValueError("object_class_wrong_prob must be in [0, 1]")
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
            self.action_confuser_indices_by_index = (
                self._build_action_confuser_indices_by_index()
            )
            self.action_wrong_object_ids_by_index = (
                self._build_action_wrong_object_ids_by_index()
            )
        else:
            self.action_names = [str(index) for index in range(self.num_classes)]
            self.action_to_index = {}
            self.action_object_map = {}
            self.action_object_ids_by_index = {}
            self.action_confuser_indices_by_index = {}
            self.action_wrong_object_ids_by_index = {}

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
        self.label_smoothing = float(hparams.get("label_smoothing", 0.0) or 0.0)
        if self.actor_prompt:
            self.train_loss = nn.CrossEntropyLoss(
                label_smoothing=self.label_smoothing
            )
        elif self.label_smoothing and hparams.mixup:
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
                "model.net.object_slot_embed",
                "model.net.object_class_embed",
                "model.net.object_box_mlp",
                "model.net.object_conf_mlp",
                "model.net.object_valid_embed",
                "model.actor_head",
                "model.actor_motion_head",
                "model.presence_head",
                "model.actor_object_base_fusion",
                "model.object_residual_action_head",
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
        )

    def _actor_prompt_param_name(self, name):
        return name.startswith(
            (
                "actor_token",
                "actor_slot_embed",
                "valid_embed",
                "bbox_mlp",
                "object_slot_embed",
                "object_class_embed",
                "object_box_mlp",
                "object_conf_mlp",
                "object_valid_embed",
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
            if getattr(self.model, "actor_head", None) is not None:
                params += list(self.model.actor_head.parameters())
            if getattr(self.model, "actor_motion_head", None) is not None:
                params += list(self.model.actor_motion_head.parameters())
            if self.model.presence_head is not None:
                params += list(self.model.presence_head.parameters())
            actor_object_base_fusion = getattr(
                self.model,
                "actor_object_base_fusion",
                None,
            )
            if actor_object_base_fusion is not None:
                params += list(actor_object_base_fusion.parameters())
            object_residual_head = getattr(
                self.model,
                "object_residual_action_head",
                None,
            )
            if object_residual_head is not None:
                params += list(object_residual_head.parameters())
            for name, param in self.model.net.named_parameters():
                if self._actor_prompt_param_name(name):
                    params.append(param)
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

    def _build_action_confuser_indices_by_index(self):
        action_confusers = {}
        for action_name, action_idx in self.action_to_index.items():
            confuser_indices = [
                int(self.action_to_index[confuser_name])
                for confuser_name in toyota_confuser_action_names(
                    action_name,
                    self.task_type,
                    self.action_taxonomy,
                )
                if confuser_name in self.action_to_index
            ]
            if confuser_indices:
                action_confusers[int(action_idx)] = torch.tensor(
                    confuser_indices,
                    dtype=torch.long,
                )
        return action_confusers

    def _build_action_wrong_object_ids_by_index(self):
        default_wrong_names = (
            "book",
            "laptop",
            "phone",
            "tv_monitor",
            "cup",
            "bottle",
            "glass",
        )
        default_wrong_ids = [
            int(OBJECT_TO_ID[name])
            for name in default_wrong_names
            if name in OBJECT_TO_ID
        ]
        action_wrong_objects = {}
        for action_name, action_idx in self.action_to_index.items():
            true_ids = set()
            for object_name in self.action_object_map.get(action_name, ()):
                if object_name in OBJECT_TO_ID:
                    true_ids.add(int(OBJECT_TO_ID[object_name]))

            wrong_ids = []
            for confuser_name in toyota_confuser_action_names(
                action_name,
                self.task_type,
                self.action_taxonomy,
            ):
                for object_name in self.action_object_map.get(confuser_name, ()):
                    if object_name not in OBJECT_TO_ID:
                        continue
                    object_id = int(OBJECT_TO_ID[object_name])
                    if object_id not in true_ids and object_id not in wrong_ids:
                        wrong_ids.append(object_id)

            if not wrong_ids:
                wrong_ids = [
                    object_id
                    for object_id in default_wrong_ids
                    if object_id not in true_ids
                ]
            if wrong_ids:
                action_wrong_objects[int(action_idx)] = torch.tensor(
                    wrong_ids,
                    dtype=torch.long,
                )
        return action_wrong_objects

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

    def _positive_balanced_interaction_heatmap_mse(
        self,
        pred_heatmap,
        target_heatmap,
        valid,
    ):
        if pred_heatmap.shape != target_heatmap.shape:
            raise RuntimeError(
                "positive-balanced interaction heatmap shape mismatch: "
                f"{tuple(pred_heatmap.shape)} vs {tuple(target_heatmap.shape)}"
            )
        valid = valid.to(device=pred_heatmap.device, dtype=torch.bool)
        if valid.shape != pred_heatmap.shape[:-2]:
            raise RuntimeError(
                "positive-balanced interaction heatmap valid shape mismatch: "
                f"{tuple(valid.shape)} vs {tuple(pred_heatmap.shape[:-2])}"
            )
        if not valid.any():
            return pred_heatmap.sum() * 0.0
        pred = pred_heatmap.float()
        target = target_heatmap.to(device=pred_heatmap.device).float()
        err = (pred - target).pow(2)
        weight = 1.0 + self.poguiseplus_interaction_heatmap_pos_weight * target
        weight = weight * valid[..., None, None].to(dtype=weight.dtype)
        return (err * weight).sum() / weight.sum().clamp_min(1e-6)

    @staticmethod
    def _heatmap_soft_center(heatmap, temperature):
        if heatmap.ndim != 3:
            raise RuntimeError(
                "soft heatmap center expects [N,H,W], got "
                f"{tuple(heatmap.shape)}"
            )
        n, height, width = heatmap.shape
        flat = (heatmap.float() * float(temperature)).flatten(1)
        prob = torch.softmax(flat, dim=-1).view(n, height, width)
        y = torch.linspace(
            0.0,
            1.0,
            height,
            device=heatmap.device,
            dtype=prob.dtype,
        )
        x = torch.linspace(
            0.0,
            1.0,
            width,
            device=heatmap.device,
            dtype=prob.dtype,
        )
        cy = (prob.sum(dim=2) * y[None, :]).sum(dim=1)
        cx = (prob.sum(dim=1) * x[None, :]).sum(dim=1)
        return torch.stack([cx, cy], dim=-1)

    def _interaction_heatmap_center_loss(self, pred_heatmap, target_heatmap, valid):
        if pred_heatmap.shape != target_heatmap.shape:
            raise RuntimeError(
                "interaction heatmap center loss shape mismatch: "
                f"{tuple(pred_heatmap.shape)} vs {tuple(target_heatmap.shape)}"
            )
        valid = valid.to(device=pred_heatmap.device, dtype=torch.bool)
        if valid.shape != pred_heatmap.shape[:-2]:
            raise RuntimeError(
                "interaction heatmap center valid shape mismatch: "
                f"{tuple(valid.shape)} vs {tuple(pred_heatmap.shape[:-2])}"
            )
        if not valid.any():
            return pred_heatmap.sum() * 0.0
        pred = pred_heatmap.float()[valid]
        target = target_heatmap.to(device=pred_heatmap.device).float()[valid]
        temperature = self.poguiseplus_interaction_heatmap_center_temperature
        pred_center = self._heatmap_soft_center(pred, temperature)
        target_center = self._heatmap_soft_center(target, temperature).detach()
        return F.smooth_l1_loss(pred_center, target_center)

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
                "Runtime object proposal paths require dataset object targets; "
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

    def _object_class_dropout_inputs(self, object_inputs, stage):
        if stage != "train" or not object_inputs:
            return object_inputs
        class_drop = float(self.object_class_dropout_prob)
        class_wrong = float(self.object_class_wrong_prob)
        if class_drop <= 0.0 and class_wrong <= 0.0:
            return object_inputs

        output = {name: value.clone() for name, value in object_inputs.items()}
        object_valid = output["object_valid"]
        object_classes = output["object_classes"]
        if not object_valid.any():
            return output

        none_id = int(self.model.hparams.get("num_object_classes", NUM_OBJECT_CLASSES))
        if class_drop > 0.0:
            drop_mask = (
                object_valid
                & (torch.rand_like(output["object_confs"]) < class_drop)
            )
            object_classes[drop_mask] = none_id
            self._log_scalar(
                "train_object_class_dropout_rate",
                drop_mask.float().mean(),
                int(drop_mask.numel()),
            )
        if class_wrong > 0.0:
            wrong_mask = (
                object_valid
                & (object_classes != none_id)
                & (torch.rand_like(output["object_confs"]) < class_wrong)
            )
            if wrong_mask.any():
                num_classes = int(
                    self.model.hparams.get("num_object_classes", NUM_OBJECT_CLASSES)
                )
                replacement = torch.randint(
                    low=0,
                    high=num_classes,
                    size=object_classes.shape,
                    device=object_classes.device,
                    dtype=object_classes.dtype,
                )
                object_classes[wrong_mask] = replacement[wrong_mask]
            self._log_scalar(
                "train_object_class_wrong_rate",
                wrong_mask.float().mean(),
                int(wrong_mask.numel()),
            )
        return output

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

    def _object_prompt_grounding_loss(self, stage, actions, valid, target):
        if (
            not self.actor_object_prompt_tokens
            or self.object_prompt_grounding_loss_weight <= 0
        ):
            return None
        prompt_logits = getattr(
            self.model,
            "last_actor_object_prompt_attention_logits",
            None,
        )
        if prompt_logits is None:
            return None
        info = self._exact_teacher_object_info(
            actions,
            valid,
            target,
            prompt_logits.device,
        )
        if info is None:
            return None
        exact_compatible = (
            info["valid"]
            & info["known_action"]
            & info["compatible_1based"]
        )
        if not exact_compatible.any():
            return None
        target_slots = info["idx_1based"][exact_compatible].to(
            device=prompt_logits.device,
            dtype=torch.long,
        )
        logits = prompt_logits[exact_compatible].float()
        if logits.ndim != 2 or logits.shape[0] != target_slots.shape[0]:
            raise RuntimeError(
                "object prompt grounding logits must have shape [N,K], got "
                f"{tuple(logits.shape)} for {int(target_slots.shape[0])} targets"
            )
        if (target_slots >= logits.shape[-1]).any():
            raise RuntimeError(
                "object prompt teacher slot is outside prompt-token count"
            )
        loss = F.cross_entropy(logits, target_slots)
        count = int(target_slots.numel())
        self.log(
            f"{stage}_loss_object_prompt_grounding",
            loss,
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        with torch.no_grad():
            probs = torch.softmax(logits, dim=-1)
            pred_slots = logits.argmax(dim=-1)
            true_prob = probs.gather(1, target_slots.unsqueeze(1)).squeeze(1)
            self._log_scalar(
                f"{stage}_object_prompt_grounding_acc",
                (pred_slots == target_slots).float().mean(),
                count,
            )
            self._log_scalar(
                f"{stage}_object_prompt_grounding_true_prob",
                true_prob.mean(),
                count,
            )
        return loss

    def _objectless_prompt_consistency_loss(
        self,
        stage,
        imgs,
        boxes,
        valid,
        actions,
        preds,
        target,
        object_inputs,
    ):
        if (
            stage != "train"
            or not self.actor_object_prompt_tokens
            or self.objectless_prompt_consistency_loss_weight <= 0
            or "object_valid" not in target
        ):
            return None
        device = preds.device
        valid = valid.to(device=device, dtype=torch.bool)
        actions = actions.to(device=device, dtype=torch.long)
        object_visible = target["object_valid"].to(device=device, dtype=torch.bool)
        object_visible = object_visible.any(dim=1)
        objectless = self._labels_in_indices(actions, self.objectless_action_indices)
        consistency_mask = valid & objectless & object_visible[:, None]
        if not consistency_mask.any():
            return None

        empty_inputs = self._empty_object_inputs(
            batch_size=int(imgs.shape[0]),
            device=imgs.device,
            dtype=torch.float32,
        )
        distractor_inputs = self._object_distractor_class_inputs(
            object_inputs
        )
        with torch.no_grad():
            no_object_preds = self._cached_object_prompt_logits(empty_inputs).detach()
            distractor_preds = self._cached_object_prompt_logits(
                distractor_inputs
            ).detach()

        with_object_logp = F.log_softmax(preds[consistency_mask].float(), dim=-1)
        no_object_prob = F.softmax(
            no_object_preds[consistency_mask].float(),
            dim=-1,
        )
        distractor_prob = F.softmax(
            distractor_preds[consistency_mask].float(),
            dim=-1,
        )
        loss_no_object = F.kl_div(
            with_object_logp,
            no_object_prob,
            reduction="batchmean",
        )
        loss_distractor = F.kl_div(
            with_object_logp,
            distractor_prob,
            reduction="batchmean",
        )
        loss = 0.5 * (loss_no_object + loss_distractor)
        count = int(consistency_mask.sum().item())
        self.log(
            f"{stage}_loss_objectless_prompt_consistency",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        with torch.no_grad():
            pred_with = preds[consistency_mask].argmax(dim=-1)
            pred_without = no_object_preds[consistency_mask].argmax(dim=-1)
            pred_distractor = distractor_preds[consistency_mask].argmax(dim=-1)
            self._log_scalar(
                "train_objectless_prompt_consistency_pred_match",
                (pred_with == pred_without).float().mean(),
                count,
            )
            self._log_scalar(
                "train_objectless_prompt_distractor_pred_match",
                (pred_with == pred_distractor).float().mean(),
                count,
            )
            self._log_scalar(
                "train_objectless_prompt_consistency_kl",
                loss.detach(),
                count,
            )
            self._log_scalar(
                "train_objectless_prompt_distractor_kl",
                loss_distractor.detach(),
                count,
            )
        return loss

    def _cached_object_prompt_logits(self, object_inputs=None):
        if not hasattr(self.model, "cached_object_prompt_action_logits"):
            raise RuntimeError(
                "Object prompt counterfactuals require cached_object_prompt_action_logits"
            )
        object_inputs = object_inputs or {}
        return self.model.cached_object_prompt_action_logits(
            object_classes=object_inputs.get("object_classes"),
            object_valid=object_inputs.get("object_valid"),
        )

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

    @staticmethod
    def _first_actor_per_sample_mask(mask):
        if mask.ndim != 2:
            raise RuntimeError(
                "first-actor mask expects [B,A], got "
                f"{tuple(mask.shape)}"
            )
        output = torch.zeros_like(mask, dtype=torch.bool)
        for batch_idx in range(mask.shape[0]):
            slots = torch.nonzero(mask[batch_idx], as_tuple=False)
            if int(slots.numel()) == 0:
                continue
            output[batch_idx, int(slots[0, 0].item())] = True
        return output

    def _wrong_object_ids_for_actions(self, actions):
        values = []
        for action in actions.detach().cpu().tolist():
            object_ids = self.action_wrong_object_ids_by_index.get(int(action))
            if object_ids is None or int(object_ids.numel()) == 0:
                values.append(int(NUM_OBJECT_CLASSES))
            else:
                values.append(int(object_ids[0].item()))
        return torch.tensor(values, device=actions.device, dtype=torch.long)

    def _teacher_object_wrong_class_inputs(
        self,
        object_inputs,
        object_slots,
        selected_valid,
        actions,
    ):
        output = {name: value.clone() for name, value in object_inputs.items()}
        if not selected_valid.any():
            return output
        rows = torch.nonzero(selected_valid, as_tuple=False)
        batch_idx = rows[:, 0]
        actor_actions = actions[selected_valid].to(dtype=torch.long)
        wrong_ids = self._wrong_object_ids_for_actions(actor_actions)
        object_slots = object_slots[selected_valid].to(
            device=batch_idx.device,
            dtype=torch.long,
        )
        valid_wrong = wrong_ids < int(NUM_OBJECT_CLASSES)
        if not valid_wrong.any():
            return output
        batch_idx = batch_idx[valid_wrong]
        object_slots = object_slots[valid_wrong]
        wrong_ids = wrong_ids[valid_wrong]
        output["object_valid"][batch_idx, object_slots] = True
        output["object_classes"][batch_idx, object_slots] = wrong_ids
        return output

    def _object_distractor_class_inputs(self, object_inputs):
        output = {name: value.clone() for name, value in object_inputs.items()}
        object_valid = output.get("object_valid")
        object_classes = output.get("object_classes")
        if object_valid is None or object_classes is None or not object_valid.any():
            return output
        distractor_ids = torch.tensor(
            [
                int(OBJECT_TO_ID[name])
                for name in ("laptop", "book", "phone", "tv_monitor")
                if name in OBJECT_TO_ID
            ],
            device=object_classes.device,
            dtype=torch.long,
        )
        if int(distractor_ids.numel()) == 0:
            return output
        slot_ids = torch.arange(
            object_classes.shape[1],
            device=object_classes.device,
            dtype=torch.long,
        )
        replacement = distractor_ids[slot_ids.remainder(distractor_ids.numel())]
        replacement = replacement.unsqueeze(0).expand_as(object_classes)
        output["object_classes"] = torch.where(
            object_valid,
            replacement,
            object_classes,
        )
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
        with torch.no_grad():
            counterfactual_preds = self._cached_object_prompt_logits(
                counterfactual_inputs
            )

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

        base_true_probs = self._action_probs(preds)[selected_valid].gather(
            1,
            true_labels.unsqueeze(1),
        ).squeeze(1)
        counterfactual_true_probs = (
            self._action_probs(counterfactual_preds)[selected_valid]
            .gather(1, true_labels.unsqueeze(1))
            .squeeze(1)
        )
        self._log_scalar(
            f"{stage}_object_counterfactual_teacher_prob_drop",
            (base_true_probs - counterfactual_true_probs).mean(),
            count,
        )
        return None

    def _log_object_prompt_drop_eval(
        self,
        imgs,
        boxes,
        valid,
        actions,
        preds,
        target,
        stage,
    ):
        if not self.uses_object_proposals or stage == "train":
            return None
        if "object_valid" not in target:
            return None

        device = preds.device
        valid = valid.to(device=device, dtype=torch.bool)
        actions = actions.to(device=device, dtype=torch.long)
        object_visible = target["object_valid"].to(device=device, dtype=torch.bool)
        object_visible = object_visible.any(dim=1)
        objectless = self._labels_in_indices(actions, self.objectless_action_indices)
        objectless_visible = valid & objectless & object_visible[:, None]

        exact_compatible = None
        info = self._exact_teacher_object_info(actions, valid, target, device)
        if info is not None:
            exact_compatible = (
                info["valid"]
                & info["known_action"]
                & info["compatible_1based"]
            )

        has_objectless = bool(objectless_visible.any().item())
        has_exact = (
            exact_compatible is not None
            and bool(exact_compatible.any().item())
        )
        if not has_objectless and not has_exact:
            return None

        empty_inputs = self._empty_object_inputs(
            batch_size=int(imgs.shape[0]),
            device=imgs.device,
            dtype=torch.float32,
        )
        distractor_inputs = self._object_distractor_class_inputs(
            self._object_inputs_from_target(target, imgs.device)
        )
        with torch.no_grad():
            dropped_preds = self._cached_object_prompt_logits(empty_inputs).detach()
            distractor_preds = self._cached_object_prompt_logits(
                distractor_inputs
            ).detach()

        with torch.no_grad():
            base_logp = F.log_softmax(preds.float(), dim=-1)
            dropped_logp = F.log_softmax(dropped_preds.float(), dim=-1)
            base_prob = base_logp.exp()
            dropped_prob = dropped_logp.exp()
            base_pred = preds.argmax(dim=-1)
            dropped_pred = dropped_preds.argmax(dim=-1)
            distractor_logp = F.log_softmax(distractor_preds.float(), dim=-1)
            distractor_pred = distractor_preds.argmax(dim=-1)

            if has_objectless:
                labels = actions[objectless_visible]
                base_true_prob = base_prob[objectless_visible].gather(
                    1,
                    labels.unsqueeze(1),
                ).squeeze(1)
                dropped_true_prob = dropped_prob[objectless_visible].gather(
                    1,
                    labels.unsqueeze(1),
                ).squeeze(1)
                consistency_kl = F.kl_div(
                    dropped_logp[objectless_visible],
                    base_prob[objectless_visible],
                    reduction="batchmean",
                )
                distractor_kl = F.kl_div(
                    distractor_logp[objectless_visible],
                    base_prob[objectless_visible],
                    reduction="batchmean",
                )
                count = int(labels.numel())
                self._log_scalar(
                    f"{stage}_object_prompt_drop_objectless_pred_match",
                    (
                        base_pred[objectless_visible]
                        == dropped_pred[objectless_visible]
                    ).float().mean(),
                    count,
                )
                self._log_scalar(
                    f"{stage}_object_prompt_drop_objectless_true_prob_delta",
                    (base_true_prob - dropped_true_prob).mean(),
                    count,
                )
                self._log_scalar(
                    f"{stage}_object_prompt_drop_objectless_kl",
                    consistency_kl,
                    count,
                )
                self._log_scalar(
                    f"{stage}_object_prompt_drop_objectless_acc",
                    (dropped_pred[objectless_visible] == labels).float().mean(),
                    count,
                )
                self._log_scalar(
                    f"{stage}_object_prompt_distractor_objectless_pred_match",
                    (
                        base_pred[objectless_visible]
                        == distractor_pred[objectless_visible]
                    ).float().mean(),
                    count,
                )
                self._log_scalar(
                    f"{stage}_object_prompt_distractor_objectless_kl",
                    distractor_kl,
                    count,
                )
                self._log_scalar(
                    f"{stage}_object_prompt_distractor_objectless_acc",
                    (distractor_pred[objectless_visible] == labels).float().mean(),
                    count,
                )
                object_action_indices = self.group_indices.get("object_mapped")
                if (
                    object_action_indices is not None
                    and int(object_action_indices.numel()) > 0
                ):
                    object_action_indices = object_action_indices.to(
                        device=device,
                        dtype=torch.long,
                    )
                    object_action_rate = (
                        self._labels_in_indices(
                            dropped_pred[objectless_visible],
                            object_action_indices,
                        )
                        .float()
                        .mean()
                    )
                    self._log_scalar(
                        f"{stage}_object_prompt_drop_objectless_object_action_pred_rate",
                        object_action_rate,
                        count,
                    )
                    distractor_object_action_rate = (
                        self._labels_in_indices(
                            distractor_pred[objectless_visible],
                            object_action_indices,
                        )
                        .float()
                        .mean()
                    )
                    self._log_scalar(
                        f"{stage}_object_prompt_distractor_objectless_object_action_pred_rate",
                        distractor_object_action_rate,
                        count,
                    )

            if has_exact:
                labels = actions[exact_compatible]
                base_true_logits = preds[exact_compatible].gather(
                    1,
                    labels.unsqueeze(1),
                ).squeeze(1)
                dropped_true_logits = dropped_preds[exact_compatible].gather(
                    1,
                    labels.unsqueeze(1),
                ).squeeze(1)
                base_true_prob = base_prob[exact_compatible].gather(
                    1,
                    labels.unsqueeze(1),
                ).squeeze(1)
                dropped_true_prob = dropped_prob[exact_compatible].gather(
                    1,
                    labels.unsqueeze(1),
                ).squeeze(1)
                count = int(labels.numel())
                self._log_scalar(
                    f"{stage}_object_prompt_drop_exact_true_logit_drop",
                    (base_true_logits - dropped_true_logits).mean(),
                    count,
                )
                self._log_scalar(
                    f"{stage}_object_prompt_drop_exact_true_prob_drop",
                    (base_true_prob - dropped_true_prob).mean(),
                    count,
                )
                self._log_scalar(
                    f"{stage}_object_prompt_drop_exact_pred_match",
                    (
                        base_pred[exact_compatible]
                        == dropped_pred[exact_compatible]
                    ).float().mean(),
                    count,
                )
                self._log_scalar(
                    f"{stage}_object_prompt_drop_exact_acc",
                    (dropped_pred[exact_compatible] == labels).float().mean(),
                    count,
                )
        return None

    def _motion_confuser_margin(self, motion_logits, actions, selected_valid):
        margins = torch.full(
            actions.shape,
            float("inf"),
            device=actions.device,
            dtype=motion_logits.dtype,
        )
        rows = torch.nonzero(selected_valid, as_tuple=False)
        for row in rows:
            batch_idx = int(row[0].item())
            actor_idx = int(row[1].item())
            action_idx = int(actions[batch_idx, actor_idx].item())
            confusers = self.action_confuser_indices_by_index.get(action_idx)
            if confusers is None or int(confusers.numel()) == 0:
                continue
            confusers = confusers.to(device=actions.device, dtype=torch.long)
            true_logit = motion_logits[batch_idx, actor_idx, action_idx]
            max_confuser = motion_logits[batch_idx, actor_idx, confusers].max()
            margins[batch_idx, actor_idx] = true_logit - max_confuser
        return margins

    def _object_prompt_action_coupling_loss(
        self,
        stage,
        imgs,
        boxes,
        valid,
        actions,
        preds,
        target,
        object_inputs,
    ):
        if not self.uses_object_proposals:
            return None
        if (
            self.object_prompt_wrong_class_loss_weight <= 0
            and self.object_prompt_sensitivity_loss_weight <= 0
        ):
            return None
        info = self._exact_teacher_object_info(actions, valid, target, preds.device)
        if info is None:
            return None
        exact_compatible = (
            info["valid"]
            & info["known_action"]
            & info["compatible_1based"]
        )
        selected_valid = self._first_actor_per_sample_mask(exact_compatible)
        if not selected_valid.any():
            return None

        actions = actions.to(device=preds.device, dtype=torch.long)
        selected_labels = actions[selected_valid]
        selected_slots = info["idx_1based"].to(device=preds.device, dtype=torch.long)
        base_true_logits = preds[selected_valid].gather(
            1,
            selected_labels.unsqueeze(1),
        ).squeeze(1)

        weighted_losses = []
        wrong_preds = None
        dropped_preds = None
        dropped_prompt_mask = None
        if self.object_prompt_wrong_class_loss_weight > 0:
            wrong_inputs = self._teacher_object_wrong_class_inputs(
                object_inputs,
                selected_slots,
                selected_valid,
                actions,
            )
            wrong_preds = self._cached_object_prompt_logits(wrong_inputs)
            wrong_true_logits = wrong_preds[selected_valid].gather(
                1,
                selected_labels.unsqueeze(1),
            ).squeeze(1)
            wrong_drop = base_true_logits - wrong_true_logits
            loss_wrong = F.relu(
                self.object_prompt_wrong_class_margin - wrong_drop
            ).mean()
            count = int(selected_labels.numel())
            self.log(
                f"{stage}_loss_object_prompt_wrong_class",
                loss_wrong,
                on_step=stage == "train",
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )
            self._log_scalar(
                f"{stage}_object_prompt_correct_minus_wrong_true_logit",
                wrong_drop.detach().mean(),
                count,
            )
            self._log_scalar(
                f"{stage}_object_prompt_wrong_class_margin_sat_rate",
                (
                    wrong_drop.detach()
                    >= self.object_prompt_wrong_class_margin
                )
                .float()
                .mean(),
                count,
            )
            weighted_losses.append(
                loss_wrong * self.object_prompt_wrong_class_loss_weight
            )

        if self.object_prompt_sensitivity_loss_weight > 0:
            motion_logits = getattr(self.model, "last_actor_motion_logits", None)
            if motion_logits is not None:
                motion_logits = motion_logits.to(device=preds.device)
                motion_margin = self._motion_confuser_margin(
                    motion_logits.float(),
                    actions,
                    selected_valid,
                )
                ambiguous = (
                    selected_valid
                    & (
                        motion_margin
                        < self.object_prompt_sensitivity_motion_margin_threshold
                    )
                )
            else:
                ambiguous = torch.zeros_like(selected_valid, dtype=torch.bool)

            self._log_scalar(
                f"{stage}_object_prompt_sensitivity_ambiguous_rate",
                ambiguous[selected_valid].float().mean(),
                int(selected_valid.sum().item()),
            )
            if ambiguous.any():
                dropped_prompt_mask = ambiguous
                dropped_inputs = self._teacher_object_removed_inputs(
                    object_inputs,
                    info["selected_indices"].to(
                        device=preds.device,
                        dtype=torch.long,
                    ),
                    ambiguous,
                )
                dropped_preds = self._cached_object_prompt_logits(dropped_inputs)
                ambiguous_labels = actions[ambiguous]
                base_ambiguous_true = preds[ambiguous].gather(
                    1,
                    ambiguous_labels.unsqueeze(1),
                ).squeeze(1)
                dropped_true = dropped_preds[ambiguous].gather(
                    1,
                    ambiguous_labels.unsqueeze(1),
                ).squeeze(1)
                dropped_drop = base_ambiguous_true - dropped_true
                loss_sensitivity = F.relu(
                    self.object_prompt_sensitivity_margin - dropped_drop
                ).mean()
                count = int(ambiguous_labels.numel())
                self.log(
                    f"{stage}_loss_object_prompt_sensitivity",
                    loss_sensitivity,
                    on_step=stage == "train",
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                    sync_dist=True,
                )
                self._log_scalar(
                    f"{stage}_object_prompt_correct_minus_dropped_true_logit_ambiguous",
                    dropped_drop.detach().mean(),
                    count,
                )
                self._log_scalar(
                    f"{stage}_object_prompt_sensitivity_margin_sat_rate",
                    (
                        dropped_drop.detach()
                        >= self.object_prompt_sensitivity_margin
                    )
                    .float()
                    .mean(),
                    count,
                )
                weighted_losses.append(
                    loss_sensitivity
                    * self.object_prompt_sensitivity_loss_weight
                )

        self._log_uselaptop_prompt_counterfactual_margins(
            stage,
            actions,
            selected_valid,
            wrong_preds,
            dropped_preds,
            dropped_prompt_mask,
        )

        if not weighted_losses:
            return None
        return torch.stack(weighted_losses).sum()

    def _log_uselaptop_prompt_counterfactual_margins(
        self,
        stage,
        actions,
        selected_valid,
        wrong_preds,
        dropped_preds,
        dropped_prompt_mask,
    ):
        uselaptop_idx = self.action_to_index.get("Uselaptop")
        readbook_idx = self.action_to_index.get("Readbook")
        watchtv_idx = self.action_to_index.get("WatchTV")
        if uselaptop_idx is None:
            return
        uselaptop_mask = selected_valid & (actions == int(uselaptop_idx))
        if not uselaptop_mask.any():
            return
        count = int(uselaptop_mask.sum().item())
        if wrong_preds is not None and readbook_idx is not None:
            margin = (
                wrong_preds[uselaptop_mask][:, int(uselaptop_idx)]
                - wrong_preds[uselaptop_mask][:, int(readbook_idx)]
            )
            self._log_scalar(
                f"{stage}_object_prompt_wrong_Uselaptop_minus_Readbook_margin",
                margin.detach().mean(),
                count,
            )
        if (
            dropped_preds is not None
            and dropped_prompt_mask is not None
            and watchtv_idx is not None
        ):
            uselaptop_mask = dropped_prompt_mask & (actions == int(uselaptop_idx))
            if not uselaptop_mask.any():
                return
            count = int(uselaptop_mask.sum().item())
            margin = (
                dropped_preds[uselaptop_mask][:, int(uselaptop_idx)]
                - dropped_preds[uselaptop_mask][:, int(watchtv_idx)]
            )
            self._log_scalar(
                f"{stage}_object_prompt_dropped_Uselaptop_minus_WatchTV_margin",
                margin.detach().mean(),
                count,
            )

    def _log_actor_object_prompt_diagnostics(
        self,
        stage,
        actions,
        valid,
        target,
    ):
        if stage == "train" or not self.actor_object_prompt_tokens:
            return
        prompt_attention = getattr(
            self.model,
            "last_actor_object_prompt_attention",
            None,
        )
        prompt_valid = getattr(
            self.model,
            "last_actor_object_prompt_valid",
            None,
        )
        if prompt_attention is None or prompt_valid is None:
            return

        device = prompt_attention.device
        info = self._exact_teacher_object_info(actions, valid, target, device)
        if info is None:
            return

        valid = valid.to(device=device, dtype=torch.bool)
        actions = actions.to(device=device, dtype=torch.long)
        prompt_valid = prompt_valid.to(device=device, dtype=torch.bool)
        prompt_attention = prompt_attention.float()
        visible_prompt = prompt_valid.any(dim=1)
        valid_prompt_attention = prompt_attention.masked_fill(
            ~prompt_valid[:, None, :],
            0.0,
        )
        objectless = self._labels_in_indices(actions, self.objectless_action_indices)
        objectless_visible = valid & objectless & visible_prompt[:, None]
        if objectless_visible.any():
            prompt_count = prompt_valid.sum(dim=1).clamp_min(1).to(
                dtype=prompt_attention.dtype,
            )
            mean_attention = (
                valid_prompt_attention.sum(dim=-1)
                / prompt_count[:, None]
            )
            max_attention = valid_prompt_attention.amax(dim=-1)
            entropy_prob = valid_prompt_attention.clamp_min(1.0e-8)
            entropy = -(entropy_prob * entropy_prob.log()).sum(dim=-1)
            entropy_norm = torch.where(
                prompt_count[:, None] > 1.0,
                entropy / prompt_count[:, None].log().clamp_min(1.0e-6),
                torch.zeros_like(entropy),
            )
            count = int(objectless_visible.sum().item())
            self._log_scalar(
                f"{stage}_object_prompt_attention_objectless_visible_mean",
                mean_attention[objectless_visible].mean(),
                count,
            )
            self._log_scalar(
                f"{stage}_object_prompt_attention_objectless_visible_max",
                max_attention[objectless_visible].mean(),
                count,
            )
            self._log_scalar(
                f"{stage}_object_prompt_attention_objectless_visible_entropy",
                entropy_norm[objectless_visible].mean(),
                count,
            )

        valid_known = info["valid"] & info["known_action"]
        if valid_known.any():
            count = int(valid_known.sum().item())
            self._log_scalar(
                f"{stage}_object_prompt_exact_teacher_valid_rate_1based",
                info["valid_1based"][valid_known].float().mean(),
                count,
            )
            self._log_scalar(
                f"{stage}_object_prompt_exact_compatible_rate_1based",
                info["compatible_1based"][valid_known].float().mean(),
                count,
            )
            self._log_scalar(
                f"{stage}_object_prompt_any_compatible_proposal_rate",
                info["any_compatible"][valid_known].float().mean(),
                count,
            )

        exact_compatible = (
            info["valid"]
            & info["known_action"]
            & info["compatible_1based"]
        )
        if not exact_compatible.any():
            return

        target_slots = info["idx_1based"][exact_compatible].to(
            device=device,
            dtype=torch.long,
        )
        attention = prompt_attention[exact_compatible].float()
        if (target_slots >= attention.shape[-1]).any():
            raise RuntimeError(
                "object prompt teacher slot is outside prompt-token count"
            )
        pred_slots = attention.argmax(dim=-1)
        true_prob = attention.gather(1, target_slots.unsqueeze(1)).squeeze(1)
        count = int(target_slots.numel())
        self._log_count(
            f"{stage}_object_prompt_exact_compatible_count",
            exact_compatible.float().sum(),
        )
        self._log_scalar(
            f"{stage}_object_prompt_exact_correct_object_rate",
            (pred_slots == target_slots).float().mean(),
            count,
        )
        self._log_scalar(
            f"{stage}_object_prompt_exact_correct_object_prob",
            true_prob.mean(),
            count,
        )
        self._log_scalar(
            f"{stage}_object_prompt_attention_exact_teacher_mean",
            true_prob.mean(),
            count,
        )
        self._log_scalar(
            f"{stage}_actor_object_prompt_token_count",
            torch.as_tensor(
                float(prompt_valid.shape[-1]),
                device=device,
                dtype=torch.float32,
            ),
            count,
        )

    def _true_minus_confuser_margin(self, values, actions, mask):
        if values is None or not mask.any():
            return None
        actions = actions.to(device=values.device, dtype=torch.long)
        mask = mask.to(device=values.device, dtype=torch.bool)
        margins = []
        for batch_idx, actor_idx in torch.nonzero(mask, as_tuple=False).tolist():
            action_idx = int(actions[batch_idx, actor_idx].item())
            confusers = self.action_confuser_indices_by_index.get(action_idx)
            if confusers is None or int(confusers.numel()) == 0:
                continue
            confusers = confusers.to(device=values.device, dtype=torch.long)
            confusers = confusers[
                (confusers >= 0) & (confusers < int(values.shape[-1]))
            ]
            if int(confusers.numel()) == 0:
                continue
            true_value = values[batch_idx, actor_idx, action_idx]
            max_confuser = values[batch_idx, actor_idx, confusers].max()
            margins.append(true_value - max_confuser)
        if not margins:
            return None
        return torch.stack(margins)

    def _gather_true_action_value(self, values, actions):
        actions = actions.to(device=values.device, dtype=torch.long)
        safe_actions = actions.clamp(0, int(values.shape[-1]) - 1)
        return values.gather(-1, safe_actions.unsqueeze(-1)).squeeze(-1)

    def _log_object_residual_diagnostics(
        self,
        stage,
        actions,
        valid,
        target,
    ):
        if stage == "train" or not self.actor_object_residual_head:
            return
        coverage = getattr(self.model, "last_object_residual_coverage", None)
        object_residual = getattr(self.model, "last_object_residual", None)
        raw_base_logits = getattr(
            self.model,
            "last_object_residual_raw_base_logits",
            None,
        )
        base_logits = getattr(self.model, "last_object_residual_base_logits", None)
        final_logits = getattr(self.model, "last_object_residual_final_logits", None)
        null_prob = getattr(self.model, "last_object_residual_prompt_null_prob", None)
        useful_mass = getattr(
            self.model,
            "last_object_residual_prompt_useful_mass",
            None,
        )
        fusion_delta = getattr(
            self.model,
            "last_actor_object_base_fusion_delta",
            None,
        )
        fusion_gate = getattr(
            self.model,
            "last_actor_object_base_fusion_gate",
            None,
        )
        fusion_null_prob = getattr(
            self.model,
            "last_actor_object_base_fusion_null_prob",
            None,
        )
        fusion_useful_mass = getattr(
            self.model,
            "last_actor_object_base_fusion_useful_mass",
            None,
        )
        if (
            coverage is None
            or object_residual is None
            or base_logits is None
            or final_logits is None
        ):
            return

        device = coverage.device
        info = self._exact_teacher_object_info(actions, valid, target, device)
        if info is None:
            return
        actions = actions.to(device=device, dtype=torch.long)
        valid = valid.to(device=device, dtype=torch.bool)
        known_objectful = info["valid"] & info["known_action"]
        exact_compatible = known_objectful & info["compatible_1based"]
        missing_compatible = known_objectful & ~info["any_compatible"]

        relation_scale = getattr(self.model, "last_object_residual_relation_scale", None)
        if relation_scale is not None and known_objectful.any():
            self._log_scalar(
                f"{stage}_object_residual_relation_scale",
                relation_scale.to(device=device).float(),
                int(known_objectful.sum().item()),
            )

        coverage = coverage.float()
        object_residual = object_residual.float()
        if raw_base_logits is not None:
            raw_base_logits = raw_base_logits.float()
        base_logits = base_logits.float()
        final_logits = final_logits.float()
        valid_count = int(valid.sum().item())
        if valid_count > 0:
            residual_abs = object_residual[valid].abs()
            self._log_scalar(
                f"{stage}_object_residual_abs_mean",
                residual_abs.mean(),
                valid_count,
            )
            self._log_scalar(
                f"{stage}_object_residual_abs_max",
                residual_abs.max(),
                valid_count,
            )
            if fusion_delta is not None:
                fusion_delta_norm = fusion_delta.float().norm(dim=-1)
                self._log_scalar(
                    f"{stage}_object_fusion_delta_norm",
                    fusion_delta_norm[valid].mean(),
                    valid_count,
                )
            if fusion_gate is not None:
                fusion_gate_mean = fusion_gate.float().mean(dim=-1)
                self._log_scalar(
                    f"{stage}_object_fusion_gate_mean",
                    fusion_gate_mean[valid].mean(),
                    valid_count,
                )

        fusion_scale = getattr(self.model, "last_actor_object_base_fusion_scale", None)
        if fusion_scale is not None and known_objectful.any():
            self._log_scalar(
                f"{stage}_object_fusion_scale",
                fusion_scale.to(device=device).float(),
                int(known_objectful.sum().item()),
            )

        object_action_indices = self.group_indices.get("object_mapped")
        objectless = self._labels_in_indices(actions, self.objectless_action_indices)
        objectless = objectless & valid
        if objectless.any():
            count = int(objectless.sum().item())
            if fusion_useful_mass is not None:
                self._log_scalar(
                    f"{stage}_object_fusion_useful_mass_objectless",
                    fusion_useful_mass.float()[objectless].mean(),
                    count,
                )
            if fusion_null_prob is not None:
                self._log_scalar(
                    f"{stage}_object_fusion_null_prob_objectless",
                    fusion_null_prob.float()[objectless].mean(),
                    count,
                )
        if (
            objectless.any()
            and object_action_indices is not None
            and int(object_action_indices.numel()) > 0
        ):
            object_action_indices = object_action_indices.to(
                device=device,
                dtype=torch.long,
            )
            true_base = self._gather_true_action_value(base_logits, actions)
            true_final = self._gather_true_action_value(final_logits, actions)
            max_object_base = base_logits[..., object_action_indices].max(dim=-1).values
            max_object_final = final_logits[..., object_action_indices].max(
                dim=-1
            ).values
            count = int(objectless.sum().item())
            self._log_scalar(
                f"{stage}_object_residual_base_objectless_true_minus_objectful_max",
                (true_base - max_object_base)[objectless].mean(),
                count,
            )
            self._log_scalar(
                f"{stage}_object_residual_final_objectless_true_minus_objectful_max",
                (true_final - max_object_final)[objectless].mean(),
                count,
            )

        raw_true = None
        fused_true = self._gather_true_action_value(base_logits, actions)
        if raw_base_logits is not None:
            raw_true = self._gather_true_action_value(raw_base_logits, actions)
        coverage_true = self._gather_true_action_value(coverage, actions)
        residual_true = self._gather_true_action_value(object_residual, actions)
        null_true = None
        useful_true = None
        if null_prob is not None:
            null_true = self._gather_true_action_value(null_prob.float(), actions)
        if useful_mass is not None:
            useful_true = self._gather_true_action_value(useful_mass.float(), actions)

        if exact_compatible.any():
            count = int(exact_compatible.sum().item())
            self._log_scalar(
                f"{stage}_object_residual_coverage_true_exact",
                coverage_true[exact_compatible].mean(),
                count,
            )
            self._log_scalar(
                f"{stage}_object_residual_true_exact",
                residual_true[exact_compatible].mean(),
                count,
            )
            if useful_true is not None:
                self._log_scalar(
                    f"{stage}_object_residual_useful_mass_true_exact",
                    useful_true[exact_compatible].mean(),
                    count,
                )
            if null_true is not None:
                self._log_scalar(
                    f"{stage}_object_residual_null_prob_true_exact",
                    null_true[exact_compatible].mean(),
                    count,
                )
            if fusion_useful_mass is not None:
                self._log_scalar(
                    f"{stage}_object_fusion_useful_mass_exact",
                    fusion_useful_mass.float()[exact_compatible].mean(),
                    count,
                )
            if fusion_null_prob is not None:
                self._log_scalar(
                    f"{stage}_object_fusion_null_prob_exact",
                    fusion_null_prob.float()[exact_compatible].mean(),
                    count,
                )
            if raw_true is not None:
                self._log_scalar(
                    f"{stage}_object_fusion_raw_to_fused_true_logit_delta_exact",
                    (fused_true - raw_true)[exact_compatible].mean(),
                    count,
                )
                raw_base_margin = self._true_minus_confuser_margin(
                    raw_base_logits,
                    actions,
                    exact_compatible,
                )
                if raw_base_margin is not None:
                    self._log_scalar(
                        f"{stage}_object_fusion_raw_base_true_minus_confuser_exact",
                        raw_base_margin.mean(),
                        int(raw_base_margin.numel()),
                    )
            residual_margin = self._true_minus_confuser_margin(
                object_residual,
                actions,
                exact_compatible,
            )
            if residual_margin is not None:
                self._log_scalar(
                    f"{stage}_object_residual_prompt_delta_true_minus_confuser_exact",
                    residual_margin.mean(),
                    int(residual_margin.numel()),
                )
            base_margin = self._true_minus_confuser_margin(
                base_logits,
                actions,
                exact_compatible,
            )
            if base_margin is not None:
                self._log_scalar(
                    f"{stage}_object_residual_base_true_minus_confuser_exact",
                    base_margin.mean(),
                    int(base_margin.numel()),
                )
            final_margin = self._true_minus_confuser_margin(
                final_logits,
                actions,
                exact_compatible,
            )
            if final_margin is not None:
                self._log_scalar(
                    f"{stage}_object_residual_final_true_minus_confuser_exact",
                    final_margin.mean(),
                    int(final_margin.numel()),
                )

        if missing_compatible.any():
            count = int(missing_compatible.sum().item())
            self._log_scalar(
                f"{stage}_object_residual_true_missing",
                residual_true[missing_compatible].mean(),
                count,
            )
            if useful_true is not None:
                self._log_scalar(
                    f"{stage}_object_residual_useful_mass_true_missing",
                    useful_true[missing_compatible].mean(),
                    count,
                )
            if null_true is not None:
                self._log_scalar(
                    f"{stage}_object_residual_null_prob_true_missing",
                    null_true[missing_compatible].mean(),
                    count,
                )
            residual_margin = self._true_minus_confuser_margin(
                object_residual,
                actions,
                missing_compatible,
            )
            if residual_margin is not None:
                self._log_scalar(
                    f"{stage}_object_residual_prompt_delta_true_minus_confuser_missing",
                    residual_margin.mean(),
                    int(residual_margin.numel()),
                )
            base_margin = self._true_minus_confuser_margin(
                base_logits,
                actions,
                missing_compatible,
            )
            if base_margin is not None:
                self._log_scalar(
                    f"{stage}_object_residual_base_true_minus_confuser_missing",
                    base_margin.mean(),
                    int(base_margin.numel()),
                )
            final_margin = self._true_minus_confuser_margin(
                final_logits,
                actions,
                missing_compatible,
            )
            if final_margin is not None:
                self._log_scalar(
                    f"{stage}_object_residual_final_true_minus_confuser_missing",
                    final_margin.mean(),
                    int(final_margin.numel()),
                )

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

    def _linear_epoch_weight(self, initial_weight, final_weight, start_epoch, end_epoch):
        epoch = int(getattr(self, "current_epoch", 0) or 0)
        initial_weight = float(initial_weight)
        final_weight = float(final_weight)
        start_epoch = int(start_epoch)
        end_epoch = int(end_epoch)
        if end_epoch <= start_epoch:
            return initial_weight
        if epoch <= start_epoch:
            return initial_weight
        if epoch >= end_epoch:
            return final_weight
        ratio = float(epoch - start_epoch) / float(end_epoch - start_epoch)
        return initial_weight + (final_weight - initial_weight) * ratio

    def _object_residual_prompt_relation_effective_weight(self):
        return self._linear_epoch_weight(
            self.object_residual_prompt_relation_loss_weight,
            self.object_residual_prompt_relation_loss_final_weight,
            self.object_residual_relation_loss_decay_start_epoch,
            self.object_residual_relation_loss_decay_end_epoch,
        )

    def _object_residual_evidence_relation_loss(self, stage, actions, valid, target):
        if not self.actor_object_residual_head:
            return None
        prompt_weight = self._object_residual_prompt_relation_effective_weight()
        null_weight = self.object_residual_null_relation_loss_weight
        if prompt_weight <= 0 and null_weight <= 0:
            return None

        prompt_delta = getattr(self.model, "last_object_residual_prompt_delta", None)
        useful_mass = getattr(self.model, "last_object_residual_prompt_useful_mass", None)
        if prompt_delta is None and useful_mass is None:
            return None

        device = prompt_delta.device if prompt_delta is not None else useful_mass.device
        info = self._exact_teacher_object_info(actions, valid, target, device)
        if info is None:
            return None

        actions = actions.to(device=device, dtype=torch.long)
        valid = valid.to(device=device, dtype=torch.bool)
        known_objectful = info["valid"] & info["known_action"]
        exact_compatible = known_objectful & info["compatible_1based"]
        missing_compatible = known_objectful & ~info["any_compatible"]
        if known_objectful.any():
            count = int(known_objectful.sum().item())
            self._log_scalar(
                f"{stage}_object_residual_prompt_relation_effective_weight",
                torch.as_tensor(prompt_weight, device=device, dtype=torch.float32),
                count,
            )
            self._log_scalar(
                f"{stage}_object_residual_null_relation_weight",
                torch.as_tensor(null_weight, device=device, dtype=torch.float32),
                count,
            )

        terms = []
        margin_target = self.object_residual_relation_confuser_margin
        if (
            prompt_weight > 0
            and prompt_delta is not None
            and exact_compatible.any()
        ):
            prompt_margins = self._true_minus_confuser_margin(
                prompt_delta.float(),
                actions,
                exact_compatible,
            )
            if prompt_margins is not None:
                loss_prompt = F.relu(margin_target - prompt_margins).mean()
                count = int(prompt_margins.numel())
                self.log(
                    f"{stage}_loss_object_residual_prompt_relation",
                    loss_prompt,
                    on_step=stage == "train",
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                    sync_dist=True,
                )
                self._log_scalar(
                    f"{stage}_object_residual_prompt_relation_margin_exact",
                    prompt_margins.detach().mean(),
                    count,
                )
                self._log_scalar(
                    f"{stage}_object_residual_prompt_relation_margin_sat_exact",
                    (prompt_margins.detach() >= margin_target).float().mean(),
                    count,
                )
                terms.append(loss_prompt * prompt_weight)

        if (
            null_weight > 0
            and useful_mass is not None
            and (exact_compatible.any() or missing_compatible.any())
        ):
            useful_mass = useful_mass.float()
            useful_true = self._gather_true_action_value(useful_mass, actions)
            values = []
            targets = []
            if exact_compatible.any():
                exact_values = useful_true[exact_compatible]
                values.append(exact_values)
                targets.append(torch.ones_like(exact_values))
                self._log_scalar(
                    f"{stage}_object_residual_useful_mass_loss_exact_mean",
                    exact_values.detach().mean(),
                    int(exact_values.numel()),
                )
            if missing_compatible.any():
                missing_values = useful_true[missing_compatible]
                values.append(missing_values)
                targets.append(torch.zeros_like(missing_values))
                self._log_scalar(
                    f"{stage}_object_residual_useful_mass_loss_missing_mean",
                    missing_values.detach().mean(),
                    int(missing_values.numel()),
                )
            useful_values = torch.cat(values).float().clamp(1.0e-4, 1.0 - 1.0e-4)
            useful_targets = torch.cat(targets).to(dtype=useful_values.dtype)
            useful_logits = torch.logit(useful_values)
            loss_null = F.binary_cross_entropy_with_logits(
                useful_logits,
                useful_targets,
            )
            self.log(
                f"{stage}_loss_object_residual_null_relation",
                loss_null,
                on_step=stage == "train",
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )
            terms.append(loss_null * null_weight)

        if not terms:
            return None
        return torch.stack(terms).sum()

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
        model_object_inputs = self._object_class_dropout_inputs(object_inputs, stage)
        data = self.model(
            imgs,
            boxes=boxes,
            valid=valid,
            action_labels=actions,
            **model_object_inputs,
        )
        preds, hm_preds, presence_logits = self._unpack_model_data(data)

        loss_action = self._action_loss(preds, actions, loss_fn, valid)
        valid_preds = preds[valid]
        valid_labels = actions[valid]
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
        loss_object_residual_relation = self._object_residual_evidence_relation_loss(
            stage,
            actions,
            valid,
            target,
        )
        if loss_object_residual_relation is not None:
            loss_main_task = loss_main_task + loss_object_residual_relation

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

        grounding_aux_terms = []
        loss_object_prompt_grounding = self._object_prompt_grounding_loss(
            stage,
            actions,
            valid,
            target,
        )
        if (
            loss_object_prompt_grounding is not None
            and self.object_prompt_grounding_loss_weight > 0
        ):
            grounding_aux_terms.append(
                loss_object_prompt_grounding
                * self.object_prompt_grounding_loss_weight
            )

        loss_object_prompt_action_coupling = (
            self._object_prompt_action_coupling_loss(
                stage,
                imgs,
                boxes,
                valid,
                actions,
                preds,
                target,
                model_object_inputs,
            )
        )
        if stage == "train" and loss_object_prompt_action_coupling is not None:
            loss_main_task = loss_main_task + loss_object_prompt_action_coupling

        self._log_actor_object_prompt_diagnostics(
            stage,
            actions,
            valid,
            target,
        )
        self._log_object_residual_diagnostics(
            stage,
            actions,
            valid,
            target,
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
        self._log_object_prompt_drop_eval(
            imgs,
            boxes,
            valid,
            actions,
            preds,
            target,
            stage,
        )
        loss_objectless_prompt_consistency = (
            self._objectless_prompt_consistency_loss(
                stage,
                imgs,
                boxes,
                valid,
                actions,
                preds,
                target,
                model_object_inputs,
            )
        )
        if (
            loss_objectless_prompt_consistency is not None
            and self.objectless_prompt_consistency_loss_weight > 0
        ):
            loss_main_task = (
                loss_main_task
                + loss_objectless_prompt_consistency
                * self.objectless_prompt_consistency_loss_weight
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
                if self.poguiseplus_interaction_heatmap_pos_loss_weight > 0:
                    loss_interaction_pos_balanced = (
                        self._positive_balanced_interaction_heatmap_mse(
                            interaction_heatmap,
                            target_heatmap,
                            positive_heatmap_valid,
                        )
                    )
                    loss_interaction_heatmap_optimized = (
                        loss_interaction_heatmap_optimized
                        + loss_interaction_pos_balanced
                        * self.poguiseplus_interaction_heatmap_pos_loss_weight
                    )
                    self.log(
                        f"{stage}_loss_interaction_heatmap_pos_balanced",
                        loss_interaction_pos_balanced,
                        on_step=stage == "train",
                        on_epoch=True,
                        prog_bar=False,
                        logger=True,
                        sync_dist=True,
                    )
                if self.poguiseplus_interaction_heatmap_center_loss_weight > 0:
                    loss_interaction_center = self._interaction_heatmap_center_loss(
                        interaction_heatmap,
                        target_heatmap,
                        positive_heatmap_valid,
                    )
                    loss_interaction_heatmap_optimized = (
                        loss_interaction_heatmap_optimized
                        + loss_interaction_center
                        * self.poguiseplus_interaction_heatmap_center_loss_weight
                    )
                    self.log(
                        f"{stage}_loss_interaction_heatmap_center",
                        loss_interaction_center,
                        on_step=stage == "train",
                        on_epoch=True,
                        prog_bar=False,
                        logger=True,
                        sync_dist=True,
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

    def _action_loss(self, scores, labels, loss_fn, valid=None):
        if self.actor_object_residual_head:
            labels = labels.to(device=scores.device, dtype=torch.long)
            if valid is not None:
                valid = valid.to(device=scores.device, dtype=torch.bool)
                if not valid.any():
                    return scores.sum() * 0.0
                return F.nll_loss(scores[valid].float(), labels[valid])
            return F.nll_loss(scores.float(), labels)
        if valid is not None:
            valid = valid.to(device=scores.device, dtype=torch.bool)
            if not valid.any():
                return scores.sum() * 0.0
            labels = labels.to(device=scores.device, dtype=torch.long)
            return loss_fn(scores[valid].float(), labels[valid])
        return loss_fn(scores, labels)

    def _action_probs(self, scores):
        return scores.float().softmax(dim=-1)

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
        probs = self._action_probs(full_preds)[hard_mask]
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
        watch_idx = self._action_index("WatchTV")
        if watch_idx is not None:
            watch_fp = (pred_labels[hard_mask] == int(watch_idx)).float().mean()
            self._log_scalar(
                f"{stage}_watchtv_fp_rate_objectless",
                watch_fp,
                count,
            )
            sit_idx = self._action_index("Sitdown")
            if sit_idx is not None:
                sit_mask = hard_mask & (actions == int(sit_idx))
                if sit_mask.any():
                    self._log_scalar(
                        f"{stage}_sitdown_to_watchtv_false_positive_rate",
                        (pred_labels[sit_mask] == int(watch_idx)).float().mean(),
                        int(sit_mask.sum().item()),
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
        loss = self._action_loss(preds, labels, self.val_loss)
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
