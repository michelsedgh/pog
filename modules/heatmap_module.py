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

DEPLOY_KEY_ACTIONS = (
    "Uselaptop",
    "Readbook",
    "Walk",
    "Getup",
    "Sitdown",
    "Laydown",
)
RELATION_ACTION_AUDIT_ACTIONS = (
    "Uselaptop",
    "Readbook",
    "WatchTV",
    "Usetelephone",
    "Usetablet",
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
                "for relation-only runtime object memory."
            )
        self.actor_object_residual_head = bool(
            hparams.get("actor_object_residual_head", 0)
        )
        self.actor_object_prompt_tokens = bool(
            hparams.get("actor_object_prompt_tokens", 0)
        )
        self.actor_object_region_visual_tokens = bool(
            hparams.get("actor_object_region_visual_tokens", 0)
        )
        self.actor_object_relation_in_transformer = bool(
            hparams.get("actor_object_relation_in_transformer", 0)
        )
        self.actor_object_pair_action_head = bool(
            hparams.get("actor_object_pair_action_head", 0)
        )
        actor_object_slot_head = bool(hparams.get("actor_object_slot_head", 0))
        if actor_object_slot_head:
            raise ValueError(
                "actor_object_slot_head was replaced by "
                "actor_object_prompt_tokens. Set --actor_object_prompt_tokens 1 "
                "and keep --actor_object_slot_head 0."
            )
        if self.actor_object_residual_head:
            raise ValueError(
                "actor_object_residual_head was removed. Use "
                "actor_object_relation_in_transformer for object-aware actor tokens."
            )
        self.uses_object_proposals = self.actor_object_prompt_tokens
        if self.actor_object_prompt_tokens and not self.actor_prompt:
            raise ValueError("actor_object_prompt_tokens requires actor_prompt")
        if self.actor_object_prompt_tokens and not self.actor_object_region_visual_tokens:
            raise ValueError(
                "actor_object_prompt_tokens requires "
                "actor_object_region_visual_tokens=1. Runtime object memory must "
                "include visual patch features pooled from object boxes."
            )
        if self.actor_object_prompt_tokens and not self.actor_interaction_heatmaps:
            raise ValueError(
                "actor_object_prompt_tokens requires actor_interaction_heatmaps"
            )
        if (
            self.actor_object_prompt_tokens
            and not self.actor_object_relation_in_transformer
        ):
            raise ValueError(
                "actor_object_prompt_tokens now has one supported training path: "
                "actor_object_relation_in_transformer=1 with learned pair action "
                "scoring."
            )
        if self.actor_object_relation_in_transformer and not self.actor_object_prompt_tokens:
            raise ValueError(
                "actor_object_relation_in_transformer requires actor_object_prompt_tokens"
            )
        if self.actor_object_relation_in_transformer and not self.actor_interaction_heatmaps:
            raise ValueError(
                "actor_object_relation_in_transformer requires actor_interaction_heatmaps"
            )
        if self.actor_object_relation_in_transformer and not self.actor_object_pair_action_head:
            raise ValueError(
                "actor_object_relation_in_transformer requires "
                "actor_object_pair_action_head=1 so action CE trains the same "
                "actor-object pairs as relation CE."
            )
        if self.actor_object_pair_action_head and not self.actor_object_relation_in_transformer:
            raise ValueError(
                "actor_object_pair_action_head requires "
                "actor_object_relation_in_transformer=1"
            )
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
        self.actor_object_relation_loss_weight = float(
            hparams.get("actor_object_relation_loss_weight", 0.0)
        )
        if self.actor_object_relation_loss_weight < 0:
            raise ValueError("actor_object_relation_loss_weight must be >= 0")
        self.actor_object_relation_null_loss_weight = float(
            hparams.get("actor_object_relation_null_loss_weight", 0.5)
        )
        if self.actor_object_relation_null_loss_weight < 0:
            raise ValueError("actor_object_relation_null_loss_weight must be >= 0")
        self.actor_object_pair_action_margin_loss_weight = float(
            hparams.get("actor_object_pair_action_margin_loss_weight", 0.0)
        )
        if self.actor_object_pair_action_margin_loss_weight < 0:
            raise ValueError(
                "actor_object_pair_action_margin_loss_weight must be >= 0"
            )
        self.actor_object_pair_action_margin = float(
            hparams.get("actor_object_pair_action_margin", 0.0)
        )
        if self.actor_object_pair_action_margin < 0:
            raise ValueError("actor_object_pair_action_margin must be >= 0")
        self.actor_object_detector_dropout_prob = float(
            hparams.get("actor_object_detector_dropout_prob", 0.0)
        )
        if not 0.0 <= self.actor_object_detector_dropout_prob <= 1.0:
            raise ValueError("actor_object_detector_dropout_prob must be in [0, 1]")
        self.actor_object_detector_dropout_action_loss_weight = float(
            hparams.get("actor_object_detector_dropout_action_loss_weight", 0.0)
        )
        if self.actor_object_detector_dropout_action_loss_weight < 0:
            raise ValueError(
                "actor_object_detector_dropout_action_loss_weight must be >= 0"
            )
        self.actor_object_detector_dropout_relation_loss_weight = float(
            hparams.get("actor_object_detector_dropout_relation_loss_weight", 0.0)
        )
        if self.actor_object_detector_dropout_relation_loss_weight < 0:
            raise ValueError(
                "actor_object_detector_dropout_relation_loss_weight must be >= 0"
            )
        self.actor_object_present_margin_loss_weight = float(
            hparams.get("actor_object_present_margin_loss_weight", 0.0)
        )
        if self.actor_object_present_margin_loss_weight < 0:
            raise ValueError("actor_object_present_margin_loss_weight must be >= 0")
        self.actor_object_present_margin = float(
            hparams.get("actor_object_present_margin", 0.0)
        )
        if self.actor_object_present_margin < 0:
            raise ValueError("actor_object_present_margin must be >= 0")
        if (
            float(hparams.get("actor_object_present_gain_loss_weight", 0.0)) != 0.0
            or float(hparams.get("actor_object_present_gain_margin", 0.0)) != 0.0
        ):
            raise ValueError(
                "actor_object_present_gain_* was replaced by "
                "actor_object_present_margin_* so object-present training improves "
                "the correct-vs-hardest-wrong action margin, not only the true logit."
            )
        self.actor_object_detector_dropout_eval = bool(
            hparams.get("actor_object_detector_dropout_eval", 0)
        )
        self.actor_object_proposal_aug_prob = float(
            hparams.get("actor_object_proposal_aug_prob", 0.0)
        )
        if not 0.0 <= self.actor_object_proposal_aug_prob <= 1.0:
            raise ValueError("actor_object_proposal_aug_prob must be in [0, 1]")
        self.actor_object_proposal_box_jitter = float(
            hparams.get("actor_object_proposal_box_jitter", 0.0)
        )
        if self.actor_object_proposal_box_jitter < 0:
            raise ValueError("actor_object_proposal_box_jitter must be >= 0")
        self.actor_object_proposal_scale_jitter = float(
            hparams.get("actor_object_proposal_scale_jitter", 0.0)
        )
        if self.actor_object_proposal_scale_jitter < 0:
            raise ValueError("actor_object_proposal_scale_jitter must be >= 0")
        if float(hparams.get("actor_object_proposal_conf_jitter", 0.0)) != 0.0:
            raise ValueError(
                "actor_object_proposal_conf_jitter was removed. Detector confidence "
                "is detector-side evidence, not an augmented actor-model feature; "
                "use box jitter and distractor dropping for proposal robustness."
            )
        self.actor_object_proposal_distractor_drop_prob = float(
            hparams.get("actor_object_proposal_distractor_drop_prob", 0.0)
        )
        if not 0.0 <= self.actor_object_proposal_distractor_drop_prob <= 1.0:
            raise ValueError(
                "actor_object_proposal_distractor_drop_prob must be in [0, 1]"
            )
        self.actor_pair_train_weight = float(hparams.get("actor_pair_train_weight", 0.0))
        if self.actor_pair_train_weight < 0:
            raise ValueError("actor_pair_train_weight must be >= 0")
        if (
            self.actor_pair_train_weight > 0
            and int(hparams.get("num_actor_tokens", 8)) < 2
        ):
            raise ValueError(
                "actor_pair_train_weight requires num_actor_tokens >= 2"
            )
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
        else:
            self.action_names = [str(index) for index in range(self.num_classes)]
            self.action_to_index = {}
            self.action_object_map = {}
            self.action_object_ids_by_index = {}

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
            "relation_action_joint": [],
            "relation_action_joint_exact": [],
            "relation_action_joint_objectless": [],
            "relation_action_joint_missing_objectful": [],
            "object_dropout_action": [],
            "object_dropout_action_Uselaptop": [],
            "object_dropout_relation_action_joint_missing_objectful": [],
            "object_dropout_relation_action_joint_missing_Uselaptop": [],
            "object_present_true_prob_gain": [],
            "object_present_true_logit_gain": [],
            "object_present_action_margin_gain": [],
            "object_present_Uselaptop_prob_gain": [],
            "object_present_Uselaptop_confuser_margin_gain": [],
        }

    def load_state_dict(self, state_dict, strict=True, assign=False):
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        stale_object_conf = [
            key
            for key in result.unexpected_keys
            if "object_conf_mlp" in key
        ]
        if stale_object_conf:
            preview = ", ".join(stale_object_conf[:12])
            raise RuntimeError(
                "Checkpoint contains removed learned object-confidence embedding "
                "weights. Detector confidence is no longer a learned actor-model "
                "feature; thresholded object_valid proposals provide object "
                "existence evidence. Retrain from a clean base checkpoint. "
                f"First stale keys: {preview}"
            )
        if self.actor_prompt and not strict:
            allowed_missing = [
                "model.net.actor_token",
                "model.net.actor_slot_embed",
                "model.net.valid_embed",
                "model.net.bbox_mlp",
                "model.net.object_slot_embed",
                "model.net.object_class_embed",
                "model.net.object_box_mlp",
                "model.net.object_valid_embed",
                "model.net.actor_object_relation_updates",
                "model.net.actor_object_final_relation_update",
                "model.actor_head",
                "model.actor_object_null_pair_token",
                "model.actor_object_pair_action_head",
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
        object_boxes=None,
        object_classes=None,
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
                "object_valid_embed",
                "actor_object_relation_updates",
                "actor_object_final_relation_update",
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

    def _head_params(self):
        if not self.actor_prompt:
            return list(self.model.head.parameters())

        params = []
        if getattr(self.model, "actor_head", None) is not None:
            params += list(self.model.actor_head.parameters())
        if getattr(self.model, "actor_object_pair_action_head", None) is not None:
            params += list(self.model.actor_object_pair_action_head.parameters())
        null_pair_token = getattr(self.model, "actor_object_null_pair_token", None)
        if isinstance(null_pair_token, nn.Parameter):
            params.append(null_pair_token)
        if self.model.presence_head is not None:
            params += list(self.model.presence_head.parameters())
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

    @staticmethod
    def _actor_model_object_inputs(object_inputs):
        if not object_inputs:
            return {}
        return {
            key: value
            for key, value in object_inputs.items()
            if key != "object_confs"
        }


    def _teacher_object_slot_mask(self, target, shape, device):
        teacher_mask = torch.zeros(shape, device=device, dtype=torch.bool)
        if (
            "interaction_object_index" not in target
            or "interaction_object_index_valid" not in target
        ):
            return teacher_mask
        selected = target["interaction_object_index"].to(device=device, dtype=torch.long)
        selected_valid = target["interaction_object_index_valid"].to(
            device=device,
            dtype=torch.bool,
        )
        if selected.ndim != 2 or selected_valid.shape != selected.shape:
            return teacher_mask
        num_objects = int(shape[1])
        one_based = selected_valid & (selected > 0) & ((selected - 1) < num_objects)
        if not one_based.any():
            return teacher_mask
        batch_idx, _ = torch.nonzero(one_based, as_tuple=True)
        object_idx = (selected[one_based] - 1).clamp(0, num_objects - 1)
        teacher_mask[batch_idx, object_idx] = True
        return teacher_mask

    def _augment_object_inputs_for_training(self, object_inputs, target, stage):
        device = object_inputs["object_valid"].device
        dropped_target_mask = torch.zeros(target["valid"].shape, dtype=torch.bool, device=device)

        if (
            not stage.startswith("train")
            or not self.uses_object_proposals
            or not object_inputs
        ):
            return object_inputs, dropped_target_mask

        object_valid = object_inputs["object_valid"].to(dtype=torch.bool)
        if not object_valid.any():
            return object_inputs, dropped_target_mask

        boxes = object_inputs["object_boxes"].clone()
        original_boxes = boxes.clone()
        confs = object_inputs["object_confs"].clone()
        output_valid = object_valid.clone()

        if self.actor_object_proposal_aug_prob > 0.0:
            aug_mask = object_valid & (
                torch.rand(object_valid.shape, device=device, dtype=torch.float32)
                < self.actor_object_proposal_aug_prob
            )
            if aug_mask.any():
                if self.actor_object_proposal_box_jitter > 0.0:
                    center = (boxes[..., :2] + boxes[..., 2:]) * 0.5
                    size = (boxes[..., 2:] - boxes[..., :2]).clamp_min(1.0e-4)
                    center_noise = (
                        torch.rand_like(center) * 2.0 - 1.0
                    ) * size * self.actor_object_proposal_box_jitter
                    center = center + torch.where(
                        aug_mask[..., None],
                        center_noise,
                        torch.zeros_like(center_noise),
                    )
                    if self.actor_object_proposal_scale_jitter > 0.0:
                        scale_noise = (
                            torch.rand_like(size) * 2.0 - 1.0
                        ) * self.actor_object_proposal_scale_jitter
                        size = size * (1.0 + scale_noise).clamp_min(0.25)
                    half = size * 0.5
                    jittered_boxes = torch.cat([center - half, center + half], dim=-1)
                    boxes = torch.where(aug_mask[..., None], jittered_boxes, boxes)
                    boxes = boxes.clamp(0.0, 1.0)
                    min_xy = torch.minimum(boxes[..., :2], boxes[..., 2:])
                    max_xy = torch.maximum(boxes[..., :2], boxes[..., 2:])
                    jittered_boxes = torch.cat([min_xy, max_xy], dim=-1)
                    jittered_size = max_xy - min_xy
                    valid_jittered = aug_mask & (jittered_size > 1.0e-4).all(dim=-1)
                    boxes = torch.where(
                        valid_jittered[..., None],
                        jittered_boxes,
                        original_boxes,
                    )
                self._log_scalar(
                    f"{stage}_object_proposal_aug_rate",
                    aug_mask.float().mean(),
                    aug_mask.numel(),
                )

        if self.actor_object_proposal_distractor_drop_prob > 0.0:
            teacher_mask = self._teacher_object_slot_mask(
                target,
                object_valid.shape,
                device,
            )
            distractor_mask = object_valid & ~teacher_mask
            drop_mask = distractor_mask & (
                torch.rand(object_valid.shape, device=device, dtype=torch.float32)
                < self.actor_object_proposal_distractor_drop_prob
            )
            output_valid = output_valid & ~drop_mask
            confs = confs.masked_fill(drop_mask, 0.0)
            self._log_scalar(
                f"{stage}_object_proposal_distractor_drop_rate",
                drop_mask.float().mean(),
                drop_mask.numel(),
            )

        if self.actor_object_detector_dropout_prob > 0.0:
            actions = target["actions"].to(device=device, dtype=torch.long)
            valid = target["valid"].to(device=device, dtype=torch.bool)
            info = self._exact_teacher_object_info(actions, valid, target, device)
            if info is not None:
                exact = info["valid"] & info["known_action"] & info["compatible_from_one_based"]
                if exact.any():
                    dropped_target_mask = exact & (
                        torch.rand(exact.shape, device=device, dtype=torch.float32)
                        < self.actor_object_detector_dropout_prob
                    )
                    if dropped_target_mask.any():
                        object_classes = object_inputs["object_classes"].to(device=device, dtype=torch.long)
                        dropped_object_mask = torch.zeros_like(object_valid, dtype=torch.bool)
                        for action_idx, object_ids in self.action_object_ids_by_index.items():
                            action_actor_mask = dropped_target_mask & (actions == int(action_idx))
                            if not action_actor_mask.any():
                                continue
                            sample_mask = action_actor_mask.any(dim=1)
                            object_ids = object_ids.to(device=device, dtype=torch.long)
                            class_match = (
                                object_classes.unsqueeze(-1) == object_ids.view(1, 1, -1)
                            ).any(dim=-1)
                            dropped_object_mask |= sample_mask[:, None] & object_valid & class_match

                        output_valid = output_valid & ~dropped_object_mask
                        confs = confs.masked_fill(dropped_object_mask, 0.0)
                        self._log_scalar(
                            f"{stage}_object_detector_dropout_rate",
                            dropped_target_mask.float().mean(),
                            dropped_target_mask.numel(),
                        )

        return {
            "object_boxes": boxes,
            "object_classes": object_inputs["object_classes"],
            "object_confs": confs,
            "object_valid": output_valid,
        }, dropped_target_mask

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

        # Toyota interaction_object_index uses one-based object slots in the
        # current preprocessed targets. Keep the zero-based interpretation
        # around only as a compatibility audit path.
        slot_index_from_one_based = (selected_indices - 1).clamp(0, num_objects - 1)
        slot_index_from_zero_based = selected_indices.clamp(0, num_objects - 1)
        in_range_one_based = (
            selected_valid
            & (selected_indices > 0)
            & ((selected_indices - 1) < num_objects)
        )
        in_range_zero_based = (
            selected_valid
            & (selected_indices >= 0)
            & (selected_indices < num_objects)
        )
        class_from_one_based = object_classes.gather(1, slot_index_from_one_based)
        class_from_zero_based = object_classes.gather(1, slot_index_from_zero_based)
        valid_from_one_based = in_range_one_based & object_valid.gather(
            1,
            slot_index_from_one_based,
        )
        valid_from_zero_based = in_range_zero_based & object_valid.gather(
            1,
            slot_index_from_zero_based,
        )

        known_action = torch.zeros_like(valid, dtype=torch.bool)
        any_compatible = torch.zeros_like(valid, dtype=torch.bool)
        compatible_from_one_based = torch.zeros_like(valid, dtype=torch.bool)
        compatible_from_zero_based = torch.zeros_like(valid, dtype=torch.bool)
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
            class_match_one_based = (
                class_from_one_based.unsqueeze(-1) == object_ids.view(1, 1, -1)
            ).any(dim=-1)
            class_match_zero_based = (
                class_from_zero_based.unsqueeze(-1) == object_ids.view(1, 1, -1)
            ).any(dim=-1)
            compatible_from_one_based |= (
                action_mask & valid_from_one_based & class_match_one_based
            )
            compatible_from_zero_based |= (
                action_mask & valid_from_zero_based & class_match_zero_based
            )

        return {
            "actions": actions,
            "valid": valid,
            "object_classes": object_classes,
            "object_valid": object_valid,
            "selected_indices": selected_indices,
            "slot_index_from_one_based": slot_index_from_one_based,
            "slot_index_from_zero_based": slot_index_from_zero_based,
            "valid_from_one_based": valid_from_one_based,
            "valid_from_zero_based": valid_from_zero_based,
            "known_action": known_action,
            "any_compatible": any_compatible,
            "compatible_from_one_based": compatible_from_one_based,
            "compatible_from_zero_based": compatible_from_zero_based,
        }

    def _log_token_selection_diagnostics(self, stage, actions, valid, target):
        if stage.startswith("train"):
            return
        net = getattr(self.model, "net", None)
        diagnostics = getattr(net, "last_token_selection_diagnostics", None)
        if not diagnostics:
            return
        selected_mask = diagnostics.get("selected_mask")
        window_size = diagnostics.get("window_size")
        if selected_mask is None or window_size is None:
            return

        device = selected_mask.device
        selected_mask = selected_mask.to(device=device, dtype=torch.float32)
        if selected_mask.ndim != 2:
            raise RuntimeError(
                "token-selection selected_mask must have shape [B,N], got "
                f"{tuple(selected_mask.shape)}"
            )
        batch_size, num_tokens = selected_mask.shape
        count = int(batch_size)
        self._log_scalar(
            f"{stage}_token_selection_visual_keep_rate",
            selected_mask.mean(),
            count,
        )
        self._log_scalar(
            f"{stage}_token_selection_visual_keep_count",
            selected_mask.sum(dim=-1).mean(),
            count,
        )

        def log_region_keep_rate(metric_name, prior, sample_mask, selected=None):
            if prior is None:
                return
            prior = prior.to(device=device, dtype=torch.float32)
            selected_for_prior = selected_mask if selected is None else selected
            selected_for_prior = selected_for_prior.to(device=device, dtype=torch.float32)
            if prior.shape != selected_for_prior.shape:
                raise RuntimeError(
                    f"{metric_name} prior shape {tuple(prior.shape)} does not "
                    f"match selected mask {tuple(selected_for_prior.shape)}"
                )
            sample_mask = sample_mask.to(device=device, dtype=torch.bool)
            denom = prior.sum(dim=-1)
            keep_valid = sample_mask & (denom > 0)
            if not keep_valid.any():
                return
            keep_rate = (prior * selected_for_prior).sum(dim=-1) / denom.clamp_min(
                1.0e-6
            )
            sample_count = int(keep_valid.sum().item())
            self._log_scalar(
                f"{stage}_{metric_name}",
                keep_rate[keep_valid].mean(),
                sample_count,
            )
            self._log_count(
                f"{stage}_{metric_name}_count",
                keep_valid.float().sum(),
            )

        if not hasattr(net, "_make_box_token_prior"):
            return

        actions = actions.to(device=device, dtype=torch.long)
        valid = valid.to(device=device, dtype=torch.bool)
        boxes = target.get("boxes")
        if boxes is not None:
            boxes = boxes.to(device=device, dtype=torch.float32)
            actor_prior = net._make_box_token_prior(
                boxes,
                valid,
                window_size,
                expand=float(self.model.hparams.get("actor_bbox_prior_expand", 1.75)),
            )
            log_region_keep_rate(
                "token_selection_actor_box_keep_rate",
                actor_prior,
                valid.any(dim=1),
            )

        object_boxes = target.get("object_boxes")
        object_valid = target.get("object_valid")
        object_classes = target.get("object_classes")
        if (
            object_boxes is not None
            and object_valid is not None
            and object_classes is not None
        ):
            object_boxes = object_boxes.to(device=device, dtype=torch.float32)
            object_valid = object_valid.to(device=device, dtype=torch.bool)
            object_classes = object_classes.to(device=device, dtype=torch.long)
            object_prior = net._make_box_token_prior(
                object_boxes,
                object_valid,
                window_size,
                expand=float(
                    self.model.hparams.get(
                        "actor_object_prompt_box_prior_expand",
                        1.25,
                    )
                ),
            )
            log_region_keep_rate(
                "token_selection_visible_object_box_keep_rate",
                object_prior,
                object_valid.any(dim=1),
            )
            for object_name in ("laptop", "book", "phone", "tv_monitor"):
                object_id = OBJECT_TO_ID.get(object_name)
                if object_id is None:
                    continue
                class_valid = object_valid & (object_classes == int(object_id))
                class_prior = net._make_box_token_prior(
                    object_boxes,
                    class_valid,
                    window_size,
                    expand=float(
                        self.model.hparams.get(
                            "actor_object_prompt_box_prior_expand",
                            1.25,
                        )
                    ),
                )
                safe_name = object_name.replace(".", "_")
                log_region_keep_rate(
                    f"token_selection_{safe_name}_box_keep_rate",
                    class_prior,
                    class_valid.any(dim=1),
                )

            info = self._exact_teacher_object_info(actions, valid, target, device)
            exact_compatible = None
            if info is not None:
                exact_compatible = (
                    info["valid"]
                    & info["known_action"]
                    & info["compatible_from_one_based"]
                )
            if exact_compatible is not None and exact_compatible.any():
                B, A = actions.shape
                if B != batch_size:
                    raise RuntimeError(
                        "token-selection diagnostics batch mismatch: "
                        f"{B} actions vs {batch_size} selected masks"
                    )
                teacher_slots = info["slot_index_from_one_based"].to(
                    device=device,
                    dtype=torch.long,
                )
                teacher_boxes = object_boxes.gather(
                    1,
                    teacher_slots.unsqueeze(-1).expand(-1, -1, 4),
                )
                flat_teacher_boxes = teacher_boxes.reshape(B * A, 1, 4)
                flat_teacher_valid = exact_compatible.reshape(B * A, 1)
                flat_prior = net._make_box_token_prior(
                    flat_teacher_boxes,
                    flat_teacher_valid,
                    window_size,
                    expand=float(
                        self.model.hparams.get(
                            "actor_object_prompt_box_prior_expand",
                            1.25,
                        )
                    ),
                )
                flat_selected = (
                    selected_mask[:, None, :]
                    .expand(-1, A, -1)
                    .reshape(B * A, num_tokens)
                )
                flat_exact = exact_compatible.reshape(B * A)
                log_region_keep_rate(
                    "token_selection_exact_teacher_object_keep_rate",
                    flat_prior,
                    flat_exact,
                    selected=flat_selected,
                )
                for action_name in (
                    "Uselaptop",
                    "Readbook",
                    "WatchTV",
                    "Usetelephone",
                ):
                    action_idx = self._action_index(action_name)
                    if action_idx is None:
                        continue
                    action_mask = exact_compatible & (actions == int(action_idx))
                    safe_name = action_name.replace(".", "_")
                    log_region_keep_rate(
                        f"token_selection_{safe_name}_teacher_object_keep_rate",
                        flat_prior,
                        action_mask.reshape(B * A),
                        selected=flat_selected,
                    )

        if (
            "interaction_heatmap" in target
            and "interaction_heatmap_valid" in target
            and len(window_size) == 3
        ):
            target_heatmap = target["interaction_heatmap"].to(
                device=device,
                dtype=torch.float32,
            )
            heatmap_valid = target["interaction_heatmap_valid"].to(
                device=device,
                dtype=torch.bool,
            )
            if target_heatmap.ndim != 4:
                raise RuntimeError(
                    "interaction_heatmap must have shape [B,A,H,W], got "
                    f"{tuple(target_heatmap.shape)}"
                )
            B, A, _, _ = target_heatmap.shape
            frames, height, width = [int(v) for v in window_size]
            flat_heatmap = target_heatmap.reshape(B * A, 1, *target_heatmap.shape[-2:])
            spatial_prior = F.interpolate(
                flat_heatmap,
                size=(height, width),
                mode="area",
            ).reshape(B * A, height * width)
            heatmap_prior = (
                spatial_prior[:, None, :]
                .expand(-1, frames, -1)
                .reshape(B * A, frames * height * width)
                .clamp_min(0.0)
            )
            flat_selected = (
                selected_mask[:, None, :]
                .expand(-1, A, -1)
                .reshape(B * A, num_tokens)
            )
            heatmap_sample_valid = (
                valid
                & heatmap_valid
                & (target_heatmap.flatten(2).sum(dim=-1) > 0)
            )
            log_region_keep_rate(
                "token_selection_interaction_heatmap_keep_rate",
                heatmap_prior,
                heatmap_sample_valid.reshape(B * A),
                selected=flat_selected,
            )

    def _actor_object_relation_loss(self, stage, actions, valid, target):
        is_train = stage.startswith("train")
        if (
            not self.actor_object_relation_in_transformer
            or self.actor_object_relation_loss_weight <= 0
        ):
            return None
        relation_aux = getattr(self.model, "last_actor_object_relation_aux", None)
        if not relation_aux:
            return None
        last_block_name = sorted(relation_aux.keys(), key=lambda value: int(value))[-1]
        relation_logit_scale = relation_aux[last_block_name].get("relation_logit_scale")
        if relation_logit_scale is not None:
            self._log_scalar(
                f"{stage}_relation_logit_scale",
                relation_logit_scale.float(),
                1,
            )
        object_region_norm = getattr(
            self.model,
            "last_actor_object_region_visual_norm",
            None,
        )
        if object_region_norm is not None:
            target_valid = target.get("object_valid")
            if target_valid is not None:
                valid_object_norm = object_region_norm.to(
                    device=target_valid.device,
                    dtype=torch.float32,
                )[target_valid.bool()]
                if valid_object_norm.numel() > 0:
                    self._log_scalar(
                        f"{stage}_object_region_visual_feature_norm",
                        valid_object_norm.mean(),
                        int(valid_object_norm.numel()),
                    )

        first_aux = next(iter(relation_aux.values()))
        logits0 = first_aux.get("logits")
        if logits0 is None:
            return None
        device = logits0.device
        info = self._exact_teacher_object_info(actions, valid, target, device)
        if info is None:
            return None

        actions = actions.to(device=device, dtype=torch.long)
        valid = valid.to(device=device, dtype=torch.bool)
        objectless = self._labels_in_indices(actions, self.objectless_action_indices)
        objectless = valid & objectless
        known_objectful = info["valid"] & info["known_action"]
        exact_compatible = known_objectful & info["compatible_from_one_based"]
        missing_compatible = known_objectful & ~info["any_compatible"]

        target_index = torch.zeros_like(actions, dtype=torch.long, device=device)
        selected_index = info["selected_indices"].to(device=device, dtype=torch.long)
        max_object_index = int(logits0.shape[-1]) - 1
        target_index[exact_compatible] = selected_index[exact_compatible].clamp(
            1,
            max_object_index,
        )

        null_weight = self.actor_object_relation_null_loss_weight
        null_supervised = missing_compatible | objectless
        supervised = exact_compatible | (null_supervised & (null_weight > 0))
        if not supervised.any():
            return None

        losses = []
        for block_name, aux in relation_aux.items():
            logits = aux.get("logits")
            if logits is None:
                continue
            if logits.shape[:2] != target_index.shape:
                raise RuntimeError(
                    "actor-object relation logits shape mismatch: "
                    f"{tuple(logits.shape[:2])} vs {tuple(target_index.shape)}"
                )
            per_item = F.cross_entropy(
                logits.float().reshape(-1, logits.shape[-1]),
                target_index.reshape(-1),
                reduction="none",
            ).reshape_as(target_index)

            block_components = []
            block_weights = []
            exact_loss = None
            if exact_compatible.any():
                exact_loss = per_item[exact_compatible].mean()
                block_components.append(exact_loss)
                block_weights.append(per_item.new_tensor(1.0))

            null_loss = None
            if null_supervised.any() and null_weight > 0:
                null_parts = []
                if missing_compatible.any():
                    null_parts.append(per_item[missing_compatible].mean())
                if objectless.any():
                    null_parts.append(per_item[objectless].mean())
                if null_parts:
                    null_loss = torch.stack(null_parts).mean()
                    block_components.append(null_loss * null_weight)
                    block_weights.append(per_item.new_tensor(null_weight))

            if not block_components:
                continue
            block_loss = torch.stack(block_components).sum() / torch.stack(
                block_weights
            ).sum().clamp_min(1e-6)
            losses.append(block_loss)
            self.log(
                f"{stage}_loss_actor_object_relation_block_{block_name}",
                block_loss,
                on_step=is_train,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )
            if exact_loss is not None:
                self.log(
                    f"{stage}_loss_actor_object_relation_exact_block_{block_name}",
                    exact_loss,
                    on_step=is_train,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                    sync_dist=True,
                )
            if null_loss is not None:
                self.log(
                    f"{stage}_loss_actor_object_relation_null_block_{block_name}",
                    null_loss,
                    on_step=is_train,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                    sync_dist=True,
                )

        if not losses:
            return None

        loss = torch.stack(losses).mean()
        weighted_loss = loss * self.actor_object_relation_loss_weight
        self.log(
            f"{stage}_loss_actor_object_relation",
            loss,
            on_step=is_train,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        self._log_actor_object_relation_metrics(
            stage,
            relation_aux,
            target_index,
            exact_compatible,
            missing_compatible,
            objectless,
            info,
        )
        return weighted_loss

    def _actor_object_pair_action_margin_loss(self, stage, actions, valid, target):
        if not self.actor_object_pair_action_head:
            return None
        pair_scores = getattr(self.model, "last_actor_object_pair_action_scores", None)
        pair_allowed = getattr(self.model, "last_actor_object_pair_action_allowed", None)
        if pair_scores is None or pair_allowed is None:
            return None
        if pair_scores.ndim != 4:
            raise RuntimeError(
                "pair action scores must have shape [B,A,P,C], got "
                f"{tuple(pair_scores.shape)}"
            )

        device = pair_scores.device
        actions = actions.to(device=device, dtype=torch.long)
        valid = valid.to(device=device, dtype=torch.bool)
        pair_scores = pair_scores.float()
        pair_allowed = pair_allowed.to(device=device, dtype=torch.bool)
        info = self._exact_teacher_object_info(actions, valid, target, device)
        if info is None:
            return None

        objectless = self._labels_in_indices(actions, self.objectless_action_indices)
        objectless = info["valid"] & objectless
        known_objectful = info["valid"] & info["known_action"]
        exact_compatible = known_objectful & info["compatible_from_one_based"]
        missing_compatible = known_objectful & ~info["any_compatible"]
        supervised = exact_compatible | objectless | missing_compatible
        supervised = supervised & (actions >= 0) & (actions < pair_scores.shape[-1])
        if not supervised.any():
            return None

        num_pairs = int(pair_scores.shape[2])
        target_pair = torch.zeros_like(actions, dtype=torch.long, device=device)
        selected_index = info["selected_indices"].to(device=device, dtype=torch.long)
        if num_pairs > 1:
            target_pair[exact_compatible] = selected_index[exact_compatible].clamp(
                1,
                num_pairs - 1,
            )

        scores_s = pair_scores[supervised]
        allowed_s = pair_allowed[supervised]
        labels_s = actions[supervised]
        pair_s = target_pair[supervised].clamp(0, num_pairs - 1)
        rows = torch.arange(scores_s.shape[0], device=device)
        target_allowed = allowed_s[rows, pair_s, labels_s]
        if not bool(target_allowed.all()):
            raise RuntimeError(
                "The supervised pair/action target is masked out. This means the "
                "Toyota action-object map, target interaction object, or object "
                "validity disagrees with pair scorer routing."
            )

        true_scores = scores_s[rows, pair_s, labels_s]
        wrong_scores = scores_s.masked_fill(~allowed_s, -1.0e4).clone()
        wrong_scores[rows, pair_s, labels_s] = -1.0e4
        max_wrong = wrong_scores.reshape(scores_s.shape[0], -1).amax(dim=-1)
        usable = max_wrong > -9999.0
        if not usable.any():
            return None

        margins = true_scores[usable] - max_wrong[usable]
        count = int(margins.numel())
        self._log_scalar(
            f"{stage}_actor_object_pair_action_margin",
            margins.mean(),
            count,
        )
        self._log_scalar(
            f"{stage}_actor_object_pair_action_margin_win_rate",
            (margins > 0).float().mean(),
            count,
        )
        uselaptop_idx = self._action_index("Uselaptop")
        if uselaptop_idx is not None:
            labels_usable = labels_s[usable]
            uselaptop_mask = labels_usable == int(uselaptop_idx)
            if uselaptop_mask.any():
                uselaptop_count = int(uselaptop_mask.sum().item())
                self._log_scalar(
                    f"{stage}_actor_object_pair_action_Uselaptop_margin",
                    margins[uselaptop_mask].mean(),
                    uselaptop_count,
                )
                self._log_scalar(
                    f"{stage}_actor_object_pair_action_Uselaptop_margin_win_rate",
                    (margins[uselaptop_mask] > 0).float().mean(),
                    uselaptop_count,
                )

        if (
            not stage.startswith("train")
            or self.actor_object_pair_action_margin_loss_weight <= 0.0
        ):
            return None

        margin_target = margins.new_tensor(float(self.actor_object_pair_action_margin))
        loss = F.relu(margin_target - margins).mean()
        self.log(
            f"{stage}_loss_actor_object_pair_action_margin",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        return loss * self.actor_object_pair_action_margin_loss_weight

    def _log_actor_object_relation_metrics(
        self,
        stage,
        relation_aux,
        target_index,
        exact_compatible,
        missing_compatible,
        objectless,
        info,
    ):
        if not relation_aux:
            return
        last_aux = relation_aux[sorted(relation_aux.keys(), key=lambda x: int(x))[-1]]
        logits = last_aux.get("logits")
        if logits is None:
            return
        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            if exact_compatible.any():
                count = int(exact_compatible.sum().item())
                self._log_scalar(
                    f"{stage}_relation_exact_teacher_acc",
                    (pred[exact_compatible] == target_index[exact_compatible])
                    .float()
                    .mean(),
                    count,
                )
                object_attention = last_aux.get("object_attention")
                if object_attention is not None:
                    teacher_slot = info["slot_index_from_one_based"].to(
                        device=object_attention.device,
                        dtype=torch.long,
                    )
                    teacher_prob = object_attention.float().gather(
                        -1,
                        teacher_slot.unsqueeze(-1),
                    ).squeeze(-1)
                    self._log_scalar(
                        f"{stage}_relation_exact_teacher_prob",
                        teacher_prob[exact_compatible].mean(),
                        count,
                    )
                useful_mass = last_aux.get("useful_mass")
                if useful_mass is not None:
                    self._log_scalar(
                        f"{stage}_relation_useful_mass_exact",
                        useful_mass.float()[exact_compatible].mean(),
                        count,
                    )
                null_prob = last_aux.get("null_prob")
                if null_prob is not None:
                    self._log_scalar(
                        f"{stage}_relation_null_prob_exact",
                        null_prob.float()[exact_compatible].mean(),
                        count,
                    )

            if missing_compatible.any():
                count = int(missing_compatible.sum().item())
                self._log_scalar(
                    f"{stage}_relation_null_rate_missing_objectful",
                    (pred[missing_compatible] == 0).float().mean(),
                    count,
                )
                useful_mass = last_aux.get("useful_mass")
                if useful_mass is not None:
                    self._log_scalar(
                        f"{stage}_relation_useful_mass_missing_objectful",
                        useful_mass.float()[missing_compatible].mean(),
                        count,
                    )
                null_prob = last_aux.get("null_prob")
                if null_prob is not None:
                    self._log_scalar(
                        f"{stage}_relation_null_prob_missing_objectful",
                        null_prob.float()[missing_compatible].mean(),
                        count,
                    )

            if objectless.any():
                count = int(objectless.sum().item())
                self._log_scalar(
                    f"{stage}_relation_null_rate_objectless",
                    (pred[objectless] == 0).float().mean(),
                    count,
                )
                useful_mass = last_aux.get("useful_mass")
                if useful_mass is not None:
                    self._log_scalar(
                        f"{stage}_relation_useful_mass_objectless",
                        useful_mass.float()[objectless].mean(),
                        count,
                    )
                null_prob = last_aux.get("null_prob")
                if null_prob is not None:
                    self._log_scalar(
                        f"{stage}_relation_null_prob_objectless",
                        null_prob.float()[objectless].mean(),
                        count,
                    )

    def _relation_action_joint_info(self, preds, actions, valid, target):
        if not self.actor_object_relation_in_transformer:
            return None
        relation_aux = getattr(self.model, "last_actor_object_relation_aux", None)
        if not relation_aux:
            return None
        last_aux = relation_aux[sorted(relation_aux.keys(), key=lambda x: int(x))[-1]]
        relation_logits = last_aux.get("logits")
        if relation_logits is None:
            return None

        device = relation_logits.device
        actions = actions.to(device=device, dtype=torch.long)
        valid = valid.to(device=device, dtype=torch.bool)
        preds = preds.to(device=device)
        if relation_logits.shape[:2] != actions.shape:
            raise RuntimeError(
                "actor-object relation logits shape mismatch for joint metrics: "
                f"{tuple(relation_logits.shape[:2])} vs {tuple(actions.shape)}"
            )
        if preds.shape[:2] != actions.shape:
            raise RuntimeError(
                "action logits shape mismatch for joint relation-action metrics: "
                f"{tuple(preds.shape[:2])} vs {tuple(actions.shape)}"
            )

        info = self._exact_teacher_object_info(actions, valid, target, device)
        if info is None:
            return None

        objectless = self._labels_in_indices(actions, self.objectless_action_indices)
        objectless = info["valid"] & objectless
        known_objectful = info["valid"] & info["known_action"]
        exact_compatible = known_objectful & info["compatible_from_one_based"]
        missing_compatible = known_objectful & ~info["any_compatible"]

        target_index = torch.zeros_like(actions, dtype=torch.long, device=device)
        selected_index = info["selected_indices"].to(device=device, dtype=torch.long)
        max_object_index = int(relation_logits.shape[-1]) - 1
        target_index[exact_compatible] = selected_index[exact_compatible].clamp(
            1,
            max_object_index,
        )

        relation_pred = relation_logits.argmax(dim=-1)
        action_pred = preds.argmax(dim=-1)
        action_correct = info["valid"] & (action_pred == actions)
        relation_goal_correct = torch.zeros_like(info["valid"], dtype=torch.bool)
        relation_goal_correct[exact_compatible] = (
            relation_pred[exact_compatible] == target_index[exact_compatible]
        )
        null_supervised = objectless | missing_compatible
        relation_goal_correct[null_supervised] = relation_pred[null_supervised] == 0
        supervised = exact_compatible | null_supervised
        joint_correct = action_correct & relation_goal_correct

        return {
            "actions": actions,
            "action_logits": preds,
            "action_correct": action_correct,
            "relation_goal_correct": relation_goal_correct,
            "joint_correct": joint_correct,
            "supervised": supervised,
            "exact_compatible": exact_compatible,
            "objectless": objectless,
            "missing_compatible": missing_compatible,
        }

    def _log_relation_action_joint_metrics(self, stage, preds, actions, valid, target):
        info = self._relation_action_joint_info(preds, actions, valid, target)
        if info is None:
            return {}

        def log_masked(name, values, mask):
            if not mask.any():
                return
            count = int(mask.sum().item())
            self._log_scalar(name, values[mask].float().mean(), count)
            self._log_count(f"{name}_count", mask.float().sum())

        with torch.no_grad():
            joint_correct = info["joint_correct"]
            action_correct = info["action_correct"]
            relation_correct = info["relation_goal_correct"]
            supervised = info["supervised"]
            exact = info["exact_compatible"]
            objectless = info["objectless"]
            missing = info["missing_compatible"]

            log_masked(
                f"{stage}_relation_action_joint_acc",
                joint_correct,
                supervised,
            )
            log_masked(
                f"{stage}_relation_action_joint_exact_acc",
                joint_correct,
                exact,
            )
            log_masked(
                f"{stage}_relation_action_joint_objectless_acc",
                joint_correct,
                objectless,
            )
            log_masked(
                f"{stage}_relation_action_joint_missing_objectful_acc",
                joint_correct,
                missing,
            )

            relation_correct_action_wrong = relation_correct & ~action_correct
            action_correct_relation_wrong = action_correct & ~relation_correct
            log_masked(
                f"{stage}_relation_correct_action_wrong_exact_rate",
                relation_correct_action_wrong,
                exact,
            )
            log_masked(
                f"{stage}_action_correct_relation_wrong_exact_rate",
                action_correct_relation_wrong,
                exact,
            )
            relation_correct_exact = exact & relation_correct
            action_correct_exact = exact & action_correct
            log_masked(
                f"{stage}_action_acc_when_relation_exact",
                action_correct,
                relation_correct_exact,
            )
            log_masked(
                f"{stage}_relation_exact_when_action_correct",
                relation_correct,
                action_correct_exact,
            )

            actions = info["actions"]
            for action_name in RELATION_ACTION_AUDIT_ACTIONS:
                action_idx = self._action_index(action_name)
                if action_idx is None:
                    continue
                action_mask = exact & (actions == int(action_idx))
                missing_action_mask = missing & (actions == int(action_idx))
                safe_name = action_name.replace(".", "_")
                log_masked(
                    f"{stage}_relation_action_joint_{safe_name}_acc",
                    joint_correct,
                    action_mask,
                )
                log_masked(
                    f"{stage}_relation_correct_action_wrong_{safe_name}_rate",
                    relation_correct_action_wrong,
                    action_mask,
                )
                log_masked(
                    f"{stage}_relation_action_joint_missing_{safe_name}_acc",
                    joint_correct,
                    missing_action_mask,
                )
                log_masked(
                    f"{stage}_relation_correct_action_wrong_missing_{safe_name}_rate",
                    relation_correct_action_wrong,
                    missing_action_mask,
                )

            uselaptop_idx = self._action_index("Uselaptop")
            if uselaptop_idx is not None:
                confuser_indices = [
                    self._action_index(action_name)
                    for action_name in RELATION_ACTION_AUDIT_ACTIONS
                    if action_name != "Uselaptop"
                ]
                confuser_indices = [
                    int(index) for index in confuser_indices if index is not None
                ]
                uselaptop_mask = exact & (actions == int(uselaptop_idx))
                uselaptop_missing_mask = missing & (actions == int(uselaptop_idx))
                if confuser_indices and uselaptop_mask.any():
                    action_logits = info["action_logits"].float()
                    true_logit = action_logits[..., int(uselaptop_idx)]
                    confuser_tensor = torch.tensor(
                        confuser_indices,
                        device=action_logits.device,
                        dtype=torch.long,
                    )
                    confuser_logit = action_logits.index_select(
                        -1,
                        confuser_tensor,
                    ).amax(dim=-1)
                    margin = true_logit - confuser_logit
                    count = int(uselaptop_mask.sum().item())
                    self._log_scalar(
                        f"{stage}_action_Uselaptop_object_confuser_margin",
                        margin[uselaptop_mask].mean(),
                        count,
                    )
                    self._log_scalar(
                        f"{stage}_action_Uselaptop_object_confuser_win_rate",
                        (margin[uselaptop_mask] > 0).float().mean(),
                        count,
                    )
                if confuser_indices and uselaptop_missing_mask.any():
                    action_logits = info["action_logits"].float()
                    true_logit = action_logits[..., int(uselaptop_idx)]
                    confuser_tensor = torch.tensor(
                        confuser_indices,
                        device=action_logits.device,
                        dtype=torch.long,
                    )
                    confuser_logit = action_logits.index_select(
                        -1,
                        confuser_tensor,
                    ).amax(dim=-1)
                    margin = true_logit - confuser_logit
                    count = int(uselaptop_missing_mask.sum().item())
                    self._log_scalar(
                        f"{stage}_action_Uselaptop_missing_object_confuser_margin",
                        margin[uselaptop_missing_mask].mean(),
                        count,
                    )
                    self._log_scalar(
                        f"{stage}_action_Uselaptop_missing_object_confuser_win_rate",
                        (margin[uselaptop_missing_mask] > 0).float().mean(),
                        count,
                    )

            if not stage.startswith("val"):
                return {}
            outputs = {}
            for name, mask in (
                ("relation_action_joint", supervised),
                ("relation_action_joint_exact", exact),
                ("relation_action_joint_objectless", objectless),
                ("relation_action_joint_missing_objectful", missing),
            ):
                if mask.any():
                    outputs[name] = joint_correct[mask].float().detach()
            return outputs

                    true_logit_gain[dropout_valid].detach()
                )
                aux_outputs["object_present_action_margin_gain"] = (
                    action_margin_gain[dropout_valid].detach()
                )

            uselaptop_idx = self._action_index("Uselaptop")
            if uselaptop_idx is not None:
                uselaptop_mask = dropout_valid & (
                    dropout_labels == int(uselaptop_idx)
                )
                if uselaptop_mask.any():
                    self._log_scalar(
                        f"{aux_stage}_object_present_Uselaptop_prob_gain",
                        true_prob_gain[uselaptop_mask].mean(),
                        int(uselaptop_mask.sum().item()),
                    )
                    if stage == "val":
                        aux_outputs["object_present_Uselaptop_prob_gain"] = (
                            true_prob_gain[uselaptop_mask].detach()
                        )
                    confuser_indices = [
                        self._action_index(action_name)
                        for action_name in RELATION_ACTION_AUDIT_ACTIONS
                        if action_name != "Uselaptop"
                    ]
                    confuser_indices = [
                        int(index) for index in confuser_indices if index is not None
                    ]
                    if confuser_indices:
                        confuser_tensor = torch.tensor(
                            confuser_indices,
                            device=dropout_preds.device,
                            dtype=torch.long,
                        )
                        present_logits = present_preds.detach().float()
                        dropout_logits = dropout_preds.detach().float()
                        present_margin = present_logits[..., int(uselaptop_idx)] - (
                            present_logits.index_select(-1, confuser_tensor).amax(dim=-1)
                        )
                        dropout_margin = dropout_logits[..., int(uselaptop_idx)] - (
                            dropout_logits.index_select(-1, confuser_tensor).amax(dim=-1)
                        )
                        margin_gain = present_margin - dropout_margin
                        self._log_scalar(
                            f"{aux_stage}_object_present_Uselaptop_confuser_margin_gain",
                            margin_gain[uselaptop_mask].mean(),
                            int(uselaptop_mask.sum().item()),
                        )
                        if stage == "val":
                            aux_outputs[
                                "object_present_Uselaptop_confuser_margin_gain"
                            ] = margin_gain[uselaptop_mask].detach()

        aux_relation_loss = self._actor_object_relation_loss(
            aux_stage,
            actions,
            selected_actor_mask,
            masked_target,
        )
        self._log_relation_action_joint_metrics(
            aux_stage,
            dropout_preds,
            actions,
            selected_actor_mask,
            masked_target,
        )
        if stage == "val" and selected_count > 0:
            dropout_labels = actions.to(device=dropout_preds.device, dtype=torch.long)
            dropout_valid = selected_actor_mask.to(
                device=dropout_preds.device,
                dtype=torch.bool,
            )
            pred_labels = dropout_preds.argmax(dim=-1)
            action_correct = pred_labels == dropout_labels
            aux_outputs["object_dropout_action"] = (
                action_correct[dropout_valid].float().detach()
            )
            uselaptop_idx = self._action_index("Uselaptop")
            if uselaptop_idx is not None:
                uselaptop_mask = dropout_valid & (
                    dropout_labels == int(uselaptop_idx)
                )
                if uselaptop_mask.any():
                    aux_outputs["object_dropout_action_Uselaptop"] = (
                        action_correct[uselaptop_mask].float().detach()
                    )

            joint_info = self._relation_action_joint_info(
                dropout_preds,
                actions,
                selected_actor_mask,
                masked_target,
            )
            if joint_info is not None:
                missing = joint_info["missing_compatible"]
                joint_correct = joint_info["joint_correct"]
                if missing.any():
                    aux_outputs[
                        "object_dropout_relation_action_joint_missing_objectful"
                    ] = joint_correct[missing].float().detach()
                if uselaptop_idx is not None:
                    joint_uselaptop_mask = missing & (
                        joint_info["actions"] == int(uselaptop_idx)
                    )
                    if joint_uselaptop_mask.any():
                        aux_outputs[
                            "object_dropout_relation_action_joint_missing_Uselaptop"
                        ] = joint_correct[joint_uselaptop_mask].float().detach()

        if not stage.startswith("train"):
            return None, aux_outputs

        aux_losses = []
        if self.actor_object_detector_dropout_action_loss_weight > 0.0:
            aux_action_loss = self._action_loss(
                dropout_preds,
                actions,
                loss_fn,
                selected_actor_mask,
            )
            self.log(
                f"{aux_stage}_loss_action",
                aux_action_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )
            aux_losses.append(
                aux_action_loss
                * self.actor_object_detector_dropout_action_loss_weight
            )
        if (
            aux_relation_loss is not None
            and self.actor_object_detector_dropout_relation_loss_weight > 0.0
        ):
            aux_losses.append(
                aux_relation_loss
                * self.actor_object_detector_dropout_relation_loss_weight
            )
        if (
            present_margin_loss is not None
            and self.actor_object_present_margin_loss_weight > 0.0
        ):
            aux_losses.append(
                present_margin_loss * self.actor_object_present_margin_loss_weight
            )
        if not aux_losses:
            return None, aux_outputs
        aux_total = torch.stack(aux_losses).sum()
        self.log(
            f"{aux_stage}_loss",
            aux_total,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        return aux_total, aux_outputs

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
                    "No trainable parameters selected. Set --lr_head > 0 or "
                    "--lr_head_hm > 0."
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
                    "Nash-MTL expects two tasks [main_deploy, heatmap_aux], "
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
                    "train_nash_weight_heatmap_aux",
                    weights[1].detach(),
                    on_step=True,
                    on_epoch=True,
                    logger=True,
                    sync_dist=True,
                )
        else:
            loss.backward()

    def _actor_step(self, imgs, target, loss_fn, stage):
        is_train = stage.startswith("train")
        actions = target["actions"].long()
        boxes = target["boxes"].float()
        valid = target["valid"].bool()
        if not valid.any():
            raise ValueError(f"{stage} actor batch has no valid actor slots")

        object_inputs = self._object_inputs_from_target(target, imgs.device)
        object_inputs, dropped_target_mask = self._augment_object_inputs_for_training(
            object_inputs,
            target,
            stage,
        )
        data = self.model(
            imgs,
            boxes=boxes,
            valid=valid,
            action_labels=actions,
            **self._actor_model_object_inputs(object_inputs),
        )
        preds, hm_preds, presence_logits = self._unpack_model_data(data)
        pair_scores = getattr(self.model, "last_actor_object_pair_action_scores", None)
        pair_allowed = getattr(self.model, "last_actor_object_pair_action_allowed", None)
        if pair_scores is not None and pair_allowed is not None and pair_allowed.any():
            pair_scores = pair_scores.detach().float()
            pair_allowed = pair_allowed.to(device=pair_scores.device, dtype=torch.bool)
            count = int(pair_allowed.sum().item())
            self._log_scalar(
                f"{stage}_actor_object_pair_action_score_abs",
                pair_scores[pair_allowed].abs().mean(),
                count,
            )

        # Mask action loss for main-pass object dropouts to avoid confuser overfit
        action_ce_mask = valid & ~dropped_target_mask
        if action_ce_mask.any():
            loss_action = self._action_loss(preds, actions, loss_fn, action_ce_mask)
            self.log(
                f"{stage}_loss_action",
                loss_action,
                on_step=is_train,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )
        else:
            loss_action = preds.sum() * 0.0

        loss_main_task = loss_action
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
                on_step=is_train,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )
        loss_actor_object_relation = self._actor_object_relation_loss(
            stage,
            actions,
            valid,
            target,
        )
        if loss_actor_object_relation is not None:
            loss_main_task = loss_main_task + loss_actor_object_relation
        loss_actor_object_pair_action_margin = (
            self._actor_object_pair_action_margin_loss(
                stage,
                actions,
                valid,
                target,
            )
        )
        if loss_actor_object_pair_action_margin is not None:
            loss_main_task = loss_main_task + loss_actor_object_pair_action_margin
        relation_action_joint_outputs = self._log_relation_action_joint_metrics(
            stage,
            preds,
            actions,
            valid,
            target,
        )

        heatmap_aux_terms = []

        self._log_token_selection_diagnostics(stage, actions, valid, target)
        loss_kp = None
        loss_pose_frobenius = None
        loss_pose_heatmap_optimized = None
        if self.model.hparams.n_landmarks > 0:
            labels_kp = target["heatmap"]
            kp_vis = target["kp_vis"]
            pose_hm_preds = self._pose_heatmap_pred(hm_preds)
            if is_train and self.model.hparams.target_kp_loss_weight:
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
                    on_step=is_train,
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
                on_step=is_train,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )

            if (
                not self.actor_poguiseplus_loss
                and is_train
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
                    teacher_missing = known_object & ~heatmap_info["valid_from_one_based"]
                    teacher_mismatch = (
                        known_object
                        & heatmap_info["valid_from_one_based"]
                        & ~heatmap_info["compatible_from_one_based"]
                    )
                    exact_compatible = known_object & heatmap_info["compatible_from_one_based"]

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
                        on_step=is_train,
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
                        on_step=is_train,
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
                        on_step=is_train,
                        on_epoch=True,
                        prog_bar=False,
                        logger=True,
                        sync_dist=True,
                    )
                self.log(
                    f"{stage}_loss_interaction_heatmap",
                    loss_interaction_heatmap,
                    on_step=is_train,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                    sync_dist=True,
                )
                self.log(
                    f"{stage}_loss_interaction_heatmap_raw_frobenius",
                    loss_interaction_raw_frobenius,
                    on_step=is_train,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                    sync_dist=True,
                )
                self.log(
                    f"{stage}_loss_interaction_heatmap_frobenius",
                    loss_interaction_frobenius,
                    on_step=is_train,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                    sync_dist=True,
                )
                if not is_train:
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
                on_step=is_train,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )
            self.log(
                f"{stage}_loss_heatmap_optimized",
                loss_heatmap_raw,
                on_step=is_train,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )
            self.log(
                f"{stage}_loss_heatmap_log",
                loss_heatmap_task,
                on_step=is_train,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )
            heatmap_aux_terms.append(
                loss_heatmap_task * self.poguiseplus_heatmap_loss_weight
            )
            loss_aux_task = torch.stack(heatmap_aux_terms).sum()
            self.log(
                f"{stage}_loss_main_deploy",
                loss_main_task,
                on_step=is_train,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )
            self.log(
                f"{stage}_loss_heatmap_aux",
                loss_aux_task,
                on_step=is_train,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )
            if is_train and self.model.hparams.grad_weights:
                loss = torch.stack([loss_main_task, loss_aux_task])
            elif self.model.hparams.get("kp_only", False):
                loss = loss_main_task * 1e-6 + loss_aux_task
            else:
                loss = loss_main_task + loss_aux_task
        else:
            loss = loss_main_task
            if heatmap_aux_terms:
                loss = loss + torch.stack(heatmap_aux_terms).sum()
            if loss_kp is not None:
                kp_loss_weight = float(self.model.hparams.kp_loss_weight)
                if self.model.hparams.get("kp_only", False):
                    loss = loss * 1e-6 + loss_kp * kp_loss_weight
                elif is_train and self.model.hparams.grad_weights:
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
            relation_action_joint_outputs,
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
            **self._actor_model_object_inputs(object_inputs),
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
            **self._actor_model_object_inputs(object_inputs),
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

    def _scale_boxes_to_side_by_side_panel(self, boxes, side, split_frac):
        boxes = boxes.clone()
        if side == "left":
            boxes[..., [0, 2]] = boxes[..., [0, 2]] * split_frac
        elif side == "right":
            boxes[..., [0, 2]] = split_frac + boxes[..., [0, 2]] * (1.0 - split_frac)
        else:
            raise ValueError(f"unknown side-by-side panel: {side}")
        return boxes.clamp(0.0, 1.0)

    def _compose_half_width_maps(self, left, right):
        if left.shape != right.shape:
            raise RuntimeError(
                "side-by-side heatmap shape mismatch: "
                f"{tuple(left.shape)} vs {tuple(right.shape)}"
            )
        if left.ndim < 3:
            raise RuntimeError(
                "side-by-side heatmap tensors must have at least 3 dims, got "
                f"{tuple(left.shape)}"
            )
        height, width = int(left.shape[-2]), int(left.shape[-1])
        split = width // 2
        if split <= 0 or split >= width:
            raise RuntimeError(f"invalid side-by-side heatmap width: {width}")

        leading = left.shape[:-2]
        left_4d = left.reshape(-1, 1, height, width).float()
        right_4d = right.reshape(-1, 1, height, width).float()
        left_scaled = F.interpolate(
            left_4d,
            size=(height, split),
            mode="bilinear",
            align_corners=False,
        ).reshape(*leading, height, split)
        right_scaled = F.interpolate(
            right_4d,
            size=(height, width - split),
            mode="bilinear",
            align_corners=False,
        ).reshape(*leading, height, width - split)

        output = left.new_zeros(*leading, height, width)
        output[..., :split] = left_scaled.to(dtype=output.dtype)
        output[..., split:] = right_scaled.to(dtype=output.dtype)
        return output

    def _copy_pair_objects_for_side(
        self,
        pair_target,
        source,
        pair_idx,
        source_idx,
        source_actor_slot,
        target_actor_slot,
        object_start,
        object_capacity,
        side,
        split_frac,
    ):
        object_valid = source["object_valid"]
        object_boxes = source["object_boxes"]
        object_classes = source["object_classes"]
        object_confs = source["object_confs"]
        num_objects = int(object_valid.shape[1])
        if object_capacity <= 0 or num_objects <= 0:
            return

        source_idx = int(source_idx)
        source_actor_slot = int(source_actor_slot)
        pair_idx = int(pair_idx)
        target_actor_slot = int(target_actor_slot)

        teacher_valid = bool(
            source["interaction_object_index_valid"][source_idx, source_actor_slot].item()
        )
        teacher_one_based = int(
            source["interaction_object_index"][source_idx, source_actor_slot].item()
        )
        if teacher_valid and teacher_one_based == 0:
            pair_target["interaction_object_index"][pair_idx, target_actor_slot] = 0
            pair_target["interaction_object_index_valid"][pair_idx, target_actor_slot] = True

        teacher_zero_based = None
        if teacher_valid and teacher_one_based > 0:
            candidate = teacher_one_based - 1
            if 0 <= candidate < num_objects and bool(object_valid[source_idx, candidate]):
                teacher_zero_based = candidate

        keep = []
        if teacher_zero_based is not None:
            keep.append(teacher_zero_based)
        for old_idx in torch.nonzero(object_valid[source_idx], as_tuple=False).flatten():
            old_idx = int(old_idx.item())
            if old_idx not in keep:
                keep.append(old_idx)
            if len(keep) >= object_capacity:
                break

        for offset, old_idx in enumerate(keep[:object_capacity]):
            new_idx = object_start + offset
            pair_target["object_boxes"][pair_idx, new_idx] = (
                self._scale_boxes_to_side_by_side_panel(
                    object_boxes[source_idx, old_idx],
                    side,
                    split_frac,
                )
            )
            pair_target["object_classes"][pair_idx, new_idx] = object_classes[
                source_idx,
                old_idx,
            ]
            pair_target["object_confs"][pair_idx, new_idx] = object_confs[
                source_idx,
                old_idx,
            ]
            pair_target["object_valid"][pair_idx, new_idx] = True
            if teacher_zero_based is not None and old_idx == teacher_zero_based:
                pair_target["interaction_object_index"][pair_idx, target_actor_slot] = (
                    new_idx + 1
                )
                pair_target["interaction_object_index_valid"][
                    pair_idx,
                    target_actor_slot,
                ] = True

    def _compose_pair_training_batch(self, imgs, target):
        if self.actor_pair_train_weight <= 0:
            return None
        valid = target["valid"].to(device=imgs.device, dtype=torch.bool)
        batch_size, num_actor_tokens = valid.shape
        if num_actor_tokens < 2:
            raise ValueError("actor pair training requires num_actor_tokens >= 2")

        available = torch.nonzero(valid.any(dim=1), as_tuple=False).flatten()
        pair_count = int(available.numel() // 2)
        if pair_count <= 0:
            return None

        left_idx = available[:pair_count]
        right_idx = available[pair_count : pair_count * 2]
        left_slot = valid[left_idx].long().argmax(dim=1)
        right_slot = valid[right_idx].long().argmax(dim=1)

        _, num_frames, channels, height, width = imgs.shape
        split = width // 2
        split_frac = float(split) / float(width)
        left_frames = imgs[left_idx].reshape(
            pair_count * num_frames,
            channels,
            height,
            width,
        )
        right_frames = imgs[right_idx].reshape(
            pair_count * num_frames,
            channels,
            height,
            width,
        )
        left_panel = F.interpolate(
            left_frames,
            size=(height, split),
            mode="bilinear",
            align_corners=False,
        ).reshape(pair_count, num_frames, channels, height, split)
        right_panel = F.interpolate(
            right_frames,
            size=(height, width - split),
            mode="bilinear",
            align_corners=False,
        ).reshape(pair_count, num_frames, channels, height, width - split)
        pair_imgs = torch.zeros(
            pair_count,
            num_frames,
            channels,
            height,
            width,
            device=imgs.device,
            dtype=imgs.dtype,
        )
        pair_imgs[:, :, :, :, :split] = left_panel
        pair_imgs[:, :, :, :, split:] = right_panel

        source = {
            key: value.to(device=imgs.device)
            for key, value in target.items()
            if torch.is_tensor(value)
        }
        pair_target = {
            "actions": torch.full(
                (pair_count, num_actor_tokens),
                -100,
                device=imgs.device,
                dtype=torch.long,
            ),
            "boxes": torch.zeros(
                pair_count,
                num_actor_tokens,
                4,
                device=imgs.device,
                dtype=torch.float32,
            ),
            "valid": torch.zeros(
                pair_count,
                num_actor_tokens,
                device=imgs.device,
                dtype=torch.bool,
            ),
        }

        left_boxes = source["boxes"][left_idx, left_slot].float()
        right_boxes = source["boxes"][right_idx, right_slot].float()
        pair_target["boxes"][:, 0] = self._scale_boxes_to_side_by_side_panel(
            left_boxes,
            "left",
            split_frac,
        )
        pair_target["boxes"][:, 1] = self._scale_boxes_to_side_by_side_panel(
            right_boxes,
            "right",
            split_frac,
        )
        pair_target["actions"][:, 0] = source["actions"][left_idx, left_slot].long()
        pair_target["actions"][:, 1] = source["actions"][right_idx, right_slot].long()
        pair_target["valid"][:, :2] = True

        if "heatmap" in source and "kp_vis" in source:
            pair_target["heatmap"] = self._compose_half_width_maps(
                source["heatmap"][left_idx],
                source["heatmap"][right_idx],
            )
            pair_target["kp_vis"] = self._compose_half_width_maps(
                source["kp_vis"][left_idx],
                source["kp_vis"][right_idx],
            )

        if "interaction_cls" in source:
            pair_target["interaction_cls"] = torch.full(
                (pair_count, num_actor_tokens),
                NUM_OBJECT_CLASSES,
                device=imgs.device,
                dtype=torch.long,
            )
            pair_target["interaction_cls"][:, 0] = source["interaction_cls"][
                left_idx,
                left_slot,
            ].long()
            pair_target["interaction_cls"][:, 1] = source["interaction_cls"][
                right_idx,
                right_slot,
            ].long()
        if "interaction_valid" in source:
            pair_target["interaction_valid"] = torch.zeros(
                pair_count,
                num_actor_tokens,
                device=imgs.device,
                dtype=torch.bool,
            )
            pair_target["interaction_valid"][:, 0] = source["interaction_valid"][
                left_idx,
                left_slot,
            ].bool()
            pair_target["interaction_valid"][:, 1] = source["interaction_valid"][
                right_idx,
                right_slot,
            ].bool()
        if "interaction_heatmap" in source:
            _, _, hm_h, hm_w = source["interaction_heatmap"].shape
            pair_target["interaction_heatmap"] = torch.zeros(
                pair_count,
                num_actor_tokens,
                hm_h,
                hm_w,
                device=imgs.device,
                dtype=source["interaction_heatmap"].dtype,
            )
            left_hm = source["interaction_heatmap"][left_idx, left_slot]
            right_hm = source["interaction_heatmap"][right_idx, right_slot]
            pair_target["interaction_heatmap"][:, 0] = self._compose_half_width_maps(
                left_hm,
                torch.zeros_like(left_hm),
            )
            pair_target["interaction_heatmap"][:, 1] = self._compose_half_width_maps(
                torch.zeros_like(right_hm),
                right_hm,
            )
        for key in ("interaction_heatmap_valid", "interaction_heatmap_positive_valid"):
            if key not in source:
                continue
            pair_target[key] = torch.zeros(
                pair_count,
                num_actor_tokens,
                device=imgs.device,
                dtype=torch.bool,
            )
            pair_target[key][:, 0] = source[key][left_idx, left_slot].bool()
            pair_target[key][:, 1] = source[key][right_idx, right_slot].bool()

        if self.uses_object_proposals:
            required = (
                "object_boxes",
                "object_classes",
                "object_confs",
                "object_valid",
                "interaction_object_index",
                "interaction_object_index_valid",
            )
            missing = [key for key in required if key not in source]
            if missing:
                raise RuntimeError(
                    "actor pair training with object proposals requires "
                    f"{missing}"
                )
            num_objects = int(source["object_boxes"].shape[1])
            left_capacity = num_objects // 2
            right_capacity = num_objects - left_capacity
            none_id = int(
                self.model.hparams.get("num_object_classes", NUM_OBJECT_CLASSES)
            )
            pair_target["object_boxes"] = torch.zeros(
                pair_count,
                num_objects,
                4,
                device=imgs.device,
                dtype=torch.float32,
            )
            pair_target["object_classes"] = torch.full(
                (pair_count, num_objects),
                none_id,
                device=imgs.device,
                dtype=torch.long,
            )
            pair_target["object_confs"] = torch.zeros(
                pair_count,
                num_objects,
                device=imgs.device,
                dtype=torch.float32,
            )
            pair_target["object_valid"] = torch.zeros(
                pair_count,
                num_objects,
                device=imgs.device,
                dtype=torch.bool,
            )
            pair_target["interaction_object_index"] = torch.zeros(
                pair_count,
                num_actor_tokens,
                device=imgs.device,
                dtype=torch.long,
            )
            pair_target["interaction_object_index_valid"] = torch.zeros(
                pair_count,
                num_actor_tokens,
                device=imgs.device,
                dtype=torch.bool,
            )
            for pair_idx in range(pair_count):
                self._copy_pair_objects_for_side(
                    pair_target,
                    source,
                    pair_idx,
                    int(left_idx[pair_idx].item()),
                    int(left_slot[pair_idx].item()),
                    0,
                    0,
                    left_capacity,
                    "left",
                    split_frac,
                )
                self._copy_pair_objects_for_side(
                    pair_target,
                    source,
                    pair_idx,
                    int(right_idx[pair_idx].item()),
                    int(right_slot[pair_idx].item()),
                    1,
                    left_capacity,
                    right_capacity,
                    "right",
                    split_frac,
                )

        return pair_imgs, pair_target

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
                **self._actor_model_object_inputs(object_inputs),
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

        def values_gathered(name):
            values = self._flatten_gathered_validation_tensor(gathered_outputs, name)
            if values is None:
                return None
            values = values.to(device=labels.device, dtype=torch.float32)
            if values.numel() == 0:
                return None
            return values

        def mean_gathered(name):
            values = values_gathered(name)
            if values is None:
                return None
            return values.mean()

        def median_value(values):
            if values is None:
                return None
            return values.median()

        def negative_rate(values):
            if values is None:
                return None
            return (values < 0).float().mean()

        relation_action_joint_acc = mean_gathered("relation_action_joint")
        relation_action_joint_exact_acc = mean_gathered("relation_action_joint_exact")
        relation_action_joint_objectless_acc = mean_gathered(
            "relation_action_joint_objectless"
        )
        relation_action_joint_missing_acc = mean_gathered(
            "relation_action_joint_missing_objectful"
        )
        relation_action_joint_components = [
            value
            for value in (
                relation_action_joint_exact_acc,
                relation_action_joint_objectless_acc,
                relation_action_joint_missing_acc,
            )
            if value is not None
        ]
        relation_action_joint_balanced_acc = (
            torch.stack(relation_action_joint_components).mean()
            if relation_action_joint_components
            else None
        )
        object_dropout_action_acc = mean_gathered("object_dropout_action")
        object_dropout_uselaptop_acc = mean_gathered(
            "object_dropout_action_Uselaptop"
        )
        object_dropout_joint_missing_acc = mean_gathered(
            "object_dropout_relation_action_joint_missing_objectful"
        )
        object_dropout_joint_uselaptop_acc = mean_gathered(
            "object_dropout_relation_action_joint_missing_Uselaptop"
        )
        object_present_true_gain_values = values_gathered(
            "object_present_true_prob_gain"
        )
        object_present_true_logit_gain_values = values_gathered(
            "object_present_true_logit_gain"
        )
        object_present_action_margin_gain_values = values_gathered(
            "object_present_action_margin_gain"
        )
        object_present_uselaptop_gain_values = values_gathered(
            "object_present_Uselaptop_prob_gain"
        )
        object_present_uselaptop_margin_gain_values = values_gathered(
            "object_present_Uselaptop_confuser_margin_gain"
        )
        object_present_true_gain = (
            object_present_true_gain_values.mean()
            if object_present_true_gain_values is not None
            else None
        )
        object_present_true_logit_gain = (
            object_present_true_logit_gain_values.mean()
            if object_present_true_logit_gain_values is not None
            else None
        )
        object_present_action_margin_gain = (
            object_present_action_margin_gain_values.mean()
            if object_present_action_margin_gain_values is not None
            else None
        )
        object_present_uselaptop_gain = (
            object_present_uselaptop_gain_values.mean()
            if object_present_uselaptop_gain_values is not None
            else None
        )
        object_present_uselaptop_margin_gain = (
            object_present_uselaptop_margin_gain_values.mean()
            if object_present_uselaptop_margin_gain_values is not None
            else None
        )
        object_present_true_gain_median = median_value(
            object_present_true_gain_values
        )
        object_present_action_margin_gain_median = median_value(
            object_present_action_margin_gain_values
        )
        object_present_uselaptop_gain_median = median_value(
            object_present_uselaptop_gain_values
        )
        object_present_true_gain_negative_rate = negative_rate(
            object_present_true_gain_values
        )
        object_present_action_margin_gain_negative_rate = negative_rate(
            object_present_action_margin_gain_values
        )
        object_present_uselaptop_gain_negative_rate = negative_rate(
            object_present_uselaptop_gain_values
        )
        object_present_true_hurt = (
            (-object_present_true_gain).clamp_min(0.0)
            if object_present_true_gain is not None
            else None
        )
        object_present_uselaptop_hurt = (
            (-object_present_uselaptop_gain).clamp_min(0.0)
            if object_present_uselaptop_gain is not None
            else None
        )
        object_present_margin_hurt = (
            (-object_present_action_margin_gain).clamp_min(0.0)
            if object_present_action_margin_gain is not None
            else None
        )
        object_present_true_median_hurt = (
            (-object_present_true_gain_median).clamp_min(0.0)
            if object_present_true_gain_median is not None
            else None
        )
        object_present_margin_median_hurt = (
            (-object_present_action_margin_gain_median).clamp_min(0.0)
            if object_present_action_margin_gain_median is not None
            else None
        )
        object_present_uselaptop_median_hurt = (
            (-object_present_uselaptop_gain_median).clamp_min(0.0)
            if object_present_uselaptop_gain_median is not None
            else None
        )

        key_action_values = self._deploy_action_accuracies(
            pred_labels,
            labels,
            DEPLOY_KEY_ACTIONS,
        )
        key_action_mean = (
            torch.stack(key_action_values).mean() if key_action_values else None
        )
        key_action_min = (
            torch.stack(key_action_values).amin() if key_action_values else None
        )
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
                (relation_action_joint_balanced_acc, 0.15),
                (object_dropout_action_acc, 0.10),
                (object_dropout_joint_missing_acc, 0.10),
                (object_dropout_uselaptop_acc, 0.05),
                (object_dropout_joint_uselaptop_acc, 0.05),
            ],
            penalties=[
                (hard_object_action_rate, 0.20),
                (key_action_floor_deficit, 0.15),
                (object_present_true_hurt, 0.50),
                (object_present_true_median_hurt, 0.75),
                (object_present_true_gain_negative_rate, 0.25),
                (object_present_margin_hurt, 0.75),
                (object_present_margin_median_hurt, 0.75),
                (object_present_action_margin_gain_negative_rate, 0.35),
                (object_present_uselaptop_hurt, 1.00),
                (object_present_uselaptop_median_hurt, 1.00),
                (object_present_uselaptop_gain_negative_rate, 0.50),
            ],
        )
        if deploy_score is None:
            return

        for name, value in (
            ("val_deploy_score", deploy_score),
            ("val_deploy_key_action_mean", key_action_mean),
            ("val_deploy_key_action_min", key_action_min),
            (
                "val_relation_action_joint_balanced_acc",
                relation_action_joint_balanced_acc,
            ),
            (
                "val_deploy_relation_action_joint_acc",
                relation_action_joint_acc,
            ),
            (
                "val_deploy_relation_action_joint_exact_acc",
                relation_action_joint_exact_acc,
            ),
            (
                "val_deploy_object_dropout_action_acc",
                object_dropout_action_acc,
            ),
            (
                "val_deploy_object_dropout_Uselaptop_acc",
                object_dropout_uselaptop_acc,
            ),
            (
                "val_deploy_object_dropout_joint_missing_acc",
                object_dropout_joint_missing_acc,
            ),
            (
                "val_deploy_object_dropout_joint_Uselaptop_acc",
                object_dropout_joint_uselaptop_acc,
            ),
            (
                "val_deploy_object_present_true_prob_gain",
                object_present_true_gain,
            ),
            (
                "val_deploy_object_present_true_prob_gain_median",
                object_present_true_gain_median,
            ),
            (
                "val_deploy_object_present_true_prob_gain_negative_rate",
                object_present_true_gain_negative_rate,
            ),
            (
                "val_deploy_object_present_true_logit_gain",
                object_present_true_logit_gain,
            ),
            (
                "val_deploy_object_present_action_margin_gain",
                object_present_action_margin_gain,
            ),
            (
                "val_deploy_object_present_action_margin_gain_median",
                object_present_action_margin_gain_median,
            ),
            (
                "val_deploy_object_present_action_margin_gain_negative_rate",
                object_present_action_margin_gain_negative_rate,
            ),
            (
                "val_deploy_object_present_Uselaptop_prob_gain",
                object_present_uselaptop_gain,
            ),
            (
                "val_deploy_object_present_Uselaptop_prob_gain_median",
                object_present_uselaptop_gain_median,
            ),
            (
                "val_deploy_object_present_Uselaptop_prob_gain_negative_rate",
                object_present_uselaptop_gain_negative_rate,
            ),
            (
                "val_deploy_object_present_Uselaptop_confuser_margin_gain",
                object_present_uselaptop_margin_gain,
            ),
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
            loss, _, _, _, loss_kp, _, _, _ = self._actor_step(
                imgs, target, self.train_loss, "train"
            )
            pair_batch = self._compose_pair_training_batch(imgs, target)
            if pair_batch is not None:
                pair_imgs, pair_target = pair_batch
                pair_loss, _, _, _, pair_loss_kp, _, _, _ = self._actor_step(
                    pair_imgs,
                    pair_target,
                    self.train_loss,
                    "train_pair",
                )
                loss = loss + pair_loss * self.actor_pair_train_weight
                pair_loss_to_log = pair_loss.sum() if pair_loss.ndim > 0 else pair_loss
                self.log(
                    "train_pair_loss",
                    pair_loss_to_log,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                    sync_dist=True,
                )
                self.log(
                    "train_pair_batch_size",
                    torch.tensor(
                        pair_imgs.shape[0],
                        device=imgs.device,
                        dtype=torch.float32,
                    ),
                    on_step=True,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                    sync_dist=True,
                )
                if pair_loss_kp is not None:
                    self.log(
                        "train_pair_loss_kp",
                        pair_loss_kp,
                        on_step=True,
                        on_epoch=True,
                        prog_bar=False,
                        logger=True,
                        sync_dist=True,
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
                relation_action_joint_outputs,
            ) = self._actor_step(
                imgs, target, self.val_loss, "val"
            )
            for name, tensor in relation_action_joint_outputs.items():
                self.validation_step_outputs[name].append(tensor)
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
                **self._actor_model_object_inputs(object_inputs),
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
