import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from argparse import ArgumentParser
from blocks.poguise import vit_base_patch16_224
from blocks.poguise import vit_small_patch16_224
from blocks.poguise import vit_large_patch16_224
import torch
import os
import os.path
import pickle

from collections import OrderedDict
import json
from datasets.object_vocab import OBJECT_SPECIALIST_GROUPS, OBJECT_TO_ID


TOYOTA_CS_ACTION_TO_INDEX = {
    "Cook.Cleandishes": 0,
    "Cook.Cleanup": 1,
    "Cook.Cut": 2,
    "Cook.Stir": 3,
    "Cook.Usestove": 4,
    "Cutbread": 5,
    "Drink.Frombottle": 6,
    "Drink.Fromcan": 7,
    "Drink.Fromcup": 8,
    "Drink.Fromglass": 9,
    "Eat.Attable": 10,
    "Eat.Snack": 11,
    "Enter": 12,
    "Getup": 13,
    "Laydown": 14,
    "Leave": 15,
    "Makecoffee.Pourgrains": 16,
    "Makecoffee.Pourwater": 17,
    "Maketea.Boilwater": 18,
    "Maketea.Insertteabag": 19,
    "Pour.Frombottle": 20,
    "Pour.Fromcan": 21,
    "Pour.Fromkettle": 22,
    "Readbook": 23,
    "Sitdown": 24,
    "Takepills": 25,
    "Uselaptop": 26,
    "Usetablet": 27,
    "Usetelephone": 28,
    "Walk": 29,
    "WatchTV": 30,
}

OBJECT_RESIDUAL_ACTION_MASKS = {
    "none": [],
    "strong": [
        "Drink.Frombottle",
        "Drink.Fromcup",
        "Drink.Fromglass",
        "Readbook",
        "Uselaptop",
        "Usetelephone",
    ],
    "strong_plus_watchtv": [
        "Drink.Frombottle",
        "Drink.Fromcup",
        "Drink.Fromglass",
        "Readbook",
        "Uselaptop",
        "Usetelephone",
        "WatchTV",
    ],
}


def load_state_dict(
    model, state_dict, prefix="", ignore_missing="relative_position_index"
):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=""):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            True,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + ".")

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split("|"):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print(
            "Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys
            )
        )
    if len(unexpected_keys) > 0:
        print(
            "Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys
            )
        )
    if len(ignore_missing_keys) > 0:
        print(
            "Ignored weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, ignore_missing_keys
            )
        )
    if len(error_msgs) > 0:
        print("\n".join(error_msgs))


class Classifier(nn.Module):
    "classifier head"

    def __init__(
        self,
        num_classes,
        feat_dim,
        l2_norm=False,
        use_bn=False,
        use_dropout=False,
        dropout=0.5,
    ):
        super(Classifier, self).__init__()
        self.use_bn = use_bn
        self.l2_norm = l2_norm
        self.use_dropout = use_dropout
        if use_bn:
            self.bn = nn.BatchNorm1d(feat_dim)
            self.bn.weight.data.fill_(1)
            self.bn.bias.data.zero_()
        if use_dropout:
            self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(feat_dim, num_classes)
        self._initialize_weights(self.classifier)

    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0.0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1)

    def forward(self, x, mode="train"):
        # save the feature before the classifier
        filename = "tmp/x_{}_{}.pickle"
        # create the directory if it does not exist
        if not os.path.exists("tmp"):
            os.makedirs("tmp")
        # check if the file exists

        i = 0
        while os.path.isfile(filename.format(i, mode)):
            i += 1
        with open(filename.format(i, mode), "wb") as f:
            pickle.dump(x, f)

        x = x.squeeze()
        # x = x.view(x.shape[0], -1)
        if self.l2_norm:
            x = nn.functional.normalize(x, p=2, dim=-1)
        if self.use_bn:
            x = self.bn(x)
        if self.use_dropout:
            x = self.dropout(x)
        return self.classifier(x)


class POGUISE(pl.LightningModule):
    def __init__(self, net_size="t", pretrained=None, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.mode = self.hparams.get("mode", "train")
        self.actor_prompt = bool(self.hparams.get("actor_prompt", 0))
        self.object_prompt = bool(self.hparams.get("object_prompt", 0))
        if self.object_prompt and not self.actor_prompt:
            raise ValueError("object_prompt requires actor_prompt")
        if self.hparams.get("object_relation_only", 0):
            if not self.object_prompt:
                raise ValueError("object_relation_only requires object_prompt")
            if not self.hparams.get("freeze_backbone", 0):
                raise ValueError("object_relation_only requires freeze_backbone")
        if self.hparams.get("object_warmup_freeze_actor_path", 0):
            if not (self.actor_prompt and self.object_prompt):
                raise ValueError(
                    "object_warmup_freeze_actor_path requires actor_prompt and object_prompt"
                )
            if not self.hparams.get("freeze_backbone", 0):
                raise ValueError(
                    "object_warmup_freeze_actor_path requires freeze_backbone"
                )
        for prob_name in ("object_dropout_prob", "object_token_dropout_prob"):
            prob_value = float(self.hparams.get(prob_name, 0.0))
            if not 0 <= prob_value <= 1:
                raise ValueError(f"{prob_name} must be in [0, 1]")
        self.use_register_tokens = bool(self.hparams.get("use_register_tokens", 0))
        self._create_network()
        # freeze backbone if specified
        if self.hparams.freeze_backbone:
            self._freeze_backbone()

    def _create_network(self):
        n_registers = (
            int(self.hparams.get("n_registers", 0) or 0)
            if self.use_register_tokens
            else 0
        )
        if self.hparams.pretrained == "small":
            self.net = vit_small_patch16_224(
                drop_rate=self.hparams.drop_rate,
                attn_drop_rate=self.hparams.attn_drop_rate,
                drop_path_rate=self.hparams.drop_path_rate,
                head_drop_rate=self.hparams.head_drop_rate,
                keep_rate=self.hparams.keep_rate,
                enhanced_weight_class=self.hparams.enhanced_weight_class,
                enhanced_weight_heatmap=self.hparams.enhanced_weight_heatmap,
                n_landmarks=self.hparams.n_landmarks,
                sim_metric=self.hparams.sim_metric,
                topk_type=self.hparams.topk_type,
                merge_mode=self.hparams.merge_mode,
                keep_rate_merge=self.hparams.keep_rate_merge,
                merge_type=self.hparams.merge_type,
                mode=self.mode,
                hw_out_conv=self.hparams.hw_out_conv,
                n_registers=n_registers,
                actor_prompt=self.actor_prompt,
                num_actor_tokens=self.hparams.get("num_actor_tokens", 8),
                actor_bbox_prior_weight=self.hparams.get(
                    "actor_bbox_prior_weight", 0.1
                ),
                actor_bbox_prior_expand=self.hparams.get(
                    "actor_bbox_prior_expand", 1.75
                ),
                object_prompt=self.object_prompt,
                num_object_tokens=self.hparams.get("num_object_tokens", 24),
                num_object_classes=self.hparams.get("num_object_classes", 0),
                object_bbox_prior_weight=self.hparams.get(
                    "object_bbox_prior_weight", 0.0
                ),
                object_bbox_prior_expand=self.hparams.get(
                    "object_bbox_prior_expand", 1.25
                ),
                return_heatmap_features=self.object_prompt,
            )
        else:
            self.net = vit_base_patch16_224(
                drop_rate=self.hparams.drop_rate,
                attn_drop_rate=self.hparams.attn_drop_rate,
                drop_path_rate=self.hparams.drop_path_rate,
                head_drop_rate=self.hparams.head_drop_rate,
                keep_rate=self.hparams.keep_rate,
                enhanced_weight_class=self.hparams.enhanced_weight_class,
                enhanced_weight_heatmap=self.hparams.enhanced_weight_heatmap,
                n_landmarks=self.hparams.n_landmarks,
                sim_metric=self.hparams.sim_metric,
                topk_type=self.hparams.topk_type,
                merge_mode=self.hparams.merge_mode,
                keep_rate_merge=self.hparams.keep_rate_merge,
                merge_type=self.hparams.merge_type,
                mode=self.mode,
                hw_out_conv=self.hparams.hw_out_conv,
                n_registers=n_registers,
                actor_prompt=self.actor_prompt,
                num_actor_tokens=self.hparams.get("num_actor_tokens", 8),
                actor_bbox_prior_weight=self.hparams.get(
                    "actor_bbox_prior_weight", 0.1
                ),
                actor_bbox_prior_expand=self.hparams.get(
                    "actor_bbox_prior_expand", 1.75
                ),
                object_prompt=self.object_prompt,
                num_object_tokens=self.hparams.get("num_object_tokens", 24),
                num_object_classes=self.hparams.get("num_object_classes", 0),
                object_bbox_prior_weight=self.hparams.get(
                    "object_bbox_prior_weight", 0.0
                ),
                object_bbox_prior_expand=self.hparams.get(
                    "object_bbox_prior_expand", 1.25
                ),
                return_heatmap_features=self.object_prompt,
            )
        if self.hparams.pretrained == "DEFAULT":
            if os.path.exists("vit_b_k710_dl_from_giant.pth"):
                print("loading from disk")
                state_dict = torch.load("vit_b_k710_dl_from_giant.pth")
            else:
                state_dict = torch.hub.load_state_dict_from_url(
                    "https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/internvideo/distill/vit_b_k710_dl_from_giant.pth",
                )
            load_state_dict(self.net, state_dict["module"])
        elif self.hparams.pretrained == "small":
            if os.path.exists("vit_s_k710_dl_from_giant.pth"):
                print("loading from disk")
                state_dict = torch.load("vit_s_k710_dl_from_giant.pth")
            else:
                state_dict = torch.hub.load_state_dict_from_url(
                    "https://huggingface.co/OpenGVLab/VideoMAE2/resolve/main/distill/vit_s_k710_dl_from_giant.pth"
                )
            load_state_dict(self.net, state_dict["module"])

        # Mapping to classification output
        self.net.head = nn.Identity(self.net.num_features, self.net.num_features)
        self.head = nn.Linear(self.net.num_features, self.hparams.num_classes)
        if self.actor_prompt:
            self.actor_head = nn.Linear(self.net.num_features, self.hparams.num_classes)
            self.presence_head = (
                nn.Linear(self.net.num_features, 1)
                if self.hparams.get("actor_presence_head", 0)
                else None
            )
            if self.object_prompt:
                self._create_object_prompt_modules()
        if self.hparams.get("linear_probe", 0):
            self._freeze_backbone()
            self.head = Classifier(
                self.hparams.num_classes,
                self.net.num_features,
                l2_norm=False,
                use_bn=False,
                use_dropout=False,
                # dropout=0.5,
            )

    def _freeze_backbone(self):
        print("Freezing backbone")
        if (
            self.actor_prompt
            and self.object_prompt
            and self.hparams.get("object_relation_only", 0)
        ):
            for param in self.parameters():
                param.requires_grad = False
            for module in self._object_relation_modules():
                for param in module.parameters():
                    param.requires_grad = True
            self.object_relation_gate_logit.requires_grad = True
            return

        # Freeze the backbone
        for param in self.net.parameters():
            param.requires_grad = False
        if (
            self.object_prompt
            and float(self.hparams.get("lr_head_hm", 0.0)) > 0.0
            and hasattr(self.net, "heatmap_head")
        ):
            for param in self.net.heatmap_head.parameters():
                param.requires_grad = True
        object_warmup_freeze_actor_path = (
            self.actor_prompt
            and self.object_prompt
            and bool(self.hparams.get("object_warmup_freeze_actor_path", 0))
        )
        if object_warmup_freeze_actor_path:
            print(
                "Object warmup: freezing base actor path; training object relation "
                "modules and heatmap head only."
            )
        # Unfreeze the head
        if not object_warmup_freeze_actor_path:
            for param in self.head.parameters():
                param.requires_grad = True
        if self.actor_prompt:
            if not object_warmup_freeze_actor_path:
                for param in self.actor_head.parameters():
                    param.requires_grad = True
                if hasattr(self.net, "actor_token"):
                    self.net.actor_token.requires_grad = True
                if hasattr(self.net, "actor_slot_embed"):
                    self.net.actor_slot_embed.requires_grad = True
                if hasattr(self.net, "valid_embed"):
                    for param in self.net.valid_embed.parameters():
                        param.requires_grad = True
                if hasattr(self.net, "bbox_mlp"):
                    for param in self.net.bbox_mlp.parameters():
                        param.requires_grad = True
                if self.presence_head is not None:
                    for param in self.presence_head.parameters():
                        param.requires_grad = True
            if self.object_prompt:
                for module in self._object_relation_modules():
                    for param in module.parameters():
                        param.requires_grad = True
                self.object_relation_gate_logit.requires_grad = True

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def _create_object_prompt_modules(self):
        num_object_classes = int(self.hparams.get("num_object_classes", 0))
        if num_object_classes <= 0:
            raise ValueError("object_prompt requires num_object_classes > 0")
        dim = self.net.num_features
        self.num_object_classes = num_object_classes
        num_classes = int(self.hparams.num_classes)
        hidden_dim = int(self.hparams.get("object_relation_hidden_dim", 512))
        if hidden_dim <= 0:
            raise ValueError("object_relation_hidden_dim must be positive")
        dropout = float(self.hparams.get("object_relation_dropout", 0.1))
        if not 0 <= dropout < 1:
            raise ValueError("object_relation_dropout must be in [0, 1)")
        action_gate_init = float(self.hparams.get("object_action_gate_init", 0.05))
        if not 0 < action_gate_init < 1:
            raise ValueError("object_action_gate_init must be in (0, 1)")
        self.object_delta_scale = float(self.hparams.get("object_delta_scale", 1.0))
        if self.object_delta_scale <= 0:
            raise ValueError("object_delta_scale must be positive")
        self.object_specialist_heads_enabled = bool(
            self.hparams.get("object_specialist_heads", 0)
        )
        residual_action_mask = str(
            self.hparams.get("object_residual_action_mask", "none")
        )
        if residual_action_mask not in OBJECT_RESIDUAL_ACTION_MASKS:
            raise ValueError(
                "object_residual_action_mask must be one of "
                + ", ".join(sorted(OBJECT_RESIDUAL_ACTION_MASKS))
            )
        if residual_action_mask != "none" and int(self.hparams.num_classes) != 31:
            raise ValueError(
                "object_residual_action_mask currently supports Toyota CS 31 classes"
            )
        residual_mask = torch.ones(num_classes, dtype=torch.float32)
        masked_actions = OBJECT_RESIDUAL_ACTION_MASKS[residual_action_mask]
        if masked_actions:
            residual_mask.zero_()
            residual_mask[
                [TOYOTA_CS_ACTION_TO_INDEX[action] for action in masked_actions]
            ] = 1.0
            print(
                "Object residual action mask: "
                + residual_action_mask
                + " -> "
                + ", ".join(masked_actions)
            )
        self.register_buffer(
            "object_residual_action_mask",
            residual_mask,
            persistent=False,
        )
        interaction_heatmap_size = int(self.hparams.get("object_heatmap_size", 56))
        if interaction_heatmap_size != 56:
            raise ValueError("object interaction heatmaps are trained at 56x56")
        self.interaction_heatmap_size = (
            interaction_heatmap_size,
            interaction_heatmap_size,
        )

        self.object_cls_embed = nn.Embedding(num_object_classes + 1, dim)
        self.object_bbox_mlp = nn.Sequential(
            nn.Linear(4, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.object_conf_mlp = nn.Sequential(
            nn.Linear(1, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.object_visual_mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.object_action_embed = nn.Embedding(num_classes, dim)
        self.object_action_query = nn.Sequential(
            nn.LayerNorm(dim * 3),
            nn.Linear(dim * 3, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.object_action_key = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
        )
        self.object_action_value = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
        )
        self.object_action_geom_bias = nn.Sequential(
            nn.LayerNorm(10),
            nn.Linear(10, max(32, hidden_dim // 4)),
            nn.GELU(),
            nn.Linear(max(32, hidden_dim // 4), 1),
        )
        self.object_action_delta_norm = nn.LayerNorm(dim * 6)
        self.object_action_delta = nn.Sequential(
            nn.Linear(dim * 6, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.object_valid_embed = nn.Embedding(2, dim)
        self.object_relation_gate_logit = nn.Parameter(
            torch.full(
                (num_classes,),
                torch.logit(torch.tensor(action_gate_init, dtype=torch.float32)),
            )
        )
        if self.object_specialist_heads_enabled:
            self._create_object_specialist_modules(dim, hidden_dim, dropout)

        nn.init.normal_(self.object_cls_embed.weight, std=0.02)
        nn.init.normal_(self.object_valid_embed.weight, std=0.02)
        nn.init.normal_(self.object_action_embed.weight, std=0.02)
        nn.init.zeros_(self.object_action_delta[-1].weight)
        nn.init.zeros_(self.object_action_delta[-1].bias)

    def _object_specialist_specs(self):
        if int(self.hparams.num_classes) != 31:
            raise ValueError("object_specialist_heads currently requires Toyota CS 31 classes")
        task_type = str(self.hparams.get("task_type", "CS"))
        if task_type != "CS":
            raise ValueError("object_specialist_heads currently supports task_type=CS only")

        specs = []
        for group_name, group in OBJECT_SPECIALIST_GROUPS.items():
            action_indices = [
                TOYOTA_CS_ACTION_TO_INDEX[action_name]
                for action_name in group["actions"]
            ]
            object_indices = [OBJECT_TO_ID[name] for name in group["objects"]]
            if any(idx >= self.num_object_classes for idx in object_indices):
                raise ValueError(
                    f"Specialist group {group_name} references object ids outside "
                    f"num_object_classes={self.num_object_classes}: {object_indices}"
                )
            specs.append((group_name, action_indices, object_indices))
        return specs

    def _create_object_specialist_modules(self, dim, hidden_dim, dropout):
        self.object_specialist_specs = self._object_specialist_specs()
        self.object_specialist_base_mlp = nn.Sequential(
            nn.Linear(1, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.object_specialist_delta_norms = nn.ModuleDict()
        self.object_specialist_delta_heads = nn.ModuleDict()
        for group_name, action_indices, object_indices in self.object_specialist_specs:
            self.register_buffer(
                f"object_specialist_{group_name}_actions",
                torch.tensor(action_indices, dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                f"object_specialist_{group_name}_objects",
                torch.tensor(object_indices, dtype=torch.long),
                persistent=False,
            )
            self.object_specialist_delta_norms[group_name] = nn.LayerNorm(dim * 7)
            self.object_specialist_delta_heads[group_name] = nn.Sequential(
                nn.Linear(dim * 7, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
            nn.init.zeros_(self.object_specialist_delta_heads[group_name][-1].weight)
            nn.init.zeros_(self.object_specialist_delta_heads[group_name][-1].bias)

    def _object_relation_modules(self):
        modules = [
            self.object_cls_embed,
            self.object_bbox_mlp,
            self.object_conf_mlp,
            self.object_visual_mlp,
            self.object_valid_embed,
            self.object_action_embed,
            self.object_action_query,
            self.object_action_key,
            self.object_action_value,
            self.object_action_geom_bias,
            self.object_action_delta_norm,
            self.object_action_delta,
        ]
        if self.object_specialist_heads_enabled:
            modules.extend(
                [
                    self.object_specialist_base_mlp,
                    self.object_specialist_delta_norms,
                    self.object_specialist_delta_heads,
                ]
            )
        return tuple(modules)

    def _apply_object_dropout(self, object_valid):
        if object_valid is None or not self.training:
            return object_valid
        object_valid = object_valid.clone()
        object_dropout_prob = float(self.hparams.get("object_dropout_prob", 0.0))
        object_token_dropout_prob = float(
            self.hparams.get("object_token_dropout_prob", 0.0)
        )
        if object_dropout_prob > 0:
            if torch.rand((), device=object_valid.device) < object_dropout_prob:
                object_valid.fill_(False)
        if object_token_dropout_prob > 0:
            drop_each = (
                torch.rand(object_valid.shape, device=object_valid.device)
                < object_token_dropout_prob
            )
            object_valid = object_valid & ~drop_each
        return object_valid

    def _object_relation_geometry(self, actor_boxes, object_boxes):
        batch_size, num_objects, _ = object_boxes.shape
        if actor_boxes is None:
            num_actors = int(self.hparams.get("num_actor_tokens", 0))
            return object_boxes.new_zeros((batch_size, num_actors, num_objects, 10))

        actor_boxes = actor_boxes.to(
            device=object_boxes.device,
            dtype=object_boxes.dtype,
        )
        actor_boxes = actor_boxes.clamp(0.0, 1.0)
        object_boxes = object_boxes.clamp(0.0, 1.0)

        actor_center = (actor_boxes[..., :2] + actor_boxes[..., 2:]) * 0.5
        actor_size = (actor_boxes[..., 2:] - actor_boxes[..., :2]).clamp_min(1e-4)
        object_center = (object_boxes[..., :2] + object_boxes[..., 2:]) * 0.5
        object_size = (object_boxes[..., 2:] - object_boxes[..., :2]).clamp_min(1e-4)

        actor_center_pair = actor_center[:, :, None, :]
        actor_size_pair = actor_size[:, :, None, :]
        object_center_pair = object_center[:, None, :, :]
        object_size_pair = object_size[:, None, :, :]
        return torch.cat(
            [
                object_center_pair - actor_center_pair,
                torch.log(object_size_pair / actor_size_pair),
                object_center_pair.expand(-1, actor_boxes.shape[1], -1, -1),
                object_size_pair.expand(-1, actor_boxes.shape[1], -1, -1),
                actor_center_pair.expand(-1, -1, num_objects, -1),
            ],
            dim=-1,
        )

    def _pool_boxes_from_feature_map(self, feature_map, boxes, valid):
        batch_size, num_objects, _ = boxes.shape
        dim = self.net.num_features
        if feature_map is None:
            return boxes.new_zeros((batch_size, num_objects, dim))

        feature_map = feature_map.to(device=boxes.device, dtype=boxes.dtype)
        _, channels, _, _ = feature_map.shape
        boxes = boxes.clamp(0.0, 1.0)
        valid = valid.to(device=boxes.device, dtype=torch.bool)
        offsets = torch.tensor(
            [1.0 / 6.0, 0.5, 5.0 / 6.0],
            device=boxes.device,
            dtype=boxes.dtype,
        )
        pooled = []
        for slot in range(num_objects):
            slot_boxes = boxes[:, slot]
            xs = slot_boxes[:, 0:1] + offsets[None, :] * (
                slot_boxes[:, 2:3] - slot_boxes[:, 0:1]
            ).clamp_min(1e-4)
            ys = slot_boxes[:, 1:2] + offsets[None, :] * (
                slot_boxes[:, 3:4] - slot_boxes[:, 1:2]
            ).clamp_min(1e-4)
            grid_y, grid_x = torch.meshgrid(
                torch.arange(3, device=boxes.device),
                torch.arange(3, device=boxes.device),
                indexing="ij",
            )
            sample_x = xs[:, grid_x.reshape(-1)]
            sample_y = ys[:, grid_y.reshape(-1)]
            grid = torch.stack(
                [sample_x * 2.0 - 1.0, sample_y * 2.0 - 1.0],
                dim=-1,
            ).reshape(batch_size, 3, 3, 2)
            sampled = F.grid_sample(
                feature_map,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            pooled.append(sampled.mean(dim=(2, 3)))

        pooled = torch.stack(pooled, dim=1)
        pooled = pooled * valid[:, :, None].to(dtype=pooled.dtype)
        if channels != dim:
            raise ValueError(
                f"Feature map channel count {channels} does not match model dim {dim}"
            )
        return pooled

    def _build_interaction_heatmap(self, object_alpha, object_boxes, object_valid, size):
        height, width = [int(v) for v in size]
        if height <= 0 or width <= 0:
            raise ValueError(f"Invalid interaction heatmap size: {size}")

        object_boxes = object_boxes.clamp(0.0, 1.0)
        object_valid = object_valid.to(device=object_boxes.device, dtype=torch.bool)
        real_alpha = object_alpha[..., : object_boxes.shape[1]]
        if real_alpha.shape[0] != object_valid.shape[0]:
            raise ValueError("object_alpha and object_valid batch dimensions differ")
        if real_alpha.shape[-1] != object_valid.shape[-1]:
            raise ValueError("object_alpha and object_valid object dimensions differ")
        valid_shape = (
            object_valid.shape[0],
            *([1] * (real_alpha.ndim - 2)),
            object_valid.shape[1],
        )
        real_alpha = real_alpha * object_valid.reshape(valid_shape).to(
            dtype=real_alpha.dtype
        )

        y = (
            torch.arange(height, device=object_boxes.device, dtype=object_boxes.dtype)
            + 0.5
        ) / float(height)
        x = (
            torch.arange(width, device=object_boxes.device, dtype=object_boxes.dtype)
            + 0.5
        ) / float(width)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        grid_x = grid_x.reshape(1, 1, height, width)
        grid_y = grid_y.reshape(1, 1, height, width)

        center = (object_boxes[..., :2] + object_boxes[..., 2:]) * 0.5
        size_xy = (object_boxes[..., 2:] - object_boxes[..., :2]).clamp_min(1e-4)
        sigma = (size_xy * 0.5).clamp_min(1.0 / float(max(height, width)))
        cx = center[:, :, 0:1, None]
        cy = center[:, :, 1:2, None]
        sx = sigma[:, :, 0:1, None]
        sy = sigma[:, :, 1:2, None]
        object_maps = torch.exp(
            -0.5 * (((grid_x - cx) / sx) ** 2 + ((grid_y - cy) / sy) ** 2)
        )
        map_shape = (
            object_maps.shape[0],
            *([1] * (real_alpha.ndim - 2)),
            object_maps.shape[1],
            height,
            width,
        )
        heatmap = (real_alpha[..., None, None] * object_maps.reshape(map_shape)).sum(
            dim=-3
        )
        return heatmap.clamp(0.0, 1.0)

    def _pool_interaction_context(self, feature_map, interaction_heatmap):
        if feature_map is None:
            batch_size = interaction_heatmap.shape[0]
            return interaction_heatmap.new_zeros(
                (*interaction_heatmap.shape[:-2], self.net.num_features)
            )
        feature_map = feature_map.to(
            device=interaction_heatmap.device,
            dtype=interaction_heatmap.dtype,
        )
        weights = interaction_heatmap.clamp_min(0.0)
        batch_size = weights.shape[0]
        prefix_shape = weights.shape[1:-2]
        weights_flat = weights.reshape(batch_size, -1, *weights.shape[-2:])
        denom = weights_flat.flatten(2).sum(dim=-1).clamp_min(1e-6)
        pooled = torch.einsum("bnhw,bdhw->bnd", weights_flat, feature_map)
        pooled = pooled / denom[:, :, None]
        return pooled.reshape(batch_size, *prefix_shape, feature_map.shape[1])

    def _object_relation(
        self,
        actor_feat,
        spatial_feat=None,
        actor_boxes=None,
        object_boxes=None,
        object_cls=None,
        object_conf=None,
        object_valid=None,
        action_labels=None,
        base_logits=None,
    ):
        num_classes = int(self.hparams.num_classes)
        if object_boxes is None:
            zero_delta = actor_feat.new_zeros(
                (*actor_feat.shape[:2], num_classes)
            )
            num_object_tokens = int(self.hparams.get("num_object_tokens", 0))
            zero_selection = actor_feat.new_zeros(
                (*actor_feat.shape[:2], num_classes, num_object_tokens + 1)
            )
            return zero_delta, zero_selection, None
        if object_cls is None or object_conf is None or object_valid is None:
            raise ValueError(
                "object_boxes, object_cls, object_conf, and object_valid must be passed together"
            )

        object_valid = object_valid.to(device=actor_feat.device, dtype=torch.bool)
        object_boxes = object_boxes.to(device=actor_feat.device, dtype=actor_feat.dtype)
        object_cls = object_cls.to(device=actor_feat.device).long()
        object_conf = object_conf.to(device=actor_feat.device, dtype=actor_feat.dtype)

        batch_size = actor_feat.shape[0]
        none_boxes = object_boxes.new_zeros((batch_size, 1, 4))
        none_cls = object_cls.new_full((batch_size, 1), self.num_object_classes)
        none_conf = object_conf.new_ones((batch_size, 1))
        none_valid = object_valid.new_ones((batch_size, 1))

        object_boxes_with_null = torch.cat([object_boxes, none_boxes], dim=1)
        object_cls_with_null = torch.cat([object_cls, none_cls], dim=1)
        object_conf_with_null = torch.cat([object_conf, none_conf], dim=1)
        object_valid_with_null = torch.cat([object_valid, none_valid], dim=1)

        object_cls_with_null = object_cls_with_null.clamp(0, self.num_object_classes)
        object_cls_with_null = object_cls_with_null.masked_fill(
            ~object_valid_with_null,
            self.num_object_classes,
        )
        object_feat = (
            self.object_cls_embed(object_cls_with_null).to(dtype=actor_feat.dtype)
            + self.object_bbox_mlp(object_boxes_with_null)
            + self.object_conf_mlp(object_conf_with_null.unsqueeze(-1))
            + self.object_valid_embed(object_valid_with_null.long()).to(
                dtype=actor_feat.dtype
            )
        )
        object_visual = self._pool_boxes_from_feature_map(
            spatial_feat,
            object_boxes,
            object_valid,
        )
        null_visual = object_visual.new_zeros((batch_size, 1, object_visual.shape[-1]))
        object_visual = torch.cat([object_visual, null_visual], dim=1)
        object_feat = object_feat + self.object_visual_mlp(object_visual)

        relation_geometry = self._object_relation_geometry(
            actor_boxes,
            object_boxes_with_null,
        )
        num_actors = actor_feat.shape[1]
        action_embed = self.object_action_embed.weight.to(dtype=actor_feat.dtype)
        actor_action = actor_feat[:, :, None, :].expand(
            -1,
            -1,
            num_classes,
            -1,
        )
        action_pair = action_embed[None, None, :, :].expand(
            batch_size,
            num_actors,
            -1,
            -1,
        )
        action_query = self.object_action_query(
            torch.cat(
                [
                    actor_action,
                    action_pair,
                    actor_action * action_pair,
                ],
                dim=-1,
            )
        )
        object_key = self.object_action_key(object_feat)
        object_value = self.object_action_value(object_feat)
        action_scores = torch.einsum(
            "bkcd,bmd->bkcm",
            action_query,
            object_key,
        ) / float(actor_feat.shape[-1]) ** 0.5
        geom_bias = self.object_action_geom_bias(relation_geometry).squeeze(-1)
        action_scores = action_scores + geom_bias[:, :, None, :]
        action_scores = action_scores.masked_fill(
            ~object_valid_with_null[:, None, None, :],
            torch.finfo(action_scores.dtype).min,
        )
        selection_logits = action_scores
        action_alpha = torch.softmax(selection_logits.float(), dim=-1).to(
            dtype=actor_feat.dtype
        )
        if self.object_specialist_heads_enabled:
            return self._object_specialist_relation(
                actor_feat=actor_feat,
                spatial_feat=spatial_feat,
                object_boxes=object_boxes,
                object_valid=object_valid,
                object_cls_with_null=object_cls_with_null,
                object_valid_with_null=object_valid_with_null,
                object_value=object_value,
                action_scores=selection_logits,
                action_alpha=action_alpha,
                action_embed=action_embed,
                base_logits=base_logits,
                action_labels=action_labels,
            )
        real_object_mass = action_alpha[..., : object_boxes.shape[1]].sum(
            dim=-1,
        )
        interaction_heatmap = None
        if action_labels is not None:
            action_labels = action_labels.to(
                device=actor_feat.device,
                dtype=torch.long,
            )
            action_labels = action_labels.clamp(0, num_classes - 1)
            gather_idx = action_labels[:, :, None, None].expand(
                -1,
                -1,
                1,
                action_alpha.shape[-1],
            )
            target_action_alpha = action_alpha.gather(dim=2, index=gather_idx).squeeze(
                2
            )
            interaction_heatmap = self._build_interaction_heatmap(
                target_action_alpha,
                object_boxes,
                object_valid,
                self.interaction_heatmap_size,
            )

        action_heatmap_for_pool = self._build_interaction_heatmap(
            action_alpha,
            object_boxes,
            object_valid,
            self.net.HW_OUT_CONV,
        )
        visual_action = self._pool_interaction_context(
            spatial_feat,
            action_heatmap_for_pool,
        )
        visual_action = visual_action * real_object_mass[..., None]
        object_value = object_value.clone()
        object_value[:, -1] = 0
        action_context = torch.einsum(
            "bkcm,bmd->bkcd",
            action_alpha,
            object_value,
        )
        action_feat = torch.cat(
            [
                action_context,
                visual_action,
                actor_action * action_context,
                actor_action * visual_action,
                action_pair * action_context,
                action_pair * visual_action,
            ],
            dim=-1,
        )
        action_feat = self.object_action_delta_norm(action_feat)
        raw_delta = self.object_action_delta(action_feat).squeeze(-1)
        bounded_delta = self.object_delta_scale * torch.tanh(raw_delta)
        class_gate = torch.sigmoid(self.object_relation_gate_logit).to(
            device=bounded_delta.device,
            dtype=bounded_delta.dtype,
        )
        residual_action_mask = self.object_residual_action_mask.to(
            device=bounded_delta.device,
            dtype=bounded_delta.dtype,
        )
        residual = bounded_delta * class_gate[None, None, :] * real_object_mass
        residual = residual * residual_action_mask[None, None, :]
        return residual, selection_logits, interaction_heatmap

    def _object_specialist_relation(
        self,
        actor_feat,
        spatial_feat,
        object_boxes,
        object_valid,
        object_cls_with_null,
        object_valid_with_null,
        object_value,
        action_scores,
        action_alpha,
        action_embed,
        base_logits,
        action_labels=None,
    ):
        if base_logits is None:
            raise ValueError("object_specialist_heads requires base_logits")

        batch_size, num_actors, num_classes = base_logits.shape
        residual = base_logits.new_zeros(base_logits.shape)
        specialist_selection_logits = action_scores.clone()
        interaction_heatmap = None

        object_value = object_value.clone()
        object_value[:, -1] = 0
        class_gate = torch.sigmoid(self.object_relation_gate_logit).to(
            device=base_logits.device,
            dtype=base_logits.dtype,
        )
        none_slot = object_cls_with_null.shape[1] - 1
        slot_ids = torch.arange(
            object_cls_with_null.shape[1],
            device=object_cls_with_null.device,
        )

        for group_name, _, _ in self.object_specialist_specs:
            action_idx = getattr(self, f"object_specialist_{group_name}_actions").to(
                device=base_logits.device
            )
            object_idx = getattr(self, f"object_specialist_{group_name}_objects").to(
                device=object_cls_with_null.device
            )
            object_allowed = torch.isin(object_cls_with_null, object_idx)
            object_allowed = object_allowed | (slot_ids[None, :] == none_slot)
            object_allowed = object_allowed & object_valid_with_null

            group_scores = action_scores.index_select(2, action_idx)
            group_scores = group_scores.masked_fill(
                ~object_allowed[:, None, None, :],
                torch.finfo(group_scores.dtype).min,
            )
            group_alpha = torch.softmax(group_scores.float(), dim=-1).to(
                dtype=actor_feat.dtype
            )
            specialist_selection_logits.index_copy_(2, action_idx, group_scores)

            real_mass = group_alpha[..., : object_boxes.shape[1]].sum(dim=-1)
            group_context = torch.einsum(
                "bkgm,bmd->bkgd",
                group_alpha,
                object_value,
            )
            group_heatmap = self._build_interaction_heatmap(
                group_alpha,
                object_boxes,
                object_valid,
                self.net.HW_OUT_CONV,
            )
            group_visual = self._pool_interaction_context(spatial_feat, group_heatmap)
            group_visual = group_visual * real_mass[..., None]

            actor_group = actor_feat[:, :, None, :].expand(
                -1,
                -1,
                action_idx.numel(),
                -1,
            )
            action_group = action_embed.index_select(0, action_idx)[None, None, :, :]
            action_group = action_group.expand(batch_size, num_actors, -1, -1)
            base_group = base_logits.index_select(2, action_idx).to(
                dtype=actor_feat.dtype
            )
            base_group_embed = self.object_specialist_base_mlp(
                base_group.unsqueeze(-1)
            )
            group_feat = torch.cat(
                [
                    group_context,
                    group_visual,
                    actor_group * group_context,
                    actor_group * group_visual,
                    action_group * group_context,
                    action_group * group_visual,
                    base_group_embed,
                ],
                dim=-1,
            )
            group_feat = self.object_specialist_delta_norms[group_name](group_feat)
            group_raw = self.object_specialist_delta_heads[group_name](
                group_feat
            ).squeeze(-1)
            group_delta = self.object_delta_scale * torch.tanh(group_raw)
            group_delta = (
                group_delta
                * class_gate.index_select(0, action_idx)[None, None, :]
                * real_mass
            )
            residual.index_add_(2, action_idx, group_delta.to(dtype=residual.dtype))

        if action_labels is not None:
            action_labels = action_labels.to(
                device=base_logits.device,
                dtype=torch.long,
            ).clamp(0, num_classes - 1)
            gather_idx = action_labels[:, :, None, None].expand(
                -1,
                -1,
                1,
                action_alpha.shape[-1],
            )
            target_action_alpha = torch.softmax(
                specialist_selection_logits.gather(dim=2, index=gather_idx)
                .squeeze(2)
                .float(),
                dim=-1,
            ).to(dtype=actor_feat.dtype)
            interaction_heatmap = self._build_interaction_heatmap(
                target_action_alpha,
                object_boxes,
                object_valid,
                self.interaction_heatmap_size,
            )

        return residual, specialist_selection_logits, interaction_heatmap

    def forward(
        self,
        x,
        boxes=None,
        valid=None,
        object_boxes=None,
        object_cls=None,
        object_conf=None,
        object_valid=None,
        action_labels=None,
    ):
        # convert to b c t h w
        x = x.permute(0, 2, 1, 3, 4)
        if self.actor_prompt:
            if self.hparams.n_landmarks > 0 or self.object_prompt:
                net_data = self.net(
                    x,
                    boxes=boxes,
                    valid=valid,
                    object_boxes=object_boxes,
                    object_valid=object_valid,
                    object_conf=object_conf,
                )
                if len(net_data) == 4:
                    _, x_actor, x_heatmap, x_spatial = net_data
                else:
                    _, x_actor, x_heatmap = net_data
                    x_spatial = None
            else:
                data = self.net(
                    x,
                    boxes=boxes,
                    valid=valid,
                    object_boxes=object_boxes,
                    object_valid=object_valid,
                    object_conf=object_conf,
                )
                _, x_actor = data
                x_heatmap = 0
                x_spatial = None
            if self.hparams.ret_feat:
                return x_actor
            action_logits = self.actor_head(x_actor)
            if self.object_prompt:
                (
                    object_delta,
                    selection_logits,
                    interaction_heatmap,
                ) = self._object_relation(
                    x_actor,
                    spatial_feat=x_spatial,
                    actor_boxes=boxes,
                    object_boxes=object_boxes,
                    object_cls=object_cls,
                    object_conf=object_conf,
                    object_valid=object_valid,
                    action_labels=action_labels,
                    base_logits=action_logits,
                )
                action_logits = action_logits + object_delta
            else:
                selection_logits = None
                interaction_heatmap = None
            if self.presence_head is not None:
                presence_logits = self.presence_head(x_actor).squeeze(-1)
                if self.object_prompt:
                    return (
                        action_logits,
                        x_heatmap,
                        presence_logits,
                        selection_logits,
                        interaction_heatmap,
                        object_delta,
                    )
                return action_logits, x_heatmap, presence_logits
            if self.object_prompt:
                return (
                    action_logits,
                    x_heatmap,
                    None,
                    selection_logits,
                    interaction_heatmap,
                    object_delta,
                )
            return action_logits, x_heatmap
        if self.hparams.n_landmarks > 0:
            x_class, x_heatmap = self.net(x)
            if self.hparams.ret_feat:
                return x_class
            x_class = (
                self.head(x_class)
                if not self.hparams.get("linear_probe", 0)
                else self.head(x_class, mode=self.mode)
            )
            return x_class, x_heatmap
        else:
            x_class = self.net(x)
            if self.hparams.ret_feat:
                return x_class
            x_class = (
                self.head(x_class)
                if not self.hparams.get("linear_probe", 0)
                else self.head(x_class, mode=self.mode)
            )
            return x_class, 0

    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--net_size", type=str, default="b")
        parser.add_argument("--pretrained", type=str, default="DEFAULT")
        parser.add_argument("--freeze_backbone", type=int, default=0)
        parser.add_argument("--freeze_stages", type=int, default=-1)
        parser.add_argument("--drop_rate", type=float, default=0.0)
        parser.add_argument("--attn_drop_rate", type=float, default=0.0)
        parser.add_argument("--drop_path_rate", type=float, default=0.0)
        parser.add_argument("--head_drop_rate", type=float, default=0.0)
        parser.add_argument("--keep_rate", type=float, default=0.6)
        parser.add_argument("--enhanced_weight_class", type=float, default=1)
        parser.add_argument("--enhanced_weight_heatmap", type=float, default=1)

        parser.add_argument(
            "--sim_metric", type=int, default=1
        )  # 0: k, 1: attn, 2: q, 3: v
        parser.add_argument(
            "--topk_type", type=int, default=1
        )  # 0: all, 1: cls_hm, 2: cls
        parser.add_argument("--merge_mode", type=int, default=1)  # 0: mean, 1: sum
        parser.add_argument("--keep_rate_merge", type=float, default=0.3)
        parser.add_argument(
            "--merge_type", type=str, default="sim"
        )  # sim :poguise, tome :tome
        # parser.add_argument("--enhanced_weight_class_obj", type=float, default=1)
        parser.add_argument("--hw_out_conv", type=int, default=8)
        parser.add_argument("--use_register_tokens", type=int, default=0)
        parser.add_argument("--n_registers", type=int, default=0)
        parser.add_argument("--actor_prompt", type=int, default=0)
        parser.add_argument("--num_actor_tokens", type=int, default=8)
        parser.add_argument("--actor_bbox_prior_weight", type=float, default=0.1)
        parser.add_argument("--actor_bbox_prior_expand", type=float, default=1.75)
        parser.add_argument("--actor_presence_head", type=int, default=0)
        parser.add_argument("--presence_loss_weight", type=float, default=0.05)
        parser.add_argument("--actor_val_diagnostics", type=int, default=1)
        parser.add_argument("--actor_val_diagnostic_max_pairs", type=int, default=8)
        parser.add_argument("--object_bbox_prior_weight", type=float, default=0.0)
        parser.add_argument("--object_bbox_prior_expand", type=float, default=1.25)
        parser.add_argument("--object_dropout_prob", type=float, default=0.0)
        parser.add_argument("--object_token_dropout_prob", type=float, default=0.0)
        parser.add_argument("--object_relation_hidden_dim", type=int, default=512)
        parser.add_argument("--object_relation_dropout", type=float, default=0.1)
        parser.add_argument("--object_relation_gate_init", type=float, default=0.25)
        parser.add_argument("--object_action_gate_init", type=float, default=0.05)
        parser.add_argument("--object_delta_scale", type=float, default=1.0)
        parser.add_argument("--object_residual_action_mask", type=str, default="none")
        parser.add_argument("--object_specialist_heads", type=int, default=0)
        parser.add_argument("--object_relation_only", type=int, default=0)
        parser.add_argument(
            "--object_interaction_heatmap_weight",
            type=float,
            default=0.0,
        )
        parser.add_argument("--ret_feat", type=int, default=0)
        parser.add_argument("--linear_probe", type=int, default=0)

        return parser
