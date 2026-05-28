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
        # Unfreeze the head
        for param in self.head.parameters():
            param.requires_grad = True
        if self.actor_prompt:
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
        hidden_dim = int(self.hparams.get("object_relation_hidden_dim", 512))
        if hidden_dim <= 0:
            raise ValueError("object_relation_hidden_dim must be positive")
        dropout = float(self.hparams.get("object_relation_dropout", 0.1))
        if not 0 <= dropout < 1:
            raise ValueError("object_relation_dropout must be in [0, 1)")
        gate_init = float(self.hparams.get("object_relation_gate_init", 0.25))
        if not 0 < gate_init < 1:
            raise ValueError("object_relation_gate_init must be in (0, 1)")
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
        self.object_selector = nn.Sequential(
            nn.LayerNorm(dim * 3 + 10),
            nn.Linear(dim * 3 + 10, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.object_relation_delta_norm = nn.LayerNorm(dim * 4)
        self.object_relation_delta = nn.Sequential(
            nn.Linear(dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.hparams.num_classes),
        )
        self.object_valid_embed = nn.Embedding(2, dim)
        self.object_relation_gate_logit = nn.Parameter(
            torch.logit(torch.tensor(gate_init, dtype=torch.float32))
        )

        nn.init.normal_(self.object_cls_embed.weight, std=0.02)
        nn.init.normal_(self.object_valid_embed.weight, std=0.02)
        nn.init.zeros_(self.object_relation_delta[-1].weight)
        nn.init.zeros_(self.object_relation_delta[-1].bias)

    def _object_relation_modules(self):
        return (
            self.object_cls_embed,
            self.object_bbox_mlp,
            self.object_conf_mlp,
            self.object_visual_mlp,
            self.object_selector,
            self.object_valid_embed,
            self.object_relation_delta_norm,
            self.object_relation_delta,
        )

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
        real_alpha = real_alpha * object_valid[:, None, :].to(dtype=real_alpha.dtype)

        y = (
            torch.arange(height, device=object_boxes.device, dtype=object_boxes.dtype)
            + 0.5
        ) / float(height)
        x = (
            torch.arange(width, device=object_boxes.device, dtype=object_boxes.dtype)
            + 0.5
        ) / float(width)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        grid_x = grid_x.reshape(1, 1, 1, height, width)
        grid_y = grid_y.reshape(1, 1, 1, height, width)

        center = (object_boxes[..., :2] + object_boxes[..., 2:]) * 0.5
        size_xy = (object_boxes[..., 2:] - object_boxes[..., :2]).clamp_min(1e-4)
        sigma = (size_xy * 0.5).clamp_min(1.0 / float(max(height, width)))
        cx = center[:, None, :, 0:1, None]
        cy = center[:, None, :, 1:2, None]
        sx = sigma[:, None, :, 0:1, None]
        sy = sigma[:, None, :, 1:2, None]
        object_maps = torch.exp(
            -0.5 * (((grid_x - cx) / sx) ** 2 + ((grid_y - cy) / sy) ** 2)
        )
        heatmap = (real_alpha[:, :, :, None, None] * object_maps).sum(dim=2)
        return heatmap.clamp(0.0, 1.0)

    def _pool_interaction_context(self, feature_map, interaction_heatmap):
        if feature_map is None:
            batch_size, num_actors = interaction_heatmap.shape[:2]
            return interaction_heatmap.new_zeros(
                (batch_size, num_actors, self.net.num_features)
            )
        feature_map = feature_map.to(
            device=interaction_heatmap.device,
            dtype=interaction_heatmap.dtype,
        )
        weights = interaction_heatmap.clamp_min(0.0)
        denom = weights.flatten(2).sum(dim=-1).clamp_min(1e-6)
        return torch.einsum("bkhw,bchw->bkc", weights, feature_map) / denom[:, :, None]

    def _object_relation(
        self,
        actor_feat,
        spatial_feat=None,
        actor_boxes=None,
        object_boxes=None,
        object_cls=None,
        object_conf=None,
        object_valid=None,
    ):
        if object_boxes is None:
            zero_delta = actor_feat.new_zeros(
                (*actor_feat.shape[:2], self.hparams.num_classes)
            )
            num_object_tokens = int(self.hparams.get("num_object_tokens", 0))
            zero_selection = actor_feat.new_zeros(
                (*actor_feat.shape[:2], num_object_tokens + 1)
            )
            zero_heatmap = actor_feat.new_zeros(
                (*actor_feat.shape[:2], *self.interaction_heatmap_size)
            )
            return zero_delta, zero_selection, zero_heatmap
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
        actor_pair = actor_feat[:, :, None, :].expand(
            -1,
            -1,
            object_feat.shape[1],
            -1,
        )
        object_pair = object_feat[:, None, :, :].expand(
            -1,
            num_actors,
            -1,
            -1,
        )
        selector_input = torch.cat(
            [actor_pair, object_pair, actor_pair * object_pair, relation_geometry],
            dim=-1,
        )
        selection_logits = self.object_selector(selector_input).squeeze(-1)
        selection_logits = selection_logits.masked_fill(
            ~object_valid_with_null[:, None, :],
            torch.finfo(selection_logits.dtype).min,
        )

        object_alpha = torch.softmax(selection_logits.float(), dim=-1).to(
            dtype=actor_feat.dtype
        )
        real_alpha = object_alpha[..., : object_boxes.shape[1]]
        real_alpha = real_alpha * object_valid[:, None, :].to(dtype=actor_feat.dtype)
        selected_context = torch.einsum(
            "bkm,bmd->bkd",
            real_alpha,
            object_feat[:, : object_boxes.shape[1], :],
        )
        interaction_heatmap = self._build_interaction_heatmap(
            object_alpha,
            object_boxes,
            object_valid,
            self.interaction_heatmap_size,
        )
        interaction_heatmap_for_pool = F.interpolate(
            interaction_heatmap.flatten(0, 1).unsqueeze(1),
            size=self.net.HW_OUT_CONV,
            mode="bilinear",
            align_corners=False,
        ).squeeze(1).reshape(
            batch_size,
            num_actors,
            *self.net.HW_OUT_CONV,
        )
        visual_context = self._pool_interaction_context(
            spatial_feat,
            interaction_heatmap_for_pool,
        )
        action_feat = torch.cat(
            [
                selected_context,
                visual_context,
                actor_feat * selected_context,
                actor_feat * visual_context,
            ],
            dim=-1,
        )
        action_feat = self.object_relation_delta_norm(action_feat)
        delta = self.object_relation_delta(action_feat)
        return delta, selection_logits, interaction_heatmap

    def forward(
        self,
        x,
        boxes=None,
        valid=None,
        object_boxes=None,
        object_cls=None,
        object_conf=None,
        object_valid=None,
    ):
        # convert to b c t h w
        x = x.permute(0, 2, 1, 3, 4)
        if self.object_prompt:
            object_valid = self._apply_object_dropout(object_valid)
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
                )
                gate = torch.sigmoid(self.object_relation_gate_logit).to(
                    device=action_logits.device,
                    dtype=action_logits.dtype,
                )
                action_logits = action_logits + gate * object_delta
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
                    )
                return action_logits, x_heatmap, presence_logits
            if self.object_prompt:
                return (
                    action_logits,
                    x_heatmap,
                    None,
                    selection_logits,
                    interaction_heatmap,
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
        parser.add_argument("--object_relation_only", type=int, default=0)
        parser.add_argument(
            "--object_interaction_heatmap_weight",
            type=float,
            default=0.0,
        )
        parser.add_argument("--ret_feat", type=int, default=0)
        parser.add_argument("--linear_probe", type=int, default=0)

        return parser
