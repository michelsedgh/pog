from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ObjectInteractionModule(nn.Module):
    """Actor-conditioned object selection and feature-level fusion.

    The module consumes every detected object candidate plus an explicit NONE token.
    It predicts an actor/object selection distribution, builds an actor-conditioned
    interaction heatmap from that distribution, pools visual context through that
    heatmap, and returns a bounded actor-feature adapter. It never writes action
    logits directly.
    """

    def __init__(
        self,
        feature_dim: int,
        num_object_classes: int,
        hidden_dim: int,
        dropout: float,
        fusion_gate_init: float,
        feature_scale: float,
        interaction_heatmap_size: int,
    ):
        super().__init__()
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if num_object_classes <= 0:
            raise ValueError("num_object_classes must be positive")
        if hidden_dim <= 0:
            raise ValueError("object_interaction_hidden_dim must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("object_interaction_dropout must be in [0, 1)")
        if not 0 < fusion_gate_init < 1:
            raise ValueError("object_fusion_gate_init must be in (0, 1)")
        if feature_scale <= 0:
            raise ValueError("object_feature_scale must be positive")
        if interaction_heatmap_size != 56:
            raise ValueError("object interaction heatmaps are trained at 56x56")

        self.feature_dim = int(feature_dim)
        self.num_object_classes = int(num_object_classes)
        self.feature_scale = float(feature_scale)
        self.interaction_heatmap_size = (
            int(interaction_heatmap_size),
            int(interaction_heatmap_size),
        )

        self.object_cls_embed = nn.Embedding(num_object_classes + 1, feature_dim)
        self.object_bbox_mlp = nn.Sequential(
            nn.Linear(4, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
        )
        self.object_conf_mlp = nn.Sequential(
            nn.Linear(1, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
        )
        self.object_visual_mlp = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
        )
        self.actor_query = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
        )
        self.object_key = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
        )
        self.object_value = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
        )
        self.geometry_bias = nn.Sequential(
            nn.LayerNorm(10),
            nn.Linear(10, max(32, hidden_dim // 4)),
            nn.GELU(),
            nn.Linear(max(32, hidden_dim // 4), 1),
        )
        self.fusion_norm = nn.LayerNorm(feature_dim * 4)
        self.feature_adapter = nn.Sequential(
            nn.Linear(feature_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim),
        )
        self.object_valid_embed = nn.Embedding(2, feature_dim)
        self.fusion_gate_logit = nn.Parameter(
            torch.logit(torch.tensor(float(fusion_gate_init), dtype=torch.float32))
        )

        nn.init.normal_(self.object_cls_embed.weight, std=0.02)
        nn.init.normal_(self.object_valid_embed.weight, std=0.02)
        nn.init.zeros_(self.feature_adapter[-1].weight)
        nn.init.zeros_(self.feature_adapter[-1].bias)

    def _geometry(self, actor_boxes, object_boxes):
        batch_size, num_objects, _ = object_boxes.shape
        if actor_boxes is None:
            return object_boxes.new_zeros((batch_size, 0, num_objects, 10))

        actor_boxes = actor_boxes.to(
            device=object_boxes.device,
            dtype=object_boxes.dtype,
        ).clamp(0.0, 1.0)
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
        if feature_map is None:
            return boxes.new_zeros((batch_size, num_objects, self.feature_dim))

        feature_map = feature_map.to(device=boxes.device, dtype=boxes.dtype)
        _, channels, _, _ = feature_map.shape
        if channels != self.feature_dim:
            raise ValueError(
                f"Feature map channel count {channels} does not match model dim "
                f"{self.feature_dim}"
            )

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
        return pooled * valid[:, :, None].to(dtype=pooled.dtype)

    def _build_interaction_heatmap(self, object_alpha, object_boxes, object_valid, size):
        if isinstance(size, int):
            height = width = int(size)
        else:
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
                (*interaction_heatmap.shape[:-2], self.feature_dim)
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

    def forward(
        self,
        actor_feat,
        spatial_feat: Optional[torch.Tensor] = None,
        actor_boxes: Optional[torch.Tensor] = None,
        object_boxes: Optional[torch.Tensor] = None,
        object_cls: Optional[torch.Tensor] = None,
        object_conf: Optional[torch.Tensor] = None,
        object_valid: Optional[torch.Tensor] = None,
        spatial_heatmap_size: Optional[Tuple[int, int]] = None,
    ):
        if object_boxes is None:
            feature_delta = actor_feat.new_zeros(actor_feat.shape)
            selection_logits = actor_feat.new_zeros((*actor_feat.shape[:2], 1))
            return feature_delta, selection_logits, None
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

        object_boxes_with_none = torch.cat([object_boxes, none_boxes], dim=1)
        object_cls_with_none = torch.cat([object_cls, none_cls], dim=1)
        object_conf_with_none = torch.cat([object_conf, none_conf], dim=1)
        object_valid_with_none = torch.cat([object_valid, none_valid], dim=1)

        object_cls_with_none = object_cls_with_none.clamp(0, self.num_object_classes)
        object_cls_with_none = object_cls_with_none.masked_fill(
            ~object_valid_with_none,
            self.num_object_classes,
        )
        object_feat = (
            self.object_cls_embed(object_cls_with_none).to(dtype=actor_feat.dtype)
            + self.object_bbox_mlp(object_boxes_with_none)
            + self.object_conf_mlp(object_conf_with_none.unsqueeze(-1))
            + self.object_valid_embed(object_valid_with_none.long()).to(
                dtype=actor_feat.dtype
            )
        )
        object_visual = self._pool_boxes_from_feature_map(
            spatial_feat,
            object_boxes,
            object_valid,
        )
        none_visual = object_visual.new_zeros((batch_size, 1, object_visual.shape[-1]))
        object_visual = torch.cat([object_visual, none_visual], dim=1)
        object_feat = object_feat + self.object_visual_mlp(object_visual)

        actor_query = self.actor_query(actor_feat)
        object_key = self.object_key(object_feat)
        object_value = self.object_value(object_feat)
        selection_scores = torch.einsum(
            "bkd,bmd->bkm",
            actor_query,
            object_key,
        ) / float(actor_feat.shape[-1]) ** 0.5
        geometry = self._geometry(actor_boxes, object_boxes_with_none)
        if geometry.shape[1] != actor_feat.shape[1]:
            geometry = object_boxes.new_zeros(
                (batch_size, actor_feat.shape[1], object_boxes_with_none.shape[1], 10)
            )
        selection_scores = selection_scores + self.geometry_bias(geometry).squeeze(-1)
        selection_scores = selection_scores.masked_fill(
            ~object_valid_with_none[:, None, :],
            torch.finfo(selection_scores.dtype).min,
        )
        selection_logits = selection_scores
        object_alpha = torch.softmax(selection_logits.float(), dim=-1).to(
            dtype=actor_feat.dtype
        )
        real_object_mass = object_alpha[..., : object_boxes.shape[1]].sum(dim=-1)
        interaction_heatmap = self._build_interaction_heatmap(
            object_alpha,
            object_boxes,
            object_valid,
            self.interaction_heatmap_size,
        )

        if spatial_heatmap_size is None:
            spatial_heatmap_size = spatial_feat.shape[-2:] if spatial_feat is not None else (1, 1)
        action_heatmap_for_pool = self._build_interaction_heatmap(
            object_alpha,
            object_boxes,
            object_valid,
            spatial_heatmap_size,
        )
        visual_context = self._pool_interaction_context(
            spatial_feat,
            action_heatmap_for_pool,
        )
        visual_context = visual_context * real_object_mass[..., None]
        object_value = object_value.clone()
        object_value[:, -1] = 0
        object_context = torch.einsum(
            "bkm,bmd->bkd",
            object_alpha,
            object_value,
        )
        fusion_feat = torch.cat(
            [
                object_context,
                visual_context,
                actor_feat * object_context,
                actor_feat * visual_context,
            ],
            dim=-1,
        )
        raw_delta = self.feature_adapter(self.fusion_norm(fusion_feat))
        bounded_delta = self.feature_scale * torch.tanh(raw_delta)
        feature_gate = torch.sigmoid(self.fusion_gate_logit).to(
            device=bounded_delta.device,
            dtype=bounded_delta.dtype,
        )
        feature_delta = bounded_delta * feature_gate * real_object_mass[..., None]
        return feature_delta, selection_logits, interaction_heatmap
