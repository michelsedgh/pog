from typing import Optional, Tuple

import torch
import torch.nn as nn


class ObjectInteractionHead(nn.Module):
    """Actor/object selection head for object-token PO-GUISE+.

    Object candidates are already embedded into the transformer token sequence
    with pooled object-region patch features. This head reads final actor/object
    tokens plus actor-object union visual features to produce:
      * actor-conditioned object/NONE selection logits
      * actor-conditioned interaction heatmaps built from selected object boxes

    It intentionally does not produce a feature update or logit residual. The
    action classifier reads actor tokens that have already attended to object
    tokens inside the transformer.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        dropout: float,
        interaction_heatmap_size: int,
    ):
        super().__init__()
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("object_interaction_hidden_dim must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("object_interaction_dropout must be in [0, 1)")
        if interaction_heatmap_size != 56:
            raise ValueError("object interaction heatmaps are trained at 56x56")

        self.feature_dim = int(feature_dim)
        self.interaction_heatmap_size = (
            int(interaction_heatmap_size),
            int(interaction_heatmap_size),
        )
        self.pair_visual_proj = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim),
        )
        selector_dim = feature_dim * 4 + 10
        self.selector = nn.Sequential(
            nn.LayerNorm(selector_dim),
            nn.Linear(selector_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.none_mlp = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

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

    def forward(
        self,
        actor_tokens: torch.Tensor,
        object_tokens: torch.Tensor,
        actor_boxes: Optional[torch.Tensor],
        object_boxes: torch.Tensor,
        object_valid: torch.Tensor,
        pair_visual_features: torch.Tensor,
        heatmap_size: Optional[Tuple[int, int]] = None,
    ):
        if object_tokens is None:
            raise ValueError("object_tokens are required for object interaction")
        if pair_visual_features is None:
            raise ValueError(
                "pair_visual_features are required for visually grounded object interaction"
            )
        object_valid = object_valid.to(device=actor_tokens.device, dtype=torch.bool)
        object_boxes = object_boxes.to(device=actor_tokens.device, dtype=actor_tokens.dtype)
        object_tokens = object_tokens.to(device=actor_tokens.device, dtype=actor_tokens.dtype)
        pair_visual_features = pair_visual_features.to(
            device=actor_tokens.device,
            dtype=actor_tokens.dtype,
        )
        expected_pair_shape = (
            actor_tokens.shape[0],
            actor_tokens.shape[1],
            object_tokens.shape[1],
            actor_tokens.shape[-1],
        )
        if tuple(pair_visual_features.shape) != expected_pair_shape:
            raise ValueError(
                "pair_visual_features must have shape "
                f"{expected_pair_shape}, got {tuple(pair_visual_features.shape)}"
            )

        batch_size = actor_tokens.shape[0]
        num_objects = object_tokens.shape[1]
        geometry = self._geometry(actor_boxes, object_boxes)
        if geometry.shape[1] != actor_tokens.shape[1]:
            geometry = object_boxes.new_zeros(
                (batch_size, actor_tokens.shape[1], num_objects, 10)
            )
        pair_visual_features = self.pair_visual_proj(pair_visual_features)

        actor_pair = actor_tokens[:, :, None, :].expand(
            -1,
            -1,
            num_objects,
            -1,
        )
        object_pair = object_tokens[:, None, :, :].expand(
            -1,
            actor_tokens.shape[1],
            -1,
            -1,
        )
        selector_input = torch.cat(
            [
                actor_pair,
                object_pair,
                actor_pair * object_pair,
                pair_visual_features,
                geometry.to(dtype=actor_tokens.dtype),
            ],
            dim=-1,
        )
        object_logits = self.selector(selector_input).squeeze(-1)
        object_logits = object_logits.masked_fill(
            ~object_valid[:, None, :],
            torch.finfo(object_logits.dtype).min,
        )
        none_logits = self.none_mlp(actor_tokens)
        selection_logits = torch.cat([object_logits, none_logits], dim=-1)

        object_alpha = torch.softmax(selection_logits.float(), dim=-1).to(
            dtype=actor_tokens.dtype
        )
        interaction_heatmap = self._build_interaction_heatmap(
            object_alpha,
            object_boxes,
            object_valid,
            heatmap_size or self.interaction_heatmap_size,
        )
        return selection_logits, interaction_heatmap
