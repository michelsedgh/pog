from typing import Optional, Tuple

import math

import torch
import torch.nn as nn


def _logit(prob: float) -> float:
    prob = min(max(float(prob), 1e-6), 1.0 - 1e-6)
    return math.log(prob / (1.0 - prob))


class ActorObjectDecoder(nn.Module):
    """Refine actor tokens with actor-conditioned object evidence.

    The transformer backbone inserts object tokens into the token sequence, but
    the final actor classifier still reads actor tokens. This decoder is the
    explicit per-actor binding step:

      actor token queries object tokens + actor/object geometry + union visual
      evidence, then the selected object context updates the actor token before
      actor_head sees it.

    It also emits the same selection logits and interaction heatmaps used by
    the object supervision and validation diagnostics. It never writes action
    logits directly.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        dropout: float,
        interaction_heatmap_size: int,
        init_update_gate: float = 0.02,
        init_ffn_gate: float = 0.02,
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

        self.actor_norm = nn.LayerNorm(feature_dim)
        self.object_norm = nn.LayerNorm(feature_dim)
        self.pair_norm = nn.LayerNorm(feature_dim)

        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.pair_value = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim),
        )
        self.geometry_bias = nn.Sequential(
            nn.LayerNorm(10),
            nn.Linear(10, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.pair_score = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
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

        self.update_norm = nn.LayerNorm(feature_dim)
        self.update_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim),
        )
        self.ffn_norm = nn.LayerNorm(feature_dim)
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim),
        )
        self.update_gate_logit = nn.Parameter(
            torch.tensor(_logit(init_update_gate), dtype=torch.float32)
        )
        self.ffn_gate_logit = nn.Parameter(
            torch.tensor(_logit(init_ffn_gate), dtype=torch.float32)
        )

    def _geometry(self, actor_boxes, object_boxes, num_actors: int):
        batch_size, num_objects, _ = object_boxes.shape
        if actor_boxes is None:
            return object_boxes.new_zeros((batch_size, num_actors, num_objects, 10))

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
                object_center_pair.expand(-1, num_actors, -1, -1),
                object_size_pair.expand(-1, num_actors, -1, -1),
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
            raise ValueError("object_tokens are required for actor-object decoding")
        if pair_visual_features is None:
            raise ValueError(
                "pair_visual_features are required for actor-object decoding"
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

        batch_size, num_actors, feature_dim = actor_tokens.shape
        num_objects = object_tokens.shape[1]
        actor_norm = self.actor_norm(actor_tokens)
        object_norm = self.object_norm(object_tokens)
        pair_norm = self.pair_norm(pair_visual_features)

        query = self.query(actor_norm)
        key = self.key(object_norm)
        value = self.value(object_norm)
        pair_value = self.pair_value(pair_norm)

        attention_logits = torch.einsum("bkd,bmd->bkm", query, key)
        attention_logits = attention_logits / math.sqrt(float(feature_dim))

        geometry = self._geometry(actor_boxes, object_boxes, num_actors).to(
            dtype=actor_tokens.dtype
        )
        attention_logits = attention_logits + self.geometry_bias(geometry).squeeze(-1)
        attention_logits = attention_logits + self.pair_score(pair_norm).squeeze(-1)
        attention_logits = attention_logits.masked_fill(
            ~object_valid[:, None, :],
            torch.finfo(attention_logits.dtype).min,
        )
        none_logits = self.none_mlp(actor_norm)
        selection_logits = torch.cat([attention_logits, none_logits], dim=-1)

        alpha = torch.softmax(selection_logits.float(), dim=-1).to(
            dtype=actor_tokens.dtype
        )
        real_alpha = alpha[..., :num_objects]
        object_context = value[:, None, :, :] + pair_value
        context = (real_alpha[..., None] * object_context).sum(dim=2)

        update_gate = torch.sigmoid(self.update_gate_logit).to(dtype=actor_tokens.dtype)
        ffn_gate = torch.sigmoid(self.ffn_gate_logit).to(dtype=actor_tokens.dtype)
        refined_actor = actor_tokens + update_gate * self.update_proj(
            self.update_norm(context)
        )
        refined_actor = refined_actor + ffn_gate * self.ffn(
            self.ffn_norm(refined_actor)
        )

        interaction_heatmap = self._build_interaction_heatmap(
            alpha,
            object_boxes,
            object_valid,
            heatmap_size or self.interaction_heatmap_size,
        )
        return refined_actor, selection_logits, interaction_heatmap
