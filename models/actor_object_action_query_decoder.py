"""Actor-object action query decoder.

This module scores actions from actor tokens and fixed object proposals without
using a global selected-object bottleneck.  For each actor and action, an action
query attends over explanation slots:

    NULL, UNKNOWN, object_1, ..., object_K

Objectless actions can attend only to NULL.  Objectful actions can attend to
UNKNOWN and valid object slots.  The decoder produces a bounded relation energy
that is added to the motion-only actor logits, so training starts from the
motion model and learns actor-object evidence as a residual energy term.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ActionObjectQuerySpec:
    """Static action/object taxonomy used by the query decoder."""

    num_actions: int
    num_object_classes: int
    objectless_action_indices: Tuple[int, ...]
    action_to_object_ids: Mapping[int, Tuple[int, ...]]

    def build_buffers(self) -> Dict[str, Tensor]:
        num_actions = int(self.num_actions)
        num_objects = int(self.num_object_classes)
        if num_actions <= 0 or num_objects <= 0:
            raise ValueError("num_actions and num_object_classes must be positive")

        objectless = torch.zeros(num_actions, dtype=torch.float32)
        for action_idx in self.objectless_action_indices:
            action_idx = int(action_idx)
            if 0 <= action_idx < num_actions:
                objectless[action_idx] = 1.0

        compat = torch.zeros(num_objects, num_actions, dtype=torch.float32)
        has_known_object = torch.zeros(num_actions, dtype=torch.float32)
        for action_idx, object_ids in self.action_to_object_ids.items():
            action_idx = int(action_idx)
            if not (0 <= action_idx < num_actions):
                continue
            for object_id in object_ids:
                object_id = int(object_id)
                if 0 <= object_id < num_objects:
                    compat[object_id, action_idx] = 1.0
                    has_known_object[action_idx] = 1.0

        return {
            "objectless_action": objectless,
            "objectful_action": (1.0 - objectless).clamp(0.0, 1.0),
            "has_known_object": has_known_object,
            "compat_by_object_action": compat,
        }


class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class ActorObjectActionQueryDecoder(nn.Module):
    """Direct action decoder from actor-object relation slots."""

    def __init__(
        self,
        dim: int,
        spec: ActionObjectQuerySpec,
        hidden_dim: int = 512,
        attn_dim: int = 256,
        compatible_bias: float = 0.75,
        incompatible_bias: float = -0.75,
        unknown_bias: float = -0.10,
        unknown_mismatch_penalty: float = 1.0,
        quality_init_bias: float = -3.0,
        relation_logit_scale_init: float = -2.0,
        neg_inf: float = -1.0e4,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_actions = int(spec.num_actions)
        self.num_object_classes = int(spec.num_object_classes)
        self.hidden_dim = int(hidden_dim)
        self.attn_dim = int(attn_dim)
        self.compatible_bias = float(compatible_bias)
        self.incompatible_bias = float(incompatible_bias)
        self.unknown_bias = float(unknown_bias)
        self.unknown_mismatch_penalty = float(unknown_mismatch_penalty)
        self.quality_init_bias = float(quality_init_bias)
        self.relation_logit_scale_init = float(relation_logit_scale_init)
        self.neg_inf = float(neg_inf)

        for name, tensor in spec.build_buffers().items():
            self.register_buffer(name, tensor, persistent=False)

        self.object_class_embed = nn.Embedding(self.num_object_classes, dim)
        self.object_meta_mlp = _MLP(10, max(hidden_dim // 2, 1), dim)
        self.object_norm = nn.LayerNorm(dim)

        self.pair_geom_mlp = _MLP(13, max(hidden_dim // 2, 1), dim)
        self.relation_mlp = nn.Sequential(
            nn.LayerNorm(dim * 5),
            nn.Linear(dim * 5, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self.relation_norm = nn.LayerNorm(dim)

        self.null_slot = nn.Parameter(torch.empty(dim))
        self.unknown_slot = nn.Parameter(torch.empty(dim))
        self.null_mlp = _MLP(dim * 2, hidden_dim, dim)
        self.unknown_mlp = _MLP(dim * 2, hidden_dim, dim)

        self.action_embed = nn.Parameter(torch.empty(self.num_actions, dim))
        self.action_query = nn.Linear(dim, attn_dim, bias=False)
        self.slot_key = nn.Linear(dim, attn_dim, bias=False)
        self.slot_value = nn.Linear(dim, dim, bias=False)

        self.actor_proj = nn.Linear(dim, dim)
        self.relation_proj = nn.Linear(dim, dim)
        self.action_proj = nn.Linear(dim, dim)
        self.logit_head = nn.Linear(dim, 1)
        self.relation_logit_scale = nn.Parameter(
            torch.tensor(self.relation_logit_scale_init, dtype=torch.float32)
        )

        self.quality_head = nn.Linear(dim, 1)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.null_slot, std=0.02)
        nn.init.normal_(self.unknown_slot, std=0.02)
        nn.init.normal_(self.action_embed, std=0.02)
        nn.init.zeros_(self.logit_head.weight)
        nn.init.zeros_(self.logit_head.bias)
        nn.init.zeros_(self.quality_head.weight)
        nn.init.constant_(self.quality_head.bias, self.quality_init_bias)

    @staticmethod
    def _box_features(boxes: Tensor, confs: Tensor) -> Tensor:
        x1, y1, x2, y2 = boxes.unbind(dim=-1)
        w = (x2 - x1).clamp_min(1.0e-4)
        h = (y2 - y1).clamp_min(1.0e-4)
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        area = (w * h).clamp(0.0, 1.0)
        return torch.stack((x1, y1, x2, y2, cx, cy, w, h, area, confs), dim=-1)

    @staticmethod
    def _pair_geometry(
        actor_boxes: Tensor,
        object_boxes: Tensor,
        object_confs: Tensor,
        heatmap_scores: Tensor,
    ) -> Tensor:
        ax1, ay1, ax2, ay2 = actor_boxes.unbind(dim=-1)
        ox1, oy1, ox2, oy2 = object_boxes.unbind(dim=-1)

        aw = (ax2 - ax1).clamp_min(1.0e-4)
        ah = (ay2 - ay1).clamp_min(1.0e-4)
        acx = (ax1 + ax2) * 0.5
        acy = (ay1 + ay2) * 0.5
        aarea = (aw * ah).clamp_min(1.0e-4)

        ow = (ox2 - ox1).clamp_min(1.0e-4)
        oh = (oy2 - oy1).clamp_min(1.0e-4)
        ocx = (ox1 + ox2) * 0.5
        ocy = (oy1 + oy2) * 0.5
        oarea = (ow * oh).clamp_min(1.0e-4)

        dx = (ocx[:, None, :] - acx[:, :, None]) / aw[:, :, None]
        dy = (ocy[:, None, :] - acy[:, :, None]) / ah[:, :, None]
        wr = ow[:, None, :] / aw[:, :, None]
        hr = oh[:, None, :] / ah[:, :, None]
        area_ratio = oarea[:, None, :] / aarea[:, :, None]

        ix1 = torch.maximum(ax1[:, :, None], ox1[:, None, :])
        iy1 = torch.maximum(ay1[:, :, None], oy1[:, None, :])
        ix2 = torch.minimum(ax2[:, :, None], ox2[:, None, :])
        iy2 = torch.minimum(ay2[:, :, None], oy2[:, None, :])
        iw = (ix2 - ix1).clamp_min(0.0)
        ih = (iy2 - iy1).clamp_min(0.0)
        inter = iw * ih
        union = aarea[:, :, None] + oarea[:, None, :] - inter
        iou = inter / union.clamp_min(1.0e-4)
        obj_inside_actor = inter / oarea[:, None, :].clamp_min(1.0e-4)
        dist2 = dx * dx + dy * dy
        conf = object_confs[:, None, :].expand_as(dx)

        return torch.stack(
            (
                dx.clamp(-4.0, 4.0),
                dy.clamp(-4.0, 4.0),
                wr.clamp(0.0, 8.0),
                hr.clamp(0.0, 8.0),
                area_ratio.clamp(0.0, 8.0),
                iou.clamp(0.0, 1.0),
                obj_inside_actor.clamp(0.0, 1.0),
                dist2.clamp(0.0, 16.0),
                conf.clamp(0.0, 1.0),
                heatmap_scores.clamp(0.0, 1.0),
                ax1[:, :, None].expand_as(dx),
                ay1[:, :, None].expand_as(dx),
                aarea[:, :, None].expand_as(dx).clamp(0.0, 1.0),
            ),
            dim=-1,
        )

    def _compat_for_objects(self, object_classes: Tensor) -> Tensor:
        safe_classes = object_classes.clamp(0, self.num_object_classes - 1).long()
        return F.embedding(safe_classes, self.compat_by_object_action)

    def _build_relation_slots(
        self,
        actor_tokens: Tensor,
        actor_boxes: Tensor,
        object_boxes: Tensor,
        object_classes: Tensor,
        object_confs: Tensor,
        object_valid_f: Tensor,
        object_heatmap_scores: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        batch, actors, dim = actor_tokens.shape
        objects = object_boxes.shape[1]

        safe_classes = object_classes.clamp(0, self.num_object_classes - 1).long()
        class_token = self.object_class_embed(safe_classes)
        object_meta = self._box_features(object_boxes, object_confs)
        object_token = self.object_norm(class_token + self.object_meta_mlp(object_meta))
        object_token = object_token * object_valid_f[:, :, None]

        actor_exp = actor_tokens[:, :, None, :].expand(-1, -1, objects, -1)
        object_exp = object_token[:, None, :, :].expand(-1, actors, -1, -1)
        geom = self.pair_geom_mlp(
            self._pair_geometry(
                actor_boxes,
                object_boxes,
                object_confs,
                object_heatmap_scores,
            )
        )
        relation_input = torch.cat(
            (
                actor_exp,
                object_exp,
                actor_exp * object_exp,
                actor_exp - object_exp,
                geom,
            ),
            dim=-1,
        )
        object_slots = self.relation_mlp(relation_input)
        object_slots = object_slots * object_valid_f[:, None, :, None]

        null_seed = self.null_slot.view(1, 1, dim).expand(batch, actors, -1)
        unknown_seed = self.unknown_slot.view(1, 1, dim).expand(batch, actors, -1)
        null_slot = self.null_mlp(torch.cat((actor_tokens, null_seed), dim=-1))
        unknown_slot = self.unknown_mlp(
            torch.cat((actor_tokens, unknown_seed), dim=-1)
        )

        slots = torch.cat(
            (
                null_slot.unsqueeze(2),
                unknown_slot.unsqueeze(2),
                object_slots,
            ),
            dim=2,
        )
        quality_logits = self.quality_head(object_slots).squeeze(-1)
        quality = torch.sigmoid(quality_logits)
        quality = quality * object_valid_f[:, None, :] * object_confs[:, None, :]
        quality = quality * (1.0 + object_heatmap_scores).clamp(1.0, 2.0)
        quality = quality.clamp(0.0, 1.0)
        return self.relation_norm(slots), quality_logits, quality

    def _slot_bias(
        self,
        object_classes: Tensor,
        object_confs: Tensor,
        object_valid_f: Tensor,
        quality: Tensor,
        dtype: torch.dtype,
    ) -> Tuple[Tensor, Tensor]:
        batch, objects = object_classes.shape
        actors = quality.shape[1]
        device = object_classes.device
        objectless = self.objectless_action.to(device=device, dtype=dtype)
        objectful = self.objectful_action.to(device=device, dtype=dtype)
        has_known = self.has_known_object.to(device=device, dtype=dtype)

        bias = torch.zeros(
            batch,
            actors,
            self.num_actions,
            objects + 2,
            device=device,
            dtype=dtype,
        )

        null_mask = (1.0 - objectless).view(1, 1, self.num_actions) * self.neg_inf
        unknown_mask = (1.0 - objectful).view(1, 1, self.num_actions) * self.neg_inf
        bias[..., 0] = bias[..., 0] + null_mask
        bias[..., 1] = bias[..., 1] + unknown_mask + objectful.view(
            1,
            1,
            self.num_actions,
        ) * self.unknown_bias

        compat_bkc = self._compat_for_objects(object_classes).to(dtype=dtype)
        compat_prior = compat_bkc * self.compatible_bias
        compat_prior = compat_prior + (
            (1.0 - compat_bkc)
            * has_known.view(1, 1, self.num_actions)
            * self.incompatible_bias
        )
        compat_prior = compat_prior[:, None, :, :] * quality.detach()[:, :, :, None]
        compat_prior = compat_prior * object_confs[:, None, :, None]
        object_prior = compat_prior.permute(0, 1, 3, 2)

        invalid_objects = (1.0 - object_valid_f)[:, None, None, :] * self.neg_inf
        objectless_object_mask = objectless.view(1, 1, self.num_actions, 1) * self.neg_inf
        bias[..., 2:] = bias[..., 2:] + object_prior
        bias[..., 2:] = bias[..., 2:] + invalid_objects + objectless_object_mask

        compat_bck = compat_bkc.permute(0, 2, 1)
        any_quality = quality.amax(dim=-1, keepdim=True)
        compatible_quality = (
            quality[:, :, None, :] * compat_bck[:, None, :, :]
        ).amax(dim=-1)
        mismatch = (any_quality - compatible_quality).clamp_min(0.0).clamp_max(1.0)
        bias[..., 1] = bias[..., 1] - (
            self.unknown_mismatch_penalty
            * mismatch
            * has_known.view(1, 1, self.num_actions)
        )
        return bias, mismatch

    def forward(
        self,
        actor_tokens: Tensor,
        actor_boxes: Tensor,
        actor_valid: Optional[Tensor],
        object_boxes: Tensor,
        object_classes: Tensor,
        object_confs: Tensor,
        object_valid: Tensor,
        object_heatmap_scores: Tensor,
        motion_logits: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        if actor_tokens.ndim != 3:
            raise ValueError("actor_tokens must have shape [B,A,D]")
        batch, actors, dim = actor_tokens.shape
        if dim != self.dim:
            raise ValueError(f"actor_tokens dim must be {self.dim}, got {dim}")
        if actor_boxes.shape[:2] != (batch, actors):
            raise ValueError("actor_boxes must have shape [B,A,4]")
        if object_boxes.shape[:2] != object_classes.shape[:2]:
            raise ValueError("object_boxes/object_classes shape mismatch")

        dtype = actor_tokens.dtype
        device = actor_tokens.device
        if actor_valid is None:
            actor_valid = torch.ones(batch, actors, device=device, dtype=torch.bool)
        actor_valid = actor_valid.to(device=device, dtype=torch.bool)

        object_classes = object_classes.to(device=device)
        class_valid = (object_classes >= 0) & (object_classes < self.num_object_classes)
        object_valid_f = object_valid.to(device=device, dtype=dtype)
        object_valid_f = object_valid_f * class_valid.to(dtype=dtype)
        object_boxes = object_boxes.to(device=device, dtype=dtype)
        actor_boxes = actor_boxes.to(device=device, dtype=dtype)
        object_confs = object_confs.to(device=device, dtype=dtype) * object_valid_f
        object_heatmap_scores = object_heatmap_scores.to(device=device, dtype=dtype)
        object_heatmap_scores = object_heatmap_scores * object_valid_f[:, None, :]

        slots, quality_logits, quality = self._build_relation_slots(
            actor_tokens,
            actor_boxes,
            object_boxes,
            object_classes,
            object_confs,
            object_valid_f,
            object_heatmap_scores,
        )
        slot_keys = self.slot_key(slots)
        action_queries = self.action_query(self.action_embed)
        slot_scores = torch.einsum(
            "cr,basr->bacs",
            action_queries,
            slot_keys,
        ) / math.sqrt(float(self.attn_dim))
        bias, mismatch = self._slot_bias(
            object_classes,
            object_confs,
            object_valid_f,
            quality,
            dtype,
        )
        slot_scores = slot_scores + bias
        slot_posterior = torch.softmax(slot_scores.float(), dim=-1).to(dtype=dtype)

        slot_values = self.slot_value(slots)
        relation_summary = torch.einsum(
            "bacs,basd->bacd",
            slot_posterior,
            slot_values,
        )
        actor_term = self.actor_proj(actor_tokens)[:, :, None, :]
        relation_term = self.relation_proj(relation_summary)
        action_term = self.action_proj(self.action_embed).view(
            1,
            1,
            self.num_actions,
            dim,
        )
        relation_logits = self.logit_head(
            torch.tanh(actor_term + relation_term + action_term)
        ).squeeze(-1)
        relation_logits = torch.tanh(relation_logits)
        relation_scale = torch.sigmoid(self.relation_logit_scale).to(dtype=dtype)
        if motion_logits is not None:
            expected_shape = (batch, actors, self.num_actions)
            if tuple(motion_logits.shape) != expected_shape:
                raise ValueError(
                    "motion_logits must have shape "
                    f"{expected_shape}, got {tuple(motion_logits.shape)}"
                )
            motion_logits = motion_logits.to(device=device, dtype=dtype)
            logits = motion_logits + relation_scale * relation_logits
        else:
            logits = relation_scale * relation_logits
        invalid_actor = (~actor_valid).to(dtype=dtype) * self.neg_inf
        logits = logits + invalid_actor[:, :, None]
        if motion_logits is not None:
            motion_logits = motion_logits + invalid_actor[:, :, None]

        return {
            "logits": logits,
            "motion_logits": motion_logits,
            "relation_logits": relation_logits,
            "relation_logit_scale": relation_scale,
            "slot_delta": slot_scores,
            "slot_posterior": slot_posterior,
            "best_slot": slot_posterior.argmax(dim=-1),
            "relation_slots": slots,
            "object_quality": quality,
            "object_quality_logits": quality_logits,
            "mismatch": mismatch,
            "unknown_delta": slot_scores[..., 1],
            "object_slot_delta": slot_scores[..., 2:],
        }


def action_slot_target_loss(
    slot_delta: Tensor,
    labels: Tensor,
    object_classes: Tensor,
    object_valid: Tensor,
    spec: ActionObjectQuerySpec,
    valid: Optional[Tensor] = None,
    interaction_object_index: Optional[Tensor] = None,
    interaction_object_index_valid: Optional[Tensor] = None,
    ignore_missing_object: bool = False,
) -> Tensor:
    """Supervise p(slot | true_action, actor, video)."""

    batch, actors, num_actions, slots = slot_delta.shape
    num_object_slots = slots - 2
    device = slot_delta.device
    labels = labels.to(device=device, dtype=torch.long).clamp(0, num_actions - 1)
    object_classes = object_classes.to(device=device, dtype=torch.long)
    object_valid = object_valid.to(device=device, dtype=torch.bool)
    if interaction_object_index is not None:
        interaction_object_index = interaction_object_index.to(
            device=device,
            dtype=torch.long,
        )
    if interaction_object_index_valid is not None:
        interaction_object_index_valid = interaction_object_index_valid.to(
            device=device,
            dtype=torch.bool,
        )

    rows = torch.arange(batch, device=device)[:, None].expand(batch, actors)
    actor_idx = torch.arange(actors, device=device)[None, :].expand(batch, actors)
    true_slot_logits = slot_delta[rows, actor_idx, labels]

    target = torch.zeros(
        (batch, actors, num_object_slots + 2),
        device=device,
        dtype=slot_delta.dtype,
    )
    valid_loss = torch.ones((batch, actors), device=device, dtype=torch.bool)
    if valid is not None:
        valid_loss &= valid.to(device=device, dtype=torch.bool)

    objectless_set = {int(x) for x in spec.objectless_action_indices}
    mapping = {
        int(key): tuple(int(v) for v in values)
        for key, values in spec.action_to_object_ids.items()
    }

    for bi in range(batch):
        for ai in range(actors):
            if not bool(valid_loss[bi, ai].item()):
                continue
            label = int(labels[bi, ai].item())
            if label in objectless_set:
                target[bi, ai, 0] = 1.0
                continue
            allowed = mapping.get(label, tuple())
            if not allowed:
                target[bi, ai, 1] = 1.0
                continue
            if (
                interaction_object_index is not None
                and interaction_object_index_valid is not None
                and bool(interaction_object_index_valid[bi, ai].item())
            ):
                # Dataset convention: 0=NULL, object proposals are 1..K.
                # Decoder convention: 0=NULL, 1=UNKNOWN, objects are 2..K+1.
                object_slot = int(interaction_object_index[bi, ai].item()) - 1
                if 0 <= object_slot < num_object_slots and bool(
                    object_valid[bi, object_slot].item()
                ):
                    object_class = int(object_classes[bi, object_slot].item())
                    if object_class in allowed:
                        target[bi, ai, object_slot + 2] = 1.0
                    else:
                        # The teacher says this proposal is the interaction
                        # object, but the detector class conflicts with the
                        # action taxonomy.  Preserve some geometry supervision
                        # while making UNKNOWN the primary explanation.
                        target[bi, ai, 1] = 0.7
                        target[bi, ai, object_slot + 2] = 0.3
                    continue
                target[bi, ai, 1] = 1.0
                continue

            match = torch.zeros(num_object_slots, device=device, dtype=torch.bool)
            for object_id in allowed:
                match |= (object_classes[bi] == int(object_id)) & object_valid[bi]
            if bool(match.any().item()):
                weights = match.to(dtype=slot_delta.dtype)
                target[bi, ai, 2:] = weights / weights.sum().clamp_min(1.0)
            elif ignore_missing_object:
                valid_loss[bi, ai] = False
            else:
                target[bi, ai, 1] = 1.0

    logp = F.log_softmax(true_slot_logits.float(), dim=-1).to(dtype=slot_delta.dtype)
    loss = -(target * logp).sum(dim=-1)
    if valid_loss.any():
        return loss[valid_loss].mean()
    return loss.sum() * 0.0


if __name__ == "__main__":
    torch.manual_seed(0)
    batch, actors, objects, dim, actions, object_classes = 2, 3, 6, 128, 8, 5
    spec = ActionObjectQuerySpec(
        num_actions=actions,
        num_object_classes=object_classes,
        objectless_action_indices=(0, 1),
        action_to_object_ids={2: (1,), 3: (2,)},
    )
    head = ActorObjectActionQueryDecoder(dim, spec)
    actor = torch.randn(batch, actors, dim)
    actor_boxes = torch.rand(batch, actors, 4)
    actor_boxes[..., 2:] = torch.maximum(actor_boxes[..., :2] + 0.05, actor_boxes[..., 2:])
    actor_boxes = actor_boxes.clamp(0, 1)
    obj_boxes = torch.rand(batch, objects, 4)
    obj_boxes[..., 2:] = torch.maximum(obj_boxes[..., :2] + 0.05, obj_boxes[..., 2:])
    obj_boxes = obj_boxes.clamp(0, 1)
    obj_classes = torch.randint(0, object_classes, (batch, objects))
    obj_confs = torch.rand(batch, objects)
    obj_valid = torch.ones(batch, objects, dtype=torch.bool)
    heatmap_scores = torch.zeros(batch, actors, objects)
    actor_valid = torch.ones(batch, actors, dtype=torch.bool)
    motion = torch.randn(batch, actors, actions)
    out = head(
        actor_tokens=actor,
        actor_boxes=actor_boxes,
        actor_valid=actor_valid,
        object_boxes=obj_boxes,
        object_classes=obj_classes,
        object_confs=obj_confs,
        object_valid=obj_valid,
        object_heatmap_scores=heatmap_scores,
        motion_logits=motion,
    )
    max_motion_delta = (out["logits"] - motion).abs().max().item()
    assert max_motion_delta < 1.0e-6, max_motion_delta
    loss = action_slot_target_loss(
        out["slot_delta"],
        torch.randint(0, actions, (batch, actors)),
        obj_classes,
        obj_valid,
        spec,
        valid=actor_valid,
    )
    print(
        out["logits"].shape,
        out["slot_delta"].shape,
        float(loss.detach()),
        max_motion_delta,
    )
