"""TensorRT-friendly actor-object explanation slot action head.

This head is designed for PO-GUISE/PO-GUISE+ actor tokens plus a fixed
top-K set of detector object proposals.  It avoids ROIAlign, dynamic loops in
forward, and a global selected-object bottleneck.

The explanation slots are:
    NULL: no object interaction; only objectless actions may use it.
    UNKNOWN: object interaction exists but detector/object class is missing.
    OBJECT_j: actor interacting with detector proposal j.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ActionObjectSlotSpec:
    """Static action/object taxonomy for the slot head."""

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

        objectful = (1.0 - objectless).clamp(0.0, 1.0)
        return {
            "objectless_action": objectless,
            "objectful_action": objectful,
            "has_known_object": has_known_object,
            "compat_by_object_action": compat,
        }


class TRTFriendlyActorObjectSlotHead(nn.Module):
    """Joint actor-object action head with NULL/UNKNOWN/object slots."""

    def __init__(
        self,
        dim: int,
        spec: ActionObjectSlotSpec,
        hidden_dim: int = 256,
        relation_rank: int = 64,
        prior_compatible: float = 1.25,
        prior_incompatible: float = -1.25,
        unknown_init_bias: float = -0.25,
        unknown_mismatch_penalty: float = 1.0,
        relation_scale: float = 1.0,
        quality_scale: float = 0.5,
        neg_inf: float = -1.0e4,
        use_hard_incompatible_mask: bool = False,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_actions = int(spec.num_actions)
        self.num_object_classes = int(spec.num_object_classes)
        self.hidden_dim = int(hidden_dim)
        self.relation_rank = int(relation_rank)
        self.prior_compatible = float(prior_compatible)
        self.prior_incompatible = float(prior_incompatible)
        self.unknown_mismatch_penalty = float(unknown_mismatch_penalty)
        self.relation_scale = float(relation_scale)
        self.quality_scale = float(quality_scale)
        self.neg_inf = float(neg_inf)
        self.use_hard_incompatible_mask = bool(use_hard_incompatible_mask)

        for name, tensor in spec.build_buffers().items():
            self.register_buffer(name, tensor, persistent=False)

        self.object_class_embed = nn.Embedding(self.num_object_classes, hidden_dim)
        self.object_meta_mlp = nn.Sequential(
            nn.Linear(10, hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.object_norm = nn.LayerNorm(hidden_dim)

        self.actor_proj = nn.Linear(dim, hidden_dim)
        self.object_proj = nn.Linear(hidden_dim, hidden_dim)
        self.pair_geom_mlp = nn.Sequential(
            nn.Linear(13, hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.relation_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=False),
        )

        self.relation_to_basis = nn.Linear(hidden_dim, relation_rank)
        self.action_basis = nn.Parameter(torch.empty(self.num_actions, relation_rank))
        self.action_bias = nn.Parameter(torch.zeros(self.num_actions))

        self.quality_head = nn.Linear(hidden_dim, 1)

        self.unknown_actor_mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, relation_rank),
        )
        self.unknown_action_basis = nn.Parameter(torch.empty(self.num_actions, relation_rank))
        self.unknown_bias = nn.Parameter(torch.full((self.num_actions,), float(unknown_init_bias)))

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.action_basis, std=0.02)
        nn.init.normal_(self.unknown_action_basis, std=0.02)
        nn.init.zeros_(self.relation_to_basis.weight)
        nn.init.zeros_(self.relation_to_basis.bias)
        nn.init.zeros_(self.quality_head.weight)
        nn.init.constant_(self.quality_head.bias, -1.0)
        last = self.unknown_actor_mlp[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

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

    def forward(
        self,
        actor_tokens: Tensor,
        motion_logits: Tensor,
        actor_boxes: Tensor,
        object_boxes: Tensor,
        object_classes: Tensor,
        object_confs: Tensor,
        object_valid: Tensor,
        object_heatmap_scores: Tensor,
    ) -> Dict[str, Tensor]:
        if actor_tokens.ndim != 3:
            raise ValueError("actor_tokens must have shape [B,A,D]")
        if motion_logits.shape[:2] != actor_tokens.shape[:2] or motion_logits.shape[-1] != self.num_actions:
            raise ValueError("motion_logits must have shape [B,A,num_actions]")

        batch, actors, _ = actor_tokens.shape
        dtype = actor_tokens.dtype
        device = actor_tokens.device

        object_classes = object_classes.to(device=device)
        class_valid = (object_classes >= 0) & (object_classes < self.num_object_classes)
        object_valid_f = object_valid.to(device=device, dtype=dtype) * class_valid.to(dtype=dtype)
        object_confs = object_confs.to(device=device, dtype=dtype) * object_valid_f
        object_boxes = object_boxes.to(device=device, dtype=dtype)
        actor_boxes = actor_boxes.to(device=device, dtype=dtype)
        object_heatmap_scores = object_heatmap_scores.to(device=device, dtype=dtype) * object_valid_f[:, None, :]

        safe_classes = object_classes.clamp(0, self.num_object_classes - 1).long()
        cls_token = self.object_class_embed(safe_classes)
        meta = self._box_features(object_boxes, object_confs)
        obj_token = self.object_norm(cls_token + self.object_meta_mlp(meta))
        obj_token = obj_token * object_valid_f[:, :, None]

        actor_h = self.actor_proj(actor_tokens)[:, :, None, :]
        object_h = self.object_proj(obj_token)[:, None, :, :]
        pair_geom = self._pair_geometry(actor_boxes, object_boxes, object_confs, object_heatmap_scores)
        geom_h = self.pair_geom_mlp(pair_geom)
        relation = self.relation_mlp(actor_h + object_h + geom_h)

        basis = self.relation_to_basis(relation)
        raw_object_delta = torch.einsum("bakr,cr->back", basis, self.action_basis)
        object_delta = torch.tanh(raw_object_delta) * self.relation_scale

        quality = torch.sigmoid(self.quality_head(relation).squeeze(-1))
        quality = quality * object_valid_f[:, None, :] * object_confs[:, None, :]
        quality = quality * (1.0 + object_heatmap_scores).clamp(1.0, 2.0)
        quality = quality.clamp(0.0, 1.0)

        compat_bkc = self._compat_for_objects(object_classes).to(dtype=dtype)
        known_action = self.has_known_object.to(device=device, dtype=dtype).view(1, 1, -1)
        compat_prior_bkc = compat_bkc * self.prior_compatible
        compat_prior_bkc = compat_prior_bkc + (1.0 - compat_bkc) * known_action * self.prior_incompatible
        compat_prior_bkc = compat_prior_bkc * object_confs[:, :, None]
        compat_prior = compat_prior_bkc.permute(0, 2, 1)[:, None, :, :]

        object_slot_delta = object_delta + compat_prior + self.quality_scale * quality[:, :, None, :]
        object_slot_delta = object_slot_delta + (1.0 - object_valid_f)[:, None, None, :] * self.neg_inf

        objectless = self.objectless_action.to(device=device, dtype=dtype).view(1, 1, self.num_actions, 1)
        object_slot_delta = object_slot_delta + objectless * self.neg_inf

        if self.use_hard_incompatible_mask:
            hard_bad = (1.0 - compat_bkc).permute(0, 2, 1)[:, None, :, :] * known_action[:, :, :, None]
            object_slot_delta = object_slot_delta + hard_bad * self.neg_inf

        unknown_basis = self.unknown_actor_mlp(actor_tokens)
        unknown_delta = torch.einsum("bar,cr->bac", unknown_basis, self.unknown_action_basis)
        unknown_delta = torch.tanh(unknown_delta) * self.relation_scale + self.unknown_bias.view(1, 1, -1)

        compat_bck = compat_bkc.permute(0, 2, 1)
        any_quality = quality.amax(dim=-1, keepdim=True)
        compatible_quality = (quality[:, :, None, :] * compat_bck[:, None, :, :]).amax(dim=-1)
        mismatch = (any_quality - compatible_quality).clamp_min(0.0).clamp_max(1.0)
        has_known = self.has_known_object.to(device=device, dtype=dtype).view(1, 1, -1)
        unknown_delta = unknown_delta - self.unknown_mismatch_penalty * mismatch * has_known

        objectful = self.objectful_action.to(device=device, dtype=dtype).view(1, 1, -1)
        unknown_delta = unknown_delta + (1.0 - objectful) * self.neg_inf

        null_delta = torch.zeros((batch, actors, self.num_actions), device=device, dtype=dtype)
        null_delta = null_delta + (1.0 - objectless.squeeze(-1)) * self.neg_inf

        slot_delta = torch.cat(
            (
                null_delta.unsqueeze(-1),
                unknown_delta.unsqueeze(-1),
                object_slot_delta,
            ),
            dim=-1,
        )

        best_delta, best_slot = slot_delta.max(dim=-1)
        final_logits = motion_logits + best_delta + self.action_bias.view(1, 1, -1)
        slot_posterior = torch.softmax(slot_delta.float(), dim=-1).to(dtype=dtype)

        return {
            "logits": final_logits,
            "motion_logits": motion_logits,
            "slot_delta": slot_delta,
            "slot_posterior": slot_posterior,
            "best_slot": best_slot,
            "object_quality": quality,
            "mismatch": mismatch,
            "unknown_delta": unknown_delta,
            "object_slot_delta": object_slot_delta,
        }


def action_slot_target_loss(
    slot_delta: Tensor,
    labels: Tensor,
    object_classes: Tensor,
    object_valid: Tensor,
    spec: ActionObjectSlotSpec,
    valid: Optional[Tensor] = None,
    ignore_missing_object: bool = False,
) -> Tensor:
    """Supervise the explanation slot for the true action."""

    batch, actors, _, slots = slot_delta.shape
    num_object_slots = slots - 2
    device = slot_delta.device
    labels = labels.to(device=device, dtype=torch.long)
    object_classes = object_classes.to(device=device, dtype=torch.long)
    object_valid = object_valid.to(device=device, dtype=torch.bool)

    rows = torch.arange(batch, device=device)[:, None].expand(batch, actors)
    actor_idx = torch.arange(actors, device=device)[None, :].expand(batch, actors)
    true_slot_logits = slot_delta[rows, actor_idx, labels]

    target = torch.zeros((batch, actors, num_object_slots + 2), device=device, dtype=slot_delta.dtype)
    valid_loss = torch.ones((batch, actors), device=device, dtype=torch.bool)
    if valid is not None:
        valid_loss &= valid.to(device=device, dtype=torch.bool)

    objectless_set = {int(x) for x in spec.objectless_action_indices}
    mapping = {int(key): tuple(int(v) for v in values) for key, values in spec.action_to_object_ids.items()}

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


def objectless_hard_negative_loss(
    logits: Tensor,
    labels: Tensor,
    spec: ActionObjectSlotSpec,
    margin: float = 0.5,
) -> Tensor:
    """For objectless examples, keep objectful classes below the true class."""

    device = logits.device
    labels = labels.to(device=device, dtype=torch.long)
    objectless = torch.zeros(spec.num_actions, device=device, dtype=torch.bool)
    if spec.objectless_action_indices:
        objectless[list(spec.objectless_action_indices)] = True
    objectful = ~objectless

    sample_objectless = objectless[labels]
    if not sample_objectless.any():
        return logits.sum() * 0.0
    true_logits = logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    max_objectful = logits[..., objectful].amax(dim=-1)
    return F.relu(float(margin) - (true_logits - max_objectful))[sample_objectless].mean()


def confuser_margin_loss(
    logits: Tensor,
    labels: Tensor,
    confusers_by_action: Mapping[int, Sequence[int]],
    margin: float = 0.75,
) -> Tensor:
    """Margin loss for visually similar object actions."""

    device = logits.device
    labels = labels.to(device=device, dtype=torch.long)
    losses = []
    for action_idx, confusers in confusers_by_action.items():
        action_idx = int(action_idx)
        confusers = [int(confuser) for confuser in confusers]
        if not confusers:
            continue
        mask = labels == action_idx
        if not mask.any():
            continue
        pos = logits[..., action_idx][mask]
        neg = logits[..., confusers][mask].amax(dim=-1)
        losses.append(F.relu(float(margin) - (pos - neg)))
    if not losses:
        return logits.sum() * 0.0
    return torch.cat(losses).mean()


if __name__ == "__main__":
    torch.manual_seed(0)
    batch, actors, objects, dim, actions, object_classes = 2, 1, 6, 128, 8, 5
    spec = ActionObjectSlotSpec(
        num_actions=actions,
        num_object_classes=object_classes,
        objectless_action_indices=(0, 1),
        action_to_object_ids={2: (1,), 3: (2,)},
    )
    head = TRTFriendlyActorObjectSlotHead(dim, spec)
    actor = torch.randn(batch, actors, dim)
    motion = torch.randn(batch, actors, actions)
    actor_boxes = torch.tensor(
        [[[0.2, 0.2, 0.8, 0.9]], [[0.1, 0.1, 0.7, 0.8]]],
        dtype=torch.float32,
    )
    obj_boxes = torch.rand(batch, objects, 4)
    obj_boxes[..., 2:] = torch.maximum(obj_boxes[..., :2] + 0.05, obj_boxes[..., 2:])
    obj_boxes = obj_boxes.clamp(0, 1)
    obj_classes = torch.randint(0, object_classes, (batch, objects))
    obj_confs = torch.rand(batch, objects)
    obj_valid = torch.ones(batch, objects, dtype=torch.bool)
    heatmap_scores = torch.zeros(batch, actors, objects)
    out = head(actor, motion, actor_boxes, obj_boxes, obj_classes, obj_confs, obj_valid, heatmap_scores)
    print(out["logits"].shape, out["slot_delta"].shape, out["best_slot"].shape)
