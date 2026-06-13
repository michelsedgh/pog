"""
Factorized interaction-action head for PO-GUISE actor prompts.

Core idea:
    p(action | actor, video, objects) =
        p(NULL-mode | actor) * p(objectless_action | NULL, actor)
      + p(INTERACTION-mode | actor) * p(objectful_action | actor, object evidence)

Objectful evidence is itself a detected/missing mixture per action:
    score_objectful[a] = motion_obj[a] + logsumexp(
        log(coverage[a])     + detected_delta[a],
        log(1-coverage[a])   + visual_missing_delta[a]
    )

This prevents detector-missing fallback losses from directly pushing objectful logits over
objectless actions. It also keeps detector object slots useful when compatible objects exist.

TensorRT-friendly ops: Linear, Embedding, LayerNorm, GELU, einsum/matmul, softmax/log_softmax,
elementwise ops, max/reduce.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class FactorizedActionObjectSpec:
    num_actions: int
    num_object_classes: int
    objectless_action_indices: Tuple[int, ...]
    # shape [num_object_classes, num_actions], 1 iff object class can explain action
    compat_matrix: Tensor
    # optional confuser action ids for objectful actions; used only by losses
    confusers_by_action: Optional[Mapping[int, Tuple[int, ...]]] = None

    def __post_init__(self) -> None:
        if self.num_actions <= 0:
            raise ValueError("num_actions must be positive")
        if self.num_object_classes <= 0:
            raise ValueError("num_object_classes must be positive")
        expected_shape = (int(self.num_object_classes), int(self.num_actions))
        if tuple(self.compat_matrix.shape) != expected_shape:
            raise ValueError(
                "compat_matrix must have shape "
                f"{expected_shape}, got {tuple(self.compat_matrix.shape)}"
            )
        objectless = tuple(sorted(set(int(i) for i in self.objectless_action_indices)))
        if any(i < 0 or i >= self.num_actions for i in objectless):
            raise ValueError("objectless_action_indices contains an invalid action id")
        if len(objectless) == 0:
            raise ValueError("factorized head requires at least one objectless action")
        if len(objectless) == self.num_actions:
            raise ValueError("factorized head requires at least one objectful action")
        object.__setattr__(self, "objectless_action_indices", objectless)

    @property
    def objectful_action_indices(self) -> Tuple[int, ...]:
        objectless = set(int(i) for i in self.objectless_action_indices)
        return tuple(i for i in range(self.num_actions) if i not in objectless)


def _gather_indices(x: Tensor, indices: Sequence[int]) -> Tensor:
    return x.index_select(-1, torch.as_tensor(indices, device=x.device, dtype=torch.long))


def _scatter_last_dim(
    values: Tensor,
    indices: Tensor,
    size: int,
    fill_value: float = -1.0e4,
) -> Tensor:
    output = values.new_full((*values.shape[:-1], int(size)), float(fill_value))
    output.index_copy_(-1, indices.to(device=values.device), values)
    return output


class GeometryEncoder(nn.Module):
    """Actor-object relative geometry encoder."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(12, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    @staticmethod
    def features(actor_boxes: Tensor, object_boxes: Tensor) -> Tensor:
        """Return [B,A,K,12] geometry features. Boxes are xyxy normalized 0..1."""
        ab = actor_boxes[:, :, None, :]  # [B,A,1,4]
        ob = object_boxes[:, None, :, :]  # [B,1,K,4]

        ax1, ay1, ax2, ay2 = ab.unbind(-1)
        ox1, oy1, ox2, oy2 = ob.unbind(-1)
        aw = (ax2 - ax1).clamp_min(1e-4)
        ah = (ay2 - ay1).clamp_min(1e-4)
        ow = (ox2 - ox1).clamp_min(1e-4)
        oh = (oy2 - oy1).clamp_min(1e-4)
        acx = 0.5 * (ax1 + ax2)
        acy = 0.5 * (ay1 + ay2)
        ocx = 0.5 * (ox1 + ox2)
        ocy = 0.5 * (oy1 + oy2)

        inter_x1 = torch.maximum(ax1, ox1)
        inter_y1 = torch.maximum(ay1, oy1)
        inter_x2 = torch.minimum(ax2, ox2)
        inter_y2 = torch.minimum(ay2, oy2)
        inter = (inter_x2 - inter_x1).clamp_min(0) * (inter_y2 - inter_y1).clamp_min(0)
        area_a = aw * ah
        area_o = ow * oh
        union = (area_a + area_o - inter).clamp_min(1e-6)
        iou = inter / union

        dx = (ocx - acx) / aw
        dy = (ocy - acy) / ah
        dist = torch.sqrt(dx * dx + dy * dy + 1e-6)
        area_ratio = area_o / area_a.clamp_min(1e-6)

        target_shape = dx.shape
        def ex(t: Tensor) -> Tensor:
            return t.expand(target_shape)

        return torch.stack([
            dx, dy, dist, iou,
            ex(ow), ex(oh), ex(area_o),
            ex(aw), ex(ah), ex(area_a),
            area_ratio,
            ((ocx >= ax1).to(ab.dtype) * (ocx <= ax2).to(ab.dtype) * (ocy >= ay1).to(ab.dtype) * (ocy <= ay2).to(ab.dtype)).expand(target_shape),
        ], dim=-1)

    def forward(self, actor_boxes: Tensor, object_boxes: Tensor) -> Tensor:
        return self.net(self.features(actor_boxes, object_boxes))


class DetectedObjectRelationHead(nn.Module):
    """Action-conditioned detected-object relation scoring."""

    def __init__(self, dim: int, spec: FactorizedActionObjectSpec, hidden_dim: int = 512, attn_dim: int = 256):
        super().__init__()
        self.spec = spec
        self.object_embed = nn.Embedding(spec.num_object_classes, dim)
        self.object_box_mlp = nn.Sequential(nn.Linear(5, dim), nn.GELU(), nn.Linear(dim, dim))
        self.geometry = GeometryEncoder(hidden_dim)
        self.relation_mlp = nn.Sequential(
            nn.LayerNorm(dim * 3 + hidden_dim + 1),
            nn.Linear(dim * 3 + hidden_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.LayerNorm(dim),
        )
        self.action_query = nn.Parameter(torch.randn(spec.num_actions, attn_dim) * 0.02)
        self.rel_proj = nn.Linear(dim, attn_dim, bias=False)
        self.quality_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))

        compat = spec.compat_matrix.float()
        self.register_buffer("compat_matrix", compat, persistent=False)

    def encode_objects(self, object_boxes: Tensor, object_classes: Tensor, object_confs: Tensor) -> Tensor:
        # object boxes normalized xyxy, conf [B,K]
        box_conf = torch.cat([object_boxes, object_confs[..., None]], dim=-1)
        safe_classes = object_classes.clamp(0, self.spec.num_object_classes - 1)
        class_valid = (object_classes >= 0) & (
            object_classes < self.spec.num_object_classes
        )
        class_token = self.object_embed(safe_classes.long())
        class_token = class_token * class_valid.to(dtype=class_token.dtype).unsqueeze(-1)
        return class_token + self.object_box_mlp(box_conf)

    def forward(
        self,
        actor_tokens: Tensor,             # [B,A,D]
        actor_boxes: Tensor,              # [B,A,4]
        object_boxes: Tensor,             # [B,K,4]
        object_classes: Tensor,           # [B,K]
        object_confs: Tensor,             # [B,K]
        object_valid: Tensor,             # [B,K] bool
        relation_logit_bound: float = 2.0,
        eps: float = 1e-4,
    ) -> Dict[str, Tensor]:
        B, A, D = actor_tokens.shape
        K = object_boxes.shape[1]
        if K <= 0:
            raise ValueError("factorized interaction head requires object proposal slots")
        object_valid = object_valid.to(device=actor_tokens.device, dtype=torch.bool)
        object_classes = object_classes.to(device=actor_tokens.device, dtype=torch.long)
        object_boxes = object_boxes.to(device=actor_tokens.device, dtype=actor_tokens.dtype)
        actor_boxes = actor_boxes.to(device=actor_tokens.device, dtype=actor_tokens.dtype)
        object_confs = (
            object_confs.to(device=actor_tokens.device, dtype=actor_tokens.dtype)
            * object_valid.to(device=actor_tokens.device, dtype=actor_tokens.dtype)
        )
        obj_tok = self.encode_objects(object_boxes, object_classes, object_confs)  # [B,K,D]
        geom = self.geometry(actor_boxes, object_boxes)  # [B,A,K,H]

        at = actor_tokens[:, :, None, :].expand(B, A, K, D)
        ot = obj_tok[:, None, :, :].expand(B, A, K, D)
        conf = object_confs[:, None, :, None].expand(B, A, K, 1)
        rel_in = torch.cat([at, ot, at * ot, geom, conf], dim=-1)
        rel = self.relation_mlp(rel_in)  # [B,A,K,D]

        q = self.rel_proj(rel)  # [B,A,K,R]
        # raw relation score for every action and object slot: [B,A,C,K]
        raw = torch.einsum("bakr,cr->back", q, self.action_query) / (q.shape[-1] ** 0.5)
        raw = relation_logit_bound * torch.tanh(raw / max(relation_logit_bound, 1e-6))

        valid = object_valid[:, None, None, :].to(raw.dtype)
        raw = raw.masked_fill(valid <= 0, -1e4)

        # posterior over detected object slots for each action
        slot_posterior = F.softmax(raw.float(), dim=-1).to(raw.dtype)
        detected_delta = torch.sum(slot_posterior * raw, dim=-1)  # [B,A,C]

        quality_logits = self.quality_head(rel).squeeze(-1)  # [B,A,K]
        quality = torch.sigmoid(quality_logits) * object_valid[:, None, :].to(quality_logits.dtype)

        safe_classes = object_classes.clamp(0, self.spec.num_object_classes - 1)
        class_valid = (object_classes >= 0) & (
            object_classes < self.spec.num_object_classes
        )
        compat_bkc = F.embedding(safe_classes, self.compat_matrix)
        compat_bkc = compat_bkc * (class_valid & object_valid).to(
            dtype=compat_bkc.dtype,
        ).unsqueeze(-1)
        compat_back = compat_bkc[:, None, :, :].permute(0, 1, 3, 2).expand(B, A, self.spec.num_actions, K)

        # action loss uses quality only as a gate; explicit quality BCE trains quality.
        quality_gate = quality.detach()
        proposal_strength = (
            quality_gate[:, :, None, :]
            * object_confs[:, None, None, :]
            * object_valid[:, None, None, :].to(quality.dtype)
        )
        coverage = (compat_back * proposal_strength).amax(dim=-1).clamp(0.0, 1.0)  # [B,A,C]

        # detected taxonomy prior is inside the objectful expert, not global action competition
        taxonomy_prior = torch.log1p(compat_back * proposal_strength / eps).amax(dim=-1)
        taxonomy_prior = taxonomy_prior.clamp(max=4.0)
        detected_delta = detected_delta + taxonomy_prior

        return {
            "detected_delta": detected_delta,
            "coverage": coverage,
            "slot_scores": raw,
            "slot_posterior": slot_posterior,
            "quality_logits": quality_logits,
            "quality": quality,
            "compat_back": compat_back,
        }


class VisualMissingFallbackHead(nn.Module):
    """Detector-free objectful fallback from heatmap/visual source tokens."""

    def __init__(self, dim: int, spec: FactorizedActionObjectSpec, hidden_dim: int = 512, attn_dim: int = 256):
        super().__init__()
        self.spec = spec
        self.q_seed = nn.Parameter(torch.randn(1, dim) * 0.02)
        self.q_proj = nn.Linear(dim, attn_dim, bias=False)
        self.k_proj = nn.Linear(dim, attn_dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.slot_norm = nn.LayerNorm(dim)
        self.action_query = nn.Parameter(torch.randn(spec.num_actions, attn_dim) * 0.02)
        self.slot_proj = nn.Linear(dim, attn_dim, bias=False)
        self.delta_mlp = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, dim))

    def forward(self, actor_tokens: Tensor, source_tokens: Tensor, source_valid: Optional[Tensor] = None, bound: float = 2.0) -> Dict[str, Tensor]:
        if source_tokens is None:
            raise RuntimeError("VisualMissingFallbackHead requires detector-free heatmap/visual source tokens.")
        B, A, D = actor_tokens.shape
        q_seed = actor_tokens + self.q_seed.view(1, 1, D)
        q = self.q_proj(q_seed)        # [B,A,R]
        k = self.k_proj(source_tokens) # [B,N,R]
        v = self.v_proj(source_tokens) # [B,N,D]
        scores = torch.einsum("bar,bnr->ban", q, k) / (q.shape[-1] ** 0.5)
        if source_valid is not None:
            scores = scores.masked_fill(~source_valid[:, None, :], -1e4)
        attn = F.softmax(scores.float(), dim=-1).to(v.dtype)
        slot = torch.einsum("ban,bnd->bad", attn, v)
        slot = self.slot_norm(slot + actor_tokens)
        slot = self.delta_mlp(slot)
        sr = self.slot_proj(slot)  # [B,A,R]
        raw = torch.einsum("bar,cr->bac", sr, self.action_query) / (sr.shape[-1] ** 0.5)
        delta = bound * torch.tanh(raw / max(bound, 1e-6))
        return {"visual_delta": delta, "visual_attn": attn, "visual_slot": slot}


class FactorizedInteractionActionHead(nn.Module):
    """Final head with explicit objectless/objectful arbitration."""

    def __init__(
        self,
        dim: int,
        spec: FactorizedActionObjectSpec,
        hidden_dim: int = 512,
        relation_logit_bound: float = 2.0,
        max_relation_scale: float = 1.5,
        relation_scale_init: float = -1.0,
    ):
        super().__init__()
        self.spec = spec
        self.objectless_idx = tuple(int(i) for i in spec.objectless_action_indices)
        self.objectful_idx = spec.objectful_action_indices
        self.register_buffer("objectless_index", torch.tensor(self.objectless_idx, dtype=torch.long), persistent=False)
        self.register_buffer("objectful_index", torch.tensor(self.objectful_idx, dtype=torch.long), persistent=False)

        self.presence_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 2))
        self.objectless_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, len(self.objectless_idx)))
        self.objectful_motion_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, len(self.objectful_idx)))
        self.motion_aux_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, spec.num_actions))

        self.detected = DetectedObjectRelationHead(dim, spec, hidden_dim=hidden_dim)
        self.visual = VisualMissingFallbackHead(dim, spec, hidden_dim=hidden_dim)
        self.relation_scale_logit = nn.Parameter(torch.tensor(float(relation_scale_init)))
        self.relation_logit_bound = float(relation_logit_bound)
        self.max_relation_scale = float(max_relation_scale)

    def forward(
        self,
        actor_tokens: Tensor,
        actor_boxes: Tensor,
        actor_valid: Tensor,
        object_boxes: Tensor,
        object_classes: Tensor,
        object_confs: Tensor,
        object_valid: Tensor,
        visual_source_tokens: Tensor,
        visual_source_valid: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        B, A, D = actor_tokens.shape
        actor_valid = actor_valid.to(device=actor_tokens.device, dtype=torch.bool)
        rel_scale = self.max_relation_scale * torch.sigmoid(self.relation_scale_logit)

        presence_logits = self.presence_head(actor_tokens)  # [B,A,2]
        presence_logp = F.log_softmax(presence_logits, dim=-1)
        objectless_logp = F.log_softmax(self.objectless_head(actor_tokens), dim=-1)  # [B,A,C0]
        objectful_motion = self.objectful_motion_head(actor_tokens)  # [B,A,C1]
        motion_aux_logits = self.motion_aux_head(actor_tokens)

        det = self.detected(
            actor_tokens, actor_boxes, object_boxes, object_classes, object_confs, object_valid,
            relation_logit_bound=self.relation_logit_bound,
        )
        vis = self.visual(actor_tokens, visual_source_tokens, visual_source_valid, bound=self.relation_logit_bound)

        det_delta_full = det["detected_delta"] * rel_scale
        vis_delta_full = vis["visual_delta"] * rel_scale
        coverage_full = det["coverage"].clamp(1e-4, 1.0 - 1e-4)

        det_delta = _gather_indices(det_delta_full, self.objectful_idx)
        vis_delta = _gather_indices(vis_delta_full, self.objectful_idx)
        coverage = _gather_indices(coverage_full, self.objectful_idx)

        # detected/missing evidence mixture INSIDE objectful branch only
        detected_path = torch.log(coverage) + det_delta
        missing_path = torch.log1p(-coverage) + vis_delta
        mix_paths = torch.stack([detected_path, missing_path], dim=-1)
        evidence_mix = torch.logsumexp(mix_paths, dim=-1)
        mix_weight = F.softmax(mix_paths.float(), dim=-1).to(objectful_motion.dtype)
        detected_mix_weight = mix_weight[..., 0]
        visual_mix_weight = mix_weight[..., 1]
        objectful_scores = objectful_motion + evidence_mix
        objectful_logp = F.log_softmax(objectful_scores, dim=-1)

        final_logp = actor_tokens.new_full((B, A, self.spec.num_actions), -1.0e4)
        final_logp.index_copy_(
            -1,
            self.objectless_index.to(actor_tokens.device),
            presence_logp[..., 0:1] + objectless_logp,
        )
        final_logp.index_copy_(
            -1,
            self.objectful_index.to(actor_tokens.device),
            presence_logp[..., 1:2] + objectful_logp,
        )

        objectful_scores_full = _scatter_last_dim(
            objectful_scores,
            self.objectful_index,
            self.spec.num_actions,
        )
        objectful_logp_full = _scatter_last_dim(
            objectful_logp,
            self.objectful_index,
            self.spec.num_actions,
        )
        visual_delta_objectful = _gather_indices(vis_delta_full, self.objectful_idx)
        visual_delta_objectful_full = _scatter_last_dim(
            visual_delta_objectful,
            self.objectful_index,
            self.spec.num_actions,
        )
        detected_mix_weight_objectful_full = _scatter_last_dim(
            detected_mix_weight,
            self.objectful_index,
            self.spec.num_actions,
            fill_value=0.0,
        )
        visual_mix_weight_objectful_full = _scatter_last_dim(
            visual_mix_weight,
            self.objectful_index,
            self.spec.num_actions,
            fill_value=0.0,
        )

        final_logp = final_logp.masked_fill(~actor_valid[:, :, None], -1e4)
        motion_aux_logits = motion_aux_logits.masked_fill(~actor_valid[:, :, None], -1e4)

        return {
            "log_probs": final_logp,
            "presence_logits": presence_logits,
            "objectless_logp": objectless_logp,
            "objectful_scores": objectful_scores,
            "objectful_scores_full": objectful_scores_full,
            "objectful_logp": objectful_logp,
            "objectful_logp_full": objectful_logp_full,
            "motion_aux_logits": motion_aux_logits,
            "detected_delta": det_delta_full,
            "visual_delta": vis_delta_full,
            "visual_delta_objectful": visual_delta_objectful,
            "visual_delta_objectful_full": visual_delta_objectful_full,
            "detected_mix_weight": detected_mix_weight,
            "visual_mix_weight": visual_mix_weight,
            "detected_mix_weight_objectful_full": detected_mix_weight_objectful_full,
            "visual_mix_weight_objectful_full": visual_mix_weight_objectful_full,
            "coverage": coverage_full,
            "relation_scale": rel_scale.detach(),
            **{f"det_{k}": v for k, v in det.items()},
            **{f"vis_{k}": v for k, v in vis.items()},
        }


# ----------------------------- losses -----------------------------

def final_nll_loss(log_probs: Tensor, labels: Tensor, actor_valid: Tensor, ignore_index: int = -100) -> Tensor:
    mask = actor_valid.bool() & (labels != ignore_index)
    if not mask.any():
        return log_probs.sum() * 0.0
    return F.nll_loss(log_probs[mask], labels[mask])


def presence_loss(presence_logits: Tensor, labels: Tensor, actor_valid: Tensor, objectless_indices: Iterable[int], ignore_index: int = -100) -> Tensor:
    objectless = torch.zeros(presence_logits.shape[:-1], dtype=torch.bool, device=presence_logits.device)
    for idx in objectless_indices:
        objectless |= labels == int(idx)
    target = (~objectless).long()  # 0=null/objectless, 1=object interaction
    mask = actor_valid.bool() & (labels != ignore_index)
    if not mask.any():
        return presence_logits.sum() * 0.0
    return F.cross_entropy(presence_logits[mask], target[mask])


def objectful_within_loss(objectful_scores: Tensor, labels: Tensor, actor_valid: Tensor, objectful_indices: Sequence[int], ignore_index: int = -100) -> Tensor:
    # Train objectful distinctions without involving objectless classes.
    idx_map = {int(a): i for i, a in enumerate(objectful_indices)}
    target = torch.full_like(labels, ignore_index)
    for a, i in idx_map.items():
        target = torch.where(labels == a, torch.full_like(target, i), target)
    mask = actor_valid.bool() & (target != ignore_index)
    if not mask.any():
        return objectful_scores.sum() * 0.0
    return F.cross_entropy(objectful_scores[mask], target[mask])


def objectless_margin_loss(log_probs: Tensor, labels: Tensor, actor_valid: Tensor, objectless_indices: Sequence[int], objectful_indices: Sequence[int], margin: float = 1.0) -> Tensor:
    objectless = torch.zeros_like(labels, dtype=torch.bool)
    for idx in objectless_indices:
        objectless |= labels == int(idx)
    mask = actor_valid.bool() & objectless
    if not mask.any():
        return log_probs.sum() * 0.0
    true_score = log_probs.gather(-1, labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    objectful_score = _gather_indices(log_probs, objectful_indices).amax(dim=-1)
    return F.relu(margin - (true_score[mask] - objectful_score[mask])).mean()


def restricted_confuser_loss(logits_or_scores: Tensor, labels: Tensor, actor_valid: Tensor, confusers_by_action: Dict[int, Tuple[int, ...]], ignore_index: int = -100) -> Tensor:
    # Use this for teacher-object-dropped missing-object training; never include objectless classes unless explicitly in confuser set.
    losses: List[Tensor] = []
    for action, confs in confusers_by_action.items():
        classes = (int(action),) + tuple(int(c) for c in confs)
        mask = actor_valid.bool() & (labels == int(action))
        if not mask.any():
            continue
        cls = torch.as_tensor(classes, device=logits_or_scores.device, dtype=torch.long)
        subset = logits_or_scores.index_select(-1, cls)[mask]
        target = torch.zeros(subset.shape[0], dtype=torch.long, device=logits_or_scores.device)
        losses.append(F.cross_entropy(subset, target))
    if not losses:
        return logits_or_scores.sum() * 0.0
    return torch.stack(losses).mean()


if __name__ == "__main__":
    B, A, K, C, O, D, N = 2, 3, 5, 10, 6, 32, 16
    compat = torch.zeros(O, C)
    compat[1, 6] = 1
    compat[2, 7] = 1
    spec = FactorizedActionObjectSpec(C, O, objectless_action_indices=(0, 1, 2, 3), compat_matrix=compat)
    head = FactorizedInteractionActionHead(D, spec, hidden_dim=64)
    out = head(
        torch.randn(B, A, D),
        torch.rand(B, A, 4),
        torch.ones(B, A, dtype=torch.bool),
        torch.rand(B, K, 4),
        torch.randint(0, O, (B, K)),
        torch.rand(B, K),
        torch.ones(B, K, dtype=torch.bool),
        torch.randn(B, N, D),
    )
    print(out["log_probs"].shape, torch.logsumexp(out["log_probs"][0, 0], dim=-1))
