"""Factorized action head with runtime object-prompt evidence.

This head keeps the final action decision split into:

    NULL/objectless mode -> objectless action distribution from actor tokens only
    INTERACTION mode     -> objectful action distribution from actor + prompt evidence

Runtime object prompt tokens can strongly affect objectful action scores, but they do
not edit the actor representation used by the objectless branch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class FactorizedActionObjectSpec:
    num_actions: int
    num_object_classes: int
    objectless_action_indices: Tuple[int, ...]
    compat_matrix: Tensor
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
        if not objectless:
            raise ValueError("factorized head requires at least one objectless action")
        if len(objectless) == self.num_actions:
            raise ValueError("factorized head requires at least one objectful action")
        if any(i < 0 or i >= self.num_actions for i in objectless):
            raise ValueError("objectless_action_indices contains an invalid action id")
        object.__setattr__(self, "objectless_action_indices", objectless)

    @property
    def objectful_action_indices(self) -> Tuple[int, ...]:
        objectless = set(int(i) for i in self.objectless_action_indices)
        return tuple(i for i in range(self.num_actions) if i not in objectless)


def _gather_last_dim(values: Tensor, indices: Sequence[int]) -> Tensor:
    idx = torch.as_tensor(indices, device=values.device, dtype=torch.long)
    return values.index_select(-1, idx)


def _scatter_last_dim(
    values: Tensor,
    indices: Tensor,
    size: int,
    fill_value: float = -1.0e4,
) -> Tensor:
    output = values.new_full((*values.shape[:-1], int(size)), float(fill_value))
    output.index_copy_(-1, indices.to(device=values.device), values)
    return output


class PromptRelationExpert(nn.Module):
    """Action-conditioned actor/object-prompt relation scorer."""

    def __init__(
        self,
        dim: int,
        spec: FactorizedActionObjectSpec,
        hidden_dim: int = 512,
        relation_dim: int = 256,
        relation_logit_bound: float = 2.0,
        taxonomy_prior_cap: float = 4.0,
    ):
        super().__init__()
        self.spec = spec
        self.relation_logit_bound = float(relation_logit_bound)
        self.taxonomy_prior_cap = float(taxonomy_prior_cap)

        self.actor_query = nn.Linear(dim, relation_dim, bias=False)
        self.prompt_key = nn.Linear(dim, relation_dim, bias=False)
        self.pair_mlp = nn.Sequential(
            nn.LayerNorm(3 * dim + 1),
            nn.Linear(3 * dim + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.LayerNorm(dim),
        )
        self.pair_proj = nn.Linear(dim, relation_dim, bias=False)
        self.action_query = nn.Parameter(
            torch.randn(spec.num_actions, relation_dim) * 0.02
        )

        self.register_buffer(
            "compat_matrix",
            spec.compat_matrix.float(),
            persistent=False,
        )

    def _compat_by_slot(
        self,
        object_classes: Tensor,
        object_valid: Tensor,
    ) -> Tensor:
        safe_classes = object_classes.clamp(0, self.spec.num_object_classes - 1)
        class_valid = (object_classes >= 0) & (
            object_classes < self.spec.num_object_classes
        )
        compat = F.embedding(safe_classes, self.compat_matrix)
        return compat * (class_valid & object_valid).to(dtype=compat.dtype).unsqueeze(-1)

    def forward(
        self,
        actor_tokens: Tensor,
        object_prompt_tokens: Tensor,
        object_classes: Tensor,
        object_confs: Tensor,
        object_valid: Tensor,
        eps: float = 1e-4,
    ) -> Dict[str, Tensor]:
        B, A, D = actor_tokens.shape
        K = object_prompt_tokens.shape[1]
        if K <= 0:
            raise RuntimeError("object prompt relation expert requires prompt tokens")

        object_prompt_tokens = object_prompt_tokens.to(
            device=actor_tokens.device,
            dtype=actor_tokens.dtype,
        )
        object_classes = object_classes.to(device=actor_tokens.device, dtype=torch.long)
        object_valid = object_valid.to(device=actor_tokens.device, dtype=torch.bool)
        object_confs = object_confs.to(
            device=actor_tokens.device,
            dtype=actor_tokens.dtype,
        ) * object_valid.to(device=actor_tokens.device, dtype=actor_tokens.dtype)

        actor_q = self.actor_query(actor_tokens)
        prompt_k = self.prompt_key(object_prompt_tokens)
        prompt_logits = torch.einsum("bar,bkr->bak", actor_q, prompt_k)
        prompt_logits = prompt_logits / (actor_q.shape[-1] ** 0.5)
        prompt_logits = prompt_logits.masked_fill(~object_valid[:, None, :], -1.0e4)
        prompt_attention = F.softmax(prompt_logits.float(), dim=-1).to(
            dtype=actor_tokens.dtype
        )
        prompt_attention = prompt_attention * object_valid[:, None, :].to(
            dtype=prompt_attention.dtype
        )
        prompt_attention_sum = prompt_attention.sum(dim=-1, keepdim=True)
        prompt_attention = torch.where(
            prompt_attention_sum > 0,
            prompt_attention / prompt_attention_sum.clamp_min(1.0e-6),
            torch.zeros_like(prompt_attention),
        )

        actor_expanded = actor_tokens[:, :, None, :].expand(B, A, K, D)
        prompt_expanded = object_prompt_tokens[:, None, :, :].expand(B, A, K, D)
        conf = object_confs[:, None, :, None].expand(B, A, K, 1)
        pair_input = torch.cat(
            [
                actor_expanded,
                prompt_expanded,
                actor_expanded * prompt_expanded,
                conf,
            ],
            dim=-1,
        )
        pair = self.pair_mlp(pair_input)
        pair_rel = self.pair_proj(pair)

        raw = torch.einsum("bakr,cr->back", pair_rel, self.action_query)
        raw = raw / (pair_rel.shape[-1] ** 0.5)
        bound = max(self.relation_logit_bound, 1.0e-6)
        raw = self.relation_logit_bound * torch.tanh(raw / bound)

        compat_bkc = self._compat_by_slot(object_classes, object_valid)
        compat_back = compat_bkc[:, None, :, :].permute(0, 1, 3, 2)
        compat_back = compat_back.expand(B, A, self.spec.num_actions, K)
        relation_valid = (compat_back > 0) & object_valid[:, None, None, :]
        raw = raw.masked_fill(~relation_valid, -1.0e4)

        posterior = F.softmax(raw.float(), dim=-1).to(dtype=actor_tokens.dtype)
        posterior = posterior * relation_valid.to(dtype=posterior.dtype)
        posterior_sum = posterior.sum(dim=-1, keepdim=True)
        posterior = torch.where(
            posterior_sum > 0,
            posterior / posterior_sum.clamp_min(1.0e-6),
            torch.zeros_like(posterior),
        )
        relation_delta = torch.sum(posterior * raw, dim=-1)

        proposal_strength = object_confs[:, None, None, :] * object_valid[
            :, None, None, :
        ].to(dtype=actor_tokens.dtype)
        coverage = (compat_back * proposal_strength).amax(dim=-1).clamp(0.0, 1.0)
        taxonomy_prior = torch.log1p(compat_back * proposal_strength / eps).amax(
            dim=-1
        )
        taxonomy_prior = taxonomy_prior.clamp(max=self.taxonomy_prior_cap)
        relation_delta = relation_delta + taxonomy_prior

        return {
            "prompt_relation_delta": relation_delta,
            "prompt_relation_scores": raw,
            "prompt_relation_posterior": posterior,
            "prompt_attention_logits": prompt_logits,
            "prompt_attention": prompt_attention,
            "coverage": coverage,
            "compat_back": compat_back,
        }


class VisualFallbackExpert(nn.Module):
    """Detector-free objectful fallback from heatmap/visual tokens."""

    def __init__(
        self,
        dim: int,
        spec: FactorizedActionObjectSpec,
        hidden_dim: int = 512,
        relation_dim: int = 256,
        relation_logit_bound: float = 2.0,
    ):
        super().__init__()
        self.spec = spec
        self.relation_logit_bound = float(relation_logit_bound)
        self.query_seed = nn.Parameter(torch.randn(1, dim) * 0.02)
        self.query = nn.Linear(dim, relation_dim, bias=False)
        self.key = nn.Linear(dim, relation_dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.slot_norm = nn.LayerNorm(dim)
        self.slot_mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self.slot_proj = nn.Linear(dim, relation_dim, bias=False)
        self.action_query = nn.Parameter(
            torch.randn(spec.num_actions, relation_dim) * 0.02
        )

    def forward(
        self,
        actor_tokens: Tensor,
        source_tokens: Optional[Tensor],
        source_valid: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        B, A, D = actor_tokens.shape
        if source_tokens is None:
            delta = actor_tokens.new_zeros(B, A, self.spec.num_actions)
            return {"visual_delta": delta, "visual_attention": None}

        source_tokens = source_tokens.to(
            device=actor_tokens.device,
            dtype=actor_tokens.dtype,
        )
        q = self.query(actor_tokens + self.query_seed.view(1, 1, D))
        k = self.key(source_tokens)
        v = self.value(source_tokens)
        scores = torch.einsum("bar,bnr->ban", q, k) / (q.shape[-1] ** 0.5)
        if source_valid is not None:
            source_valid = source_valid.to(device=actor_tokens.device, dtype=torch.bool)
            scores = scores.masked_fill(~source_valid[:, None, :], -1.0e4)
        attention = F.softmax(scores.float(), dim=-1).to(dtype=actor_tokens.dtype)
        if source_valid is not None:
            attention = attention * source_valid[:, None, :].to(dtype=attention.dtype)
            attention_sum = attention.sum(dim=-1, keepdim=True)
            attention = torch.where(
                attention_sum > 0,
                attention / attention_sum.clamp_min(1.0e-6),
                torch.zeros_like(attention),
            )
        slot = torch.einsum("ban,bnd->bad", attention, v)
        slot = self.slot_norm(slot + actor_tokens)
        slot = self.slot_mlp(slot)
        rel = self.slot_proj(slot)
        raw = torch.einsum("bar,cr->bac", rel, self.action_query)
        raw = raw / (rel.shape[-1] ** 0.5)
        bound = max(self.relation_logit_bound, 1.0e-6)
        delta = self.relation_logit_bound * torch.tanh(raw / bound)
        return {"visual_delta": delta, "visual_attention": attention}


class FactorizedInteractionActionHead(nn.Module):
    """Prompt-aware final head with explicit objectless/objectful arbitration."""

    def __init__(
        self,
        dim: int,
        spec: FactorizedActionObjectSpec,
        hidden_dim: int = 512,
        relation_logit_bound: float = 2.0,
        relation_scale_init: float = -1.0,
        max_relation_scale: float = 1.5,
    ):
        super().__init__()
        self.spec = spec
        self.objectless_idx = tuple(int(i) for i in spec.objectless_action_indices)
        self.objectful_idx = spec.objectful_action_indices
        self.register_buffer(
            "objectless_index",
            torch.tensor(self.objectless_idx, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "objectful_index",
            torch.tensor(self.objectful_idx, dtype=torch.long),
            persistent=False,
        )

        self.presence_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )
        self.objectless_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, len(self.objectless_idx)),
        )
        self.objectful_motion_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, len(self.objectful_idx)),
        )
        self.motion_aux_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, spec.num_actions),
        )
        self.prompt_relation = PromptRelationExpert(
            dim,
            spec,
            hidden_dim=hidden_dim,
            relation_logit_bound=relation_logit_bound,
        )
        self.visual_fallback = VisualFallbackExpert(
            dim,
            spec,
            hidden_dim=hidden_dim,
            relation_logit_bound=relation_logit_bound,
        )
        self.relation_scale_logit = nn.Parameter(torch.tensor(float(relation_scale_init)))
        self.max_relation_scale = float(max_relation_scale)

    def forward(
        self,
        actor_tokens: Tensor,
        actor_valid: Tensor,
        object_prompt_tokens: Tensor,
        object_classes: Tensor,
        object_confs: Tensor,
        object_valid: Tensor,
        visual_source_tokens: Optional[Tensor] = None,
        visual_source_valid: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        B, A, _ = actor_tokens.shape
        actor_valid = actor_valid.to(device=actor_tokens.device, dtype=torch.bool)
        relation_scale = self.max_relation_scale * torch.sigmoid(
            self.relation_scale_logit
        )

        presence_logits = self.presence_head(actor_tokens)
        presence_logp = F.log_softmax(presence_logits, dim=-1)
        objectless_logp = F.log_softmax(self.objectless_head(actor_tokens), dim=-1)
        objectful_motion = self.objectful_motion_head(actor_tokens)
        motion_aux_logits = self.motion_aux_head(actor_tokens)

        prompt = self.prompt_relation(
            actor_tokens,
            object_prompt_tokens,
            object_classes,
            object_confs,
            object_valid,
        )
        visual = self.visual_fallback(
            actor_tokens,
            visual_source_tokens,
            visual_source_valid,
        )

        prompt_delta_full = prompt["prompt_relation_delta"] * relation_scale
        visual_delta_full = visual["visual_delta"] * relation_scale
        coverage_full = prompt["coverage"].clamp(1.0e-4, 1.0 - 1.0e-4)

        prompt_delta = _gather_last_dim(prompt_delta_full, self.objectful_idx)
        visual_delta = _gather_last_dim(visual_delta_full, self.objectful_idx)
        coverage = _gather_last_dim(coverage_full, self.objectful_idx)

        detected_path = torch.log(coverage) + prompt_delta
        missing_path = torch.log1p(-coverage) + visual_delta
        evidence_mix = torch.logsumexp(
            torch.stack([detected_path, missing_path], dim=-1),
            dim=-1,
        )
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
        final_logp = final_logp.masked_fill(~actor_valid[:, :, None], -1.0e4)
        motion_aux_logits = motion_aux_logits.masked_fill(
            ~actor_valid[:, :, None],
            -1.0e4,
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

        return {
            "log_probs": final_logp,
            "presence_logits": presence_logits,
            "objectless_logp": objectless_logp,
            "objectful_scores": objectful_scores,
            "objectful_scores_full": objectful_scores_full,
            "objectful_logp": objectful_logp,
            "objectful_logp_full": objectful_logp_full,
            "motion_aux_logits": motion_aux_logits,
            "prompt_delta": prompt_delta_full,
            "visual_delta": visual_delta_full,
            "coverage": prompt["coverage"],
            "relation_scale": relation_scale.detach(),
            "prompt_attention_logits": prompt["prompt_attention_logits"],
            "prompt_attention": prompt["prompt_attention"],
            "prompt_relation_scores": prompt["prompt_relation_scores"],
            "prompt_relation_posterior": prompt["prompt_relation_posterior"],
            "visual_attention": visual["visual_attention"],
        }


def final_nll_loss(
    log_probs: Tensor,
    labels: Tensor,
    actor_valid: Tensor,
    ignore_index: int = -100,
) -> Tensor:
    labels = labels.to(device=log_probs.device, dtype=torch.long)
    actor_valid = actor_valid.to(device=log_probs.device, dtype=torch.bool)
    mask = actor_valid & (labels != ignore_index)
    if not mask.any():
        return log_probs.sum() * 0.0
    return F.nll_loss(log_probs[mask].float(), labels[mask])


def presence_loss(
    presence_logits: Tensor,
    labels: Tensor,
    actor_valid: Tensor,
    objectless_indices: Iterable[int],
    ignore_index: int = -100,
) -> Tensor:
    labels = labels.to(device=presence_logits.device, dtype=torch.long)
    actor_valid = actor_valid.to(device=presence_logits.device, dtype=torch.bool)
    objectless = torch.zeros_like(labels, dtype=torch.bool)
    for idx in objectless_indices:
        objectless |= labels == int(idx)
    target = (~objectless).long()
    mask = actor_valid & (labels != ignore_index)
    if not mask.any():
        return presence_logits.sum() * 0.0
    return F.cross_entropy(presence_logits[mask].float(), target[mask])
