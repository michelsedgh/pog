"""Bounded runtime object-evidence head for PO-GUISE+ actor logits.

The PO-GUISE+ actor/video head owns the action decision. Runtime object prompts only
add bounded, actor-specific, taxonomy-compatible evidence to objectful classes. If no
useful compatible object is detected, the object residual is near zero and the base
actor/video logits carry the prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ObjectResidualActionSpec:
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
            raise ValueError("object_residual head requires at least one objectless action")
        if len(objectless) == self.num_actions:
            raise ValueError("object_residual head requires at least one objectful action")
        if any(i < 0 or i >= self.num_actions for i in objectless):
            raise ValueError("objectless_action_indices contains an invalid action id")
        object.__setattr__(self, "objectless_action_indices", objectless)

    @property
    def objectful_action_indices(self) -> Tuple[int, ...]:
        objectless = set(int(i) for i in self.objectless_action_indices)
        return tuple(i for i in range(self.num_actions) if i not in objectless)


class PromptRelationExpert(nn.Module):
    """Action-conditioned actor/object-prompt relation scorer."""

    def __init__(
        self,
        dim: int,
        spec: ObjectResidualActionSpec,
        hidden_dim: int = 512,
        relation_dim: int = 256,
        relation_logit_bound: float = 2.0,
        compat_prior_scale: float = 1.0,
    ):
        super().__init__()
        self.spec = spec
        self.relation_logit_bound = float(relation_logit_bound)
        self.compat_prior_scale = float(compat_prior_scale)
        if self.compat_prior_scale < 0:
            raise ValueError("compat_prior_scale must be >= 0")

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
        self.null_relation_logit = nn.Parameter(torch.zeros(spec.num_actions))

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
        prompt_logits = torch.matmul(actor_q, prompt_k.transpose(1, 2))
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

        raw = torch.matmul(
            pair_rel.reshape(B * A * K, pair_rel.shape[-1]),
            self.action_query.transpose(0, 1),
        )
        raw = raw.reshape(B, A, K, self.spec.num_actions).permute(0, 1, 3, 2)
        raw = raw / (pair_rel.shape[-1] ** 0.5)
        bound = max(self.relation_logit_bound, 1.0e-6)
        raw = self.relation_logit_bound * torch.tanh(raw / bound)

        compat_bkc = self._compat_by_slot(object_classes, object_valid)
        compat_back = compat_bkc[:, None, :, :].permute(0, 1, 3, 2)
        compat_back = compat_back.expand(B, A, self.spec.num_actions, K)
        relation_valid = (compat_back > 0) & object_valid[:, None, None, :]
        raw_masked = raw.masked_fill(~relation_valid, -1.0e4)

        null_logit = self.null_relation_logit.to(
            device=actor_tokens.device,
            dtype=actor_tokens.dtype,
        )
        null_logit = null_logit.view(1, 1, self.spec.num_actions, 1)
        null_logit = null_logit.expand(B, A, -1, -1)
        posterior_logits = torch.cat([null_logit, raw_masked], dim=-1)
        posterior = F.softmax(posterior_logits.float(), dim=-1).to(
            dtype=actor_tokens.dtype
        )
        null_prob = posterior[..., 0]
        object_posterior_unnorm = posterior[..., 1:] * relation_valid.to(
            dtype=posterior.dtype
        )
        object_posterior_sum = object_posterior_unnorm.sum(dim=-1, keepdim=True)
        useful_mass = object_posterior_sum.squeeze(-1).clamp(0.0, 1.0)
        object_posterior = torch.where(
            object_posterior_sum > 0,
            object_posterior_unnorm / object_posterior_sum.clamp_min(1.0e-6),
            torch.zeros_like(object_posterior_unnorm),
        )
        relation_delta = torch.sum(object_posterior * raw_masked, dim=-1)
        relation_delta = relation_delta * useful_mass.detach()

        valid_f = object_valid[:, None, None, :].to(dtype=actor_tokens.dtype)
        conf_f = object_confs[:, None, None, :].to(dtype=actor_tokens.dtype).clamp(
            0.0,
            1.0,
        )
        attn_f = prompt_attention.detach()[:, :, None, :]
        posterior_gate = object_posterior_unnorm.detach()
        actor_object_relevance = (
            0.20 + 0.40 * attn_f + 0.40 * posterior_gate
        ).clamp(0.0, 1.0)
        actor_object_strength = (
            valid_f
            * conf_f
            * actor_object_relevance
            * useful_mass.detach()[..., None]
        )
        coverage = (compat_back * actor_object_strength).amax(dim=-1).clamp(0.0, 1.0)
        compat_prior = self.compat_prior_scale * coverage
        relation_delta = relation_delta + compat_prior

        return {
            "prompt_relation_delta": relation_delta,
            "prompt_relation_scores": raw_masked,
            "prompt_relation_posterior": object_posterior,
            "prompt_relation_null_prob": null_prob,
            "prompt_relation_useful_mass": useful_mass,
            "prompt_attention_logits": prompt_logits,
            "prompt_attention": prompt_attention,
            "coverage": coverage,
            "compat_back": compat_back,
        }


class ActorObjectContextFusion(nn.Module):
    """Fuse runtime object context into actor tokens before action classification."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int = 512,
        relation_dim: int = 256,
        fusion_scale_init: float = -2.0,
        max_fusion_scale: float = 1.0,
    ):
        super().__init__()
        self.actor_norm = nn.LayerNorm(dim)
        self.object_norm = nn.LayerNorm(dim)
        self.actor_query = nn.Linear(dim, relation_dim, bias=False)
        self.object_key = nn.Linear(dim, relation_dim, bias=False)
        self.object_value = nn.Linear(dim, dim, bias=False)
        self.null_key = nn.Parameter(torch.zeros(relation_dim))
        self.null_logit = nn.Parameter(torch.tensor(0.0))
        self.context_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(3 * dim),
            nn.Linear(3 * dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Sigmoid(),
        )
        self.fusion_scale_logit = nn.Parameter(torch.tensor(float(fusion_scale_init)))
        self.max_fusion_scale = float(max_fusion_scale)
        if self.max_fusion_scale < 0:
            raise ValueError("max_fusion_scale must be >= 0")
        nn.init.zeros_(self.context_proj[-1].weight)
        nn.init.zeros_(self.context_proj[-1].bias)
        nn.init.zeros_(self.gate[-2].weight)
        nn.init.constant_(self.gate[-2].bias, -1.0)

    def forward(
        self,
        actor_tokens: Tensor,
        object_prompt_tokens: Tensor,
        object_confs: Tensor,
        object_valid: Tensor,
        object_attention_bias: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        object_prompt_tokens = object_prompt_tokens.to(
            device=actor_tokens.device,
            dtype=actor_tokens.dtype,
        )
        object_valid = object_valid.to(device=actor_tokens.device, dtype=torch.bool)
        object_confs = object_confs.to(
            device=actor_tokens.device,
            dtype=actor_tokens.dtype,
        ).clamp(0.0, 1.0)
        object_confs = object_confs * object_valid.to(dtype=object_confs.dtype)

        q = self.actor_query(self.actor_norm(actor_tokens))
        k = self.object_key(self.object_norm(object_prompt_tokens))
        scores_obj = torch.matmul(q, k.transpose(1, 2)) / (q.shape[-1] ** 0.5)
        conf_bias = torch.log(object_confs.clamp_min(1.0e-4))[:, None, :]
        scores_obj = scores_obj + conf_bias
        if object_attention_bias is not None:
            object_attention_bias = object_attention_bias.to(
                device=actor_tokens.device,
                dtype=actor_tokens.dtype,
            )
            if tuple(object_attention_bias.shape) != tuple(scores_obj.shape):
                raise ValueError(
                    "object_attention_bias must have shape "
                    f"{tuple(scores_obj.shape)}, got {tuple(object_attention_bias.shape)}"
                )
            scores_obj = scores_obj + object_attention_bias
        scores_obj = scores_obj.masked_fill(~object_valid[:, None, :], -1.0e4)

        null_key = self.null_key.to(device=actor_tokens.device, dtype=actor_tokens.dtype)
        null_score = torch.sum(q * null_key.view(1, 1, -1), dim=-1, keepdim=True)
        null_score = null_score / (q.shape[-1] ** 0.5)
        null_score = null_score + self.null_logit.to(
            device=actor_tokens.device,
            dtype=actor_tokens.dtype,
        )
        scores = torch.cat([null_score, scores_obj], dim=-1)

        attention_full = F.softmax(scores.float(), dim=-1).to(dtype=actor_tokens.dtype)
        null_prob = attention_full[..., 0]
        attention = attention_full[..., 1:] * object_valid[:, None, :].to(
            dtype=actor_tokens.dtype
        )
        useful_mass = attention.sum(dim=-1).clamp(0.0, 1.0)

        values = self.object_value(object_prompt_tokens)
        context = torch.matmul(attention, values)

        gate = self.gate(
            torch.cat([actor_tokens, context, actor_tokens * context], dim=-1)
        )
        delta = self.context_proj(context) * gate * useful_mass[..., None]
        scale = self.max_fusion_scale * torch.sigmoid(self.fusion_scale_logit)
        scaled_delta = scale * delta
        fused = actor_tokens + scale * delta
        return {
            "actor_tokens": fused,
            "object_context": context,
            "object_attention": attention,
            "object_null_prob": null_prob.detach(),
            "object_useful_mass": useful_mass.detach(),
            "fusion_delta": scaled_delta.detach(),
            "fusion_gate": gate.detach(),
            "fusion_scale": scale.detach(),
        }


class ObjectResidualActionHead(nn.Module):
    """Prompt-aware bounded residual over base PO-GUISE+ actor logits."""

    def __init__(
        self,
        dim: int,
        spec: ObjectResidualActionSpec,
        hidden_dim: int = 512,
        relation_logit_bound: float = 2.0,
        relation_scale_init: float = -1.0,
        max_relation_scale: float = 1.5,
        compat_prior_scale: float = 1.0,
    ):
        super().__init__()
        self.spec = spec
        self.objectless_idx = tuple(int(i) for i in spec.objectless_action_indices)
        objectless_mask = torch.zeros(spec.num_actions, dtype=torch.bool)
        objectless_mask[list(self.objectless_idx)] = True
        self.register_buffer("objectless_mask", objectless_mask, persistent=False)
        self.register_buffer("objectful_mask", ~objectless_mask, persistent=False)

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
            compat_prior_scale=compat_prior_scale,
        )
        self.relation_scale_logit = nn.Parameter(torch.tensor(float(relation_scale_init)))
        self.max_relation_scale = float(max_relation_scale)

    def forward(
        self,
        actor_tokens: Tensor,
        actor_valid: Tensor,
        base_logits: Tensor,
        object_prompt_tokens: Tensor,
        object_classes: Tensor,
        object_confs: Tensor,
        object_valid: Tensor,
    ) -> Dict[str, Tensor]:
        actor_valid = actor_valid.to(device=actor_tokens.device, dtype=torch.bool)
        base_logits = base_logits.to(device=actor_tokens.device, dtype=actor_tokens.dtype)
        relation_scale = self.max_relation_scale * torch.sigmoid(
            self.relation_scale_logit
        )

        motion_aux_logits = self.motion_aux_head(actor_tokens)
        prompt = self.prompt_relation(
            actor_tokens,
            object_prompt_tokens,
            object_classes,
            object_confs,
            object_valid,
        )

        raw_residual = prompt["prompt_relation_delta"]
        object_residual = relation_scale * torch.tanh(raw_residual)
        object_residual = object_residual.masked_fill(
            self.objectless_mask.view(1, 1, -1),
            0.0,
        )

        final_logits = base_logits + object_residual
        base_logp = F.log_softmax(base_logits, dim=-1)
        final_logp = F.log_softmax(final_logits, dim=-1)
        final_logp = final_logp.masked_fill(~actor_valid[:, :, None], -1.0e4)
        final_logits = final_logits.masked_fill(~actor_valid[:, :, None], -1.0e4)
        motion_aux_logits = motion_aux_logits.masked_fill(
            ~actor_valid[:, :, None],
            -1.0e4,
        )

        return {
            "log_probs": final_logp,
            "base_logits": base_logits,
            "base_logp": base_logp,
            "final_logits": final_logits,
            "object_residual": object_residual,
            "motion_aux_logits": motion_aux_logits,
            "prompt_delta": object_residual,
            "coverage": prompt["coverage"],
            "relation_scale": relation_scale.detach(),
            "prompt_attention_logits": prompt["prompt_attention_logits"],
            "prompt_attention": prompt["prompt_attention"],
            "prompt_relation_scores": prompt["prompt_relation_scores"],
            "prompt_relation_posterior": prompt["prompt_relation_posterior"],
            "prompt_relation_null_prob": prompt["prompt_relation_null_prob"],
            "prompt_relation_useful_mass": prompt["prompt_relation_useful_mass"],
        }
