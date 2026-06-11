import torch.nn as nn
import pytorch_lightning as pl
from argparse import ArgumentParser
from blocks.poguise import vit_base_patch16_224
from blocks.poguise import vit_small_patch16_224
from blocks.poguise import vit_large_patch16_224
import torch
import os
import os.path
import pickle
from datasets.object_vocab import NUM_OBJECT_CLASSES, OBJECT_TO_ID
from datasets.toyota_action_taxonomy import (
    toyota_action_object_map,
    toyota_action_to_index,
    toyota_objectless_action_names,
)
from models.trt_actor_object_slot_head import (
    ActionObjectSlotSpec,
    TRTFriendlyActorObjectSlotHead,
)


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


class ActorObjectSelectionHead(nn.Module):
    def __init__(self, dim, hidden_dim=512):
        super().__init__()
        self.actor_norm = nn.LayerNorm(dim)
        self.object_norm = nn.LayerNorm(dim)
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.none_mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, actor_tokens, object_tokens, object_valid):
        if actor_tokens.ndim != 3 or object_tokens.ndim != 3:
            raise ValueError("actor_tokens and object_tokens must be rank-3 tensors")
        if object_valid.shape != object_tokens.shape[:2]:
            raise ValueError(
                "object_valid must have shape "
                f"{tuple(object_tokens.shape[:2])}, got {tuple(object_valid.shape)}"
            )
        query = self.query(self.actor_norm(actor_tokens))
        key = self.key(self.object_norm(object_tokens))
        scale = query.shape[-1] ** -0.5
        object_logits = torch.einsum("bkd,bmd->bkm", query, key) * scale
        object_logits = object_logits.masked_fill(
            ~object_valid[:, None, :].bool(),
            torch.finfo(object_logits.dtype).min,
        )
        none_logits = self.none_mlp(actor_tokens).squeeze(-1).unsqueeze(-1)
        return torch.cat([none_logits, object_logits], dim=-1)


class ActorObjectCompatibilityExpert(nn.Module):
    def __init__(
        self,
        dim,
        num_classes,
        object_action_indices,
        allowed_action_indices,
        num_object_classes=NUM_OBJECT_CLASSES,
        hidden_dim=512,
        relevance_gate_init=-2.0,
        compatibility_eps=0.05,
        compatibility_scale=1.0,
        pair_residual_scale=1.0,
    ):
        super().__init__()
        num_classes = int(num_classes)
        num_object_classes = int(num_object_classes)
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if num_object_classes <= 0:
            raise ValueError("num_object_classes must be positive")
        if compatibility_eps <= 0:
            raise ValueError("compatibility_eps must be > 0")
        if compatibility_scale < 0:
            raise ValueError("compatibility_scale must be >= 0")
        if pair_residual_scale < 0:
            raise ValueError("pair_residual_scale must be >= 0")

        self.num_classes = num_classes
        self.num_object_classes = num_object_classes
        self.compatibility_eps = float(compatibility_eps)
        self.compatibility_scale = float(compatibility_scale)
        self.pair_residual_scale = float(pair_residual_scale)
        self.object_norm = nn.LayerNorm(dim)
        self.actor_norm = nn.LayerNorm(dim)
        self.relevance_head = nn.Sequential(
            nn.LayerNorm(dim * 3 + 3),
            nn.Linear(dim * 3 + 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.pair_head = nn.Sequential(
            nn.LayerNorm(dim * 3 + 3),
            nn.Linear(dim * 3 + 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
        )
        object_action_mask = torch.zeros(num_object_classes, num_classes)
        for object_id, action_indices in object_action_indices.items():
            object_id = int(object_id)
            if object_id < 0 or object_id >= num_object_classes:
                continue
            for action_idx in action_indices:
                action_idx = int(action_idx)
                if 0 <= action_idx < num_classes:
                    object_action_mask[object_id, action_idx] = 1.0
        allowed_action_mask = torch.zeros(num_classes)
        for action_idx in allowed_action_indices:
            action_idx = int(action_idx)
            if 0 <= action_idx < num_classes:
                allowed_action_mask[action_idx] = 1.0
        if float(object_action_mask.sum()) == 0.0:
            allowed_action_mask.zero_()
        self.register_buffer(
            "object_action_mask",
            object_action_mask,
            persistent=False,
        )
        self.register_buffer(
            "allowed_action_mask",
            allowed_action_mask,
            persistent=False,
        )
        nn.init.zeros_(self.relevance_head[-1].weight)
        nn.init.constant_(self.relevance_head[-1].bias, float(relevance_gate_init))
        nn.init.zeros_(self.pair_head[-1].weight)
        nn.init.zeros_(self.pair_head[-1].bias)

    def _selection_distribution(self, object_selection_logits, object_valid, dtype):
        if object_selection_logits is None:
            raise ValueError(
                "object_selection_logits is required for actor-object compatibility"
            )
        if object_selection_logits.ndim != 3:
            raise ValueError("object_selection_logits must be rank-3")
        if object_selection_logits.shape[-1] != object_valid.shape[1] + 1:
            raise ValueError(
                "object_selection_logits last dimension must be num_objects + 1"
            )
        object_logits = object_selection_logits[..., 1:].masked_fill(
            ~object_valid[:, None, :],
            torch.finfo(object_selection_logits.dtype).min,
        )
        selection_logits = torch.cat(
            (object_selection_logits[..., :1], object_logits),
            dim=-1,
        )
        # The action stack must use the same selector distribution at train and
        # inference time. The selector itself is trained by its own supervised
        # loss, so the action loss consumes a detached distribution here.
        return selection_logits.float().softmax(dim=-1).to(dtype=dtype).detach()

    def _selector_features(self, selection_probs):
        none_prob = selection_probs[..., :1]
        object_probs = selection_probs[..., 1:]
        object_mass = object_probs.sum(dim=-1, keepdim=True)
        top_object_prob = object_probs.amax(dim=-1, keepdim=True)
        entropy = -(selection_probs.clamp_min(1e-6).log() * selection_probs).sum(
            dim=-1,
            keepdim=True,
        )
        entropy = entropy / selection_probs.shape[-1]
        return torch.cat((1.0 - none_prob, top_object_prob, entropy), dim=-1), object_mass

    def _object_context(self, object_tokens, object_probs, object_mass):
        conditional_object_probs = object_probs / object_mass.clamp_min(1e-6)
        return torch.einsum(
            "bkm,bmd->bkd",
            conditional_object_probs,
            object_tokens.detach(),
        )

    def _object_compatibility(self, object_probs, object_classes, object_valid):
        if object_classes is None:
            raise ValueError(
                "object_classes is required for actor-object compatibility"
            )
        if object_classes.shape != object_valid.shape:
            raise ValueError(
                "object_classes must have shape "
                f"{tuple(object_valid.shape)}, got {tuple(object_classes.shape)}"
            )
        device = object_probs.device
        object_action_mask = self.object_action_mask.to(
            device=device,
            dtype=object_probs.dtype,
        )
        object_valid = object_valid.to(device=device, dtype=torch.bool)
        raw_classes = object_classes.to(device=device, dtype=torch.long)
        class_valid = (raw_classes >= 0) & (raw_classes < self.num_object_classes)
        safe_classes = raw_classes.clamp(0, self.num_object_classes - 1)
        slot_action_mask = object_action_mask[safe_classes]
        valid_class_slots = object_valid & class_valid
        slot_action_mask = slot_action_mask * valid_class_slots[:, :, None].to(
            dtype=object_probs.dtype
        )
        return torch.einsum("bkm,bmc->bkc", object_probs, slot_action_mask)

    def forward(
        self,
        actor_tokens,
        object_tokens,
        object_valid,
        object_selection_logits,
        object_classes,
        base_logits,
    ):
        if actor_tokens.ndim != 3 or object_tokens.ndim != 3:
            raise ValueError("actor_tokens and object_tokens must be rank-3 tensors")
        if object_valid.shape != object_tokens.shape[:2]:
            raise ValueError(
                "object_valid must have shape "
                f"{tuple(object_tokens.shape[:2])}, got {tuple(object_valid.shape)}"
            )
        if (
            object_selection_logits is not None
            and object_selection_logits.shape[:2] != actor_tokens.shape[:2]
        ):
            raise ValueError(
                "object_selection_logits must have shape [batch, actors, objects+1]"
            )
        if base_logits.shape != actor_tokens.shape[:2] + (self.num_classes,):
            raise ValueError(
                "base_logits must have shape "
                f"{actor_tokens.shape[:2] + (self.num_classes,)}, "
                f"got {tuple(base_logits.shape)}"
            )

        object_valid = object_valid.to(device=actor_tokens.device, dtype=torch.bool)
        selection_probs = self._selection_distribution(
            object_selection_logits,
            object_valid,
            actor_tokens.dtype,
        )
        selector_features, object_mass = self._selector_features(selection_probs)
        object_probs = selection_probs[..., 1:] * object_valid[:, None, :].to(
            dtype=selection_probs.dtype
        )
        object_context = self._object_context(
            object_tokens,
            object_probs,
            object_mass,
        )
        compatibility = self._object_compatibility(
            object_probs,
            object_classes,
            object_valid,
        ).to(dtype=actor_tokens.dtype)
        object_mass_for_actions = object_mass.to(dtype=actor_tokens.dtype)

        relevance_input = torch.cat(
            (
                self.actor_norm(actor_tokens),
                self.object_norm(object_context),
                self.actor_norm(actor_tokens) * self.object_norm(object_context),
                selector_features.to(dtype=actor_tokens.dtype),
            ),
            dim=-1,
        )
        relevance_logits = self.relevance_head(relevance_input).squeeze(-1)
        relevance_gate = torch.sigmoid(relevance_logits).unsqueeze(-1)
        final_gate = relevance_gate * object_mass_for_actions

        eps = torch.as_tensor(
            self.compatibility_eps,
            device=actor_tokens.device,
            dtype=actor_tokens.dtype,
        )
        incompatible_mass = (object_mass_for_actions - compatibility).clamp_min(0.0)
        compatibility_prior = torch.log1p(compatibility / eps) - torch.log1p(
            incompatible_mass / eps
        )
        allowed_action_mask = self.allowed_action_mask.to(
            device=actor_tokens.device,
            dtype=actor_tokens.dtype,
        ).view(1, 1, -1)
        compatibility_prior = compatibility_prior * allowed_action_mask

        pair_residual = torch.tanh(self.pair_head(relevance_input))
        pair_residual = pair_residual * allowed_action_mask
        pair_residual = pair_residual * compatibility.detach()
        pair_residual = pair_residual * self.pair_residual_scale

        adjustment = final_gate * (
            self.compatibility_scale * compatibility_prior + pair_residual
        )
        final_logits = base_logits + adjustment.to(dtype=base_logits.dtype)
        return {
            "logits": final_logits,
            "relevance_logits": relevance_logits,
            "object_mass": object_mass.squeeze(-1),
            "compatibility": compatibility,
            "compatibility_prior": compatibility_prior,
            "pair_residual": pair_residual,
            "adjustment": adjustment,
        }


class POGUISE(pl.LightningModule):
    def __init__(self, net_size="t", pretrained=None, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.mode = self.hparams.get("mode", "train")
        self.actor_prompt = bool(self.hparams.get("actor_prompt", 0))
        self.actor_interaction_heatmaps = bool(
            self.hparams.get("actor_interaction_heatmaps", 0)
        )
        self.scene_object_tokens = bool(self.hparams.get("scene_object_tokens", 0))
        self.actor_object_slot_head_enabled = bool(
            self.hparams.get("actor_object_slot_head", 0)
        )
        self.actor_object_logit_residual = False
        self.actor_object_conditioned_action = False
        if self.scene_object_tokens and not self.actor_prompt:
            raise ValueError("scene_object_tokens requires actor_prompt")
        if self.actor_object_slot_head_enabled and not self.actor_prompt:
            raise ValueError("actor_object_slot_head requires actor_prompt")
        if self.actor_object_slot_head_enabled and self.scene_object_tokens:
            raise ValueError(
                "actor_object_slot_head and scene_object_tokens are mutually "
                "exclusive. The slot head is a late sidecar and detector object "
                "tokens must not enter the main transformer trunk."
            )
        if self.actor_interaction_heatmaps and not self.actor_prompt:
            raise ValueError("actor_interaction_heatmaps requires actor_prompt")
        if "interaction_object_classes" in self.hparams:
            raise ValueError(
                "interaction_object_classes was removed. Actor-object heatmaps "
                "are now one interacted-object channel per actor; object class "
                "semantics come from scene object tokens."
            )
        if self.hparams.get("interaction_warmup_freeze_actor_path", 0):
            if not (self.actor_prompt and self.actor_interaction_heatmaps):
                raise ValueError(
                    "interaction_warmup_freeze_actor_path requires actor_prompt "
                    "and actor_interaction_heatmaps"
                )
            if not self.hparams.get("freeze_backbone", 0):
                raise ValueError(
                    "interaction_warmup_freeze_actor_path requires freeze_backbone"
                )
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
                actor_interaction_heatmaps=self.actor_interaction_heatmaps,
                scene_object_tokens=self.scene_object_tokens,
                num_scene_object_tokens=self.hparams.get("num_scene_object_tokens", 32),
                num_object_classes=self.hparams.get("num_object_classes", 19),
                return_heatmap_features=False,
                trt_safe_attention=self.hparams.get("trt_safe_attention", 0),
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
                actor_interaction_heatmaps=self.actor_interaction_heatmaps,
                scene_object_tokens=self.scene_object_tokens,
                num_scene_object_tokens=self.hparams.get("num_scene_object_tokens", 32),
                num_object_classes=self.hparams.get("num_object_classes", 19),
                return_heatmap_features=False,
                trt_safe_attention=self.hparams.get("trt_safe_attention", 0),
            )
        if self.hparams.pretrained == "DEFAULT":
            if os.path.exists("vit_b_k710_dl_from_giant.pth"):
                print("loading from disk")
                state_dict = torch.load("vit_b_k710_dl_from_giant.pth")
            else:
                state_dict = torch.hub.load_state_dict_from_url(
                    "https://huggingface.co/OpenGVLab/VideoMAE2/resolve/main/distill/vit_b_k710_dl_from_giant.pth",
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
            self.actor_head = nn.Linear(
                self.net.num_features,
                self.hparams.num_classes,
            )
            self.actor_motion_head = (
                nn.Linear(self.net.num_features, self.hparams.num_classes)
                if (
                    not self.actor_object_slot_head_enabled
                    and float(self.hparams.get("motion_aux_loss_weight", 0.25)) > 0.0
                )
                else None
            )
            self.presence_head = (
                nn.Linear(self.net.num_features, 1)
                if self.hparams.get("actor_presence_head", 0)
                else None
            )
            self.object_selection_head = (
                ActorObjectSelectionHead(self.net.num_features)
                if self.scene_object_tokens
                else None
            )
            object_action_indices = self._object_action_indices()
            object_action_indices_by_object = self._object_action_indices_by_object()
            if self.scene_object_tokens and not object_action_indices:
                raise ValueError(
                    "scene_object_tokens requires a Toyota action-object taxonomy "
                    "mapping for the compatibility expert."
                )
            self.actor_object_relation = (
                ActorObjectCompatibilityExpert(
                    self.net.num_features,
                    num_classes=self.hparams.num_classes,
                    object_action_indices=object_action_indices_by_object,
                    allowed_action_indices=object_action_indices,
                    num_object_classes=self.hparams.get(
                        "num_object_classes",
                        NUM_OBJECT_CLASSES,
                    ),
                    hidden_dim=int(
                        self.hparams.get("actor_object_relation_hidden_dim", 512)
                    ),
                    relevance_gate_init=float(
                        self.hparams.get("actor_object_relevance_gate_init", -2.0)
                    ),
                    compatibility_eps=float(
                        self.hparams.get("actor_object_compatibility_eps", 0.05)
                    ),
                    compatibility_scale=float(
                        self.hparams.get("actor_object_compatibility_scale", 1.0)
                    ),
                    pair_residual_scale=float(
                        self.hparams.get("actor_object_pair_residual_scale", 1.0)
                    ),
                )
                if self.scene_object_tokens
                else None
            )
            if self.actor_object_slot_head_enabled:
                slot_spec = self._actor_object_slot_spec()
                if not slot_spec.action_to_object_ids:
                    raise ValueError(
                        "actor_object_slot_head requires a Toyota action-object "
                        "taxonomy mapping."
                    )
                self.actor_object_slot_head = TRTFriendlyActorObjectSlotHead(
                    self.net.num_features,
                    spec=slot_spec,
                    hidden_dim=int(
                        self.hparams.get("actor_object_slot_hidden_dim", 256)
                    ),
                    relation_rank=int(
                        self.hparams.get("actor_object_slot_relation_rank", 64)
                    ),
                    prior_compatible=float(
                        self.hparams.get("actor_object_slot_prior_compatible", 1.25)
                    ),
                    prior_incompatible=float(
                        self.hparams.get("actor_object_slot_prior_incompatible", -1.25)
                    ),
                    unknown_init_bias=float(
                        self.hparams.get("actor_object_slot_unknown_init_bias", -0.25)
                    ),
                    unknown_mismatch_penalty=float(
                        self.hparams.get(
                            "actor_object_slot_unknown_mismatch_penalty",
                            1.0,
                        )
                    ),
                    relation_scale=float(
                        self.hparams.get("actor_object_slot_relation_scale", 1.0)
                    ),
                    quality_scale=float(
                        self.hparams.get("actor_object_slot_quality_scale", 0.5)
                    ),
                    use_hard_incompatible_mask=bool(
                        self.hparams.get(
                            "actor_object_slot_hard_incompatible_mask",
                            0,
                        )
                    ),
                )
            else:
                self.actor_object_slot_head = None
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

    def _toyota_action_settings(self):
        task_type = self.hparams.get("task_type", "CS")
        action_taxonomy = self.hparams.get("toyota_action_taxonomy", "toyota_31")
        return task_type, action_taxonomy

    def _object_action_indices(self):
        if self.hparams.get("dataset", None) != "toyotasm":
            return []
        task_type, action_taxonomy = self._toyota_action_settings()
        action_to_index = toyota_action_to_index(task_type, action_taxonomy)
        action_object_map = toyota_action_object_map(task_type, action_taxonomy)
        output = []
        for action_name in action_object_map:
            action_idx = action_to_index.get(action_name)
            if action_idx is not None:
                output.append(int(action_idx))
        return sorted(set(output))

    def _object_action_indices_by_object(self):
        if self.hparams.get("dataset", None) != "toyotasm":
            return {}
        task_type, action_taxonomy = self._toyota_action_settings()
        action_to_index = toyota_action_to_index(task_type, action_taxonomy)
        action_object_map = toyota_action_object_map(task_type, action_taxonomy)
        output = {}
        for action_name, object_names in action_object_map.items():
            action_idx = action_to_index.get(action_name)
            if action_idx is None:
                continue
            for object_name in object_names:
                object_id = OBJECT_TO_ID.get(object_name)
                if object_id is None:
                    continue
                output.setdefault(int(object_id), set()).add(int(action_idx))
        return {
            object_id: sorted(action_indices)
            for object_id, action_indices in output.items()
        }

    def _actor_object_slot_spec(self):
        if self.hparams.get("dataset", None) != "toyotasm":
            raise ValueError("actor_object_slot_head currently requires toyotasm")
        task_type, action_taxonomy = self._toyota_action_settings()
        action_to_index = toyota_action_to_index(task_type, action_taxonomy)
        action_object_map = toyota_action_object_map(task_type, action_taxonomy)
        objectless_names = toyota_objectless_action_names(task_type, action_taxonomy)

        objectless_indices = []
        for action_name in objectless_names:
            action_idx = action_to_index.get(action_name)
            if action_idx is not None:
                objectless_indices.append(int(action_idx))

        action_to_object_ids = {}
        for action_name, object_names in action_object_map.items():
            action_idx = action_to_index.get(action_name)
            if action_idx is None:
                continue
            object_ids = []
            for object_name in object_names:
                object_id = OBJECT_TO_ID.get(object_name)
                if object_id is not None:
                    object_ids.append(int(object_id))
            if object_ids:
                action_to_object_ids[int(action_idx)] = tuple(sorted(set(object_ids)))

        return ActionObjectSlotSpec(
            num_actions=int(self.hparams.num_classes),
            num_object_classes=int(
                self.hparams.get("num_object_classes", NUM_OBJECT_CLASSES)
            ),
            objectless_action_indices=tuple(sorted(set(objectless_indices))),
            action_to_object_ids=action_to_object_ids,
        )

    def _freeze_backbone(self):
        print("Freezing backbone")
        # Freeze the backbone
        for param in self.net.parameters():
            param.requires_grad = False
        if (
            self.actor_interaction_heatmaps
            and float(self.hparams.get("lr_head_hm", 0.0)) > 0.0
            and hasattr(self.net, "heatmap_head")
        ):
            for param in self.net.heatmap_head.parameters():
                param.requires_grad = True
        interaction_warmup_freeze_actor_path = (
            self.actor_prompt
            and self.actor_interaction_heatmaps
            and bool(self.hparams.get("interaction_warmup_freeze_actor_path", 0))
        )
        if interaction_warmup_freeze_actor_path:
            print(
                "Interaction warmup: freezing base actor path; training interaction "
                "heatmap/object-token path and optional late transformer blocks only."
            )
            for param in self.head.parameters():
                param.requires_grad = False
            if self.actor_prompt:
                for param in self.actor_head.parameters():
                    param.requires_grad = False
                if self.actor_motion_head is not None:
                    for param in self.actor_motion_head.parameters():
                        param.requires_grad = False
                if self.presence_head is not None:
                    for param in self.presence_head.parameters():
                        param.requires_grad = False
        if self.scene_object_tokens:
            object_param_names = (
                "object_slot_embed",
                "object_cls_embed",
                "object_valid_embed",
                "object_bbox_mlp",
                "object_conf_mlp",
                "object_visual_proj",
            )
            for name, param in self.net.named_parameters():
                if name.startswith(object_param_names):
                    param.requires_grad = True
            if self.object_selection_head is not None:
                for param in self.object_selection_head.parameters():
                    param.requires_grad = True
            if self.actor_object_relation is not None:
                for param in self.actor_object_relation.parameters():
                    param.requires_grad = True
            if self.actor_object_slot_head is not None:
                for param in self.actor_object_slot_head.parameters():
                    param.requires_grad = True
        interaction_unfreeze_last_blocks = int(
            self.hparams.get("interaction_unfreeze_last_blocks", 0) or 0
        )
        if self.actor_interaction_heatmaps and interaction_unfreeze_last_blocks > 0:
            blocks = getattr(self.net, "blocks", None)
            if blocks is None:
                raise ValueError("interaction_unfreeze_last_blocks requires net.blocks")
            if interaction_unfreeze_last_blocks > len(blocks):
                raise ValueError(
                    "interaction_unfreeze_last_blocks exceeds transformer depth: "
                    f"{interaction_unfreeze_last_blocks} > {len(blocks)}"
                )
            print(
                "Interaction heatmap path: unfreezing last "
                f"{interaction_unfreeze_last_blocks} transformer blocks."
            )
            for block in blocks[-interaction_unfreeze_last_blocks:]:
                for param in block.parameters():
                    param.requires_grad = True
            if getattr(self.net, "fc_norm", None) is not None:
                for param in self.net.fc_norm.parameters():
                    param.requires_grad = True
            if getattr(self.net, "norm", None) is not None:
                for param in self.net.norm.parameters():
                    param.requires_grad = True
        # Unfreeze the head
        if not interaction_warmup_freeze_actor_path:
            for param in self.head.parameters():
                param.requires_grad = True
        if self.actor_prompt:
            if not interaction_warmup_freeze_actor_path:
                for param in self.actor_head.parameters():
                    param.requires_grad = True
                if self.actor_motion_head is not None:
                    for param in self.actor_motion_head.parameters():
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
                if self.object_selection_head is not None:
                    for param in self.object_selection_head.parameters():
                        param.requires_grad = True
                if self.actor_object_relation is not None:
                    for param in self.actor_object_relation.parameters():
                        param.requires_grad = True
                if self.actor_object_slot_head is not None:
                    for param in self.actor_object_slot_head.parameters():
                        param.requires_grad = True

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

    def forward(
        self,
        x,
        boxes=None,
        valid=None,
        action_labels=None,
        object_boxes=None,
        object_classes=None,
        object_confs=None,
        object_valid=None,
        interaction_object_index=None,
        interaction_object_index_valid=None,
        object_heatmap_scores=None,
    ):
        # convert to b c t h w
        x = x.permute(0, 2, 1, 3, 4)
        if self.actor_prompt:
            object_kwargs = {}
            if self.scene_object_tokens:
                object_kwargs = {
                    "object_boxes": object_boxes,
                    "object_classes": object_classes,
                    "object_confs": object_confs,
                    "object_valid": object_valid,
                }
            if self.hparams.n_landmarks > 0 or self.actor_interaction_heatmaps:
                net_data = self.net(
                    x,
                    boxes=boxes,
                    valid=valid,
                    **object_kwargs,
                )
                if self.scene_object_tokens:
                    if len(net_data) == 5:
                        _, x_actor, x_object, x_heatmap, _ = net_data
                    else:
                        _, x_actor, x_object, x_heatmap = net_data
                else:
                    if len(net_data) == 4:
                        _, x_actor, x_heatmap, _ = net_data
                    else:
                        _, x_actor, x_heatmap = net_data
            else:
                data = self.net(
                    x,
                    boxes=boxes,
                    valid=valid,
                    **object_kwargs,
                )
                if self.scene_object_tokens:
                    _, x_actor, x_object = data[:3]
                else:
                    _, x_actor = data[:2]
                x_heatmap = 0
            object_selection_logits = None
            if self.object_selection_head is not None:
                if object_valid is None:
                    raise ValueError(
                        "object_valid is required when scene_object_tokens is enabled"
                    )
                object_selection_logits = self.object_selection_head(
                    x_actor,
                    x_object,
                    object_valid.to(device=x_actor.device, dtype=torch.bool),
                )
            self.last_actor_object_relation_delta = None
            self.last_actor_object_relevance_logits = None
            self.last_actor_object_relation_mass = None
            self.last_actor_object_compatibility = None
            self.last_actor_object_compatibility_prior = None
            self.last_actor_object_pair_residual_logits = None
            self.last_actor_object_compatibility_adjustment = None
            self.last_actor_object_slot_delta = None
            self.last_actor_object_slot_posterior = None
            self.last_actor_object_best_slot = None
            self.last_actor_object_quality = None
            self.last_actor_object_mismatch = None
            self.last_actor_object_unknown_delta = None
            self.last_actor_object_object_slot_delta = None
            self.last_actor_motion_logits = None
            if self.actor_motion_head is not None:
                self.last_actor_motion_logits = self.actor_motion_head(x_actor)
            if self.hparams.ret_feat:
                return x_actor
            base_action_logits = self.actor_head(x_actor)
            self.last_actor_action_logits = base_action_logits
            action_logits = base_action_logits
            if self.actor_object_slot_head is not None:
                self.last_actor_motion_logits = base_action_logits
                if boxes is None:
                    raise ValueError("actor_object_slot_head requires actor boxes")
                if object_boxes is None or object_classes is None:
                    raise ValueError(
                        "actor_object_slot_head requires object_boxes and "
                        "object_classes"
                    )
                if object_confs is None or object_valid is None:
                    raise ValueError(
                        "actor_object_slot_head requires object_confs and "
                        "object_valid"
                    )
                if object_heatmap_scores is None:
                    object_heatmap_scores = torch.zeros(
                        (
                            x_actor.shape[0],
                            x_actor.shape[1],
                            object_boxes.shape[1],
                        ),
                        device=x_actor.device,
                        dtype=x_actor.dtype,
                    )
                slot_output = self.actor_object_slot_head(
                    actor_tokens=x_actor,
                    motion_logits=base_action_logits,
                    actor_boxes=boxes,
                    object_boxes=object_boxes,
                    object_classes=object_classes,
                    object_confs=object_confs,
                    object_valid=object_valid,
                    object_heatmap_scores=object_heatmap_scores,
                )
                action_logits = slot_output["logits"]
                self.last_actor_object_slot_delta = slot_output["slot_delta"]
                self.last_actor_object_slot_posterior = slot_output[
                    "slot_posterior"
                ]
                self.last_actor_object_best_slot = slot_output["best_slot"]
                self.last_actor_object_quality = slot_output["object_quality"]
                self.last_actor_object_mismatch = slot_output["mismatch"]
                self.last_actor_object_unknown_delta = slot_output["unknown_delta"]
                self.last_actor_object_object_slot_delta = slot_output[
                    "object_slot_delta"
                ]
                self.last_actor_object_relation_delta = (
                    action_logits - base_action_logits
                )
            if self.actor_object_relation is not None:
                if object_valid is None:
                    raise ValueError(
                        "object_valid is required when scene_object_tokens is enabled"
                    )
                if object_classes is None:
                    raise ValueError(
                        "object_classes is required when scene_object_tokens is enabled"
                    )
                expert_output = self.actor_object_relation(
                    x_actor,
                    x_object,
                    object_valid.to(device=x_actor.device, dtype=torch.bool),
                    object_selection_logits,
                    object_classes=object_classes,
                    base_logits=base_action_logits,
                )
                action_logits = expert_output["logits"]
                self.last_actor_object_relevance_logits = expert_output[
                    "relevance_logits"
                ]
                self.last_actor_object_relation_mass = expert_output["object_mass"]
                self.last_actor_object_compatibility = expert_output["compatibility"]
                self.last_actor_object_compatibility_prior = expert_output[
                    "compatibility_prior"
                ]
                self.last_actor_object_pair_residual_logits = expert_output[
                    "pair_residual"
                ]
                self.last_actor_object_compatibility_adjustment = expert_output[
                    "adjustment"
                ]
                self.last_actor_object_relation_delta = expert_output["adjustment"]
            if self.presence_head is not None:
                presence_logits = self.presence_head(x_actor).squeeze(-1)
                if object_selection_logits is not None:
                    return (
                        action_logits,
                        x_heatmap,
                        presence_logits,
                        object_selection_logits,
                    )
                return action_logits, x_heatmap, presence_logits
            if object_selection_logits is not None:
                return action_logits, x_heatmap, None, object_selection_logits
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
        parser.add_argument("--actor_interaction_heatmaps", type=int, default=0)
        parser.add_argument("--scene_object_tokens", type=int, default=0)
        parser.add_argument("--num_scene_object_tokens", type=int, default=32)
        parser.add_argument("--num_object_classes", type=int, default=19)
        parser.add_argument("--actor_object_slot_head", type=int, default=0)
        parser.add_argument("--actor_object_slot_hidden_dim", type=int, default=256)
        parser.add_argument("--actor_object_slot_relation_rank", type=int, default=64)
        parser.add_argument(
            "--actor_object_slot_prior_compatible",
            type=float,
            default=1.25,
        )
        parser.add_argument(
            "--actor_object_slot_prior_incompatible",
            type=float,
            default=-1.25,
        )
        parser.add_argument(
            "--actor_object_slot_unknown_init_bias",
            type=float,
            default=-0.25,
        )
        parser.add_argument(
            "--actor_object_slot_unknown_mismatch_penalty",
            type=float,
            default=1.0,
        )
        parser.add_argument(
            "--actor_object_slot_relation_scale",
            type=float,
            default=1.0,
        )
        parser.add_argument(
            "--actor_object_slot_quality_scale",
            type=float,
            default=0.5,
        )
        parser.add_argument(
            "--actor_object_slot_hard_incompatible_mask",
            type=int,
            default=0,
        )
        parser.add_argument(
            "--actor_object_relation_hidden_dim",
            type=int,
            default=512,
        )
        parser.add_argument(
            "--actor_object_relevance_gate_init",
            type=float,
            default=-2.0,
        )
        parser.add_argument(
            "--actor_object_compatibility_eps",
            type=float,
            default=0.05,
        )
        parser.add_argument(
            "--actor_object_compatibility_scale",
            type=float,
            default=1.0,
        )
        parser.add_argument(
            "--actor_object_pair_residual_scale",
            type=float,
            default=1.0,
        )
        parser.add_argument("--trt_safe_attention", type=int, default=0)
        parser.add_argument("--interaction_unfreeze_last_blocks", type=int, default=0)
        parser.add_argument("--ret_feat", type=int, default=0)
        parser.add_argument("--linear_probe", type=int, default=0)

        return parser
