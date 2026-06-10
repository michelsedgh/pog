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


class ActorObjectRelationLayer(nn.Module):
    def __init__(self, dim, num_heads=8, hidden_dim=None):
        super().__init__()
        if dim % int(num_heads) != 0:
            raise ValueError("dim must be divisible by num_heads")
        hidden_dim = int(hidden_dim or dim * 4)
        self.num_heads = int(num_heads)
        self.head_dim = dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.actor_norm = nn.LayerNorm(dim)
        self.object_norm = nn.LayerNorm(dim)
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self.geometry_bias = nn.Sequential(
            nn.Linear(7, 64),
            nn.GELU(),
            nn.Linear(64, self.num_heads),
        )
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        nn.init.zeros_(self.ffn[-1].weight)
        nn.init.zeros_(self.ffn[-1].bias)
        nn.init.zeros_(self.geometry_bias[-1].weight)
        nn.init.zeros_(self.geometry_bias[-1].bias)

    def _box_geometry_bias(
        self,
        actor_boxes,
        object_boxes,
        object_confs,
        batch,
        num_actors,
        num_objects,
        device,
        dtype,
    ):
        if actor_boxes is None or object_boxes is None:
            return torch.zeros(
                batch,
                self.num_heads,
                num_actors,
                num_objects,
                device=device,
                dtype=dtype,
            )

        actor_boxes = actor_boxes.to(device=device, dtype=dtype)
        object_boxes = object_boxes.to(device=device, dtype=dtype)
        actor_xy1 = actor_boxes[..., :2]
        actor_xy2 = actor_boxes[..., 2:].clamp_min(actor_xy1 + 1e-4)
        object_xy1 = object_boxes[..., :2]
        object_xy2 = object_boxes[..., 2:].clamp_min(object_xy1 + 1e-4)

        actor_ctr = (actor_xy1 + actor_xy2) * 0.5
        object_ctr = (object_xy1 + object_xy2) * 0.5
        actor_size = (actor_xy2 - actor_xy1).clamp_min(1e-4)
        object_size = (object_xy2 - object_xy1).clamp_min(1e-4)

        delta = object_ctr[:, None, :, :] - actor_ctr[:, :, None, :]
        normalized_delta = delta / actor_size[:, :, None, :]
        size_ratio = torch.log(object_size[:, None, :, :] / actor_size[:, :, None, :])
        actor_area = actor_size[..., 0] * actor_size[..., 1]
        object_area = object_size[..., 0] * object_size[..., 1]
        area_ratio = torch.log(object_area[:, None, :] / actor_area[:, :, None])

        inter_xy1 = torch.maximum(actor_xy1[:, :, None, :], object_xy1[:, None, :, :])
        inter_xy2 = torch.minimum(actor_xy2[:, :, None, :], object_xy2[:, None, :, :])
        inter_size = (inter_xy2 - inter_xy1).clamp_min(0.0)
        inter_area = inter_size[..., 0] * inter_size[..., 1]
        union_area = (
            actor_area[:, :, None] + object_area[:, None, :] - inter_area
        ).clamp_min(1e-6)
        iou = inter_area / union_area

        if object_confs is None:
            conf = torch.ones(batch, num_objects, device=device, dtype=dtype)
        else:
            conf = object_confs.to(device=device, dtype=dtype).clamp(0.0, 1.0)
        conf = conf[:, None, :].expand(-1, num_actors, -1)

        features = torch.stack(
            (
                normalized_delta[..., 0],
                normalized_delta[..., 1],
                size_ratio[..., 0],
                size_ratio[..., 1],
                area_ratio,
                iou,
                conf,
            ),
            dim=-1,
        )
        bias = self.geometry_bias(features.float()).to(dtype=dtype)
        return bias.permute(0, 3, 1, 2).contiguous()

    def forward(
        self,
        actor_tokens,
        object_tokens,
        object_valid,
        object_prior,
        actor_boxes=None,
        object_boxes=None,
        object_confs=None,
    ):
        batch, num_actors, dim = actor_tokens.shape
        num_objects = object_tokens.shape[1]
        if object_prior.shape != (batch, num_actors, num_objects):
            raise ValueError(
                "object_prior must have shape "
                f"{(batch, num_actors, num_objects)}, got {tuple(object_prior.shape)}"
            )

        q = self.query(self.actor_norm(actor_tokens))
        k = self.key(self.object_norm(object_tokens))
        v = self.value(self.object_norm(object_tokens))
        q = q.view(batch, num_actors, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, num_objects, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, num_objects, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        prior = object_prior.to(device=scores.device, dtype=scores.dtype).clamp_min(1e-6)
        scores = scores + prior[:, None, :, :].log()
        scores = scores + self._box_geometry_bias(
            actor_boxes,
            object_boxes,
            object_confs,
            batch,
            num_actors,
            num_objects,
            scores.device,
            scores.dtype,
        )

        object_valid = object_valid.to(device=scores.device, dtype=torch.bool)
        mask = object_valid[:, None, None, :]
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        has_object = object_valid.any(dim=-1)
        scores = torch.where(
            has_object[:, None, None, None],
            scores,
            torch.zeros_like(scores),
        )
        attn = scores.float().softmax(dim=-1).to(dtype=scores.dtype)
        attn = attn * mask.to(dtype=attn.dtype)

        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).reshape(batch, num_actors, dim)
        actor_tokens = actor_tokens + self.out(context)
        actor_tokens = actor_tokens + self.ffn(actor_tokens)
        return actor_tokens


class ActorObjectRelationBlock(nn.Module):
    def __init__(
        self,
        dim,
        hidden_dim=512,
        num_layers=2,
        num_heads=8,
        relevance_gate_init=-2.0,
    ):
        super().__init__()
        self.object_norm = nn.LayerNorm(dim)
        self.actor_norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList(
            [
                ActorObjectRelationLayer(dim, num_heads=num_heads, hidden_dim=hidden_dim)
                for _ in range(int(num_layers))
            ]
        )
        self.relevance_head = nn.Sequential(
            nn.LayerNorm(dim * 3 + 3),
            nn.Linear(dim * 3 + 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.relation_scale = nn.Parameter(torch.ones(()))
        nn.init.zeros_(self.relevance_head[-1].weight)
        nn.init.constant_(self.relevance_head[-1].bias, float(relevance_gate_init))

    def _selection_distribution(self, object_selection_logits, object_valid, dtype):
        if object_selection_logits is None:
            raise ValueError(
                "object_selection_logits is required for actor-object relation"
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

    def forward(
        self,
        actor_tokens,
        object_tokens,
        object_valid,
        object_selection_logits,
        actor_boxes=None,
        object_boxes=None,
        object_confs=None,
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
        conditional_object_probs = object_probs / object_mass.clamp_min(1e-6)
        object_context = torch.einsum(
            "bkm,bmd->bkd",
            conditional_object_probs,
            object_tokens,
        )

        relation_actor = actor_tokens
        for layer in self.layers:
            relation_actor = layer(
                relation_actor,
                object_tokens,
                object_valid,
                conditional_object_probs,
                actor_boxes=actor_boxes,
                object_boxes=object_boxes,
                object_confs=object_confs,
            )
        relation_delta = relation_actor - actor_tokens

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
        final_gate = relevance_gate * object_mass.to(dtype=actor_tokens.dtype)
        fused_actor = actor_tokens + self.relation_scale * final_gate * relation_delta
        return fused_actor, relevance_logits, object_mass.squeeze(-1)


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
        self.actor_object_logit_residual = False
        self.actor_object_conditioned_action = False
        if self.scene_object_tokens and not self.actor_prompt:
            raise ValueError("scene_object_tokens requires actor_prompt")
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
                if float(self.hparams.get("motion_aux_loss_weight", 0.25)) > 0.0
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
            self.actor_object_relation = (
                ActorObjectRelationBlock(
                    self.net.num_features,
                    hidden_dim=int(
                        self.hparams.get("actor_object_relation_hidden_dim", 512)
                    ),
                    num_layers=int(
                        self.hparams.get("actor_object_relation_layers", 2)
                    ),
                    num_heads=int(
                        self.hparams.get("actor_object_relation_heads", 8)
                    ),
                    relevance_gate_init=float(
                        self.hparams.get("actor_object_relevance_gate_init", -2.0)
                    ),
                )
                if self.scene_object_tokens
                else None
            )
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
            self.last_actor_motion_logits = None
            if self.actor_motion_head is not None:
                self.last_actor_motion_logits = self.actor_motion_head(x_actor)
            action_actor = x_actor
            if self.actor_object_relation is not None:
                if object_valid is None:
                    raise ValueError(
                        "object_valid is required when scene_object_tokens is enabled"
                    )
                relation_output = self.actor_object_relation(
                    x_actor,
                    x_object,
                    object_valid.to(device=x_actor.device, dtype=torch.bool),
                    object_selection_logits,
                    actor_boxes=boxes,
                    object_boxes=object_boxes,
                    object_confs=object_confs,
                )
                action_actor, relevance_logits, relation_mass = relation_output
                self.last_actor_object_relation_delta = action_actor - x_actor
                self.last_actor_object_relevance_logits = relevance_logits
                self.last_actor_object_relation_mass = relation_mass
            if self.hparams.ret_feat:
                return action_actor
            self.last_actor_action_logits = None
            action_logits = self.actor_head(action_actor)
            self.last_actor_action_logits = action_logits
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
        parser.add_argument("--actor_object_relation_layers", type=int, default=2)
        parser.add_argument("--actor_object_relation_heads", type=int, default=8)
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
        parser.add_argument("--trt_safe_attention", type=int, default=0)
        parser.add_argument("--interaction_unfreeze_last_blocks", type=int, default=0)
        parser.add_argument("--ret_feat", type=int, default=0)
        parser.add_argument("--linear_probe", type=int, default=0)

        return parser
