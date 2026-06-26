import torch.nn as nn
import torch.nn.functional as F
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
    toyota_action_names,
    toyota_action_object_map,
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


class POGUISE(pl.LightningModule):
    def __init__(self, net_size="t", pretrained=None, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.mode = self.hparams.get("mode", "train")
        self.actor_prompt = bool(self.hparams.get("actor_prompt", 0))
        self.actor_interaction_heatmaps = bool(
            self.hparams.get("actor_interaction_heatmaps", 0)
        )
        self.actor_object_prompt_tokens_enabled = bool(
            self.hparams.get("actor_object_prompt_tokens", 0)
        )
        self.actor_object_residual_head_enabled = bool(
            self.hparams.get("actor_object_residual_head", 0)
        )

        removed_action_prior_keys = (
            "actor_object_action_prior_weight",
            "actor_object_action_prior_negative_weight",
            "actor_object_action_prior_map",
            "object_action_prior_weight",
            "object_action_prior_negative_weight",
        )
        stale_action_prior = [
            key
            for key in removed_action_prior_keys
            if key in self.hparams
            and self.hparams.get(key) not in (None, 0, 0.0, "0", "0.0")
        ]
        if stale_action_prior:
            raise ValueError(
                "Manual object-to-action logit priors were removed. The clean "
                "path is PO-GUISE+ video/heatmap features plus learned HOI-style "
                "actor-object relation binding and learned pair action scoring. "
                f"Remove stale hparams: {', '.join(stale_action_prior)}"
            )
        if self.actor_object_prompt_tokens_enabled and not self.actor_prompt:
            raise ValueError("actor_object_prompt_tokens requires actor_prompt")
        if (
            self.actor_object_prompt_tokens_enabled
            and not self.actor_interaction_heatmaps
        ):
            raise ValueError(
                "actor_object_prompt_tokens requires actor_interaction_heatmaps"
            )

        if "interaction_object_classes" in self.hparams:
            raise ValueError(
                "interaction_object_classes was removed. Actor-object heatmaps "
                "are now one interacted-object channel per actor; object class "
                "semantics come from relation-only runtime object memory."
            )
        self.use_register_tokens = bool(self.hparams.get("use_register_tokens", 0))
        self._create_network()
        # freeze backbone if specified
        if self.hparams.freeze_backbone:
            self._freeze_backbone()

    def load_state_dict(self, state_dict, strict=True, assign=False):
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        stale_object_conf = [
            key for key in result.unexpected_keys if "object_conf_mlp" in key
        ]
        if stale_object_conf:
            preview = ", ".join(stale_object_conf[:12])
            raise RuntimeError(
                "Checkpoint contains removed learned object-confidence embedding "
                "weights. Detector confidence is no longer a learned actor-model "
                "feature; thresholded object_valid proposals provide object "
                "existence evidence. Retrain from a clean base checkpoint. "
                f"First stale keys: {preview}"
            )
        return result

    def _create_network(self):
        n_registers = (
            int(self.hparams.get("n_registers", 0) or 0)
            if self.use_register_tokens
            else 0
        )
        all_frames = int(self.hparams.get("n_frames", 16))
        if all_frames <= 0:
            raise ValueError("n_frames must be positive")

        if self.hparams.pretrained == "small":
            return_heatmap_features = bool(self.actor_object_prompt_tokens_enabled)
            self.net = vit_small_patch16_224(
                all_frames=all_frames,
                drop_rate=self.hparams.drop_rate,
                attn_drop_rate=self.hparams.attn_drop_rate,
                drop_path_rate=self.hparams.drop_path_rate,
                head_drop_rate=self.hparams.head_drop_rate,

                mode=self.mode,
                hw_out_conv=self.hparams.hw_out_conv,
                n_registers=n_registers,
                actor_prompt=self.actor_prompt,
                num_actor_tokens=self.hparams.get("num_actor_tokens", 8),
                actor_interaction_heatmaps=self.actor_interaction_heatmaps,
                actor_object_prompt_tokens=self.actor_object_prompt_tokens_enabled,
                num_scene_object_tokens=self.hparams.get("num_scene_object_tokens", 32),
                num_object_classes=self.hparams.get(
                    "num_object_classes",
                    NUM_OBJECT_CLASSES,
                ),

                return_heatmap_features=return_heatmap_features,
                trt_safe_attention=self.hparams.get("trt_safe_attention", 0),
            )
        else:
            return_heatmap_features = bool(self.actor_object_prompt_tokens_enabled)
            self.net = vit_base_patch16_224(
                all_frames=all_frames,
                drop_rate=self.hparams.drop_rate,
                attn_drop_rate=self.hparams.attn_drop_rate,
                drop_path_rate=self.hparams.drop_path_rate,
                head_drop_rate=self.hparams.head_drop_rate,

                mode=self.mode,
                hw_out_conv=self.hparams.hw_out_conv,
                n_registers=n_registers,
                actor_prompt=self.actor_prompt,
                num_actor_tokens=self.hparams.get("num_actor_tokens", 8),
                actor_interaction_heatmaps=self.actor_interaction_heatmaps,
                actor_object_prompt_tokens=self.actor_object_prompt_tokens_enabled,
                num_scene_object_tokens=self.hparams.get("num_scene_object_tokens", 32),
                num_object_classes=self.hparams.get(
                    "num_object_classes",
                    NUM_OBJECT_CLASSES,
                ),

                return_heatmap_features=return_heatmap_features,
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
            self.actor_object_pair_action_head = None
            if self.hparams.get("actor_object_pair_action_head", 0):
                self.actor_head = None
                self.actor_object_relation_in_transformer = True
                self.actor_object_null_pair_token = nn.Parameter(
                    torch.zeros(1, 1, 1, self.net.num_features)
                )
                self.actor_object_pair_action_head = nn.Linear(
                    self.net.num_features * 2,
                    self.hparams.num_classes,
                )
            else:
                self.actor_head = nn.Linear(
                    self.net.num_features,
                    self.hparams.num_classes,
                )
            self.presence_head = (
                nn.Linear(self.net.num_features, 1)
                if self.hparams.get("actor_presence_head", 0)
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
        # Unfreeze the head
        for param in self.head.parameters():
            param.requires_grad = True
        if self.actor_prompt:
            if self.actor_head is not None:
                for param in self.actor_head.parameters():
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
            for attr in (
                "object_slot_embed",
                "object_class_embed",
                "object_box_mlp",
                "object_valid_embed",
            ):
                module = getattr(self.net, attr, None)
                if module is None:
                    continue
                if isinstance(module, nn.Parameter):
                    module.requires_grad = True
                else:
                    for param in module.parameters():
                        param.requires_grad = True
            if self.presence_head is not None:
                for param in self.presence_head.parameters():
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
        object_valid=None,
    ):
        # convert to b c t h w
        x = x.permute(0, 2, 1, 3, 4)
        if self.actor_prompt:
            if self.hparams.n_landmarks > 0 or self.actor_interaction_heatmaps:
                net_data = self.net(
                    x,
                    boxes=boxes,
                    valid=valid,
                    object_boxes=object_boxes,
                    object_classes=object_classes,
                    object_valid=object_valid,
                )
                x_heatmap_feat = None
                x_visual_final = None
                x_object_prompt = None
                if len(net_data) == 6:
                    (
                        _,
                        x_actor,
                        x_heatmap,
                        x_heatmap_feat,
                        x_visual_final,
                        x_object_prompt,
                    ) = net_data
                elif len(net_data) == 5:
                    _, x_actor, x_heatmap, x_heatmap_feat, x_visual_final = net_data
                elif len(net_data) == 4:
                    _, x_actor, x_heatmap, x_heatmap_feat = net_data
                else:
                    _, x_actor, x_heatmap = net_data
            else:
                data = self.net(
                    x,
                    boxes=boxes,
                    valid=valid,
                    object_boxes=object_boxes,
                    object_classes=object_classes,
                    object_valid=object_valid,
                )
                _, x_actor = data[:2]
                x_heatmap = 0
                x_heatmap_feat = None
                x_visual_final = None
                x_object_prompt = None
            self.last_actor_tokens = x_actor
            self.last_actor_action_tokens = x_actor
            if self.actor_object_pair_action_head is not None and x_object_prompt is not None:
                batch_size, num_actors, feature_dim = x_actor.shape
                num_objects = x_object_prompt.shape[1]
                
                null_feat = self.actor_object_null_pair_token.expand(batch_size, 1, 1, feature_dim).squeeze(1)
                all_objects = torch.cat([null_feat, x_object_prompt], dim=1)
                
                relation_logits = torch.einsum('bac,boc->bao', x_actor, all_objects)
                relation_log_probs = F.log_softmax(relation_logits, dim=-1)
                
                actor_expanded = x_actor.unsqueeze(2).expand(batch_size, num_actors, num_objects + 1, feature_dim)
                objects_expanded = all_objects.unsqueeze(1).expand(batch_size, num_actors, num_objects + 1, feature_dim)
                
                pair_features = torch.cat([actor_expanded, objects_expanded], dim=-1)
                pair_action_logits = self.actor_object_pair_action_head(pair_features)
                
                # Zero out logits for invalid objects to prevent them from affecting logsumexp
                # object_valid shape is [B, 32]. We need to prepend a True for the NULL object.
                if object_valid is not None:
                    valid_mask = torch.cat([torch.ones(batch_size, 1, dtype=torch.bool, device=x.device), object_valid.to(torch.bool)], dim=1)
                    valid_mask = valid_mask.unsqueeze(1).unsqueeze(-1) # [B, 1, 33, 1]
                    pair_action_logits = pair_action_logits.masked_fill(~valid_mask, -1.0e4)
                    relation_log_probs = relation_log_probs.masked_fill(~valid_mask.squeeze(-1), -1.0e4)
                    # Re-normalize log probs
                    relation_log_probs = F.log_softmax(relation_log_probs, dim=-1)
                    
                pair_scores = pair_action_logits + relation_log_probs.unsqueeze(-1)
                action_scores = torch.logsumexp(pair_scores, dim=2)
                
                self.last_actor_object_pair_action_logits = pair_action_logits
                self.last_actor_object_pair_action_scores = pair_scores
                self.last_actor_object_pair_action_log_probs = relation_log_probs
                
                # Make allowed mask for dashboard
                if object_valid is not None:
                    allowed = torch.cat([torch.ones(batch_size, num_actors, 1, dtype=torch.bool, device=x.device), object_valid.unsqueeze(1).expand(batch_size, num_actors, num_objects)], dim=2)
                else:
                    allowed = torch.ones(batch_size, num_actors, num_objects + 1, dtype=torch.bool, device=x.device)
                self.last_actor_object_pair_action_allowed = allowed
            else:
                if self.actor_head is None:
                    raise RuntimeError("actor_head is not initialized")
                action_scores = self.actor_head(x_actor)
            self.last_actor_action_logits = action_scores
            if self.presence_head is not None:
                presence_logits = self.presence_head(x_actor).squeeze(-1)
                return action_scores, x_heatmap, presence_logits
            return action_scores, x_heatmap
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

        # parser.add_argument("--enhanced_weight_class_obj", type=float, default=1)
        parser.add_argument("--hw_out_conv", type=int, default=8)
        parser.add_argument("--use_register_tokens", type=int, default=0)
        parser.add_argument("--n_registers", type=int, default=0)
        parser.add_argument("--actor_prompt", type=int, default=0)
        parser.add_argument("--num_actor_tokens", type=int, default=8)
        parser.add_argument("--actor_presence_head", type=int, default=0)
        parser.add_argument("--presence_loss_weight", type=float, default=0.05)
        parser.add_argument("--actor_pair_train_weight", type=float, default=0.0)
        parser.add_argument("--actor_val_diagnostics", type=int, default=1)
        parser.add_argument("--actor_val_diagnostic_max_pairs", type=int, default=8)
        parser.add_argument("--actor_interaction_heatmaps", type=int, default=0)
        parser.add_argument("--num_scene_object_tokens", type=int, default=32)
        parser.add_argument("--num_object_classes", type=int, default=19)
        parser.add_argument("--actor_object_prompt_tokens", type=int, default=0)

        parser.add_argument("--trt_safe_attention", type=int, default=0)
        parser.add_argument("--ret_feat", type=int, default=0)
        parser.add_argument("--linear_probe", type=int, default=0)

        return parser
