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

from datasets.object_vocab import OBJECT_SUMMARY_FEATURE_DIM


from collections import OrderedDict
import json


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
        self.object_prompt = bool(self.hparams.get("object_prompt", 0))
        if self.object_prompt and not self.actor_prompt:
            raise ValueError("object_prompt requires actor_prompt")
        for prob_name in ("object_dropout_prob", "object_token_dropout_prob"):
            prob_value = float(self.hparams.get(prob_name, 0.0))
            if not 0 <= prob_value <= 1:
                raise ValueError(f"{prob_name} must be in [0, 1]")
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
                object_prompt=self.object_prompt,
                num_object_tokens=self.hparams.get("num_object_tokens", 24),
                num_object_classes=self.hparams.get("num_object_classes", 0),
                object_bbox_prior_weight=self.hparams.get(
                    "object_bbox_prior_weight", 0.0
                ),
                object_bbox_prior_expand=self.hparams.get(
                    "object_bbox_prior_expand", 1.25
                ),
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
                object_prompt=self.object_prompt,
                num_object_tokens=self.hparams.get("num_object_tokens", 24),
                num_object_classes=self.hparams.get("num_object_classes", 0),
                object_bbox_prior_weight=self.hparams.get(
                    "object_bbox_prior_weight", 0.0
                ),
                object_bbox_prior_expand=self.hparams.get(
                    "object_bbox_prior_expand", 1.25
                ),
            )
        if self.hparams.pretrained == "DEFAULT":
            if os.path.exists("vit_b_k710_dl_from_giant.pth"):
                print("loading from disk")
                state_dict = torch.load("vit_b_k710_dl_from_giant.pth")
            else:
                state_dict = torch.hub.load_state_dict_from_url(
                    "https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/internvideo/distill/vit_b_k710_dl_from_giant.pth",
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
            self.actor_head = nn.Linear(self.net.num_features, self.hparams.num_classes)
            self.presence_head = (
                nn.Linear(self.net.num_features, 1)
                if self.hparams.get("actor_presence_head", 0)
                else None
            )
            if self.object_prompt:
                self._create_object_prompt_modules()
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
        # Unfreeze the head
        for param in self.head.parameters():
            param.requires_grad = True
        if self.actor_prompt:
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
            if self.presence_head is not None:
                for param in self.presence_head.parameters():
                    param.requires_grad = True
            if self.object_prompt:
                for module in (
                    self.object_summary_norm,
                    self.object_summary_delta,
                    self.interaction_head,
                ):
                    for param in module.parameters():
                        param.requires_grad = True
                self.object_summary_gate_logit.requires_grad = True

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

    def _create_object_prompt_modules(self):
        num_object_classes = int(self.hparams.get("num_object_classes", 0))
        if num_object_classes <= 0:
            raise ValueError("object_prompt requires num_object_classes > 0")
        dim = self.net.num_features
        self.num_object_classes = num_object_classes
        summary_dim = OBJECT_SUMMARY_FEATURE_DIM * num_object_classes
        hidden_dim = int(self.hparams.get("object_summary_hidden_dim", 256))
        if hidden_dim <= 0:
            raise ValueError("object_summary_hidden_dim must be positive")
        dropout = float(self.hparams.get("object_summary_dropout", 0.15))
        if not 0 <= dropout < 1:
            raise ValueError("object_summary_dropout must be in [0, 1)")
        gate_init = float(self.hparams.get("object_summary_gate_init", 0.12))
        if not 0 < gate_init < 1:
            raise ValueError("object_summary_gate_init must be in (0, 1)")

        self.object_summary_dim = summary_dim
        self.object_summary_norm = nn.LayerNorm(summary_dim)
        self.object_summary_delta = nn.Sequential(
            nn.Linear(summary_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.hparams.num_classes),
        )
        self.object_summary_gate_logit = nn.Parameter(
            torch.logit(torch.tensor(gate_init, dtype=torch.float32))
        )
        self.interaction_head = nn.Linear(dim, num_object_classes + 1)

        nn.init.zeros_(self.object_summary_delta[-1].weight)
        nn.init.zeros_(self.object_summary_delta[-1].bias)
        nn.init.zeros_(self.interaction_head.weight)
        nn.init.zeros_(self.interaction_head.bias)

    def _apply_object_dropout(self, object_valid):
        if object_valid is None or not self.training:
            return object_valid
        object_valid = object_valid.clone()
        object_dropout_prob = float(self.hparams.get("object_dropout_prob", 0.0))
        object_token_dropout_prob = float(
            self.hparams.get("object_token_dropout_prob", 0.0)
        )
        if object_dropout_prob > 0:
            if torch.rand((), device=object_valid.device) < object_dropout_prob:
                object_valid.fill_(False)
        if object_token_dropout_prob > 0:
            drop_each = (
                torch.rand(object_valid.shape, device=object_valid.device)
                < object_token_dropout_prob
            )
            object_valid = object_valid & ~drop_each
        return object_valid

    def _apply_object_summary_dropout(self, object_summary):
        if object_summary is None or not self.training:
            return object_summary
        object_summary = object_summary.clone()
        object_dropout_prob = float(self.hparams.get("object_dropout_prob", 0.0))
        object_token_dropout_prob = float(
            self.hparams.get("object_token_dropout_prob", 0.0)
        )
        if object_dropout_prob > 0:
            if torch.rand((), device=object_summary.device) < object_dropout_prob:
                object_summary.zero_()
        if object_token_dropout_prob > 0:
            summary = object_summary.view(
                object_summary.shape[0],
                OBJECT_SUMMARY_FEATURE_DIM,
                self.num_object_classes,
            )
            drop_cls = (
                torch.rand(
                    summary.shape[0],
                    1,
                    summary.shape[2],
                    device=summary.device,
                )
                < object_token_dropout_prob
            )
            summary = summary.masked_fill(drop_cls, 0.0)
            object_summary = summary.reshape(object_summary.shape)
        return object_summary

    def _object_summary_fusion(self, action_logits, object_summary=None):
        if object_summary is None:
            return action_logits
        object_summary = object_summary.to(
            device=action_logits.device,
            dtype=action_logits.dtype,
        )
        if object_summary.ndim != 2 or object_summary.shape[1] != self.object_summary_dim:
            raise ValueError(
                "object_summary must have shape "
                f"[B,{self.object_summary_dim}], got {tuple(object_summary.shape)}"
            )
        normalized = self.object_summary_norm(object_summary)
        delta = self.object_summary_delta(normalized)
        gate = torch.sigmoid(self.object_summary_gate_logit).to(
            device=action_logits.device,
            dtype=action_logits.dtype,
        )
        return action_logits + gate * delta

    def forward(
        self,
        x,
        boxes=None,
        valid=None,
        object_boxes=None,
        object_cls=None,
        object_conf=None,
        object_valid=None,
        object_summary=None,
    ):
        # convert to b c t h w
        x = x.permute(0, 2, 1, 3, 4)
        if self.object_prompt:
            object_valid = self._apply_object_dropout(object_valid)
            object_summary = self._apply_object_summary_dropout(object_summary)
        if self.actor_prompt:
            if self.hparams.n_landmarks > 0 or self.object_prompt:
                _, x_actor, x_heatmap = self.net(
                    x,
                    boxes=boxes,
                    valid=valid,
                    object_boxes=object_boxes,
                    object_valid=object_valid,
                    object_conf=object_conf,
                )
            else:
                data = self.net(
                    x,
                    boxes=boxes,
                    valid=valid,
                    object_boxes=object_boxes,
                    object_valid=object_valid,
                    object_conf=object_conf,
                )
                _, x_actor = data
                x_heatmap = 0
            if self.hparams.ret_feat:
                return x_actor
            action_logits = self.actor_head(x_actor)
            if self.object_prompt:
                action_logits = self._object_summary_fusion(
                    action_logits,
                    object_summary=object_summary,
                )
            interaction_logits = (
                self.interaction_head(x_actor) if self.object_prompt else None
            )
            if self.presence_head is not None:
                presence_logits = self.presence_head(x_actor).squeeze(-1)
                if self.object_prompt:
                    return action_logits, x_heatmap, presence_logits, interaction_logits
                return action_logits, x_heatmap, presence_logits
            if self.object_prompt:
                return action_logits, x_heatmap, None, interaction_logits
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
        parser.add_argument("--object_bbox_prior_weight", type=float, default=0.0)
        parser.add_argument("--object_bbox_prior_expand", type=float, default=1.25)
        parser.add_argument("--object_dropout_prob", type=float, default=0.0)
        parser.add_argument("--object_token_dropout_prob", type=float, default=0.0)
        parser.add_argument("--object_summary_hidden_dim", type=int, default=256)
        parser.add_argument("--object_summary_dropout", type=float, default=0.15)
        parser.add_argument("--object_summary_gate_init", type=float, default=0.12)
        parser.add_argument("--object_summary_delta_only", type=int, default=0)
        parser.add_argument("--ret_feat", type=int, default=0)
        parser.add_argument("--linear_probe", type=int, default=0)

        return parser
