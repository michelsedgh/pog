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
        self.actor_object_relation_in_transformer = bool(
            self.hparams.get("actor_object_relation_in_transformer", 0)
        )
        self.actor_relation_action_fusion_enabled = bool(
            self.hparams.get("actor_relation_action_fusion", 0)
        )
        self.actor_object_pair_action_head_enabled = bool(
            self.hparams.get("actor_object_pair_action_head", 0)
        )
        self.actor_object_pair_action_hidden_dim = int(
            self.hparams.get("actor_object_pair_action_hidden_dim", 0)
        )
        self.actor_object_pair_action_init_scale = float(
            self.hparams.get("actor_object_pair_action_init_scale", 0.01)
        )
        if self.actor_object_pair_action_hidden_dim < 0:
            raise ValueError("actor_object_pair_action_hidden_dim must be >= 0")
        if self.actor_object_pair_action_init_scale < 0:
            raise ValueError("actor_object_pair_action_init_scale must be >= 0")
        if self.actor_relation_action_fusion_enabled:
            raise ValueError(
                "actor_relation_action_fusion was removed. Use "
                "--actor_object_pair_action_head 1 so action logits are scored "
                "from actor-object pairs instead of one soft-selected object memory."
            )
        if float(self.hparams.get("actor_relation_action_fusion_init_scale", 0.0)) != 0.0:
            raise ValueError(
                "actor_relation_action_fusion_init_scale is stale because "
                "actor_relation_action_fusion was removed."
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
        removed_relation_prior_keys = (
            "actor_object_relation_valid_logit_bonus",
            "actor_object_relation_learned_valid_bonus",
            "actor_object_relation_geometry_bias_weight",
            "actor_object_relation_heatmap_bias_weight",
            "token_selection_object_weight",
        )
        stale_relation_prior = [
            key
            for key in removed_relation_prior_keys
            if key in self.hparams
            and self.hparams.get(key) not in (None, 0, 0.0, "0", "0.0")
        ]
        if stale_relation_prior:
            raise ValueError(
                "Manual actor-object relation logit priors were removed. "
                "The clean ROI-object path uses object-region visual tokens, "
                "normalized learned relation pointers, relation CE, and learned "
                "pair action scoring. Remove stale hparams: "
                f"{', '.join(stale_relation_prior)}"
            )
        if bool(self.hparams.get("actor_object_slot_head", 0)):
            raise ValueError(
                "actor_object_slot_head was replaced by "
                "actor_object_prompt_tokens. Set --actor_object_prompt_tokens 1 "
                "and keep --actor_object_slot_head 0."
            )
        if bool(self.hparams.get("scene_object_tokens", 0)):
            raise ValueError(
                "scene_object_tokens was removed. Use actor_object_prompt_tokens=1 "
                "for relation-only runtime object memory."
            )
        if bool(self.hparams.get("actor_object_base_fusion", 0)):
            raise ValueError(
                "actor_object_base_fusion was removed. Runtime objects now enter "
                "through relation-only memory, token selection priors, and optional "
                "actor_object_relation_in_transformer."
            )
        if self.actor_object_residual_head_enabled:
            raise ValueError(
                "actor_object_residual_head was removed from the clean PO-GUISE+ "
                "path. Use actor_object_relation_in_transformer for object-aware "
                "actor-token updates."
            )
        if self.actor_object_prompt_tokens_enabled and not self.actor_prompt:
            raise ValueError("actor_object_prompt_tokens requires actor_prompt")
        if self.actor_object_prompt_tokens_enabled and not bool(
            self.hparams.get("actor_object_region_visual_tokens", 0)
        ):
            raise ValueError(
                "actor_object_prompt_tokens requires "
                "actor_object_region_visual_tokens=1. Runtime object memory must "
                "include object-region visual patch features, not only class/box "
                "metadata."
            )
        if (
            self.actor_object_prompt_tokens_enabled
            and not self.actor_interaction_heatmaps
        ):
            raise ValueError(
                "actor_object_prompt_tokens requires actor_interaction_heatmaps"
            )
        if (
            self.actor_object_prompt_tokens_enabled
            and not self.actor_object_relation_in_transformer
        ):
            raise ValueError(
                "actor_object_prompt_tokens now has one supported path: "
                "enable actor_object_relation_in_transformer so object proposals "
                "are bound to actor slots instead of existing as passive prompt "
                "tokens."
            )
        if self.actor_object_relation_in_transformer and not self.actor_object_prompt_tokens_enabled:
            raise ValueError(
                "actor_object_relation_in_transformer requires actor_object_prompt_tokens"
            )
        if self.actor_object_relation_in_transformer and not self.actor_object_pair_action_head_enabled:
            raise ValueError(
                "actor_object_relation_in_transformer requires "
                "actor_object_pair_action_head so relation pointers and action "
                "classification are optimized through the same actor-object pairs."
            )
        if self.actor_object_pair_action_head_enabled and not self.actor_object_relation_in_transformer:
            raise ValueError(
                "actor_object_pair_action_head requires "
                "actor_object_relation_in_transformer"
            )
        if self.actor_interaction_heatmaps and not self.actor_prompt:
            raise ValueError("actor_interaction_heatmaps requires actor_prompt")
        if self.actor_object_relation_in_transformer and not self.actor_interaction_heatmaps:
            raise ValueError(
                "actor_object_relation_in_transformer requires "
                "actor_interaction_heatmaps so relation bias is grounded in "
                "actor-conditioned object heatmap tokens."
            )
        if "interaction_object_classes" in self.hparams:
            raise ValueError(
                "interaction_object_classes was removed. Actor-object heatmaps "
                "are now one interacted-object channel per actor; object class "
                "semantics come from relation-only runtime object memory."
            )
        self.use_register_tokens = bool(self.hparams.get("use_register_tokens", 0))
        self._register_actor_object_action_buffers()
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

    def _register_actor_object_action_buffers(self):
        num_classes = int(self.hparams.get("num_classes", 0))
        num_object_classes = int(
            self.hparams.get("num_object_classes", NUM_OBJECT_CLASSES)
        )
        action_object_mask = torch.zeros(
            num_classes,
            num_object_classes,
            dtype=torch.bool,
        )
        action_has_object = torch.zeros(num_classes, dtype=torch.bool)
        if not self.actor_object_pair_action_head_enabled:
            self.register_buffer(
                "actor_object_action_mask",
                action_object_mask,
                persistent=False,
            )
            self.register_buffer(
                "actor_object_action_has_object",
                action_has_object,
                persistent=False,
            )
            return

        task_type = self.hparams.get("task_type", "CS")
        action_taxonomy = self.hparams.get("toyota_action_taxonomy", "toyota_31")
        action_names = toyota_action_names(task_type, action_taxonomy)
        if len(action_names) != num_classes:
            raise ValueError(
                "actor_object_pair_action_head needs the Toyota action taxonomy "
                f"to match num_classes. Got {len(action_names)} action names for "
                f"num_classes={num_classes}."
            )
        action_to_index = {name: index for index, name in enumerate(action_names)}
        action_object_map = toyota_action_object_map(task_type, action_taxonomy)
        if not action_object_map:
            raise ValueError(
                "actor_object_pair_action_head requires a non-empty Toyota "
                "action-to-object map."
            )

        missing_objects = []
        mapped_actions = 0
        for action_name, object_names in action_object_map.items():
            action_idx = action_to_index.get(action_name)
            if action_idx is None:
                continue
            for object_name in object_names:
                object_idx = OBJECT_TO_ID.get(object_name)
                if object_idx is None or object_idx >= num_object_classes:
                    missing_objects.append(object_name)
                    continue
                action_object_mask[int(action_idx), int(object_idx)] = True
            if action_object_mask[int(action_idx)].any():
                action_has_object[int(action_idx)] = True
                mapped_actions += 1
        if missing_objects:
            names = ", ".join(sorted(set(missing_objects)))
            raise ValueError(
                "actor_object_pair_action_head found Toyota action-object names "
                f"missing from object_vocab: {names}"
            )
        if mapped_actions == 0:
            raise ValueError(
                "actor_object_pair_action_head built zero mapped object actions. "
                "Check --toyota_action_taxonomy and --num_object_classes."
            )

        self.register_buffer(
            "actor_object_action_mask",
            action_object_mask,
            persistent=False,
        )
        self.register_buffer(
            "actor_object_action_has_object",
            action_has_object,
            persistent=False,
        )

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
                actor_object_prompt_tokens=self.actor_object_prompt_tokens_enabled,
                num_scene_object_tokens=self.hparams.get("num_scene_object_tokens", 32),
                num_object_classes=self.hparams.get(
                    "num_object_classes",
                    NUM_OBJECT_CLASSES,
                ),
                actor_object_region_visual_tokens=self.hparams.get(
                    "actor_object_region_visual_tokens",
                    0,
                ),
                actor_object_prompt_box_prior_weight=self.hparams.get(
                    "actor_object_prompt_box_prior_weight",
                    0.05,
                ),
                actor_object_prompt_box_prior_expand=self.hparams.get(
                    "actor_object_prompt_box_prior_expand",
                    1.25,
                ),
                token_selection_cls_weight=self.hparams.get(
                    "token_selection_cls_weight",
                    0.25,
                ),
                token_selection_actor_weight=self.hparams.get(
                    "token_selection_actor_weight",
                    0.25,
                ),
                token_selection_register_weight=self.hparams.get(
                    "token_selection_register_weight",
                    0.0,
                ),
                token_selection_heatmap_weight=self.hparams.get(
                    "token_selection_heatmap_weight",
                    0.35,
                ),
                actor_object_relation_in_transformer=self.actor_object_relation_in_transformer,
                actor_object_relation_blocks=self.hparams.get(
                    "actor_object_relation_blocks",
                    "2,5,8",
                ),
                actor_object_relation_dim=self.hparams.get(
                    "actor_object_relation_dim",
                    256,
                ),
                actor_object_relation_hidden_dim=self.hparams.get(
                    "actor_object_relation_hidden_dim",
                    512,
                ),
                actor_object_relation_max_scale=self.hparams.get(
                    "actor_object_relation_max_scale",
                    1.0,
                ),
                actor_object_relation_null_logit_init=self.hparams.get(
                    "actor_object_relation_null_logit_init",
                    4.0,
                ),
                actor_object_relation_logit_scale_init=self.hparams.get(
                    "actor_object_relation_logit_scale_init",
                    1.0,
                ),
                actor_object_relation_learned_logit_scale=self.hparams.get(
                    "actor_object_relation_learned_logit_scale",
                    0,
                ),
                actor_object_relation_normalize_pointers=self.hparams.get(
                    "actor_object_relation_normalize_pointers",
                    0,
                ),
                actor_object_relation_learned_scale=self.hparams.get(
                    "actor_object_relation_learned_scale",
                    0,
                ),
                actor_object_relation_layer_scale_init=self.hparams.get(
                    "actor_object_relation_layer_scale_init",
                    0.25,
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
                actor_object_prompt_tokens=self.actor_object_prompt_tokens_enabled,
                num_scene_object_tokens=self.hparams.get("num_scene_object_tokens", 32),
                num_object_classes=self.hparams.get(
                    "num_object_classes",
                    NUM_OBJECT_CLASSES,
                ),
                actor_object_region_visual_tokens=self.hparams.get(
                    "actor_object_region_visual_tokens",
                    0,
                ),
                actor_object_prompt_box_prior_weight=self.hparams.get(
                    "actor_object_prompt_box_prior_weight",
                    0.05,
                ),
                actor_object_prompt_box_prior_expand=self.hparams.get(
                    "actor_object_prompt_box_prior_expand",
                    1.25,
                ),
                token_selection_cls_weight=self.hparams.get(
                    "token_selection_cls_weight",
                    0.25,
                ),
                token_selection_actor_weight=self.hparams.get(
                    "token_selection_actor_weight",
                    0.25,
                ),
                token_selection_register_weight=self.hparams.get(
                    "token_selection_register_weight",
                    0.0,
                ),
                token_selection_heatmap_weight=self.hparams.get(
                    "token_selection_heatmap_weight",
                    0.35,
                ),
                actor_object_relation_in_transformer=self.actor_object_relation_in_transformer,
                actor_object_relation_blocks=self.hparams.get(
                    "actor_object_relation_blocks",
                    "2,5,8",
                ),
                actor_object_relation_dim=self.hparams.get(
                    "actor_object_relation_dim",
                    256,
                ),
                actor_object_relation_hidden_dim=self.hparams.get(
                    "actor_object_relation_hidden_dim",
                    512,
                ),
                actor_object_relation_max_scale=self.hparams.get(
                    "actor_object_relation_max_scale",
                    1.0,
                ),
                actor_object_relation_null_logit_init=self.hparams.get(
                    "actor_object_relation_null_logit_init",
                    4.0,
                ),
                actor_object_relation_logit_scale_init=self.hparams.get(
                    "actor_object_relation_logit_scale_init",
                    1.0,
                ),
                actor_object_relation_learned_logit_scale=self.hparams.get(
                    "actor_object_relation_learned_logit_scale",
                    0,
                ),
                actor_object_relation_normalize_pointers=self.hparams.get(
                    "actor_object_relation_normalize_pointers",
                    0,
                ),
                actor_object_relation_learned_scale=self.hparams.get(
                    "actor_object_relation_learned_scale",
                    0,
                ),
                actor_object_relation_layer_scale_init=self.hparams.get(
                    "actor_object_relation_layer_scale_init",
                    0.25,
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
            self.actor_head = None
            self.actor_object_null_pair_token = None
            self.actor_object_pair_action_head = None
            if self.actor_object_pair_action_head_enabled:
                feature_dim = int(self.net.num_features)
                pair_dim = feature_dim * 3 + 1
                hidden_dim = self.actor_object_pair_action_hidden_dim or feature_dim
                self.actor_object_null_pair_token = nn.Parameter(
                    torch.zeros(1, 1, 1, feature_dim)
                )
                self.actor_object_pair_action_head = nn.Sequential(
                    nn.LayerNorm(pair_dim),
                    nn.Linear(pair_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, self.hparams.num_classes),
                )
                nn.init.trunc_normal_(
                    self.actor_object_pair_action_head[-1].weight,
                    std=self.actor_object_pair_action_init_scale,
                )
                nn.init.zeros_(self.actor_object_pair_action_head[-1].bias)
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
            if self.actor_object_pair_action_head is not None:
                for param in self.actor_object_pair_action_head.parameters():
                    param.requires_grad = True
            if self.actor_object_null_pair_token is not None:
                self.actor_object_null_pair_token.requires_grad = True
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
                "object_region_norm",
                "object_region_proj",
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

    def _actor_object_pair_allowed_mask(
        self,
        object_classes,
        object_valid,
        actor_valid,
        num_actors,
        num_pairs,
        num_actions,
    ):
        batch_size, num_objects = object_classes.shape
        device = object_classes.device
        max_object_class = int(self.actor_object_action_mask.shape[-1]) - 1
        object_valid = object_valid.to(device=device, dtype=torch.bool)
        class_in_range = (object_classes >= 0) & (object_classes <= max_object_class)
        object_valid = object_valid & class_in_range
        object_classes = object_classes.clamp(0, max_object_class)
        action_object_mask = self.actor_object_action_mask.to(device=device)
        action_has_object = self.actor_object_action_has_object.to(device=device)
        object_compat = action_object_mask.t()[object_classes]
        object_compat = object_compat & object_valid.unsqueeze(-1)
        object_compat = object_compat[:, None, :, :].expand(
            batch_size,
            num_actors,
            num_objects,
            num_actions,
        )
        compatible_object_present = object_compat.any(dim=2)
        null_allowed = (
            ~action_has_object.view(1, 1, num_actions)
        ) | ~compatible_object_present
        allowed = torch.cat([null_allowed.unsqueeze(2), object_compat], dim=2)
        if allowed.shape[2] != num_pairs:
            raise RuntimeError(
                "actor-object pair allowed mask shape mismatch: "
                f"{allowed.shape[2]} pairs vs expected {num_pairs}"
            )
        if actor_valid is not None:
            actor_valid = actor_valid.to(device=device, dtype=torch.bool)
            allowed = allowed & actor_valid[:, :, None, None]
        return allowed

    def _actor_object_pair_action_scores(
        self,
        x_actor,
        relation_logits,
        object_tokens,
        object_classes,
        object_valid,
        actor_valid,
    ):
        if self.actor_object_pair_action_head is None:
            raise RuntimeError("actor_object_pair_action_head is not initialized")
        if object_tokens is None or object_classes is None or object_valid is None:
            raise RuntimeError(
                "actor_object_pair_action_head requires object prompt tokens, "
                "object classes, and object_valid."
            )
        batch_size, num_actors, feature_dim = x_actor.shape
        if object_tokens.ndim != 3:
            raise RuntimeError(
                "object prompt tokens must have shape [B,O,D], got "
                f"{tuple(object_tokens.shape)}"
            )
        if object_tokens.shape[0] != batch_size or object_tokens.shape[-1] != feature_dim:
            raise RuntimeError(
                "object prompt tokens shape does not match actor tokens: "
                f"{tuple(object_tokens.shape)} vs {tuple(x_actor.shape)}"
            )
        num_objects = int(object_tokens.shape[1])
        num_pairs = num_objects + 1
        num_actions = int(self.hparams.num_classes)
        if relation_logits.shape != (batch_size, num_actors, num_pairs):
            raise RuntimeError(
                "actor-object relation logits must have shape [B,A,O+1], got "
                f"{tuple(relation_logits.shape)} for actor/object shapes "
                f"{tuple(x_actor.shape)}, {tuple(object_tokens.shape)}"
            )
        if object_classes.shape != (batch_size, num_objects):
            raise RuntimeError(
                "object_classes must have shape [B,O], got "
                f"{tuple(object_classes.shape)}"
            )
        if object_valid.shape != (batch_size, num_objects):
            raise RuntimeError(
                "object_valid must have shape [B,O], got "
                f"{tuple(object_valid.shape)}"
            )

        object_tokens = object_tokens.to(device=x_actor.device, dtype=x_actor.dtype)
        null_token = self.actor_object_null_pair_token.to(
            device=x_actor.device,
            dtype=x_actor.dtype,
        ).expand(batch_size, num_actors, 1, feature_dim)
        object_pair_tokens = object_tokens[:, None, :, :].expand(
            batch_size,
            num_actors,
            num_objects,
            feature_dim,
        )
        pair_tokens = torch.cat([null_token, object_pair_tokens], dim=2)
        actor_tokens = x_actor[:, :, None, :].expand(
            batch_size,
            num_actors,
            num_pairs,
            feature_dim,
        )
        relation_log_probs = F.log_softmax(relation_logits.float(), dim=-1)
        valid_pair = torch.cat(
            [
                torch.ones(
                    batch_size,
                    num_actors,
                    1,
                    device=x_actor.device,
                    dtype=x_actor.dtype,
                ),
                object_valid.to(device=x_actor.device, dtype=x_actor.dtype)[
                    :, None, :
                ].expand(batch_size, num_actors, num_objects),
            ],
            dim=2,
        )
        pair_features = torch.cat(
            [
                actor_tokens,
                pair_tokens,
                actor_tokens * pair_tokens,
                valid_pair.unsqueeze(-1),
            ],
            dim=-1,
        )
        pair_logits = self.actor_object_pair_action_head(pair_features).float()
        pair_scores = pair_logits + relation_log_probs.unsqueeze(-1)
        allowed = self._actor_object_pair_allowed_mask(
            object_classes.to(device=x_actor.device, dtype=torch.long),
            object_valid.to(device=x_actor.device, dtype=torch.bool),
            actor_valid,
            num_actors,
            num_pairs,
            num_actions,
        )
        masked_pair_logits = pair_logits.masked_fill(~allowed, -1.0e4)
        pair_action_probs = F.softmax(masked_pair_logits, dim=-1)
        relation_probs = torch.exp(relation_log_probs)
        action_probs = torch.sum(pair_action_probs * relation_probs.unsqueeze(-1), dim=2)
        action_scores = torch.log(action_probs.clamp_min(1e-6))

        self.last_actor_object_pair_action_logits = pair_logits
        self.last_actor_object_pair_action_scores = pair_scores
        self.last_actor_object_pair_action_allowed = allowed
        self.last_actor_object_pair_action_log_probs = relation_log_probs
        return action_scores.to(dtype=x_actor.dtype)

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
            self.last_actor_object_prompt_classes = None
            self.last_actor_object_prompt_tokens = None
            self.last_actor_object_prompt_valid = None
            self.last_actor_object_relation_context = None
            self.last_actor_object_relation_mass = None
            self.last_actor_object_pair_action_logits = None
            self.last_actor_object_pair_action_scores = None
            self.last_actor_object_pair_action_allowed = None
            self.last_actor_object_pair_action_log_probs = None
            self.last_actor_action_tokens = None
            self.last_actor_object_relation_aux = getattr(
                self.net,
                "last_actor_object_relation_aux",
                {},
            )
            self.last_actor_object_region_visual_norm = getattr(
                self.net,
                "last_object_region_visual_norm",
                None,
            )
            if self.hparams.ret_feat:
                return x_actor

            prompt_valid = None
            if (
                self.actor_object_prompt_tokens_enabled
                and x_object_prompt is not None
                and object_valid is not None
            ):
                prompt_valid = object_valid.to(
                    device=x_object_prompt.device,
                    dtype=torch.bool,
                )
                if object_classes is not None:
                    none_id = int(self.hparams.get("num_object_classes", 19))
                    prompt_classes = object_classes.to(
                        device=x_object_prompt.device,
                        dtype=torch.long,
                    )
                    prompt_classes = torch.where(
                        prompt_valid,
                        prompt_classes.clamp(0, none_id),
                        torch.full_like(prompt_classes, none_id),
                    )
                    self.last_actor_object_prompt_classes = prompt_classes
                    self.last_actor_object_prompt_tokens = x_object_prompt
                    self.last_actor_object_prompt_valid = prompt_valid
            x_action = x_actor
            self.last_actor_action_tokens = x_action
            if self.actor_object_pair_action_head is not None:
                if not self.last_actor_object_relation_aux:
                    raise RuntimeError(
                        "actor_object_pair_action_head requires relation logits"
                    )
                last_block = sorted(
                    self.last_actor_object_relation_aux.keys(),
                    key=lambda value: int(value),
                )[-1]
                last_relation_aux = self.last_actor_object_relation_aux[last_block]
                relation_logits = last_relation_aux.get("logits")
                if relation_logits is None:
                    raise RuntimeError(
                        "actor_object_pair_action_head requires final relation logits"
                    )
                action_scores = self._actor_object_pair_action_scores(
                    x_actor,
                    relation_logits.to(device=x_actor.device),
                    x_object_prompt,
                    self.last_actor_object_prompt_classes,
                    prompt_valid,
                    valid,
                )
            else:
                if self.actor_head is None:
                    raise RuntimeError("actor_head is not initialized")
                action_scores = self.actor_head(x_action)
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
        parser.add_argument("--actor_pair_train_weight", type=float, default=0.0)
        parser.add_argument("--actor_val_diagnostics", type=int, default=1)
        parser.add_argument("--actor_val_diagnostic_max_pairs", type=int, default=8)
        parser.add_argument("--actor_interaction_heatmaps", type=int, default=0)
        parser.add_argument("--num_scene_object_tokens", type=int, default=32)
        parser.add_argument("--num_object_classes", type=int, default=19)
        parser.add_argument("--actor_object_prompt_tokens", type=int, default=0)
        parser.add_argument("--actor_object_region_visual_tokens", type=int, default=0)
        parser.add_argument(
            "--actor_object_prompt_box_prior_weight",
            type=float,
            default=0.05,
        )
        parser.add_argument(
            "--actor_object_prompt_box_prior_expand",
            type=float,
            default=1.25,
        )
        parser.add_argument("--token_selection_cls_weight", type=float, default=0.25)
        parser.add_argument("--token_selection_actor_weight", type=float, default=0.25)
        parser.add_argument("--token_selection_register_weight", type=float, default=0.0)
        parser.add_argument("--token_selection_heatmap_weight", type=float, default=0.35)
        parser.add_argument("--actor_object_relation_in_transformer", type=int, default=0)
        parser.add_argument("--actor_object_relation_blocks", type=str, default="2,5,8")
        parser.add_argument("--actor_object_relation_dim", type=int, default=256)
        parser.add_argument("--actor_object_relation_hidden_dim", type=int, default=512)
        parser.add_argument("--actor_object_relation_max_scale", type=float, default=1.0)
        parser.add_argument("--actor_object_relation_learned_scale", type=int, default=0)
        parser.add_argument(
            "--actor_object_relation_layer_scale_init",
            type=float,
            default=0.25,
        )
        parser.add_argument("--actor_relation_action_fusion", type=int, default=0)
        parser.add_argument(
            "--actor_relation_action_fusion_init_scale",
            type=float,
            default=0.0,
        )
        parser.add_argument("--actor_object_pair_action_head", type=int, default=0)
        parser.add_argument(
            "--actor_object_pair_action_hidden_dim",
            type=int,
            default=0,
        )
        parser.add_argument(
            "--actor_object_pair_action_init_scale",
            type=float,
            default=0.01,
        )
        parser.add_argument(
            "--actor_object_relation_null_logit_init",
            type=float,
            default=4.0,
        )
        parser.add_argument(
            "--actor_object_relation_logit_scale_init",
            type=float,
            default=1.0,
        )
        parser.add_argument(
            "--actor_object_relation_learned_logit_scale",
            type=int,
            default=0,
        )
        parser.add_argument(
            "--actor_object_relation_normalize_pointers",
            type=int,
            default=0,
        )
        parser.add_argument("--trt_safe_attention", type=int, default=0)
        parser.add_argument("--ret_feat", type=int, default=0)
        parser.add_argument("--linear_probe", type=int, default=0)

        return parser
