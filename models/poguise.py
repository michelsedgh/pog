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
    toyota_confuser_action_names,
    toyota_objectless_action_names,
)
from models.object_residual_action_head import (
    ActorObjectContextFusion,
    ObjectResidualActionSpec,
    ObjectResidualActionHead,
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
        if bool(self.hparams.get("actor_object_slot_head", 0)):
            raise ValueError(
                "actor_object_slot_head was replaced by "
                "actor_object_prompt_tokens. Set --actor_object_prompt_tokens 1 "
                "and keep --actor_object_slot_head 0."
            )
        if bool(self.hparams.get("scene_object_tokens", 0)):
            raise ValueError(
                "scene_object_tokens was removed. Use actor_object_prompt_tokens=1 "
                "for runtime object prompts inside the transformer trunk."
            )
        if self.actor_object_residual_head_enabled:
            if not self.actor_object_prompt_tokens_enabled:
                raise ValueError(
                    "actor_object_residual_head requires actor_object_prompt_tokens"
                )
        if self.actor_object_prompt_tokens_enabled and not self.actor_prompt:
            raise ValueError("actor_object_prompt_tokens requires actor_prompt")
        if self.actor_object_prompt_tokens_enabled and not self.actor_object_residual_head_enabled:
            raise ValueError(
                "actor_object_prompt_tokens requires actor_object_residual_head; "
                "runtime objects must use the bounded actor-object action path."
            )
        if self.actor_interaction_heatmaps and not self.actor_prompt:
            raise ValueError("actor_interaction_heatmaps requires actor_prompt")
        if (
            self.actor_object_residual_head_enabled
            and not self.actor_interaction_heatmaps
        ):
            raise ValueError(
                "actor_object_residual_head requires actor_interaction_heatmaps "
                "so the base PO-GUISE+ trunk has actor-object heatmap evidence."
            )
        if "interaction_object_classes" in self.hparams:
            raise ValueError(
                "interaction_object_classes was removed. Actor-object heatmaps "
                "are now one interacted-object channel per actor; object class "
                "semantics come from runtime object prompt tokens."
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
        self.object_residual_action_head = None
        self.actor_object_base_fusion = None
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
            return_heatmap_features = bool(self.actor_object_prompt_tokens_enabled)
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
                actor_object_prompt_tokens=self.actor_object_prompt_tokens_enabled,
                num_scene_object_tokens=self.hparams.get("num_scene_object_tokens", 32),
                num_object_classes=self.hparams.get(
                    "num_object_classes",
                    NUM_OBJECT_CLASSES,
                ),
                actor_object_prompt_box_prior_weight=self.hparams.get(
                    "actor_object_prompt_box_prior_weight",
                    0.05,
                ),
                actor_object_prompt_box_prior_expand=self.hparams.get(
                    "actor_object_prompt_box_prior_expand",
                    1.25,
                ),
                return_heatmap_features=return_heatmap_features,
                trt_safe_attention=self.hparams.get("trt_safe_attention", 0),
            )
        else:
            return_heatmap_features = bool(self.actor_object_prompt_tokens_enabled)
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
                actor_object_prompt_tokens=self.actor_object_prompt_tokens_enabled,
                num_scene_object_tokens=self.hparams.get("num_scene_object_tokens", 32),
                num_object_classes=self.hparams.get(
                    "num_object_classes",
                    NUM_OBJECT_CLASSES,
                ),
                actor_object_prompt_box_prior_weight=self.hparams.get(
                    "actor_object_prompt_box_prior_weight",
                    0.05,
                ),
                actor_object_prompt_box_prior_expand=self.hparams.get(
                    "actor_object_prompt_box_prior_expand",
                    1.25,
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
            self.actor_head = nn.Linear(
                self.net.num_features,
                self.hparams.num_classes,
            )
            if self.hparams.get("actor_object_base_fusion", 0):
                if not self.actor_object_prompt_tokens_enabled:
                    raise ValueError(
                        "actor_object_base_fusion requires actor_object_prompt_tokens"
                    )
                self.actor_object_base_fusion = ActorObjectContextFusion(
                    self.net.num_features,
                    hidden_dim=int(
                        self.hparams.get(
                            "actor_object_base_fusion_hidden_dim",
                            512,
                        )
                    ),
                    fusion_scale_init=float(
                        self.hparams.get(
                            "actor_object_base_fusion_scale_init",
                            -2.0,
                        )
                    ),
                    max_fusion_scale=float(
                        self.hparams.get(
                            "actor_object_base_fusion_max_scale",
                            1.0,
                        )
                    ),
                )
            self.actor_motion_head = (
                nn.Linear(self.net.num_features, self.hparams.num_classes)
                if (
                    not self.actor_object_residual_head_enabled
                    and float(self.hparams.get("motion_aux_loss_weight", 0.25)) > 0.0
                )
                else None
            )
            self.presence_head = (
                nn.Linear(self.net.num_features, 1)
                if self.hparams.get("actor_presence_head", 0)
                else None
            )
            if self.actor_object_residual_head_enabled:
                spec = self._object_residual_action_spec()
                self.object_residual_action_head = (
                    ObjectResidualActionHead(
                        self.net.num_features,
                        spec=spec,
                        hidden_dim=int(
                            self.hparams.get(
                                "actor_object_residual_hidden_dim",
                                512,
                            )
                        ),
                        relation_scale_init=float(
                            self.hparams.get(
                                "actor_object_residual_relation_scale_init",
                                -1.0,
                            )
                        ),
                        relation_logit_bound=float(
                            self.hparams.get(
                                "actor_object_residual_relation_logit_bound",
                                2.0,
                            )
                        ),
                        max_relation_scale=float(
                            self.hparams.get(
                                "actor_object_residual_max_relation_scale",
                                1.5,
                            )
                        ),
                        compat_prior_scale=float(
                            self.hparams.get(
                                "actor_object_residual_compat_prior_scale",
                                1.0,
                            )
                        ),
                    )
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

    def _toyota_action_settings(self):
        task_type = self.hparams.get("task_type", "CS")
        action_taxonomy = self.hparams.get("toyota_action_taxonomy", "toyota_31")
        return task_type, action_taxonomy

    def _object_residual_action_spec(self):
        dataset = self.hparams.get("dataset", None)
        dataset_name = (
            dataset
            if isinstance(dataset, str)
            else getattr(dataset, "__name__", str(dataset))
        )
        dataset_module = "" if isinstance(dataset, str) else getattr(
            dataset,
            "__module__",
            "",
        )
        is_toyotasm = (
            dataset_name == "toyotasm"
            or dataset_name == "ToyotaSMDataset"
            or dataset_module == "datasets.toyotasm"
            or self.hparams.get("dataset_artifact", None) == "toyotasm"
        )
        if not is_toyotasm:
            raise ValueError("actor_object_residual_head currently requires toyotasm")

        task_type, action_taxonomy = self._toyota_action_settings()
        action_to_index = toyota_action_to_index(task_type, action_taxonomy)
        action_object_map = toyota_action_object_map(task_type, action_taxonomy)
        objectless_names = toyota_objectless_action_names(
            task_type,
            action_taxonomy,
        )
        num_actions = int(self.hparams.num_classes)
        num_object_classes = int(
            self.hparams.get("num_object_classes", NUM_OBJECT_CLASSES)
        )

        objectless_indices = []
        for action_name in objectless_names:
            action_idx = action_to_index.get(action_name)
            if action_idx is not None:
                objectless_indices.append(int(action_idx))
        if not objectless_indices:
            raise ValueError("object_residual head found no objectless actions")

        compat = torch.zeros(num_object_classes, num_actions, dtype=torch.float32)
        for action_name, object_names in action_object_map.items():
            action_idx = action_to_index.get(action_name)
            if action_idx is None:
                continue
            for object_name in object_names:
                object_id = OBJECT_TO_ID.get(object_name)
                if object_id is not None and int(object_id) < num_object_classes:
                    compat[int(object_id), int(action_idx)] = 1.0
        if not compat.any():
            raise ValueError("object_residual head found no object/action compatibility map")

        confusers_by_action = {}
        for action_name, action_idx in action_to_index.items():
            confusers = []
            for confuser_name in toyota_confuser_action_names(
                action_name,
                task_type,
                action_taxonomy,
            ):
                confuser_idx = action_to_index.get(confuser_name)
                if confuser_idx is not None:
                    confusers.append(int(confuser_idx))
            if confusers:
                confusers_by_action[int(action_idx)] = tuple(sorted(set(confusers)))

        return ObjectResidualActionSpec(
            num_actions=num_actions,
            num_object_classes=num_object_classes,
            objectless_action_indices=tuple(sorted(set(objectless_indices))),
            compat_matrix=compat,
            confusers_by_action=confusers_by_action,
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
                if self.actor_head is not None:
                    for param in self.actor_head.parameters():
                        param.requires_grad = False
                if self.actor_motion_head is not None:
                    for param in self.actor_motion_head.parameters():
                        param.requires_grad = False
                if self.presence_head is not None:
                    for param in self.presence_head.parameters():
                        param.requires_grad = False
                if self.actor_object_base_fusion is not None:
                    for param in self.actor_object_base_fusion.parameters():
                        param.requires_grad = False
                if self.object_residual_action_head is not None:
                    for param in self.object_residual_action_head.parameters():
                        param.requires_grad = False
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
                if self.actor_head is not None:
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
                for attr in (
                    "object_slot_embed",
                    "object_class_embed",
                    "object_box_mlp",
                    "object_conf_mlp",
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
                if self.actor_object_base_fusion is not None:
                    for param in self.actor_object_base_fusion.parameters():
                        param.requires_grad = True
                if self.object_residual_action_head is not None:
                    for param in self.object_residual_action_head.parameters():
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

    def _actor_object_geometry_attention_bias(
        self,
        actor_boxes,
        object_boxes,
        actor_valid,
        object_valid,
        device,
        dtype,
    ):
        if (
            actor_boxes is None
            or object_boxes is None
            or actor_valid is None
            or object_valid is None
        ):
            return None
        actor_boxes = actor_boxes.to(device=device, dtype=dtype).clamp(0.0, 1.0)
        object_boxes = object_boxes.to(device=device, dtype=dtype).clamp(0.0, 1.0)
        actor_valid = actor_valid.to(device=device, dtype=torch.bool)
        object_valid = object_valid.to(device=device, dtype=torch.bool)

        actor_center = (actor_boxes[..., :2] + actor_boxes[..., 2:]) * 0.5
        object_center = (object_boxes[..., :2] + object_boxes[..., 2:]) * 0.5
        actor_size = (actor_boxes[..., 2:] - actor_boxes[..., :2]).clamp_min(1.0e-4)
        object_size = (object_boxes[..., 2:] - object_boxes[..., :2]).clamp_min(1.0e-4)

        center_dist = torch.linalg.vector_norm(
            actor_center[:, :, None, :] - object_center[:, None, :, :],
            dim=-1,
        )
        actor_diag = torch.linalg.vector_norm(actor_size, dim=-1)
        object_diag = torch.linalg.vector_norm(object_size, dim=-1)
        distance_scale = (
            0.5 * actor_diag[:, :, None] + 0.5 * object_diag[:, None, :]
        ).clamp_min(0.05)
        norm_dist = center_dist / distance_scale
        proximity = torch.exp(-0.5 * norm_dist.square()).clamp(0.0, 1.0)

        inter_min = torch.maximum(
            actor_boxes[:, :, None, :2],
            object_boxes[:, None, :, :2],
        )
        inter_max = torch.minimum(
            actor_boxes[:, :, None, 2:],
            object_boxes[:, None, :, 2:],
        )
        inter_size = (inter_max - inter_min).clamp_min(0.0)
        inter_area = inter_size[..., 0] * inter_size[..., 1]
        actor_area = actor_size[..., 0] * actor_size[..., 1]
        object_area = object_size[..., 0] * object_size[..., 1]
        union = actor_area[:, :, None] + object_area[:, None, :] - inter_area
        iou = inter_area / union.clamp_min(1.0e-6)

        pair_valid = actor_valid[:, :, None] & object_valid[:, None, :]
        bias = 4.0 * proximity + 2.0 * iou - 3.0
        bias = bias.clamp(-4.0, 3.0)
        return bias.masked_fill(~pair_valid, -1.0e4)

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
                    object_confs=object_confs,
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
                    object_confs=object_confs,
                    object_valid=object_valid,
                )
                _, x_actor = data[:2]
                x_heatmap = 0
                x_heatmap_feat = None
                x_visual_final = None
                x_object_prompt = None
            self.last_actor_motion_logits = None
            self.last_actor_tokens = x_actor
            self.last_actor_object_prompt_classes = None
            self.last_actor_object_prompt_tokens = None
            self.last_actor_object_prompt_attention_logits = None
            self.last_actor_object_prompt_attention = None
            self.last_actor_object_prompt_valid = None
            self.last_actor_object_base_fusion_attention = None
            self.last_actor_object_base_fusion_null_prob = None
            self.last_actor_object_base_fusion_useful_mass = None
            self.last_actor_object_base_fusion_delta = None
            self.last_actor_object_base_fusion_gate = None
            self.last_actor_object_base_fusion_scale = None
            self.last_actor_object_base_fusion_attention_bias = None
            self.last_actor_action_tokens = None
            self.last_object_residual_head_output = None
            self.last_object_residual_raw_base_logits = None
            self.last_object_residual_base_logits = None
            self.last_object_residual_final_logits = None
            self.last_object_residual = None
            self.last_object_residual_prompt_delta = None
            self.last_object_residual_coverage = None
            self.last_object_residual_prompt_null_prob = None
            self.last_object_residual_prompt_useful_mass = None
            self.last_object_residual_relation_scale = None
            self.last_object_residual_cache = None
            if self.hparams.ret_feat:
                return x_actor

            action_scores = None
            prompt_valid = None
            prompt_classes = None
            raw_base_logits = None
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

            x_actor_action = x_actor
            if self.actor_object_base_fusion is not None:
                raw_base_logits = self.actor_head(x_actor)
                if x_object_prompt is None or prompt_valid is None:
                    raise RuntimeError(
                        "actor_object_base_fusion requires runtime object prompt tokens"
                    )
                if object_confs is None:
                    raise ValueError("actor_object_base_fusion requires object_confs")
                object_attention_bias = self._actor_object_geometry_attention_bias(
                    boxes,
                    object_boxes,
                    valid,
                    prompt_valid,
                    x_actor.device,
                    x_actor.dtype,
                )
                fusion_output = self.actor_object_base_fusion(
                    x_actor,
                    x_object_prompt,
                    object_confs,
                    prompt_valid,
                    object_attention_bias=object_attention_bias,
                )
                self.last_actor_object_base_fusion_attention_bias = (
                    None
                    if object_attention_bias is None
                    else object_attention_bias.detach()
                )
                x_actor_action = fusion_output["actor_tokens"]
                self.last_actor_object_base_fusion_attention = fusion_output[
                    "object_attention"
                ]
                self.last_actor_object_base_fusion_null_prob = fusion_output[
                    "object_null_prob"
                ]
                self.last_actor_object_base_fusion_useful_mass = fusion_output[
                    "object_useful_mass"
                ]
                self.last_actor_object_base_fusion_delta = fusion_output[
                    "fusion_delta"
                ]
                self.last_actor_object_base_fusion_gate = fusion_output[
                    "fusion_gate"
                ]
                self.last_actor_object_base_fusion_scale = fusion_output[
                    "fusion_scale"
                ]
            self.last_actor_action_tokens = x_actor_action
            base_logits = self.actor_head(x_actor_action)
            if raw_base_logits is None:
                raw_base_logits = base_logits
            self.last_object_residual_raw_base_logits = raw_base_logits

            if self.object_residual_action_head is not None:
                if boxes is None or valid is None:
                    raise ValueError("actor_object_residual_head requires actor boxes")
                if x_object_prompt is None or prompt_valid is None:
                    raise RuntimeError(
                        "actor_object_residual_head requires runtime object prompt tokens"
                    )
                if object_classes is None or object_confs is None or object_valid is None:
                    raise ValueError(
                        "actor_object_residual_head requires object_classes, "
                        "object_confs, and object_valid"
                    )
                head_output = self.object_residual_action_head(
                    actor_tokens=x_actor,
                    actor_valid=valid,
                    base_logits=base_logits,
                    object_prompt_tokens=x_object_prompt,
                    object_classes=object_classes,
                    object_confs=object_confs,
                    object_valid=object_valid,
                )
                action_scores = head_output["log_probs"]
                self.last_object_residual_head_output = head_output
                self.last_object_residual_base_logits = head_output["base_logits"]
                self.last_object_residual_final_logits = head_output["final_logits"]
                self.last_object_residual = head_output["object_residual"]
                self.last_object_residual_prompt_delta = head_output["prompt_delta"]
                self.last_object_residual_coverage = head_output["coverage"]
                self.last_object_residual_prompt_null_prob = head_output[
                    "prompt_relation_null_prob"
                ]
                self.last_object_residual_prompt_useful_mass = head_output[
                    "prompt_relation_useful_mass"
                ]
                self.last_object_residual_relation_scale = head_output["relation_scale"]
                self.last_actor_motion_logits = head_output["motion_aux_logits"]
                self.last_actor_object_prompt_attention_logits = head_output[
                    "prompt_attention_logits"
                ]
                self.last_actor_object_prompt_attention = head_output[
                    "prompt_attention"
                ]
                self.last_actor_object_prompt_valid = prompt_valid
                self.last_object_residual_cache = {
                    "raw_actor_tokens": x_actor,
                    "actor_tokens": x_actor_action,
                    "actor_valid": valid,
                    "raw_base_logits": raw_base_logits,
                    "base_logits": base_logits,
                    "object_prompt_tokens": x_object_prompt,
                    "actor_boxes": boxes,
                    "object_boxes": object_boxes,
                    "object_classes": object_classes,
                    "object_confs": object_confs,
                    "object_valid": object_valid,
                }
            if self.actor_motion_head is not None:
                self.last_actor_motion_logits = self.actor_motion_head(x_actor)
            if action_scores is None:
                action_scores = base_logits
            self.last_actor_action_logits = action_scores
            if self.last_actor_motion_logits is None:
                self.last_actor_motion_logits = action_scores
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

    def cached_object_prompt_action_logits(
        self,
        object_classes=None,
        object_valid=None,
    ):
        cache = getattr(self, "last_object_residual_cache", None)
        if self.object_residual_action_head is None or cache is None:
            raise RuntimeError(
                "No cached runtime-object residual state; run forward before "
                "counterfactual logits"
            )

        prompt_tokens = cache["object_prompt_tokens"]
        prompt_classes = cache["object_classes"]
        if object_classes is not None:
            base_classes = getattr(self, "last_actor_object_prompt_classes", None)
            if base_classes is None:
                raise RuntimeError(
                    "No cached object classes for counterfactual prompt logits"
                )
            class_embed = getattr(self.net, "object_class_embed", None)
            if class_embed is None:
                raise RuntimeError("Cached object prompt logits require object_class_embed")
            none_id = int(self.hparams.get("num_object_classes", 19))
            prompt_classes = object_classes.to(
                device=prompt_tokens.device,
                dtype=torch.long,
            ).clamp(0, none_id)
            class_delta = class_embed(prompt_classes) - class_embed(base_classes)
            prompt_tokens = prompt_tokens + class_delta.to(dtype=prompt_tokens.dtype)

        prompt_valid = (
            cache["object_valid"]
            if object_valid is None
            else object_valid.to(device=prompt_tokens.device, dtype=torch.bool)
        )
        raw_actor_tokens = cache.get("raw_actor_tokens", cache["actor_tokens"])
        residual_actor_tokens = raw_actor_tokens
        base_logits = cache["base_logits"]
        if self.actor_object_base_fusion is not None:
            object_attention_bias = self._actor_object_geometry_attention_bias(
                cache.get("actor_boxes"),
                cache.get("object_boxes"),
                cache["actor_valid"],
                prompt_valid,
                raw_actor_tokens.device,
                raw_actor_tokens.dtype,
            )
            fusion_output = self.actor_object_base_fusion(
                raw_actor_tokens,
                prompt_tokens,
                cache["object_confs"],
                prompt_valid,
                object_attention_bias=object_attention_bias,
            )
            base_actor_tokens = fusion_output["actor_tokens"]
            base_logits = self.actor_head(base_actor_tokens)
        out = self.object_residual_action_head(
            actor_tokens=residual_actor_tokens,
            actor_valid=cache["actor_valid"],
            base_logits=base_logits,
            object_prompt_tokens=prompt_tokens,
            object_classes=prompt_classes,
            object_confs=cache["object_confs"],
            object_valid=prompt_valid,
        )
        return out["log_probs"]

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
        parser.add_argument("--num_scene_object_tokens", type=int, default=32)
        parser.add_argument("--num_object_classes", type=int, default=19)
        parser.add_argument("--actor_object_prompt_tokens", type=int, default=0)
        parser.add_argument("--actor_object_base_fusion", type=int, default=0)
        parser.add_argument(
            "--actor_object_base_fusion_hidden_dim",
            type=int,
            default=512,
        )
        parser.add_argument(
            "--actor_object_base_fusion_scale_init",
            type=float,
            default=-2.0,
        )
        parser.add_argument(
            "--actor_object_base_fusion_max_scale",
            type=float,
            default=1.0,
        )
        parser.add_argument("--actor_object_residual_head", type=int, default=0)
        parser.add_argument("--actor_object_residual_hidden_dim", type=int, default=512)
        parser.add_argument(
            "--actor_object_residual_relation_scale_init",
            type=float,
            default=-1.0,
        )
        parser.add_argument(
            "--actor_object_residual_relation_logit_bound",
            type=float,
            default=2.0,
        )
        parser.add_argument(
            "--actor_object_residual_max_relation_scale",
            type=float,
            default=1.5,
        )
        parser.add_argument(
            "--actor_object_residual_compat_prior_scale",
            type=float,
            default=1.0,
        )
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
        parser.add_argument("--trt_safe_attention", type=int, default=0)
        parser.add_argument("--interaction_unfreeze_last_blocks", type=int, default=0)
        parser.add_argument("--ret_feat", type=int, default=0)
        parser.add_argument("--linear_probe", type=int, default=0)

        return parser
