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
from models.actor_object_action_query_decoder import (
    ActionObjectQuerySpec,
    ActorObjectActionQueryDecoder,
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
        self.actor_object_slot_head_enabled = bool(
            self.hparams.get("actor_object_slot_head", 0)
        )
        self.actor_object_logit_residual = False
        self.actor_object_conditioned_action = False
        if bool(self.hparams.get("scene_object_tokens", 0)):
            raise ValueError(
                "scene_object_tokens was removed from the actor-object action "
                "path. Use actor_object_slot_head=1, which keeps detector object "
                "classes out of the transformer trunk and uses the action-query "
                "decoder."
            )
        if self.actor_object_slot_head_enabled and not self.actor_prompt:
            raise ValueError("actor_object_slot_head requires actor_prompt")
        if self.actor_interaction_heatmaps and not self.actor_prompt:
            raise ValueError("actor_interaction_heatmaps requires actor_prompt")
        if "interaction_object_classes" in self.hparams:
            raise ValueError(
                "interaction_object_classes was removed. Actor-object heatmaps "
                "are now one interacted-object channel per actor; object class "
                "semantics come from the actor-object action query decoder."
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
            if self.actor_object_slot_head_enabled:
                slot_spec = self._actor_object_slot_spec()
                if not slot_spec.action_to_object_ids:
                    raise ValueError(
                        "actor_object_slot_head requires a Toyota action-object "
                        "taxonomy mapping."
                    )
                self.actor_object_slot_head = ActorObjectActionQueryDecoder(
                    self.net.num_features,
                    spec=slot_spec,
                    hidden_dim=int(
                        self.hparams.get("actor_object_slot_hidden_dim", 512)
                    ),
                    attn_dim=int(
                        self.hparams.get("actor_object_slot_attn_dim", 256)
                    ),
                    compatible_bias=float(
                        self.hparams.get("actor_object_slot_prior_compatible", 0.75)
                    ),
                    incompatible_bias=float(
                        self.hparams.get("actor_object_slot_prior_incompatible", -0.75)
                    ),
                    unknown_bias=float(
                        self.hparams.get("actor_object_slot_unknown_init_bias", -0.10)
                    ),
                    unknown_mismatch_penalty=float(
                        self.hparams.get(
                            "actor_object_slot_unknown_mismatch_penalty",
                            1.0,
                        )
                    ),
                    quality_init_bias=float(
                        self.hparams.get("actor_object_slot_quality_init_bias", -3.0)
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

        return ActionObjectQuerySpec(
            num_actions=int(self.hparams.num_classes),
            num_object_classes=int(
                self.hparams.get("num_object_classes", NUM_OBJECT_CLASSES)
            ),
            objectless_action_indices=tuple(sorted(set(objectless_indices))),
            action_to_object_ids=action_to_object_ids,
        )

    def _object_heatmap_scores_from_predictions(
        self,
        heatmap_preds,
        object_boxes,
        object_valid,
        dtype,
    ):
        if not torch.is_tensor(heatmap_preds) or not self.actor_interaction_heatmaps:
            return None
        n_pose = int(self.hparams.get("n_landmarks", 0) or 0)
        n_actor = int(self.hparams.get("num_actor_tokens", 8) or 8)
        end = n_pose + n_actor
        if heatmap_preds.shape[1] < end:
            return None
        interaction_heatmaps = heatmap_preds[:, n_pose:end].to(dtype=dtype).clamp(
            0.0,
            1.0,
        )
        object_boxes = object_boxes.to(
            device=interaction_heatmaps.device,
            dtype=dtype,
        ).clamp(0.0, 1.0)
        object_valid_f = object_valid.to(
            device=interaction_heatmaps.device,
            dtype=dtype,
        )
        height = int(interaction_heatmaps.shape[-2])
        width = int(interaction_heatmaps.shape[-1])
        ys = torch.linspace(
            0.0,
            1.0,
            height,
            device=interaction_heatmaps.device,
            dtype=dtype,
        ).view(1, 1, height, 1)
        xs = torch.linspace(
            0.0,
            1.0,
            width,
            device=interaction_heatmaps.device,
            dtype=dtype,
        ).view(1, 1, 1, width)
        x1 = object_boxes[..., 0].view(object_boxes.shape[0], object_boxes.shape[1], 1, 1)
        y1 = object_boxes[..., 1].view(object_boxes.shape[0], object_boxes.shape[1], 1, 1)
        x2 = object_boxes[..., 2].view(object_boxes.shape[0], object_boxes.shape[1], 1, 1)
        y2 = object_boxes[..., 3].view(object_boxes.shape[0], object_boxes.shape[1], 1, 1)
        mask = (xs >= x1) & (xs <= x2) & (ys >= y1) & (ys <= y2)
        mask = mask.to(dtype=dtype) * object_valid_f[:, :, None, None]
        area = mask.sum(dim=(-2, -1)).clamp_min(1.0)
        score_sum = torch.einsum("bahw,bkhw->bak", interaction_heatmaps, mask)
        return (score_sum / area[:, None, :]).clamp(0.0, 1.0)

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
            if self.hparams.n_landmarks > 0 or self.actor_interaction_heatmaps:
                net_data = self.net(
                    x,
                    boxes=boxes,
                    valid=valid,
                )
                if len(net_data) == 4:
                    _, x_actor, x_heatmap, _ = net_data
                else:
                    _, x_actor, x_heatmap = net_data
            else:
                data = self.net(
                    x,
                    boxes=boxes,
                    valid=valid,
                )
                _, x_actor = data[:2]
                x_heatmap = 0
            self.last_actor_object_slot_logit_delta = None
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
            motion_action_logits = self.actor_head(x_actor)
            self.last_actor_action_logits = motion_action_logits
            if self.last_actor_motion_logits is None:
                self.last_actor_motion_logits = motion_action_logits
            action_logits = motion_action_logits
            if self.actor_object_slot_head is not None:
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
                    object_heatmap_scores = self._object_heatmap_scores_from_predictions(
                        x_heatmap,
                        object_boxes,
                        object_valid,
                        x_actor.dtype,
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
                    actor_boxes=boxes,
                    actor_valid=valid,
                    object_boxes=object_boxes,
                    object_classes=object_classes,
                    object_confs=object_confs,
                    object_valid=object_valid,
                    object_heatmap_scores=object_heatmap_scores,
                    motion_logits=motion_action_logits,
                )
                action_logits = slot_output["logits"]
                if slot_output.get("motion_logits") is not None:
                    self.last_actor_motion_logits = slot_output["motion_logits"]
                self.last_actor_object_slot_delta = slot_output["slot_delta"]
                self.last_actor_object_slot_posterior = slot_output[
                    "slot_posterior"
                ]
                self.last_actor_object_best_slot = slot_output["best_slot"]
                self.last_actor_object_quality = slot_output["object_quality"]
                self.last_actor_object_quality_logits = slot_output[
                    "object_quality_logits"
                ]
                self.last_actor_object_mismatch = slot_output["mismatch"]
                self.last_actor_object_unknown_delta = slot_output["unknown_delta"]
                self.last_actor_object_object_slot_delta = slot_output[
                    "object_slot_delta"
                ]
                self.last_actor_object_slot_logit_delta = (
                    action_logits - motion_action_logits
                )
            if self.presence_head is not None:
                presence_logits = self.presence_head(x_actor).squeeze(-1)
                return action_logits, x_heatmap, presence_logits
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
        parser.add_argument("--num_scene_object_tokens", type=int, default=32)
        parser.add_argument("--num_object_classes", type=int, default=19)
        parser.add_argument("--actor_object_slot_head", type=int, default=0)
        parser.add_argument("--actor_object_slot_hidden_dim", type=int, default=512)
        parser.add_argument("--actor_object_slot_attn_dim", type=int, default=256)
        parser.add_argument(
            "--actor_object_slot_prior_compatible",
            type=float,
            default=0.75,
        )
        parser.add_argument(
            "--actor_object_slot_prior_incompatible",
            type=float,
            default=-0.75,
        )
        parser.add_argument(
            "--actor_object_slot_unknown_init_bias",
            type=float,
            default=-0.10,
        )
        parser.add_argument(
            "--actor_object_slot_unknown_mismatch_penalty",
            type=float,
            default=1.0,
        )
        parser.add_argument(
            "--actor_object_slot_quality_init_bias",
            type=float,
            default=-3.0,
        )
        parser.add_argument("--trt_safe_attention", type=int, default=0)
        parser.add_argument("--interaction_unfreeze_last_blocks", type=int, default=0)
        parser.add_argument("--ret_feat", type=int, default=0)
        parser.add_argument("--linear_probe", type=int, default=0)

        return parser
