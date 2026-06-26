# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from timm.models.registry import register_model
import math


def _adaptive_avg_pool_weights(input_size, output_size):
    weights = torch.zeros(output_size, input_size, dtype=torch.float32)
    for out_idx in range(output_size):
        start = math.floor(out_idx * input_size / output_size)
        end = math.ceil((out_idx + 1) * input_size / output_size)
        weights[out_idx, start:end] = 1.0 / max(end - start, 1)
    return weights


class FixedAdaptiveAvgPool2d(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        input_h, input_w = to_2tuple(input_size)
        output_h, output_w = to_2tuple(output_size)
        self.input_h = int(input_h)
        self.input_w = int(input_w)
        self.output_h = int(output_h)
        self.output_w = int(output_w)
        self.register_buffer(
            "height_weight",
            _adaptive_avg_pool_weights(self.input_h, self.output_h),
            persistent=False,
        )
        self.register_buffer(
            "width_weight",
            _adaptive_avg_pool_weights(self.input_w, self.output_w),
            persistent=False,
        )

    def forward(self, x):
        batch, channels, height, width = x.shape
        if height != self.input_h or width != self.input_w:
            raise ValueError(
                "FixedAdaptiveAvgPool2d expected input "
                f"{self.input_h}x{self.input_w}, got {height}x{width}"
            )
        height_weight = self.height_weight.to(device=x.device, dtype=x.dtype)
        width_weight = self.width_weight.to(device=x.device, dtype=x.dtype)
        y = x.reshape(batch * channels, height, width)
        y = torch.matmul(height_weight, y)
        y = torch.matmul(y, width_weight.transpose(0, 1))
        return y.reshape(batch, channels, self.output_h, self.output_w)


class HeatmapHead(nn.Module):
    """
        HeatmapHead(
    (loss_module): KeypointMSELoss()
    (deconv_layers): Sequential(
        (0): ConvTranspose2d(768, 224, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        (1): BatchNorm2d(224, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): ConvTranspose2d(224, 224, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        (4): BatchNorm2d(224, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU(inplace=True)
    )
    (conv_layers): Identity()
    (final_layer): Conv2d(224, 13, kernel_size=(1, 1), stride=(1, 1))
    )
    """

    def __init__(
        self,
        in_channels=768,
        in_size=14,
        out_channels=13,
        loss_keypoint=None,
        deconv_out_channels=(224, 224),
        deconv_kernel_sizes=(4, 4),
    ):
        super(HeatmapHead, self).__init__()
        if in_size != 14:
            deconv_layers = [FixedAdaptiveAvgPool2d(in_size, (14, 14))]
        else:
            deconv_layers = []
        for i in range(len(deconv_out_channels)):
            deconv_layers.extend(
                [
                    nn.ConvTranspose2d(
                        in_channels,
                        deconv_out_channels[i],
                        kernel_size=deconv_kernel_sizes[i],
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(deconv_out_channels[i]),
                    nn.ReLU(inplace=True),
                ]
            )
            in_channels = deconv_out_channels[i]
        self.deconv_layers = nn.Sequential(*deconv_layers)
        self.conv_layers = nn.Identity()
        self.final_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x, target=None):
        x = x[-1]
        x = self.deconv_layers(x)
        x = self.conv_layers(x)
        x = self.final_layer(x)
        if target is None:
            return x

        return x, 0


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 400,
        "input_size": (3, 224, 224),
        "pool_size": None,
        "crop_pct": 0.9,
        "interpolation": "bicubic",
        "mean": (0.5, 0.5, 0.5),
        "std": (0.5, 0.5, 0.5),
        **kwargs,
    }


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CosAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        attn_head_dim=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        # self.scale = qk_scale or head_dim**-0.5
        # DO NOT RENAME [self.scale] (for no weight decay)
        if qk_scale is None:
            self.scale = nn.Parameter(
                torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True
            )
        else:
            self.scale = qk_scale

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat(
                (
                    self.q_bias,
                    torch.zeros_like(self.v_bias, requires_grad=False),
                    self.v_bias,
                )
            )
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)

        # torch.log(torch.tensor(1. / 0.01)) = 4.6052
        logit_scale = torch.clamp(self.scale, max=4.6052).exp()

        attn = attn * logit_scale

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x




class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        attn_head_dim=None,
        trt_safe_attention=False,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.trt_safe_attention = bool(trt_safe_attention)

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, key_padding_mask=None):
        B, N, C = x.shape
        if key_padding_mask is not None:
            if key_padding_mask.shape != (B, N):
                raise ValueError(
                    "key_padding_mask must have shape "
                    f"{(B, N)}, got {tuple(key_padding_mask.shape)}"
                )
            key_padding_mask = key_padding_mask.to(device=x.device, dtype=torch.bool)
            
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat(
                (
                    self.q_bias,
                    torch.zeros_like(self.v_bias, requires_grad=False),
                    self.v_bias,
                )
            )
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if not self.trt_safe_attention:
            dropout_p = self.attn_drop.p if self.training else 0.0
            attn_mask = None
            if key_padding_mask is not None and key_padding_mask.any():
                attn_mask = torch.zeros(
                    B, 1, 1, N, device=x.device, dtype=q.dtype
                )
                attn_mask.masked_fill_(
                    key_padding_mask[:, None, None, :],
                    torch.finfo(q.dtype).min,
                )
            x = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=dropout_p
            )
            x = x.transpose(1, 2).reshape(B, N, -1)
        else:
            attn = (q * self.scale) @ k.transpose(-2, -1)
            if key_padding_mask is not None and (
                self.trt_safe_attention or key_padding_mask.any()
            ):
                attn.masked_fill_(
                    key_padding_mask[:, None, None, :],
                    torch.finfo(attn.dtype).min,
                )
            attn = attn.softmax(dim=-1)
            attn_for_output = self.attn_drop(attn)
            x = (attn_for_output @ v).transpose(1, 2).reshape(B, N, -1)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        init_values=None,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        attn_head_dim=None,
        trt_safe_attention=False,
        **kwargs,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            attn_head_dim=attn_head_dim,
            trt_safe_attention=trt_safe_attention,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        if init_values > 0:
            self.gamma_1 = nn.Parameter(
                init_values * torch.ones((dim)), requires_grad=True
            )
            self.gamma_2 = nn.Parameter(
                init_values * torch.ones((dim)), requires_grad=True
            )
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, key_padding_mask=None):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x), key_padding_mask=key_padding_mask))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), key_padding_mask=key_padding_mask))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        if key_padding_mask is not None:
            x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        num_frames=16,
        tubelet_size=2,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_spatial_patches = (img_size[0] // patch_size[0]) * (
            img_size[1] // patch_size[1]
        )
        num_patches = num_spatial_patches * (num_frames // tubelet_size)

        self.img_size = img_size
        self.tubelet_size = tubelet_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv3d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=(self.tubelet_size, patch_size[0], patch_size[1]),
            stride=(self.tubelet_size, patch_size[0], patch_size[1]),
        )

    def forward(self, x, **kwargs):
        B, C, T, H, W = x.shape
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # b, c, l -> b, l, c
        x = self.proj(x)  # .flatten(2).transpose(1, 2)
        return x


# sin-cos position encoding
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py#L31
def get_sinusoid_encoding_table(n_position, d_hid):
    """Sinusoid position encoding table"""

    # TODO: make it with torch instead of numpy
    def get_position_angle_vec(position):
        return [
            position / np.power(10000, 2 * (hid_j // 2) / d_hid)
            for hid_j in range(d_hid)
        ]

    sinusoid_table = np.array(
        [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
    )
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.tensor(
        sinusoid_table, dtype=torch.float, requires_grad=False
    ).unsqueeze(0)


class VisionTransformer(nn.Module):
    """Vision Transformer with support for patch or hybrid CNN input stage"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        head_drop_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_values=0.0,
        use_learnable_pos_emb=False,
        init_scale=0.0,
        all_frames=16,
        tubelet_size=2,
        use_mean_pooling=True,
        with_cp=False,
        cos_attn=False,
        keep_rate=1.0,
        keep_rate_merge=1.0,
        n_landmarks=17,
        mode="train",
        hw_out_conv=14,
        n_registers=0,
        actor_prompt=0,
        num_actor_tokens=8,

        actor_interaction_heatmaps=0,
        actor_object_prompt_tokens=0,
        num_scene_object_tokens=32,
        num_object_classes=19,

        return_heatmap_features=False,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        # num_features for consistency with other models
        self.num_features = self.embed_dim = embed_dim
        self.tubelet_size = tubelet_size
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            num_frames=all_frames,
            tubelet_size=tubelet_size,
        )
        num_patches = self.patch_embed.num_patches
        self.with_cp = with_cp
        self.n_registers = n_registers
        self.actor_prompt = bool(actor_prompt)
        self.n_actor_tokens = int(num_actor_tokens) if self.actor_prompt else 0
        if self.n_actor_tokens < 0:
            raise ValueError("num_actor_tokens must be non-negative")
        self.actor_interaction_heatmaps = bool(actor_interaction_heatmaps)
        if self.actor_interaction_heatmaps and not self.actor_prompt:
            raise ValueError("actor_interaction_heatmaps requires actor_prompt")
        self.actor_object_prompt_tokens = bool(actor_object_prompt_tokens)
        if self.actor_object_prompt_tokens and not self.actor_prompt:
            raise ValueError("actor_object_prompt_tokens requires actor_prompt")
        if self.actor_object_prompt_tokens and not self.actor_interaction_heatmaps:
            raise ValueError(
                "actor_object_prompt_tokens requires actor_interaction_heatmaps"
            )
        self.n_object_tokens = (
            int(num_scene_object_tokens) if self.actor_object_prompt_tokens else 0
        )
        self.num_object_classes = int(num_object_classes)

        if "interaction_object_classes" in kwargs:
            raise ValueError(
                "interaction_object_classes was removed. Actor-object heatmaps "
                "are now one interacted-object channel per actor; object class "
                "semantics come from relation-only runtime object memory."
            )
        self.return_heatmap_features = bool(return_heatmap_features)
        self.HW_OUT_CONV = (hw_out_conv, hw_out_conv)
        self.n_heatmap_tokens = self.HW_OUT_CONV[0] * self.HW_OUT_CONV[1]
        self.n_landmarks = int(n_landmarks)
        self.n_interaction_heatmap_channels = (
            self.n_actor_tokens
            if self.actor_interaction_heatmaps
            else 0
        )
        self.n_heatmap_out_channels = (
            self.n_landmarks + self.n_interaction_heatmap_channels
        )
        if self.n_heatmap_out_channels == 0:
            self.n_heatmap_tokens = 0
        self.mode = mode

        self.N_KEY_TOKENS = (
            1 + self.n_actor_tokens + self.n_object_tokens + self.n_registers
        )

        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        else:
            # sine-cosine positional embeddings is on the way
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        self.pos_drop = nn.Dropout(p=drop_rate)
        self.depth = depth

        if depth == 12:
            keep_rate = [1, 1, 1, keep_rate, 1, 1, keep_rate, 1, 1, keep_rate, 1, 1]
            keep_rate_merge = [
                1,
                1,
                1,
                keep_rate_merge,
                1,
                1,
                keep_rate_merge,
                1,
                1,
                keep_rate_merge,
                1,
                1,
            ]
        elif depth == 24:
            keep_rate = [
                1,
                1,
                1,
                keep_rate,
            ] * 6
            keep_rate_merge = [
                1,
                1,
                1,
                keep_rate_merge,
            ] * 6

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    init_values=init_values,
                    cos_attn=cos_attn,
                    keep_rate=keep_rate[i],
                    keep_rate_merge=keep_rate_merge[i],
                    n_heatmap_tokens=self.n_heatmap_tokens,
                    n_key_tokens=self.N_KEY_TOKENS,
                    **kwargs,
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        self.head_dropout = nn.Dropout(head_drop_rate)
        self.head = (
            nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=0.02)

        self.apply(self._init_weights)

        self.head.weight.data.mul_(init_scale)
        self.head.bias.data.mul_(init_scale)
        self.class_token = nn.Parameter(torch.zeros(1, 1, self.num_features))
        if self.n_actor_tokens > 0:
            self.actor_token = nn.Parameter(torch.zeros(1, 1, self.num_features))
            self.actor_slot_embed = nn.Parameter(
                torch.zeros(1, self.n_actor_tokens, self.num_features)
            )
            self.valid_embed = nn.Embedding(2, self.num_features)
            self.bbox_mlp = nn.Sequential(
                nn.Linear(4, self.num_features),
                nn.GELU(),
                nn.Linear(self.num_features, self.num_features),
            )
            trunc_normal_(self.actor_token, std=0.02)
            nn.init.zeros_(self.actor_slot_embed)
            nn.init.zeros_(self.valid_embed.weight)
            nn.init.zeros_(self.bbox_mlp[-1].weight)
            nn.init.zeros_(self.bbox_mlp[-1].bias)
        if self.actor_object_prompt_tokens:
            self.object_slot_embed = nn.Parameter(
                torch.zeros(1, self.n_object_tokens, self.num_features)
            )
            self.object_class_embed = nn.Embedding(
                self.num_object_classes + 1,
                self.num_features,
            )
            self.object_box_mlp = nn.Sequential(
                nn.Linear(4, self.num_features),
                nn.GELU(),
                nn.Linear(self.num_features, self.num_features),
            )
            self.object_valid_embed = nn.Embedding(2, self.num_features)
            self.object_region_norm = nn.LayerNorm(self.num_features)
            self.object_region_proj = nn.Linear(self.num_features, self.num_features)
            object_grid_h = self.patch_embed.img_size[0] // self.patch_embed.patch_size[0]
            object_grid_w = self.patch_embed.img_size[1] // self.patch_embed.patch_size[1]
            y_centers = (torch.arange(object_grid_h, dtype=torch.float32) + 0.5) / float(
                object_grid_h
            )
            x_centers = (torch.arange(object_grid_w, dtype=torch.float32) + 0.5) / float(
                object_grid_w
            )
            grid_y, grid_x = torch.meshgrid(y_centers, x_centers, indexing="ij")
            self.register_buffer(
                "object_region_grid_x",
                grid_x.reshape(1, 1, -1),
                persistent=False,
            )
            self.register_buffer(
                "object_region_grid_y",
                grid_y.reshape(1, 1, -1),
                persistent=False,
            )
            trunc_normal_(self.object_class_embed.weight, std=0.02)
            nn.init.zeros_(self.object_slot_embed)
            nn.init.zeros_(self.object_valid_embed.weight)
            trunc_normal_(self.object_box_mlp[-1].weight, std=0.01)
            nn.init.zeros_(self.object_box_mlp[-1].bias)
            trunc_normal_(self.object_region_proj.weight, std=0.01)
            nn.init.zeros_(self.object_region_proj.bias)
        if self.n_heatmap_out_channels > 0:
            self.heatmap_tokens = nn.Parameter(
                torch.randn(1, self.HW_OUT_CONV[0] * self.HW_OUT_CONV[1], embed_dim)
            )
            self.heatmap_head = HeatmapHead(
                in_channels=embed_dim,
                in_size=self.HW_OUT_CONV[0],
                out_channels=self.n_heatmap_out_channels,
                deconv_out_channels=(224, 224),
                deconv_kernel_sizes=(4, 4),
            )
        if self.n_registers > 0:
            self.register_tokens = nn.Parameter(
                torch.randn(1, self.n_registers, self.num_features)
            )

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        self.num_classes = num_classes
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

    @staticmethod
    def _parse_relation_blocks(blocks):
        if blocks is None:
            return tuple()
        if isinstance(blocks, str):
            blocks = blocks.strip()
            if not blocks:
                return tuple()
            values = [part.strip() for part in blocks.split(",")]
            return tuple(sorted(set(int(value) for value in values if value)))
        if isinstance(blocks, (list, tuple, set)):
            return tuple(sorted(set(int(value) for value in blocks)))
        return (int(blocks),)



    def forward(
        self,
        x,
        boxes=None,
        valid=None,
        object_boxes=None,
        object_classes=None,
        object_valid=None,
    ):
        # x_list, images_whwh = self.preprocess_image(x_list)

        x = self.patch_embed(x)  # (B, C_e, t, h, w)
        ws = x.shape[2:]

        num_frames = x.shape[2]
        assert num_frames % 2 == 0, "Only consider even case, check frames {}".format(
            num_frames
        )

        x = x.flatten(2).transpose(1, 2)  # (B, N, C_e), N = t x h x w
        B, N, C = x.shape

        if self.pos_embed is not None:
            x = (
                x
                + self.pos_embed.expand(B, -1, -1)
                .type_as(x)
                .to(x.device)
                .clone()
                .detach()
        )
        x = self.pos_drop(x)
        token_key_padding_mask = None
        object_memory_tokens = None

        prefix_tokens = []
        prefix_key_masks = []
        if self.n_actor_tokens > 0:
            if boxes is None:
                raise ValueError("boxes are required when actor_prompt is enabled")
            if boxes.ndim != 3 or boxes.shape[1:] != (self.n_actor_tokens, 4):
                raise ValueError(
                    "boxes must have shape "
                    f"[B,{self.n_actor_tokens},4], got {tuple(boxes.shape)}"
                )
            if boxes.shape[0] != B:
                raise ValueError(
                    f"boxes batch size {boxes.shape[0]} does not match video batch {B}"
                )
            boxes = boxes.to(device=x.device, dtype=x.dtype)
            if valid is None:
                raise ValueError("valid is required when actor_prompt is enabled")
            if valid.shape != boxes.shape[:2]:
                raise ValueError(
                    "valid must have shape "
                    f"[B,{self.n_actor_tokens}], got {tuple(valid.shape)}"
                )
            valid = valid.to(device=x.device, dtype=torch.bool)
            actor_tokens = (
                self.actor_token.expand(B, self.n_actor_tokens, -1)
                + self.actor_slot_embed.expand(B, -1, -1)
                + self.bbox_mlp(boxes)
                + self.valid_embed(valid.long()).to(dtype=x.dtype)
            )
            prefix_tokens.append(actor_tokens)
            prefix_key_masks.append(valid == 0)

        if self.n_registers > 0:
            prefix_tokens.append(self.register_tokens.expand(B, -1, -1))
            prefix_key_masks.append(
                torch.zeros(B, self.n_registers, dtype=torch.bool, device=x.device)
            )
        if self.actor_object_prompt_tokens:
            if object_boxes is None or object_classes is None:
                raise ValueError(
                    "object_boxes and object_classes are required when "
                    "actor_object_prompt_tokens is enabled"
                )
            if object_valid is None:
                raise ValueError(
                    "object_valid is required when "
                    "actor_object_prompt_tokens is enabled"
                )
            if object_boxes.ndim != 3 or object_boxes.shape[1:] != (
                self.n_object_tokens,
                4,
            ):
                raise ValueError(
                    "object_boxes must have shape "
                    f"[B,{self.n_object_tokens},4], got {tuple(object_boxes.shape)}"
                )
            if object_boxes.shape[0] != B:
                raise ValueError(
                    "object_boxes batch size "
                    f"{object_boxes.shape[0]} does not match video batch {B}"
                )
            if object_classes.shape != object_boxes.shape[:2]:
                raise ValueError(
                    "object_classes must have shape "
                    f"{tuple(object_boxes.shape[:2])}, got {tuple(object_classes.shape)}"
                )
            if object_valid.shape != object_boxes.shape[:2]:
                raise ValueError(
                    "object_valid must have shape "
                    f"{tuple(object_boxes.shape[:2])}, got {tuple(object_valid.shape)}"
            )
            object_boxes = object_boxes.to(device=x.device, dtype=x.dtype).clamp(0.0, 1.0)
            object_classes = object_classes.to(device=x.device, dtype=torch.long)
            object_valid = object_valid.to(device=x.device, dtype=torch.bool)
            none_id = self.num_object_classes
            object_classes = torch.where(
                object_valid,
                object_classes.clamp(0, none_id),
                torch.full_like(object_classes, none_id),
            )
            object_tokens = (
                self.object_slot_embed.expand(B, -1, -1)
                + self.object_class_embed(object_classes).to(dtype=x.dtype)
                + self.object_box_mlp(object_boxes)
                + self.object_valid_embed(object_valid.long()).to(dtype=x.dtype)
            )
            object_memory_tokens = object_tokens
            prefix_tokens.append(object_tokens)
            prefix_key_masks.append(object_valid == 0)
        if self.n_heatmap_out_channels > 0:
            prefix_tokens.append(self.heatmap_tokens.expand(B, -1, -1))
            prefix_key_masks.append(
                torch.zeros(B, self.n_heatmap_tokens, dtype=torch.bool, device=x.device)
            )

        if prefix_tokens:
            x = torch.cat([*prefix_tokens, x], dim=1)
            prefix_mask = torch.cat(prefix_key_masks, dim=1)
            video_mask = torch.zeros(B, N, dtype=torch.bool, device=x.device)
            token_key_padding_mask = torch.cat([prefix_mask, video_mask], dim=1)

        # append class token to the beginning of the sequence
        x = torch.cat([self.class_token.expand(B, -1, -1), x], dim=1)
        if token_key_padding_mask is not None:
            class_mask = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
            token_key_padding_mask = torch.cat(
                [class_mask, token_key_padding_mask],
                dim=1,
            )
        actor_start = 1
        actor_end = actor_start + self.n_actor_tokens
        heatmap_start = self.N_KEY_TOKENS
        heatmap_end = heatmap_start + self.n_heatmap_tokens


        for i in range(self.depth):
            blk = self.blocks[i]
            x = blk(x, key_padding_mask=token_key_padding_mask)



        if self.n_actor_tokens > 0:
            x_actor = x[:, actor_start:actor_end, :]
        x_object = None
        if self.actor_object_prompt_tokens:
            object_start = 1 + self.n_actor_tokens + self.n_registers
            object_end = object_start + self.n_object_tokens
            x_object = x[:, object_start:object_end, :]
        if self.n_heatmap_out_channels > 0:
            x_heatmap = x[
                :,
                heatmap_start:heatmap_end,
                :,
            ]
        x_visual = None
        if self.return_heatmap_features:
            visual_start = self.N_KEY_TOKENS + self.n_heatmap_tokens
            x_visual = x[:, visual_start:, :]
        x_class = x[:, 0, :]
        if self.fc_norm is not None:
            x_class = self.fc_norm(x_class)
            if self.n_actor_tokens > 0:
                x_actor = self.fc_norm(x_actor)
            if x_object is not None:
                x_object = self.fc_norm(x_object)
            if x_visual is not None:
                x_visual = self.fc_norm(x_visual)
        else:
            x_class = self.norm(x_class)
            if self.n_actor_tokens > 0:
                x_actor = self.norm(x_actor)
            if x_object is not None:
                x_object = self.norm(x_object)
            if x_visual is not None:
                x_visual = self.norm(x_visual)
        x_class = self.head_dropout(x_class)
        x_class = self.head(x_class)
        if self.n_actor_tokens > 0:
            x_actor = self.head_dropout(x_actor)
        if x_object is not None:
            x_object = self.head_dropout(x_object)
        if x_visual is not None:
            x_visual = self.head_dropout(x_visual)

        if self.n_heatmap_out_channels == 0:
            if self.n_actor_tokens > 0:
                return x_class, x_actor
            return x_class
        x_heatmap_feat = x_heatmap.reshape(B, *self.HW_OUT_CONV, -1).permute(
            0,
            3,
            1,
            2,
        )
        x_heatmap = tuple([0, x_heatmap_feat])
        x_heatmap = self.heatmap_head(x_heatmap)
        if self.n_actor_tokens > 0:
            if self.return_heatmap_features:
                return x_class, x_actor, x_heatmap, x_heatmap_feat, x_visual, x_object
            return x_class, x_actor, x_heatmap
        return x_class, x_heatmap


@register_model
def vit_small_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    model.default_cfg = _cfg()
    return model


@register_model
def vit_base_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    model.default_cfg = _cfg()
    return model


@register_model
def vit_large_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    model.default_cfg = _cfg()
    return model


@register_model
def vit_huge_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    model.default_cfg = _cfg()
    return model


@register_model
def vit_giant_patch14_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=14,
        embed_dim=1408,
        depth=40,
        num_heads=16,
        mlp_ratio=48 / 11,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    model.default_cfg = _cfg()
    return model
