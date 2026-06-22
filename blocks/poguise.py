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
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim**-0.5

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

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def sim_matrixv2_batch(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=-1)[:, :, None], b.norm(dim=-1)[:, :, None]
    a_norm = a / torch.clamp(a_n, min=eps)
    b_norm = b / torch.clamp(b_n, min=eps)
    sim_mt = torch.bmm(a_norm, b_norm.transpose(-2, -1))
    return sim_mt


class KTPAttention(Attention):
    """Attention with Keyframe-centric Token Pruning (KTP)."""

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        attn_head_dim=None,
        use_checkpoint=False,
        keep_rate=0.0,
        enhanced_weight_class=1,
        enhanced_weight_heatmap=1,
        n_heatmap_tokens=196,
        sim_metric=0,
        topk_type=0,
        n_key_tokens=1,
        bbox_prior_weight=0.0,
        semantic_token_score_weights=None,
        needs_full_attention=False,
        trt_safe_attention=False,
    ):
        super(KTPAttention, self).__init__(
            dim, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop, attn_head_dim
        )
        self.keep_rate = keep_rate
        assert 0 < keep_rate <= 1, "keep_rate must > 0 and <= 1, got {0}".format(
            keep_rate
        )
        self.enhanced_weight_class = enhanced_weight_class
        self.enhanced_weight_heatmap = enhanced_weight_heatmap
        self.n_heatmap_tokens = n_heatmap_tokens
        self.use_checkpoint = use_checkpoint
        # self.merge_type = "sim"
        self.topk_type = topk_type  # 0: all, 1: cls_hm
        self.sim_metric = sim_metric  # 0: k, 1: attn
        self.n_key_tokens = n_key_tokens
        self.bbox_prior_weight = float(bbox_prior_weight)
        if semantic_token_score_weights is None:
            self.semantic_token_score_weights = None
        else:
            semantic_token_score_weights = torch.as_tensor(
                semantic_token_score_weights,
                dtype=torch.float32,
            )
            if semantic_token_score_weights.ndim != 1:
                raise ValueError("semantic_token_score_weights must be a 1D tensor")
            self.register_buffer(
                "semantic_token_score_weights",
                semantic_token_score_weights,
                persistent=False,
            )
        self.needs_full_attention = bool(needs_full_attention)
        self.trt_safe_attention = bool(trt_safe_attention)

    def forward_part1(self, x, size=None, key_padding_mask=None):
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
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        if (
            self.keep_rate >= 1
            and not self.needs_full_attention
            and not self.trt_safe_attention
        ):
            # use flash attention
            dropout_p = self.attn_drop.p if self.training else 0.0
            attn_mask = None
            if key_padding_mask is not None and key_padding_mask.any():
                attn_mask = torch.zeros(
                    B,
                    1,
                    1,
                    N,
                    device=x.device,
                    dtype=q.dtype,
                )
                attn_mask.masked_fill_(
                    key_padding_mask[:, None, None, :],
                    torch.finfo(q.dtype).min,
                )
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
            )
            x = x.transpose(1, 2).reshape(B, N, -1)
            attn = None
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
        sim_feat = x
        if self.sim_metric == 1:
            sim_feat = attn
        elif self.sim_metric == 0:
            sim_feat = k
        elif self.sim_metric == 2:
            sim_feat = q
        elif self.sim_metric == 3:
            sim_feat = v
        return x, attn, sim_feat

    def forward(
        self,
        x,
        last_idx=None,
        ws=None,
        size=None,
        bbox_token_prior=None,
        key_padding_mask=None,
    ):
        B, N, C = x.shape
        if self.keep_rate < 1:
            x, attn, feature = self.forward_part1(
                x, size, key_padding_mask=key_padding_mask
            )
        else:
            x, attn, feature = self.forward_part1(
                x, size, key_padding_mask=key_padding_mask
            )
            return x, last_idx, attn if self.sim_metric == 1 else feature

        # get top-k tokens and the corresponding indexes
        if self.keep_rate < 1:
            num_s_tokens = self.n_heatmap_tokens + self.n_key_tokens
            num_keep_tokens = math.ceil(self.keep_rate * (N - num_s_tokens))
            if num_keep_tokens > 0:
                attn_topk = attn.clone()
                if key_padding_mask is not None and (
                    self.trt_safe_attention or key_padding_mask.any()
                ):
                    attn_topk = attn_topk.masked_fill(
                        key_padding_mask[:, None, :, None],
                        0.0,
                    )
                prefix_count = self.n_heatmap_tokens + self.n_key_tokens
                if self.semantic_token_score_weights is not None:
                    if self.semantic_token_score_weights.numel() != prefix_count:
                        raise RuntimeError(
                            "semantic token score weights must match protected "
                            f"prefix length {prefix_count}, got "
                            f"{self.semantic_token_score_weights.numel()}"
                        )
                    semantic_weights = self.semantic_token_score_weights.to(
                        device=attn_topk.device,
                        dtype=attn_topk.dtype,
                    )
                    attn_topk[:, :, :prefix_count] *= semantic_weights.view(
                        1,
                        1,
                        -1,
                        1,
                    )
                else:
                    # Legacy PO-GUISE weighting: class plus heatmap tokens guide pruning.
                    attn_topk[:, :, 0] *= self.enhanced_weight_class
                    if self.n_heatmap_tokens:
                        attn_topk[
                            :,
                            :,
                            self.n_key_tokens : self.n_heatmap_tokens + self.n_key_tokens,
                        ] *= self.enhanced_weight_heatmap
                if self.topk_type == 0:
                    attn_topk = attn_topk.sum(dim=-2).mean(
                        dim=1
                    )  # (B, N) for each token sum how much it is attended by all other tokens

                elif self.topk_type == 1:
                    attn_topk = (
                        attn_topk[:, :, :prefix_count]
                        .sum(dim=-2)
                        .mean(dim=1)
                    )
                elif self.topk_type == 2:
                    # only class token
                    attn_topk = attn_topk[:, :, 0].mean(dim=1)
                attn_topk = attn_topk[
                    :, num_s_tokens:
                ]  # remove class token and heatmap tokens
                if bbox_token_prior is not None and self.bbox_prior_weight > 0:
                    bbox_prior = torch.gather(
                        bbox_token_prior.to(dtype=attn_topk.dtype),
                        dim=1,
                        index=last_idx,
                    )
                    attn_topk = attn_topk + self.bbox_prior_weight * bbox_prior
                # average on all queries and num_heads
                _, idx_evad = torch.topk(
                    attn_topk,
                    num_keep_tokens,
                    dim=1,
                    largest=True,
                )  # (B, N_keep)
                last_idx = idx_evad.sort(dim=1)[0]
        if self.sim_metric == 1:
            return x, last_idx, attn
        else:
            return x, last_idx, feature


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
        cos_attn=False,
        keep_rate=1.0,
        n_heatmap_tokens=196,
        merge_mode=0,
        sim_metric=0,
        merge_type="sim",
        keep_rate_merge=1.0,
        n_key_tokens=1,
        **kwargs,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.keep_rate = keep_rate
        self.n_heatmap_tokens = n_heatmap_tokens
        self.merge_mode = merge_mode
        self.sim_metric = sim_metric
        self.merge_type = merge_type
        self.keep_rate_merge = keep_rate_merge
        self.n_key_tokens = n_key_tokens
        self.attn = KTPAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            attn_head_dim=attn_head_dim,
            keep_rate=keep_rate,
            n_heatmap_tokens=n_heatmap_tokens,
            sim_metric=sim_metric,
            n_key_tokens=n_key_tokens,
            bbox_prior_weight=kwargs.pop("bbox_prior_weight", 0.0),
            semantic_token_score_weights=kwargs.pop(
                "semantic_token_score_weights",
                None,
            ),
            needs_full_attention=keep_rate_merge < 1,
            **kwargs,
        )

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self._size_attn = None

        if init_values > 0:
            self.gamma_1 = nn.Parameter(
                init_values * torch.ones((dim)), requires_grad=True
            )
            self.gamma_2 = nn.Parameter(
                init_values * torch.ones((dim)), requires_grad=True
            )
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward_KTPpart1(
        self,
        x,
        last_idx,
        window_size,
        bbox_token_prior=None,
        key_padding_mask=None,
    ):
        # check if self._size_attn shape matches with x
        if self.gamma_1 is None:
            tmp, idx, attn = self.attn(
                self.norm1(x),
                last_idx,
                window_size,
                self._size_attn,
                bbox_token_prior=bbox_token_prior,
                key_padding_mask=key_padding_mask,
            )
            return self.drop_path(tmp), idx, attn
        else:
            attn, idx = self.attn(
                self.norm1(x),
                last_idx,
                window_size,
                self._size_attn,
                key_padding_mask=key_padding_mask,
            )
            return self.drop_path(self.gamma_1 * attn), idx

    def forward_KTP_part2(self, x):
        if self.gamma_2 is None:
            return self.drop_path(self.mlp(self.norm2(x)))
        else:
            return self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))

    def forward(
        self,
        x,
        last_idx,
        window_size,
        bbox_token_prior=None,
        key_padding_mask=None,
    ):
        # attn
        tmp, idx, attn = self.forward_KTPpart1(
            x,
            last_idx,
            window_size,
            bbox_token_prior=bbox_token_prior,
            key_padding_mask=key_padding_mask,
        )
        x = x + tmp

        # class token heatmap -centric token pruning
        B, N, C = x.shape
        num_s_tokens = self.n_heatmap_tokens + self.n_key_tokens  # 1 for class token
        if key_padding_mask is not None:
            if key_padding_mask.shape != (B, N):
                raise ValueError(
                    "key_padding_mask must have shape "
                    f"{(B, N)}, got {tuple(key_padding_mask.shape)}"
                )
            key_padding_mask = key_padding_mask.to(device=x.device, dtype=torch.bool)
        if self.keep_rate < 1:
            x_key = x[:, :num_s_tokens]
            x_nonkey_keep = x[:, num_s_tokens:]
            index_nonkey_keep = idx.unsqueeze(-1).expand(-1, -1, C)  # (B, N_keep, C_e)
            x_nonkey_keep = torch.gather(
                x_nonkey_keep, dim=1, index=index_nonkey_keep
            )  # (B, N_keep, C_e)

            if self.keep_rate_merge < 1:
                selected = torch.zeros(
                    B,
                    N - num_s_tokens,
                    dtype=x.dtype,
                    device=x.device,
                )
                selected = selected.scatter(
                    1,
                    idx.long(),
                    torch.ones_like(idx, dtype=x.dtype, device=x.device),
                )
                num_nonselected = (N - num_s_tokens) - idx.shape[1]
                idx_nonselected = torch.topk(
                    1.0 - selected,
                    num_nonselected,
                    dim=1,
                ).indices
                idx_nonselected = torch.sort(idx_nonselected, dim=1).values
                x_nonselected = torch.gather(
                    x[:, num_s_tokens:],
                    dim=1,
                    index=idx_nonselected.unsqueeze(-1).expand(-1, -1, C),
                )
                attn_merge = attn.clone()
                attn_merge = attn_merge.mean(
                    dim=1
                )  # (B, N) for each token sum how much it is attended by all other tokens
                attn_merge = attn_merge[
                    :, num_s_tokens:
                ]  # remove class token and heatmap tokens
                # filter attn by idx_nonselected
                if self.sim_metric == 1:
                    attn_merge = attn_merge.gather(
                        dim=2,
                        index=idx_nonselected.unsqueeze(1).expand(
                            -1, attn_merge.shape[1], -1
                        ),
                    )
                    # filter on the second attention
                    attn_merge = attn_merge.gather(
                        dim=1,
                        index=idx_nonselected.unsqueeze(-1).expand(
                            -1, -1, attn_merge.shape[-1]
                        ),
                    )
                else:
                    # filter attn_merge only on the first dimension
                    attn_merge = attn_merge.gather(
                        dim=1,
                        index=idx_nonselected.unsqueeze(-1).expand(
                            -1, -1, attn_merge.shape[-1]
                        ),
                    )
                idx = torch.gather(last_idx, dim=1, index=idx)
                merge, src_idx = self.bipartite_soft_matching(
                    attn_merge, int(self.keep_rate_merge * attn_merge.shape[1])
                )
                x_nonselected = self.merge_wavg(merge, x_nonselected)
                # update idx_nonselected by src_idx
                idx_nonselected = idx_nonselected.gather(
                    dim=1,
                    index=src_idx.squeeze(-1),
                )
                idx_nonselected = torch.gather(last_idx, dim=1, index=idx_nonselected)
                x_nonkey_keep = torch.cat([x_nonkey_keep, x_nonselected], dim=1)
                # update the global index in video sequence
                idx = torch.cat([idx, idx_nonselected], dim=1)
            else:
                idx = torch.gather(last_idx, dim=1, index=idx)

            x = torch.cat([x_key, x_nonkey_keep], dim=1)
            if key_padding_mask is not None:
                key_mask = key_padding_mask[:, :num_s_tokens]
                nonkey_mask = torch.zeros(
                    B,
                    x_nonkey_keep.shape[1],
                    dtype=torch.bool,
                    device=x.device,
                )
                key_padding_mask = torch.cat([key_mask, nonkey_mask], dim=1)
        elif self.keep_rate_merge < 1:
            num_s_tokens = self.n_heatmap_tokens + self.n_key_tokens
            x_key = x[:, :num_s_tokens]
            x_nonkey_keep = x[:, num_s_tokens:]
            attn_merge = attn.clone()
            attn_merge = attn_merge.mean(dim=1)
            attn_merge = attn_merge[:, num_s_tokens:]
            merge, src_idx = self.bipartite_soft_matching(
                attn_merge, int(self.keep_rate_merge * attn_merge.shape[1])
            )
            x_nonkey_keep = self.merge_wavg(merge, x_nonkey_keep)
            x = torch.cat([x_key, x_nonkey_keep], dim=1)
            if key_padding_mask is not None:
                key_mask = key_padding_mask[:, :num_s_tokens]
                nonkey_mask = torch.zeros(
                    B,
                    x_nonkey_keep.shape[1],
                    dtype=torch.bool,
                    device=x.device,
                )
                key_padding_mask = torch.cat([key_mask, nonkey_mask], dim=1)
        x = x + self.forward_KTP_part2(x)
        if key_padding_mask is not None:
            x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

        return x, idx, key_padding_mask

    def bipartite_soft_matching(
        self,
        metric,
        r,
    ):
        """
        Modified from ToMe:
        https://github.com/facebookresearch/ToMe/blob/main/tome/merge.py#L228
        """

        def sim_matrixv2_batch(a, b, eps=1e-8):
            """
            added eps for numerical stability
            """
            a_n, b_n = a.norm(dim=-1)[:, :, None], b.norm(dim=-1)[:, :, None]
            a_norm = a / torch.clamp(a_n, min=eps)
            b_norm = b / torch.clamp(b_n, min=eps)
            sim_mt = torch.bmm(a_norm, b_norm.transpose(-2, -1))
            return sim_mt

        with torch.no_grad():

            if self.merge_type == "sim":
                scores = sim_matrixv2_batch(metric, metric)
                diag_mask = torch.eye(
                    scores.shape[-1],
                    device=scores.device,
                    dtype=torch.bool,
                ).unsqueeze(0)
                scores = scores.masked_fill(diag_mask, -1.0e4)
                node_max, node_idx = scores.max(dim=-1)
                edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]
                src_idx = edge_idx[..., :r, :]  # Merged Tokens
                dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)
            else:
                with torch.no_grad():
                    metric = metric / metric.norm(dim=-1, keepdim=True)
                    a, b = metric[..., ::2, :], metric[..., 1::2, :]
                    scores = a @ b.transpose(-1, -2)

                    node_max, node_idx = scores.max(dim=-1)
                    edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

                    src_idx = edge_idx[..., :r, :]  # Merged Tokens
                    dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

        def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
            if self.merge_type == "sim":
                src, dst = x, x

            else:
                src, dst = x[..., ::2, :], x[..., 1::2, :]
            n, t1, c = src.shape
            src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
            dst = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))
            if mode == "mean":
                dst = dst + src
                dst = dst / 2
            elif mode == "sum":
                dst = dst + src
            return dst

        return merge, src_idx

    def merge_wavg(self, merge, x: torch.Tensor):
        """
        Applies the merge function by taking a weighted average based on token size.
        Returns the merged tensor and the new token sizes.
        """
        mode = {0: "mean", 1: "sum"}
        x = merge(x, mode=mode[self.merge_mode])

        return x


class ActorObjectRelationUpdate(nn.Module):
    """Update actor tokens from selected runtime object memory.

    The relation logits choose NULL or one object slot for each actor. The same
    selected object context is then fused into the actor token that ultimately
    feeds the action head; there is no separate object-action classifier.
    """

    def __init__(
        self,
        dim,
        relation_dim=256,
        hidden_dim=512,
        max_scale=1.0,
        null_logit_init=None,
        relation_logit_scale_init=1.0,
        learned_relation_logit_scale=False,
        normalize_relation_pointers=False,
        learned_scale=False,
        layer_scale_init=0.25,
    ):
        super().__init__()
        if null_logit_init is None:
            null_logit_init = 4.0

        self.actor_norm = nn.LayerNorm(dim)
        self.object_norm = nn.LayerNorm(dim)
        self.actor_q = nn.Linear(dim, relation_dim, bias=False)
        self.object_k = nn.Linear(dim, relation_dim, bias=False)
        self.object_v = nn.Linear(dim, dim, bias=False)

        self.null_logit = nn.Parameter(torch.tensor(float(null_logit_init)))
        relation_logit_scale_init = float(relation_logit_scale_init)
        if relation_logit_scale_init <= 0:
            raise ValueError("relation_logit_scale_init must be > 0")
        self.normalize_relation_pointers = bool(normalize_relation_pointers)
        self.learned_relation_logit_scale = bool(learned_relation_logit_scale)
        if self.learned_relation_logit_scale:
            raw_init = math.log(math.expm1(max(relation_logit_scale_init, 1.0e-6)))
            self.relation_logit_scale_raw = nn.Parameter(torch.tensor(raw_init))
            self.register_buffer(
                "relation_logit_scale",
                torch.empty(0),
                persistent=False,
            )
        else:
            self.relation_logit_scale_raw = None
            self.register_buffer(
                "relation_logit_scale",
                torch.tensor(relation_logit_scale_init),
                persistent=False,
            )

        fusion_dim = 3 * dim
        self.out = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        gate_hidden = max(hidden_dim // 4, 64)
        self.gate = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, 1),
            nn.Sigmoid(),
        )

        self.max_scale = float(max_scale)
        if self.max_scale < 0:
            raise ValueError("max_scale must be non-negative")
        self.learned_scale = bool(learned_scale)
        self.layer_scale_init = float(layer_scale_init)
        if self.learned_scale:
            if self.max_scale <= 0:
                raise ValueError("learned relation scale requires max_scale > 0")
            if self.layer_scale_init <= 0:
                raise ValueError("layer_scale_init must be positive")
            scale_ratio = min(
                max(self.layer_scale_init / self.max_scale, 1.0e-4),
                1.0 - 1.0e-4,
            )
            self.layer_scale_raw = nn.Parameter(
                torch.tensor(math.log(scale_ratio / (1.0 - scale_ratio)))
            )
        else:
            self.layer_scale_raw = None

        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def _relation_logit_scale(self, device, dtype):
        if self.relation_logit_scale_raw is not None:
            return F.softplus(
                self.relation_logit_scale_raw.to(device=device, dtype=dtype)
            )
        return self.relation_logit_scale.to(device=device, dtype=dtype)

    def forward(
        self,
        actor_tokens,
        object_tokens,
        actor_valid,
        object_valid,
    ):
        B, A, _ = actor_tokens.shape

        q = self.actor_q(self.actor_norm(actor_tokens))
        k = self.object_k(self.object_norm(object_tokens))
        v = self.object_v(object_tokens)

        relation_logit_scale = self._relation_logit_scale(q.device, q.dtype)
        if self.normalize_relation_pointers:
            q_scores = F.normalize(q.float(), p=2, dim=-1).to(dtype=q.dtype)
            k_scores = F.normalize(k.float(), p=2, dim=-1).to(dtype=k.dtype)
            obj_scores = torch.matmul(q_scores, k_scores.transpose(1, 2))
            obj_scores = obj_scores * relation_logit_scale
        else:
            obj_scores = torch.matmul(q, k.transpose(1, 2))
            obj_scores = obj_scores / (q.shape[-1] ** 0.5)
            obj_scores = obj_scores * relation_logit_scale

        actor_valid = actor_valid.to(device=obj_scores.device, dtype=torch.bool)
        if tuple(actor_valid.shape) != tuple(actor_tokens.shape[:2]):
            raise ValueError(
                "actor_valid must have shape "
                f"{tuple(actor_tokens.shape[:2])}, got {tuple(actor_valid.shape)}"
            )
        object_valid = object_valid.to(device=obj_scores.device, dtype=torch.bool)
        obj_scores = obj_scores.masked_fill(~object_valid[:, None, :], -1.0e4)

        null_score = self.null_logit.to(
            device=actor_tokens.device,
            dtype=actor_tokens.dtype,
        ).view(1, 1, 1)
        null_score = null_score.expand(B, A, 1)

        logits = torch.cat([null_score, obj_scores], dim=-1)
        attn = torch.softmax(logits.float(), dim=-1).to(actor_tokens.dtype)

        null_prob = attn[..., 0]
        object_posterior = attn[..., 1:] * object_valid[:, None, :].to(
            dtype=attn.dtype
        )
        object_mass = object_posterior.sum(dim=-1, keepdim=True).clamp(0.0, 1.0)
        object_attn = torch.where(
            object_mass > 0,
            object_posterior / object_mass.clamp_min(1.0e-6),
            torch.zeros_like(object_posterior),
        )
        useful_mass = object_mass.squeeze(-1)

        selected_object_memory = torch.matmul(object_attn, object_tokens)
        object_context = torch.matmul(object_attn, v)
        object_context = object_context * object_mass

        update_in = torch.cat(
            [
                actor_tokens,
                object_context,
                actor_tokens * object_context,
            ],
            dim=-1,
        )
        delta = self.out(update_in)
        gate = self.gate(update_in)
        # Relation NULL means "no interacted object"; in that case the
        # object-conditioned residual should be a near no-op on actor tokens.
        update_strength = gate * object_mass
        update_strength = update_strength * actor_valid[:, :, None].to(
            dtype=update_strength.dtype
        )

        if self.layer_scale_raw is None:
            scale = actor_tokens.new_tensor(self.max_scale)
        else:
            scale = actor_tokens.new_tensor(self.max_scale) * torch.sigmoid(
                self.layer_scale_raw.to(
                    device=actor_tokens.device,
                    dtype=actor_tokens.dtype,
                )
            )
        actor_tokens = actor_tokens + scale * update_strength * delta

        aux = {
            "logits": logits,
            "object_attention": object_posterior,
            "object_attention_norm": object_attn,
            "null_prob": null_prob,
            "useful_mass": useful_mass,
            "object_context": object_context,
            "selected_object_memory": selected_object_memory,
            "relation_logit_scale": relation_logit_scale.detach(),
            "scale": scale.detach(),
            "gate": gate.detach(),
            "update_strength": update_strength.detach(),
        }
        return actor_tokens, aux


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
        actor_bbox_prior_weight=0.1,
        actor_bbox_prior_expand=1.75,
        actor_interaction_heatmaps=0,
        actor_object_prompt_tokens=0,
        num_scene_object_tokens=32,
        num_object_classes=19,
        actor_object_region_visual_tokens=0,
        actor_object_prompt_box_prior_weight=0.05,
        actor_object_prompt_box_prior_expand=1.25,
        token_selection_cls_weight=0.25,
        token_selection_actor_weight=0.25,
        token_selection_register_weight=0.0,
        token_selection_heatmap_weight=0.35,
        actor_object_relation_in_transformer=0,
        actor_object_relation_blocks="2,5,8",
        actor_object_relation_dim=256,
        actor_object_relation_hidden_dim=512,
        actor_object_relation_max_scale=1.0,
        actor_object_relation_null_logit_init=0.5,
        actor_object_relation_logit_scale_init=1.0,
        actor_object_relation_learned_logit_scale=False,
        actor_object_relation_normalize_pointers=False,
        actor_object_relation_learned_scale=False,
        actor_object_relation_layer_scale_init=0.25,
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
        self.actor_bbox_prior_weight = float(actor_bbox_prior_weight)
        self.actor_bbox_prior_expand = float(actor_bbox_prior_expand)
        if self.actor_bbox_prior_weight < 0:
            raise ValueError("actor_bbox_prior_weight must be non-negative")
        if self.actor_bbox_prior_expand <= 0:
            raise ValueError("actor_bbox_prior_expand must be positive")
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
        if self.n_object_tokens < 0:
            raise ValueError("num_scene_object_tokens must be non-negative")
        self.num_object_classes = int(num_object_classes)
        if self.num_object_classes <= 0:
            raise ValueError("num_object_classes must be positive")
        self.actor_object_region_visual_tokens = bool(actor_object_region_visual_tokens)
        if self.actor_object_prompt_tokens and not self.actor_object_region_visual_tokens:
            raise ValueError(
                "actor_object_prompt_tokens requires actor_object_region_visual_tokens. "
                "Runtime object memory must include visual patch features pooled from "
                "the object box, not only class/box metadata."
            )
        self.actor_object_prompt_box_prior_weight = float(
            actor_object_prompt_box_prior_weight
        )
        if self.actor_object_prompt_box_prior_weight < 0:
            raise ValueError("actor_object_prompt_box_prior_weight must be >= 0")
        self.actor_object_prompt_box_prior_expand = float(
            actor_object_prompt_box_prior_expand
        )
        if self.actor_object_prompt_box_prior_expand <= 0:
            raise ValueError("actor_object_prompt_box_prior_expand must be positive")
        self.token_selection_cls_weight = float(token_selection_cls_weight)
        self.token_selection_actor_weight = float(token_selection_actor_weight)
        self.token_selection_register_weight = float(token_selection_register_weight)
        self.token_selection_heatmap_weight = float(token_selection_heatmap_weight)
        self.actor_object_relation_in_transformer = bool(
            actor_object_relation_in_transformer
        )
        if self.actor_object_relation_in_transformer and not self.actor_object_prompt_tokens:
            raise ValueError(
                "actor_object_relation_in_transformer requires actor_object_prompt_tokens"
            )
        self.actor_object_relation_blocks = self._parse_relation_blocks(
            actor_object_relation_blocks
        )
        if self.actor_object_relation_in_transformer and not self.actor_object_relation_blocks:
            raise ValueError(
                "actor_object_relation_in_transformer requires at least one relation block"
            )
        self.actor_object_relation_dim = int(actor_object_relation_dim)
        self.actor_object_relation_hidden_dim = int(actor_object_relation_hidden_dim)
        self.actor_object_relation_max_scale = float(actor_object_relation_max_scale)
        self.actor_object_relation_null_logit_init = float(
            actor_object_relation_null_logit_init
        )
        self.actor_object_relation_logit_scale_init = float(
            actor_object_relation_logit_scale_init
        )
        self.actor_object_relation_learned_logit_scale = bool(
            actor_object_relation_learned_logit_scale
        )
        self.actor_object_relation_normalize_pointers = bool(
            actor_object_relation_normalize_pointers
        )
        self.actor_object_relation_learned_scale = bool(
            actor_object_relation_learned_scale
        )
        self.actor_object_relation_layer_scale_init = float(
            actor_object_relation_layer_scale_init
        )
        if self.actor_object_relation_dim <= 0:
            raise ValueError("actor_object_relation_dim must be positive")
        if self.actor_object_relation_hidden_dim <= 0:
            raise ValueError("actor_object_relation_hidden_dim must be positive")
        if self.actor_object_relation_max_scale < 0:
            raise ValueError("actor_object_relation_max_scale must be non-negative")
        if self.actor_object_relation_logit_scale_init <= 0:
            raise ValueError("actor_object_relation_logit_scale_init must be > 0")
        if self.actor_object_relation_layer_scale_init <= 0:
            raise ValueError("actor_object_relation_layer_scale_init must be positive")
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

        # Class, actor, and register tokens are protected from pruning. Runtime
        # object detections are kept as relation-only memory so they cannot
        # leak object-presence shortcuts through generic self-attention.
        self.N_KEY_TOKENS = (
            1 + self.n_actor_tokens + self.n_registers
        )
        self.semantic_token_score_weights = self._semantic_token_score_weights()

        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        else:
            # sine-cosine positional embeddings is on the way
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        self.pos_drop = nn.Dropout(p=drop_rate)
        self.depth = depth
        invalid_relation_blocks = [
            block_idx
            for block_idx in self.actor_object_relation_blocks
            if block_idx < 0 or block_idx >= self.depth
        ]
        if invalid_relation_blocks:
            raise ValueError(
                "actor_object_relation_blocks contains invalid block indices "
                f"for depth {self.depth}: {invalid_relation_blocks}"
            )
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
                    bbox_prior_weight=1.0,
                    semantic_token_score_weights=self.semantic_token_score_weights,
                    **kwargs,
                )
                for i in range(depth)
            ]
        )
        self.actor_object_relation_updates = nn.ModuleDict()
        self.actor_object_final_relation_update = None
        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        self.head_dropout = nn.Dropout(head_drop_rate)
        self.head = (
            nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=0.02)

        self.apply(self._init_weights)
        relation_modules = list(self.actor_object_relation_updates.values())
        if self.actor_object_final_relation_update is not None:
            relation_modules.append(self.actor_object_final_relation_update)
        for relation_update in relation_modules:
            nn.init.zeros_(relation_update.out[-1].weight)
            nn.init.zeros_(relation_update.out[-1].bias)

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

    def _semantic_token_score_weights(self):
        def distribute(group_weight, count):
            count = int(count)
            if count <= 0:
                return []
            return [float(group_weight) / float(count)] * count

        weights = [self.token_selection_cls_weight]
        weights.extend(distribute(self.token_selection_actor_weight, self.n_actor_tokens))
        weights.extend(distribute(self.token_selection_register_weight, self.n_registers))
        weights.extend(distribute(self.token_selection_heatmap_weight, self.n_heatmap_tokens))
        return torch.tensor(weights, dtype=torch.float32)

    def _expand_boxes(self, boxes, expand):
        boxes = boxes.clamp(0.0, 1.0)
        center = (boxes[..., :2] + boxes[..., 2:]) * 0.5
        size = (boxes[..., 2:] - boxes[..., :2]).clamp_min(1e-4)
        size = size * float(expand)
        mins = (center - size * 0.5).clamp(0.0, 1.0)
        maxs = (center + size * 0.5).clamp(0.0, 1.0)
        return torch.cat([mins, maxs], dim=-1)

    def _make_box_token_prior(self, boxes, valid, window_size, expand=1.0):
        frames, height, width = [int(v) for v in window_size]
        if height <= 0 or width <= 0:
            return None
        if boxes is None or valid is None or boxes.shape[1] == 0:
            return None

        boxes = boxes.clamp(0.0, 1.0)
        valid = valid.bool()
        center = (boxes[..., :2] + boxes[..., 2:]) * 0.5
        size = (boxes[..., 2:] - boxes[..., :2]).clamp_min(1e-4)
        expanded = size * float(expand)
        mins = (center - expanded * 0.5).clamp(0.0, 1.0)
        maxs = (center + expanded * 0.5).clamp(0.0, 1.0)

        y_centers = (
            torch.arange(height, device=boxes.device, dtype=boxes.dtype) + 0.5
        ) / height
        x_centers = (
            torch.arange(width, device=boxes.device, dtype=boxes.dtype) + 0.5
        ) / width
        grid_y, grid_x = torch.meshgrid(y_centers, x_centers, indexing="ij")
        grid_x = grid_x.reshape(1, 1, -1)
        grid_y = grid_y.reshape(1, 1, -1)

        inside = (
            (grid_x >= mins[..., 0:1])
            & (grid_x <= maxs[..., 0:1])
            & (grid_y >= mins[..., 1:2])
            & (grid_y <= maxs[..., 1:2])
            & valid.unsqueeze(-1)
        )
        box_prior = inside.to(dtype=boxes.dtype)
        spatial_prior = box_prior.max(dim=1).values
        return spatial_prior.unsqueeze(1).expand(-1, frames, -1).reshape(
            boxes.shape[0], frames * height * width
        )

    def _object_region_visual_features(
        self,
        visual_tokens,
        object_boxes,
        object_valid,
        window_size,
    ):
        frames, height, width = [int(v) for v in window_size]
        if height <= 0 or width <= 0:
            raise ValueError(f"Invalid patch grid for object ROI pooling: {window_size}")
        if visual_tokens.ndim != 3:
            raise ValueError(
                "visual_tokens must have shape [B,N,C], got "
                f"{tuple(visual_tokens.shape)}"
            )
        expected_tokens = frames * height * width
        if int(visual_tokens.shape[1]) != expected_tokens:
            raise ValueError(
                "visual token count does not match patch grid: "
                f"N={visual_tokens.shape[1]}, grid={window_size}"
            )

        dtype = visual_tokens.dtype
        device = visual_tokens.device
        boxes = object_boxes.to(device=device, dtype=dtype).clamp(0.0, 1.0)
        valid = object_valid.to(device=device, dtype=torch.bool)
        grid_x = self.object_region_grid_x.to(device=device, dtype=dtype)
        grid_y = self.object_region_grid_y.to(device=device, dtype=dtype)
        if int(grid_x.shape[-1]) != height * width:
            raise ValueError(
                "object ROI grid does not match patch grid: "
                f"buffer={grid_x.shape[-1]}, forward={height * width}"
            )

        mins = boxes[..., :2]
        maxs = boxes[..., 2:]
        sharpness = visual_tokens.new_tensor(40.0)
        left = torch.sigmoid((grid_x - mins[..., 0:1]) * sharpness)
        right = torch.sigmoid((maxs[..., 0:1] - grid_x) * sharpness)
        top = torch.sigmoid((grid_y - mins[..., 1:2]) * sharpness)
        bottom = torch.sigmoid((maxs[..., 1:2] - grid_y) * sharpness)
        spatial_weights = left * right * top * bottom
        weights = spatial_weights.unsqueeze(2).expand(
            -1,
            -1,
            frames,
            -1,
        )
        weights = weights.reshape(boxes.shape[0], boxes.shape[1], expected_tokens)
        weights = weights * valid.unsqueeze(-1).to(dtype=dtype)
        denom = weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
        pooled = torch.matmul(weights, visual_tokens) / denom
        pooled = pooled * valid.unsqueeze(-1).to(dtype=dtype)
        pooled = self.object_region_proj(self.object_region_norm(pooled))
        return pooled

    def _bbox_token_prior(
        self,
        actor_boxes,
        actor_valid,
        object_boxes,
        object_valid,
        window_size,
    ):
        priors = []
        if self.actor_bbox_prior_weight > 0:
            actor_prior = self._make_box_token_prior(
                actor_boxes,
                actor_valid,
                window_size,
                expand=self.actor_bbox_prior_expand,
            )
            if actor_prior is not None:
                priors.append(actor_prior * self.actor_bbox_prior_weight)
        if self.actor_object_prompt_box_prior_weight > 0:
            object_prior = self._make_box_token_prior(
                object_boxes,
                object_valid,
                window_size,
                expand=self.actor_object_prompt_box_prior_expand,
            )
            if object_prior is not None:
                priors.append(object_prior * self.actor_object_prompt_box_prior_weight)
        if not priors:
            return None
        return torch.stack(priors, dim=0).sum(dim=0).clamp(0.0, 1.0)

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
        bbox_token_prior = None
        token_key_padding_mask = None
        object_memory_tokens = None
        self.last_object_region_visual_norm = None

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
            prefix_key_masks.append(~valid)

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
            object_region_tokens = self._object_region_visual_features(
                x,
                object_boxes,
                object_valid,
                ws,
            )
            self.last_object_region_visual_norm = (
                object_region_tokens.detach().float().norm(dim=-1)
            )
            object_tokens = object_tokens + object_region_tokens
            object_memory_tokens = object_tokens
            prefix_tokens.append(object_tokens)
            prefix_key_masks.append(~object_valid)
        if self.n_heatmap_out_channels > 0:
            prefix_tokens.append(self.heatmap_tokens.expand(B, -1, -1))
            prefix_key_masks.append(
                torch.zeros(B, self.n_heatmap_tokens, dtype=torch.bool, device=x.device)
            )
        if self.n_actor_tokens > 0:
            bbox_token_prior = self._bbox_token_prior(
                boxes,
                valid,
                object_boxes if self.actor_object_prompt_tokens else None,
                object_valid if self.actor_object_prompt_tokens else None,
                ws,
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
        self.last_actor_object_relation_aux = {}
        self.last_token_selection_diagnostics = None
        # keep the global indexes of non-keyframe tokens during pruning
        idx = torch.arange(0, N, device=x.device).unsqueeze(0).repeat(B, 1)
        for i in range(self.depth):
            blk = self.blocks[i]
            x, idx, token_key_padding_mask = blk(
                x,
                idx,
                ws,
                bbox_token_prior=bbox_token_prior,
                key_padding_mask=token_key_padding_mask,
            )

        selected_mask = torch.zeros(B, N, dtype=torch.bool, device=x.device)
        if idx is not None and idx.numel() > 0:
            selected_idx = idx.clamp(0, N - 1).long()
            selected_mask = selected_mask.scatter(
                1,
                selected_idx,
                torch.ones_like(selected_idx, dtype=torch.bool, device=x.device),
            )
        self.last_token_selection_diagnostics = {
            "selected_indices": idx.detach() if idx is not None else None,
            "selected_mask": selected_mask.detach(),
            "num_visual_tokens": int(N),
            "window_size": tuple(int(v) for v in ws),
        }

        if self.n_actor_tokens > 0:
            x_actor = x[:, actor_start:actor_end, :]
        x_object = None
        if self.actor_object_prompt_tokens:
            x_object = object_memory_tokens
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
