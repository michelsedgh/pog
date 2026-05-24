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
            deconv_layers = [nn.AdaptiveAvgPool2d((14, 14))]
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

    def forward_part1(self, x, size=None):
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
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        if self.keep_rate >= 1:
            # use flash attention
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
            x = x.transpose(1, 2).reshape(B, N, -1)
            attn = None
        else:
            attn = (q * self.scale) @ k.transpose(-2, -1)
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

    def forward(self, x, last_idx=None, ws=None, size=None):
        B, N, C = x.shape
        if self.keep_rate < 1:
            x, attn, feature = self.forward_part1(x, size)
        else:
            x, attn, _ = self.forward_part1(x, size)
            return x, last_idx, attn

        # get top-k tokens and the corresponding indexes
        if self.keep_rate < 1:
            num_s_tokens = self.n_heatmap_tokens + self.n_key_tokens
            num_keep_tokens = math.ceil(self.keep_rate * (N - num_s_tokens))
            if num_keep_tokens > 0:
                # class token query enhancement
                attn_topk = attn.clone()
                attn_topk[:, :, 0] *= self.enhanced_weight_class
                if self.n_heatmap_tokens:
                    # heatmap token query enhancement
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
                        attn_topk[:, :, : self.n_heatmap_tokens + self.n_key_tokens]
                        .sum(dim=-2)
                        .mean(dim=1)
                    )
                elif self.topk_type == 2:
                    # only class token
                    attn_topk = attn_topk[:, :, 0].mean(dim=1)
                attn_topk = attn_topk[
                    :, num_s_tokens:
                ]  # remove class token and heatmap tokens
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

    def forward_KTPpart1(self, x, last_idx, window_size):
        # check if self._size_attn shape matches with x
        if self.gamma_1 is None:
            tmp, idx, attn = self.attn(
                self.norm1(x), last_idx, window_size, self._size_attn
            )
            return self.drop_path(tmp), idx, attn
        else:
            attn, idx = self.attn(self.norm1(x), last_idx, window_size, self._size_attn)
            return self.drop_path(self.gamma_1 * attn), idx

    def forward_KTP_part2(self, x):
        if self.gamma_2 is None:
            return self.drop_path(self.mlp(self.norm2(x)))
        else:
            return self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))

    def forward(self, x, last_idx, window_size):
        # attn
        tmp, idx, attn = self.forward_KTPpart1(x, last_idx, window_size)
        x = x + tmp

        # class token heatmap -centric token pruning
        B, N, C = x.shape
        num_s_tokens = self.n_heatmap_tokens + self.n_key_tokens  # 1 for class token
        if self.keep_rate < 1:
            x_key = x[:, :num_s_tokens]
            x_nonkey_keep = x[:, num_s_tokens:]
            index_nonkey_keep = idx.unsqueeze(-1).expand(-1, -1, C)  # (B, N_keep, C_e)
            x_nonkey_keep = torch.gather(
                x_nonkey_keep, dim=1, index=index_nonkey_keep
            )  # (B, N_keep, C_e)

            if self.keep_rate_merge < 1:
                idx_nonselected = (
                    torch.arange(N - num_s_tokens, device=x.device)
                    .unsqueeze(0)
                    .repeat(B, 1)
                )
                # Get a boolean mask where True indicates the indices of non-selected tokens
                idx_non_list = []
                for i in range(B):
                    mask = torch.isin(
                        idx_nonselected[i], idx[i], invert=True, assume_unique=True
                    )
                    # Get the indices of the non-selected tokens
                    idx_non_list.append(idx_nonselected[i][mask])
                idx_nonselected = torch.stack(idx_non_list)
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
        x = x + self.forward_KTP_part2(x)

        return x, idx

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
                # make diagonal 0
                scores = scores - torch.diag_embed(
                    torch.diagonal(scores, dim1=-2, dim2=-1)
                )
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
        self.HW_OUT_CONV = (hw_out_conv, hw_out_conv)
        self.n_heatmap_tokens = self.HW_OUT_CONV[0] * self.HW_OUT_CONV[1]
        self.n_landmarks = n_landmarks
        if self.n_landmarks == 0:
            self.n_heatmap_tokens = 0
        self.mode = mode

        # Tokens before heatmap/video tokens are protected from pruning.
        self.N_KEY_TOKENS = 1 + self.n_actor_tokens + self.n_registers

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
            self.actor_tokens = nn.Parameter(
                torch.randn(1, self.n_actor_tokens, self.num_features)
            )
            self.bbox_mlp = nn.Sequential(
                nn.Linear(4, self.num_features),
                nn.GELU(),
                nn.Linear(self.num_features, self.num_features),
            )
        if self.n_landmarks > 0:
            self.heatmap_tokens = nn.Parameter(
                torch.randn(1, self.HW_OUT_CONV[0] * self.HW_OUT_CONV[1], embed_dim)
            )
            self.heatmap_head = HeatmapHead(
                in_channels=embed_dim,
                in_size=self.HW_OUT_CONV[0],
                out_channels=self.n_landmarks,
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

    def forward(self, x, boxes=None, valid=None):
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
        if self.n_landmarks > 0:
            # append heatmap tokens
            x = torch.cat([self.heatmap_tokens.expand(B, -1, -1), x], dim=1)
        if self.n_registers > 0:
            x = torch.cat([self.register_tokens.expand(B, -1, -1), x], dim=1)
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
            actor_tokens = self.actor_tokens.expand(B, -1, -1) + self.bbox_mlp(boxes)
            if valid is None:
                raise ValueError("valid is required when actor_prompt is enabled")
            if valid.shape != boxes.shape[:2]:
                raise ValueError(
                    "valid must have shape "
                    f"[B,{self.n_actor_tokens}], got {tuple(valid.shape)}"
                )
            x = torch.cat([actor_tokens, x], dim=1)

        # append class token to the beginning of the sequence
        x = torch.cat([self.class_token.expand(B, -1, -1), x], dim=1)
        # keep the global indexes of non-keyframe tokens during pruning
        idx = torch.arange(0, N, device=x.device).unsqueeze(0).repeat(B, 1)
        for i in range(self.depth):
            blk = self.blocks[i]
            x, idx = blk(x, idx, ws)

        if self.n_actor_tokens > 0:
            x_actor = x[:, 1 : 1 + self.n_actor_tokens, :]
        if self.n_landmarks > 0:
            x_heatmap = x[
                :,
                self.N_KEY_TOKENS : self.heatmap_tokens.shape[1] + self.N_KEY_TOKENS,
                :,
            ]
        x_class = x[:, 0, :]
        if self.fc_norm is not None:
            x_class = self.fc_norm(x_class)
            if self.n_actor_tokens > 0:
                x_actor = self.fc_norm(x_actor)
        else:
            x_class = self.norm(x_class)
            if self.n_actor_tokens > 0:
                x_actor = self.norm(x_actor)
        x_class = self.head_dropout(x_class)
        x_class = self.head(x_class)
        if self.n_actor_tokens > 0:
            x_actor = self.head_dropout(x_actor)

        if self.n_landmarks == 0:
            if self.n_actor_tokens > 0:
                return x_class, x_actor
            return x_class
        x_heatmap = x_heatmap.reshape(B, *self.HW_OUT_CONV, -1).permute(0, 3, 1, 2)
        x_heatmap = tuple([0, x_heatmap])
        x_heatmap = self.heatmap_head(x_heatmap)
        if self.n_actor_tokens > 0:
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
