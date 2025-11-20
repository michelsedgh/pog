from typing import Optional

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# https://mmpretrain.readthedocs.io/en/latest/get_started.html
class KeypointMSELoss(nn.Module):
    """MSE loss for heatmaps.

    Args:
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
            Defaults to ``False``
        skip_empty_channel (bool): If ``True``, heatmap channels with no
            non-zero value (which means no visible ground-truth keypoint
            in the image) will not be used to calculate the loss. Defaults to
            ``False``
        loss_weight (float): Weight of the loss. Defaults to 1.0
    """

    def __init__(
        self,
        use_target_weight: bool = False,
        skip_empty_channel: bool = False,
        loss_weight: float = 1.0,
    ):
        super().__init__()
        self.use_target_weight = use_target_weight
        self.skip_empty_channel = skip_empty_channel
        self.loss_weight = loss_weight

    def forward(
        self,
        output: Tensor,
        target: Tensor,
        target_weights: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward function of loss.

        Note:
            - batch_size: B
            - num_keypoints: K
            - heatmaps height: H
            - heatmaps weight: W

        Args:
            output (Tensor): The output heatmaps with shape [B, K, H, W]
            target (Tensor): The target heatmaps with shape [B, K, H, W]
            target_weights (Tensor, optional): The target weights of differet
                keypoints, with shape [B, K] (keypoint-wise) or
                [B, K, H, W] (pixel-wise).
            mask (Tensor, optional): The masks of valid heatmap pixels in
                shape [B, K, H, W] or [B, 1, H, W]. If ``None``, no mask will
                be applied. Defaults to ``None``

        Returns:
            Tensor: The calculated loss.
        """

        _mask = self._get_mask(target, target_weights, mask)
        if _mask is None:
            loss = F.mse_loss(output, target)
        else:
            _loss = F.mse_loss(output, target, reduction="none")
            loss = (_loss * _mask).mean()

        return loss * self.loss_weight

    def _get_mask(
        self, target: Tensor, target_weights: Optional[Tensor], mask: Optional[Tensor]
    ) -> Optional[Tensor]:
        """Generate the heatmap mask w.r.t. the given mask, target weight and
        `skip_empty_channel` setting.

        Returns:
            Tensor: The mask in shape (B, K, *) or ``None`` if no mask is
            needed.
        """
        # Given spatial mask
        if mask is not None:
            # check mask has matching type with target
            assert mask.ndim == target.ndim and all(
                d_m == d_t or d_m == 1 for d_m, d_t in zip(mask.shape, target.shape)
            ), (
                f"mask and target have mismatched shapes {mask.shape} v.s."
                f"{target.shape}"
            )

        # Mask by target weights (keypoint-wise mask)
        if target_weights is not None:
            # check target weight has matching shape with target
            assert (
                target_weights.ndim in (2, 4)
                and target_weights.shape == target.shape[: target_weights.ndim]
            ), (
                "target_weights and target have mismatched shapes "
                f"{target_weights.shape} v.s. {target.shape}"
            )

            ndim_pad = target.ndim - target_weights.ndim
            _mask = target_weights.view(target_weights.shape + (1,) * ndim_pad)

            if mask is None:
                mask = _mask
            else:
                mask = mask * _mask

        # Mask by ``skip_empty_channel``
        if self.skip_empty_channel:
            _mask = (target != 0).flatten(2).any(dim=2)
            ndim_pad = target.ndim - _mask.ndim
            _mask = _mask.view(_mask.shape + (1,) * ndim_pad)

            if mask is None:
                mask = _mask
            else:
                mask = mask * _mask

        return mask
