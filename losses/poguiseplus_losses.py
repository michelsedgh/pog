import torch


def heatmap_frobenius_loss(pred_heatmap, target_heatmap, valid=None, mask=None):
    if pred_heatmap.shape != target_heatmap.shape:
        raise RuntimeError(
            "heatmap target/prediction shape mismatch: "
            f"{tuple(target_heatmap.shape)} vs {tuple(pred_heatmap.shape)}"
        )

    sq_error = (pred_heatmap - target_heatmap).float().pow(2)
    active = torch.ones(
        pred_heatmap.shape[0],
        dtype=torch.bool,
        device=pred_heatmap.device,
    )

    if mask is not None:
        mask = mask.to(device=pred_heatmap.device, dtype=sq_error.dtype)
        if mask.shape != sq_error.shape:
            raise RuntimeError(
                "heatmap mask shape mismatch: "
                f"{tuple(mask.shape)} vs {tuple(sq_error.shape)}"
            )
        sq_error = sq_error * mask
        active = active & (mask.flatten(1).sum(dim=1) > 0)

    if valid is not None:
        valid = valid.to(device=pred_heatmap.device, dtype=torch.bool)
        if valid.shape != pred_heatmap.shape[:2]:
            raise RuntimeError(
                "heatmap valid mask shape mismatch: "
                f"{tuple(valid.shape)} vs {tuple(pred_heatmap.shape[:2])}"
            )
        sq_error = sq_error * valid[:, :, None, None].to(dtype=sq_error.dtype)
        active = active & valid.any(dim=1)

    if not active.any():
        return pred_heatmap.sum() * 0.0

    per_sample = sq_error.flatten(1).sum(dim=1)
    return per_sample[active].mean()


def log_heatmap_frobenius_loss(
    pred_heatmap,
    target_heatmap,
    valid=None,
    mask=None,
    eps=1e-6,
):
    loss = heatmap_frobenius_loss(
        pred_heatmap,
        target_heatmap,
        valid=valid,
        mask=mask,
    )
    return torch.log(loss + float(eps))
