import torch.nn.functional as F


def interaction_heatmap_loss(pred_heatmap, target_heatmap, valid):
    if pred_heatmap is None:
        return None
    if pred_heatmap.shape != target_heatmap.shape:
        raise RuntimeError(
            "interaction_heatmap target/prediction shape mismatch: "
            f"{tuple(target_heatmap.shape)} vs {tuple(pred_heatmap.shape)}"
        )
    if target_heatmap.shape[-2:] != pred_heatmap.shape[-2:]:
        raise RuntimeError(
            "interaction_heatmap target/prediction size mismatch: "
            f"{target_heatmap.shape[-2:]} vs {pred_heatmap.shape[-2:]}"
        )
    valid = valid.to(device=pred_heatmap.device).bool()
    if valid.shape != pred_heatmap.shape[:-2]:
        raise RuntimeError(
            "interaction_heatmap valid mask shape mismatch: "
            f"{tuple(valid.shape)} vs {tuple(pred_heatmap.shape[:-2])}"
        )
    if not valid.any():
        return None
    return F.mse_loss(pred_heatmap[valid], target_heatmap[valid])
