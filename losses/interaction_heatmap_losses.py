import torch.nn.functional as F


def interaction_heatmap_loss(pred_heatmap, target_heatmap, valid):
    if pred_heatmap is None or not valid.any():
        return None
    if target_heatmap.shape[-2:] != pred_heatmap.shape[-2:]:
        raise RuntimeError(
            "interaction_heatmap target/prediction size mismatch: "
            f"{target_heatmap.shape[-2:]} vs {pred_heatmap.shape[-2:]}"
        )
    return F.mse_loss(pred_heatmap[valid], target_heatmap[valid])
