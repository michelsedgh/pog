import torch
import torch.nn.functional as F


def positive_object_selection_loss(selection_logits, positive_mask, valid):
    if selection_logits is None or not valid.any():
        return None
    log_probs = F.log_softmax(selection_logits.float(), dim=-1)
    valid_log_probs = log_probs[valid]
    valid_positive_mask = positive_mask[valid]
    mask_value = torch.finfo(valid_log_probs.dtype).min
    selected_log_prob = torch.logsumexp(
        valid_log_probs.masked_fill(~valid_positive_mask, mask_value),
        dim=-1,
    )
    return -selected_log_prob.mean()


def interaction_heatmap_loss(pred_heatmap, target_heatmap, valid):
    if pred_heatmap is None or not valid.any():
        return None
    if target_heatmap.shape[-2:] != pred_heatmap.shape[-2:]:
        raise RuntimeError(
            "interaction_heatmap target/prediction size mismatch: "
            f"{target_heatmap.shape[-2:]} vs {pred_heatmap.shape[-2:]}"
        )
    return F.mse_loss(pred_heatmap[valid], target_heatmap[valid])
