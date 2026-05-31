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


def feature_update_l2(feature_update, valid):
    if feature_update is None or not valid.any():
        return None
    return feature_update[valid].float().square().mean()


def positive_erased_margin_loss(
    normal_logits,
    labels,
    object_mask,
    margin,
    erased_logits=None,
    shuffled_logits=None,
):
    if not object_mask.any():
        return None
    normal_object = normal_logits[object_mask]
    labels_object = labels[object_mask]
    normal_true = normal_object.gather(1, labels_object[:, None]).squeeze(1)

    margin_losses = []
    for comparison_logits in (erased_logits, shuffled_logits):
        if comparison_logits is None:
            continue
        comparison_true = comparison_logits[object_mask].gather(
            1,
            labels_object[:, None],
        ).squeeze(1)
        margin_losses.append(F.relu(margin - (normal_true - comparison_true)))
    if not margin_losses:
        return None
    return torch.cat(margin_losses).mean()


def objectless_consistency_loss(normal_logits, off_logits, none_mask):
    if off_logits is None or not none_mask.any():
        return None
    return F.kl_div(
        F.log_softmax(normal_logits[none_mask], dim=-1),
        F.softmax(off_logits[none_mask], dim=-1),
        reduction="batchmean",
    )
