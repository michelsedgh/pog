#!/usr/bin/env python3
"""Smoke-test ActorObjectDecoder identity and object-mass gating."""

import argparse

import torch

from models.actor_object_decoder import ActorObjectDecoder


def sorted_boxes(batch_size, count, device):
    boxes = torch.rand(batch_size, count, 4, device=device)
    mins = torch.minimum(boxes[..., :2], boxes[..., 2:])
    maxs = torch.maximum(boxes[..., :2], boxes[..., 2:])
    return torch.cat([mins, maxs], dim=-1)


def run(device):
    torch.manual_seed(0)
    batch_size, actors, objects, dim = 2, 3, 5, 32
    decoder = ActorObjectDecoder(
        feature_dim=dim,
        hidden_dim=64,
        dropout=0.0,
        interaction_heatmap_size=56,
        init_update_gate=0.5,
        init_ffn_gate=0.5,
    ).to(device)
    decoder.eval()

    actor_tokens = torch.randn(batch_size, actors, dim, device=device)
    object_tokens_a = torch.randn(batch_size, objects, dim, device=device)
    object_tokens_b = torch.randn(batch_size, objects, dim, device=device)
    actor_boxes = sorted_boxes(batch_size, actors, device)
    object_boxes_a = sorted_boxes(batch_size, objects, device)
    object_boxes_b = sorted_boxes(batch_size, objects, device)
    pair_visual_a = torch.randn(batch_size, actors, objects, dim, device=device)
    pair_visual_b = torch.randn(batch_size, actors, objects, dim, device=device)
    all_invalid = torch.zeros(batch_size, objects, dtype=torch.bool, device=device)
    all_valid = torch.ones(batch_size, objects, dtype=torch.bool, device=device)

    with torch.no_grad():
        off_a, sel_a, heat_a = decoder(
            actor_tokens,
            object_tokens_a,
            actor_boxes,
            object_boxes_a,
            all_invalid,
            pair_visual_a,
        )
        off_b, sel_b, heat_b = decoder(
            actor_tokens,
            object_tokens_b,
            actor_boxes,
            object_boxes_b,
            all_invalid,
            pair_visual_b,
        )
        on, sel_on, heat_on = decoder(
            actor_tokens,
            object_tokens_a,
            actor_boxes,
            object_boxes_a,
            all_valid,
            pair_visual_a,
        )

    off_probs = torch.softmax(sel_a.float(), dim=-1)
    max_off_delta = (off_a - actor_tokens).abs().max().item()
    max_off_payload_delta = (off_a - off_b).abs().max().item()
    max_off_real_prob = off_probs[..., :-1].max().item()
    min_off_none_prob = off_probs[..., -1].min().item()
    max_off_heat = heat_a.abs().max().item()
    max_on_delta = (on - actor_tokens).abs().max().item()
    max_on_heat = heat_on.abs().max().item()

    print(f"max_off_delta={max_off_delta:.6e}")
    print(f"max_off_payload_delta={max_off_payload_delta:.6e}")
    print(f"max_off_real_prob={max_off_real_prob:.6e}")
    print(f"min_off_none_prob={min_off_none_prob:.6e}")
    print(f"max_off_heat={max_off_heat:.6e}")
    print(f"max_on_delta_initial={max_on_delta:.6e}")
    print(f"max_on_heat={max_on_heat:.6e}")

    assert max_off_delta < 1e-6
    assert max_off_payload_delta < 1e-6
    assert max_off_real_prob < 1e-6
    assert min_off_none_prob > 1.0 - 1e-6
    assert max_off_heat < 1e-6
    assert max_on_delta < 1e-6
    assert max_on_heat > 0.0
    assert torch.isfinite(sel_on).all()
    print("ActorObjectDecoder identity smoke passed.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    run(torch.device(args.device))


if __name__ == "__main__":
    main()
