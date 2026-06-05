import gc

import torch

from models.poguise import POGUISE
from train import _load_checkpoint


def load_actor_model(checkpoint_path, device, return_metadata=False, dtype=torch.float32):
    checkpoint = _load_checkpoint(checkpoint_path)
    hparams = {}
    hparams.update(checkpoint.get("hyper_parameters", {}))
    hparams.update(checkpoint.get("datamodule_hyper_parameters", {}))
    if not hparams:
        raise RuntimeError(f"No hyperparameters found in checkpoint: {checkpoint_path}")
    if not hparams.get("actor_prompt", 0):
        raise RuntimeError("Checkpoint is not an actor-prompt checkpoint.")

    hparams["pretrained"] = "none"
    hparams["mode"] = "test"
    hparams["ret_feat"] = 0

    model = POGUISE(**hparams)
    state_dict = {
        key.removeprefix("model."): value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith("model.")
    }
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint load mismatch. Missing={missing}, unexpected={unexpected}"
        )

    metadata = {
        "epoch": checkpoint.get("epoch"),
        "global_step": checkpoint.get("global_step"),
    }
    del checkpoint
    del state_dict
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if device.type == "cuda" and dtype != torch.float32:
        model.to(dtype=dtype)
        model.to(device=device)
    else:
        model.to(device=device, dtype=dtype)
    model.eval()
    if return_metadata:
        return model, hparams, metadata
    return model, hparams
