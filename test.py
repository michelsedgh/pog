# %%
import os
from argparse import ArgumentParser
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers.wandb import WandbLogger

from datamodule.base_datamod import BaseDataModule

from models.poguise import POGUISE

from modules.heatmap_module import HeatmapModule

import torch
import argparse
import pandas as pd
import numpy as np
from collections import defaultdict
import json

# from datasets.ntu120 import NTUDataset
from datasets.toyotasm import ToyotaSMDataset
from datasets.driveact import DriveActDataset

seed_everything(42)

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"


def to_normalized_float_tensor(vid):
    return vid.to(torch.float32) / 255


def main(hparams, network, dataset):
    if hparams.gpus == -1:
        hparams.gpus = torch.cuda.device_count()
    num_cpu = len(os.sched_getaffinity(0))
    # if using slurm do not modify num_workers
    if "SLURM_JOB_ID" in os.environ:
        num_cpu = hparams.num_workers
    else:
        num_cpu = 8 if num_cpu < 16 else num_cpu
        hparams.num_workers = num_cpu // hparams.gpus
    print(hparams)

    checkpoint = torch.load(
        hparams.model_file,
        map_location=lambda storage, loc: storage,
        weights_only=False,
    )
    file_hparams = dict(checkpoint["datamodule_hyper_parameters"])
    # convert namespace to dict
    hparams = vars(hparams)
    # overwrite file hparams with command line hparams Namespace
    file_hparams.update(hparams)
    hparams.update(file_hparams)
    # convert back to namespace
    hparams = argparse.Namespace(**hparams)
    # add mode = 'test' to hparams
    hparams.mode = "test"
    print(hparams)

    model = HeatmapModule.load_from_checkpoint(
        hparams.model_file, model=network, **vars(hparams), strict=True
    )

    project_folder = hparams.project_folder
    checkpoint_path = os.path.join(
        "./checkpoints/", hparams.model_name
    )  # "/opt/ml/checkpoints/"

    wandb_logger = WandbLogger(
        name=hparams.model_name,
        project=project_folder,
    )
    if os.getenv("NODE_RANK", 0) == 0 and os.getenv("LOCAL_RANK", 0) == 0:
        run_id = wandb_logger.experiment.id
        hparams.wandb_id = run_id
        # save
        gpus = torch.cuda.device_count()
        if gpus > 1:
            with open(checkpoint_path + "/wandb_id.json", "w") as json_file:
                json.dump(
                    {
                        "wandb_id": run_id,
                        "model_name": hparams.model_name,
                        "project_folder": project_folder,
                    },
                    json_file,
                )
        checkpoint_path = os.path.join(checkpoint_path, run_id)

    else:
        with open(checkpoint_path + "/wandb_id.json", "r") as json_file:
            data = json.load(json_file)
            run_id = data["wandb_id"]
            model_name = data["model_name"]
            project_folder = data["project_folder"]
    wandb_logger.watch(model, log="all")
    # wandb_logger.experiment.use_artifact(hparams.dataset_artifact + ":latest")

    trainer = Trainer(
        accelerator="gpu",
        devices=hparams.gpus,
        max_epochs=1,
        logger=wandb_logger,
        profiler="simple",
        # deterministic=False,
        benchmark=True,
        precision=hparams.precision,
        default_root_dir=checkpoint_path,
        enable_checkpointing=True,
        strategy="ddp",
        num_sanity_val_steps=0,
    )
    datamodule_params = vars(hparams).copy()
    if "dataset" in datamodule_params:
        del datamodule_params["dataset"]
    datamodule = BaseDataModule(
        dataset,
        **datamodule_params,
    )
    datamodule.setup("test")
    print(f"Number of GPUs used: {trainer.device_ids}")
    model.eval()
    model.freeze()
    trainer.test(model, datamodule=datamodule)
    file_path = os.path.join(checkpoint_path, run_id, "test_results.csv")
    df = pd.read_csv(file_path)
    # group by id

    final_top1 = []
    final_top5 = []
    per_class_top1 = defaultdict(
        lambda: [0, 0]
    )  # [correct predictions, total predictions]
    for id, group in df.groupby("id"):
        feats = group["preds"].values
        feats = np.array(feats)
        # to float
        feats = np.array(
            [
                np.fromstring(f.split("[")[1].split("]")[0], dtype=float, sep=",")
                for f in feats
            ]
        )
        feats = np.mean(feats, axis=0)
        pred = np.argmax(feats)
        labels = group["labels"].values
        label = labels[0]
        top1 = (int(pred) == int(label)) * 1.0
        top5 = (int(label) in np.argsort(-feats)[:5]) * 1.0
        final_top1.append(top1)
        final_top5.append(top5)
        per_class_top1[label][1] += 1  # increment total predictions for this class
        if top1:
            per_class_top1[label][
                0
            ] += 1  # increment correct predictions for this class

    # calculate per class accuracy
    per_class_accuracy = {k: v[0] / v[1] for k, v in per_class_top1.items()}

    final_top1 = np.mean(np.array(final_top1))
    final_top5 = np.mean(np.array(final_top5))
    final_top1.shape
    # calculate mean per class accuracy
    per_class_accuracy = np.mean(list(per_class_accuracy.values()))
    tnc = hparams.test_num_crop
    tns = hparams.test_num_segment
    # update wandb
    wandb_logger.experiment.log(
        {
            f"c_{tnc}_s_{tns}_test_top1": float(final_top1),
            f"c_{tnc}_s_{tns}_test_top5": float(final_top5),
            f"c_{tnc}_s_{tns}_per_class_accuracy": float(per_class_accuracy),
        }
    )


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    # trainer args
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--precision", type=str, default="bf16-mixed")
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--test_num_crop", default=1, type=int)
    parser.add_argument("--test_num_segment", default=1, type=int)
    parser.add_argument("--n_frames_stride", default=-3, type=int)
    parser.add_argument("--dataset", type=str)

    parser.add_argument(
        "--model_file",
        type=str,
        required=True,
        help="Path to the checkpoint file",
    )

    parser.add_argument("--num_workers", type=int, default=16)

    hparams, _ = parser.parse_known_args()
    network = POGUISE
    if hparams.dataset == "driveact":
        dataset = DriveActDataset
    elif hparams.dataset == "toyotasm":
        dataset = ToyotaSMDataset
    main(hparams, network, dataset)
