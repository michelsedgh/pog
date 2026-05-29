import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import default_collate
from datamodule.mixup import Mixup
import torch
import torch
import numpy as np
import json
import os


def _toyota_specialist_action_indices():
    from datasets.object_vocab import OBJECT_SPECIALIST_GROUPS
    from datasets.toyotasm import CS_DICT

    action_indices = set()
    for group in OBJECT_SPECIALIST_GROUPS.values():
        for action_name in group["actions"]:
            if action_name in CS_DICT:
                action_indices.add(int(CS_DICT[action_name]) - 1)
    if not action_indices:
        raise ValueError("specialist_sampler found no Toyota specialist actions")
    return torch.tensor(sorted(action_indices), dtype=torch.long)


# collate_fn for THUMOS14Dataset
def thumos_collate_fn(batch):
    """
    Collate function for THUMOS14Dataset.
    Args:
        batch (list): A list of tuples, where each tuple is (frames, results_dict)
                      as returned by THUMOS14Dataset.__getitem__.
                      - frames: torch.Tensor of shape (T, C, H, W)
                      - results_dict: a dictionary of metadata and ground truth.
    Returns:
        collated_frames (torch.Tensor): Batched frames, shape (B, T, C, H, W).
        collated_results (dict): A dictionary where each key corresponds to a key
                                 in results_dict, and values are batched:
                                 - Tensors for stackable items (e.g., masks, frame_inds).
                                 - Lists of Tensors/Nones for variable-length items (e.g., gt_segments).
                                 - Tensors for scalar metadata if numeric.
                                 - Lists for string metadata or other non-stackable items.
    """

    frames_list = [item[0] for item in batch]
    results_list = [item[1] for item in batch]

    # Collate frames: stack them along a new batch dimension
    # Input frames are (T, C, H, W), output will be (B, T, C, H, W)
    collated_frames = torch.stack(frames_list, 0)

    # Collate results dictionaries
    collated_results = {}
    if not results_list:
        return collated_frames, collated_results

    # Get all unique keys from all dictionaries in the batch
    # This handles cases where some optional keys (like GTs) might be missing.
    all_keys = set()
    for r_dict in results_list:
        all_keys.update(r_dict.keys())

    for key in all_keys:
        # Collect all values for the current key from each sample in the batch
        # If a key is missing in a particular sample's dict, .get() will return None
        values = [r_dict.get(key) for r_dict in results_list]

        # Special handling for ground truth segments and labels
        # These are typically lists of tensors because their count varies per sample.
        if key in ["gt_segments", "gt_labels"]:
            # Convert np.ndarray to torch.Tensor.
            # The result will be a list, e.g., [tensor_sample1, tensor_sample2, None, tensor_sample4, ...]
            # where None indicates missing ground truth for that sample.
            collated_results[key] = [
                torch.from_numpy(v) if isinstance(v, np.ndarray) else v for v in values
            ]
            continue  # Move to the next key

        # Determine the type from the first non-None value in the list
        # This helps decide how to collate this key's values.
        first_non_none_val = next((v for v in values if v is not None), None)

        if first_non_none_val is None:  # All values for this key are None
            collated_results[key] = values  # Store as a list of Nones
            continue

        # --- Batching logic for different types ---

        if isinstance(first_non_none_val, torch.Tensor):
            # For keys where values are torch.Tensors (e.g., 'masks')
            # Assuming if one is a Tensor, all non-None are Tensors and stackable.
            # The dataset __getitem__ ensures 'masks' are always present and have consistent shape.
            if any(v is None for v in values):  # Should not happen for 'masks'
                # If Nones are possible for a tensor field, they can't be stacked directly.
                # Default to storing as a list in such cases.
                collated_results[key] = values
            else:
                try:
                    collated_results[key] = torch.stack(values)
                except RuntimeError:  # If shapes mismatch unexpectedly
                    collated_results[key] = values

        elif isinstance(first_non_none_val, np.ndarray):
            # For keys where values are np.ndarrays (e.g., 'frame_inds')
            # Assuming if one is np.ndarray, all non-None are np.ndarray and stackable.
            # The dataset __getitem__ ensures 'frame_inds' are always present and have consistent shape.
            if any(v is None for v in values):  # Should not happen for 'frame_inds'
                collated_results[key] = [
                    torch.from_numpy(v) if isinstance(v, np.ndarray) else v
                    for v in values
                ]
            else:
                try:
                    tensor_list = [torch.from_numpy(v) for v in values]
                    collated_results[key] = torch.stack(tensor_list)
                except RuntimeError:  # If shapes mismatch unexpectedly
                    collated_results[key] = [
                        torch.from_numpy(v) if isinstance(v, np.ndarray) else v
                        for v in values
                    ]

        elif isinstance(first_non_none_val, (int, float, bool)):
            # For scalar numeric types.
            # If Nones are present, torch.tensor might error or convert them undesirably.
            if any(v is None for v in values):
                collated_results[key] = values
            else:
                try:
                    # Attempt to convert the list of scalars to a torch.Tensor.
                    # torch.tensor is quite flexible with mixed int/float.
                    collated_results[key] = torch.tensor(values)
                except (
                    Exception
                ):  # Fallback if conversion fails (e.g. mixed with strings)
                    collated_results[key] = values

        elif isinstance(first_non_none_val, str):
            # For string types (e.g., 'video_name'). Store as a list of strings.
            # This also correctly handles Nones if a string field is optional.
            collated_results[key] = values

        else:
            # Fallback for any other types or complex unhandled structures.
            collated_results[key] = values

    return collated_frames, collated_results


class BaseDataModule(pl.LightningDataModule):
    def __init__(self, dataset, **kwargs):
        super().__init__()
        # self.__dict__.update(kwargs)
        self.save_hyperparameters()
        self.dataset = dataset

    def prepare_data(self):
        pass

    def _one_hot(self, x, num_classes, smoothing=0.0):
        off_value = smoothing / num_classes
        on_value = 1.0 - smoothing + off_value
        x = x.long().view(-1, 1)
        return torch.full(
            (x.size()[0], num_classes), off_value, device=x.device
        ).scatter_(1, x, on_value)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.train_dataset = self.dataset(
                set_type="train",
                **self.hparams,
                # task_type=self.task_type,
                # modal=self.modal,
                # fold=self.fold,
                # transforms=self.transforms,
                # lazy_load=self.lazy_load,
            )
            self.train_dataset.setup()
        limit_val_batches = getattr(self.hparams, "limit_val_batches", None)
        val_disabled = limit_val_batches is not None and float(limit_val_batches) == 0.0
        if stage == "validate" or (stage == "fit" and not val_disabled):
            self.val_dataset = self.dataset(
                set_type="val",
                **self.hparams,
            )
            self.val_dataset.setup()

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.test_dataset = self.dataset(
                set_type="test",
                **self.hparams,
            )
            self.test_dataset.setup()

        if stage == "predict":
            pass

    def _loader_worker_kwargs(self):
        num_workers = int(getattr(self.hparams, "num_workers", 0) or 0)
        kwargs = {"num_workers": num_workers, "pin_memory": True}
        if num_workers > 0:
            kwargs["persistent_workers"] = bool(
                getattr(self.hparams, "persistent_workers", 1)
            )
            prefetch_factor = getattr(self.hparams, "prefetch_factor", None)
            if prefetch_factor is not None:
                kwargs["prefetch_factor"] = int(prefetch_factor)
        return kwargs

    def _class_balanced_train_sampler(self):
        specialist_sampler = self._specialist_train_sampler()
        if specialist_sampler is not None:
            return specialist_sampler

        if not getattr(self.hparams, "class_balanced_sampler", 0):
            weights = self._hard_negative_train_sampler(None)
            if weights is None:
                return None
            return WeightedRandomSampler(
                weights.double(), num_samples=len(weights), replacement=True
            )
        labels = getattr(self.train_dataset, "y", None)
        if labels is None:
            raise ValueError(
                "class_balanced_sampler requires the training dataset to expose y labels"
            )
        labels = torch.as_tensor(labels, dtype=torch.long)
        if labels.numel() == 0:
            raise ValueError("class_balanced_sampler received an empty training dataset")
        class_counts = torch.bincount(labels)
        if torch.any(class_counts[labels] == 0):
            raise ValueError("class_balanced_sampler found a label with zero count")
        weights = 1.0 / class_counts[labels].float()
        weights = self._hard_negative_train_sampler(weights)
        return WeightedRandomSampler(
            weights.double(), num_samples=len(weights), replacement=True
        )

    def _hard_negative_file_ids(self):
        manifest_path = getattr(self.hparams, "hard_negative_manifest", None)
        if not manifest_path:
            raise ValueError(
                "hard_negative_sampler requires --hard_negative_manifest"
            )
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(manifest_path)
        with open(manifest_path) as f:
            entries = json.load(f)
        if not isinstance(entries, list):
            raise ValueError("hard_negative_manifest must be a JSON list")
        file_ids = set()
        for idx, entry in enumerate(entries):
            if not isinstance(entry, dict) or "file_id" not in entry:
                raise ValueError(
                    f"hard_negative_manifest entry {idx} is missing file_id"
                )
            file_ids.add(str(entry["file_id"]))
        if not file_ids:
            raise ValueError("hard_negative_manifest contains no file_id entries")
        return file_ids

    def _hard_negative_train_sampler(self, weights):
        if not getattr(self.hparams, "hard_negative_sampler", 0):
            return weights

        data_df = getattr(self.train_dataset, "data_df", None)
        if data_df is None or "file_id" not in data_df:
            raise ValueError(
                "hard_negative_sampler requires train_dataset.data_df.file_id"
            )
        dataset_file_ids = [str(file_id) for file_id in data_df.file_id.tolist()]
        if not dataset_file_ids:
            raise ValueError("hard_negative_sampler received an empty training dataset")

        hard_file_ids = self._hard_negative_file_ids()
        hard_mask = torch.tensor(
            [file_id in hard_file_ids for file_id in dataset_file_ids],
            dtype=torch.bool,
        )
        hard_count = int(hard_mask.sum().item())
        if hard_count == 0:
            raise ValueError(
                "hard_negative_sampler found no manifest file_ids in the training split"
            )

        if weights is None:
            weights = torch.ones(len(dataset_file_ids), dtype=torch.float32)
        else:
            weights = torch.as_tensor(weights, dtype=torch.float32).clone()
            if weights.numel() != len(dataset_file_ids):
                raise ValueError(
                    "hard_negative_sampler weight count does not match dataset: "
                    f"{weights.numel()} vs {len(dataset_file_ids)}"
                )

        target_prob = float(getattr(self.hparams, "hard_negative_prob", 0.15))
        if not 0.0 < target_prob < 1.0:
            raise ValueError("hard_negative_prob must be in (0, 1)")

        hard_mass = weights[hard_mask].sum()
        total_mass = weights.sum()
        if hard_mass <= 0 or total_mass <= 0:
            raise ValueError("hard_negative_sampler received non-positive weights")

        current_prob = float((hard_mass / total_mass).item())
        multiplier = 1.0
        if current_prob < target_prob:
            non_hard_mass = total_mass - hard_mass
            multiplier = float(
                target_prob * non_hard_mass / (hard_mass * (1.0 - target_prob))
            )
            weights[hard_mask] *= multiplier

        final_prob = float((weights[hard_mask].sum() / weights.sum()).item())
        print(
            "Hard-negative sampler: "
            f"{hard_count}/{len(dataset_file_ids)} train clips from manifest; "
            f"sampling mass {current_prob:.3f} -> {final_prob:.3f}; "
            f"multiplier {multiplier:.3f}"
        )
        return weights

    def _specialist_train_sampler(self):
        if not getattr(self.hparams, "specialist_sampler", 0):
            return None
        if getattr(self.hparams, "class_balanced_sampler", 0):
            raise ValueError(
                "specialist_sampler owns train sampling and cannot be combined with "
                "class_balanced_sampler"
            )

        labels = getattr(self.train_dataset, "y", None)
        if labels is None:
            raise ValueError(
                "specialist_sampler requires the training dataset to expose y labels"
            )
        labels = torch.as_tensor(labels, dtype=torch.long)
        if labels.numel() == 0:
            raise ValueError("specialist_sampler received an empty training dataset")

        specialist_actions = _toyota_specialist_action_indices().to(labels.device)
        positive_mask = torch.isin(labels, specialist_actions)
        if not positive_mask.any():
            raise ValueError(
                "specialist_sampler found no specialist-positive samples in train split"
            )

        hard_prob = float(getattr(self.hparams, "hard_negative_prob", 0.15))
        if not getattr(self.hparams, "hard_negative_sampler", 0):
            hard_prob = 0.0
            hard_mask = torch.zeros_like(positive_mask)
        else:
            data_df = getattr(self.train_dataset, "data_df", None)
            if data_df is None or "file_id" not in data_df:
                raise ValueError(
                    "specialist_sampler with hard_negative_sampler requires "
                    "train_dataset.data_df.file_id"
                )
            hard_file_ids = self._hard_negative_file_ids()
            hard_mask = torch.tensor(
                [str(file_id) in hard_file_ids for file_id in data_df.file_id.tolist()],
                dtype=torch.bool,
            )
            if hard_mask.numel() != labels.numel():
                raise ValueError(
                    "specialist_sampler hard mask length does not match labels: "
                    f"{hard_mask.numel()} vs {labels.numel()}"
                )
            if hard_prob > 0.0 and not hard_mask.any():
                raise ValueError(
                    "specialist_sampler found no hard-negative manifest clips in train split"
                )

        positive_prob = float(getattr(self.hparams, "specialist_positive_prob", 0.55))
        anchor_prob = float(getattr(self.hparams, "normal_anchor_prob", 0.30))
        if min(positive_prob, hard_prob, anchor_prob) < 0.0:
            raise ValueError(
                "specialist_positive_prob, hard_negative_prob, and normal_anchor_prob "
                "must be non-negative"
            )
        total_prob = positive_prob + hard_prob + anchor_prob
        if abs(total_prob - 1.0) > 1e-6:
            raise ValueError(
                "specialist sampler probabilities must sum to 1.0: "
                f"specialist_positive_prob={positive_prob}, "
                f"hard_negative_prob={hard_prob}, normal_anchor_prob={anchor_prob}"
            )

        anchor_mask = ~(positive_mask | hard_mask)
        if anchor_prob > 0.0 and not anchor_mask.any():
            raise ValueError("specialist_sampler found no normal anchor samples")

        weights = torch.zeros(labels.numel(), dtype=torch.float32)
        if positive_prob > 0.0:
            weights[positive_mask] += positive_prob / positive_mask.float().sum()
        if hard_prob > 0.0:
            weights[hard_mask] += hard_prob / hard_mask.float().sum()
        if anchor_prob > 0.0:
            weights[anchor_mask] += anchor_prob / anchor_mask.float().sum()
        if weights.sum() <= 0:
            raise ValueError("specialist_sampler produced non-positive weights")

        overlap = int((positive_mask & hard_mask).sum().item())
        print(
            "Specialist sampler: "
            f"positives {int(positive_mask.sum().item())}/{labels.numel()} "
            f"mass {positive_prob:.3f}; "
            f"hard {int(hard_mask.sum().item())}/{labels.numel()} "
            f"mass {hard_prob:.3f}; "
            f"anchors {int(anchor_mask.sum().item())}/{labels.numel()} "
            f"mass {anchor_prob:.3f}; overlap positive&hard {overlap}"
        )
        return WeightedRandomSampler(
            weights.double(), num_samples=len(weights), replacement=True
        )

    def train_dataloader(self):
        # # Balanced batch sampler
        # loss_weights = self.train_dataset.calc_class_weights()
        # weight_sample = torch.tensor([loss_weights[i] for i in self.train_dataset.y])
        # sampler = torch.utils.data.sampler.WeightedRandomSampler(weight_sample, len(self.train_dataset))
        if getattr(self.hparams, "actor_prompt", 0) and self.hparams.mixup:
            raise ValueError("actor_prompt training does not support mixup/cutmix")
        sampler = self._class_balanced_train_sampler()
        if self.hparams.mixup:
            mixup_fn = Mixup(
                mixup_alpha=0.8,
                cutmix_alpha=1,
                cutmix_minmax=None,
                prob=1.0,
                switch_prob=0.5,
                mode="batch",
                label_smoothing=self.hparams.label_smoothing,
                num_classes=self.hparams.num_classes,
            )

            def collate_fn(batch):
                x, y = default_collate(batch)
                B, C, T, H, W = x.shape
                x = x.view(B, C * T, H, W)
                if isinstance(y, list):
                    y_tmp = y[0]
                    y_rest = y[1:]
                    x, y_res = mixup_fn(x, y_tmp)
                    y = [y_res] + y_rest
                    x = x.view(B, C, T, H, W)
                    return x, y
                x, y = mixup_fn(x, y)
                x = x.view(B, C, T, H, W)
                return x, y

            return DataLoader(
                self.train_dataset,
                batch_size=self.hparams.batch_size,
                shuffle=sampler is None,
                sampler=sampler,
                **self._loader_worker_kwargs(),
                collate_fn=collate_fn,
                drop_last=True,
                # worker_init_fn=dataload_init
            )
        else:
            if self.hparams.dataset_artifact == "thumos14":
                return DataLoader(
                    self.train_dataset,
                    batch_size=self.hparams.batch_size,
                    shuffle=sampler is None,
                    sampler=sampler,
                    **self._loader_worker_kwargs(),
                    collate_fn=thumos_collate_fn,
                    drop_last=True,
                )
            else:
                return DataLoader(
                    self.train_dataset,
                    batch_size=self.hparams.batch_size,
                    shuffle=sampler is None,
                    sampler=sampler,
                    **self._loader_worker_kwargs(),
                    drop_last=True,
                )

    def val_dataloader(self):
        if self.hparams.dataset_artifact == "thumos14":
            return DataLoader(
                self.val_dataset,
                batch_size=self.hparams.batch_size,
                **self._loader_worker_kwargs(),
                collate_fn=thumos_collate_fn,
            )
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            **self._loader_worker_kwargs(),
            # worker_init_fn=dataload_init
        )

    def test_dataloader(self):
        if self.hparams.dataset_artifact == "thumos14":
            return DataLoader(
                self.test_dataset,
                batch_size=self.hparams.batch_size,
                **self._loader_worker_kwargs(),
                collate_fn=thumos_collate_fn,
            )
        else:
            return DataLoader(
                self.test_dataset,
                batch_size=self.hparams.batch_size,
                **self._loader_worker_kwargs(),
            )

    def predict_dataloader(self):
        pass
