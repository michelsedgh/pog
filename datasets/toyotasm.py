# %%
from torch.utils.data import Dataset, get_worker_info
import torch
import os
import pandas as pd
import numpy as np
import torchvision
import torch.nn.functional as F
from argparse import ArgumentParser
import json
import re
import shutil
import tempfile
import zipfile
from utils.ntu import frame_utils as utils

try:
    import av
except ImportError:
    av = None

try:
    from mmpose.codecs import UDPHeatmap
except ImportError:
    UDPHeatmap = None

CS_DICT = {
    "Cook.Cleandishes": 1,
    "Cook.Cleanup": 2,
    "Cook.Cut": 3,
    "Cook.Stir": 4,
    "Cook.Usestove": 5,
    "Cutbread": 6,
    "Drink.Frombottle": 7,
    "Drink.Fromcan": 8,
    "Drink.Fromcup": 9,
    "Drink.Fromglass": 10,
    "Eat.Attable": 11,
    "Eat.Snack": 12,
    "Enter": 13,
    "Getup": 14,
    "Laydown": 15,
    "Leave": 16,
    "Makecoffee.Pourgrains": 17,
    "Makecoffee.Pourwater": 18,
    "Maketea.Boilwater": 19,
    "Maketea.Insertteabag": 20,
    "Pour.Frombottle": 21,
    "Pour.Fromcan": 22,
    "Pour.Fromkettle": 23,
    "Readbook": 24,
    "Sitdown": 25,
    "Takepills": 26,
    "Uselaptop": 27,
    "Usetablet": 28,
    "Usetelephone": 29,
    "Walk": 30,
    "WatchTV": 31,
}
CV_DICT = {
    "Cutbread": 1,
    "Drink.Frombottle": 2,
    "Drink.Fromcan": 3,
    "Drink.Fromcup": 4,
    "Drink.Fromglass": 5,
    "Eat.Attable": 6,
    "Eat.Snack": 7,
    "Enter": 8,
    "Getup": 9,
    "Leave": 10,
    "Pour.Frombottle": 11,
    "Pour.Fromcan": 12,
    "Readbook": 13,
    "Sitdown": 14,
    "Takepills": 15,
    "Uselaptop": 16,
    "Usetablet": 17,
    "Usetelephone": 18,
    "Walk": 19,
}


class ToyotaSMDataset(Dataset):
    def __init__(
        self,
        data_dir,
        set_type,
        test_num_segment=1,
        test_num_crop=1,
        **kwargs,
    ):
        """
        Args:
            root_dir (string): Directory with all the images.
            set_type (string): train, val, test
            task_type (string): cross_subject, cross_view
            modal (string): kinect_color, kinect_depth, kinect_ir, inner_mirror, a_column_co_driver, a_column_driver, ceiling,
                            steering_wheel
        """
        self.data_dir = data_dir
        self.set_type = set_type
        self.task_type = kwargs["task_type"]
        self.n_frames = kwargs["n_frames"]
        self.n_frames_stride = kwargs.get("n_frames_stride", 1)
        self.multi_thread_decode = bool(kwargs.get("multi_thread_decode", 1))
        self.n_landmarks = kwargs["n_landmarks"]
        self.heatmap_agg = kwargs["heatmap_agg"]
        self.actor_prompt = bool(kwargs.get("actor_prompt", 0))
        self.num_actor_tokens = int(kwargs.get("num_actor_tokens", 8))
        if self.num_actor_tokens <= 0:
            raise ValueError("num_actor_tokens must be positive")
        self.toyota_actor_box_expand = float(kwargs.get("toyota_actor_box_expand", 1.15))
        if self.toyota_actor_box_expand <= 0:
            raise ValueError("toyota_actor_box_expand must be positive")
        self.toyota_actor_box_jitter_prob = float(
            kwargs.get("toyota_actor_box_jitter_prob", 0.8)
        )
        self.toyota_actor_box_center_jitter = float(
            kwargs.get("toyota_actor_box_center_jitter", 0.08)
        )
        self.toyota_actor_box_scale_min = float(
            kwargs.get("toyota_actor_box_scale_min", 0.9)
        )
        self.toyota_actor_box_scale_max = float(
            kwargs.get("toyota_actor_box_scale_max", 1.3)
        )
        if self.toyota_actor_box_scale_min <= 0:
            raise ValueError("toyota_actor_box_scale_min must be positive")
        if self.toyota_actor_box_scale_max < self.toyota_actor_box_scale_min:
            raise ValueError(
                "toyota_actor_box_scale_max must be greater than or equal to min"
            )
        self.toyota_actor_background_box_prob = float(
            kwargs.get("toyota_actor_background_box_prob", 0.5)
        )
        if not 0 <= self.toyota_actor_background_box_prob <= 1:
            raise ValueError("toyota_actor_background_box_prob must be in [0, 1]")
        self.toyota_pose_guided_sampling = bool(
            kwargs.get("toyota_pose_guided_sampling", 1)
        )
        self.toyota_min_pose_frames = int(kwargs.get("toyota_min_pose_frames", 1))
        if self.toyota_min_pose_frames < 1:
            raise ValueError("toyota_min_pose_frames must be >= 1")
        self.current_epoch = int(kwargs.get("toyota_current_epoch", 0))
        self.synthetic_warmup_epochs = int(
            kwargs.get("toyota_synthetic_warmup_epochs", 3)
        )
        self.synthetic_two_actor_prob = float(
            kwargs.get("toyota_synthetic_two_actor_prob", 0.0)
        )
        self.synthetic_three_actor_prob = float(
            kwargs.get("toyota_synthetic_three_actor_prob", 0.0)
        )
        self.synthetic_same_class_prob = float(
            kwargs.get("toyota_synthetic_same_class_prob", 0.3)
        )
        total_synthetic_prob = (
            self.synthetic_two_actor_prob + self.synthetic_three_actor_prob
        )
        if total_synthetic_prob > 1.0:
            raise ValueError("Toyota synthetic actor probabilities must sum to <= 1")
        if self.synthetic_three_actor_prob > 0 and self.num_actor_tokens < 3:
            raise ValueError("3-person synthetic samples require at least 3 actor slots")
        self.pose_landmarks = int(kwargs.get("toyota_pose_landmarks", 13))
        if self.n_landmarks > 0 and self.pose_landmarks != self.n_landmarks:
            raise ValueError(
                "Toyota heatmap generation requires toyota_pose_landmarks to match "
                f"n_landmarks, got {self.pose_landmarks} and {self.n_landmarks}"
            )
        self.jitter_scales_min = kwargs["jitter_scales_min"]
        self.jitter_scales_max = kwargs["jitter_scales_max"]
        self.test_num_crop = test_num_crop
        self.test_num_segment = test_num_segment
        self.frame_source = kwargs.get("toyota_frame_source", "auto")
        self.split_source = kwargs.get("toyota_split_source", "auto")
        self.toyota_seed = int(kwargs.get("toyota_seed", 42))
        self.toyota_val_fraction = float(kwargs.get("toyota_val_fraction", 0.15))
        self.toyota_test_fraction = float(kwargs.get("toyota_test_fraction", 0.20))
        self.toyota_max_samples = int(kwargs.get("toyota_max_samples", 0) or 0)
        self.mp4_zip_path = kwargs.get("toyota_mp4_zip")
        if not self.mp4_zip_path and str(data_dir).lower().endswith(".zip"):
            self.mp4_zip_path = data_dir
            self.data_dir = os.path.dirname(data_dir) or "."
        self.video_cache_dir = kwargs.get("toyota_video_cache_dir") or os.path.join(
            tempfile.gettempdir(), "poguise_toyota_mp4_cache"
        )
        self.skeleton_zip_path = kwargs.get("toyota_skeleton_zip") or os.path.join(
            self.data_dir, "toyota_smarthome_skeleton_v1.2.zip"
        )
        self.frame_count_cache_path = kwargs.get(
            "toyota_frame_count_cache"
        ) or os.path.join(self.data_dir, "toyota_mp4_frame_counts.json")
        self._frame_count_cache = {}
        self._mp4_zip_names = None
        self.mean = torch.tensor([0.485, 0.456, 0.406])  # videomae normalization
        self.std = torch.tensor([0.229, 0.224, 0.225])
        # self.mean = torch.tensor([1,1,1])
        # self.std = torch.tensor([1,1,1])
        self._num_retries = 5
        self.needs_skeleton = self.n_landmarks > 0 or self.actor_prompt
        if self.n_landmarks:
            if UDPHeatmap is None:
                raise ImportError("mmpose is required when n_landmarks is greater than 0")
            self.heatmap_size = (56, 56)
            self.heatmap_generator = UDPHeatmap(
                input_size=(224, 224), heatmap_size=self.heatmap_size, sigma=1.5
            )
        self.frame_source = self._resolve_frame_source()
        if self.frame_source in ["mp4", "mp4_zip"]:
            self._load_frame_count_cache()
        self.data_df = self._load_split()
        if self.toyota_max_samples > 0:
            self.data_df = self._limit_samples(self.data_df, self.toyota_max_samples)
        if self.actor_prompt and self.needs_skeleton:
            self.data_df = self._filter_actor_pose_samples(self.data_df)
        self.data_df["label"] -= 1
        self.y = torch.tensor(self.data_df.label.values, dtype=torch.long)
        self.class_to_indices = {
            int(label): np.flatnonzero(self.y.numpy() == int(label))
            for label in torch.unique(self.y).tolist()
        }

        self.length = len(self.data_df)
        print(
            self.length,
            set_type,
            "num classes",
            len(torch.unique(self.y)),
            "num samples per class",
            torch.unique(self.y, return_counts=True),
            "n_frames_stride",
            self.n_frames_stride,
        )

    def setup(self, stage=None):
        print(f"Setting up ToyotaSMDataset for {self.set_type}...")
        if self.needs_skeleton:
            self.landmark_list = []
            # read landmarks into memory
            file_folder = "skeleton"
            skeleton_zip = None
            if not os.path.isdir(os.path.join(self.data_dir, file_folder)):
                if os.path.exists(self.skeleton_zip_path):
                    skeleton_zip = zipfile.ZipFile(self.skeleton_zip_path)
                else:
                    raise FileNotFoundError(
                        "Toyota skeleton labels were not found. Expected either "
                        f"{os.path.join(self.data_dir, file_folder)} or "
                        f"{self.skeleton_zip_path}."
                    )

            try:
                for i in range(self.length):
                    if i % 1000 == 0:
                        print(f"Loading landmarks: {i}/{self.length}")
                    row = self.data_df.iloc[i]
                    file_name = f"{row.file_id}_pose3d.json"
                    data = self._read_skeleton_json(file_folder, file_name, skeleton_zip)
                    landmarks_file = []
                    for frame in data["frames"]:
                        if len(frame) > 1:
                            print(frame, row.file_id)
                            raise ValueError("More than one person in frame")
                        if len(frame) == 0:
                            landmarks_file.append(torch.zeros((self.pose_landmarks, 2)))
                            continue
                        landmarks_x = frame[0]["pose2d"][: self.pose_landmarks]
                        landmarks_y = frame[0]["pose2d"][
                            self.pose_landmarks : self.pose_landmarks * 2
                        ]
                        landmarks = list(zip(landmarks_x, landmarks_y))
                        landmarks = torch.tensor(landmarks)
                        landmarks = torch.round(landmarks).to(torch.int)
                        landmarks_file.append(landmarks)
                    if not landmarks_file:
                        landmarks_file.append(torch.zeros((self.n_landmarks, 2)))
                    # repeat last landmark to match number of frames
                    landmarks_file.append(landmarks_file[-1])
                    self.landmark_list.append(landmarks_file)
            finally:
                if skeleton_zip is not None:
                    skeleton_zip.close()
            # iterate over frames and landmarks in memory

    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--data_dir", type=str, default="/datasets/toyotasm")
        parser.add_argument("--heatmap_agg", type=int, default=1)
        parser.add_argument("--num_classes", type=int, default=31)
        parser.add_argument("--n_frames", type=int, default=16)
        parser.add_argument("--n_frames_stride", type=int, default=1)
        parser.add_argument("--n_landmarks", type=int, default=13)
        parser.add_argument("--vis", type=float, default=0.0)
        parser.add_argument("--jitter_scales_min", type=int, default=256)
        parser.add_argument("--jitter_scales_max", type=int, default=320)
        parser.add_argument("--multi_thread_decode", type=int, default=1)
        parser.add_argument("--uniform_sampling", type=int, default=1)
        parser.add_argument("--backend_video", type=str, default="torch")
        parser.add_argument("--task_type", type=str, default="CS")
        parser.add_argument(
            "--toyota_frame_source",
            type=str,
            default="auto",
            choices=["auto", "frames", "mp4", "mp4_zip"],
            help="Toyota input source. auto uses extracted frames, then mp4, then a video zip.",
        )
        parser.add_argument(
            "--toyota_split_source",
            type=str,
            default="auto",
            choices=["auto", "files"],
            help="Toyota split source. auto uses official split files when present, otherwise creates a deterministic subject split.",
        )
        parser.add_argument("--toyota_seed", type=int, default=42)
        parser.add_argument("--toyota_val_fraction", type=float, default=0.15)
        parser.add_argument("--toyota_test_fraction", type=float, default=0.20)
        parser.add_argument("--toyota_max_samples", type=int, default=0)
        parser.add_argument("--toyota_skeleton_zip", type=str, default=None)
        parser.add_argument("--toyota_mp4_zip", type=str, default=None)
        parser.add_argument("--toyota_video_cache_dir", type=str, default=None)
        parser.add_argument("--toyota_frame_count_cache", type=str, default=None)
        parser.add_argument("--toyota_actor_box_expand", type=float, default=1.15)
        parser.add_argument("--toyota_actor_box_jitter_prob", type=float, default=0.8)
        parser.add_argument("--toyota_actor_box_center_jitter", type=float, default=0.08)
        parser.add_argument("--toyota_actor_box_scale_min", type=float, default=0.9)
        parser.add_argument("--toyota_actor_box_scale_max", type=float, default=1.3)
        parser.add_argument("--toyota_actor_background_box_prob", type=float, default=0.5)
        parser.add_argument("--toyota_pose_guided_sampling", type=int, default=1)
        parser.add_argument("--toyota_min_pose_frames", type=int, default=1)
        parser.add_argument("--toyota_pose_landmarks", type=int, default=13)
        parser.add_argument("--toyota_current_epoch", type=int, default=0)
        parser.add_argument("--toyota_synthetic_warmup_epochs", type=int, default=3)
        parser.add_argument("--toyota_synthetic_two_actor_prob", type=float, default=0.0)
        parser.add_argument("--toyota_synthetic_three_actor_prob", type=float, default=0.0)
        parser.add_argument("--toyota_synthetic_same_class_prob", type=float, default=0.3)

        return parser

    def _limit_samples(self, data_df, max_samples):
        if max_samples <= 0 or len(data_df) <= max_samples:
            return data_df.copy().reset_index(drop=True)

        split_offsets = {"train": 0, "val": 1, "test": 2}
        rng = np.random.default_rng(
            self.toyota_seed + split_offsets.get(self.set_type, 3)
        )
        labels = sorted(data_df["label"].unique().tolist())
        rng.shuffle(labels)

        buckets = {}
        pointers = {}
        selected = []
        for label in labels:
            indices = data_df.index[data_df["label"] == label].to_numpy()
            rng.shuffle(indices)
            buckets[label] = indices.tolist()
            pointers[label] = 0

        if max_samples >= 2:
            for label in labels:
                if len(selected) + 2 > max_samples:
                    break
                if len(buckets[label]) < 2:
                    continue
                selected.extend(buckets[label][:2])
                pointers[label] = 2

        while len(selected) < max_samples:
            progressed = False
            for label in labels:
                pointer = pointers[label]
                if pointer >= len(buckets[label]):
                    continue
                selected.append(buckets[label][pointer])
                pointers[label] = pointer + 1
                progressed = True
                if len(selected) >= max_samples:
                    break
            if not progressed:
                break

        limited = data_df.loc[selected].copy().reset_index(drop=True)
        print(
            f"Limited Toyota {self.set_type} split to {len(limited)} "
            f"class-balanced samples across {limited.label.nunique()} classes."
        )
        return limited

    def _filter_actor_pose_samples(self, data_df):
        file_folder = "skeleton"
        skeleton_zip = None
        if not os.path.isdir(os.path.join(self.data_dir, file_folder)):
            if os.path.exists(self.skeleton_zip_path):
                skeleton_zip = zipfile.ZipFile(self.skeleton_zip_path)
            else:
                raise FileNotFoundError(
                    "Toyota skeleton labels were not found. Expected either "
                    f"{os.path.join(self.data_dir, file_folder)} or "
                    f"{self.skeleton_zip_path}."
                )

        try:
            keep = []
            for row in data_df.itertuples(index=False):
                file_name = f"{row.file_id}_pose3d.json"
                data = self._read_skeleton_json(file_folder, file_name, skeleton_zip)
                keep.append(self._skeleton_has_pose(data, row.file_id))
        finally:
            if skeleton_zip is not None:
                skeleton_zip.close()

        filtered = data_df.loc[keep].copy().reset_index(drop=True)
        dropped = len(data_df) - len(filtered)
        if dropped > 0:
            print(
                f"Filtered Toyota {self.set_type} actor split: dropped "
                f"{dropped}/{len(data_df)} clips with no usable skeleton pose."
            )
        if len(filtered) == 0:
            raise RuntimeError("No Toyota actor samples have usable skeleton pose.")
        return filtered

    def _skeleton_has_pose(self, data, file_id):
        for frame in data["frames"]:
            if len(frame) > 1:
                print(frame, file_id)
                raise ValueError("More than one person in frame")
            if len(frame) == 0:
                continue
            pose2d = frame[0]["pose2d"]
            landmarks_x = pose2d[: self.pose_landmarks]
            landmarks_y = pose2d[self.pose_landmarks : self.pose_landmarks * 2]
            landmarks = np.asarray(list(zip(landmarks_x, landmarks_y)), dtype=np.float32)
            finite = np.isfinite(landmarks).all(axis=-1)
            non_zero = ~np.all(landmarks == 0, axis=-1)
            if (finite & non_zero).any():
                return True
        return False

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        synthetic_actor_count = self._synthetic_actor_count()
        if synthetic_actor_count > 1:
            return self._getitem_synthetic(idx, synthetic_actor_count)
        return self._getitem_single(idx)

    def set_epoch(self, epoch):
        self.current_epoch = int(epoch)

    def _pose_available_by_frame(self, idx, n_frames):
        if not self.needs_skeleton or not hasattr(self, "landmark_list"):
            return None
        keypoints = torch.stack(self.landmark_list[idx]).numpy()[:n_frames]
        if keypoints.ndim != 3 or keypoints.shape[0] == 0:
            return None
        finite = np.isfinite(keypoints).all(axis=-1)
        non_zero = ~np.all(keypoints == 0, axis=-1)
        return (finite & non_zero).any(axis=1)

    def _sample_pose_guided_start(self, start_min, start_max, pose_available):
        if pose_available is None or not pose_available.any():
            return None

        start_min = int(start_min)
        start_max = int(start_max)
        if start_max < start_min:
            start_max = start_min

        starts = np.arange(start_min, start_max + 1, dtype=int)
        if starts.size == 0:
            starts = np.array([start_min], dtype=int)

        hits = np.zeros(starts.shape[0], dtype=int)
        for start_idx, start in enumerate(starts):
            end = min(start + 128, len(pose_available) - 1)
            frame_idx = np.linspace(start, end, self.n_frames, dtype=int)
            frame_idx = np.clip(frame_idx, 0, len(pose_available) - 1)
            hits[start_idx] = int(pose_available[frame_idx].sum())

        enough_pose = hits >= self.toyota_min_pose_frames
        if enough_pose.any():
            candidates = starts[enough_pose]
            if self.set_type == "train":
                return int(candidates[np.random.randint(0, len(candidates))])
            center = (start_min + start_max) * 0.5
            return int(candidates[np.argmin(np.abs(candidates - center))])

        best_hit = int(hits.max())
        if best_hit <= 0:
            return None
        best_starts = starts[hits == best_hit]
        if self.set_type == "train":
            return int(best_starts[np.random.randint(0, len(best_starts))])
        center = (start_min + start_max) * 0.5
        return int(best_starts[np.argmin(np.abs(best_starts - center))])

    def _synthetic_actor_count(self):
        if not self.actor_prompt or self.set_type != "train":
            return 1
        if self.current_epoch < self.synthetic_warmup_epochs:
            return 1
        draw = np.random.random()
        if draw < self.synthetic_three_actor_prob:
            return 3
        if draw < self.synthetic_three_actor_prob + self.synthetic_two_actor_prob:
            return 2
        return 1

    def _getitem_single(self, idx, actor_slot=None):
        if self.set_type in ["train", "val"]:
            # -1 indicates random sampling.
            temporal_sample_index = -1
            spatial_sample_index = -1
            min_scale = self.jitter_scales_min
            max_scale = self.jitter_scales_max
            crop_size = 224
        elif self.set_type in ["test"]:
            temporal_sample_index = 1
            # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
            # center, or right if width is larger than height, and top, middle,
            # or bottom if height is larger than width.
            spatial_sample_index = 1
            min_scale, max_scale, crop_size = (
                [224] * 3
                if self.test_num_crop > 1
                else [self.jitter_scales_min] * 2 + [224]
            )
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale}) == 1
        else:
            raise NotImplementedError("Does not support {} mode".format(self.mode))

        for i_try in range(self._num_retries):

            file_id = self.data_df.iloc[idx].file_id
            n_frames = self._num_frames(file_id)
            label = self.y[idx]
            pose_available = None
            if (
                self.actor_prompt
                and self.needs_skeleton
                and self.toyota_pose_guided_sampling
            ):
                pose_available = self._pose_available_by_frame(idx, n_frames)
            if self.set_type == "test":
                start_frame = self.data_df.iloc[idx].start
                end_frame = self.data_df.iloc[idx].end
                if end_frame < 0 or end_frame >= n_frames:
                    end_frame = n_frames - 1
            elif n_frames > 128:  # test has 128 frames segments
                max_start = max(0, n_frames - 129)
                if self.set_type == "train":
                    start_frame = None
                    if pose_available is not None:
                        start_frame = self._sample_pose_guided_start(
                            0, max_start, pose_available
                        )
                    if start_frame is None:
                        start_frame = np.random.randint(0, n_frames - 128)
                    end_frame = min(start_frame + 128, n_frames - 1)
                else:
                    start_frame = None
                    if pose_available is not None:
                        start_frame = self._sample_pose_guided_start(
                            0, max_start, pose_available
                        )
                    if start_frame is None:
                        # get the middle 128 frames
                        start_frame = n_frames // 2 - 64
                        end_frame = n_frames // 2 + 64
                    else:
                        end_frame = min(start_frame + 128, n_frames - 1)
            else:
                start_frame = 0
                end_frame = n_frames - 1
            # evenly sample n frames from a list of frames

            frames_idx = np.linspace(start_frame, end_frame, self.n_frames, dtype=int)
            if len(frames_idx) < self.n_frames:
                frames_idx = np.pad(
                    frames_idx, (0, self.n_frames - len(frames_idx)), "edge"
                )
            frames = self._read_sampled_frames(file_id, frames_idx)
            # frames = frames[frames_idx]
            # convert frames from T, C, H, W to T H W C
            frames = frames.permute(0, 2, 3, 1)
            if self.needs_skeleton:
                keypoints = torch.stack(self.landmark_list[idx]).numpy()
                frames_idx_clamped = np.clip(frames_idx, 0, len(keypoints) - 1)
                keypoints = [keypoints[frames_idx_clamped]]
            else:
                keypoints = None

            frames = utils.tensor_normalize(frames, mean=self.mean, std=self.std)
            frames = frames.permute(3, 0, 1, 2)
            frames, keypoints = utils.spatial_sampling(
                frames,
                spatial_idx=spatial_sample_index if self.set_type != "val" else 1,
                min_scale=min_scale if self.set_type == "train" else min_scale,
                max_scale=max_scale if self.set_type == "train" else min_scale,
                crop_size=crop_size,
                random_horizontal_flip=True if self.set_type == "train" else False,
                inverse_uniform_sampling=False,
                keypoints=keypoints,
                keypoint_aware_crop=self.actor_prompt,
            )
            frames = frames.permute(1, 0, 2, 3)
            actor_target = None
            if self.actor_prompt:
                actor_target = self._build_actor_target(
                    keypoints,
                    label,
                    height=frames.shape[2],
                    width=frames.shape[3],
                    slot=actor_slot,
                )
                if not actor_target["valid"].any():
                    if i_try < self._num_retries - 1:
                        continue
                    raise ValueError(f"No valid actor box found for {file_id}")
            if self.n_landmarks:
                lnd_heatmap = torch.zeros(
                    frames.shape[0], self.n_landmarks, *self.heatmap_size
                )
                kp_vis = torch.zeros(self.n_landmarks, *self.heatmap_size)
                for person_idx in range(len(keypoints)):
                    for frame_idx in range(frames.shape[0]):
                        kp_frame = keypoints[person_idx][frame_idx]
                        kp_frame = np.expand_dims(kp_frame, axis=0)
                        # make negative values nan
                        kp_frame[kp_frame < 0] = np.nan
                        vis = np.ones(kp_frame.shape[1])
                        # Check if any value in kp_frame is out of bounds
                        out_of_bounds = (kp_frame[:, :, 0] > frames.shape[2]) | (
                            kp_frame[:, :, 1] > frames.shape[3]
                        )
                        # Set corresponding rows in vis to 0
                        vis[np.any(out_of_bounds, axis=0)] = 0

                        # set corresponding rows to 0 if kp_frame row is nan
                        nan_rows = np.isnan(kp_frame).any(axis=2)
                        # convert to 1d array
                        nan_rows = np.any(nan_rows, axis=0)
                        vis[nan_rows] = 0
                        vis = np.expand_dims(vis, axis=0)
                        hm = self.heatmap_generator.encode(kp_frame, vis)
                        lnd_heatmap[frame_idx] += torch.Tensor(hm["heatmaps"])
                        vis = vis[0]
                        mask = torch.zeros((vis.shape[0], *self.heatmap_size))
                        for i, kp in enumerate(vis):
                            if kp == 1:
                                mask[i] = 1
                        kp_vis += mask
                kp_vis = torch.clamp(kp_vis, 0, 1)
                num_zeros = kp_vis.sum(dim=1).sum(dim=1)
                # count number of zeros in kp_vis
                num_zeros = (num_zeros == 0).sum()
                if num_zeros > 10 and i_try < self._num_retries - 1:
                    continue
                if self.heatmap_agg == 1:
                    # calculate average heatmap over all time frames
                    lnd_heatmap = torch.sum(lnd_heatmap, dim=0)
                elif self.heatmap_agg == 2:
                    lnd_heatmap = torch.mean(lnd_heatmap, dim=0)
                # kp_vis = torch.ones_like(lnd_heatmap)
                if self.set_type == "test":
                    if self.actor_prompt:
                        actor_target["heatmap"] = lnd_heatmap
                        actor_target["kp_vis"] = kp_vis
                        return (
                            frames,
                            actor_target,
                            self.data_df.iloc[idx].file_id,
                            temporal_sample_index,
                            spatial_sample_index,
                        )
                    return (
                        frames,
                        label,
                        self.data_df.iloc[idx].file_id,
                        temporal_sample_index,
                        spatial_sample_index,
                        lnd_heatmap,
                    )
                if self.actor_prompt:
                    actor_target["heatmap"] = lnd_heatmap
                    actor_target["kp_vis"] = kp_vis
                    return frames, actor_target
                return frames, [label, lnd_heatmap, kp_vis]
            if self.actor_prompt:
                if self.set_type == "test":
                    return (
                        frames,
                        actor_target,
                        self.data_df.iloc[idx].file_id,
                        temporal_sample_index,
                        spatial_sample_index,
                    )
                return frames, actor_target
            if self.set_type == "test":
                return (
                    frames,
                    label,
                    self.data_df.iloc[idx].file_id,
                    temporal_sample_index,
                    spatial_sample_index,
            )
            return frames, label

    def _getitem_synthetic(self, idx, actor_count):
        if actor_count not in (2, 3):
            raise ValueError(f"Unsupported synthetic actor count: {actor_count}")

        indices = self._sample_synthetic_indices(idx, actor_count)
        slots = np.random.choice(self.num_actor_tokens, actor_count, replace=False)
        order = np.random.permutation(actor_count)
        indices = [indices[int(i)] for i in order]
        slots = slots[order]
        samples = [
            self._getitem_single(int(sample_idx), actor_slot=int(slots[i]))
            for i, sample_idx in enumerate(indices)
        ]

        canvas_width = samples[0][0].shape[-1]
        bounds = self._sample_panel_bounds(actor_count, canvas_width)
        frames = self._compose_synthetic_frames(
            [sample[0] for sample in samples], bounds
        )
        target = self._compose_synthetic_actor_target(
            [sample[1] for sample in samples],
            slots,
            bounds,
            canvas_width,
        )
        return frames, target

    def _sample_synthetic_indices(self, idx, actor_count):
        base_label = int(self.y[idx])
        indices = [int(idx)]
        if actor_count == 2:
            if np.random.random() < self.synthetic_same_class_prob:
                indices.append(self._sample_same_class_index(base_label, exclude=indices))
            else:
                indices.append(
                    self._sample_different_class_index(base_label, exclude=indices)
                )
            return indices

        while len(indices) < actor_count:
            if np.random.random() < self.synthetic_same_class_prob:
                labels = [int(self.y[i]) for i in indices]
                same_label = labels[np.random.randint(0, len(labels))]
                indices.append(
                    self._sample_same_class_index(same_label, exclude=indices)
                )
            else:
                labels = {int(self.y[i]) for i in indices}
                indices.append(self._sample_label_outside(labels, exclude=indices))
        return indices

    def _sample_same_class_index(self, label, exclude):
        candidates = [
            int(i)
            for i in self.class_to_indices.get(int(label), [])
            if int(i) not in set(exclude)
        ]
        if not candidates:
            raise RuntimeError(
                f"Cannot create same-class synthetic sample for class {label}"
            )
        return candidates[int(np.random.randint(0, len(candidates)))]

    def _sample_different_class_index(self, label, exclude):
        return self._sample_label_outside({int(label)}, exclude)

    def _sample_label_outside(self, labels, exclude):
        excluded = set(exclude)
        candidates = [
            int(i)
            for i, sample_label in enumerate(self.y.tolist())
            if int(sample_label) not in labels and i not in excluded
        ]
        if not candidates:
            raise RuntimeError(
                "Cannot create different-class synthetic Toyota sample from this split"
            )
        return candidates[int(np.random.randint(0, len(candidates)))]

    def _sample_panel_bounds(self, actor_count, width):
        if actor_count == 2:
            split = int(round(width * np.random.uniform(0.45, 0.55)))
            split = int(np.clip(split, int(width * 0.42), int(width * 0.58)))
            return [(0, split), (split, width)]
        left = int(round(width * np.random.uniform(0.30, 0.36)))
        middle = int(round(width * np.random.uniform(0.64, 0.70)))
        left = int(np.clip(left, int(width * 0.28), int(width * 0.38)))
        middle = int(np.clip(middle, int(width * 0.62), int(width * 0.72)))
        if middle <= left:
            left = width // 3
            middle = (2 * width) // 3
        return [(0, left), (left, middle), (middle, width)]

    def _scale_panel_bounds(self, bounds, source_width, target_width):
        scaled = []
        for x0, x1 in bounds:
            sx0 = int(round(x0 * target_width / source_width))
            sx1 = int(round(x1 * target_width / source_width))
            scaled.append((sx0, sx1))
        scaled[0] = (0, scaled[0][1])
        scaled[-1] = (scaled[-1][0], target_width)
        for idx in range(1, len(scaled)):
            prev_end = scaled[idx - 1][1]
            scaled[idx] = (prev_end, scaled[idx][1])
        return scaled

    def _compose_synthetic_frames(self, frames_list, bounds):
        _, _, height, width = frames_list[0].shape
        canvas = torch.zeros_like(frames_list[0])
        for frames, (x0, x1) in zip(frames_list, bounds):
            panel = F.interpolate(
                frames,
                size=(height, x1 - x0),
                mode="bilinear",
                align_corners=False,
            )
            canvas[:, :, :, x0:x1] = panel
        return canvas

    def _compose_synthetic_actor_target(self, targets, slots, bounds, canvas_width):
        boxes = torch.zeros((self.num_actor_tokens, 4), dtype=torch.float32)
        valid = torch.zeros(self.num_actor_tokens, dtype=torch.bool)
        actions = torch.full((self.num_actor_tokens,), -100, dtype=torch.long)

        for target, slot, (x0, x1) in zip(targets, slots, bounds):
            slot = int(slot)
            src_box = target["boxes"][slot]
            panel_x0 = x0 / float(canvas_width)
            panel_w = (x1 - x0) / float(canvas_width)
            boxes[slot] = torch.tensor(
                [
                    panel_x0 + src_box[0] * panel_w,
                    src_box[1],
                    panel_x0 + src_box[2] * panel_w,
                    src_box[3],
                ],
                dtype=torch.float32,
            ).clamp_(0.0, 1.0)
            valid[slot] = True
            actions[slot] = target["actions"][slot]

        self._fill_invalid_actor_boxes(boxes, valid)
        output = {"actions": actions, "boxes": boxes, "valid": valid}
        if self.n_landmarks > 0:
            heatmap_width = targets[0]["heatmap"].shape[-1]
            heatmap_bounds = self._scale_panel_bounds(
                bounds, source_width=canvas_width, target_width=heatmap_width
            )
            output["heatmap"] = self._compose_synthetic_heatmaps(
                [target["heatmap"] for target in targets],
                heatmap_bounds,
            )
            output["kp_vis"] = self._compose_synthetic_heatmaps(
                [target["kp_vis"] for target in targets],
                heatmap_bounds,
                nearest=True,
            ).clamp_(0, 1)
        return output

    def _compose_synthetic_heatmaps(self, heatmaps, bounds, nearest=False):
        channels, height, width = heatmaps[0].shape
        canvas = torch.zeros((channels, height, width), dtype=heatmaps[0].dtype)
        for heatmap, (x0, x1) in zip(heatmaps, bounds):
            mode = "nearest" if nearest else "bilinear"
            kwargs = {} if nearest else {"align_corners": False}
            panel = F.interpolate(
                heatmap.unsqueeze(0),
                size=(height, x1 - x0),
                mode=mode,
                **kwargs,
            ).squeeze(0)
            canvas[:, :, x0:x1] += panel
        return canvas

    def _build_actor_target(self, keypoints, label, height, width, slot=None):
        if keypoints is None:
            raise ValueError("Toyota actor_prompt requires skeleton keypoints")

        boxes = torch.zeros((self.num_actor_tokens, 4), dtype=torch.float32)
        valid = torch.zeros(self.num_actor_tokens, dtype=torch.bool)
        actions = torch.full((self.num_actor_tokens,), -100, dtype=torch.long)

        person_keypoints = np.asarray(keypoints[0], dtype=np.float32)
        finite = np.isfinite(person_keypoints).all(axis=-1)
        non_zero = ~np.all(person_keypoints == 0, axis=-1)
        in_frame = (
            (person_keypoints[..., 0] >= 0)
            & (person_keypoints[..., 0] < width)
            & (person_keypoints[..., 1] >= 0)
            & (person_keypoints[..., 1] < height)
        )
        visible = finite & non_zero & in_frame
        if not visible.any():
            return {
                "actions": actions,
                "boxes": boxes,
                "valid": valid,
            }

        points = person_keypoints[visible]
        x1 = float(points[:, 0].min())
        y1 = float(points[:, 1].min())
        x2 = float(points[:, 0].max())
        y2 = float(points[:, 1].max())

        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        bw = max(x2 - x1, 1.0) * self.toyota_actor_box_expand
        bh = max(y2 - y1, 1.0) * self.toyota_actor_box_expand
        x1 = max(0.0, cx - bw * 0.5)
        y1 = max(0.0, cy - bh * 0.5)
        x2 = min(float(width), cx + bw * 0.5)
        y2 = min(float(height), cy + bh * 0.5)

        slot = self._actor_slot(slot)
        box = torch.tensor(
            [x1 / width, y1 / height, x2 / width, y2 / height],
            dtype=torch.float32,
        ).clamp_(0.0, 1.0)
        if self.set_type == "train":
            box = self._jitter_actor_box(box)
        boxes[slot] = box
        valid[slot] = True
        actions[slot] = label.long() if torch.is_tensor(label) else int(label)
        self._fill_invalid_actor_boxes(boxes, valid)
        return {
            "actions": actions,
            "boxes": boxes,
            "valid": valid,
        }

    def _actor_slot(self, slot):
        if slot is not None:
            slot = int(slot)
            if slot < 0 or slot >= self.num_actor_tokens:
                raise ValueError(f"actor slot {slot} is outside K={self.num_actor_tokens}")
            return slot
        if self.set_type == "train":
            return int(np.random.randint(0, self.num_actor_tokens))
        return 0

    def _jitter_actor_box(self, box):
        if self.toyota_actor_box_jitter_prob <= 0:
            return box
        if np.random.random() >= self.toyota_actor_box_jitter_prob:
            return box

        x1, y1, x2, y2 = [float(v) for v in box.tolist()]
        bw = max(x2 - x1, 1e-4)
        bh = max(y2 - y1, 1e-4)
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        cx += np.random.uniform(
            -self.toyota_actor_box_center_jitter,
            self.toyota_actor_box_center_jitter,
        ) * bw
        cy += np.random.uniform(
            -self.toyota_actor_box_center_jitter,
            self.toyota_actor_box_center_jitter,
        ) * bh
        scale = np.random.uniform(
            self.toyota_actor_box_scale_min,
            self.toyota_actor_box_scale_max,
        )
        bw *= scale
        bh *= scale
        jittered = torch.tensor(
            [
                max(0.0, cx - bw * 0.5),
                max(0.0, cy - bh * 0.5),
                min(1.0, cx + bw * 0.5),
                min(1.0, cy + bh * 0.5),
            ],
            dtype=torch.float32,
        )
        if jittered[2] <= jittered[0] or jittered[3] <= jittered[1]:
            return box
        return jittered

    def _fill_invalid_actor_boxes(self, boxes, valid):
        if self.set_type != "train" or self.toyota_actor_background_box_prob <= 0:
            return
        invalid_slots = torch.nonzero(~valid, as_tuple=False).flatten().tolist()
        if not invalid_slots:
            return
        valid_slots = torch.nonzero(valid, as_tuple=False).flatten().tolist()
        for slot in invalid_slots:
            if np.random.random() > self.toyota_actor_background_box_prob:
                continue
            if valid_slots and np.random.random() < 0.5:
                src_slot = valid_slots[int(np.random.randint(0, len(valid_slots)))]
                boxes[slot] = self._shift_actor_box_negative(boxes[src_slot])
            else:
                boxes[slot] = self._random_actor_box()

    def _random_actor_box(self):
        width = float(np.random.uniform(0.08, 0.45))
        height = float(np.random.uniform(0.10, 0.65))
        x1 = float(np.random.uniform(0.0, 1.0 - width))
        y1 = float(np.random.uniform(0.0, 1.0 - height))
        return torch.tensor([x1, y1, x1 + width, y1 + height], dtype=torch.float32)

    def _shift_actor_box_negative(self, box):
        x1, y1, x2, y2 = [float(v) for v in box.tolist()]
        bw = max(x2 - x1, 1e-4)
        bh = max(y2 - y1, 1e-4)
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        cx += np.random.choice([-1.0, 1.0]) * np.random.uniform(0.65, 1.35) * bw
        cy += np.random.uniform(-0.75, 0.75) * bh
        scale_w = np.random.uniform(0.6, 1.6)
        scale_h = np.random.uniform(0.6, 1.6)
        bw *= scale_w
        bh *= scale_h
        shifted = torch.tensor(
            [
                max(0.0, cx - bw * 0.5),
                max(0.0, cy - bh * 0.5),
                min(1.0, cx + bw * 0.5),
                min(1.0, cy + bh * 0.5),
            ],
            dtype=torch.float32,
        )
        if shifted[2] <= shifted[0] or shifted[3] <= shifted[1]:
            return self._random_actor_box()
        return shifted

    def _resolve_frame_source(self):
        if self.frame_source != "auto":
            if self.frame_source == "mp4_zip":
                self.mp4_zip_path = self.mp4_zip_path or self._find_default_mp4_zip()
            return self.frame_source
        frames_dir = os.path.join(self.data_dir, "frames")
        if os.path.isdir(frames_dir) and any(os.scandir(frames_dir)):
            return "frames"
        mp4_dir = os.path.join(self.data_dir, "mp4")
        if os.path.isdir(mp4_dir):
            return "mp4"
        self.mp4_zip_path = self.mp4_zip_path or self._find_default_mp4_zip()
        if self.mp4_zip_path and os.path.exists(self.mp4_zip_path):
            return "mp4_zip"
        raise FileNotFoundError(
            f"Could not find Toyota frames, mp4 directory, or mp4 zip under {self.data_dir}."
        )

    def _find_default_mp4_zip(self):
        candidates = [
            "toyota_smarthome_mp4.zip",
            "toyota_smarthome_videos.zip",
            "toyotasm_mp4.zip",
            "mp4.zip",
            "videos.zip",
        ]
        for candidate in candidates:
            path = os.path.join(self.data_dir, candidate)
            if os.path.exists(path):
                return path
        return None

    def _label_dict(self):
        return CS_DICT if self.task_type == "CS" else CV_DICT

    def _load_split(self):
        if self.set_type == "test":
            split_path = os.path.join(self.data_dir, f"test_Labels_{self.task_type}.csv")
        else:
            split_path = os.path.join(
                self.data_dir, "splits", self.set_type + f"_{self.task_type}.txt"
            )
        if os.path.exists(split_path):
            return self._load_split_file(split_path)
        if self.split_source == "files":
            raise FileNotFoundError(
                f"Toyota split file not found: {split_path}. Use "
                "--toyota_split_source auto to generate a deterministic split."
            )
        print(
            f"Toyota split file not found: {split_path}. "
            "Generating deterministic subject split from mp4 filenames."
        )
        return self._build_auto_split()

    def _load_split_file(self, split_path):
        if self.set_type == "test":
            data_df = pd.read_csv(split_path)
            data_df.columns = ["file_id", "start", "end"]
        else:
            data_df = pd.read_csv(split_path)
            data_df.columns = ["file_id"]
            data_df["file_id"] = data_df["file_id"].apply(lambda x: x[:-4])
        data_df["label"] = data_df.file_id.apply(lambda x: x.split("_")[0])
        data_df["label"] = data_df.label.map(self._label_dict())
        data_df = data_df.dropna(subset=["label"]).copy()
        data_df["label"] = data_df["label"].astype(int)
        return data_df

    def _build_auto_split(self):
        mp4_dir = os.path.join(self.data_dir, "mp4")
        if self.frame_source == "mp4_zip":
            names = self._mp4_zip_index().keys()
        elif os.path.isdir(mp4_dir):
            names = [name[:-4] for name in os.listdir(mp4_dir) if name.endswith(".mp4")]
        else:
            raise FileNotFoundError(
                f"Automatic Toyota split creation needs mp4 files in {mp4_dir} "
                "or --toyota_mp4_zip."
            )
        rows = []
        label_dict = self._label_dict()
        for file_id in sorted(names):
            label_name = file_id.split("_")[0]
            if label_name not in label_dict:
                continue
            subject = self._subject_id(file_id)
            if subject is None:
                continue
            rows.append(
                {
                    "file_id": file_id,
                    "label": label_dict[label_name],
                    "subject": subject,
                }
            )
        if not rows:
            raise RuntimeError("No Toyota mp4 files with known labels found.")

        df = pd.DataFrame(rows)
        subjects = np.array(sorted(df.subject.unique()))
        rng = np.random.default_rng(self.toyota_seed)
        rng.shuffle(subjects)
        n_subjects = len(subjects)
        n_test = int(round(n_subjects * self.toyota_test_fraction))
        n_val = int(round((n_subjects - n_test) * self.toyota_val_fraction))
        if n_subjects >= 3:
            if self.toyota_test_fraction > 0:
                n_test = max(1, n_test)
            if self.toyota_val_fraction > 0:
                n_val = max(1, n_val)
        test_subjects = set(subjects[:n_test])
        val_subjects = set(subjects[n_test : n_test + n_val])

        if self.set_type == "test":
            split_df = df[df.subject.isin(test_subjects)].copy()
            split_df["start"] = 0
            split_df["end"] = -1
        elif self.set_type == "val":
            split_df = df[df.subject.isin(val_subjects)].copy()
        else:
            held_out = test_subjects | val_subjects
            split_df = df[~df.subject.isin(held_out)].copy()

        split_df = split_df.drop(columns=["subject"]).reset_index(drop=True)
        if split_df.empty:
            raise RuntimeError(
                f"Generated Toyota {self.set_type} split is empty. Adjust "
                "--toyota_val_fraction/--toyota_test_fraction."
            )
        return split_df

    def _subject_id(self, file_id):
        match = re.search(r"_p(\d+)_", file_id)
        if match is None:
            return None
        return int(match.group(1))

    def _read_skeleton_json(self, file_folder, file_name, skeleton_zip=None):
        skeleton_path = os.path.join(self.data_dir, file_folder, file_name)
        if os.path.exists(skeleton_path):
            with open(skeleton_path) as f:
                return json.load(f)
        if skeleton_zip is not None:
            with skeleton_zip.open(file_name) as f:
                return json.load(f)
        raise FileNotFoundError(skeleton_path)

    def _load_frame_count_cache(self):
        if self.frame_count_cache_path and os.path.exists(self.frame_count_cache_path):
            with open(self.frame_count_cache_path) as f:
                self._frame_count_cache = json.load(f)

    def _save_frame_count_cache(self):
        if not self.frame_count_cache_path:
            return
        if get_worker_info() is not None:
            return
        cache_dir = os.path.dirname(self.frame_count_cache_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        tmp_path = f"{self.frame_count_cache_path}.{os.getpid()}.tmp"
        with open(tmp_path, "w") as f:
            json.dump(self._frame_count_cache, f)
        os.replace(tmp_path, self.frame_count_cache_path)

    def _num_frames(self, file_id):
        if self.frame_source == "frames":
            return len(os.listdir(os.path.join(self.data_dir, "frames", file_id)))

        if file_id in self._frame_count_cache:
            return int(self._frame_count_cache[file_id])

        video_path = self._video_path(file_id)
        n_frames = self._count_video_frames(video_path)
        self._frame_count_cache[file_id] = int(n_frames)
        self._save_frame_count_cache()
        return n_frames

    def _count_video_frames(self, video_path):
        try:
            import cv2

            cap = cv2.VideoCapture(video_path)
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if n_frames > 0:
                return n_frames
        except Exception:
            pass

        if av is None:
            raise ImportError("PyAV is required to count Toyota mp4 frames.")
        with av.open(video_path) as container:
            stream = container.streams.video[0]
            if stream.frames:
                return int(stream.frames)
            return sum(1 for _ in container.decode(stream))

    def _read_sampled_frames(self, file_id, frames_idx):
        if self.frame_source == "frames":
            return self.read_all_frames(
                os.path.join(self.data_dir, "frames", file_id),
                frames_idx,
            )
        return self.read_all_video_frames(
            self._video_path(file_id),
            frames_idx,
        )

    def _video_path(self, file_id):
        if self.frame_source == "mp4":
            return os.path.join(self.data_dir, "mp4", file_id + ".mp4")
        if self.frame_source == "mp4_zip":
            return self._extract_video_from_zip(file_id)
        raise ValueError(f"Unsupported frame source for video path: {self.frame_source}")

    def _mp4_zip_index(self):
        if self._mp4_zip_names is not None:
            return self._mp4_zip_names
        if not self.mp4_zip_path or not os.path.exists(self.mp4_zip_path):
            raise FileNotFoundError(
                "Toyota mp4 zip not found. Pass --toyota_mp4_zip /path/to/videos.zip."
            )
        with zipfile.ZipFile(self.mp4_zip_path) as zf:
            self._mp4_zip_names = {
                os.path.splitext(os.path.basename(name))[0]: name
                for name in zf.namelist()
                if name.lower().endswith(".mp4")
            }
        return self._mp4_zip_names

    def _extract_video_from_zip(self, file_id):
        os.makedirs(self.video_cache_dir, exist_ok=True)
        cache_path = os.path.join(self.video_cache_dir, file_id + ".mp4")
        if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
            return cache_path

        zip_name = self._mp4_zip_index().get(file_id)
        if zip_name is None:
            raise FileNotFoundError(f"{file_id}.mp4 was not found inside {self.mp4_zip_path}")

        tmp_path = f"{cache_path}.{os.getpid()}.tmp"
        try:
            with zipfile.ZipFile(self.mp4_zip_path) as zf:
                with zf.open(zip_name) as src, open(tmp_path, "wb") as dst:
                    shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
            os.replace(tmp_path, cache_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        return cache_path

    def read_all_video_frames(self, video_path, frames_idx):
        if av is None:
            raise ImportError("PyAV is required for Toyota mp4 frame sampling.")

        frames_idx = np.asarray(frames_idx, dtype=np.int64)
        if frames_idx.size == 0:
            raise ValueError("frames_idx must contain at least one frame index")
        frames_idx = np.maximum(frames_idx, 0)

        frames_by_index = self._read_video_frames_pyav(video_path, frames_idx)
        frames = torch.stack([frames_by_index[int(frame_idx)] for frame_idx in frames_idx])
        if len(frames) < self.n_frames:
            frames = torch.cat(
                [
                    frames,
                    frames[-1]
                    .unsqueeze(0)
                    .repeat(self.n_frames - len(frames), 1, 1, 1),
                ]
            )
        return frames

    def _read_video_frames_pyav(self, video_path, frames_idx):
        unique_indices = np.unique(frames_idx.astype(np.int64))
        min_target = int(unique_indices[0])
        max_target = int(unique_indices[-1])
        frames_by_index = {}

        with av.open(video_path) as container:
            stream = container.streams.video[0]
            if self.multi_thread_decode:
                stream.thread_type = "AUTO"
            fps = self._video_stream_fps(stream)
            self._seek_video_container(container, stream, fps, min_target)

            target_pos = 0
            last_frame = None
            fallback_index = None
            for frame in container.decode(stream):
                decoded_index = self._frame_index_from_pts(frame, stream, fps)
                if decoded_index is None:
                    fallback_index = (
                        max(min_target - 1, 0)
                        if fallback_index is None
                        else fallback_index + 1
                    )
                    decoded_index = fallback_index

                if decoded_index < min_target:
                    continue

                if target_pos >= len(unique_indices):
                    break

                frame_tensor = None
                while (
                    target_pos < len(unique_indices)
                    and int(unique_indices[target_pos]) <= decoded_index
                ):
                    if frame_tensor is None:
                        frame_tensor = self._pyav_frame_to_tensor(frame)
                        last_frame = frame_tensor
                    frames_by_index[int(unique_indices[target_pos])] = frame_tensor
                    target_pos += 1

                if decoded_index >= max_target and target_pos >= len(unique_indices):
                    break

            if last_frame is not None and len(frames_by_index) < len(unique_indices):
                for frame_idx in unique_indices:
                    frames_by_index.setdefault(int(frame_idx), last_frame)

        if len(frames_by_index) != len(unique_indices):
            raise RuntimeError(f"No frames decoded from {video_path}")
        return frames_by_index

    def _video_stream_fps(self, stream):
        rate = stream.average_rate or stream.base_rate or stream.guessed_rate
        if rate is None:
            return 30.0
        return float(rate)

    def _seek_video_container(self, container, stream, fps, min_target):
        if min_target <= 0:
            return
        seek_frame = max(min_target - int(round(fps)), 0)
        seek_seconds = seek_frame / fps
        try:
            seek_offset = int(seek_seconds / float(stream.time_base))
            container.seek(seek_offset, any_frame=False, backward=True, stream=stream)
        except Exception:
            container.seek(0)

    def _frame_index_from_pts(self, frame, stream, fps):
        if frame.pts is None:
            return None
        start_time = stream.start_time or 0
        return int(round((frame.pts - start_time) * float(stream.time_base) * fps))

    def _pyav_frame_to_tensor(self, frame):
        frame_array = frame.to_ndarray(format="rgb24")
        return torch.from_numpy(frame_array).permute(2, 0, 1).contiguous()

    def read_all_frames(self, frame_folder, frames_idx=None):
        # read all frames in a folder
        frame_indices = range(1, 1 + len(os.listdir(frame_folder)))
        frames = []
        for i in frame_indices:
            if frames_idx is not None and i not in frames_idx:
                continue
            frame_path = os.path.join(frame_folder, f"img_{i:05d}.png")
            if not os.path.exists(frame_path):
                frame_path = os.path.join(frame_folder, f"img_{i:05d}.jpg")
            # use torchvision to read image
            frame = torchvision.io.read_image(frame_path)
            frames.append(frame)
        frames = torch.stack(frames)
        if len(frames) < self.n_frames:
            frames = torch.cat(
                [
                    frames,
                    frames[-1]
                    .unsqueeze(0)
                    .repeat(self.n_frames - len(frames), 1, 1, 1),
                ]
            )
        return frames
