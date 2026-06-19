# %%
from torch.utils.data import Dataset, get_worker_info
import torch
import os
import pandas as pd
import numpy as np
import torchvision
from argparse import ArgumentParser
import hashlib
import json
import math
import re
import shutil
import tempfile
import zipfile
from utils.ntu import frame_utils as utils
from datasets.object_vocab import (
    DETECTOR_TO_OBJECT,
    NONE_OBJECT_ID,
    NUM_OBJECT_CLASSES,
    OBJECT_TO_ID,
    OBJECTLESS_ACTIONS,
    OBJECT_ACTIONS_WITHOUT_RELIABLE_TEACHER,
    QUALITY_GATED_ACTION_OBJECTS,
    RELIABLE_ACTION_OBJECTS,
    STRONG_ACTION_OBJECTS,
    object_box_ignored_for_file_id,
    object_allowed_for_file_id,
    parse_object_camera_allowlist,
    parse_object_ignore_regions,
)
from datasets.toyota_action_taxonomy import (
    CS_DICT,
    CV_DICT,
    TOYOTA_ACTION_TAXONOMIES,
    normalize_toyota_action_taxonomy,
    toyota_action_names,
    toyota_action_object_map,
    toyota_label_dict,
    toyota_num_classes,
)

try:
    import av
except ImportError:
    av = None

try:
    from mmpose.codecs import UDPHeatmap
except ImportError:
    UDPHeatmap = None

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
        if "interaction_object_classes" in kwargs:
            raise ValueError(
                "interaction_object_classes was removed. Actor-object heatmaps "
                "are now one interacted-object channel per actor; object class "
                "semantics come from relation-only runtime object memory."
            )
        self.task_type = kwargs["task_type"]
        self.action_taxonomy = normalize_toyota_action_taxonomy(
            kwargs.get("toyota_action_taxonomy", "toyota_31")
        )
        self.action_names = toyota_action_names(
            self.task_type,
            self.action_taxonomy,
        )
        expected_num_classes = toyota_num_classes(
            self.task_type,
            self.action_taxonomy,
        )
        configured_num_classes = int(kwargs.get("num_classes", expected_num_classes))
        if configured_num_classes != expected_num_classes:
            raise ValueError(
                "Toyota action taxonomy/num_classes mismatch: "
                f"{self.action_taxonomy} {self.task_type} expects "
                f"{expected_num_classes} classes, got {configured_num_classes}."
            )
        self.n_frames = kwargs["n_frames"]
        self.n_frames_stride = kwargs.get("n_frames_stride", 1)
        self.multi_thread_decode = bool(kwargs.get("multi_thread_decode", 1))
        self.n_landmarks = kwargs["n_landmarks"]
        self.heatmap_agg = kwargs["heatmap_agg"]
        self.actor_prompt = bool(kwargs.get("actor_prompt", 0))
        self.num_actor_tokens = int(kwargs.get("num_actor_tokens", 8))
        if self.num_actor_tokens <= 0:
            raise ValueError("num_actor_tokens must be positive")
        self.actor_interaction_heatmaps = bool(
            kwargs.get("actor_interaction_heatmaps", 0)
        )
        if self.actor_interaction_heatmaps and not self.actor_prompt:
            raise ValueError("actor_interaction_heatmaps requires actor_prompt")
        if bool(kwargs.get("scene_object_tokens", 0)):
            raise ValueError(
                "scene_object_tokens was removed. Use actor_object_prompt_tokens=1 "
                "for relation-only runtime object memory."
            )
        self.actor_object_prompt_tokens = bool(
            kwargs.get("actor_object_prompt_tokens", 0)
        )
        if bool(kwargs.get("actor_object_slot_head", 0)):
            raise ValueError(
                "actor_object_slot_head was replaced by "
                "actor_object_prompt_tokens. Set --actor_object_prompt_tokens 1 "
                "and keep --actor_object_slot_head 0."
            )
        self.actor_object_residual_head = bool(
            kwargs.get("actor_object_residual_head", 0)
        )
        if self.actor_object_residual_head:
            raise ValueError(
                "actor_object_residual_head was removed. Use "
                "actor_object_relation_in_transformer with actor_object_prompt_tokens."
            )
        if self.actor_object_prompt_tokens and not self.actor_prompt:
            raise ValueError("actor_object_prompt_tokens requires actor_prompt")
        if self.actor_object_prompt_tokens and not self.actor_interaction_heatmaps:
            raise ValueError(
                "actor_object_prompt_tokens requires actor_interaction_heatmaps"
            )
        self.requires_object_proposals = self.actor_object_prompt_tokens
        self.num_scene_object_tokens = int(kwargs.get("num_scene_object_tokens", 32))
        if self.num_scene_object_tokens <= 0:
            raise ValueError("num_scene_object_tokens must be positive")
        self.interaction_teacher_enabled = (
            self.actor_interaction_heatmaps and self.set_type != "test"
        )
        self.num_object_classes = NUM_OBJECT_CLASSES
        self.object_detector_cache = kwargs.get("object_detector_cache")
        self.object_cache_dir = kwargs.get("toyota_object_cache_dir")
        self.object_camera_allowlist = parse_object_camera_allowlist(
            kwargs.get("object_camera_allowlist", None)
        )
        self.object_ignore_regions = parse_object_ignore_regions(
            kwargs.get("object_ignore_regions", None)
        )
        self.object_conf_threshold = float(kwargs.get("object_conf_threshold", 0.25))
        if not 0 <= self.object_conf_threshold <= 1:
            raise ValueError("object_conf_threshold must be in [0, 1]")
        if float(kwargs.get("object_temporal_conf_power", 0.0)) != 0.0:
            raise ValueError(
                "object_temporal_conf_power was removed. Object existence is controlled "
                "by detector-side thresholding and object_valid; sparse valid detections "
                "are not down-weighted by temporal coverage inside the actor model."
            )
        self.interaction_heatmap_size = int(
            kwargs.get("interaction_heatmap_size", 56)
        )
        if self.interaction_heatmap_size != 56:
            raise ValueError("Toyota interaction_heatmap_size must be 56")
        self.object_track_iou_threshold = float(
            kwargs.get("object_track_iou_threshold", 0.2)
        )
        if not 0 <= self.object_track_iou_threshold <= 1:
            raise ValueError("object_track_iou_threshold must be in [0, 1]")
        self.interaction_heatmap_sigma = float(
            kwargs.get("interaction_heatmap_sigma", 1.5)
        )
        if self.interaction_heatmap_sigma <= 0:
            raise ValueError("interaction_heatmap_sigma must be positive")
        self.action_object_map = toyota_action_object_map(
            self.task_type,
            self.action_taxonomy,
        )
        self.interaction_quality_min_actor_score = float(
            kwargs.get("interaction_quality_min_actor_score", 1.0)
        )
        self.interaction_quality_min_track_frames = int(
            kwargs.get("interaction_quality_min_track_frames", 1)
        )
        if self.interaction_quality_min_track_frames < 0:
            raise ValueError("interaction_quality_min_track_frames must be >= 0")
        self.interaction_quality_min_track_coverage = float(
            kwargs.get("interaction_quality_min_track_coverage", 0.0)
        )
        if not 0 <= self.interaction_quality_min_track_coverage <= 1:
            raise ValueError("interaction_quality_min_track_coverage must be in [0, 1]")
        self._expected_object_frame_cache = {}
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
        self.landmark_cache_dir = kwargs.get("toyota_landmark_cache_dir")
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
        self._video_size_cache = {}
        self.needs_skeleton = self.n_landmarks > 0 or self.actor_prompt
        self.heatmap_size = (
            self.interaction_heatmap_size,
            self.interaction_heatmap_size,
        )
        if self.n_landmarks:
            if UDPHeatmap is None:
                raise ImportError("mmpose is required when n_landmarks is greater than 0")
            self.heatmap_generator = UDPHeatmap(
                input_size=(224, 224), heatmap_size=self.heatmap_size, sigma=1.5
            )
        self.frame_source = self._resolve_frame_source()
        if self.frame_source in ["mp4", "mp4_zip"]:
            self._load_frame_count_cache()
        self._actor_train_window_starts = {}
        self.data_df = self._load_split()
        if self.toyota_max_samples > 0:
            self.data_df = self._limit_samples(self.data_df, self.toyota_max_samples)
        if self.actor_prompt and self.needs_skeleton:
            self.data_df = self._filter_actor_pose_samples(self.data_df)
        self.data_df["label"] -= 1
        self.y = torch.tensor(self.data_df.label.values, dtype=torch.long)
        self._object_cache = {}
        if self.interaction_teacher_enabled or self.requires_object_proposals:
            self._object_cache = self._load_object_cache(set(self.data_df.file_id))
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
            landmark_cache_path = self._landmark_cache_path()
            if landmark_cache_path and os.path.exists(landmark_cache_path):
                cached_landmarks = torch.load(
                    landmark_cache_path,
                    map_location="cpu",
                    weights_only=False,
                )
                if len(cached_landmarks) == self.length:
                    self.landmark_list = cached_landmarks
                    print(
                        f"Loaded preprocessed Toyota landmark cache: "
                        f"{landmark_cache_path}"
                    )
                    return
                print(
                    "Ignoring stale Toyota landmark cache with length "
                    f"{len(cached_landmarks)}; current split has {self.length} "
                    f"samples. Rebuilding {landmark_cache_path}."
                )

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
            if landmark_cache_path:
                tmp_path = f"{landmark_cache_path}.{os.getpid()}.tmp"
                torch.save(self.landmark_list, tmp_path)
                os.replace(tmp_path, landmark_cache_path)
                print(f"Saved preprocessed Toyota landmark cache: {landmark_cache_path}")
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
            "--toyota_action_taxonomy",
            type=str,
            default="toyota_31",
            choices=TOYOTA_ACTION_TAXONOMIES,
            help=(
                "Toyota action label space. product_v1 merges Drink.* into "
                "Drink, Cook.Cut+Cutbread into Cut, and drops Takepills."
            ),
        )
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
        parser.add_argument("--toyota_object_cache_dir", type=str, default=None)
        parser.add_argument("--toyota_landmark_cache_dir", type=str, default=None)
        parser.add_argument("--toyota_actor_box_expand", type=float, default=1.15)
        parser.add_argument("--toyota_actor_box_jitter_prob", type=float, default=0.8)
        parser.add_argument("--toyota_actor_box_center_jitter", type=float, default=0.08)
        parser.add_argument("--toyota_actor_box_scale_min", type=float, default=0.9)
        parser.add_argument("--toyota_actor_box_scale_max", type=float, default=1.3)
        parser.add_argument("--object_detector_cache", type=str, default=None)
        parser.add_argument("--object_camera_allowlist", type=str, default=None)
        parser.add_argument("--object_ignore_regions", type=str, default=None)
        parser.add_argument("--object_conf_threshold", type=float, default=0.25)
        parser.add_argument("--interaction_heatmap_size", type=int, default=56)
        parser.add_argument("--object_track_iou_threshold", type=float, default=0.2)
        parser.add_argument("--interaction_heatmap_sigma", type=float, default=1.5)
        parser.add_argument("--interaction_quality_min_actor_score", type=float, default=1.0)
        parser.add_argument("--interaction_quality_min_track_frames", type=int, default=1)
        parser.add_argument(
            "--interaction_quality_min_track_coverage", type=float, default=0.0
        )
        parser.add_argument("--toyota_pose_landmarks", type=int, default=13)

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
                height, width = self._video_size(row.file_id)
                if self.set_type == "train":
                    starts = self._skeleton_train_actor_window_starts(
                        data,
                        row.file_id,
                        self._num_frames(row.file_id),
                        height,
                        width,
                    )
                    self._actor_train_window_starts[str(row.file_id)] = starts
                    keep.append(len(starts) > 0)
                else:
                    keep.append(
                        self._skeleton_has_sampled_actor_pose(
                            data,
                            row,
                            self._num_frames(row.file_id),
                            height,
                            width,
                        )
                    )
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

    def _skeleton_keypoints_array(self, data, file_id):
        keypoints = []
        for frame in data["frames"]:
            if len(frame) > 1:
                print(frame, file_id)
                raise ValueError("More than one person in frame")
            if len(frame) == 0:
                keypoints.append(np.zeros((self.pose_landmarks, 2), dtype=np.float32))
                continue
            pose2d = frame[0]["pose2d"]
            landmarks_x = pose2d[: self.pose_landmarks]
            landmarks_y = pose2d[self.pose_landmarks : self.pose_landmarks * 2]
            landmarks = np.asarray(list(zip(landmarks_x, landmarks_y)), dtype=np.float32)
            keypoints.append(np.round(landmarks).astype(np.float32))
        if not keypoints:
            keypoints.append(np.zeros((self.pose_landmarks, 2), dtype=np.float32))
        keypoints.append(keypoints[-1].copy())
        return np.stack(keypoints, axis=0)

    def _visible_pose_by_frame(self, keypoints, n_frames, height, width):
        keypoints = np.asarray(keypoints, dtype=np.float32)[:n_frames]
        if keypoints.ndim != 3 or keypoints.shape[0] == 0:
            return None
        return self._visible_keypoints(keypoints, height, width).any(axis=1)

    def _skeleton_has_pose(self, data, file_id, height, width):
        keypoints = self._skeleton_keypoints_array(data, file_id)
        pose_available = self._visible_pose_by_frame(
            keypoints, len(keypoints), height, width
        )
        return pose_available is not None and bool(pose_available.any())

    def _skeleton_train_actor_window_starts(
        self,
        data,
        file_id,
        n_frames,
        height,
        width,
    ):
        keypoints = self._skeleton_keypoints_array(data, file_id)
        pose_available = self._visible_pose_by_frame(keypoints, n_frames, height, width)
        if pose_available is None or not pose_available.any():
            return []

        if n_frames > 128:
            start_candidates = range(max(1, int(n_frames) - 128))
        else:
            start_candidates = (0,)

        valid_starts = []
        for start_frame in start_candidates:
            if n_frames > 128:
                end_frame = min(int(start_frame) + 128, int(n_frames) - 1)
            else:
                end_frame = int(n_frames) - 1
            frames_idx = self._sample_frame_indices(start_frame, end_frame)
            frames_idx = np.clip(frames_idx, 0, len(keypoints) - 1)
            if bool(pose_available[frames_idx].any()):
                valid_starts.append(int(start_frame))
        return valid_starts

    def _skeleton_has_sampled_actor_pose(self, data, row, n_frames, height, width):
        keypoints = self._skeleton_keypoints_array(data, row.file_id)
        pose_available = self._visible_pose_by_frame(keypoints, n_frames, height, width)
        if pose_available is None or not pose_available.any():
            return False

        if self.set_type == "test":
            start_frame = int(getattr(row, "start", 0))
            end_frame = int(getattr(row, "end", n_frames - 1))
            if end_frame < 0 or end_frame >= n_frames:
                end_frame = n_frames - 1
        elif n_frames > 128:
            start_frame = n_frames // 2 - 64
            end_frame = n_frames // 2 + 64
        else:
            start_frame = 0
            end_frame = n_frames - 1

        frames_idx = self._sample_frame_indices(start_frame, end_frame)
        frames_idx = np.clip(frames_idx, 0, len(keypoints) - 1)
        return self._sampled_keypoints_survive_eval_crop(
            keypoints[frames_idx], height, width
        )

    def _keypoint_aware_axis_offset(self, length, size, coord):
        max_offset = int(length - size)
        if max_offset <= 0:
            return 0

        coord = float(coord)
        low = max(0, int(math.ceil(coord - size + 1)))
        high = min(max_offset, int(math.floor(coord)))
        if high >= low:
            return int(round((low + high) * 0.5))
        return int(np.clip(round(coord - size * 0.5), 0, max_offset))

    def _keypoint_aware_eval_crop_offsets(self, width, height, size, crop_points):
        center = np.median(crop_points, axis=0)
        anchor = crop_points[
            int(np.argmin(np.linalg.norm(crop_points - center, axis=1)))
        ]
        x_offset = self._keypoint_aware_axis_offset(width, size, anchor[0])
        y_offset = self._keypoint_aware_axis_offset(height, size, anchor[1])
        return x_offset, y_offset

    def _sampled_keypoints_survive_eval_crop(self, keypoints, height, width):
        # Match the deterministic eval path in utils.ntu.transform without reading frames.
        size = int(self.jitter_scales_min)
        crop_size = 224

        new_width = size
        new_height = size
        if width < height:
            new_height = int(math.floor((float(height) / width) * size))
        else:
            new_width = int(math.floor((float(width) / height) * size))

        scaled = np.asarray(keypoints, dtype=np.float32).copy()
        scaled *= np.asarray(
            [float(new_height) / height, float(new_width) / width],
            dtype=np.float32,
        )
        visible = self._visible_keypoints(scaled, new_height, new_width)
        if not visible.any():
            return False

        x_offset, y_offset = self._keypoint_aware_eval_crop_offsets(
            new_width, new_height, crop_size, scaled[visible]
        )
        cropped = scaled - np.asarray([x_offset, y_offset], dtype=np.float32)
        return bool(self._visible_keypoints(cropped, crop_size, crop_size).any())

    def _visible_keypoints(self, keypoints, height, width):
        finite = np.isfinite(keypoints).all(axis=-1)
        non_zero = ~np.all(keypoints == 0, axis=-1)
        in_frame = (
            (keypoints[..., 0] >= 0)
            & (keypoints[..., 0] < width)
            & (keypoints[..., 1] >= 0)
            & (keypoints[..., 1] < height)
        )
        return finite & non_zero & in_frame

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self._getitem_single(idx)

    def _sample_frame_indices(self, start_frame, end_frame):
        return np.linspace(start_frame, end_frame, self.n_frames, dtype=int)

    def _expected_interaction_object_ids(self, action_name):
        return {
            int(OBJECT_TO_ID[object_name])
            for object_name in self._expected_interaction_object_names(action_name)
            if object_name in OBJECT_TO_ID
        }

    def _expected_interaction_object_names(self, action_name):
        object_names = STRONG_ACTION_OBJECTS.get(action_name)
        if object_names is None:
            object_names = self.action_object_map.get(action_name, ())
        return object_names

    def _expected_interaction_frames(self, file_id, action_name):
        expected_ids = self._expected_interaction_object_ids(action_name)
        if not expected_ids:
            return np.asarray([], dtype=int)

        key = (str(file_id), str(action_name))
        cached = self._expected_object_frame_cache.get(key)
        if cached is not None:
            return cached

        frames = []
        for frame_idx, objects in self._object_cache.get(file_id, {}).items():
            if any(int(obj["cls_id"]) in expected_ids for obj in objects):
                frames.append(int(frame_idx))
        output = np.asarray(sorted(set(frames)), dtype=int)
        self._expected_object_frame_cache[key] = output
        return output

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
            row = self.data_df.iloc[idx]
            label = self.y[idx]
            raw_action_name = str(row.raw_action)
            if self.set_type == "test":
                start_frame = self.data_df.iloc[idx].start
                end_frame = self.data_df.iloc[idx].end
                if end_frame < 0 or end_frame >= n_frames:
                    end_frame = n_frames - 1
            elif n_frames > 128:  # test has 128 frames segments
                if self.set_type == "train":
                    actor_window_starts = self._actor_train_window_starts.get(
                        str(file_id),
                        None,
                    )
                    if self.actor_prompt and self.needs_skeleton:
                        if not actor_window_starts:
                            raise ValueError(
                                f"No actor-visible training window found for {file_id}"
                            )
                        start_frame = int(
                            actor_window_starts[
                                np.random.randint(0, len(actor_window_starts))
                            ]
                        )
                    else:
                        start_frame = np.random.randint(0, n_frames - 128)
                    end_frame = min(start_frame + 128, n_frames - 1)
                else:
                    # get the middle 128 frames
                    start_frame = n_frames // 2 - 64
                    end_frame = n_frames // 2 + 64
            else:
                start_frame = 0
                end_frame = n_frames - 1
            # evenly sample n frames from a list of frames

            frames_idx = self._sample_frame_indices(start_frame, end_frame)
            action_name = (
                raw_action_name
                if (self.interaction_teacher_enabled or self.requires_object_proposals)
                else None
            )
            if len(frames_idx) < self.n_frames:
                frames_idx = np.pad(
                    frames_idx, (0, self.n_frames - len(frames_idx)), "edge"
                )
            object_entries = (
                self._sample_object_entries(file_id, frames_idx, action_name)
                if (self.interaction_teacher_enabled or self.requires_object_proposals)
                else []
            )
            object_keypoints = (
                self._object_entries_to_keypoints(object_entries, len(frames_idx))
                if object_entries
                else None
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
            sampled = utils.spatial_sampling(
                frames,
                spatial_idx=spatial_sample_index if self.set_type != "val" else 1,
                min_scale=min_scale if self.set_type == "train" else min_scale,
                max_scale=max_scale if self.set_type == "train" else min_scale,
                crop_size=crop_size,
                random_horizontal_flip=True if self.set_type == "train" else False,
                inverse_uniform_sampling=False,
                keypoints=keypoints,
                obj_keypoints=object_keypoints,
                keypoint_aware_crop=self.actor_prompt,
            )
            if object_keypoints is not None:
                frames, keypoints, object_keypoints = sampled
            else:
                frames, keypoints = sampled
            frames = frames.permute(1, 0, 2, 3)
            object_entries = (
                self._transformed_object_entries(
                    object_entries,
                    object_keypoints,
                    height=frames.shape[2],
                    width=frames.shape[3],
                )
                if object_entries
                else []
            )
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
                if self.interaction_teacher_enabled or self.requires_object_proposals:
                    actor_target.update(
                        self._build_interaction_target(
                            object_entries,
                            actor_target,
                            height=frames.shape[2],
                            width=frames.shape[3],
                            action_names_by_slot={
                                int(slot): raw_action_name
                                for slot in torch.nonzero(
                                    actor_target["valid"],
                                    as_tuple=False,
                                ).flatten()
                            },
                        )
                    )
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
                    lnd_heatmap = torch.mean(lnd_heatmap, dim=0)
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

    def _box_iou(self, box_a, box_b):
        ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
        bx1, by1, bx2, by2 = [float(v) for v in box_b]
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter
        if union <= 0:
            return 0.0
        return inter / union

    def _object_tracks(self, object_entries):
        tracks = []
        entries = sorted(
            object_entries,
            key=lambda item: (
                int(item["sample_pos"]),
                int(item["cls_id"]),
                -float(item["conf"]),
            ),
        )
        for entry in entries:
            best_track = None
            best_iou = 0.0
            for track in tracks:
                if track["cls_id"] != int(entry["cls_id"]):
                    continue
                iou = self._box_iou(track["last_box"], entry["xyxy"])
                if iou > best_iou:
                    best_iou = iou
                    best_track = track
            if best_track is not None and best_iou >= self.object_track_iou_threshold:
                best_track["boxes"].append(entry["xyxy"])
                best_track["confs"].append(float(entry["conf"]))
                best_track["frames"].add(int(entry["sample_pos"]))
                best_track["last_box"] = entry["xyxy"]
                best_track["entries"].append(entry)
            else:
                tracks.append(
                    {
                        "cls_id": int(entry["cls_id"]),
                        "boxes": [entry["xyxy"]],
                        "confs": [float(entry["conf"])],
                        "frames": {int(entry["sample_pos"])},
                        "last_box": entry["xyxy"],
                        "entries": [entry],
                    }
                )
        return tracks

    def _action_name_from_label(self, label):
        label = int(label)
        if 0 <= label < len(self.action_names):
            return self.action_names[label]
        return None

    def _interaction_heatmap_from_box(self, box):
        hm_h, hm_w = self.heatmap_size
        box = box.float().clamp(0.0, 1.0)
        center = (box[:2] + box[2:]) * 0.5
        center_x = center[0] * float(hm_w) - 0.5
        center_y = center[1] * float(hm_h) - 0.5
        sigma = float(self.interaction_heatmap_sigma)
        y = torch.arange(hm_h, dtype=torch.float32)
        x = torch.arange(hm_w, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        heatmap = torch.exp(
            -0.5
            * (
                ((grid_x - center_x) / sigma) ** 2
                + ((grid_y - center_y) / sigma) ** 2
            )
        )
        return heatmap.clamp_(0.0, 1.0)

    def _interaction_motion_heatmap_from_entries(self, entries, height, width):
        frame_heatmaps = torch.zeros(
            (self.n_frames, *self.heatmap_size),
            dtype=torch.float32,
        )
        if not entries:
            return frame_heatmaps.max(dim=0).values

        width = float(width)
        height = float(height)
        if width <= 0 or height <= 0:
            raise ValueError("Toyota object heatmap dimensions must be positive")

        for entry in entries:
            sample_pos = int(entry["sample_pos"])
            if sample_pos < 0 or sample_pos >= self.n_frames:
                continue
            x1, y1, x2, y2 = [float(v) for v in entry["xyxy"].tolist()]
            box = torch.tensor(
                [
                    x1 / width,
                    y1 / height,
                    x2 / width,
                    y2 / height,
                ],
                dtype=torch.float32,
            ).clamp_(0.0, 1.0)
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            frame_heatmaps[sample_pos] = torch.maximum(
                frame_heatmaps[sample_pos],
                self._interaction_heatmap_from_box(box),
            )
        return frame_heatmaps.mean(dim=0).clamp_(0.0, 1.0)

    def _normalized_object_box(self, entry, height, width):
        width = float(width)
        height = float(height)
        if width <= 0 or height <= 0:
            raise ValueError("Toyota object dimensions must be positive")
        x1, y1, x2, y2 = [float(v) for v in entry["xyxy"].tolist()]
        box = torch.tensor(
            [
                x1 / width,
                y1 / height,
                x2 / width,
                y2 / height,
            ],
            dtype=torch.float32,
        ).clamp_(0.0, 1.0)
        return box

    @staticmethod
    def _expand_normalized_box(box, expand):
        x1, y1, x2, y2 = [float(v) for v in box.tolist()]
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        bw = max(x2 - x1, 1e-4) * float(expand)
        bh = max(y2 - y1, 1e-4) * float(expand)
        return torch.tensor(
            [
                max(0.0, cx - bw * 0.5),
                max(0.0, cy - bh * 0.5),
                min(1.0, cx + bw * 0.5),
                min(1.0, cy + bh * 0.5),
            ],
            dtype=torch.float32,
        )

    @staticmethod
    def _normalized_box_area(box):
        return max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))

    @staticmethod
    def _normalized_intersection_area(box_a, box_b):
        ix1 = max(float(box_a[0]), float(box_b[0]))
        iy1 = max(float(box_a[1]), float(box_b[1]))
        ix2 = min(float(box_a[2]), float(box_b[2]))
        iy2 = min(float(box_a[3]), float(box_b[3]))
        return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)

    @staticmethod
    def _point_box_distance(point, box):
        px, py = float(point[0]), float(point[1])
        dx = max(float(box[0]) - px, 0.0, px - float(box[2]))
        dy = max(float(box[1]) - py, 0.0, py - float(box[3]))
        return math.sqrt(dx * dx + dy * dy)

    def _interaction_track_actor_score(self, track, actor_box, height, width):
        actor_box = actor_box.float().clamp(0.0, 1.0)
        expanded_actor_box = self._expand_normalized_box(actor_box, expand=1.45)
        actor_center = (actor_box[:2] + actor_box[2:]) * 0.5
        actor_size = (actor_box[2:] - actor_box[:2]).clamp_min(1e-4)
        actor_diag = float(torch.linalg.norm(actor_size).clamp_min(1e-4))

        best_entry_score = -float("inf")
        for entry in track["entries"]:
            object_box = self._normalized_object_box(entry, height, width)
            if object_box[2] <= object_box[0] or object_box[3] <= object_box[1]:
                continue
            object_center = (object_box[:2] + object_box[2:]) * 0.5
            object_area = self._normalized_box_area(object_box)
            containment = 0.0
            if object_area > 0:
                containment = self._normalized_intersection_area(
                    object_box,
                    expanded_actor_box,
                ) / object_area
            center_inside = (
                float(expanded_actor_box[0])
                <= float(object_center[0])
                <= float(expanded_actor_box[2])
                and float(expanded_actor_box[1])
                <= float(object_center[1])
                <= float(expanded_actor_box[3])
            )
            outside_distance = self._point_box_distance(
                object_center,
                expanded_actor_box,
            )
            center_distance = (
                float(torch.linalg.norm(object_center - actor_center)) / actor_diag
            )
            conf = float(entry.get("conf", 0.0))
            entry_score = (
                conf
                + 2.0 * float(center_inside)
                + 1.5 * containment
                - 2.0 * outside_distance
                - 0.25 * center_distance
            )
            best_entry_score = max(best_entry_score, entry_score)

        if best_entry_score == -float("inf"):
            return best_entry_score
        coverage = len(track["frames"]) / float(max(self.n_frames, 1))
        return best_entry_score + 0.2 * coverage

    def _interaction_track_quality_passes(self, action_name, score, track):
        if action_name in RELIABLE_ACTION_OBJECTS:
            return True
        if action_name not in QUALITY_GATED_ACTION_OBJECTS:
            return False
        if float(score) < self.interaction_quality_min_actor_score:
            return False
        track_frames = len(track["frames"])
        track_coverage = track_frames / float(max(self.n_frames, 1))
        if track_frames < self.interaction_quality_min_track_frames:
            return False
        if track_coverage < self.interaction_quality_min_track_coverage:
            return False
        return True

    def _empty_scene_object_target(self):
        return {
            "object_boxes": torch.zeros(
                (self.num_scene_object_tokens, 4),
                dtype=torch.float32,
            ),
            "object_classes": torch.full(
                (self.num_scene_object_tokens,),
                NONE_OBJECT_ID,
                dtype=torch.long,
            ),
            "object_confs": torch.zeros(
                self.num_scene_object_tokens,
                dtype=torch.float32,
            ),
            "object_valid": torch.zeros(
                self.num_scene_object_tokens,
                dtype=torch.bool,
            ),
        }

    def _scene_object_track_sort_key(self, track):
        max_conf = max(float(conf) for conf in track["confs"]) if track["confs"] else 0.0
        mean_conf = (
            float(np.mean([float(conf) for conf in track["confs"]]))
            if track["confs"]
            else 0.0
        )
        coverage = len(track["frames"]) / float(max(self.n_frames, 1))
        return (-max_conf, -mean_conf, -coverage, int(track["cls_id"]))

    def _track_normalized_box(self, track, height, width):
        boxes = []
        weights = []
        for entry in track["entries"]:
            box = self._normalized_object_box(entry, height, width)
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            boxes.append(box)
            weights.append(max(float(entry.get("conf", 0.0)), 1e-4))
        if not boxes:
            return None
        box_tensor = torch.stack(boxes, dim=0)
        weight = torch.tensor(weights, dtype=torch.float32)
        weight = weight / weight.sum().clamp_min(1e-6)
        return (box_tensor * weight[:, None]).sum(dim=0).clamp_(0.0, 1.0)

    def _build_scene_object_target(self, object_tracks, height, width):
        target = self._empty_scene_object_target()
        track_to_slot = {}
        sorted_tracks = sorted(object_tracks, key=self._scene_object_track_sort_key)
        for slot, track in enumerate(sorted_tracks[: self.num_scene_object_tokens]):
            box = self._track_normalized_box(track, height, width)
            if box is None or box[2] <= box[0] or box[3] <= box[1]:
                continue
            target["object_boxes"][slot] = box
            target["object_classes"][slot] = int(track["cls_id"])
            target["object_confs"][slot] = (
                max(float(conf) for conf in track["confs"]) if track["confs"] else 0.0
            )
            target["object_valid"][slot] = True
            track_to_slot[id(track)] = int(slot)
        return target, track_to_slot

    def _build_interaction_targets(
        self,
        actor_target,
        object_tracks,
        track_to_object_slot,
        height,
        width,
        action_names_by_slot=None,
    ):
        interaction_cls = torch.full(
            (self.num_actor_tokens,), NONE_OBJECT_ID, dtype=torch.long
        )
        interaction_valid = torch.zeros(self.num_actor_tokens, dtype=torch.bool)
        interaction_heatmap = torch.zeros(
            (self.num_actor_tokens, *self.heatmap_size),
            dtype=torch.float32,
        )
        interaction_heatmap_valid = torch.zeros(
            self.num_actor_tokens,
            dtype=torch.bool,
        )
        interaction_heatmap_positive_valid = torch.zeros(
            self.num_actor_tokens,
            dtype=torch.bool,
        )
        interaction_object_index = torch.zeros(
            self.num_actor_tokens,
            dtype=torch.long,
        )
        interaction_object_index_valid = torch.zeros(
            self.num_actor_tokens,
            dtype=torch.bool,
        )

        for slot in torch.nonzero(actor_target["valid"], as_tuple=False).flatten():
            slot = int(slot)
            action_label = int(actor_target["actions"][slot])
            if action_label < 0:
                continue
            action_name = None
            if action_names_by_slot is not None:
                action_name = action_names_by_slot.get(slot)
            if action_name is None:
                action_name = self._action_name_from_label(action_label)
            if action_name is None:
                continue
            positive_ids = [
                int(OBJECT_TO_ID[object_name])
                for object_name in self._expected_interaction_object_names(action_name)
                if object_name in OBJECT_TO_ID
            ]
            if not positive_ids:
                if action_name in OBJECTLESS_ACTIONS:
                    interaction_object_index[slot] = 0
                    interaction_object_index_valid[slot] = True
                elif action_name not in OBJECT_ACTIONS_WITHOUT_RELIABLE_TEACHER:
                    raise RuntimeError(
                        "Toyota action is missing an object-target policy: "
                        f"{action_name}"
                    )
                continue
            positive_id_set = set(positive_ids)
            positive_tracks = [
                track
                for track in object_tracks
                if int(track["cls_id"]) in positive_id_set
            ]
            if not positive_tracks:
                continue
            actor_box = actor_target["boxes"][slot]
            scored_tracks = [
                (
                    self._interaction_track_actor_score(
                        track,
                        actor_box,
                        height,
                        width,
                    ),
                    track,
                )
                for track in positive_tracks
            ]
            scored_tracks = [
                (score, track)
                for score, track in scored_tracks
                if math.isfinite(float(score))
            ]
            if not scored_tracks:
                continue
            best_score, best_track = max(scored_tracks, key=lambda item: item[0])
            if not best_track["entries"]:
                continue
            if not self._interaction_track_quality_passes(
                action_name,
                best_score,
                best_track,
            ):
                continue
            interaction_cls[slot] = int(best_track["cls_id"])
            interaction_valid[slot] = True
            if id(best_track) in track_to_object_slot:
                interaction_object_index[slot] = int(track_to_object_slot[id(best_track)]) + 1
                interaction_object_index_valid[slot] = True
            interaction_heatmap[slot] = self._interaction_motion_heatmap_from_entries(
                best_track["entries"],
                height,
                width,
            )
            interaction_heatmap_valid[slot] = bool(
                interaction_heatmap[slot].max() > 0
            )
            interaction_heatmap_positive_valid[slot] = bool(
                interaction_heatmap_valid[slot]
            )

        return (
            interaction_cls,
            interaction_valid,
            interaction_heatmap,
            interaction_heatmap_valid,
            interaction_heatmap_positive_valid,
            interaction_object_index,
            interaction_object_index_valid,
        )

    def _build_interaction_target(
        self,
        object_entries,
        actor_target,
        height,
        width,
        action_names_by_slot=None,
    ):
        object_tracks = self._object_tracks(object_entries)
        scene_object_target = {}
        track_to_object_slot = {}
        if self.requires_object_proposals:
            scene_object_target, track_to_object_slot = self._build_scene_object_target(
                object_tracks,
                height,
                width,
            )
        (
            interaction_cls,
            interaction_valid,
            interaction_heatmap,
            interaction_heatmap_valid,
            interaction_heatmap_positive_valid,
            interaction_object_index,
            interaction_object_index_valid,
        ) = self._build_interaction_targets(
            actor_target,
            object_tracks,
            track_to_object_slot,
            height,
            width,
            action_names_by_slot=action_names_by_slot,
        )
        output = {
            "interaction_cls": interaction_cls,
            "interaction_valid": interaction_valid,
            "interaction_heatmap": interaction_heatmap,
            "interaction_heatmap_valid": interaction_heatmap_valid,
            "interaction_heatmap_positive_valid": interaction_heatmap_positive_valid,
            "interaction_object_index": interaction_object_index,
            "interaction_object_index_valid": interaction_object_index_valid,
        }
        output.update(scene_object_target)
        return output

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
        return toyota_label_dict(self.task_type, self.action_taxonomy)

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
            "Generating deterministic subject split from available video ids."
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
        data_df["raw_action"] = data_df.file_id.apply(lambda x: x.split("_")[0])
        data_df["label"] = data_df.raw_action.map(self._label_dict())
        data_df = data_df.dropna(subset=["label"]).copy()
        data_df["label"] = data_df["label"].astype(int)
        return data_df

    def _build_auto_split(self):
        frames_dir = os.path.join(self.data_dir, "frames")
        mp4_dir = os.path.join(self.data_dir, "mp4")
        if self.frame_source == "frames":
            if not os.path.isdir(frames_dir):
                raise FileNotFoundError(
                    f"Automatic Toyota split creation needs frame folders in {frames_dir}."
                )
            names = [
                name
                for name in os.listdir(frames_dir)
                if os.path.isdir(os.path.join(frames_dir, name))
                and not name.startswith(".")
            ]
        elif self.frame_source == "mp4_zip":
            names = self._mp4_zip_index().keys()
        elif os.path.isdir(mp4_dir):
            names = [name[:-4] for name in os.listdir(mp4_dir) if name.endswith(".mp4")]
        else:
            raise FileNotFoundError(
                f"Automatic Toyota split creation needs frame folders in {frames_dir}, "
                f"mp4 files in {mp4_dir}, or --toyota_mp4_zip."
            )
        rows = []
        label_dict = self._label_dict()
        for file_id in sorted(names):
            raw_action = file_id.split("_")[0]
            if raw_action not in label_dict:
                continue
            subject = self._subject_id(file_id)
            if subject is None:
                continue
            rows.append(
                {
                    "file_id": file_id,
                    "raw_action": raw_action,
                    "label": label_dict[raw_action],
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

    def _cache_digest(self, payload):
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        return hashlib.sha256(encoded).hexdigest()[:20]

    def _path_signature(self, path):
        if path is None:
            return None
        if not os.path.exists(path):
            return {"path": os.path.abspath(path), "missing": True}
        stat = os.stat(path)
        return {
            "path": os.path.abspath(path),
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }

    def _jsonable_mapping(self, value):
        if value is None:
            return None
        if isinstance(value, dict):
            return {
                str(key): self._jsonable_mapping(value[key])
                for key in sorted(value.keys(), key=str)
            }
        if isinstance(value, (list, tuple, set)):
            return [self._jsonable_mapping(item) for item in sorted(value, key=str)]
        if isinstance(value, np.generic):
            return value.item()
        return value

    def _cache_path(self, cache_dir, prefix, payload):
        if not cache_dir:
            return None
        os.makedirs(cache_dir, exist_ok=True)
        digest = self._cache_digest(payload)
        return os.path.join(cache_dir, f"{prefix}_{digest}.pt")

    def _object_cache_path(self, file_ids):
        payload = {
            "kind": "toyota_object_cache_v5_actor_object_heatmap_relation_evidence",
            "set_type": self.set_type,
            "source": self._path_signature(self.object_detector_cache),
            "file_ids": sorted(str(file_id) for file_id in file_ids),
            "object_conf_threshold": self.object_conf_threshold,
            "object_camera_allowlist": self._jsonable_mapping(
                self.object_camera_allowlist
            ),
            "object_ignore_regions": self._jsonable_mapping(self.object_ignore_regions),
            "num_object_classes": self.num_object_classes,
            "interaction_heatmap_channels": "per_actor_interacted_object",
        }
        return self._cache_path(self.object_cache_dir, "objects", payload)

    def _landmark_cache_path(self):
        skeleton_source = self.skeleton_zip_path
        skeleton_folder = os.path.join(self.data_dir, "skeleton")
        if not os.path.exists(skeleton_source) and os.path.isdir(skeleton_folder):
            skeleton_source = skeleton_folder
        payload = {
            "kind": "toyota_landmark_cache_v1",
            "set_type": self.set_type,
            "source": self._path_signature(skeleton_source),
            "file_ids": [str(file_id) for file_id in self.data_df.file_id.tolist()],
            "pose_landmarks": self.pose_landmarks,
        }
        return self._cache_path(self.landmark_cache_dir, "landmarks", payload)

    def _load_object_cache(self, file_ids):
        if not self.object_detector_cache:
            raise ValueError(
                "actor interaction/object-proposal training requires "
                "--object_detector_cache"
            )
        if not os.path.exists(self.object_detector_cache):
            raise FileNotFoundError(self.object_detector_cache)

        file_ids = set(file_ids)
        cache_path = self._object_cache_path(file_ids)
        if cache_path and os.path.exists(cache_path):
            cache = torch.load(cache_path, map_location="cpu", weights_only=False)
            print(
                f"Loaded preprocessed Toyota object cache: {cache_path} "
                f"for {len(file_ids)} clips."
            )
            return cache

        cache = {file_id: {} for file_id in file_ids}
        loaded_objects = 0
        with open(self.object_detector_cache) as f:
            for line_idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                file_id = record.get("file_id")
                if file_id not in file_ids:
                    continue
                if "frame_idx" not in record:
                    raise ValueError(
                        f"Object cache line {line_idx} is missing frame_idx"
                    )
                frame_idx = int(record["frame_idx"])
                frame_width = record.get("width")
                frame_height = record.get("height")
                objects = []
                for obj in record.get("objects", []):
                    parsed = self._parse_cache_object(
                        obj,
                        file_id=file_id,
                        width=frame_width,
                        height=frame_height,
                    )
                    if parsed is not None:
                        objects.append(parsed)
                if objects:
                    cache.setdefault(file_id, {}).setdefault(frame_idx, []).extend(
                        objects
                    )
                    loaded_objects += len(objects)
        print(
            f"Loaded {loaded_objects} cached Toyota objects from "
            f"{self.object_detector_cache} for {len(file_ids)} clips."
        )
        if cache_path:
            tmp_path = f"{cache_path}.{os.getpid()}.tmp"
            torch.save(cache, tmp_path)
            os.replace(tmp_path, cache_path)
            print(f"Saved preprocessed Toyota object cache: {cache_path}")
        return cache

    def _parse_cache_object(self, obj, file_id=None, width=None, height=None):
        conf = float(obj.get("conf", 0.0))
        if conf < self.object_conf_threshold:
            return None

        cls_id = None
        for cls_name in (obj.get("detector_cls"), obj.get("cls")):
            object_name = self._object_name_from_name(cls_name)
            if object_name is not None:
                if not object_allowed_for_file_id(
                    object_name,
                    file_id,
                    self.object_camera_allowlist,
                ):
                    return None
                cls_id = int(OBJECT_TO_ID[object_name])
                break
        if cls_id is None:
            return None

        xyxy = np.asarray(obj.get("xyxy", []), dtype=np.float32)
        if xyxy.shape != (4,) or not np.isfinite(xyxy).all():
            return None
        x1, y1, x2, y2 = [float(v) for v in xyxy.tolist()]
        if width is not None and height is not None:
            width = float(width)
            height = float(height)
            if width <= 0 or height <= 0:
                return None
            x1 = max(0.0, min(width, x1))
            y1 = max(0.0, min(height, y1))
            x2 = max(0.0, min(width, x2))
            y2 = max(0.0, min(height, y2))
        if x2 <= x1 or y2 <= y1:
            return None
        if object_box_ignored_for_file_id(
            (x1, y1, x2, y2),
            file_id,
            width,
            height,
            self.object_ignore_regions,
        ):
            return None
        return {
            "cls_id": int(cls_id),
            "conf": conf,
            "xyxy": np.asarray([x1, y1, x2, y2], dtype=np.float32),
        }

    def _object_class_id_from_name(self, cls_name):
        object_name = self._object_name_from_name(cls_name)
        if object_name is None:
            return None
        return int(OBJECT_TO_ID[object_name])

    def _object_name_from_name(self, cls_name):
        if cls_name is None:
            return None
        cls_name = str(cls_name).strip().lower()
        object_name = DETECTOR_TO_OBJECT.get(cls_name, cls_name)
        if object_name not in OBJECT_TO_ID:
            return None
        return object_name

    def _get_frame_objects(self, file_id, frame_idx):
        return self._object_cache.get(file_id, {}).get(int(frame_idx), [])

    def _sample_object_entries(self, file_id, frames_idx, action_name=None):
        entries = []
        for sample_pos, frame_idx in enumerate(frames_idx):
            frame_idx = int(frame_idx)
            frame_objects = list(self._get_frame_objects(file_id, frame_idx))
            for obj in frame_objects:
                entries.append(
                    {
                        "sample_pos": int(sample_pos),
                        "cls_id": int(obj["cls_id"]),
                        "conf": float(obj["conf"]),
                        "xyxy": np.asarray(obj["xyxy"], dtype=np.float32).copy(),
                    }
                )
        return entries

    def _object_entries_to_keypoints(self, object_entries, n_sampled_frames):
        if not object_entries:
            return None
        points = np.zeros(
            (int(n_sampled_frames), len(object_entries) * 2, 2),
            dtype=np.float32,
        )
        for entry_idx, entry in enumerate(object_entries):
            point_idx = entry_idx * 2
            frame_pos = int(entry["sample_pos"])
            x1, y1, x2, y2 = entry["xyxy"].tolist()
            points[frame_pos, point_idx] = [x1, y1]
            points[frame_pos, point_idx + 1] = [x2, y2]
            entry["point_idx"] = point_idx
        return [points]

    def _transformed_object_entries(self, object_entries, object_keypoints, height, width):
        if not object_entries:
            return []
        if object_keypoints is None:
            raise RuntimeError("Object entries were created but object keypoints are missing")

        points = np.asarray(object_keypoints[0], dtype=np.float32)
        transformed = []
        for entry in object_entries:
            frame_pos = int(entry["sample_pos"])
            point_idx = int(entry["point_idx"])
            p1 = points[frame_pos, point_idx]
            p2 = points[frame_pos, point_idx + 1]
            if not np.isfinite(p1).all() or not np.isfinite(p2).all():
                continue
            x1 = float(min(p1[0], p2[0]))
            y1 = float(min(p1[1], p2[1]))
            x2 = float(max(p1[0], p2[0]))
            y2 = float(max(p1[1], p2[1]))
            x1 = float(np.clip(x1, 0.0, width - 1.0))
            y1 = float(np.clip(y1, 0.0, height - 1.0))
            x2 = float(np.clip(x2, 0.0, width - 1.0))
            y2 = float(np.clip(y2, 0.0, height - 1.0))
            if x2 <= x1 or y2 <= y1:
                continue
            transformed.append(
                {
                    "sample_pos": frame_pos,
                    "cls_id": int(entry["cls_id"]),
                    "conf": float(entry["conf"]),
                    "xyxy": np.asarray([x1, y1, x2, y2], dtype=np.float32),
                }
            )
        return transformed

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

    def _video_size(self, file_id):
        if file_id in self._video_size_cache:
            return self._video_size_cache[file_id]

        if self.frame_source == "frames":
            frame_folder = os.path.join(self.data_dir, "frames", file_id)
            frame_names = sorted(
                name
                for name in os.listdir(frame_folder)
                if name.lower().endswith((".jpg", ".jpeg", ".png"))
            )
            if not frame_names:
                raise RuntimeError(f"No frames found in {frame_folder}")
            frame = torchvision.io.read_image(os.path.join(frame_folder, frame_names[0]))
            size = (int(frame.shape[1]), int(frame.shape[2]))
            self._video_size_cache[file_id] = size
            return size

        video_path = self._video_path(file_id)
        size = self._video_size_from_file(video_path)
        self._video_size_cache[file_id] = size
        return size

    def _video_size_from_file(self, video_path):
        try:
            import cv2

            cap = cv2.VideoCapture(video_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            if width > 0 and height > 0:
                return height, width
        except Exception:
            pass

        if av is None:
            raise ImportError("PyAV is required to inspect Toyota mp4 size.")
        with av.open(video_path) as container:
            stream = container.streams.video[0]
            width = int(stream.codec_context.width or stream.width or 0)
            height = int(stream.codec_context.height or stream.height or 0)
            if width <= 0 or height <= 0:
                frame = next(container.decode(stream), None)
                if frame is None:
                    raise RuntimeError(f"No frames decoded from {video_path}")
                width = int(frame.width)
                height = int(frame.height)
            return height, width

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
        frame_files = sorted(
            name
            for name in os.listdir(frame_folder)
            if name.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        if not frame_files:
            raise RuntimeError(f"No image frames found in {frame_folder}")

        if frames_idx is None:
            frame_indices = np.arange(len(frame_files), dtype=np.int64)
        else:
            frame_indices = np.asarray(frames_idx, dtype=np.int64)
            frame_indices = np.clip(frame_indices, 0, len(frame_files) - 1)

        frames = []
        for frame_idx in frame_indices:
            frame_path = os.path.join(frame_folder, frame_files[int(frame_idx)])
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
