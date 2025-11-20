# %%
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import random
import torch
import torch.utils.data
from fvcore.common.file_io import PathManager
import sys

# from torchcodec.decoders import VideoDecoder

import numpy as np
from utils.ntu import decoder_ntu as decoder
from utils.ntu import frame_utils as utils

from utils.ntu import pose_utils
from argparse import ArgumentParser
import av
from mmpose.codecs import UDPHeatmap
import pandas as pd
import torchvision


class NTUDataset(torch.utils.data.Dataset):
    """
    NTU video loader. Construct the NTU video loader, then sample
    clips from the videos. For training and validation, a single clip is
    randomly sampled from every video with random cropping, scaling, and
    flipping. For testing, multiple clips are uniformaly sampled from every
    video with uniform cropping. For uniform cropping, we take the left, center,
    and right crop if the width is larger than height, or take top, center, and
    bottom crop if the height is larger than the width.
    """

    def __init__(
        self,
        data_dir,
        set_type,
        test_num_segment=1,
        test_num_crop=1,
        num_retries=10,
        **kwargs,
    ):
        """
        Construct the NTU video loader with a given csv file. The format of
        the csv file is:
        ```
        path_to_video_1 path_to_pose_1 label_1
        path_to_video_2 path_to_pose_2 label_2
        ...
        path_to_video_N path_to_pose_N label_N
        ```
        Args:
            cfg (CfgNode): configs.
            mode (string): Options includes `train`, `val`, or `test` mode.
                For the train and val mode, the data loader will take data
                from the train or val set, and sample one clip per video.
                For the test mode, the data loader will take data from test set,
                and sample multiple clips per video.
            num_retries (int): number of retries.
        """
        # Only support train, val, and test mode.
        assert set_type in [
            "train",
            "val",
            "test",
        ], "Split '{}' not supported for NTU".format(set_type)
        self.mode = set_type
        self._video_meta = {}
        self._num_retries = num_retries
        self.test_num_segment = test_num_segment
        self.test_num_crop = test_num_crop
        self.data_dir = data_dir
        self.kwargs = kwargs
        self.n_frames = kwargs["n_frames"]
        self.n_landmarks = kwargs["n_landmarks"]
        self.heatmap_agg = kwargs["heatmap_agg"]
        self.jitter_scales_min = kwargs["jitter_scales_min"]
        self.jitter_scales_max = kwargs["jitter_scales_max"]
        self.multi_thread_decode = kwargs["multi_thread_decode"]
        self.uniform_sampling = kwargs["uniform_sampling"]
        self.backend_video = kwargs["backend_video"]
        self.num_classes = kwargs["num_classes"]
        self.cross_setup = kwargs["cross_setup"]
        print("Cross setup: ", self.cross_setup)
        self.mean = (
            torch.tensor([0.485, 0.456, 0.406])
            if self.n_landmarks <= 25
            else torch.tensor([1, 1, 1])
        )
        self.std = (
            torch.tensor([0.229, 0.224, 0.225])
            if self.n_landmarks <= 25
            else torch.tensor([1 / 2, 1 / 2, 1 / 2])
        )
        if self.n_landmarks:
            self.heatmap_size = (56, 56)
            self.heatmap_generator = UDPHeatmap(
                input_size=(224, 224), heatmap_size=self.heatmap_size, sigma=1.5
            )

        if self.uniform_sampling:
            print(
                "Uniformly sampling frames for training and testing. This will nulify TEST.NUM_ENSEMBLE_VIEWS and DATA.SAMPLING_RATE"
            )
            self.test_num_segment = 1

        if self.mode in ["train", "val"]:
            self._num_clips = 1
        elif self.mode in ["test"]:
            self._num_clips = self.test_num_segment * self.test_num_crop

    def setup(self, stage=None):
        """
        Construct the video loader.
        """
        print("Constructing NTU {}... Data path: {}".format(self.mode, self.data_dir))
        data_affix = ""
        path_to_file = os.path.join(
            self.data_dir, "{}.csv".format(self.mode + data_affix, self.data_dir)
        )

        assert PathManager.exists(path_to_file), "{} dir not found".format(path_to_file)

        self._path_to_videos = []
        self._path_to_poses = []
        self._labels = []
        self._spatial_temporal_idx = []

        # Load and store all video paths, pose paths, and labels
        with PathManager.open(path_to_file, "r") as f:
            # skip header
            f.readline()
            for clip_idx, path_label in enumerate(f.read().splitlines()):
                data = path_label.split(",")
                path = data[0]
                pose_path = data[1]
                label = data[2]
                for idx in range(self._num_clips):
                    self._path_to_videos.append(os.path.join(self.data_dir, path))
                    self._path_to_poses.append(os.path.join(self.data_dir, pose_path))
                    self._labels.append(int(label) - 1)  # -1 to make it 0-indexed
                    self._spatial_temporal_idx.append(idx)
                    self._video_meta[clip_idx * self._num_clips + idx] = {}
        print(
            "Constructing NTU dataloader (size: {}) from {}".format(
                len(self._path_to_videos), path_to_file
            )
        )
        # convert all lists to numpy arrays
        self._path_to_videos = pd.Series(self._path_to_videos)
        self._path_to_poses = pd.Series(self._path_to_poses)
        self._labels = torch.tensor(self._labels)
        self._spatial_temporal_idx = np.array(self._spatial_temporal_idx).astype(
            np.int32
        )
        data_limit_percent = 0.5
        if self.mode == "train":
            # Limit the training data to 50% of the original data
            num_samples = int(len(self._path_to_videos) * data_limit_percent)
            self._path_to_videos = self._path_to_videos[:num_samples]
            self._path_to_poses = self._path_to_poses[:num_samples]
            self._labels = self._labels[:num_samples]
            self._spatial_temporal_idx = self._spatial_temporal_idx[:num_samples]

    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--data_dir", type=str, default="/datasets/ntu120")
        parser.add_argument("--heatmap_agg", type=int, default=1)
        parser.add_argument("--num_classes", type=int, default=120)
        parser.add_argument("--n_landmarks", type=int, default=0)  #
        parser.add_argument("--vis", type=float, default=0.0)
        parser.add_argument("--jitter_scales_min", type=int, default=256)
        parser.add_argument("--jitter_scales_max", type=int, default=320)
        parser.add_argument("--multi_thread_decode", type=int, default=0)
        parser.add_argument("--uniform_sampling", type=int, default=1)
        parser.add_argument("--backend_video", type=str, default="torch")
        parser.add_argument("--cross_setup", type=int, default=0)

        return parser

    def __getitem__(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index if the video can be fetched and decoded successfully, otherwise
        repeatly find a random video that can be decoded as a replacement.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            keypoint_attention_mask (tensor): the indices of the PatchEmbedded video containing keypoints.
                The dimension is `num_frames * num_patches`
            label (int): the label of the current video.
            index (int): if the video provided by pytorch sampler can be
                decoded, then return the index of the video. If not, return the
                index of the video replacement that can be decoded.
        """
        short_cycle_idx = None
        # When short cycle is used, input index is a tupple.
        if isinstance(index, tuple):
            index, short_cycle_idx = index

        if self.mode in ["train", "val"]:
            # -1 indicates random sampling.
            temporal_sample_index = -1
            spatial_sample_index = -1
            min_scale = self.jitter_scales_min
            max_scale = self.jitter_scales_max
            crop_size = 224
        elif self.mode in ["test"]:
            temporal_sample_index = (
                self._spatial_temporal_idx[index] // self.test_num_crop
            )
            # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
            # center, or right if width is larger than height, and top, middle,
            # or bottom if height is larger than width.
            spatial_sample_index = (
                (self._spatial_temporal_idx[index] % self.test_num_crop)
                if self.test_num_crop > 1
                else 1
            )
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
        sampling_rate = 32
        # Try to decode and sample a clip from a video. If the video can not be
        # decoded, repeatly find a random video replacement that can be decoded.
        for i_try in range(self._num_retries):
            video_container = None
            if self.backend_video == "pyav":
                try:
                    video_container = self._get_video_container(
                        self._path_to_videos[index],
                        self.multi_thread_decode,
                        "pyav",
                    )
                except Exception as e:
                    print(
                        "Failed to load video from {} with error {}".format(
                            self._path_to_videos[index], e
                        )
                    )
                # Select a random video if the current video was not able to access.
                if video_container is None:
                    print(
                        "Failed to meta load video idx {} from {}; trial {}".format(
                            index, self._path_to_videos[index], i_try
                        )
                    )
                    if self.mode not in ["test"] and i_try > self._num_retries // 2:
                        # let's try another one
                        index = random.randint(0, len(self._path_to_videos) - 1)
                    continue

                # Decode video. Meta info is used to perform selective decoding.
                if self.uniform_sampling:
                    # do uniform sampling for train and test
                    temporal_sample_index = -2

                frames, sampled_frames = decoder.decode(
                    video_container,
                    sampling_rate,
                    self.n_frames,
                    temporal_sample_index,
                    self.test_num_segment,
                    video_meta=self._video_meta[index],
                    target_fps=30,
                    backend="pyav",
                    max_spatial_scale=min_scale,
                )
                # Ensure video container is closed
                video_container.close()
                del video_container
            else:
                decoder = VideoDecoder(self._path_to_videos[index], device="cpu")
                # get video length
                sz = decoder.metadata.num_frames
                sampled_frames = (
                    torch.linspace(0, sz - 1, self.n_frames).numpy().astype(int)
                )
                frames = decoder.get_frames_at(indices=sampled_frames).data
                # TCHW -> THWC
                frames = frames.permute(0, 2, 3, 1)
            # If decoding failed (wrong format, video is too short, and etc),
            # select another video.
            if frames is None:
                print(
                    "Failed to decode video idx {} from {}; trial {}".format(
                        index, self._path_to_videos[index], i_try
                    )
                )
                if self.mode not in ["test"] and i_try > self._num_retries // 2:
                    # let's try another one
                    index = random.randint(0, len(self._path_to_videos) - 1)
                continue

            label = self._labels[index]

            # Load keypoints before data augmentation
            keypoints = pose_utils.npy_to_keypoints(self._path_to_poses[index])

            keypoints = [kp[sampled_frames] for kp in keypoints]
            if "output" in self.data_dir:
                # special case for videos with different resolution
                width_scale = 1280 / 1920
                height_scale = 720 / 1080

                scaled_keypoints = []
                for kp in keypoints:
                    kp[:, 0] *= width_scale
                    kp[:, 1] *= height_scale
                    scaled_keypoints.append(kp)

                keypoints = scaled_keypoints

            """
            Data augmentation
            """
            # Perform color normalization.
            frames = utils.tensor_normalize(
                frames, mean=self.mean.to(frames.device), std=self.std.to(frames.device)
            )
            # T H W C -> C T H W.
            frames = frames.permute(3, 0, 1, 2)
            # Perform data augmentation
            frames, keypoints = utils.spatial_sampling(
                frames,
                spatial_idx=spatial_sample_index if self.mode != "val" else 1,
                min_scale=min_scale if self.mode == "train" else min_scale,
                max_scale=max_scale if self.mode == "train" else min_scale,
                crop_size=crop_size,
                random_horizontal_flip=True if self.mode == "train" else False,
                inverse_uniform_sampling=False,
                keypoints=keypoints,
            )
            frames = frames.permute(1, 0, 2, 3)
            if self.mode == "test" and self.n_landmarks == 0:
                return (
                    frames,
                    label,
                    self._path_to_videos[index],
                    temporal_sample_index,
                    spatial_sample_index,
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
                    lnd_heatmap = torch.sum(lnd_heatmap, dim=0)
                elif self.heatmap_agg == 2:
                    lnd_heatmap = torch.mean(lnd_heatmap, dim=0)

                if self.mode == "test":
                    return (
                        frames,
                        label,
                        self._path_to_videos[index],
                        temporal_sample_index,
                        spatial_sample_index,
                        lnd_heatmap,
                    )
                return frames, [label, lnd_heatmap, kp_vis]

            return frames, label

        else:
            raise RuntimeError(
                "Failed to fetch video after {} retries.".format(self._num_retries)
            )

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._path_to_videos)

    def _get_video_container(
        self, path_to_vid, multi_thread_decode=False, backend="pyav"
    ):
        """
        Given the path to the video, return the pyav video container.
        Args:
            path_to_vid (str): path to the video.
            multi_thread_decode (bool): if True, perform multi-thread decoding.
            backend (str): decoder backend, options include `pyav` and
                `torchvision`, default is `pyav`.
        Returns:
            container (container): video container.
        """
        if backend == "torchvision":
            with open(path_to_vid, "rb") as fp:
                container = fp.read()
            return container
        elif backend == "pyav":
            # try:
            container = av.open(path_to_vid)
            if multi_thread_decode:
                # Enable multiple threads for decoding.
                container.streams.video[0].thread_type = "AUTO"
            # except:
            #  container = None
            return container
        else:
            raise NotImplementedError("Unknown backend {}".format(backend))
