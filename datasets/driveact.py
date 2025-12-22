# %%
import os
import pickle
import time
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision
from mmpose.codecs import UDPHeatmap
from torch.utils.data import Dataset

from utils.ntu import frame_utils as utils

DRIVEACT_DCT = {
    "closing_bottle": 0,
    "closing_door_inside": 1,
    "closing_door_outside": 2,
    "closing_laptop": 3,
    "drinking": 4,
    "eating": 5,
    "entering_car": 6,
    "exiting_car": 7,
    "fastening_seat_belt": 8,
    "fetching_an_object": 9,
    "interacting_with_phone": 10,
    "looking_or_moving_around (e.g. searching)": 11,
    "opening_backpack": 12,
    "opening_bottle": 13,
    "opening_door_inside": 14,
    "opening_door_outside": 15,
    "opening_laptop": 16,
    "placing_an_object": 17,
    "preparing_food": 18,
    "pressing_automation_button": 19,
    "putting_laptop_into_backpack": 20,
    "putting_on_jacket": 21,
    "putting_on_sunglasses": 22,
    "reading_magazine": 23,
    "reading_newspaper": 24,
    "sitting_still": 25,
    "taking_laptop_from_backpack": 26,
    "taking_off_jacket": 27,
    "taking_off_sunglasses": 28,
    "talking_on_phone": 29,
    "unfastening_seat_belt": 30,
    "using_multimedia_display": 31,
    "working_on_laptop": 32,
    "writing": 33,
}


class DriveActDataset(Dataset):
    def __init__(
        self,
        data_dir,
        set_type="train",
        task_type="midlevel",
        modal="inner_mirror",
        fold="1",
        n_frames=16,
        n_landmarks=0,
        vis=0.1,
        mean_heatmap=False,
        **kwargs,
    ):
        """
        Args:
            root_dir (string): Directory with all the images.
            set_type (string): train, val, test
            task_type (string): midlevel, objectlevel, tasklevel
            modal (string): kinect_color, kinect_depth, kinect_ir, inner_mirror, a_column_co_driver, a_column_driver, ceiling,
                            steering_wheel
        """
        self.data_dir = data_dir
        self.set_type = set_type
        self.task_type = task_type
        self.modal = modal
        self.n_frames = n_frames
        self.n_frames_stride = kwargs.get("n_frames_stride", 1)
        self.n_landmarks = n_landmarks
        self.h = 270
        self.w = 480
        self.heatmap_size = (56, 56)
        self.h_adjust = self.h
        self.w_adjust = self.w
        self.vis = vis
        self.mean_heatmap = mean_heatmap
        self.jitter_scales_min = kwargs["jitter_scales_min"]
        self.jitter_scales_max = kwargs["jitter_scales_max"]
        self.num_clips = kwargs["num_clips"]
        self.random_clip_train = getattr(kwargs, "random_clip_train", 1)
        self.mean = torch.tensor([0.485, 0.456, 0.406])  # videomae normalization
        self.std = torch.tensor([0.229, 0.224, 0.225])
        self.num_retries = 4
        if self.n_landmarks:
            self.heatmap_generator = UDPHeatmap(
                input_size=(224, 224), heatmap_size=self.heatmap_size, sigma=1
            )

        split = set_type if set_type != "predict" else "test"
        self.data_df = pd.read_csv(
            os.path.join(
                data_dir,
                "activities_3s",
                modal,
                task_type + ".chunks_90.split_" + fold + "." + split + ".csv",
            )
        )
        self.data_df["activity"] = self.data_df["activity"].map(DRIVEACT_DCT)
        self.y = torch.tensor(self.data_df.activity.values, dtype=torch.long)
        if self.set_type == "test":
            self.test_num_segment = kwargs["test_num_segment"]
            self.test_num_crop = kwargs["test_num_crop"]
            self.test_seg = []
            self.test_dataset = []
            self.test_label_array = []
            for ck in range(self.test_num_segment):
                for cp in range(self.test_num_crop):
                    for idx in range(len(self.y)):
                        sample_label = self.y[idx]
                        self.test_label_array.append(sample_label)
                        self.test_dataset.append(self.data_df.iloc[idx])
                        self.test_seg.append((ck, cp))
            print(
                "test dataset",
                len(self.test_dataset),
                len(self.test_label_array),
                self.test_num_crop,
                self.test_num_segment,
            )
            self.length = len(self.test_dataset)
        else:
            self.length = len(self.data_df)
        print(
            self.length,
            set_type,
            "fold",
            fold,
            "num classes",
            len(torch.unique(self.y)),
            "num samples per class",
            torch.unique(self.y, return_counts=True),
            "task_type",
            task_type,
            "modal",
            modal,
            "n_frames",
            n_frames,
            "n_frames_stride",
            self.n_frames_stride,
        )

    def setup(self, stage=None):
        if self.n_landmarks:
            self.landmark_list = []
            # read landmarks into memory
            start = time.time()
            file_folder = "_landmarks_mmpose"
            self.h_adjust = 1
            self.w_adjust = 1

            length = len(self.data_df)
            for i in range(length):
                row = self.data_df.iloc[i]
                frame_indices = list(range(row.frame_start, row.frame_end + 1))
                file_name = (
                    f"{frame_indices[0]}_{frame_indices[-1]}_landmarks_mmpose.pkl"
                )
                with open(
                    os.path.join(
                        self.data_dir,
                        self.modal + file_folder,
                        row.file_id.split(".")[0],
                        file_name,
                    ),
                    "rb",
                ) as f:
                    landmarks_file = pickle.load(f)
                    landmarks_file = landmarks_file[:, :19, :]
                    # remove wrists (indices 15 and 16)
                    landmarks_file = np.delete(landmarks_file, [15, 16], axis=1)
                    landmarks_file = landmarks_file.to(torch.float)
                    # get mean of first 3 landmarks (nose and eyes) to represent head center
                    average_landmark = torch.mean(landmarks_file[:, :3, :], axis=1)
                    # stack average landmark with remaining landmarks
                    landmarks_file = torch.cat(
                        (average_landmark.unsqueeze(1), landmarks_file[:, 3:, :]),
                        axis=1,
                    )
                    scale_x = 224 / self.w
                    scale_y = 224 / self.h
                    conf = landmarks_file[:, :, 2]
                    landmarks_file = landmarks_file[:, :, :2]
                    landmarks_file = landmarks_file * np.array([scale_x, scale_y])
                    # round the scaled x and y coordinates to the nearest integer
                    landmarks_file = torch.round(landmarks_file).to(torch.int)
                    landmarks_file = torch.cat(
                        (landmarks_file, conf.unsqueeze(2)), axis=2
                    )

                    # landmarks_dict[i] = landmarks_file
                    self.landmark_list.append(landmarks_file)
            print("eager loading landmarks", "time", time.time() - start)
            # iterate over frames and landmarks in memory

    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--data_dir", type=str, default="./data/driveact")
        parser.add_argument("--task_type", type=str, default="midlevel")
        parser.add_argument("--modal", type=str, default="inner_mirror")
        parser.add_argument("--mean_heatmap", type=int, default=1)
        parser.add_argument("--num_classes", type=int, default=34)
        parser.add_argument("--n_landmarks", type=int, default=13)  # 13 driveact
        parser.add_argument("--fold", type=str, default="0")
        parser.add_argument("--vis", type=float, default=0.0)
        parser.add_argument("--jitter_scales_min", type=int, default=224)
        parser.add_argument("--jitter_scales_max", type=int, default=270)
        parser.add_argument("--num_clips", type=int, default=1)
        parser.add_argument("--random_clip_train", type=int, default=1)
        return parser

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.set_type == "test":
            row = self.test_dataset[idx]
            chunk_nb, split_nb = self.test_seg[idx]
            frame_indices = list(range(row.frame_start, row.frame_end + 1))
            if len(frame_indices) < self.n_frames * self.num_clips:
                frame_indices = np.pad(
                    frame_indices,
                    (0, self.num_clips * self.n_frames - len(frame_indices)),
                    "edge",
                )
            frame_clip_indices = np.array_split(frame_indices, self.num_clips)
            clips = []
            clips_labels = []
            for i in range(self.num_clips):
                frame_indices = frame_clip_indices[i]
                if self.modal == "inner_mirror":
                    video_name = os.path.join(
                        self.data_dir,
                        self.modal + "_frames",
                        row.file_id.split(".")[0],
                    )
                    buffer = self.read_all_frames(
                        video_name,
                        frame_indices,
                    )
                    # print(buffer.shape,row.frame_start, row.frame_end)
                else:
                    # read from a .mp4 file
                    video_name = os.path.join(
                        self.data_dir,
                        self.modal,
                        row.file_id + ".mp4",
                    )
                    start_sec = row.frame_start / 30
                    end_sec = (1 + row.frame_end) / 30
                    buffer = torchvision.io.read_video(
                        video_name, start_pts=start_sec, end_pts=end_sec, pts_unit="sec"
                    )[0]
                    buffer = buffer.permute(0, 3, 1, 2)
                    # rescale to 270 x 480
                    # buffer = torch.nn.functional.interpolate(buffer, (270, 480), mode='bilinear')
                    # print(buffer.shape,row.frame_start, row.frame_end)

                video_name += f"_{frame_indices[0]}_{frame_indices[-1]}"
                if self.n_landmarks > 0:
                    idx_lnd = idx % len(self.data_df)

                if self.n_frames_stride <= -1:
                    if self.n_frames_stride == -1:
                        # evenly sample 16 frames from buffer
                        frame_indices = np.linspace(
                            0, buffer.shape[0] - 1, self.n_frames, dtype=int
                        )
                    elif self.n_frames_stride == -2:
                        # randomly sample 16 frames from buffer
                        frame_indices = np.random.choice(
                            np.arange(0, buffer.shape[0]),
                            self.n_frames,
                            replace=False,
                        )
                        frame_indices = np.sort(frame_indices)
                    elif self.n_frames_stride == -3:
                        # evenly sample 16 frames from buffer taking into account temporal_step if self.test_num_segment > 1
                        if self.test_num_segment > 1:
                            temporal_step = max(
                                1.0
                                * (buffer.shape[0] - self.n_frames)
                                / (self.test_num_segment - 1),
                                0,
                            )
                        else:
                            temporal_step = 0
                        temporal_start = int(chunk_nb * temporal_step)
                        temporal_end = int(temporal_start + self.n_frames * 2)
                        buffer = buffer[temporal_start:temporal_end,]
                        frame_indices = np.linspace(
                            0, buffer.shape[0] - 1, self.n_frames, dtype=int
                        )
                    elif self.n_frames_stride == -4:
                        # randomly sample 16 frames from buffer taking into account temporal_step if self.test_num_segment > 1
                        if self.test_num_segment > 1:
                            temporal_step = max(
                                1.0
                                * (buffer.shape[0] - self.n_frames)
                                / (self.test_num_segment - 1),
                                0,
                            )
                        else:
                            temporal_step = 0
                        temporal_start = int(chunk_nb * temporal_step)
                        temporal_end = int(temporal_start + self.n_frames * 2)
                        buffer = buffer[temporal_start:temporal_end,]
                        frame_indices = np.random.choice(
                            np.arange(0, buffer.shape[0]),
                            self.n_frames,
                            replace=False,
                        )

                    buffer = buffer[frame_indices]
                    if self.n_landmarks:
                        lnd = self.landmark_list[idx_lnd]
                        if lnd.shape[0] < self.n_frames:
                            # pad the buffer
                            lnd = torch.cat(
                                (
                                    lnd,
                                    lnd[-1]
                                    .unsqueeze(0)
                                    .repeat(self.n_frames - lnd.shape[0], 1, 1),
                                ),
                                dim=0,
                            )
                        lnd = lnd[frame_indices, :, :2].clone().detach()
                else:
                    if self.test_num_segment > 1:
                        temporal_step = max(
                            1.0
                            * (buffer.shape[0] - self.n_frames)
                            / (self.test_num_segment - 1),
                            0,
                        )
                    else:
                        temporal_step = 0

                    if self.test_num_segment > 1:
                        temporal_start = int(chunk_nb * temporal_step)
                    else:
                        # center crop
                        temporal_start = int((buffer.shape[0] - self.n_frames) / 2)

                    # check if the temporal stride fits in the buffer
                    if (
                        temporal_start + self.n_frames * self.n_frames_stride
                        <= buffer.shape[0]
                    ):
                        buffer = buffer[
                            temporal_start : temporal_start
                            + self.n_frames
                            * self.n_frames_stride : self.n_frames_stride,
                            :,
                            :,
                            :,
                        ]

                        if self.n_landmarks:
                            indices = list(
                                range(
                                    temporal_start,
                                    temporal_start
                                    + self.n_frames * self.n_frames_stride,
                                    self.n_frames_stride,
                                )
                            )
                            lnd = self.landmark_list[idx_lnd]
                            if lnd.shape[0] < self.n_frames:
                                # pad the buffer
                                lnd = torch.cat(
                                    (
                                        lnd,
                                        lnd[-1]
                                        .unsqueeze(0)
                                        .repeat(self.n_frames - lnd.shape[0], 1, 1),
                                    ),
                                    dim=0,
                                )
                            lnd = lnd[indices, :, :2].clone().detach()

                    else:
                        buffer = buffer[
                            temporal_start : temporal_start + self.n_frames,
                            :,
                            :,
                            :,
                        ]
                        if self.n_landmarks:
                            lnd = (
                                self.landmark_list[idx_lnd][
                                    temporal_start : temporal_start + self.n_frames,
                                    :,
                                    :2,
                                ]
                                .clone()
                                .detach()
                            )

                        if buffer.shape[0] != self.n_frames:
                            # pad the buffer
                            buffer = torch.cat(
                                (
                                    buffer,
                                    buffer[-1]
                                    .unsqueeze(0)
                                    .repeat(self.n_frames - buffer.shape[0], 1, 1, 1),
                                ),
                                dim=0,
                            )
                            if self.n_landmarks:
                                lnd = torch.cat(
                                    (
                                        lnd,
                                        lnd[-1]
                                        .unsqueeze(0)
                                        .repeat(self.n_frames - lnd.shape[0], 1, 1),
                                    ),
                                    dim=0,
                                )
                # T C H W -> T H W C.
                buffer = buffer.permute(0, 2, 3, 1)
                buffer = utils.tensor_normalize(buffer, mean=self.mean, std=self.std)
                # T H W C -> C T H W
                buffer = buffer.permute(3, 0, 1, 2)
                # if self.set_type != "train":
                # resize to 224, 224
                buffer = F.interpolate(buffer, size=(224, 224), mode="bilinear")

                # buffer, _ = utils.spatial_sampling(
                #     buffer,
                #     spatial_idx=1 if self.test_num_crop == 1 else split_nb,
                #     min_scale=224 if self.test_num_crop > 1 else self.jitter_scales_min,
                #     max_scale=224 if self.test_num_crop > 1 else self.jitter_scales_min,
                #     crop_size=224,
                #     random_horizontal_flip=False,
                #     inverse_uniform_sampling=False,
                # )
                buffer = buffer.permute(1, 0, 2, 3)
                clips.append(buffer)
                if self.n_landmarks:
                    keypoints = lnd
                    lnd_heatmap = torch.zeros(
                        buffer.shape[0],
                        self.n_landmarks,
                        *self.heatmap_size,
                    )
                    kp_vis = torch.zeros(
                        self.n_landmarks,
                        *self.heatmap_size,
                    )

                    for frame_idx in range(buffer.shape[0]):
                        kp_frame = keypoints[frame_idx]
                        kp_frame = np.expand_dims(kp_frame, axis=0)
                        # make negative values nan
                        kp_frame[kp_frame < 0] = np.nan
                        vis = np.ones(kp_frame.shape[1])
                        # Check if any value in kp_frame is out of bounds
                        out_of_bounds = (kp_frame[:, :, 0] > buffer.shape[2]) | (
                            kp_frame[:, :, 1] > buffer.shape[3]
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

                    if self.mean_heatmap == 2:
                        # calculate average heatmap over all time frames
                        lnd_heatmap = torch.sum(lnd_heatmap, dim=0)
                    elif self.mean_heatmap == 1:
                        lnd_heatmap = torch.mean(lnd_heatmap, dim=0)
                    return (
                        buffer,
                        self.test_label_array[idx],
                        video_name,
                        chunk_nb,
                        split_nb,
                        lnd_heatmap,
                    )
                # return buffer, self.test_label_array[idx], video_name, chunk_nb, split_nb
            if self.num_clips > 1:
                return clips, self.test_label_array[idx], video_name, chunk_nb, split_nb
            else:
                return (
                    clips[0],
                    self.test_label_array[idx],
                    video_name,
                    chunk_nb,
                    split_nb,
                )
        else:
            row = self.data_df.iloc[idx]
            label = self.y[idx]

            frame_indices = list(range(row.frame_start, row.frame_end + 1))
            if len(frame_indices) < self.n_frames * self.num_clips:
                frame_indices = np.pad(
                    frame_indices,
                    (0, self.num_clips * self.n_frames - len(frame_indices)),
                    "edge",
                )
            frame_clip_indices = np.array_split(frame_indices, self.num_clips)
            clips = []
            clips_labels = []
            for i in range(self.num_clips):
                frame_indices = frame_clip_indices[i]
                if self.n_landmarks:
                    for i_try in range(self.num_retries):
                        frames, keypoints, keypoints_visible = (
                            self.read_frame_landmarks_folder(
                                os.path.join(
                                    self.data_dir,
                                    self.modal + "_frames",
                                    row.file_id.split(".")[0],
                                ),
                                frame_indices,
                                idx,
                                r_sample=(
                                    True
                                    if self.num_clips == 1
                                    and self.set_type == "train"
                                    and self.random_clip_train
                                    else False
                                ),
                            )
                        )
                        keypoints = keypoints.numpy()
                        # scale keypoints from 224, 224 to 480, 270
                        scale_x = 480 / 224
                        scale_y = 270 / 224
                        keypoints = keypoints * np.array([scale_x, scale_y])

                        # read img folder
                        frames = frames.permute(0, 2, 3, 1)
                        frames = utils.tensor_normalize(
                            frames, mean=self.mean, std=self.std
                        )
                        # T H W C -> C T H W
                        frames = frames.permute(3, 0, 1, 2)
                        # with 50% probability resize to 224, 224
                        if np.random.rand() > 0.5 or self.set_type != "train":
                            frames = F.interpolate(
                                frames, size=(224, 224), mode="bilinear"
                            )
                            # scale keypoints to 224, 224
                            scale_x = 224 / 480
                            scale_y = 224 / 270
                            keypoints = keypoints * np.array([scale_x, scale_y])

                        data_res = utils.spatial_sampling(
                            frames,
                            spatial_idx=-1 if self.set_type == "train" else 1,
                            min_scale=(
                                self.jitter_scales_min
                                if self.set_type == "train"
                                else 224
                            ),
                            max_scale=(
                                self.jitter_scales_max
                                if self.set_type == "train"
                                else 224
                            ),
                            crop_size=224,
                            random_horizontal_flip=(
                                True if self.set_type == "train" else False
                            ),
                            inverse_uniform_sampling=False,
                            keypoints=[
                                keypoints,
                            ],
                        )
                        frames, keypoints = data_res
                        frames = frames.permute(1, 0, 2, 3)
                        lnd_heatmap = torch.zeros(
                            frames.shape[0],
                            self.n_landmarks,
                            *self.heatmap_size,
                        )
                        kp_vis = torch.zeros(
                            self.n_landmarks,
                            *self.heatmap_size,
                        )
                        for person_idx in range(len(keypoints)):
                            for frame_idx in range(frames.shape[0]):
                                kp_frame = keypoints[person_idx][frame_idx]
                                kp_frame = np.expand_dims(kp_frame, axis=0)
                                # make negative values nan
                                kp_frame[kp_frame < 0] = np.nan
                                vis = np.ones(kp_frame.shape[1])
                                # Check if any value in kp_frame is out of bounds
                                out_of_bounds = (
                                    kp_frame[:, :, 0] > frames.shape[2]
                                ) | (kp_frame[:, :, 1] > frames.shape[3])
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
                        if (
                            num_zeros > 8
                            and label not in [2, 15, 6]
                            and i_try < self.num_retries - 1
                        ):
                            continue
                        if self.mean_heatmap == 2:
                            # calculate average heatmap over all time frames
                            lnd_heatmap = torch.sum(lnd_heatmap, dim=0)
                        elif self.mean_heatmap == 1:
                            lnd_heatmap = torch.mean(lnd_heatmap, dim=0)

                        clip_label = [
                            label,
                            lnd_heatmap,
                            kp_vis,
                        ]
                        clips.append(frames)
                        clips_labels.append(clip_label)
                        break
                else:
                    buffer = self.read_frame_folder(
                        os.path.join(
                            self.data_dir,
                            self.modal + "_frames",
                            row.file_id.split(".")[0],
                        ),
                        frame_indices,
                        r_sample=True if self.num_clips == 1 else False,
                    )
                    # read img folder
                    buffer = buffer.permute(0, 2, 3, 1)
                    buffer = utils.tensor_normalize(
                        buffer, mean=self.mean, std=self.std
                    )
                    # T H W C -> C T H W
                    buffer = buffer.permute(3, 0, 1, 2)
                    buffer, _ = utils.spatial_sampling(
                        buffer,
                        spatial_idx=-1,
                        min_scale=self.jitter_scales_min,
                        max_scale=self.jitter_scales_max,
                        crop_size=224,
                        random_horizontal_flip=True,
                        inverse_uniform_sampling=False,
                    )
                    buffer = buffer.permute(1, 0, 2, 3)
                    clips.append(buffer)
                    clips_labels.append(label)
            if self.num_clips > 1:
                return clips, clips_labels[0]
            else:
                return clips[0], clips_labels[0]

    def read_all_frames(self, frame_folder, frame_indices):
        if len(frame_indices) < self.n_frames:
            frame_indices = np.pad(
                frame_indices, (0, self.n_frames - len(frame_indices)), "edge"
            )
        frames = []
        for i in frame_indices:
            frame_path = os.path.join(frame_folder, f"frame_{i:05d}.png")
            if not os.path.exists(frame_path):
                frame_path = os.path.join(frame_folder, f"frame_{i:05d}.jpg")
            # use torchvision to read image
            frame = torchvision.io.read_image(frame_path)
            frames.append(frame)
        frames = torch.stack(frames)
        return frames

    def read_frame_folder(self, frame_folder, frame_indices, r_sample=True):
        frame_indices = self.sample_pad_frames(frame_indices, r_sample=r_sample)
        frames = []

        for i in frame_indices:
            frame_path = os.path.join(frame_folder, f"frame_{i:05d}.png")
            if not os.path.exists(frame_path):
                frame_path = os.path.join(frame_folder, f"frame_{i:05d}.jpg")
            # use torchvision to read image
            frame = torchvision.io.read_image(frame_path)
            frames.append(frame)

        frames = torch.stack(frames)

        return frames

    def read_frame_landmarks_folder(
        self, frame_folder, frame_indices, data_idx, r_sample=True
    ):
        frame_indices = self.sample_pad_frames(frame_indices, r_sample=r_sample)
        frames = []
        landmarks = []
        keypoints_visible = []
        for idx, i in enumerate(frame_indices):
            if idx >= len(self.landmark_list[data_idx]):
                idx = len(self.landmark_list[data_idx]) - 1
            frame_path = os.path.join(frame_folder, f"frame_{i:05d}.png")
            if not os.path.exists(frame_path):
                frame_path = os.path.join(frame_folder, f"frame_{i:05d}.jpg")
            # use torchvision to read image
            frame = torchvision.io.read_image(frame_path)

            lnd = self.landmark_list[data_idx][idx][:, :2]
            vis = self.landmark_list[data_idx][idx][:, 2]
            if lnd is None:
                # create landmark tensor by sampling from the middle of the image where each landmark is 16x16 and following one next to the other
                lnd = torch.zeros((self.n_landmarks, 2))
            landmark_list = []
            vis_list = []
            for i, landmark in enumerate(lnd):
                # if all zeros skip
                if (landmark == 0).all():
                    landmark_list.append(landmark)
                    if self.vis == 0:
                        vis_list.append(1)
                    else:
                        vis_list.append(0)
                    continue
                # landmark format x_min, y_min, x_max, y_max
                # scale to 224, 224
                # scale from 480, 270 to 224, 224
                # scale_x = 224 / self.w
                # scale_y = 224 / self.h
                # landmark = landmark * np.array([scale_x, scale_y])
                # # round the scaled x and y coordinates to the nearest integer
                # landmark = torch.round(landmark).to(torch.int)

                landmark_list.append(landmark)
                if vis[i] < self.vis:
                    vis_list.append(0)
                else:
                    vis_list.append(1)

            landmark_list = torch.stack(landmark_list)
            vis_list = torch.tensor(vis_list)
            keypoints_visible.append(vis_list)
            landmarks.append(landmark_list)
            frames.append(frame)

        frames = torch.stack(frames)
        landmarks = torch.stack(landmarks)
        keypoints_visible = torch.stack(keypoints_visible).to(int).numpy()

        return frames, landmarks, keypoints_visible

    def sample_pad_frames(self, frame_indices, r_sample=True):
        # evenly sample n frames from a list of frames
        if len(frame_indices) >= self.n_frames:
            start_frame = frame_indices[0]
            end_frame = frame_indices[-1]
            if self.set_type == "train" and r_sample:
                # randomly sample n frames

                frame_indices = np.random.choice(
                    frame_indices, self.n_frames, replace=False
                )

                # sort the frames
                frame_indices = np.sort(frame_indices)
            else:
                # calculate the indices of the sampled frames
                frame_indices = np.linspace(
                    start_frame, end_frame, self.n_frames, dtype=int
                )
        else:
            # pad the frames with the last frame
            frame_indices = np.pad(
                frame_indices, (0, self.n_frames - len(frame_indices)), "edge"
            )
        return frame_indices.tolist()
