# %%
from torch.utils.data import Dataset
import torch
import os
import pandas as pd
import numpy as np
import torchvision
from mmpose.codecs import UDPHeatmap
from argparse import ArgumentParser
import json
from utils.ntu import frame_utils as utils

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
        self.n_landmarks = kwargs["n_landmarks"]
        self.heatmap_agg = kwargs["heatmap_agg"]
        self.jitter_scales_min = kwargs["jitter_scales_min"]
        self.jitter_scales_max = kwargs["jitter_scales_max"]
        self.test_num_crop = test_num_crop
        self.test_num_segment = test_num_segment
        self.mean = torch.tensor([0.485, 0.456, 0.406])  # videomae normalization
        self.std = torch.tensor([0.229, 0.224, 0.225])
        # self.mean = torch.tensor([1,1,1])
        # self.std = torch.tensor([1,1,1])
        self._num_retries = 5
        if self.n_landmarks:
            self.heatmap_size = (56, 56)
            self.heatmap_generator = UDPHeatmap(
                input_size=(224, 224), heatmap_size=self.heatmap_size, sigma=1.5
            )
        if self.set_type == "test":
            self.data_df = pd.read_csv(
                os.path.join(data_dir, f"test_Labels_{self.task_type}.csv")
            )
            self.data_df.columns = ["file_id", "start", "end"]
            self.data_df["label"] = self.data_df.file_id.apply(
                lambda x: x.split("_")[0]
            )
            self.data_df["label"] = self.data_df.label.map(
                CS_DICT if self.task_type == "CS" else CV_DICT
            )
        else:
            self.data_df = pd.read_csv(
                os.path.join(
                    data_dir,
                    "splits",
                    set_type + f"_{self.task_type}.txt",
                ),
            )
            self.data_df.columns = ["file_id"]
            self.data_df["file_id"] = self.data_df["file_id"].apply(lambda x: x[:-4])
            # Apply the split and retain the first element for each row
            self.data_df["label"] = self.data_df["file_id"].apply(
                lambda x: x.split("_")[0]
            )
            self.data_df["label"] = self.data_df["label"].map(
                CS_DICT if self.task_type == "CS" else CV_DICT
            )
        self.data_df["label"] -= 1
        self.y = torch.tensor(self.data_df.label.values, dtype=torch.long)

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
        if self.n_landmarks:
            self.landmark_list = []
            # read landmarks into memory
            file_folder = "skeleton"

            for i in range(self.length):
                if i % 1000 == 0:
                    print(f"Loading landmarks: {i}/{self.length}")
                row = self.data_df.iloc[i]
                file_name = f"{row.file_id}_pose3d.json"
                # read json
                data = json.load(
                    open(os.path.join(self.data_dir, file_folder, file_name))
                )
                landmarks_file = []
                for frame in data["frames"]:
                    if len(frame) > 1:
                        print(frame, row.file_id)
                        raise ValueError("More than one person in frame")
                    if len(frame) == 0:
                        landmarks_file.append(torch.zeros((self.n_landmarks, 2)))
                        continue
                    landmarks_x = frame[0]["pose2d"][:13]
                    landmarks_y = frame[0]["pose2d"][13:]
                    landmarks = list(zip(landmarks_x, landmarks_y))
                    landmarks = torch.tensor(landmarks)
                    landmarks = torch.round(landmarks).to(torch.int)
                    landmarks_file.append(landmarks)
                # repeat last landmark to match number of frames
                landmarks_file.append(landmarks_file[-1])
                self.landmark_list.append(landmarks_file)
            # iterate over frames and landmarks in memory

    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--data_dir", type=str, default="/datasets/toyotasm")
        parser.add_argument("--heatmap_agg", type=int, default=1)
        parser.add_argument("--num_classes", type=int, default=31)
        parser.add_argument("--n_landmarks", type=int, default=13)
        parser.add_argument("--vis", type=float, default=0.0)
        parser.add_argument("--jitter_scales_min", type=int, default=256)
        parser.add_argument("--jitter_scales_max", type=int, default=320)
        parser.add_argument("--multi_thread_decode", type=int, default=0)
        parser.add_argument("--uniform_sampling", type=int, default=1)
        parser.add_argument("--backend_video", type=str, default="torch")
        parser.add_argument("--task_type", type=str, default="CS")

        return parser

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
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

            n_frames = len(
                os.listdir(
                    os.path.join(
                        self.data_dir, "frames", self.data_df.iloc[idx].file_id
                    )
                )
            )
            label = self.y[idx]
            if self.set_type == "test":
                start_frame = self.data_df.iloc[idx].start
                end_frame = self.data_df.iloc[idx].end
                if end_frame == n_frames:
                    end_frame -= 1
            elif n_frames > 128:  # test has 128 frames segments
                if self.set_type == "train":
                    start_frame = np.random.randint(0, n_frames - 128)
                    end_frame = start_frame + 128
                else:
                    # get the middle 128 frames
                    start_frame = n_frames // 2 - 64
                    end_frame = n_frames // 2 + 64
            else:
                start_frame = 0
                end_frame = n_frames - 1
            # evenly sample n frames from a list of frames

            frames_idx = np.linspace(start_frame, end_frame, self.n_frames, dtype=int)
            if len(frames_idx) < self.n_frames:
                frames_idx = np.pad(
                    frames_idx, (0, self.n_frames - len(frames_idx)), "edge"
                )
            frames = self.read_all_frames(
                os.path.join(self.data_dir, "frames", self.data_df.iloc[idx].file_id),
                frames_idx,
            )
            # frames = frames[frames_idx]
            # convert frames from T, C, H, W to T H W C
            frames = frames.permute(0, 2, 3, 1)
            if self.n_landmarks:
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
            )
            frames = frames.permute(1, 0, 2, 3)
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
                    return (
                        frames,
                        label,
                        self.data_df.iloc[idx].file_id,
                        temporal_sample_index,
                        spatial_sample_index,
                        lnd_heatmap,
                    )
                return frames, [label, lnd_heatmap, kp_vis]
            if self.set_type == "test":
                return (
                    frames,
                    label,
                    self.data_df.iloc[idx].file_id,
                    temporal_sample_index,
                    spatial_sample_index,
                )
            return frames, label

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
