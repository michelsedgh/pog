# %%
import os
import argparse
import cv2
import shlex
import mmcv

# print number of cpu
print(os.cpu_count(), "cpus")
# print ram available
print(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024.0**3), "GB RAM")

parser = argparse.ArgumentParser()
parser.add_argument("--root_folder", type=str, default="/datasets/toyotasm/mp4/")
parser.add_argument("--dest_folder", type=str, default="/datasets/toyotasm/frames/")
args = parser.parse_args()

root_folder = args.root_folder
dest_folder = args.dest_folder

# create dest folder if not exists
if not os.path.exists(dest_folder):
    os.makedirs(dest_folder)

# walk through root folder
for dirpath, dirnames, filenames in os.walk(root_folder):
    for filename in filenames:
        # full path to the file
        file_path = os.path.join(dirpath, filename)
        if "mp4" not in file_path:
            continue
        # process the file
        # class_name = file_path.split("/")[-2]
        # dest_folder_frames = os.path.join(
        #     dest_folder, class_name, filename.split(".")[0]
        # )
        # toyota
        class_name = file_path.split("/")[-1][:-4]
        # create class folder if not exists
        dest_folder_frames = os.path.join(dest_folder, class_name)
        print(dest_folder_frames, file_path, class_name)
        if not os.path.exists(dest_folder_frames):
            os.makedirs(dest_folder_frames)
        cap = cv2.VideoCapture(file_path)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        # find if there are already frames
        n_frames_folder = len(os.listdir(dest_folder_frames))
        if n_frames_folder > 0:
            # delete 0 byte files
            os.system(f"find {shlex.quote(dest_folder_frames)} -size 0 -delete")
        n_frames_folder = len(os.listdir(dest_folder_frames))
        if n_frames == n_frames_folder:
            # print("Already converted")
            continue
        print(dest_folder_frames, n_frames, n_frames_folder - n_frames)

        video = mmcv.VideoReader(file_path)
        for i, frame in enumerate(video, start=1):
            mmcv.imwrite(frame, f"{dest_folder_frames}/img_{i:05d}.jpg")
