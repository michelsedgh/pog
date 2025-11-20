# %%
import os
import argparse
import cv2

# print number of cpu
print(os.cpu_count(), "cpus")
# print ram available
print(
    os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024.0**3), "GB RAM"
)
# get params from command line

# parser = argparse.ArgumentParser()
# parser.add_argument("--folder", type=str, default="a_column_co_driver")
# args = parser.parse_args()
# folders = [
#     args.folder,
# ]
# folders = [
#     "inner_mirror",
#     # "kinect_color",
# ]  # 'a_column_driver', 'ceiling', 'inner_mirror', 'steering_wheel']
# for folder in folders:
#     video_folders = f"/home/ricardo/Documents/datasets/driveact/{folder}/"
#     dest_folder = f"/media/ricardo/data/datasets/driveact/{folder}_frames/"
#     # create dest folder if not exists
#     if not os.path.exists(dest_folder):
#         os.makedirs(dest_folder)
#     # convert each video in folder to frames
#     for folder in os.listdir(video_folders):
#         video_folder = os.path.join(video_folders, folder)
#         for video in os.listdir(video_folder):
#             if video.split(".")[-1].lower() != "mp4":
#                 continue
#             video_path = os.path.join(video_folder, video)
#             dest_path = os.path.join(dest_folder, folder, video.split(".")[0])
#             print(video_path)
#             print(dest_path)
#             if not os.path.exists(dest_path):
#                 os.makedirs(dest_path)
#                 # os.system(f'ffmpeg -i {video_path} {dest_path}/frame_%05d.png')
#                 # downscale to 480x270
#             # get number of frames using opencv
#             cap = cv2.VideoCapture(video_path)
#             n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#             cap.release()
#             print(n_frames)
#             # find if there are already frames
#             n_frames_folder = len(os.listdir(dest_path))
#             if n_frames_folder > 0:
#                 # delete 0 byte files
#                 os.system(f"find {dest_path} -size 0 -delete")
#             n_frames_folder = len(os.listdir(dest_path))
#             if n_frames_folder == n_frames:
#                 print("Already converted")
#                 continue
#             os.system(
#                 f"ffmpeg -n -i {video_path} -vf scale=480:270 {dest_path}/frame_%05d.png"
#             )


# %%

root_folder = "/datasets/toyotasm/mp4/"
dest_folder = "/datasets/toyotasm/frames/"
# create dest folder if not exists
if not os.path.exists(dest_folder):
    os.makedirs(dest_folder)
# create dest folder if not exists
if not os.path.exists(dest_folder):
    os.makedirs(dest_folder)
import shlex
import mmcv

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
        #toyota
        class_name = file_path.split("/")[-1][:-4]
        # create class folder if not exists
        dest_folder_frames = os.path.join(
            dest_folder, class_name
        )
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
            # file_path = shlex.quote(file_path)
        # dest_folder_frames = shlex.quote(dest_folder_frames)
        # res = os.system(f"ffmpeg -n -i {file_path} {dest_folder_frames}/img_%05d.jpg")
        # print(res)
        # if res != 0:
        #     print("Error")
        #     print(file_path, dest_folder_frames)
        #     exit()

        video = mmcv.VideoReader(file_path)
        for i, frame in enumerate(video, start=1):
            mmcv.imwrite(frame, f"{dest_folder_frames}/img_{i:05d}.jpg")
        
