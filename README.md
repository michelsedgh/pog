# Pose-guided token selection for the recognition of activities of daily living

This is the official implementation of the paper:

**Pose-guided token selection for the recognition of activities of daily living**  
Ricardo Pizarro, Roberto Valle, José M. Buenaposada, Luis M. Bergasa, Luis Baumela  
*Image and Vision Computing*, 2025  
[Paper Link](https://www.sciencedirect.com/science/article/pii/S0262885625002744)

**Note:** This is a limited release. Currently, weights for the Toyota Smarthome Cross-Subject (CS) protocol and DriveAct are available.

## Docker Usage

To run the code using the Docker image, you need to mount your dataset and checkpoints directories.

```bash
docker run --gpus 0 -it -v /path/to/your/datasets:/datasets poguise sh
```

Ensure that your dataset structure inside the container matches what the scripts expect (e.g., `/datasets/toyotasm`).

## Dataset Preparation

Before testing, you need to prepare the dataset by extracting frames and generating label files.

### Toyota Smarthome

#### 1. Extract Frames

Use the `utils/video_to_frames.py` script to extract frames from the video files. This script expects the videos to be located in `/datasets/toyotasm/mp4/` and will output frames to `/datasets/toyotasm/frames/`.

```bash
python utils/video_to_frames.py
```

#### 2. Generate Test Labels

Use the `utils/preproc_toyota_labels.py` script to generate the test labels CSV file. This script requires the frames to be already extracted as it counts the number of frames per video.

```bash
python utils/preproc_toyota_labels.py --root /datasets/toyotasm --protocol CS
```

Replace `CS` with `CV` for the Cross-View protocol if needed.

### DriveAct

For DriveAct, videos must be extracted into frames.
Landmarks are required and can be downloaded from [here](https://universidaddealcala-my.sharepoint.com/:u:/g/personal/ricardo_pizarroc_edu_uah_es/IQDKAt1U4UfyS7ssdC_PwaKyAZ05QALA84befXL70VHx4k4?e=b6kk0f).

## Testing

To test the model on Toyota Smarthome, use the following command:

```bash
python test.py --model_file "poguise_c2hntf6v_epoch=51-val_loss=0.507.ckpt" --data_dir "/datasets/toyotasm" --dataset toyotasm
```

To test the model on DriveAct, use the following command:

```bash
python test.py --model_file "poguise_lazn9q41_epoch=200-val_loss=0.519.ckpt" --data_dir "/datasets/driveact" --dataset driveact --n_frames_stride -3 --test_num_segment 3
```

## Weights

### Toyota Smarthome Cross-Subject (CS) protocol
Link: [poguise_c2hntf6v_epoch=51-val_loss=0.507.ckpt](https://universidaddealcala-my.sharepoint.com/:u:/g/personal/ricardo_pizarroc_edu_uah_es/IQDZ86hAZSr3QpGbCqoAzd5xAU3B6Aup_XPLxnY-cJQfWTw?e=GpduYJ)

### DriveAct Fold 0
Filename: [poguise_lazn9q41_epoch=200-val_loss=0.519.ckpt](https://universidaddealcala-my.sharepoint.com/:u:/g/personal/ricardo_pizarroc_edu_uah_es/IQBM6zi0llQwTpo5i6fiqUs6AXpA_j2i2coOsTnx1TZ3fII?e=HMRPtO)
