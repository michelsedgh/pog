import argparse
import json
import os
import shutil
import tempfile
import zipfile

import numpy as np
import torch
import torchvision
from PIL import Image, ImageDraw

from utils.ntu import frame_utils as utils


def build_parser():
    parser = argparse.ArgumentParser(
        description="Render Toyota actor-prompt sampling overlays for one clip."
    )
    parser.add_argument("--file_id")
    parser.add_argument("--file_ids", nargs="*")
    parser.add_argument("--toyota_mp4_zip", default=os.getenv("MP4_ZIP"))
    parser.add_argument("--toyota_skeleton_zip", default=os.getenv("SKELETON_ZIP"))
    parser.add_argument(
        "--toyota_video_cache_dir",
        default=os.getenv(
            "VIDEO_CACHE_DIR",
            os.path.join(tempfile.gettempdir(), "poguise_toyota_mp4_cache"),
        ),
    )
    parser.add_argument("--output_dir", default="toyota_actor_visualizations")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_frames", type=int, default=16)
    parser.add_argument("--pose_landmarks", type=int, default=13)
    parser.add_argument("--min_scale", type=int, default=256)
    parser.add_argument("--max_scale", type=int, default=320)
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--box_expand", type=float, default=1.15)
    parser.add_argument("--min_pose_frames", type=int, default=1)
    parser.add_argument("--temporal_start", type=int, default=None)
    parser.add_argument("--no_flip", action="store_true")
    parser.add_argument(
        "--full_video",
        action="store_true",
        help="Render full source-video overlays instead of a sampled training crop.",
    )
    parser.add_argument("--full_video_stride", type=int, default=1)
    parser.add_argument("--max_full_frames", type=int, default=0)
    parser.add_argument(
        "--fps",
        type=float,
        default=0.0,
        help="Output FPS for full-video mode. Uses source FPS when 0.",
    )
    return parser


def extract_video_from_zip(file_id, zip_path, cache_dir):
    if not zip_path or not os.path.exists(zip_path):
        raise FileNotFoundError(f"Toyota mp4 zip not found: {zip_path}")

    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, file_id + ".mp4")
    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
        return cache_path

    with zipfile.ZipFile(zip_path) as zf:
        zip_name = None
        suffix = file_id + ".mp4"
        for name in zf.namelist():
            if name.lower().endswith(".mp4") and os.path.basename(name) == suffix:
                zip_name = name
                break
        if zip_name is None:
            raise FileNotFoundError(f"{suffix} was not found in {zip_path}")

        tmp_path = f"{cache_path}.{os.getpid()}.tmp"
        try:
            with zf.open(zip_name) as src, open(tmp_path, "wb") as dst:
                shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
            os.replace(tmp_path, cache_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    return cache_path


def load_skeleton(file_id, zip_path, pose_landmarks):
    if not zip_path or not os.path.exists(zip_path):
        raise FileNotFoundError(f"Toyota skeleton zip not found: {zip_path}")

    file_name = file_id + "_pose3d.json"
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(file_name) as f:
            data = json.load(f)

    landmarks = []
    for frame in data["frames"]:
        if len(frame) == 0:
            landmarks.append(np.zeros((pose_landmarks, 2), dtype=np.float32))
            continue
        if len(frame) > 1:
            raise ValueError(f"More than one skeleton in {file_name}")
        pose2d = frame[0]["pose2d"]
        xs = pose2d[:pose_landmarks]
        ys = pose2d[pose_landmarks : pose_landmarks * 2]
        landmarks.append(np.round(np.asarray(list(zip(xs, ys)), dtype=np.float32)))

    if not landmarks:
        landmarks.append(np.zeros((pose_landmarks, 2), dtype=np.float32))
    landmarks.append(landmarks[-1].copy())
    return np.stack(landmarks)


def pose_available_by_frame(keypoints, n_frames):
    keypoints = keypoints[:n_frames]
    finite = np.isfinite(keypoints).all(axis=-1)
    non_zero = ~np.all(keypoints == 0, axis=-1)
    return (finite & non_zero).any(axis=1)


def sample_pose_guided_start(n_frames, n_out, pose_available, min_pose_frames, seed):
    if n_frames <= 128:
        return 0

    start_max = max(0, n_frames - 129)
    starts = np.arange(0, start_max + 1, dtype=int)
    hits = np.zeros(starts.shape[0], dtype=int)
    for i, start in enumerate(starts):
        end = min(start + 128, len(pose_available) - 1)
        frame_idx = np.linspace(start, end, n_out, dtype=int)
        frame_idx = np.clip(frame_idx, 0, len(pose_available) - 1)
        hits[i] = int(pose_available[frame_idx].sum())

    candidates = starts[hits >= min_pose_frames]
    if len(candidates) == 0:
        best_hit = int(hits.max())
        if best_hit <= 0:
            return None
        candidates = starts[hits == best_hit]

    rng = np.random.default_rng(seed)
    return int(candidates[rng.integers(0, len(candidates))])


def actor_box_from_keypoints(keypoints, height, width, expand):
    finite = np.isfinite(keypoints).all(axis=-1)
    non_zero = ~np.all(keypoints == 0, axis=-1)
    in_frame = (
        (keypoints[..., 0] >= 0)
        & (keypoints[..., 0] < width)
        & (keypoints[..., 1] >= 0)
        & (keypoints[..., 1] < height)
    )
    visible = finite & non_zero & in_frame
    if not visible.any():
        return None, visible

    points = keypoints[visible]
    x1 = float(points[:, 0].min())
    y1 = float(points[:, 1].min())
    x2 = float(points[:, 0].max())
    y2 = float(points[:, 1].max())
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    bw = max(x2 - x1, 1.0) * expand
    bh = max(y2 - y1, 1.0) * expand
    x1 = max(0.0, cx - bw * 0.5)
    y1 = max(0.0, cy - bh * 0.5)
    x2 = min(float(width), cx + bw * 0.5)
    y2 = min(float(height), cy + bh * 0.5)
    return (x1, y1, x2, y2), visible


def draw_contact_sheet(frames, keypoints, visible, box, frame_idx, output_path):
    frames = frames.permute(1, 2, 3, 0).detach().cpu().numpy()
    frames = np.clip(frames * 255.0, 0, 255).astype(np.uint8)

    tiles = []
    for i, frame in enumerate(frames):
        image = Image.fromarray(frame)
        draw = ImageDraw.Draw(image)
        for x, y in keypoints[i][visible[i]]:
            r = 2
            draw.ellipse((x - r, y - r, x + r, y + r), fill=(0, 255, 80))
        if box is not None:
            draw.rectangle(box, outline=(255, 30, 30), width=3)
        draw.text((6, 6), f"src {int(frame_idx[i])}", fill=(255, 255, 0))
        tiles.append(image)

    cols = 4
    rows = int(np.ceil(len(tiles) / cols))
    width, height = tiles[0].size
    sheet = Image.new("RGB", (cols * width, rows * height), (0, 0, 0))
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % cols) * width, (i // cols) * height))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sheet.save(output_path)


def draw_frame(frame, keypoints, visible, box, text):
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    for x, y in keypoints[visible]:
        r = 3
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(0, 255, 80))
    if box is not None:
        draw.rectangle(box, outline=(255, 30, 30), width=4)
    draw.text((8, 8), text, fill=(255, 255, 0))
    return np.asarray(image)


def render_full_video(args, file_id):
    import cv2

    video_path = extract_video_from_zip(
        file_id, args.toyota_mp4_zip, args.toyota_video_cache_dir
    )
    video, _, info = torchvision.io.read_video(video_path, pts_unit="sec")
    if video.shape[0] == 0:
        raise RuntimeError(f"No frames decoded from {video_path}")

    keypoints = load_skeleton(file_id, args.toyota_skeleton_zip, args.pose_landmarks)
    stride = max(1, int(args.full_video_stride))
    frame_indices = np.arange(0, video.shape[0], stride, dtype=int)
    if args.max_full_frames > 0:
        frame_indices = frame_indices[: args.max_full_frames]

    fps = args.fps or float(info.get("video_fps", 10.0) or 10.0)
    fps = fps / stride if args.fps <= 0 and stride > 1 else fps
    fps = max(1.0, fps)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{file_id}_full_overlay.mp4")
    height, width = int(video.shape[1]), int(video.shape[2])
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output_path}")

    visible_frames = 0
    visible_points = 0
    try:
        for out_i, frame_i in enumerate(frame_indices):
            kp = keypoints[min(int(frame_i), len(keypoints) - 1)]
            box, visible = actor_box_from_keypoints(
                kp,
                height=height,
                width=width,
                expand=args.box_expand,
            )
            visible_frames += int(visible.any())
            visible_points += int(visible.sum())
            frame = video[int(frame_i)].numpy()
            overlay = draw_frame(
                frame,
                kp,
                visible,
                box,
                f"{file_id} src {int(frame_i)}",
            )
            writer.write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()

    print(f"file_id: {file_id}")
    print(f"video_shape: {tuple(video.shape)}")
    print(f"skeleton_frames: {len(keypoints)}")
    print(f"rendered_frames: {len(frame_indices)}")
    print(f"visible_points: {visible_points}")
    print(f"visible_frames: {visible_frames}/{len(frame_indices)}")
    print(f"output: {output_path}")
    return output_path


def render_sample_contact_sheet(args, file_id):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    video_path = extract_video_from_zip(
        file_id, args.toyota_mp4_zip, args.toyota_video_cache_dir
    )
    video, _, _ = torchvision.io.read_video(video_path, pts_unit="sec")
    if video.shape[0] == 0:
        raise RuntimeError(f"No frames decoded from {video_path}")

    keypoints = load_skeleton(
        file_id, args.toyota_skeleton_zip, args.pose_landmarks
    )
    pose_available = pose_available_by_frame(keypoints, video.shape[0])

    start = args.temporal_start
    if start is None:
        start = sample_pose_guided_start(
            video.shape[0],
            args.n_frames,
            pose_available,
            args.min_pose_frames,
            args.seed,
        )
    if start is None:
        raise RuntimeError(f"No pose-guided temporal start found for {file_id}")

    end = min(start + 128, video.shape[0] - 1)
    frame_idx = np.linspace(start, end, args.n_frames, dtype=int)
    frame_idx = np.clip(frame_idx, 0, video.shape[0] - 1)

    frames = video[frame_idx].permute(0, 3, 1, 2).float() / 255.0
    sampled_keypoints = keypoints[np.clip(frame_idx, 0, len(keypoints) - 1)]
    frames = frames.permute(1, 0, 2, 3)
    frames, sampled_keypoints = utils.spatial_sampling(
        frames,
        spatial_idx=-1,
        min_scale=args.min_scale,
        max_scale=args.max_scale,
        crop_size=args.crop_size,
        random_horizontal_flip=not args.no_flip,
        inverse_uniform_sampling=False,
        keypoints=[sampled_keypoints.copy()],
        keypoint_aware_crop=True,
    )
    sampled_keypoints = sampled_keypoints[0]
    box, visible = actor_box_from_keypoints(
        sampled_keypoints,
        height=frames.shape[2],
        width=frames.shape[3],
        expand=args.box_expand,
    )

    output_path = os.path.join(
        args.output_dir,
        f"{file_id}_seed{args.seed}_start{start}.png",
    )
    draw_contact_sheet(frames, sampled_keypoints, visible, box, frame_idx, output_path)

    print(f"file_id: {file_id}")
    print(f"video_shape: {tuple(video.shape)}")
    print(f"skeleton_frames: {len(keypoints)}")
    print(f"temporal_start: {start}")
    print(f"sampled_frames: {frame_idx.tolist()}")
    print(f"visible_points: {int(visible.sum())}")
    print(f"visible_frames: {int(visible.any(axis=1).sum())}/{args.n_frames}")
    print(f"actor_box_xyxy: {box}")
    print(f"output: {output_path}")
    return output_path


def main():
    args = build_parser().parse_args()
    file_ids = []
    if args.file_id:
        file_ids.append(args.file_id)
    if args.file_ids:
        file_ids.extend(args.file_ids)
    if not file_ids:
        raise ValueError("Pass --file_id or --file_ids")

    outputs = []
    for file_id in file_ids:
        if args.full_video:
            outputs.append(render_full_video(args, file_id))
        else:
            outputs.append(render_sample_contact_sheet(args, file_id))

    print("outputs:")
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
