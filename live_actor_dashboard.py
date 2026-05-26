import argparse
import html
import json
import os
import socket
import threading
import time
import warnings
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from utils.actor_tensorrt import TensorRTActorEngine


ACTION_CLASSES = [
    "Cook.Cleandishes",
    "Cook.Cleanup",
    "Cook.Cut",
    "Cook.Stir",
    "Cook.Usestove",
    "Cutbread",
    "Drink.Frombottle",
    "Drink.Fromcan",
    "Drink.Fromcup",
    "Drink.Fromglass",
    "Eat.Attable",
    "Eat.Snack",
    "Enter",
    "Getup",
    "Laydown",
    "Leave",
    "Makecoffee.Pourgrains",
    "Makecoffee.Pourwater",
    "Maketea.Boilwater",
    "Maketea.Insertteabag",
    "Pour.Frombottle",
    "Pour.Fromcan",
    "Pour.Fromkettle",
    "Readbook",
    "Sitdown",
    "Takepills",
    "Uselaptop",
    "Usetablet",
    "Usetelephone",
    "Walk",
    "WatchTV",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="epoch=003.ckpt")
    parser.add_argument("--camera", type=str, default="0")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--detector-device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--detector-optimize", type=int, default=1, choices=[0, 1])
    parser.add_argument("--detector-compile", type=int, default=1, choices=[0, 1])
    parser.add_argument("--detector-fp16", type=int, default=0, choices=[0, 1])
    parser.add_argument("--engine", type=str, default=None)
    parser.add_argument("--det-threshold", type=float, default=0.35)
    parser.add_argument("--person-class-id", type=int, default=None)
    parser.add_argument("--max-actors", type=int, default=8)
    parser.add_argument("--buffer-frames", type=int, default=32)
    parser.add_argument("--clip-frames", type=int, default=16)
    parser.add_argument("--clip-stride", type=int, default=1)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--short-side", type=int, default=256)
    parser.add_argument("--crop-mode", type=str, default="actor", choices=["actor", "center"])
    parser.add_argument("--detect-every", type=int, default=5)
    parser.add_argument("--action-every", type=int, default=5)
    parser.add_argument("--action-smoothing-window", type=int, default=2)
    parser.add_argument("--track-iou-threshold", type=float, default=0.30)
    parser.add_argument("--track-hold-frames", type=int, default=10)
    parser.add_argument("--camera-buffer-size", type=int, default=1)
    parser.add_argument("--jpeg-quality", type=int, default=80)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.detect_every < 1:
        raise ValueError("--detect-every must be >= 1")
    if args.action_every < 1:
        raise ValueError("--action-every must be >= 1")
    if args.action_smoothing_window < 1:
        raise ValueError("--action-smoothing-window must be >= 1")
    if args.clip_stride < 1:
        raise ValueError("--clip-stride must be >= 1")
    min_buffer = required_clip_buffer(args.clip_frames, args.clip_stride)
    if args.buffer_frames < min_buffer:
        raise ValueError(
            f"--buffer-frames must be >= {min_buffer} for "
            f"--clip-frames {args.clip_frames} and --clip-stride {args.clip_stride}"
        )
    if args.track_hold_frames < 0:
        raise ValueError("--track-hold-frames must be >= 0")
    if not 0.0 <= args.track_iou_threshold <= 1.0:
        raise ValueError("--track-iou-threshold must be in [0, 1]")
    return args


def required_clip_buffer(clip_frames, clip_stride):
    return 1 + (int(clip_frames) - 1) * int(clip_stride)


def camera_source(value):
    try:
        return int(value)
    except ValueError:
        return value


def resolve_device(value):
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def local_ip_addresses():
    addresses = []
    try:
        hostname = socket.gethostname()
        for item in socket.getaddrinfo(hostname, None, socket.AF_INET):
            address = item[4][0]
            if address and not address.startswith("127.") and address not in addresses:
                addresses.append(address)
    except OSError:
        pass

    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.connect(("8.8.8.8", 80))
        address = probe.getsockname()[0]
        if address and not address.startswith("127.") and address not in addresses:
            addresses.append(address)
    except OSError:
        pass
    finally:
        try:
            probe.close()
        except UnboundLocalError:
            pass

    return addresses


def load_rfdetr(person_class_id, detector_device, optimize=True, compile_model=True, fp16=False):
    if detector_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--detector-device cuda was requested, but CUDA is not available.")
    if fp16 and detector_device != "cuda":
        raise RuntimeError("--detector-fp16 1 requires --detector-device cuda.")

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    from rfdetr import RFDETRNano

    print(
        "Loading RF-DETR Nano:",
        {
            "device": detector_device,
            "optimize": bool(optimize),
            "compile": bool(compile_model),
            "fp16": bool(fp16),
        },
        flush=True,
    )
    detector = RFDETRNano(device=detector_device)
    if optimize:
        dtype = torch.float16 if fp16 else torch.float32
        started = time.perf_counter()
        with warnings.catch_warnings():
            tracer_warning = getattr(torch.jit, "TracerWarning", Warning)
            warnings.filterwarnings("ignore", category=tracer_warning)
            detector.optimize_for_inference(
                compile=bool(compile_model),
                batch_size=1,
                dtype=dtype,
            )
        print(
            "RF-DETR optimized:",
            {
                "seconds": round(time.perf_counter() - started, 2),
                "dtype": str(dtype).replace("torch.", ""),
            },
            flush=True,
        )

    person_ids = []
    if person_class_id is not None:
        person_ids = [int(person_class_id)]
    else:
        class_names = getattr(detector, "class_names", None)
        if isinstance(class_names, dict):
            person_ids = [
                int(class_id)
                for class_id, name in class_names.items()
                if str(name).lower() == "person"
            ]
        elif isinstance(class_names, (list, tuple)):
            person_ids = [
                class_id
                for class_id, name in enumerate(class_names)
                if str(name).lower() == "person"
            ]
    if not person_ids:
        raise RuntimeError(
            "RF-DETR class_names did not expose a 'person' class. "
            "Pass --person-class-id explicitly."
        )
    return detector, set(person_ids)


def crop_center_from_boxes(boxes_xyxy):
    boxes = np.asarray(boxes_xyxy, dtype=np.float32)
    if len(boxes) == 0:
        return None
    x1 = float(np.min(boxes[:, 0]))
    y1 = float(np.min(boxes[:, 1]))
    x2 = float(np.max(boxes[:, 2]))
    y2 = float(np.max(boxes[:, 3]))
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def resize_crop_frame(frame_rgb, short_side, input_size, crop_center_xy=None):
    height, width = frame_rgb.shape[:2]
    if width < height:
        new_width = short_side
        new_height = int(np.floor(float(height) / width * short_side))
    else:
        new_height = short_side
        new_width = int(np.floor(float(width) / height * short_side))

    resized = cv2.resize(frame_rgb, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    scale_x = new_width / float(width)
    scale_y = new_height / float(height)
    if crop_center_xy is None:
        x_offset = int(np.ceil((new_width - input_size) / 2.0))
        y_offset = int(np.ceil((new_height - input_size) / 2.0))
    else:
        center_x = float(crop_center_xy[0]) * scale_x
        center_y = float(crop_center_xy[1]) * scale_y
        x_offset = int(round(center_x - input_size * 0.5))
        y_offset = int(round(center_y - input_size * 0.5))
        x_offset = max(0, min(x_offset, max(0, new_width - input_size)))
        y_offset = max(0, min(y_offset, max(0, new_height - input_size)))
    crop = resized[y_offset : y_offset + input_size, x_offset : x_offset + input_size]
    transform = {
        "scale_x": scale_x,
        "scale_y": scale_y,
        "x_offset": x_offset,
        "y_offset": y_offset,
    }
    return crop, transform


def transform_boxes_to_crop(boxes_xyxy, transform, input_size):
    if len(boxes_xyxy) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=bool)

    boxes = np.asarray(boxes_xyxy, dtype=np.float32).copy()
    boxes[:, [0, 2]] = boxes[:, [0, 2]] * transform["scale_x"] - transform["x_offset"]
    boxes[:, [1, 3]] = boxes[:, [1, 3]] * transform["scale_y"] - transform["y_offset"]

    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, input_size)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, input_size)
    keep = (boxes[:, 2] - boxes[:, 0] >= 2) & (boxes[:, 3] - boxes[:, 1] >= 2)
    boxes_norm = boxes / float(input_size)
    return boxes_norm, keep


def preprocess_clip(frames_rgb, short_side, input_size, device, crop_center_xy=None):
    crops = []
    transform = None
    for frame_rgb in frames_rgb:
        crop, transform = resize_crop_frame(
            frame_rgb,
            short_side,
            input_size,
            crop_center_xy=crop_center_xy,
        )
        crop = crop.astype(np.float32) / 255.0
        crop = (crop - np.asarray([0.485, 0.456, 0.406], dtype=np.float32)) / np.asarray(
            [0.229, 0.224, 0.225],
            dtype=np.float32,
        )
        crops.append(torch.from_numpy(crop).permute(2, 0, 1))
    clip = torch.stack(crops, dim=0).unsqueeze(0).to(device=device, dtype=torch.float32)
    return clip, transform


def sample_clip(buffer, clip_frames, clip_stride):
    span = required_clip_buffer(clip_frames, clip_stride)
    if len(buffer) < span:
        raise ValueError(f"Need {span} buffered frames, got {len(buffer)}")
    start = len(buffer) - span
    indices = start + np.arange(clip_frames, dtype=int) * int(clip_stride)
    return [buffer[int(index)] for index in indices]


def detections_to_people(detector, person_ids, frame_rgb, threshold, max_actors):
    detections = detector.predict(Image.fromarray(frame_rgb), threshold=threshold)
    xyxy = np.asarray(getattr(detections, "xyxy", []), dtype=np.float32)
    class_id = getattr(detections, "class_id", None)
    confidence = getattr(detections, "confidence", None)
    if class_id is None or confidence is None:
        raise RuntimeError("RF-DETR detections must expose class_id and confidence.")
    class_id = np.asarray(class_id)
    confidence = np.asarray(confidence, dtype=np.float32)
    keep = np.asarray([int(class_value) in person_ids for class_value in class_id], dtype=bool)
    xyxy = xyxy[keep]
    confidence = confidence[keep]
    if len(xyxy) == 0:
        return xyxy, confidence
    order = np.argsort(-confidence)[:max_actors]
    return xyxy[order], confidence[order]


def pack_actor_boxes(boxes_norm, valid_box_mask, max_actors, device):
    boxes = torch.zeros((1, max_actors, 4), dtype=torch.float32, device=device)
    valid = torch.zeros((1, max_actors), dtype=torch.bool, device=device)
    kept = np.asarray(boxes_norm, dtype=np.float32)[valid_box_mask][:max_actors]
    if len(kept) > 0:
        boxes[0, : len(kept)] = torch.from_numpy(kept).to(device=device, dtype=torch.float32)
        valid[0, : len(kept)] = True
    return boxes, valid


def bbox_iou_xyxy(box_a, box_b):
    ax1, ay1, ax2, ay2 = [float(value) for value in box_a]
    bx1, by1, bx2, by2 = [float(value) for value in box_b]
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter_area
    if denom <= 0.0:
        return 0.0
    return inter_area / denom


class ActionTrack:
    def __init__(self, track_id, bbox_xyxy, frame_index, smoothing_window):
        self.track_id = int(track_id)
        self.bbox_xyxy = np.asarray(bbox_xyxy, dtype=np.float32)
        self.last_seen = int(frame_index)
        self.action_probs = deque(maxlen=int(smoothing_window))
        self.presence_probs = deque(maxlen=int(smoothing_window))

    def update_detection(self, bbox_xyxy, frame_index):
        self.bbox_xyxy = np.asarray(bbox_xyxy, dtype=np.float32)
        self.last_seen = int(frame_index)

    def update_action(self, probs, presence):
        self.action_probs.append(np.asarray(probs, dtype=np.float32).copy())
        self.presence_probs.append(float(presence))

    def action_payload(self):
        if not self.action_probs:
            return {}
        probs = np.mean(np.stack(tuple(self.action_probs), axis=0), axis=0)
        action_id = int(probs.argmax())
        payload = {
            "track_id": self.track_id,
            "label": ACTION_CLASSES[action_id],
            "action_conf": float(probs[action_id]),
            "smooth_count": len(self.action_probs),
        }
        if self.presence_probs:
            payload["presence"] = float(np.mean(self.presence_probs))
        return payload


class TorchActorBackend:
    def __init__(self, checkpoint_path, device):
        from utils.actor_model import load_actor_model

        self.model, self.hparams = load_actor_model(checkpoint_path, device)
        self.device = device
        self.num_actor_tokens = int(self.hparams.get("num_actor_tokens", 0))
        self.clip_frames = int(self.hparams.get("n_frames", 16))
        self.input_size = 224
        self.backend_name = "pytorch"

    def __call__(self, clip, boxes, valid):
        with torch.inference_mode():
            logits, _heatmap, presence = self.model(clip, boxes=boxes, valid=valid)
        return logits, presence


class TensorRTActorBackend:
    def __init__(self, engine_path):
        self.engine = TensorRTActorEngine(engine_path)
        self.device = self.engine.device
        self.num_actor_tokens = self.engine.num_actor_tokens
        self.clip_frames = self.engine.clip_frames
        self.input_size = self.engine.input_size
        self.backend_name = "tensorrt"

    def __call__(self, clip, boxes, valid):
        return self.engine(clip, boxes, valid)


def load_actor_backend(args):
    if args.engine:
        return TensorRTActorBackend(args.engine)
    return TorchActorBackend(args.checkpoint, resolve_device(args.device))


def run_actor_smoke(args, actor):
    frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(args.clip_frames)]
    boxes_xyxy = np.asarray([[160.0, 80.0, 480.0, 420.0]], dtype=np.float32)
    clip, transform = preprocess_clip(
        frames,
        args.short_side,
        args.input_size,
        actor.device,
        crop_center_xy=crop_center_from_boxes(boxes_xyxy)
        if args.crop_mode == "actor"
        else None,
    )
    boxes_norm, keep = transform_boxes_to_crop(boxes_xyxy, transform, args.input_size)
    boxes, valid = pack_actor_boxes(boxes_norm, keep, args.max_actors, actor.device)
    logits, presence = actor(clip, boxes, valid)
    probs = torch.softmax(logits[0, 0], dim=-1)
    print(
        "smoke ok:",
        {
            "checkpoint": args.checkpoint,
            "engine": args.engine,
            "backend": actor.backend_name,
            "device": str(actor.device),
            "valid_slots": int(valid.sum().item()),
            "top_action": ACTION_CLASSES[int(probs.argmax().item())],
            "top_prob": float(probs.max().item()),
            "presence": float(torch.sigmoid(presence[0, 0]).item()),
        },
        flush=True,
    )


class DashboardState:
    def __init__(self):
        self.lock = threading.Lock()
        self.jpeg = None
        self.status = {
            "state": "starting",
            "message": "initializing",
            "frame": 0,
            "actors": [],
            "actor_backend": None,
            "actor_device": None,
            "detector_device": None,
            "detector_optimized": None,
            "last_detector_ms": None,
            "last_actor_ms": None,
            "action_smoothing_window": None,
            "clip_stride": None,
            "crop_mode": None,
            "det_age_frames": None,
        }

    def update(self, jpeg=None, **status):
        with self.lock:
            if jpeg is not None:
                self.jpeg = jpeg
            self.status.update(status)

    def snapshot(self):
        with self.lock:
            return self.jpeg, dict(self.status)


def draw_overlay(frame_bgr, actors, message):
    out = frame_bgr.copy()
    for actor in actors:
        x1, y1, x2, y2 = [int(v) for v in actor["xyxy"]]
        label = actor.get("label", "person")
        action_conf = actor.get("action_conf")
        det_conf = actor.get("det_conf")
        presence = actor.get("presence")
        text = label
        if action_conf is not None:
            text += f" {action_conf:.2f}"
        if presence is not None:
            text += f" pres={presence:.2f}"
        if det_conf is not None:
            text += f" det={det_conf:.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.putText(
            out,
            text,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 220, 0),
            2,
            cv2.LINE_AA,
        )
    cv2.putText(
        out,
        message,
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


class LiveRunner:
    def __init__(self, args, state):
        self.args = args
        self.state = state
        self.actor = load_actor_backend(args)
        self.device = self.actor.device
        if self.actor.num_actor_tokens != args.max_actors:
            raise RuntimeError(
                f"--max-actors must match actor slots={self.actor.num_actor_tokens}"
            )
        if self.actor.clip_frames != args.clip_frames:
            raise RuntimeError(
                f"--clip-frames must match actor clip length={self.actor.clip_frames}"
            )
        if self.actor.input_size != args.input_size:
            raise RuntimeError(
                f"--input-size must match actor input size={self.actor.input_size}"
            )
        self.detector, self.person_ids = load_rfdetr(
            args.person_class_id,
            args.detector_device,
            optimize=bool(args.detector_optimize),
            compile_model=bool(args.detector_compile),
            fp16=bool(args.detector_fp16),
        )
        self.state.update(
            actor_backend=self.actor.backend_name,
            actor_device=str(self.actor.device),
            detector_device=args.detector_device,
            detector_optimized=bool(args.detector_optimize),
        )
        self.buffer = deque(maxlen=args.buffer_frames)
        self.frame_count = 0
        self.last_boxes_xyxy = np.zeros((0, 4), dtype=np.float32)
        self.last_det_conf = np.zeros((0,), dtype=np.float32)
        self.last_detection_frame = None
        self.last_detector_ms = None
        self.last_actor_ms = None
        self.next_track_id = 1
        self.tracks = {}
        self.current_track_ids = []
        self.state.update(
            action_smoothing_window=args.action_smoothing_window,
            clip_stride=args.clip_stride,
            crop_mode=args.crop_mode,
        )

    def smoke(self):
        run_actor_smoke(self.args, self.actor)

    def _update_tracks(self, boxes_xyxy):
        boxes = np.asarray(boxes_xyxy, dtype=np.float32)
        if len(boxes) == 0:
            self.current_track_ids = []
            self._drop_stale_tracks()
            return

        candidates = []
        for det_index, box in enumerate(boxes):
            for track_id, track in self.tracks.items():
                iou = bbox_iou_xyxy(box, track.bbox_xyxy)
                if iou >= self.args.track_iou_threshold:
                    candidates.append((iou, det_index, track_id))

        matches = {}
        used_tracks = set()
        for _iou, det_index, track_id in sorted(candidates, reverse=True):
            if det_index in matches or track_id in used_tracks:
                continue
            matches[det_index] = track_id
            used_tracks.add(track_id)

        track_ids = []
        for det_index, box in enumerate(boxes):
            track_id = matches.get(det_index)
            if track_id is None:
                track_id = self.next_track_id
                self.next_track_id += 1
                self.tracks[track_id] = ActionTrack(
                    track_id,
                    box,
                    self.frame_count,
                    self.args.action_smoothing_window,
                )
            else:
                self.tracks[track_id].update_detection(box, self.frame_count)
            track_ids.append(track_id)

        self.current_track_ids = track_ids
        self._drop_stale_tracks()

    def _drop_stale_tracks(self):
        stale = [
            track_id
            for track_id, track in self.tracks.items()
            if self.frame_count - track.last_seen > self.args.track_hold_frames
        ]
        for track_id in stale:
            del self.tracks[track_id]

    def run(self):
        cap = cv2.VideoCapture(camera_source(self.args.camera))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera source: {self.args.camera}")
        if self.args.camera_buffer_size > 0:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, int(self.args.camera_buffer_size))

        self.state.update(state="running", message="camera open")
        try:
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    self.state.update(state="error", message="camera read failed")
                    time.sleep(0.25)
                    continue

                self.frame_count += 1
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                self.buffer.append(frame_rgb)

                clip_ready = len(self.buffer) >= required_clip_buffer(
                    self.args.clip_frames,
                    self.args.clip_stride,
                )
                action_due = clip_ready and self.frame_count % self.args.action_every == 0
                run_detection = self.frame_count % self.args.detect_every == 0 or action_due

                if run_detection:
                    started = time.perf_counter()
                    self.last_boxes_xyxy, self.last_det_conf = detections_to_people(
                        self.detector,
                        self.person_ids,
                        frame_rgb,
                        self.args.det_threshold,
                        self.args.max_actors,
                    )
                    self.last_detector_ms = (time.perf_counter() - started) * 1000.0
                    self.last_detection_frame = self.frame_count
                    self._update_tracks(self.last_boxes_xyxy)

                det_age = (
                    None
                    if self.last_detection_frame is None
                    else self.frame_count - self.last_detection_frame
                )
                message = (
                    f"buffer {len(self.buffer)}/{self.args.buffer_frames} "
                    f"stride={self.args.clip_stride}"
                )
                actors = [
                    {
                        "xyxy": box.tolist(),
                        "det_conf": float(conf),
                        "label": "person",
                        "track_id": int(track_id),
                    }
                    for box, conf, track_id in zip(
                        self.last_boxes_xyxy,
                        self.last_det_conf,
                        self.current_track_ids,
                    )
                ]
                for actor in actors:
                    track = self.tracks.get(actor["track_id"])
                    if track is not None:
                        actor.update(track.action_payload())

                should_run_action = action_due and len(self.last_boxes_xyxy) > 0
                if should_run_action:
                    clip_frames = sample_clip(
                        self.buffer,
                        self.args.clip_frames,
                        self.args.clip_stride,
                    )
                    crop_center_xy = (
                        crop_center_from_boxes(self.last_boxes_xyxy)
                        if self.args.crop_mode == "actor"
                        else None
                    )
                    clip, transform = preprocess_clip(
                        clip_frames,
                        self.args.short_side,
                        self.args.input_size,
                        self.device,
                        crop_center_xy=crop_center_xy,
                    )
                    boxes_norm, keep = transform_boxes_to_crop(
                        self.last_boxes_xyxy,
                        transform,
                        self.args.input_size,
                    )
                    boxes, valid = pack_actor_boxes(
                        boxes_norm,
                        keep,
                        self.args.max_actors,
                        self.device,
                    )
                    if valid.any():
                        started = time.perf_counter()
                        logits, presence_logits = self.actor(clip, boxes, valid)
                        self.last_actor_ms = (time.perf_counter() - started) * 1000.0
                        action_probs = torch.softmax(logits[0], dim=-1).detach().cpu().numpy()
                        presence_probs = (
                            torch.sigmoid(presence_logits[0]).detach().cpu().numpy()
                        )
                        kept_actor_idx = np.flatnonzero(keep)[: self.args.max_actors]
                        for slot, actor_idx in enumerate(kept_actor_idx):
                            if actor_idx >= len(self.current_track_ids):
                                continue
                            track_id = self.current_track_ids[int(actor_idx)]
                            track = self.tracks.get(track_id)
                            if track is None:
                                continue
                            track.update_action(action_probs[slot], presence_probs[slot])
                            action_id = int(action_probs[slot].argmax())
                            raw_payload = {
                                "raw_label": ACTION_CLASSES[action_id],
                                "raw_action_conf": float(action_probs[slot, action_id]),
                            }
                            actors[int(actor_idx)].update(
                                track.action_payload()
                            )
                            actors[int(actor_idx)].update(raw_payload)
                        for actor in actors:
                            track = self.tracks.get(actor["track_id"])
                            if track is None:
                                continue
                            payload = track.action_payload()
                            if payload:
                                actor.update(payload)
                        message = (
                            f"{self.actor.backend_name} actors={int(valid.sum().item())} "
                            f"det={self.last_detector_ms:.0f}ms "
                            f"actor={self.last_actor_ms:.0f}ms "
                            f"det_age={det_age} "
                            f"stride={self.args.clip_stride} "
                            f"crop={self.args.crop_mode} "
                            f"smooth={self.args.action_smoothing_window} "
                            f"frame={self.frame_count}"
                        )
                    else:
                        message = "detections outside model crop"
                elif len(self.last_boxes_xyxy) == 0:
                    message = "no RF-DETR person detections"

                overlay = draw_overlay(frame_bgr, actors, message)
                ok, encoded = cv2.imencode(
                    ".jpg",
                    overlay,
                    [int(cv2.IMWRITE_JPEG_QUALITY), int(self.args.jpeg_quality)],
                )
                if ok:
                    self.state.update(
                        jpeg=encoded.tobytes(),
                        state="running",
                        message=message,
                        frame=self.frame_count,
                        actors=actors,
                        last_detector_ms=self.last_detector_ms,
                        last_actor_ms=self.last_actor_ms,
                        det_age_frames=det_age,
                    )
        finally:
            cap.release()


class DashboardHandler(BaseHTTPRequestHandler):
    state = None

    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(
                b"""<!doctype html>
<html><head><title>PO-GUISE Actor Dashboard</title>
<style>
body{margin:0;background:#101214;color:#f2f4f5;font-family:Arial,sans-serif}
main{max-width:1120px;margin:0 auto;padding:16px}
img{max-width:100%;height:auto;border:1px solid #333}
pre{white-space:pre-wrap;background:#181b1f;padding:12px}
</style></head>
<body><main>
<h2>PO-GUISE Actor Dashboard</h2>
<img src="/stream.mjpg">
<pre id="status"></pre>
<script>
async function poll(){
  const r = await fetch('/status.json');
  document.getElementById('status').textContent = JSON.stringify(await r.json(), null, 2);
}
setInterval(poll, 1000); poll();
</script>
</main></body></html>"""
            )
            return
        if self.path == "/status.json":
            _, status = self.state.snapshot()
            body = json.dumps(status).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/stream.mjpg":
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            while True:
                jpeg, _ = self.state.snapshot()
                if jpeg is None:
                    time.sleep(0.1)
                    continue
                try:
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii"))
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
                    time.sleep(0.05)
                except BrokenPipeError:
                    break
            return
        self.send_error(404, html.escape(self.path))

    def log_message(self, fmt, *args):
        return


def main():
    args = parse_args()
    if args.smoke:
        actor = load_actor_backend(args)
        if actor.num_actor_tokens != args.max_actors:
            raise RuntimeError(
                f"--max-actors must match actor slots={actor.num_actor_tokens}"
            )
        run_actor_smoke(args, actor)
        return

    state = DashboardState()
    runner = LiveRunner(args, state)

    DashboardHandler.state = state
    thread = threading.Thread(target=runner.run, daemon=True)
    thread.start()
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"Dashboard bind: http://{args.host}:{args.port}", flush=True)
    if args.host in {"0.0.0.0", ""}:
        for address in local_ip_addresses():
            print(f"Dashboard LAN:  http://{address}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
