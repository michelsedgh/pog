import argparse
import html
import json
import socket
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import torch
from PIL import Image

from datasets.object_vocab import DETECTOR_TO_OBJECT, NONE_OBJECT_ID, OBJECT_CLASSES, OBJECT_TO_ID
from utils.rfdetr_tensorrt import TensorRTRFDETRNano

TRAINING_CLIP_FRAMES = 16
TRAINING_SPAN_FRAMES = 128
MODEL_INPUT_SIZE = 224
MODEL_SHORT_SIDE = 256
DETECTION_EVERY_FRAME = 1
ACTION_EVERY_FRAME = 1
ACTION_SMOOTHING_WINDOW = 1
MIN_OBJECT_TRACK_SAMPLE_COUNT = 2


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
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--camera", type=str, default="0")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--detector-engine", type=str, default=None)
    parser.add_argument("--det-threshold", type=float, default=0.35)
    parser.add_argument("--object-threshold", type=float, default=0.25)
    parser.add_argument("--person-class-id", type=int, default=None)
    parser.add_argument("--track-iou-threshold", type=float, default=0.30)
    parser.add_argument("--track-hold-frames", type=int, default=10)
    parser.add_argument("--camera-buffer-size", type=int, default=1)
    parser.add_argument("--jpeg-quality", type=int, default=80)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not args.smoke and not args.detector_engine:
        raise ValueError("--detector-engine is required for live inference.")
    if args.track_hold_frames < 0:
        raise ValueError("--track-hold-frames must be >= 0")
    if not 0.0 <= args.track_iou_threshold <= 1.0:
        raise ValueError("--track-iou-threshold must be in [0, 1]")
    if not 0.0 <= args.det_threshold <= 1.0:
        raise ValueError("--det-threshold must be in [0, 1]")
    if not 0.0 <= args.object_threshold <= 1.0:
        raise ValueError("--object-threshold must be in [0, 1]")
    return args


def required_clip_buffer():
    return TRAINING_SPAN_FRAMES


def sample_buffer_items(buffer):
    span = required_clip_buffer()
    if len(buffer) < span:
        raise ValueError(f"Need {span} buffered items, got {len(buffer)}")
    start = len(buffer) - span
    end = len(buffer) - 1
    indices = np.linspace(start, end, TRAINING_CLIP_FRAMES, dtype=int)
    return [buffer[int(index)] for index in indices]


def camera_source(value):
    try:
        return int(value)
    except ValueError:
        return value


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


def load_detector_backend(args):
    detector = TensorRTRFDETRNano(args.detector_engine)
    if args.person_class_id is not None:
        person_ids = [int(args.person_class_id)]
    else:
        person_ids = [
            int(class_id)
            for class_id, name in detector.class_names.items()
            if str(name).lower() == "person"
        ]
    if not person_ids:
        raise RuntimeError(
            "TensorRT RF-DETR class_names did not expose a 'person' class. "
            "Pass --person-class-id explicitly."
        )
    return detector, set(person_ids), "tensorrt"


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


def sample_clip(buffer):
    return sample_buffer_items(buffer)


def detector_class_name(detector, class_id):
    class_names = getattr(detector, "class_names", None)
    if isinstance(class_names, dict):
        name = class_names.get(int(class_id))
        if name is None:
            name = class_names.get(str(int(class_id)))
        return None if name is None else str(name)
    if isinstance(class_names, (list, tuple)):
        class_id = int(class_id)
        if 0 <= class_id < len(class_names):
            return str(class_names[class_id])
    return None


def detector_class_to_object_id(detector, class_id):
    class_name = detector_class_name(detector, class_id)
    if class_name is None:
        return None
    object_name = DETECTOR_TO_OBJECT.get(class_name.strip().lower())
    if object_name is None:
        return None
    return OBJECT_TO_ID.get(object_name)


def detections_to_people_and_objects(
    detector,
    person_ids,
    frame_rgb,
    person_threshold,
    object_threshold,
    max_actors,
    max_objects,
):
    threshold = min(float(person_threshold), float(object_threshold))
    detections = detector.predict(Image.fromarray(frame_rgb), threshold=threshold)
    xyxy = np.asarray(getattr(detections, "xyxy", []), dtype=np.float32)
    class_id = getattr(detections, "class_id", None)
    confidence = getattr(detections, "confidence", None)
    if class_id is None or confidence is None:
        raise RuntimeError("RF-DETR detections must expose class_id and confidence.")
    class_id = np.asarray(class_id)
    confidence = np.asarray(confidence, dtype=np.float32)

    person_keep = np.asarray(
        [
            int(class_value) in person_ids and float(conf) >= float(person_threshold)
            for class_value, conf in zip(class_id, confidence)
        ],
        dtype=bool,
    )
    people_xyxy = xyxy[person_keep]
    people_conf = confidence[person_keep]
    if len(people_xyxy) > 0:
        order = np.argsort(-people_conf)[:max_actors]
        people_xyxy = people_xyxy[order]
        people_conf = people_conf[order]

    object_records = []
    if max_objects > 0:
        for box, class_value, conf in zip(xyxy, class_id, confidence):
            if int(class_value) in person_ids or float(conf) < float(object_threshold):
                continue
            object_id = detector_class_to_object_id(detector, int(class_value))
            if object_id is None:
                continue
            object_records.append(
                (
                    float(conf),
                    np.asarray(box, dtype=np.float32),
                    int(object_id),
                    OBJECT_CLASSES[int(object_id)],
                )
            )
    object_records.sort(key=lambda item: item[0], reverse=True)
    if object_records:
        object_conf = np.asarray([item[0] for item in object_records], dtype=np.float32)
        object_xyxy = np.stack([item[1] for item in object_records], axis=0).astype(
            np.float32
        )
        object_class_ids = np.asarray([item[2] for item in object_records], dtype=np.int64)
        object_labels = [item[3] for item in object_records]
    else:
        object_xyxy = np.zeros((0, 4), dtype=np.float32)
        object_class_ids = np.zeros((0,), dtype=np.int64)
        object_conf = np.zeros((0,), dtype=np.float32)
        object_labels = []

    return (
        people_xyxy,
        people_conf,
        object_xyxy,
        object_class_ids,
        object_conf,
        object_labels,
    )


def pack_actor_boxes(boxes_norm, valid_box_mask, max_actors, device):
    boxes = torch.zeros((1, max_actors, 4), dtype=torch.float32, device=device)
    valid = torch.zeros((1, max_actors), dtype=torch.bool, device=device)
    kept = np.asarray(boxes_norm, dtype=np.float32)[valid_box_mask][:max_actors]
    if len(kept) > 0:
        boxes[0, : len(kept)] = torch.from_numpy(kept).to(device=device, dtype=torch.float32)
        valid[0, : len(kept)] = True
    return boxes, valid


def _object_track_sort_key(track, clip_frames):
    confs = [float(item["conf"]) for item in track["entries"]]
    max_conf = max(confs) if confs else 0.0
    mean_conf = float(np.mean(confs)) if confs else 0.0
    coverage = len(track["sample_positions"]) / float(max(int(clip_frames), 1))
    return (-max_conf, -mean_conf, -coverage, int(track["class_id"]))


def _weighted_track_box(track):
    boxes = [np.asarray(item["box_norm"], dtype=np.float32) for item in track["entries"]]
    weights = np.asarray(
        [max(float(item["conf"]), 1e-4) for item in track["entries"]],
        dtype=np.float32,
    )
    if not boxes:
        return None
    weights = weights / max(float(weights.sum()), 1e-6)
    box = np.stack(boxes, axis=0)
    return np.sum(box * weights[:, None], axis=0).clip(0.0, 1.0).astype(np.float32)


def pack_temporal_object_tokens(
    detection_records,
    transform,
    input_size,
    max_objects,
    device,
    track_iou_threshold=0.2,
    min_sample_count=MIN_OBJECT_TRACK_SAMPLE_COUNT,
):
    entries = []
    for sample_pos, record in enumerate(detection_records):
        if not record:
            continue
        boxes_xyxy = np.asarray(record["boxes_xyxy"], dtype=np.float32)
        if len(boxes_xyxy) == 0:
            continue
        boxes_norm, keep = transform_boxes_to_crop(
            boxes_xyxy,
            transform,
            input_size,
        )
        kept_indices = np.flatnonzero(keep)
        for source_index in kept_indices:
            class_id = int(record["class_ids"][source_index])
            entries.append(
                {
                    "sample_pos": int(sample_pos),
                    "class_id": class_id,
                    "conf": float(record["confs"][source_index]),
                    "label": str(record["labels"][source_index]),
                    "box_norm": boxes_norm[source_index].astype(np.float32),
                }
            )

    tracks = []
    for entry in sorted(
        entries,
        key=lambda item: (
            int(item["sample_pos"]),
            int(item["class_id"]),
            -float(item["conf"]),
        ),
    ):
        best_track = None
        best_iou = 0.0
        for track in tracks:
            if int(track["class_id"]) != int(entry["class_id"]):
                continue
            iou = bbox_iou_xyxy(track["last_box"], entry["box_norm"])
            if iou > best_iou:
                best_iou = iou
                best_track = track
        if best_track is not None and best_iou >= float(track_iou_threshold):
            best_track["entries"].append(entry)
            best_track["sample_positions"].add(int(entry["sample_pos"]))
            best_track["last_box"] = entry["box_norm"]
        else:
            tracks.append(
                {
                    "class_id": int(entry["class_id"]),
                    "label": str(entry["label"]),
                    "entries": [entry],
                    "sample_positions": {int(entry["sample_pos"])},
                    "last_box": entry["box_norm"],
                }
            )

    boxes = torch.zeros((1, max_objects, 4), dtype=torch.float32, device=device)
    classes = torch.full(
        (1, max_objects),
        int(NONE_OBJECT_ID),
        dtype=torch.long,
        device=device,
    )
    confs = torch.zeros((1, max_objects), dtype=torch.float32, device=device)
    valid = torch.zeros((1, max_objects), dtype=torch.bool, device=device)
    packed = []
    min_sample_count = max(int(min_sample_count), 1)
    supported_tracks = [
        track
        for track in tracks
        if len(track["sample_positions"]) >= min_sample_count
    ]
    sorted_tracks = sorted(
        supported_tracks,
        key=lambda track: _object_track_sort_key(track, len(detection_records)),
    )
    for slot, track in enumerate(sorted_tracks[:max_objects]):
        box = _weighted_track_box(track)
        if box is None or box[2] <= box[0] or box[3] <= box[1]:
            continue
        track_confs = [float(item["conf"]) for item in track["entries"]]
        boxes[0, slot] = torch.from_numpy(box).to(device=device)
        classes[0, slot] = int(track["class_id"])
        confs[0, slot] = float(np.mean(track_confs)) if track_confs else 0.0
        valid[0, slot] = True
        packed.append(
            {
                "slot": int(slot),
                "label": str(track["label"]),
                "object_class_id": int(track["class_id"]),
                "conf": float(confs[0, slot].detach().cpu().item()),
                "box_norm": box.astype(float).tolist(),
                "sample_count": int(len(track["sample_positions"])),
            }
        )

    return (
        {
            "object_boxes": boxes,
            "object_classes": classes,
            "object_confs": confs,
            "object_valid": valid,
        },
        packed,
    )


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
    def __init__(self, track_id, bbox_xyxy, frame_index):
        self.track_id = int(track_id)
        self.bbox_xyxy = np.asarray(bbox_xyxy, dtype=np.float32)
        self.last_seen = int(frame_index)
        self.action_probs = deque(maxlen=ACTION_SMOOTHING_WINDOW)
        self.presence_probs = deque(maxlen=ACTION_SMOOTHING_WINDOW)
        self.latest_extra_payload = {}

    def update_detection(self, bbox_xyxy, frame_index):
        self.bbox_xyxy = np.asarray(bbox_xyxy, dtype=np.float32)
        self.last_seen = int(frame_index)

    def update_action(self, probs, presence, extra_payload=None):
        self.action_probs.append(np.asarray(probs, dtype=np.float32).copy())
        self.presence_probs.append(float(presence))
        if extra_payload is not None:
            self.latest_extra_payload = dict(extra_payload)

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
        payload.update(self.latest_extra_payload)
        return payload


class TorchActorBackend:
    def __init__(self, checkpoint_path):
        from utils.actor_model import load_actor_model

        if not torch.cuda.is_available():
            raise RuntimeError("Live actor inference requires CUDA.")
        self.device = torch.device("cuda")
        self.model, self.hparams = load_actor_model(
            checkpoint_path,
            self.device,
            dtype=torch.float32,
        )
        self.precision = "fp32"
        self.num_actor_tokens = int(self.hparams.get("num_actor_tokens", 0))
        self.clip_frames = int(self.hparams.get("n_frames", 16))
        self.input_size = MODEL_INPUT_SIZE
        self.backend_name = "pytorch"
        self.scene_object_tokens = bool(self.hparams.get("scene_object_tokens", 0))
        self.actor_object_logit_residual = self.scene_object_tokens
        self.num_scene_object_tokens = (
            int(self.hparams.get("num_scene_object_tokens", 0))
            if self.scene_object_tokens
            else 0
        )
        if self.clip_frames != TRAINING_CLIP_FRAMES:
            raise RuntimeError(
                f"Checkpoint n_frames={self.clip_frames}; live inference is fixed to "
                f"{TRAINING_CLIP_FRAMES} frames to match the trained object-fused model."
            )
        if not self.scene_object_tokens:
            raise RuntimeError(
                "This dashboard is fixed to the object-fused checkpoints and requires "
                "scene_object_tokens=1."
            )
        if self.num_scene_object_tokens <= 0:
            raise RuntimeError("Checkpoint scene object token count must be positive.")

    def __call__(self, clip, boxes, valid, object_inputs=None):
        model_kwargs = {"boxes": boxes, "valid": valid}
        if self.scene_object_tokens:
            if object_inputs is None:
                raise RuntimeError(
                    "This checkpoint has scene_object_tokens=1, so live inference "
                    "must pass object_boxes, object_classes, object_confs, and object_valid."
                )
            expected_keys = {
                "object_boxes",
                "object_classes",
                "object_confs",
                "object_valid",
            }
            missing = sorted(expected_keys - set(object_inputs))
            if missing:
                raise RuntimeError(f"Missing object input keys: {missing}")
            model_kwargs.update(object_inputs)
        elif object_inputs is not None:
            raise RuntimeError("Object inputs were passed to a checkpoint without object tokens.")

        with torch.inference_mode():
            output = self.model(clip, **model_kwargs)
            if not isinstance(output, (tuple, list)):
                raise RuntimeError(f"Unexpected model output type: {type(output)}")
            if len(output) == 4:
                logits, _heatmap, presence, object_selection = output
            elif len(output) == 3:
                logits, _heatmap, presence = output
                object_selection = None
            else:
                raise RuntimeError(f"Unexpected model output length: {len(output)}")
        return logits, presence, object_selection


def load_actor_backend(args):
    return TorchActorBackend(args.checkpoint)


def run_actor_smoke(args, actor):
    frames = [
        np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(TRAINING_CLIP_FRAMES)
    ]
    boxes_xyxy = np.asarray([[160.0, 80.0, 480.0, 420.0]], dtype=np.float32)
    clip, transform = preprocess_clip(
        frames,
        MODEL_SHORT_SIDE,
        MODEL_INPUT_SIZE,
        actor.device,
        crop_center_xy=crop_center_from_boxes(boxes_xyxy),
    )
    boxes_norm, keep = transform_boxes_to_crop(
        boxes_xyxy,
        transform,
        MODEL_INPUT_SIZE,
    )
    boxes, valid = pack_actor_boxes(
        boxes_norm,
        keep,
        actor.num_actor_tokens,
        actor.device,
    )
    object_inputs, _packed_objects = pack_temporal_object_tokens(
        [],
        transform,
        MODEL_INPUT_SIZE,
        actor.num_scene_object_tokens,
        actor.device,
        track_iou_threshold=0.2,
    )
    logits, presence, object_selection = actor(clip, boxes, valid, object_inputs)
    probs = torch.softmax(logits[0, 0], dim=-1)
    print(
        "smoke ok:",
        {
            "checkpoint": args.checkpoint,
            "backend": actor.backend_name,
            "precision": actor.precision,
            "device": str(actor.device),
            "scene_object_tokens": bool(actor.scene_object_tokens),
            "actor_object_logit_residual": bool(actor.actor_object_logit_residual),
            "num_scene_object_tokens": int(actor.num_scene_object_tokens),
            "clip_frames": TRAINING_CLIP_FRAMES,
            "span_frames": TRAINING_SPAN_FRAMES,
            "sampling": "linspace",
            "min_object_track_sample_count": MIN_OBJECT_TRACK_SAMPLE_COUNT,
            "valid_slots": int(valid.sum().item()),
            "object_selection_shape": None
            if object_selection is None
            else tuple(object_selection.shape),
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
            "objects": [],
            "actor_backend": None,
            "actor_device": None,
            "scene_object_tokens": None,
            "num_scene_object_tokens": None,
            "detector_backend": None,
            "last_detector_ms": None,
            "last_actor_ms": None,
            "action_smoothing_window": None,
            "clip_frames": None,
            "clip_span_frames": None,
            "clip_sampling": None,
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


def draw_overlay(frame_bgr, actors, objects, message):
    out = frame_bgr.copy()
    for obj in objects:
        x1, y1, x2, y2 = [int(v) for v in obj["xyxy"]]
        text = f"{obj.get('label', 'object')} {obj.get('conf', 0.0):.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 180, 255), 2)
        cv2.putText(
            out,
            text,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 180, 255),
            2,
            cv2.LINE_AA,
        )
    for actor in actors:
        x1, y1, x2, y2 = [int(v) for v in actor["xyxy"]]
        label = actor.get("label", "person")
        action_conf = actor.get("action_conf")
        det_conf = actor.get("det_conf")
        presence = actor.get("presence")
        selected_object = actor.get("selected_object")
        selected_object_conf = actor.get("selected_object_conf")
        text = label
        if action_conf is not None:
            text += f" {action_conf:.2f}"
        if presence is not None:
            text += f" pres={presence:.2f}"
        if selected_object is not None:
            text += f" obj={selected_object}"
            if selected_object_conf is not None:
                text += f":{selected_object_conf:.2f}"
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
        if self.actor.input_size != MODEL_INPUT_SIZE:
            raise RuntimeError(
                f"Actor input size {self.actor.input_size} does not match "
                f"fixed live input size {MODEL_INPUT_SIZE}."
            )
        self.max_actors = int(self.actor.num_actor_tokens)
        self.max_objects = int(self.actor.num_scene_object_tokens)
        self.detector, self.person_ids, self.detector_backend_name = load_detector_backend(args)
        self.state.update(
            actor_backend=self.actor.backend_name,
            actor_precision=self.actor.precision,
            actor_device=str(self.actor.device),
            scene_object_tokens=bool(self.actor.scene_object_tokens),
            actor_object_logit_residual=bool(self.actor.actor_object_logit_residual),
            num_scene_object_tokens=int(self.actor.num_scene_object_tokens),
            detector_backend=self.detector_backend_name,
        )
        self.buffer = deque(maxlen=TRAINING_SPAN_FRAMES)
        self.detection_buffer = deque(maxlen=TRAINING_SPAN_FRAMES)
        self.frame_count = 0
        self.last_boxes_xyxy = np.zeros((0, 4), dtype=np.float32)
        self.last_det_conf = np.zeros((0,), dtype=np.float32)
        self.last_object_boxes_xyxy = np.zeros((0, 4), dtype=np.float32)
        self.last_object_class_ids = np.zeros((0,), dtype=np.int64)
        self.last_object_conf = np.zeros((0,), dtype=np.float32)
        self.last_object_labels = []
        self.last_detection_frame = None
        self.last_detector_ms = None
        self.last_actor_ms = None
        self.next_track_id = 1
        self.tracks = {}
        self.current_track_ids = []
        self.state.update(
            action_smoothing_window=ACTION_SMOOTHING_WINDOW,
            action_every=ACTION_EVERY_FRAME,
            detect_every=DETECTION_EVERY_FRAME,
            clip_frames=TRAINING_CLIP_FRAMES,
            clip_sampling="linspace",
            clip_span_frames=TRAINING_SPAN_FRAMES,
            crop_mode="actor",
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

                started = time.perf_counter()
                (
                    self.last_boxes_xyxy,
                    self.last_det_conf,
                    self.last_object_boxes_xyxy,
                    self.last_object_class_ids,
                    self.last_object_conf,
                    self.last_object_labels,
                ) = detections_to_people_and_objects(
                    self.detector,
                    self.person_ids,
                    frame_rgb,
                    self.args.det_threshold,
                    self.args.object_threshold,
                    self.max_actors,
                    self.max_objects,
                )
                self.last_detector_ms = (time.perf_counter() - started) * 1000.0
                self.last_detection_frame = self.frame_count
                self._update_tracks(self.last_boxes_xyxy)
                detection_record = {
                    "boxes_xyxy": self.last_object_boxes_xyxy.copy(),
                    "class_ids": self.last_object_class_ids.copy(),
                    "confs": self.last_object_conf.copy(),
                    "labels": list(self.last_object_labels),
                }
                self.detection_buffer.append(detection_record)
                clip_ready = len(self.buffer) >= required_clip_buffer()
                action_due = clip_ready

                det_age = (
                    None
                    if self.last_detection_frame is None
                    else self.frame_count - self.last_detection_frame
                )
                message = (
                    f"buffer {len(self.buffer)}/{TRAINING_SPAN_FRAMES} "
                    f"sampling=linspace span={TRAINING_SPAN_FRAMES} "
                    f"frames={TRAINING_CLIP_FRAMES}"
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

                objects = [
                    {
                        "xyxy": box.tolist(),
                        "label": label,
                        "object_class_id": int(object_id),
                        "conf": float(conf),
                    }
                    for box, label, object_id, conf in zip(
                        self.last_object_boxes_xyxy,
                        self.last_object_labels,
                        self.last_object_class_ids,
                        self.last_object_conf,
                    )
                ]

                should_run_action = action_due and len(self.last_boxes_xyxy) > 0
                if should_run_action:
                    clip_frames = sample_clip(self.buffer)
                    clip_detection_records = sample_buffer_items(self.detection_buffer)
                    crop_center_xy = crop_center_from_boxes(self.last_boxes_xyxy)
                    clip, transform = preprocess_clip(
                        clip_frames,
                        MODEL_SHORT_SIDE,
                        MODEL_INPUT_SIZE,
                        self.device,
                        crop_center_xy=crop_center_xy,
                    )
                    boxes_norm, keep = transform_boxes_to_crop(
                        self.last_boxes_xyxy,
                        transform,
                        MODEL_INPUT_SIZE,
                    )
                    boxes, valid = pack_actor_boxes(
                        boxes_norm,
                        keep,
                        self.max_actors,
                        self.device,
                    )
                    object_inputs, packed_objects = pack_temporal_object_tokens(
                        clip_detection_records,
                        transform,
                        MODEL_INPUT_SIZE,
                        self.actor.num_scene_object_tokens,
                        self.device,
                        track_iou_threshold=0.2,
                    )
                    if valid.any():
                        started = time.perf_counter()
                        logits, presence_logits, object_selection_logits = self.actor(
                            clip,
                            boxes,
                            valid,
                            object_inputs,
                        )
                        self.last_actor_ms = (time.perf_counter() - started) * 1000.0
                        action_probs = torch.softmax(logits[0], dim=-1).detach().cpu().numpy()
                        presence_probs = (
                            torch.sigmoid(presence_logits[0]).detach().cpu().numpy()
                        )
                        selection_probs = None
                        if object_selection_logits is not None:
                            selection_probs = (
                                torch.softmax(object_selection_logits[0], dim=-1)
                                .detach()
                                .cpu()
                                .numpy()
                            )
                        kept_actor_idx = np.flatnonzero(keep)[: self.max_actors]
                        for slot, actor_idx in enumerate(kept_actor_idx):
                            if actor_idx >= len(self.current_track_ids):
                                continue
                            track_id = self.current_track_ids[int(actor_idx)]
                            track = self.tracks.get(track_id)
                            if track is None:
                                continue
                            action_id = int(action_probs[slot].argmax())
                            raw_payload = {
                                "raw_label": ACTION_CLASSES[action_id],
                                "raw_action_conf": float(action_probs[slot, action_id]),
                            }
                            if selection_probs is not None:
                                selected_index = int(selection_probs[slot].argmax())
                                selected_conf = float(selection_probs[slot, selected_index])
                                selection_payload = {
                                    "selected_object": "none",
                                    "selected_object_conf": selected_conf,
                                }
                                if selected_index > 0:
                                    object_slot = selected_index - 1
                                    if object_slot < len(packed_objects):
                                        selected_object = packed_objects[object_slot]
                                        selection_payload.update(
                                            {
                                                "selected_object": selected_object["label"],
                                                "selected_object_slot": int(object_slot),
                                                "selected_object_det_conf": float(
                                                    selected_object["conf"]
                                                ),
                                            }
                                        )
                                    else:
                                        selection_payload.update(
                                            {
                                                "selected_object": "invalid_slot",
                                                "selected_object_slot": int(object_slot),
                                            }
                                        )
                                raw_payload.update(selection_payload)
                            track.update_action(
                                action_probs[slot],
                                presence_probs[slot],
                                raw_payload,
                            )
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
                            f"objects={len(packed_objects)}/{len(objects)} "
                            f"det={self.last_detector_ms:.0f}ms "
                            f"actor={self.last_actor_ms:.0f}ms "
                            f"det_age={det_age} "
                            f"sampling=linspace "
                            f"span={TRAINING_SPAN_FRAMES} "
                            f"frames={TRAINING_CLIP_FRAMES} "
                            f"crop=actor "
                            f"smooth={ACTION_SMOOTHING_WINDOW} "
                            f"min_obj_samples={MIN_OBJECT_TRACK_SAMPLE_COUNT} "
                            f"frame={self.frame_count}"
                        )
                        packed_object_payload = [
                            {
                                "slot": int(item.get("slot", index)),
                                "label": item.get("label"),
                                "object_class_id": int(item.get("object_class_id", -1)),
                                "conf": float(item.get("conf", 0.0)),
                                "sample_count": int(item.get("sample_count", 0)),
                                "box_norm": item.get("box_norm"),
                            }
                            for index, item in enumerate(packed_objects)
                        ]
                    else:
                        message = "detections outside model crop"
                        packed_object_payload = []
                elif len(self.last_boxes_xyxy) == 0:
                    message = "no RF-DETR person detections"
                    packed_object_payload = []
                else:
                    packed_object_payload = []

                overlay = draw_overlay(frame_bgr, actors, objects, message)
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
                        objects=objects,
                        packed_objects=packed_object_payload,
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
