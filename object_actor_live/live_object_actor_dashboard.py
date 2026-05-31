#!/usr/bin/env python3
import argparse
import html
import json
import math
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

from datasets.object_vocab import DETECTOR_TO_OBJECT, NONE_OBJECT_ID, OBJECT_CLASSES, OBJECT_TO_ID
from live_actor_dashboard import (
    ACTION_CLASSES,
    ActionTrack,
    bbox_iou_xyxy,
    camera_source,
    crop_center_from_boxes,
    local_ip_addresses,
    pack_actor_boxes,
    preprocess_clip,
    required_clip_buffer,
    resize_crop_frame,
    resolve_device,
    sample_clip,
    transform_boxes_to_crop,
)
from utils.actor_tensorrt import TensorRTActorEngine


def parse_args():
    parser = argparse.ArgumentParser(
        description="Live dashboard for object-prompt Actor-Slot PO-GUISE+ models."
    )
    parser.add_argument("--checkpoint", type=str, default="checkpoints/object_actor/epoch=004.ckpt")
    parser.add_argument("--engine", type=str, default=None)
    parser.add_argument("--camera", type=str, default="0")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument(
        "--detector-model-size",
        default="nano",
        choices=["nano", "small", "medium", "base", "large", "xlarge", "2xlarge"],
    )
    parser.add_argument("--detector-weights", type=str, default=None)
    parser.add_argument("--detector-device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--detector-optimize", type=int, default=1, choices=[0, 1])
    parser.add_argument("--detector-compile", type=int, default=1, choices=[0, 1])
    parser.add_argument("--detector-fp16", type=int, default=0, choices=[0, 1])
    parser.add_argument("--person-threshold", type=float, default=0.35)
    parser.add_argument(
        "--object-threshold",
        type=str,
        default=(
            "book=0.35,laptop=0.35,phone=0.45,tv_monitor=0.45,remote=0.35,"
            "keyboard_mouse=0.35,cup=0.35,bottle=0.35,glass=0.35"
        ),
        help="Comma-separated object threshold overrides. Use 'none' for built-in defaults.",
    )
    parser.add_argument("--raw-threshold", type=float, default=0.25)
    parser.add_argument("--person-class-id", type=int, default=None)
    parser.add_argument("--max-actors", type=int, default=8)
    parser.add_argument("--max-objects", type=int, default=24)
    parser.add_argument("--buffer-frames", type=int, default=32)
    parser.add_argument("--clip-frames", type=int, default=16)
    parser.add_argument("--clip-stride", type=int, default=1)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--short-side", type=int, default=256)
    parser.add_argument("--crop-mode", type=str, default="actor", choices=["actor", "center"])
    parser.add_argument("--detect-every", type=int, default=3)
    parser.add_argument("--action-every", type=int, default=5)
    parser.add_argument("--action-smoothing-window", type=int, default=2)
    parser.add_argument("--track-iou-threshold", type=float, default=0.30)
    parser.add_argument("--track-hold-frames", type=int, default=10)
    parser.add_argument("--object-track-iou-threshold", type=float, default=0.20)
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
    if not 0.0 <= args.object_track_iou_threshold <= 1.0:
        raise ValueError("--object-track-iou-threshold must be in [0, 1]")
    return args


def parse_object_thresholds(text):
    thresholds = {name: 0.50 for name in OBJECT_TO_ID}
    text = "" if text is None else str(text).strip()
    if text.lower() in {"", "none", "off", "0"}:
        return thresholds
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid object threshold '{item}'. Expected name=value.")
        name, value = [part.strip() for part in item.split("=", 1)]
        if name not in OBJECT_TO_ID:
            raise ValueError(f"Unknown object class in threshold override: {name}")
        threshold = float(value)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold for {name} must be in [0, 1].")
        thresholds[name] = threshold
    return thresholds


def load_rfdetr(model_size, weights, detector_device, optimize=True, compile_model=True, fp16=False):
    if detector_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--detector-device cuda was requested, but CUDA is not available.")
    if fp16 and detector_device != "cuda":
        raise RuntimeError("--detector-fp16 1 requires --detector-device cuda.")

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    from rfdetr import RFDETRBase, RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall

    model_classes = {
        "nano": RFDETRNano,
        "small": RFDETRSmall,
        "medium": RFDETRMedium,
        "base": RFDETRBase,
        "large": RFDETRLarge,
    }
    if model_size in {"xlarge", "2xlarge"}:
        try:
            from rfdetr_plus import RFDETR2XLarge, RFDETRXLarge
        except ImportError as exc:
            raise RuntimeError(
                f"--detector-model-size {model_size} requires rfdetr_plus."
            ) from exc
        model_classes.update({"xlarge": RFDETRXLarge, "2xlarge": RFDETR2XLarge})

    kwargs = {"device": detector_device}
    if weights:
        kwargs["pretrain_weights"] = weights
    detector = model_classes[model_size](**kwargs)

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
            {"seconds": round(time.perf_counter() - started, 2), "dtype": str(dtype)},
            flush=True,
        )

    class_names = detector_class_names(detector)
    person_ids = detector_person_ids(class_names)
    return detector, class_names, person_ids


def detector_class_names(detector):
    class_names = getattr(detector, "class_names", None)
    if isinstance(class_names, dict):
        return {int(class_id): str(name).lower() for class_id, name in class_names.items()}
    if isinstance(class_names, (list, tuple)):
        return {int(class_id): str(name).lower() for class_id, name in enumerate(class_names)}

    from rfdetr.util.coco_classes import COCO_CLASSES

    if hasattr(COCO_CLASSES, "items"):
        return {int(class_id): str(name).lower() for class_id, name in COCO_CLASSES.items()}
    return {int(class_id): str(name).lower() for class_id, name in enumerate(COCO_CLASSES)}


def detector_person_ids(class_names, explicit_id=None):
    if explicit_id is not None:
        return {int(explicit_id)}
    person_ids = {class_id for class_id, name in class_names.items() if name == "person"}
    if not person_ids:
        raise RuntimeError("Detector class map did not expose a 'person' class.")
    return person_ids


def detections_to_people_and_objects(
    detector,
    class_names,
    person_ids,
    frame_rgb,
    raw_threshold,
    person_threshold,
    object_thresholds,
    max_actors,
):
    height, width = frame_rgb.shape[:2]
    detections = detector.predict(Image.fromarray(frame_rgb), threshold=raw_threshold)
    xyxy = np.asarray(getattr(detections, "xyxy", []), dtype=np.float32)
    class_id = getattr(detections, "class_id", None)
    confidence = getattr(detections, "confidence", None)
    if class_id is None or confidence is None:
        raise RuntimeError("RF-DETR detections must expose class_id and confidence.")

    class_id = np.asarray(class_id)
    confidence = np.asarray(confidence, dtype=np.float32)
    people = []
    objects = []
    for box, cls_value, conf in zip(xyxy, class_id, confidence):
        det_cls_id = int(cls_value)
        det_conf = float(conf)
        det_name = class_names.get(det_cls_id)
        if det_name is None:
            continue
        x1, y1, x2, y2 = [float(v) for v in box.tolist()]
        x1 = max(0.0, min(float(width), x1))
        y1 = max(0.0, min(float(height), y1))
        x2 = max(0.0, min(float(width), x2))
        y2 = max(0.0, min(float(height), y2))
        if x2 <= x1 or y2 <= y1:
            continue

        if det_cls_id in person_ids:
            if det_conf >= person_threshold:
                people.append((det_conf, [x1, y1, x2, y2]))
            continue

        object_name = DETECTOR_TO_OBJECT.get(det_name, det_name)
        object_cls_id = OBJECT_TO_ID.get(object_name)
        if object_cls_id is None:
            continue
        if det_conf < object_thresholds[object_name]:
            continue
        objects.append(
            {
                "xyxy": np.asarray([x1, y1, x2, y2], dtype=np.float32),
                "conf": det_conf,
                "cls_id": int(object_cls_id),
                "cls": object_name,
                "detector_cls": det_name,
            }
        )

    people.sort(key=lambda item: item[0], reverse=True)
    people = people[:max_actors]
    person_boxes = np.asarray([item[1] for item in people], dtype=np.float32)
    person_conf = np.asarray([item[0] for item in people], dtype=np.float32)
    if len(person_boxes) == 0:
        person_boxes = np.zeros((0, 4), dtype=np.float32)
        person_conf = np.zeros((0,), dtype=np.float32)
    return person_boxes, person_conf, objects


def _track_object_entries(entries, iou_threshold):
    tracks = []
    for entry in sorted(entries, key=lambda item: item["sample_pos"]):
        best_track = None
        best_iou = 0.0
        for track in tracks:
            if int(track["cls_id"]) != int(entry["cls_id"]):
                continue
            iou = bbox_iou_xyxy(track["last_box"], entry["xyxy"])
            if iou > best_iou:
                best_iou = iou
                best_track = track
        if best_track is not None and best_iou >= iou_threshold:
            best_track["boxes"].append(entry["xyxy"])
            best_track["confs"].append(float(entry["conf"]))
            best_track["frames"].add(int(entry["sample_pos"]))
            best_track["last_box"] = entry["xyxy"]
        else:
            tracks.append(
                {
                    "cls_id": int(entry["cls_id"]),
                    "cls": entry["cls"],
                    "boxes": [entry["xyxy"]],
                    "confs": [float(entry["conf"])],
                    "frames": {int(entry["sample_pos"])},
                    "last_box": entry["xyxy"],
                }
            )
    return tracks


def pack_object_boxes(objects, transform, input_size, max_objects, device, object_track_iou):
    object_boxes = torch.zeros((1, max_objects, 4), dtype=torch.float32, device=device)
    object_cls = torch.full(
        (1, max_objects), NONE_OBJECT_ID, dtype=torch.long, device=device
    )
    object_conf = torch.zeros((1, max_objects), dtype=torch.float32, device=device)
    object_valid = torch.zeros((1, max_objects), dtype=torch.bool, device=device)
    if not objects:
        return object_boxes, object_cls, object_conf, object_valid, []

    raw_boxes = np.stack([item["xyxy"] for item in objects], axis=0).astype(np.float32)
    boxes_norm, keep = transform_boxes_to_crop(raw_boxes, transform, input_size)

    entries = []
    for obj, box_norm, valid in zip(objects, boxes_norm, keep):
        if not bool(valid):
            continue
        pixel_box = (box_norm * float(input_size)).astype(np.float32)
        entries.append(
            {
                "sample_pos": 0,
                "cls_id": int(obj["cls_id"]),
                "cls": str(obj["cls"]),
                "conf": float(obj["conf"]),
                "xyxy": pixel_box,
                "box_norm": box_norm.astype(np.float32),
            }
        )

    candidates = []
    for track in _track_object_entries(entries, object_track_iou):
        confs = np.asarray(track["confs"], dtype=np.float32)
        boxes = np.asarray(track["boxes"], dtype=np.float32)
        weights = np.maximum(confs, 1e-4)
        mean_box = (boxes * weights[:, None]).sum(axis=0) / weights.sum()
        score = float(confs.max()) * math.sqrt(max(len(track["frames"]), 1))
        candidates.append(
            {
                "score": score,
                "cls_id": int(track["cls_id"]),
                "cls": str(track["cls"]),
                "conf": float(confs.max()),
                "box_norm": np.clip(mean_box / float(input_size), 0.0, 1.0),
            }
        )
    candidates.sort(key=lambda item: item["score"], reverse=True)

    packed = []
    for slot, candidate in enumerate(candidates[:max_objects]):
        box = candidate["box_norm"].astype(np.float32)
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        object_boxes[0, slot] = torch.from_numpy(box).to(device=device)
        object_cls[0, slot] = int(candidate["cls_id"])
        object_conf[0, slot] = float(candidate["conf"])
        object_valid[0, slot] = True
        packed.append(candidate)
    return object_boxes, object_cls, object_conf, object_valid, packed


class TorchObjectActorBackend:
    def __init__(self, checkpoint_path, device):
        from utils.actor_model import load_actor_model

        self.model, self.hparams = load_actor_model(checkpoint_path, device)
        if not bool(self.hparams.get("object_prompt", 0)):
            raise RuntimeError("Checkpoint is not an object-prompt checkpoint.")
        self.device = device
        self.num_actor_tokens = int(self.hparams.get("num_actor_tokens", 0))
        self.num_object_tokens = int(self.hparams.get("num_object_tokens", 0))
        self.clip_frames = int(self.hparams.get("n_frames", 16))
        self.input_size = 224
        self.backend_name = "pytorch-object"

    def __call__(self, clip, boxes, valid, object_boxes, object_cls, object_conf, object_valid):
        with torch.inference_mode():
            output = self.model(
                clip,
                boxes=boxes,
                valid=valid,
                object_boxes=object_boxes,
                object_cls=object_cls.long(),
                object_conf=object_conf,
                object_valid=object_valid,
            )
            if not isinstance(output, (tuple, list)) or len(output) < 3:
                raise RuntimeError("Object actor model did not return presence logits.")
            logits = output[0]
            presence = output[2]
            if presence is None:
                raise RuntimeError("Object actor model did not return presence logits.")
        return logits, presence


class TensorRTObjectActorBackend:
    def __init__(self, engine_path):
        self.engine = TensorRTActorEngine(engine_path)
        if not self.engine.object_prompt:
            raise RuntimeError("TensorRT engine does not expose object inputs.")
        self.device = self.engine.device
        self.num_actor_tokens = self.engine.num_actor_tokens
        self.num_object_tokens = self.engine.num_object_tokens
        self.clip_frames = self.engine.clip_frames
        self.input_size = self.engine.input_size
        self.backend_name = "tensorrt-object"

    def __call__(self, clip, boxes, valid, object_boxes, object_cls, object_conf, object_valid):
        return self.engine(
            clip,
            boxes,
            valid,
            object_boxes=object_boxes,
            object_cls=object_cls,
            object_conf=object_conf,
            object_valid=object_valid,
        )


def load_actor_backend(args):
    if args.engine:
        return TensorRTObjectActorBackend(args.engine)
    return TorchObjectActorBackend(args.checkpoint, resolve_device(args.device))


def run_actor_smoke(args, actor):
    frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(args.clip_frames)]
    boxes_xyxy = np.asarray([[160.0, 80.0, 480.0, 420.0]], dtype=np.float32)
    objects = [
        {
            "xyxy": np.asarray([230.0, 220.0, 430.0, 365.0], dtype=np.float32),
            "conf": 0.90,
            "cls_id": OBJECT_TO_ID["laptop"],
            "cls": "laptop",
        }
    ]
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
    object_boxes, object_cls, object_conf, object_valid, packed_objects = pack_object_boxes(
        objects,
        transform,
        args.input_size,
        args.max_objects,
        actor.device,
        args.object_track_iou_threshold,
    )
    logits, presence = actor(
        clip,
        boxes,
        valid,
        object_boxes,
        object_cls,
        object_conf,
        object_valid,
    )
    probs = torch.softmax(logits[0, 0], dim=-1)
    print(
        "smoke ok:",
        {
            "checkpoint": args.checkpoint,
            "engine": args.engine,
            "backend": actor.backend_name,
            "device": str(actor.device),
            "valid_slots": int(valid.sum().item()),
            "valid_objects": int(object_valid.sum().item()),
            "packed_objects": packed_objects,
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
            "detector_device": None,
            "last_detector_ms": None,
            "last_actor_ms": None,
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
        label = f"{obj['cls']} {obj['conf']:.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), (220, 160, 20), 1)
        cv2.putText(
            out,
            label,
            (x1, max(18, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (220, 160, 20),
            1,
            cv2.LINE_AA,
        )
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
        if self.actor.num_object_tokens != args.max_objects:
            raise RuntimeError(
                f"--max-objects must match object tokens={self.actor.num_object_tokens}"
            )
        if self.actor.clip_frames != args.clip_frames:
            raise RuntimeError(
                f"--clip-frames must match actor clip length={self.actor.clip_frames}"
            )
        if self.actor.input_size != args.input_size:
            raise RuntimeError(f"--input-size must match actor input size={self.actor.input_size}")

        self.object_thresholds = parse_object_thresholds(args.object_threshold)
        self.detector, self.class_names, detected_person_ids = load_rfdetr(
            args.detector_model_size,
            args.detector_weights,
            args.detector_device,
            optimize=bool(args.detector_optimize),
            compile_model=bool(args.detector_compile),
            fp16=bool(args.detector_fp16),
        )
        self.person_ids = detector_person_ids(self.class_names, args.person_class_id)
        if args.person_class_id is None:
            self.person_ids = detected_person_ids

        self.buffer = deque(maxlen=args.buffer_frames)
        self.frame_count = 0
        self.last_boxes_xyxy = np.zeros((0, 4), dtype=np.float32)
        self.last_det_conf = np.zeros((0,), dtype=np.float32)
        self.last_objects = []
        self.last_detection_frame = None
        self.last_detector_ms = None
        self.last_actor_ms = None
        self.next_track_id = 1
        self.tracks = {}
        self.current_track_ids = []
        self.state.update(
            actor_backend=self.actor.backend_name,
            actor_device=str(self.actor.device),
            detector_device=args.detector_device,
            object_thresholds=self.object_thresholds,
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
                    (
                        self.last_boxes_xyxy,
                        self.last_det_conf,
                        self.last_objects,
                    ) = detections_to_people_and_objects(
                        self.detector,
                        self.class_names,
                        self.person_ids,
                        frame_rgb,
                        self.args.raw_threshold,
                        self.args.person_threshold,
                        self.object_thresholds,
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

                message = (
                    f"buffer {len(self.buffer)}/{self.args.buffer_frames} "
                    f"objects={len(self.last_objects)} stride={self.args.clip_stride}"
                )
                should_run_action = action_due and len(self.last_boxes_xyxy) > 0
                packed_objects = []
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
                    (
                        object_boxes,
                        object_cls,
                        object_conf,
                        object_valid,
                        packed_objects,
                    ) = pack_object_boxes(
                        self.last_objects,
                        transform,
                        self.args.input_size,
                        self.args.max_objects,
                        self.device,
                        self.args.object_track_iou_threshold,
                    )
                    if valid.any():
                        started = time.perf_counter()
                        logits, presence_logits = self.actor(
                            clip,
                            boxes,
                            valid,
                            object_boxes,
                            object_cls,
                            object_conf,
                            object_valid,
                        )
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
                            actors[int(actor_idx)].update(track.action_payload())
                            actors[int(actor_idx)].update(
                                {
                                    "raw_label": ACTION_CLASSES[action_id],
                                    "raw_action_conf": float(action_probs[slot, action_id]),
                                }
                            )
                        message = (
                            f"{self.actor.backend_name} actors={int(valid.sum().item())} "
                            f"objects={int(object_valid.sum().item())} "
                            f"det={self.last_detector_ms:.0f}ms "
                            f"actor={self.last_actor_ms:.0f}ms "
                            f"det_age={det_age} "
                            f"crop={self.args.crop_mode} frame={self.frame_count}"
                        )
                    else:
                        message = "detections outside model crop"
                elif len(self.last_boxes_xyxy) == 0:
                    message = f"no RF-DETR person detections; objects={len(self.last_objects)}"

                overlay = draw_overlay(frame_bgr, actors, self.last_objects, message)
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
                        objects=[
                            {
                                "cls": obj["cls"],
                                "conf": float(obj["conf"]),
                                "xyxy": obj["xyxy"].tolist(),
                            }
                            for obj in self.last_objects
                        ],
                        packed_objects=[
                            {
                                "cls": item["cls"],
                                "conf": float(item["conf"]),
                                "box_norm": item["box_norm"].tolist(),
                            }
                            for item in packed_objects
                        ],
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
<html><head><title>PO-GUISE Object Actor Dashboard</title>
<style>
body{margin:0;background:#101214;color:#f2f4f5;font-family:Arial,sans-serif}
main{max-width:1120px;margin:0 auto;padding:16px}
img{max-width:100%;height:auto;border:1px solid #333}
pre{white-space:pre-wrap;background:#181b1f;padding:12px}
</style></head>
<body><main>
<h2>PO-GUISE Object Actor Dashboard</h2>
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
        if actor.num_object_tokens != args.max_objects:
            raise RuntimeError(
                f"--max-objects must match object tokens={actor.num_object_tokens}"
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
