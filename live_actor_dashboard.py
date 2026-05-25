import argparse
import gc
import html
import json
import socket
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from models.poguise import POGUISE
from train import _load_checkpoint


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
    parser.add_argument("--det-threshold", type=float, default=0.35)
    parser.add_argument("--person-class-id", type=int, default=None)
    parser.add_argument("--max-actors", type=int, default=8)
    parser.add_argument("--buffer-frames", type=int, default=128)
    parser.add_argument("--clip-frames", type=int, default=16)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--short-side", type=int, default=256)
    parser.add_argument("--detect-every", type=int, default=5)
    parser.add_argument("--action-every", type=int, default=10)
    parser.add_argument("--jpeg-quality", type=int, default=80)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


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


def load_actor_model(checkpoint_path, device):
    checkpoint = _load_checkpoint(checkpoint_path)
    hparams = {}
    hparams.update(checkpoint.get("hyper_parameters", {}))
    hparams.update(checkpoint.get("datamodule_hyper_parameters", {}))
    if not hparams:
        raise RuntimeError(f"No hyperparameters found in checkpoint: {checkpoint_path}")
    if not hparams.get("actor_prompt", 0):
        raise RuntimeError("Checkpoint is not an actor-prompt checkpoint.")
    hparams["pretrained"] = "none"
    hparams["mode"] = "test"
    hparams["ret_feat"] = 0

    model = POGUISE(**hparams)
    state_dict = {
        key.removeprefix("model."): value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith("model.")
    }
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint load mismatch. Missing={missing}, unexpected={unexpected}")
    del checkpoint
    del state_dict
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    model.to(device=device, dtype=torch.float32)
    model.eval()
    return model, hparams


def load_rfdetr(person_class_id):
    from rfdetr import RFDETRNano

    detector = RFDETRNano()
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


def resize_crop_frame(frame_rgb, short_side, input_size):
    height, width = frame_rgb.shape[:2]
    if width < height:
        new_width = short_side
        new_height = int(np.floor(float(height) / width * short_side))
    else:
        new_height = short_side
        new_width = int(np.floor(float(width) / height * short_side))

    resized = cv2.resize(frame_rgb, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    x_offset = int(np.ceil((new_width - input_size) / 2.0))
    y_offset = int(np.ceil((new_height - input_size) / 2.0))
    crop = resized[y_offset : y_offset + input_size, x_offset : x_offset + input_size]
    transform = {
        "scale_x": new_width / float(width),
        "scale_y": new_height / float(height),
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


def preprocess_clip(frames_rgb, short_side, input_size, device):
    crops = []
    transform = None
    for frame_rgb in frames_rgb:
        crop, transform = resize_crop_frame(frame_rgb, short_side, input_size)
        crop = crop.astype(np.float32) / 255.0
        crop = (crop - np.asarray([0.485, 0.456, 0.406], dtype=np.float32)) / np.asarray(
            [0.229, 0.224, 0.225],
            dtype=np.float32,
        )
        crops.append(torch.from_numpy(crop).permute(2, 0, 1))
    clip = torch.stack(crops, dim=0).unsqueeze(0).to(device=device, dtype=torch.float32)
    return clip, transform


def sample_clip(buffer, clip_frames):
    indices = np.linspace(0, len(buffer) - 1, clip_frames, dtype=int)
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


def run_actor_smoke(args, model, device):
    frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(args.clip_frames)]
    boxes_xyxy = np.asarray([[160.0, 80.0, 480.0, 420.0]], dtype=np.float32)
    clip, transform = preprocess_clip(
        frames,
        args.short_side,
        args.input_size,
        device,
    )
    boxes_norm, keep = transform_boxes_to_crop(boxes_xyxy, transform, args.input_size)
    boxes, valid = pack_actor_boxes(boxes_norm, keep, args.max_actors, device)
    with torch.inference_mode():
        logits, _, presence = model(clip, boxes=boxes, valid=valid)
    probs = torch.softmax(logits[0, 0], dim=-1)
    print(
        "smoke ok:",
        {
            "checkpoint": args.checkpoint,
            "device": str(device),
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
        self.device = resolve_device(args.device)
        self.model, self.hparams = load_actor_model(args.checkpoint, self.device)
        checkpoint_tokens = int(self.hparams.get("num_actor_tokens", args.max_actors))
        if checkpoint_tokens != args.max_actors:
            raise RuntimeError(
                f"--max-actors must match checkpoint num_actor_tokens={checkpoint_tokens}"
            )
        self.detector, self.person_ids = load_rfdetr(args.person_class_id)
        self.buffer = deque(maxlen=args.buffer_frames)
        self.frame_count = 0
        self.last_boxes_xyxy = np.zeros((0, 4), dtype=np.float32)
        self.last_det_conf = np.zeros((0,), dtype=np.float32)
        self.last_actions = []

    def smoke(self):
        run_actor_smoke(self.args, self.model, self.device)

    def run(self):
        cap = cv2.VideoCapture(camera_source(self.args.camera))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera source: {self.args.camera}")

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

                if self.frame_count % self.args.detect_every == 0:
                    self.last_boxes_xyxy, self.last_det_conf = detections_to_people(
                        self.detector,
                        self.person_ids,
                        frame_rgb,
                        self.args.det_threshold,
                        self.args.max_actors,
                    )

                message = f"buffer {len(self.buffer)}/{self.args.buffer_frames}"
                actors = [
                    {
                        "xyxy": box.tolist(),
                        "det_conf": float(conf),
                        "label": "person",
                    }
                    for box, conf in zip(self.last_boxes_xyxy, self.last_det_conf)
                ]

                should_run_action = (
                    len(self.buffer) >= self.args.clip_frames
                    and len(self.last_boxes_xyxy) > 0
                    and self.frame_count % self.args.action_every == 0
                )
                if should_run_action:
                    clip_frames = sample_clip(self.buffer, self.args.clip_frames)
                    clip, transform = preprocess_clip(
                        clip_frames,
                        self.args.short_side,
                        self.args.input_size,
                        self.device,
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
                        with torch.inference_mode():
                            logits, _, presence_logits = self.model(
                                clip,
                                boxes=boxes,
                                valid=valid,
                            )
                        action_probs = torch.softmax(logits[0], dim=-1).detach().cpu().numpy()
                        presence_probs = (
                            torch.sigmoid(presence_logits[0]).detach().cpu().numpy()
                        )
                        kept_actor_idx = np.flatnonzero(keep)[: self.args.max_actors]
                        for slot, actor_idx in enumerate(kept_actor_idx):
                            action_id = int(action_probs[slot].argmax())
                            actors[int(actor_idx)].update(
                                {
                                    "label": ACTION_CLASSES[action_id],
                                    "action_conf": float(action_probs[slot, action_id]),
                                    "presence": float(presence_probs[slot]),
                                }
                            )
                        message = f"actors={int(valid.sum().item())} frame={self.frame_count}"
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
        device = resolve_device(args.device)
        model, hparams = load_actor_model(args.checkpoint, device)
        checkpoint_tokens = int(hparams.get("num_actor_tokens", args.max_actors))
        if checkpoint_tokens != args.max_actors:
            raise RuntimeError(
                f"--max-actors must match checkpoint num_actor_tokens={checkpoint_tokens}"
            )
        run_actor_smoke(args, model, device)
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
