from pathlib import Path

import numpy as np
import torch


COCO_CLASSES = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic light",
    11: "fire hydrant",
    13: "stop sign",
    14: "parking meter",
    15: "bench",
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",
    27: "backpack",
    28: "umbrella",
    31: "handbag",
    32: "tie",
    33: "suitcase",
    34: "frisbee",
    35: "skis",
    36: "snowboard",
    37: "sports ball",
    38: "kite",
    39: "baseball bat",
    40: "baseball glove",
    41: "skateboard",
    42: "surfboard",
    43: "tennis racket",
    44: "bottle",
    46: "wine glass",
    47: "cup",
    48: "fork",
    49: "knife",
    50: "spoon",
    51: "bowl",
    52: "banana",
    53: "apple",
    54: "sandwich",
    55: "orange",
    56: "broccoli",
    57: "carrot",
    58: "hot dog",
    59: "pizza",
    60: "donut",
    61: "cake",
    62: "chair",
    63: "couch",
    64: "potted plant",
    65: "bed",
    67: "dining table",
    70: "toilet",
    72: "tv",
    73: "laptop",
    74: "mouse",
    75: "remote",
    76: "keyboard",
    77: "cell phone",
    78: "microwave",
    79: "oven",
    80: "toaster",
    81: "sink",
    82: "refrigerator",
    84: "book",
    85: "clock",
    86: "vase",
    87: "scissors",
    88: "teddy bear",
    89: "hair drier",
    90: "toothbrush",
}


def _torch_dtype(trt_dtype):
    import tensorrt as trt

    mapping = {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.int32: torch.int32,
        trt.int64: torch.int64,
        trt.bool: torch.bool,
    }
    if trt_dtype not in mapping:
        raise TypeError(f"Unsupported TensorRT dtype: {trt_dtype}")
    return mapping[trt_dtype]


def _box_cxcywh_to_xyxy(boxes):
    cx, cy, w, h = boxes.unbind(-1)
    half_w = 0.5 * w
    half_h = 0.5 * h
    return torch.stack((cx - half_w, cy - half_h, cx + half_w, cy + half_h), dim=-1)


class TensorRTRFDETRNano:
    def __init__(self, engine_path, num_select=300):
        if not torch.cuda.is_available():
            raise RuntimeError("TensorRT RF-DETR engine requires CUDA.")

        import tensorrt as trt

        engine_path = Path(engine_path)
        if not engine_path.is_file():
            raise FileNotFoundError(f"TensorRT RF-DETR engine not found: {engine_path}")

        self.class_names = dict(COCO_CLASSES)
        self.num_select = int(num_select)
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"Failed to create TensorRT context: {engine_path}")

        self.device = torch.device("cuda")
        self.input_names = []
        self.output_names = []
        self.shapes = {}
        self.dtypes = {}
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            mode = self.engine.get_tensor_mode(name)
            shape = tuple(int(dim) for dim in self.engine.get_tensor_shape(name))
            dtype = _torch_dtype(self.engine.get_tensor_dtype(name))
            self.shapes[name] = shape
            self.dtypes[name] = dtype
            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            elif mode == trt.TensorIOMode.OUTPUT:
                self.output_names.append(name)
            else:
                raise RuntimeError(f"Unknown TensorRT tensor mode for {name}: {mode}")

        if len(self.input_names) != 1:
            raise RuntimeError(f"RF-DETR engine must have one input, got {self.input_names}")
        if set(self.output_names) != {"dets", "labels"}:
            raise RuntimeError(
                f"RF-DETR engine outputs must be ['dets', 'labels'], got {self.output_names}"
            )

        self.input_name = self.input_names[0]
        input_shape = self.shapes[self.input_name]
        dets_shape = self.shapes["dets"]
        labels_shape = self.shapes["labels"]
        if len(input_shape) != 4 or input_shape[0] != 1 or input_shape[1] != 3:
            raise RuntimeError(f"Unsupported RF-DETR input shape: {input_shape}")
        if len(dets_shape) != 3 or dets_shape[0] != 1 or dets_shape[-1] != 4:
            raise RuntimeError(f"Unsupported RF-DETR dets shape: {dets_shape}")
        if len(labels_shape) != 3 or labels_shape[:2] != dets_shape[:2]:
            raise RuntimeError(
                f"Inconsistent RF-DETR outputs: dets={dets_shape}, labels={labels_shape}"
            )
        self.resolution = int(input_shape[-1])
        if input_shape[-2] != self.resolution:
            raise RuntimeError(f"RF-DETR input must be square, got {input_shape}")
        self.stream = torch.cuda.Stream()

    def _prepare_image(self, image):
        from PIL import Image
        import torchvision.transforms.functional as F

        if isinstance(image, Image.Image):
            pil_image = image.convert("RGB")
        else:
            array = np.asarray(image)
            if array.ndim != 3 or array.shape[2] != 3:
                raise ValueError(f"Expected RGB image with shape HxWx3, got {array.shape}")
            pil_image = Image.fromarray(array.astype(np.uint8), mode="RGB")

        orig_h, orig_w = pil_image.height, pil_image.width
        tensor = F.to_tensor(pil_image)
        if (tensor > 1).any():
            raise ValueError("RF-DETR input image tensor must be normalized to [0, 1].")
        tensor = F.normalize(
            tensor,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
        tensor = F.resize(tensor, (self.resolution, self.resolution))
        tensor = tensor.unsqueeze(0).to(
            device=self.device,
            dtype=self.dtypes[self.input_name],
        )
        return tensor.contiguous(), (orig_h, orig_w)

    def _run_raw(self, batch):
        batch = batch.contiguous()
        dets = torch.empty(
            self.shapes["dets"],
            dtype=self.dtypes["dets"],
            device=self.device,
        )
        labels = torch.empty(
            self.shapes["labels"],
            dtype=self.dtypes["labels"],
            device=self.device,
        )
        tensors = {
            self.input_name: batch,
            "dets": dets,
            "labels": labels,
        }
        current_stream = torch.cuda.current_stream()
        self.stream.wait_stream(current_stream)
        with torch.cuda.stream(self.stream):
            for name, tensor in tensors.items():
                self.context.set_tensor_address(name, tensor.data_ptr())
            ok = self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
        if not ok:
            raise RuntimeError("TensorRT RF-DETR execution failed.")
        self.stream.synchronize()
        current_stream.wait_stream(self.stream)
        return dets, labels

    def predict(self, image, threshold=0.5):
        import supervision as sv

        batch, (orig_h, orig_w) = self._prepare_image(image)
        dets, labels = self._run_raw(batch)

        logits = labels.float()
        boxes_cxcywh = dets.float()
        probs = logits.sigmoid().view(1, -1)
        k = min(int(self.num_select), int(probs.shape[1]))
        scores, indexes = torch.topk(probs, k, dim=1)
        topk_boxes = indexes // logits.shape[-1]
        class_ids = indexes % logits.shape[-1]
        boxes = _box_cxcywh_to_xyxy(boxes_cxcywh)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).expand(-1, -1, 4))
        scale = torch.tensor(
            [orig_w, orig_h, orig_w, orig_h],
            dtype=torch.float32,
            device=self.device,
        )
        boxes = boxes * scale[None, None, :]

        keep = scores[0] > float(threshold)
        boxes_np = boxes[0, keep].detach().cpu().numpy().astype(np.float32)
        scores_np = scores[0, keep].detach().cpu().numpy().astype(np.float32)
        class_np = class_ids[0, keep].detach().cpu().numpy().astype(np.int64)
        if len(boxes_np) == 0:
            boxes_np = np.zeros((0, 4), dtype=np.float32)
            scores_np = np.zeros((0,), dtype=np.float32)
            class_np = np.zeros((0,), dtype=np.int64)
        return sv.Detections(
            xyxy=boxes_np,
            confidence=scores_np,
            class_id=class_np,
        )
