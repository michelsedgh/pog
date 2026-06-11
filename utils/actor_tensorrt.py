import json
from pathlib import Path

import torch


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


class TensorRTActorEngine:
    def __init__(self, engine_path):
        if not torch.cuda.is_available():
            raise RuntimeError("TensorRT actor engine requires CUDA.")

        import tensorrt as trt

        engine_path = Path(engine_path)
        if not engine_path.is_file():
            raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")
        self.engine_path = engine_path
        self.metadata = {}
        metadata_path = Path(str(engine_path) + ".json")
        if metadata_path.is_file():
            self.metadata = json.loads(metadata_path.read_text())
        self.precision = str(self.metadata.get("precision", "unknown"))

        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")
        streamable_weights = int(getattr(self.engine, "streamable_weights_size", 0))
        if streamable_weights > 0 and hasattr(self.engine, "weight_streaming_budget_v2"):
            self.engine.weight_streaming_budget_v2 = streamable_weights
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"Failed to create TensorRT execution context: {engine_path}")

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

        base_inputs = {"video", "boxes", "valid"}
        object_inputs = {
            "object_boxes",
            "object_classes",
            "object_confs",
            "object_valid",
        }
        input_set = set(self.input_names)
        self.scene_object_tokens = object_inputs.issubset(input_set)
        self.actor_object_logit_residual = bool(
            self.metadata.get("actor_object_logit_residual", False)
        )
        self.actor_object_conditioned_action = bool(
            self.metadata.get(
                "actor_object_conditioned_action",
                False,
            )
        )
        self.actor_object_relation = bool(
            self.metadata.get(
                "actor_object_relation",
                self.scene_object_tokens,
            )
        )
        self.actor_object_compatibility_expert = bool(
            self.metadata.get(
                "actor_object_compatibility_expert",
                self.actor_object_relation,
            )
        )
        expected_inputs = base_inputs | (object_inputs if self.scene_object_tokens else set())
        expected_outputs = {"logits", "presence"} | (
            {"object_selection"} if self.scene_object_tokens else set()
        )
        if input_set != expected_inputs:
            raise RuntimeError(
                f"Engine inputs must be {sorted(expected_inputs)}, got {self.input_names}"
            )
        if set(self.output_names) != expected_outputs:
            raise RuntimeError(
                f"Engine outputs must be {sorted(expected_outputs)}, got {self.output_names}"
            )

        video_shape = self.shapes["video"]
        boxes_shape = self.shapes["boxes"]
        valid_shape = self.shapes["valid"]
        logits_shape = self.shapes["logits"]
        presence_shape = self.shapes["presence"]
        if len(video_shape) != 5 or video_shape[0] != 1 or video_shape[2] != 3:
            raise RuntimeError(f"Unsupported video input shape: {video_shape}")
        if boxes_shape[:2] != valid_shape or boxes_shape[-1] != 4:
            raise RuntimeError(
                f"Inconsistent actor input shapes: boxes={boxes_shape}, valid={valid_shape}"
            )
        if logits_shape[:2] != valid_shape or presence_shape != valid_shape:
            raise RuntimeError(
                "Inconsistent actor output shapes: "
                f"logits={logits_shape}, presence={presence_shape}, valid={valid_shape}"
            )

        self.batch_size = video_shape[0]
        self.clip_frames = video_shape[1]
        self.input_size = video_shape[3]
        self.num_actor_tokens = boxes_shape[1]
        self.num_classes = logits_shape[2]
        self.num_scene_object_tokens = 0
        if self.scene_object_tokens:
            object_boxes_shape = self.shapes["object_boxes"]
            object_classes_shape = self.shapes["object_classes"]
            object_confs_shape = self.shapes["object_confs"]
            object_valid_shape = self.shapes["object_valid"]
            object_selection_shape = self.shapes["object_selection"]
            if object_boxes_shape[0] != self.batch_size or object_boxes_shape[-1] != 4:
                raise RuntimeError(f"Unsupported object_boxes shape: {object_boxes_shape}")
            if (
                object_classes_shape != object_boxes_shape[:2]
                or object_confs_shape != object_boxes_shape[:2]
                or object_valid_shape != object_boxes_shape[:2]
            ):
                raise RuntimeError(
                    "Inconsistent object input shapes: "
                    f"object_boxes={object_boxes_shape}, "
                    f"object_classes={object_classes_shape}, "
                    f"object_confs={object_confs_shape}, "
                    f"object_valid={object_valid_shape}"
                )
            if object_selection_shape != (
                self.batch_size,
                self.num_actor_tokens,
                object_boxes_shape[1] + 1,
            ):
                raise RuntimeError(
                    "Inconsistent object_selection shape: "
                    f"{object_selection_shape}, expected "
                    f"{(self.batch_size, self.num_actor_tokens, object_boxes_shape[1] + 1)}"
                )
            self.num_scene_object_tokens = object_boxes_shape[1]
        self.stream = torch.cuda.Stream()

    def _prepare_input(self, tensor, name):
        expected_shape = self.shapes[name]
        expected_dtype = self.dtypes[name]
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}, got {tuple(tensor.shape)}")
        if tensor.dtype != expected_dtype:
            tensor = tensor.to(dtype=expected_dtype)
        if tensor.device.type != "cuda":
            tensor = tensor.to(device=self.device)
        return tensor.contiguous()

    def __call__(self, video, boxes, valid, object_inputs=None):
        video = self._prepare_input(video, "video")
        boxes = self._prepare_input(boxes, "boxes")
        valid = self._prepare_input(valid, "valid")
        if self.scene_object_tokens:
            if object_inputs is None:
                raise RuntimeError(
                    "This TensorRT actor engine requires object inputs: "
                    "object_boxes, object_classes, object_confs, object_valid."
                )
            missing = sorted(
                {
                    "object_boxes",
                    "object_classes",
                    "object_confs",
                    "object_valid",
                }
                - set(object_inputs)
            )
            if missing:
                raise RuntimeError(f"Missing TensorRT object input keys: {missing}")
            object_boxes = self._prepare_input(object_inputs["object_boxes"], "object_boxes")
            object_classes = self._prepare_input(
                object_inputs["object_classes"],
                "object_classes",
            )
            object_confs = self._prepare_input(object_inputs["object_confs"], "object_confs")
            object_valid = self._prepare_input(object_inputs["object_valid"], "object_valid")
        elif object_inputs is not None:
            raise RuntimeError("Object inputs were passed to an actor-only TensorRT engine.")

        logits = torch.empty(
            self.shapes["logits"],
            dtype=self.dtypes["logits"],
            device=self.device,
        )
        presence = torch.empty(
            self.shapes["presence"],
            dtype=self.dtypes["presence"],
            device=self.device,
        )
        object_selection = None
        if self.scene_object_tokens:
            object_selection = torch.empty(
                self.shapes["object_selection"],
                dtype=self.dtypes["object_selection"],
                device=self.device,
            )

        tensors = {
            "video": video,
            "boxes": boxes,
            "valid": valid,
            "logits": logits,
            "presence": presence,
        }
        if self.scene_object_tokens:
            tensors.update(
                {
                    "object_boxes": object_boxes,
                    "object_classes": object_classes,
                    "object_confs": object_confs,
                    "object_valid": object_valid,
                    "object_selection": object_selection,
                }
            )
        current_stream = torch.cuda.current_stream()
        self.stream.wait_stream(current_stream)
        with torch.cuda.stream(self.stream):
            for name, tensor in tensors.items():
                self.context.set_tensor_address(name, tensor.data_ptr())
            ok = self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
        if not ok:
            raise RuntimeError("TensorRT actor execution failed.")
        self.stream.synchronize()
        current_stream.wait_stream(self.stream)
        return logits, presence, object_selection
