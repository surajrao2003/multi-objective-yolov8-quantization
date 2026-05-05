"""Shared TensorRT session + benchmark loop for trt_inference*.py entry points."""

from __future__ import annotations

import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class _TensorBinding:
    name: str
    is_input: bool
    shape: tuple[int, ...]
    dtype: Any
    host: Any
    device: Any


class TrtYoloSession:
    """Deserialize a TensorRT engine and run batch-1 inference (ORT-like output list)."""

    def __init__(self, engine_path: str) -> None:
        import pycuda.autoinit  # noqa: F401
        import pycuda.driver as cuda
        import tensorrt as trt

        self._cuda = cuda
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        data = Path(engine_path).read_bytes()
        engine = runtime.deserialize_cuda_engine(data)
        if engine is None:
            raise RuntimeError("deserialize_cuda_engine returned None")
        self._context = engine.create_execution_context()
        self._stream = cuda.Stream()
        self._stream_handle = int(
            getattr(self._stream, "handle", getattr(self._stream, "ptr", 0))
        )
        self._bindings: list[_TensorBinding] = []
        self._name_to_binding: dict[str, _TensorBinding] = {}

        for i in range(int(engine.num_io_tensors)):
            name = engine.get_tensor_name(i)
            mode = engine.get_tensor_mode(name)
            is_input = mode == trt.TensorIOMode.INPUT
            shape = tuple(self._context.get_tensor_shape(name))
            trt_dtype = engine.get_tensor_dtype(name)
            np_dtype = np.dtype(trt.nptype(trt_dtype))
            vol = int(np.prod(shape)) if shape else 0
            host = cuda.pagelocked_empty(vol, np_dtype)
            device = cuda.mem_alloc(host.nbytes)
            b = _TensorBinding(name, is_input, shape, np_dtype, host, device)
            self._bindings.append(b)
            self._name_to_binding[name] = b

        self.input_names = [b.name for b in self._bindings if b.is_input]
        if len(self.input_names) != 1:
            raise RuntimeError(
                f"Expected exactly 1 TRT input tensor; got {self.input_names!r}"
            )

    @property
    def input_feed_dtype(self):
        return self._name_to_binding[self.input_names[0]].dtype

    def infer(self, input_tensor: np.ndarray) -> list[np.ndarray]:
        in_name = self.input_names[0]
        b_in = self._name_to_binding[in_name]
        if input_tensor.shape != b_in.shape or input_tensor.dtype != b_in.dtype:
            raise ValueError("input shape/dtype must match engine")

        np.copyto(b_in.host, input_tensor.ravel())
        sh = self._stream_handle or 0
        if sh:
            self._cuda.memcpy_htod_async(b_in.device, b_in.host, self._stream)
        else:
            self._cuda.memcpy_htod(b_in.device, b_in.host)

        for b in self._bindings:
            self._context.set_tensor_address(b.name, int(b.device))

        if not self._context.execute_async_v3(sh):
            raise RuntimeError("execute_async_v3 returned False")

        for b in self._bindings:
            if not b.is_input:
                if sh:
                    self._cuda.memcpy_dtoh_async(b.host, b.device, self._stream)
                else:
                    self._cuda.memcpy_dtoh(b.host, b.device)
        if sh:
            self._stream.synchronize()

        outs: list[np.ndarray] = []
        for b in self._bindings:
            if b.is_input:
                continue
            outs.append(np.array(b.host).reshape(b.shape).copy())
        return outs


def run_trt_benchmark(
    *,
    engine: Path,
    data_root: Path,
    input_size: tuple[int, int],
    output_dir: str,
    device_line: str,
) -> int:
    """
    Same dataset contract as code_files/main_execution.py (images/ + labels/).
    """
    if not engine.is_file():
        print(f"Engine not found: {engine.resolve()}", file=sys.stderr)
        return 1
    images_dir, labels_dir = data_root / "images", data_root / "labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        print(
            f"Expected {data_root}/images and {data_root}/labels (YOLO layout).",
            file=sys.stderr,
        )
        return 1

    sys.path.insert(0, str(ROOT / "code_files"))
    from inference import draw_predictions, format_model_disk_size
    from metrics_map import load_yolo_ground_truth, mean_average_precision_person
    from postprocessing import postprocess_and_log_outputs
    from preprocessing import preprocess_frame

    iou, conf, map_iou = 0.7, 0.3, 0.5
    person_class_id = 0
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")

    session = TrtYoloSession(str(engine))
    feed_dtype = session.input_feed_dtype

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(images_dir) if f.lower().endswith(exts))
    total_time, total_frames = 0.0, 0
    list_gt: list = []
    list_preds: list = []

    for image_file in files:
        path = images_dir / image_file
        stem = path.stem
        t0 = time.time()
        input_frame = cv2.imread(str(path))
        if input_frame is None:
            continue

        pre, _orig, shape_hw, scale, pad = preprocess_frame(
            input_frame, input_size, feed_dtype=feed_dtype
        )
        outs = session.infer(pre)
        _img, dets = postprocess_and_log_outputs(
            input_frame,
            outs,
            shape_hw,
            scale,
            pad,
            conf_threshold=conf,
            iou_threshold=iou,
            person_class_id=person_class_id,
        )
        if not dets:
            boxes = np.zeros((0, 4))
            scores = np.zeros((0,), dtype=np.float32)
            class_ids = np.zeros((0,), dtype=np.int64)
        else:
            boxes = np.array([d[:4] for d in dets])
            scores = np.array([d[4] for d in dets], dtype=np.float32)
            class_ids = np.full(len(boxes), person_class_id, dtype=np.int64)

        out_img = draw_predictions(input_frame, boxes, scores, class_ids)
        total_time += time.time() - t0
        total_frames += 1
        h, w = shape_hw
        list_gt.append(
            load_yolo_ground_truth(labels_dir, stem, h, w, person_class_id)
        )
        list_preds.append((boxes, scores))
        cv2.imwrite(os.path.join(output_dir, f"{stem}.jpg"), out_img)

    sz = format_model_disk_size(str(engine))
    if total_frames == 0:
        print("FPS: N/A (no images processed)")
        print(device_line)
        print(f"Engine size (on disk): {sz}")
        print(f"mAP@IoU={map_iou:.2f} : N/A (no images processed)")
        return 0

    fps = total_frames / total_time
    ap, n_gt, _ = mean_average_precision_person(list_gt, list_preds, iou_match=map_iou)
    print(f"FPS: {fps:.4f}")
    print(device_line)
    print(f"Engine size (on disk): {sz}")
    if n_gt == 0:
        print(
            f"mAP@IoU={map_iou:.2f} : undefined (zero person boxes in GT labels)"
        )
    elif np.isnan(ap):
        print(f"mAP@IoU={map_iou:.2f} : nan")
    else:
        print(f"mAP@IoU={map_iou:.2f} : {ap:.6f}")
    return 0
