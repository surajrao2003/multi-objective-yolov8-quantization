"""
TensorRT inference on NVIDIA Jetson Orin Nano (8 GB).

Same CLI and metrics as trt_inference.py (FPS, engine size, mAP@0.5; images/ + labels/).
Writes overlays to outputfolder_trt_jetson/.

Important for Jetson
----------------------
- Build the .engine **on the Orin** (or for its exact GPU architecture). A TensorRT
  engine built on an x86 PC will generally **not** load on Jetson.
- Use JetPack’s TensorRT / PyCUDA (or pip wheels built for aarch64 + your JetPack).

From project root on the Jetson (Linux, L4T):

    python3 TRT_edge_deployment/trt_inference_jetson.py \\
      --engine models/trt_engines/yolov8n_640.engine \\
      --input-size 640 \\
      --data-root imagedata
"""

from __future__ import annotations

import argparse
from pathlib import Path

from trt_runtime import run_trt_benchmark


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "TensorRT benchmark on Jetson Orin Nano: images/ + labels/, "
            "FPS, size, mAP@0.5."
        ),
    )
    p.add_argument("--engine", required=True, help="Path to .engine (built on this Jetson).")
    p.add_argument(
        "--input-size",
        type=int,
        choices=(320, 640, 1280),
        required=True,
        help="Letterbox size; must match how the ONNX/engine was exported.",
    )
    p.add_argument(
        "--data-root",
        required=True,
        help="Folder with images/ and labels/ (same as main_execution.py).",
    )
    args = p.parse_args()

    return run_trt_benchmark(
        engine=Path(args.engine).resolve(),
        data_root=Path(args.data_root).resolve(),
        input_size=(args.input_size, args.input_size),
        output_dir="outputfolder_trt_jetson",
        device_line="device=gpu (TensorRT, Jetson Orin Nano 8GB)",
    )


if __name__ == "__main__":
    raise SystemExit(main())
