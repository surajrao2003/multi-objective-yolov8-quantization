"""

From project root:

    python TRT_edge_deployment/trt_inference.py --engine ... --input-size 640 --data-root ...
"""

from __future__ import annotations

import argparse
from pathlib import Path

from trt_runtime import run_trt_benchmark


def main() -> int:
    p = argparse.ArgumentParser(
        description="TensorRT benchmark: images/ + labels/, FPS, size, mAP@0.5 (desktop GPU).",
    )
    p.add_argument("--engine", required=True, help="Path to .engine file.")
    p.add_argument(
        "--input-size",
        type=int,
        choices=(320, 640, 1280),
        required=True,
        help="Letterbox size; must match engine export.",
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
        output_dir="outputfolder_trt",
        device_line="device=gpu (TensorRT)",
    )


if __name__ == "__main__":
    raise SystemExit(main())
