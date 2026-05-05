"""
Run YOLOv8 ONNX inference on a YOLO-style image folder (same layout as imagedata/).

Layout
------
Your --data-root folder must contain two subfolders:

    <data-root>/images/   ... image files (.jpg, .png, etc.)
    <data-root>/labels/   ... YOLO .txt label files used for mAP (person class 0 vs predictions)

Example: if your tree is imagedata/images and imagedata/labels, pass --data-root imagedata

How to run
----------
1. Open your terminal in the project root (folder that contains "code_files" and "models").
2. Pass --model, --device, --input-size, and --data-root.

Example (CPU):

    python code_files/main_execution.py --model models/yolo_onnx_models/FP32_onnx_models/640/yolov8n_640.onnx --device cpu --input-size 640 --data-root imagedata

Windows PowerShell example:

    python .\\code_files\\main_execution.py --model ".\\models\\yolo_onnx_models\\FP32_onnx_models\\640\\yolov8n_640.onnx" --device cpu --input-size 640 --data-root ".\\imagedata"

After a run you get FPS, device=cpu|gpu, model size on disk, and person mAP@0.5 (see code_files/inference.py). Annotated images go to outputfolder/

For all arguments:

    python code_files/main_execution.py --help
"""

import argparse
import os
import shutil
from pathlib import Path

from inference import initialize_model, run_image_inference

OUTPUT_DIR = "outputfolder"

IOU = 0.7
CONF = 0.3


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLOv8 ONNX inference on images (YOLO folder layout: images/ + labels/).",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the ONNX model file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=("cpu", "gpu"),
        required=True,
        help="Inference device for ONNX Runtime (cpu or gpu).",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        choices=(320, 640, 1280),
        required=True,
        help="Letterbox resolution (square). Must match how the ONNX was exported.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Dataset root folder containing 'images' and 'labels' subfolders (like imagedata/).",
    )

    args = parser.parse_args()

    model_file = Path(args.model)
    if not model_file.is_file():
        parser.error(f"Model file not found: {model_file.resolve()}")

    root = Path(args.data_root)
    if not root.is_dir():
        parser.error(f"Data root not found or not a folder: {root.resolve()}")

    images_dir = root / "images"
    labels_dir = root / "labels"
    if not images_dir.is_dir():
        parser.error(
            f"Missing 'images' folder under data root. Expected: {images_dir.resolve()}"
        )
    if not labels_dir.is_dir():
        parser.error(
            f"Missing 'labels' folder under data root. Expected: {labels_dir.resolve()}"
        )

    return args


def main() -> None:
    args = _parse_args()
    model_path = str(Path(args.model))
    device = args.device.lower()
    input_size = (args.input_size, args.input_size)
    images_dir = str(Path(args.data_root) / "images")
    labels_dir = str(Path(args.data_root) / "labels")

    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ort_model = initialize_model(model_path, device)

    run_image_inference(
        ort_model,
        images_dir,
        labels_dir,
        OUTPUT_DIR,
        input_size,
        CONF,
        IOU,
        model_path=model_path,
    )


if __name__ == "__main__":
    main()
