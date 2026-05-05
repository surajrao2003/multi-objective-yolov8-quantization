# Optimization-Based Quantization Strategy Selection for YOLOv8 Edge Deployment

This repo compares **YOLOv8** ONNX builds on **speed** (FPS), **model file size**, and a **person detection metric** (mAP at IoU 0.5). Lower precision (FP16, INT8) and smaller inputs usually shrink the model and can speed things up, but accuracy and GPU vs CPU behavior depend on your setup.

The project goal is to **find an optimized deployment configuration** (model size, quantization/precision, input resolution, and hardware target) under edge constraints, then **select it as the recommended** quantization strategy for edge deployment.

### What we cover (conceptually)

Typical experiments vary:

- **m** ∈ {YOLOv8n, YOLOv8s, YOLOv8m} - model size  
- **q** ∈ {FP32, FP16, INT8 Dynamic, INT8 Static} - numerical precision / quantization  
- **r** ∈ {320, 640, 1280} - square input size used at export  
- **h** ∈ {CPU1, GPU1, CPU2, GPU2} - benchmark machine (see **Hardware** below). **Edge TensorRT** on Jetson Orin Nano 8GB is a separate on-device step after optimization (not a column in the benchmark CSV unless you add it yourself).

Together **(m, q, r, h)** defines one ONNX configuration to benchmark. There is rarely one best combo for everyone; this project keeps exports and scripts repeatable so you can compare runs fairly.

### Hardware (benchmark targets)

**Machine A (laptop)**

- **CPU1:** 13th Gen Intel(R) Core(TM) i5-13420H, 8 cores, 12 logical processors  
- **GPU1:** NVIDIA GeForce RTX 4050, 6GB VRAM  
- **CUDA version:** 12.8

**Machine B (workstation)**

- **CPU2:** Intel(R) Xeon(R) w5-3435X, 16 cores, 32 logical processors  
- **GPU2:** NVIDIA RTX 4000 Ada Generation, 20GB VRAM  
- **CUDA version:** 12.8

**Edge device** 

- **NVIDIA Jetson Orin Nano (8 GB)** - We use the NVIDIA Jetson Orin Nano, an Arm-based SoC with an integrated NVIDIA GPU and 8 GB unified memory, to evaluate edge deployment. The optimal YOLOv8 configuration obtained from the optimization framework is deployed on this device and benchmarked to assess its performance under resource-constrained conditions.

### Repo layout


| Path                                          | Role                                                                                               |
| --------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| `code_files/main_execution.py`                | CLI: run inference on a folder of images + labels                                                  |
| `code_files/inference.py`                     | Loads the ONNX session, runs images, prints FPS / device / size / mAP                              |
| `quantization_code_files/`                    | Scripts to export FP32, FP16, INT8 dynamic, INT8 static ONNX                                       |
| `optimization/run_optimization.py`            | One-command pipeline: generate benchmark CSV (using csv_generator) + compute utility ranking       |
| `optimization/output_csv_results/`            | Outputs: scored table, report and trade-off plots                                                  |
| `models/yolo_onnx_models/`                    | ONNX files grouped by precision type and resolution                                                |
| `models/trt_engines/`                         | Optional: TensorRT `.engine` files to test an optimized configuration from the optimization report |
| `models/config_custom_data.yaml`              | Class names for dataset config                                                                     |
| `TRT_edge_deployment/build_trt_engine.py`     | After the report: build a TensorRT `.engine` from the chosen ONNX (FP32 / FP16)                    |
| `TRT_edge_deployment/trt_inference.py`        | Optional: TensorRT on desktop GPU (FPS, size, mAP; `outputfolder_trt/`)                            |
| `TRT_edge_deployment/trt_inference_jetson.py` | Optional: same on **Jetson Orin Nano** (`outputfolder_trt_jetson/`)                                |
| `TRT_edge_deployment/trt_runtime.py`          | Shared TensorRT session + benchmark for both inference scripts above                               |


### ONNX folders under `models/yolo_onnx_models/`

Each precision type uses the same layout: `**<subfolder>/<320 \| 640 \| 1280>/`** holds ONNX files for that input size. Naming patterns match the export scripts.


| Subfolder                               | What goes here                                         | Built with                                                                       |
| --------------------------------------- | ------------------------------------------------------ | -------------------------------------------------------------------------------- |
| `FP32_onnx_models/<size>/`              | FP32 ONNX (baseline, no quantization)                  | `quantization_code_files/FP32_onnx_export.py`                                    |
| `FP16_quantized_models/<size>/`         | FP16 quantization                                      | `quantization_code_files/FP16_quantization/export_fp16_onnx.py`                  |
| `INT8_dynamic_quantized_models/<size>/` | INT8 dynamic quantization                              | `quantization_code_files/INT8_dynamic_quantization/dynamic_quantization_onnx.py` |
| `INT8_static_quantized_models/<size>/`  | INT8 static quantization (calibration images required) | `quantization_code_files/INT8_static_quantization/static_quantization_onnx.py`   |

**Note:** This repository ships sample **YOLOv8n** ONNX checkpoints at input sizes **320, 640, and 1280** for each precision folder above; other sizes and variants can be produced with the export scripts.

### Project structure

```
multi-objective-yolov8-quantization/
├── code_files/
│   ├── main_execution.py      # CLI entry
│   ├── inference.py           # ORT session + batch inference + metrics
│   ├── preprocessing.py
│   ├── postprocessing.py
│   └── metrics_map.py
├── quantization_code_files/                # contains export and quantization scripts
│   ├── FP32_onnx_export.py                 # pytorch -> onnx (default fp32 precision)
│   ├── FP16_quantization/             
│   │   └── export_fp16_onnx.py             # fp16 quantization script
│   ├── INT8_dynamic_quantization/          
│   │   └── dynamic_quantization_onnx.py    # dynamic quantization (INT8) script 
│   └── INT8_static_quantization/
│       └── static_quantization_onnx.py     # static quantization (INT8) script
├── models/
│   ├── config_custom_data.yaml
│   ├── yolo_pytorch_models/             # .pt weights 
│   ├── yolo_onnx_models/
│   │   ├── FP32_onnx_models/            # models segregated according to their sizes (320,640,1280)
│   │   │   ├── 320/                        
│   │   │   ├── 640/
│   │   │   └── 1280/
│   │   ├── FP16_quantized_models/
│   │   │   ├── 320/
│   │   │   ├── 640/
│   │   │   └── 1280/
│   │   ├── INT8_dynamic_quantized_models/
│   │   │   ├── 320/
│   │   │   ├── 640/
│   │   │   └── 1280/
│   │   └── INT8_static_quantized_models/
│   │       ├── 320/
│   │       ├── 640/
│   │       └── 1280/
│   └── trt_engines/                # optional: to test optimized configuration model based on optimization report
├── imagedata/                      # example: images/ + labels/
├── optimization/
│   ├── csv_generator.py            # benchmark table → configurations.csv
│   ├── run_optimization.py         # generate utility scores, report and tradeoff images.
│   ├── configurations.csv
│   └── output_csv_results/         # full_results_with_utility.csv, optimization_report.md, PNG figures
├── TRT_edge_deployment/            # after optimization report: build .engine, test on desktop or Jetson
│   ├── build_trt_engine.py         # ONNX → .engine
│   ├── trt_runtime.py              # shared TRT session + benchmark
│   ├── trt_inference.py            # desktop/server GPU
│   └── trt_inference_jetson.py     # Jetson Orin Nano
├── outputfolder/                   # inference outputs (created/overwritten on run)
├── requirements.txt
└── README.md
```

### Flow of the Project (High-Level)

1. **Export / quantize models** (FP32, FP16, INT8 dynamic, INT8 static) at target input resolutions.
2. **Benchmark** the exported ONNX models across the hardware targets (CPU1, GPU1, CPU2, GPU2) to populate `optimization/configurations.csv`.
3. **Run optimization** over all benchmarked configurations to compute utility scores, apply constraints, and generate the report (`python .\optimization\run_optimization.py`).
4. **Choose the best configuration** (per your weights/constraints) using the optimization report.
5. **Optional (TensorRT / Jetson):** After you know the best **(model, precision, resolution)** from the report, build a `.engine` from the matching ONNX and run inference on either local GPU using `trt_inference.py` or on edge device like **Jetson Orin Nano** using `trt_inference_jetson.py` .

### Setup

1. Open a terminal in the **project root** (folder that contains `code_files`, `models`, `quantization_code_files`).
2. Install **PyTorch** for your machine from [pytorch.org](https://pytorch.org/get-started/locally/).
3. Install Python deps:

```bash
pip install -r requirements.txt
```

1. **GPU (optional):** Install CUDA/cuDNN that match **onnxruntime-gpu** (see `requirements.txt` comments), or use:

```bash
pip install "onnxruntime-gpu[cuda,cudnn]==1.23.2"
```

Inference calls `preload_dlls()` so pip-installed CUDA libraries are found even when `where cublasLt64_12.dll` fails.

### Run inference

1. Ensure your dataset folder (`--data-root`) contains `**images/`** and `**labels/`** (YOLO txt labels; class 0 = person for mAP).
2. Pick an ONNX model that matches your target input size.
3. Run inference:

```powershell
python .\code_files\main_execution.py --model "models\yolo_onnx_models\FP32_onnx_models\640\yolov8n_640.onnx" --device cpu --input-size 640 --data-root "imagedata"
```

1. For CUDA, use `**--device gpu**` (if your ORT + CUDA stack loads). `**--input-size**` must match how that ONNX was exported.

**Printed lines:**

1. **FPS** - includes load, preprocess, inference, drawing, saving images
2. **device=cpu** or **device=gpu** - ONNX Runtime’s **primary** execution provider (not every single op)
3. **Model size** - ONNX file size on disk
4. **mAP@IoU=0.5** - person class (id 0) vs labels; “undefined” if there are no person boxes in the labels

Outputs are written to `**outputfolder/`**. Thresholds `**CONF`** and `**IOU**` in `main_execution.py` control score cutoff and NMS.

### Optimization: select the best deployment configuration

1. Generate the benchmark CSV:

```powershell
python .\optimization\csv_generator.py
```

1. Run optimization over all benchmarked configurations:

```powershell
python .\optimization\run_optimization.py
```

Utility definition (defaults in the script):  U = alpha,A_{norm} + beta,F_{norm} - gamma,S_{norm}

Where A_{norm} uses **[mAP@0.5](mailto:mAP@0.5)**, F_{norm} uses **FPS**, and S_{norm} uses **model size (MB)**. Normalization is **min-max over the full benchmark CSV**.

Default weights: **α=0.50**, **β=0.30**, **γ=0.20**.

We also support feasibility constraints (defaults shown):

- **Accuracy:** `map50 >= 0.65`
- **Latency:** `latency_ms <= 100`
- **Size:** `size_mb <= 50`

You can override weights/constraints via CLI flags (e.g. `--alpha`, `--beta`, `--gamma`, `--amin`, `--lmax`, `--smax`).

For detailed outputs, tables, and plots, see:

- `optimization/output_csv_results/optimization_report.md`

### TensorRT deployment

**Do this after** you read the optimization report and pick the best **model / precision / resolution** for edge deployment. Build the TensorRT engine from the matching ONNX into `models/trt_engines/`, then benchmark with the same `images/` + `labels/` layout (FPS, engine size, mAP). Install `tensorrt` and `pycuda` per `requirements.txt` (Jetson: use JetPack / build the `.engine` on the Jetson so it matches the GPU).

```powershell
python .\TRT_edge_deployment\build_trt_engine.py --onnx "<path to chosen ONNX>" --out "models\trt_engines\<name>.engine"
python .\TRT_edge_deployment\trt_inference.py --engine "models\trt_engines\<name>.engine" --input-size 640 --data-root "imagedata"
```

(`--input-size` must match the resolution of the ONNX you built from, e.g. 320 / 640 / 1280.)

**Jetson Orin Nano:** on the device, `python3 TRT_edge_deployment/trt_inference_jetson.py` with the same `--engine`, `--input-size`, `--data-root` (outputs go to `outputfolder_trt_jetson/`).

### Notes

- **FP16 / INT8** inputs: preprocessing dtype must match the model’s ONNX input type; mismatch shows up as errors at `session.run()`.
- **Quantization scripts** each have a short “how to run” block at the top of the file.
- **PowerShell:** use one long `python ...` line, or line continuation with a **backtick** (```), not `^`.

### Conclusion

This repo gives you a **repeatable way to compare** YOLOv8 quantization and input sizes on your hardware, **rank configurations** with the optimization step, and **carry the chosen setup to the edge** optionally as a TensorRT engine on a local GPU or a Jetson Orin Nano. The right trade-off among accuracy, speed, and model size is the one that meets your constraints and what you read in the optimization report.