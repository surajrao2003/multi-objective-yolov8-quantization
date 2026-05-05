"""
Build a TensorRT engine from a YOLOv8 ONNX file (FP32 or FP16).
Uses the default CUDA device visible to this process (no GPU index override).

From project root:

    python TRT_edge_deployment/build_trt_engine.py --onnx ... --out ...
"""

from __future__ import annotations

import argparse
import ctypes
import sys
from pathlib import Path


def _serialized_engine_to_bytes(serialized: object) -> bytes:
    """
    TensorRT's build_serialized_network returns raw bytes on some versions and
    IHostMemory on others; normalize to bytes for Path.write_bytes / len.
    """
    if serialized is None:
        raise ValueError("serialized is None")
    if isinstance(serialized, (bytes, bytearray)):
        return bytes(serialized)
    try:
        return bytes(memoryview(serialized))
    except TypeError:
        pass
    nbytes = getattr(serialized, "nbytes", None)
    if nbytes is None:
        sz = getattr(serialized, "size", None)
        nbytes = int(sz() if callable(sz) else sz)
    ptr = getattr(serialized, "data", None)
    if ptr is None:
        raise TypeError(
            f"Cannot convert build_serialized_network result {type(serialized)!r} to bytes"
        )
    return ctypes.string_at(ctypes.c_void_p(int(ptr)), int(nbytes))


def main() -> int:
    p = argparse.ArgumentParser(description="ONNX → TensorRT .engine")
    p.add_argument("--onnx", required=True, help="Path to ONNX model.")
    p.add_argument("--out", required=True, help="Output .engine path.")
    p.add_argument("--fp16", action="store_true", help="Enable FP16 builder flag.")
    p.add_argument(
        "--workspace-mib",
        type=int,
        default=4096,
        help="TensorRT workspace pool limit (MiB).",
    )
    args = p.parse_args()

    onnx_path = Path(args.onnx).resolve()
    out_path = Path(args.out).resolve()
    if not onnx_path.is_file():
        print(f"ONNX not found: {onnx_path}", file=sys.stderr)
        return 1
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    with onnx_path.open("rb") as f:
        data = f.read()
    if not parser.parse(data):
        for i in range(parser.num_errors):
            print(parser.get_error(i), file=sys.stderr)
        return 1

    cfg = builder.create_builder_config()
    if args.fp16 and builder.platform_has_fast_fp16:
        cfg.set_flag(trt.BuilderFlag.FP16)
    mib = max(256, int(args.workspace_mib))
    try:
        cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, mib * (1 << 20))
    except AttributeError:
        try:
            cfg.max_workspace_size = mib * (1 << 20)
        except AttributeError:
            pass

    serialized = builder.build_serialized_network(network, cfg)
    if serialized is None:
        print("TensorRT build failed.", file=sys.stderr)
        return 1
    engine_bytes = _serialized_engine_to_bytes(serialized)
    out_path.write_bytes(engine_bytes)
    print(f"Wrote engine ({len(engine_bytes):,} bytes) → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
