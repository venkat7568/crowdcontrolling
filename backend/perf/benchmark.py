"""
CPU vs GPU benchmark utility for YOLOv8-nano inference.

Runs the same model on CPU and GPU (if available) and compares:
  - Inference latency (ms/frame)
  - Throughput (FPS)
  - Memory usage

Usage:
    python -m backend.perf.benchmark
    python -m backend.perf.benchmark --frames 200 --imgsz 480
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

# Ensure project root on path
_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def benchmark_backend(model_path: str, device: str, imgsz: int, warmup: int, frames: int) -> dict:
    """Benchmark a single backend (cpu/gpu)."""
    from ultralytics import YOLO

    model = YOLO(model_path)

    # Generate synthetic test frames (random noise at target resolution)
    test_frame = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

    # Warmup
    print(f"  Warming up ({warmup} frames)...")
    for _ in range(warmup):
        model.predict(test_frame, conf=0.4, classes=[0], imgsz=imgsz, verbose=False, device=device)

    # Benchmark
    print(f"  Benchmarking ({frames} frames)...")
    latencies = []
    for i in range(frames):
        start = time.perf_counter()
        results = model.predict(test_frame, conf=0.4, classes=[0], imgsz=imgsz, verbose=False, device=device)
        elapsed = (time.perf_counter() - start) * 1000
        latencies.append(elapsed)

        if (i + 1) % 50 == 0:
            print(f"    Frame {i+1}/{frames}: {elapsed:.1f}ms")

    latencies = np.array(latencies)

    return {
        "device": device,
        "frames": frames,
        "imgsz": imgsz,
        "latency_avg_ms": round(float(latencies.mean()), 2),
        "latency_min_ms": round(float(latencies.min()), 2),
        "latency_max_ms": round(float(latencies.max()), 2),
        "latency_p50_ms": round(float(np.percentile(latencies, 50)), 2),
        "latency_p95_ms": round(float(np.percentile(latencies, 95)), 2),
        "latency_p99_ms": round(float(np.percentile(latencies, 99)), 2),
        "throughput_fps": round(1000 / float(latencies.mean()), 1),
    }


def check_gpu_available() -> dict:
    """Check what GPU/acceleration is available."""
    info = {"cuda": False, "tensorrt": False, "device_name": "CPU only"}

    try:
        import torch
        if torch.cuda.is_available():
            info["cuda"] = True
            info["device_name"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = torch.version.cuda
            props = torch.cuda.get_device_properties(0)
            mem = getattr(props, 'total_memory', None) or getattr(props, 'total_mem', 0)
            info["gpu_memory_mb"] = round(mem / 1024 / 1024)
    except ImportError:
        pass

    try:
        import tensorrt
        info["tensorrt"] = True
        info["tensorrt_version"] = tensorrt.__version__
    except ImportError:
        pass

    return info


def format_comparison(cpu_result: dict, gpu_result: Optional[dict]) -> str:
    """Format a comparison table."""
    lines = []
    lines.append("\n" + "=" * 65)
    lines.append("  YOLOv8-nano INFERENCE BENCHMARK — CPU vs GPU")
    lines.append("=" * 65)

    def row(label, cpu_val, gpu_val=None):
        if gpu_val is not None:
            speedup = cpu_val / gpu_val if gpu_val > 0 else 0
            lines.append(f"  {label:<25} {cpu_val:>10}  {gpu_val:>10}  {speedup:>6.1f}x")
        else:
            lines.append(f"  {label:<25} {cpu_val:>10}")

    if gpu_result:
        lines.append(f"  {'Metric':<25} {'CPU':>10}  {'GPU':>10}  {'Speedup':>6}")
        lines.append("  " + "-" * 59)
        row("Avg latency (ms)", cpu_result["latency_avg_ms"], gpu_result["latency_avg_ms"])
        row("Min latency (ms)", cpu_result["latency_min_ms"], gpu_result["latency_min_ms"])
        row("P95 latency (ms)", cpu_result["latency_p95_ms"], gpu_result["latency_p95_ms"])
        row("P99 latency (ms)", cpu_result["latency_p99_ms"], gpu_result["latency_p99_ms"])
        row("Throughput (FPS)", cpu_result["throughput_fps"], gpu_result["throughput_fps"])
        lines.append("  " + "-" * 59)

        speedup = cpu_result["latency_avg_ms"] / gpu_result["latency_avg_ms"]
        lines.append(f"\n  GPU is {speedup:.1f}x faster than CPU")

        # Can we sustain 2 cameras at 15 FPS?
        budget_ms = 1000 / 15  # 66.7ms per frame
        total_gpu = gpu_result["latency_avg_ms"] * 2  # 2 cameras
        total_cpu = cpu_result["latency_avg_ms"] * 2
        lines.append(f"\n  2-Camera Budget (15 FPS = {budget_ms:.0f}ms per cycle):")
        lines.append(f"    CPU: {total_cpu:.0f}ms for 2 frames → {'OK' if total_cpu < budget_ms else 'TOO SLOW'}")
        lines.append(f"    GPU: {total_gpu:.0f}ms for 2 frames → {'OK' if total_gpu < budget_ms else 'TOO SLOW'}")
    else:
        lines.append(f"  {'Metric':<25} {'CPU':>10}")
        lines.append("  " + "-" * 37)
        row("Avg latency (ms)", cpu_result["latency_avg_ms"])
        row("Min latency (ms)", cpu_result["latency_min_ms"])
        row("P95 latency (ms)", cpu_result["latency_p95_ms"])
        row("Throughput (FPS)", cpu_result["throughput_fps"])
        lines.append("\n  No GPU available — CPU-only results")

    lines.append("=" * 65)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="CPU vs GPU inference benchmark")
    parser.add_argument("--model", default="yolov8n.pt", help="Model path")
    parser.add_argument("--imgsz", type=int, default=480, help="Input size")
    parser.add_argument("--frames", type=int, default=100, help="Frames to benchmark")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup frames")
    args = parser.parse_args()

    print("\n=== Crowd Control — Inference Benchmark ===\n")

    # Check hardware
    gpu_info = check_gpu_available()
    print(f"GPU: {gpu_info['device_name']}")
    if gpu_info["cuda"]:
        print(f"CUDA: {gpu_info.get('cuda_version', 'N/A')}")
        print(f"VRAM: {gpu_info.get('gpu_memory_mb', 'N/A')} MB")
    if gpu_info["tensorrt"]:
        print(f"TensorRT: {gpu_info.get('tensorrt_version', 'N/A')}")

    # CPU benchmark
    print(f"\n[1/2] CPU Benchmark (model={args.model}, imgsz={args.imgsz})")
    cpu_result = benchmark_backend(args.model, "cpu", args.imgsz, args.warmup, args.frames)

    # GPU benchmark (if available)
    gpu_result = None
    if gpu_info["cuda"]:
        print(f"\n[2/2] GPU Benchmark (model={args.model}, imgsz={args.imgsz})")
        gpu_result = benchmark_backend(args.model, "0", args.imgsz, args.warmup, args.frames)
    else:
        print("\n[2/2] Skipping GPU benchmark — no CUDA available")

    # Results
    report = format_comparison(cpu_result, gpu_result)
    print(report)

    # Save results
    import json
    results = {
        "gpu_info": gpu_info,
        "cpu": cpu_result,
        "gpu": gpu_result,
        "model": args.model,
    }
    out_path = Path("benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
