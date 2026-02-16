from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
from ultralytics import YOLO


def _detect_device() -> str:
    """Auto-detect best available device: CUDA GPU > CPU."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            print(f"[DETECTOR] Using GPU: {name}")
            return "0"
    except ImportError:
        pass
    print("[DETECTOR] Using CPU (no CUDA available)")
    return "cpu"


def _find_best_model(model_path: str) -> str:
    """Find the best available model format: TensorRT > ONNX > PyTorch.

    On Jetson Nano, TensorRT (.engine) is ~7x faster than PyTorch.
    If the requested model doesn't exist, check for faster alternatives
    in the same directory.
    """
    requested = Path(model_path)
    model_dir = requested.parent
    stem = requested.stem.replace("_crowd", "").replace("_fp16", "")

    # Priority order: TensorRT engine > ONNX > PyTorch
    candidates = [
        model_dir / f"{stem}.engine",
        model_dir / f"{stem}_fp16.engine",
        model_dir / f"{stem}_crowd.engine",
        model_dir / f"{stem}.onnx",
        requested,
    ]

    for candidate in candidates:
        if candidate.exists():
            if candidate != requested:
                print(f"[DETECTOR] Found faster model: {candidate.name} (over {requested.name})")
            return str(candidate)

    return model_path


class PersonDetector:
    """YOLOv8-nano person detector with automatic backend selection."""

    PERSON_CLASS_ID = 0  # COCO class 0 = person

    def __init__(self, model_path: str = "yolov8n.pt", confidence: float = 0.4, input_size: int = 480, device: str = None):
        self.confidence = confidence
        self.input_size = input_size
        self.device = device or _detect_device()

        # Auto-detect best model format: .engine > .onnx > .pt
        model_path = _find_best_model(model_path)

        resolved = Path(model_path)
        if not resolved.exists():
            print(f"[DETECTOR] Model not found at {model_path}, will auto-download yolov8n.pt")
            model_path = "yolov8n.pt"

        print(f"[DETECTOR] Loading: {Path(model_path).name}")
        try:
            self.model = YOLO(model_path)
            self._warm_up()
        except Exception as e:
            print(f"[DETECTOR] FATAL: Failed to load model '{model_path}': {e}")
            raise RuntimeError(f"Model load failed: {e}") from e

    def _warm_up(self):
        """Run a dummy inference to warm up the model."""
        dummy = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        self.model.predict(dummy, conf=self.confidence, classes=[self.PERSON_CLASS_ID],
                           imgsz=self.input_size, device=self.device, verbose=False)

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Detect persons in a frame.

        Returns list of:
            {
                "bbox": [x1, y1, x2, y2],
                "confidence": float,
                "centroid": [cx, cy]
            }
        """
        results = self.model.predict(
            frame,
            conf=self.confidence,
            classes=[self.PERSON_CLASS_ID],
            imgsz=self.input_size,
            device=self.device,
            verbose=False,
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                x1, y1, x2, y2 = xyxy.tolist()
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                detections.append({
                    "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    "confidence": round(conf, 3),
                    "centroid": [round(cx, 1), round(cy, 1)],
                })
        return detections
