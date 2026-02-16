"""
Pipeline latency profiler — tracks per-stage timing for every frame.

Measures:
  - capture_ms:   Camera frame grab
  - detect_ms:    YOLOv8 inference (the bottleneck)
  - track_ms:     IOU tracker matching
  - analyze_ms:   Density + movement + status logic
  - encode_ms:    JPEG compression for WebSocket
  - broadcast_ms: WebSocket send to all clients
  - total_ms:     Full pipeline end-to-end

Reports rolling averages, min, max, and FPS.
"""

from __future__ import annotations

import time
from collections import deque


class StageTimer:
    """Context manager that measures a single pipeline stage."""

    def __init__(self):
        self.elapsed_ms = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000


class FrameProfile:
    """Timing data for one complete frame."""

    def __init__(self):
        self.capture_ms = 0.0
        self.detect_ms = 0.0
        self.track_ms = 0.0
        self.analyze_ms = 0.0
        self.encode_ms = 0.0
        self.broadcast_ms = 0.0
        self.total_ms = 0.0
        self.timestamp = 0.0

    def to_dict(self) -> dict:
        return {
            "capture_ms": round(self.capture_ms, 2),
            "detect_ms": round(self.detect_ms, 2),
            "track_ms": round(self.track_ms, 2),
            "analyze_ms": round(self.analyze_ms, 2),
            "encode_ms": round(self.encode_ms, 2),
            "broadcast_ms": round(self.broadcast_ms, 2),
            "total_ms": round(self.total_ms, 2),
        }


class PipelineProfiler:
    """
    Collects frame profiles and computes rolling statistics.

    Usage in processing loop:
        prof = pipeline_profiler.new_frame()

        with StageTimer() as t:
            frame = camera.get_frame()
        prof.capture_ms = t.elapsed_ms

        with StageTimer() as t:
            detections = detector.detect(frame)
        prof.detect_ms = t.elapsed_ms

        ... etc ...

        pipeline_profiler.finish_frame(prof)
        stats = pipeline_profiler.get_stats()
    """

    def __init__(self, window_size: int = 150):
        self.window_size = window_size
        self._history: deque[FrameProfile] = deque(maxlen=window_size)
        self._frame_start = 0.0

    def new_frame(self) -> FrameProfile:
        self._frame_start = time.perf_counter()
        return FrameProfile()

    def finish_frame(self, profile: FrameProfile):
        profile.total_ms = (time.perf_counter() - self._frame_start) * 1000
        profile.timestamp = time.time()
        self._history.append(profile)

    def get_stats(self) -> dict:
        """Get rolling statistics over the last N frames."""
        if not self._history:
            return self._empty_stats()

        profiles = list(self._history)
        n = len(profiles)

        def stat(attr):
            values = [getattr(p, attr) for p in profiles]
            avg = sum(values) / n
            return {
                "avg": round(avg, 2),
                "min": round(min(values), 2),
                "max": round(max(values), 2),
            }

        # Calculate actual FPS from timestamps
        if n >= 2:
            duration = profiles[-1].timestamp - profiles[0].timestamp
            fps = (n - 1) / duration if duration > 0 else 0
        else:
            fps = 0

        total_stats = stat("total_ms")
        detect_stats = stat("detect_ms")

        return {
            "fps": round(fps, 1),
            "frames_analyzed": n,
            "pipeline": {
                "capture": stat("capture_ms"),
                "detection": detect_stats,
                "tracking": stat("track_ms"),
                "analysis": stat("analyze_ms"),
                "encoding": stat("encode_ms"),
                "broadcast": stat("broadcast_ms"),
                "total": total_stats,
            },
            "bottleneck": self._find_bottleneck(profiles),
            "efficiency": self._calc_efficiency(profiles),
        }

    def get_latest(self) -> dict:
        """Get the most recent frame's profile."""
        if not self._history:
            return {}
        return self._history[-1].to_dict()

    def _find_bottleneck(self, profiles) -> dict:
        """Identify which stage takes the most time."""
        stages = ["capture_ms", "detect_ms", "track_ms", "analyze_ms", "encode_ms", "broadcast_ms"]
        avgs = {}
        for stage in stages:
            avgs[stage] = sum(getattr(p, stage) for p in profiles) / len(profiles)

        worst = max(avgs, key=avgs.get)
        total_avg = sum(getattr(p, "total_ms") for p in profiles) / len(profiles)
        pct = (avgs[worst] / total_avg * 100) if total_avg > 0 else 0

        return {
            "stage": worst.replace("_ms", ""),
            "avg_ms": round(avgs[worst], 2),
            "percent_of_total": round(pct, 1),
        }

    def _calc_efficiency(self, profiles) -> dict:
        """Calculate pipeline efficiency metrics."""
        avgs = {}
        for attr in ["detect_ms", "total_ms"]:
            avgs[attr] = sum(getattr(p, attr) for p in profiles) / len(profiles)

        # How much of the total time is actual inference vs overhead
        inference_pct = (avgs["detect_ms"] / avgs["total_ms"] * 100) if avgs["total_ms"] > 0 else 0
        overhead_pct = 100 - inference_pct

        return {
            "inference_percent": round(inference_pct, 1),
            "overhead_percent": round(overhead_pct, 1),
            "target_fps": 15,
            "can_sustain_target": avgs["total_ms"] < (1000 / 15),
        }

    def _empty_stats(self) -> dict:
        empty = {"avg": 0, "min": 0, "max": 0}
        return {
            "fps": 0,
            "frames_analyzed": 0,
            "pipeline": {
                "capture": empty, "detection": empty, "tracking": empty,
                "analysis": empty, "encoding": empty, "broadcast": empty, "total": empty,
            },
            "bottleneck": {"stage": "none", "avg_ms": 0, "percent_of_total": 0},
            "efficiency": {"inference_percent": 0, "overhead_percent": 0, "target_fps": 15, "can_sustain_target": False},
        }
