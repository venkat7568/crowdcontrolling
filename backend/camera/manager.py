from __future__ import annotations

import math
import platform
import random
import threading
import time
from typing import Dict, List, Optional, Union

import cv2
import numpy as np


def _get_camera_backends() -> list:
    """Return camera backends to try, ordered by platform preference."""
    system = platform.system()
    if system == "Windows":
        return [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    elif system == "Linux":
        # Jetson Nano and other Linux: V4L2 is the native backend
        return [cv2.CAP_V4L2, cv2.CAP_ANY]
    else:
        return [cv2.CAP_ANY]


class CameraStream:
    """Threaded camera capture — always holds the latest frame."""

    def __init__(self, camera_index: int, width: int = 640, height: int = 480, fps: int = 15):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps

        self._cap = None
        self._frame = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self.is_open = False

    def start(self) -> bool:
        # Try each backend in platform-preferred order
        for backend in _get_camera_backends():
            self._cap = cv2.VideoCapture(self.camera_index, backend)
            if self._cap.isOpened():
                break
            self._cap.release()

        if not self._cap.isOpened():
            self.is_open = False
            return False

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)

        self.is_open = True
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        return True

    def _capture_loop(self):
        while self._running:
            ret, frame = self._cap.read()
            if ret:
                with self._lock:
                    self._frame = frame
            else:
                time.sleep(0.01)

    def read(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        if self._cap:
            self._cap.release()
        self.is_open = False
        self._frame = None


class DemoStream:
    """
    Generates synthetic frames with simulated walking people.
    Used when no real cameras are available (development/demo mode).

    Draws person-shaped silhouettes that wander around the frame,
    so the YOLO detector has something realistic to detect.
    """

    def __init__(self, camera_index: int, width: int = 640, height: int = 480, fps: int = 15, num_people: int = 8):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.is_open = False

        self._frame = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

        # Simulated people: each has position, velocity, size
        self._people = []
        for _ in range(num_people):
            self._people.append({
                "x": random.randint(50, width - 50),
                "y": random.randint(100, height - 50),
                "vx": random.uniform(-2, 2),
                "vy": random.uniform(-1, 1),
                "h": random.randint(80, 140),  # person height in pixels
                "color": (random.randint(40, 180), random.randint(40, 180), random.randint(40, 180)),
                "phase": random.uniform(0, 2 * math.pi),  # walking phase
            })

    def start(self) -> bool:
        self.is_open = True
        self._running = True
        self._thread = threading.Thread(target=self._render_loop, daemon=True)
        self._thread.start()
        return True

    def _render_loop(self):
        interval = 1.0 / self.fps
        t = 0

        while self._running:
            frame_start = time.perf_counter()

            # Dark background simulating a room / corridor
            frame = np.full((self.height, self.width, 3), (35, 30, 25), dtype=np.uint8)

            # Draw a floor line
            cv2.line(frame, (0, self.height - 40), (self.width, self.height - 40), (60, 55, 50), 2)

            # Draw "DEMO MODE" watermark
            cv2.putText(frame, "DEMO MODE", (self.width // 2 - 80, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 200), 2)
            cv2.putText(frame, f"CAM {self.camera_index}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 150, 0), 2)

            # Update and draw each person
            for p in self._people:
                # Move
                p["x"] += p["vx"]
                p["y"] += p["vy"]

                # Bounce off walls
                if p["x"] < 30 or p["x"] > self.width - 30:
                    p["vx"] *= -1
                if p["y"] < 80 or p["y"] > self.height - 50:
                    p["vy"] *= -1

                # Slight random direction changes
                if random.random() < 0.02:
                    p["vx"] += random.uniform(-1, 1)
                    p["vy"] += random.uniform(-0.5, 0.5)
                    p["vx"] = max(-3, min(3, p["vx"]))
                    p["vy"] = max(-1.5, min(1.5, p["vy"]))

                # Draw person silhouette (head + body + legs)
                cx, cy = int(p["x"]), int(p["y"])
                h = p["h"]
                w = int(h * 0.35)
                color = p["color"]

                # Body (rectangle)
                body_top = cy - h // 2
                body_bot = cy + h // 3
                cv2.rectangle(frame, (cx - w // 2, body_top), (cx + w // 2, body_bot), color, -1)

                # Head (circle)
                head_r = int(h * 0.12)
                cv2.circle(frame, (cx, body_top - head_r), head_r, color, -1)

                # Legs (two lines with walking motion)
                phase = p["phase"] + t * 3
                leg_spread = int(math.sin(phase) * w * 0.4)
                cv2.line(frame, (cx, body_bot), (cx - leg_spread, cy + h // 2), color, 3)
                cv2.line(frame, (cx, body_bot), (cx + leg_spread, cy + h // 2), color, 3)

                p["phase"] += 0.1

            t += interval

            with self._lock:
                self._frame = frame

            elapsed = time.perf_counter() - frame_start
            sleep_time = max(0, interval - elapsed)
            time.sleep(sleep_time)

    def read(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        self.is_open = False
        self._frame = None


class CameraManager:
    """Discovers and manages multiple camera streams. Falls back to demo mode."""

    def __init__(self, demo_mode: bool = False):
        self.streams: Dict[int, Union[CameraStream, DemoStream]] = {}
        self.demo_mode = demo_mode

    def discover_cameras(self, max_index: int = 8) -> list[int]:
        """Scan camera indices 0..max_index and return available ones."""
        available = []
        backends = _get_camera_backends()
        for i in range(max_index):
            found = False
            for backend in backends:
                cap = cv2.VideoCapture(i, backend)
                if cap.isOpened():
                    available.append(i)
                    cap.release()
                    found = True
                    break
                cap.release()
            if not found:
                pass  # Camera not available at this index
        return available

    def start_cameras(self, indices: list[int]) -> dict[int, bool]:
        """Start capture threads for the given camera indices. Falls back to demo if no real camera."""
        results = {}
        for idx in indices:
            if idx in self.streams and self.streams[idx].is_open:
                results[idx] = True
                continue

            if not self.demo_mode:
                # Try real camera first
                stream = CameraStream(camera_index=idx)
                ok = stream.start()
                if ok:
                    self.streams[idx] = stream
                    results[idx] = True
                    continue

            # Fallback to demo stream
            print(f"[CAMERA] No real camera at index {idx}, using DEMO mode")
            demo = DemoStream(camera_index=idx, num_people=random.randint(5, 15))
            demo.start()
            self.streams[idx] = demo
            self.demo_mode = True
            results[idx] = True

        return results

    def get_frame(self, camera_index: int) -> Optional[np.ndarray]:
        stream = self.streams.get(camera_index)
        if stream and stream.is_open:
            return stream.read()
        return None

    def stop_all(self):
        for stream in self.streams.values():
            stream.stop()
        self.streams.clear()

    def get_active_indices(self) -> list[int]:
        return [idx for idx, s in self.streams.items() if s.is_open]
