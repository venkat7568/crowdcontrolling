from __future__ import annotations

from collections import deque
from typing import List, Tuple

import numpy as np


class CrowdAnalyzer:
    """
    Analyzes crowd status per area using:
    - Person count
    - Density heatmap (Gaussian KDE on centroids)
    - Movement analysis (velocity magnitude + std)
    """

    NORMAL = "NORMAL"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"

    def __init__(
        self,
        max_count: int = 50,
        density_limit: float = 0.8,
        chaos_threshold: float = 3.0,
        smoothing_window: int = 30,
        hysteresis_frames: int = 5,
        grid_size: int = 20,
        frame_width: int = 640,
        frame_height: int = 480,
    ):
        self.max_count = max_count
        self.density_limit = density_limit
        self.chaos_threshold = chaos_threshold
        self.smoothing_window = smoothing_window
        self.hysteresis_frames = hysteresis_frames
        self.grid_size = grid_size
        self.frame_width = frame_width
        self.frame_height = frame_height

        self._raw_density_peak = 0.0

        # History buffers for temporal smoothing
        self._count_history = deque(maxlen=smoothing_window)
        self._density_history = deque(maxlen=smoothing_window)
        self._chaos_history = deque(maxlen=smoothing_window)

        # Hysteresis state
        self._current_status = self.NORMAL
        self._pending_status = self.NORMAL
        self._pending_count = 0

    def analyze(self, tracked_detections: list[dict]) -> dict:
        """
        Analyze a single frame's tracked detections.

        Args:
            tracked_detections: list of {"centroid": [cx,cy], "velocity": [vx,vy], "track_id": int, ...}

        Returns:
            {
                "person_count": int,
                "status": "NORMAL"|"WARNING"|"CRITICAL",
                "density_grid": list[list[float]],  # grid_size x grid_size
                "max_density": float,
                "avg_velocity": float,
                "velocity_std": float,
            }
        """
        count = len(tracked_detections)
        centroids = [d["centroid"] for d in tracked_detections]
        velocities = [d["velocity"] for d in tracked_detections]

        # Density heatmap
        density_grid = self._compute_density(centroids)
        max_density = self._raw_density_peak

        # Movement analysis
        avg_velocity, velocity_std = self._compute_movement(velocities)

        # Update histories
        self._count_history.append(count)
        self._density_history.append(max_density)
        self._chaos_history.append(velocity_std)

        # Smoothed values
        smooth_count = self._weighted_avg(self._count_history)
        smooth_density = self._weighted_avg(self._density_history)
        smooth_chaos = self._weighted_avg(self._chaos_history)

        # Determine raw status
        raw_status = self._evaluate_status(smooth_count, smooth_density, smooth_chaos)

        # Apply hysteresis
        status = self._apply_hysteresis(raw_status)

        return {
            "person_count": count,
            "status": status,
            "density_grid": density_grid.tolist(),
            "max_density": round(max_density, 3),
            "avg_velocity": round(avg_velocity, 3),
            "velocity_std": round(velocity_std, 3),
            "smooth_count": round(smooth_count, 1),
        }

    def _compute_density(self, centroids: list) -> np.ndarray:
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        if not centroids:
            self._raw_density_peak = 0.0
            return grid

        cell_w = self.frame_width / self.grid_size
        cell_h = self.frame_height / self.grid_size
        sigma = 1.5  # Gaussian spread in grid cells

        for cx, cy in centroids:
            gx = cx / cell_w
            gy = cy / cell_h

            # Apply Gaussian contribution to nearby cells
            for i in range(max(0, int(gy - 3)), min(self.grid_size, int(gy + 4))):
                for j in range(max(0, int(gx - 3)), min(self.grid_size, int(gx + 4))):
                    dist_sq = (i - gy + 0.5) ** 2 + (j - gx + 0.5) ** 2
                    grid[i, j] += np.exp(-dist_sq / (2 * sigma ** 2))

        # Compute density peak scaled by max_count (for status evaluation)
        # 1 person / max_count=50 → 0.02, 40 clustered / 50 → 0.80
        self._raw_density_peak = float(grid.max()) / max(1.0, float(self.max_count))

        # Self-normalize grid to [0, 1] for heatmap visualization
        if grid.max() > 0:
            grid /= grid.max()

        return grid

    def _compute_movement(self, velocities: list) -> tuple[float, float]:
        if not velocities:
            return 0.0, 0.0

        speeds = [np.sqrt(vx ** 2 + vy ** 2) for vx, vy in velocities]
        avg = float(np.mean(speeds))
        std = float(np.std(speeds))
        return avg, std

    def _weighted_avg(self, history: deque) -> float:
        if not history:
            return 0.0
        values = np.array(list(history))
        # Exponential weights: recent frames matter more
        weights = np.exp(np.linspace(-1, 0, len(values)))
        return float(np.average(values, weights=weights))

    def _evaluate_status(self, count: float, density: float, chaos: float) -> str:
        # CRITICAL conditions
        if (count > self.max_count
                or density > self.density_limit
                or chaos > self.chaos_threshold):
            return self.CRITICAL

        # WARNING conditions (75% thresholds)
        if (count > self.max_count * 0.75
                or density > self.density_limit * 0.75):
            return self.WARNING

        return self.NORMAL

    def _apply_hysteresis(self, raw_status: str) -> str:
        if raw_status == self._pending_status:
            self._pending_count += 1
        else:
            self._pending_status = raw_status
            self._pending_count = 1

        if self._pending_count >= self.hysteresis_frames:
            self._current_status = self._pending_status

        return self._current_status
