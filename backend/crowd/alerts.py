from __future__ import annotations

import time
from typing import Dict, List, Optional


class AlertManager:
    """Manages crowd control alerts with cooldown and history."""

    def __init__(self, cooldown_seconds: int = 30, max_history: int = 100):
        self.cooldown_seconds = cooldown_seconds
        self.max_history = max_history
        self._last_alert_time: dict[str, float] = {}  # area_name → timestamp
        self._previous_status: dict[str, str] = {}  # area_name → last status
        self.history: list[dict] = []

    def check(self, area_name: str, status: str, person_count: int, analysis: dict) -> Optional[dict]:
        """
        Check if an alert should fire for this area.

        Returns alert dict if triggered, None otherwise.
        """
        prev = self._previous_status.get(area_name, "NORMAL")
        self._previous_status[area_name] = status

        # Only alert on transition TO critical
        if status != "CRITICAL" or prev == "CRITICAL":
            return None

        # Check cooldown
        now = time.time()
        last = self._last_alert_time.get(area_name, 0)
        if now - last < self.cooldown_seconds:
            return None

        # Build reason
        reasons = []
        if analysis.get("smooth_count", 0) > analysis.get("max_count_threshold", float("inf")):
            reasons.append(f"Person count high ({person_count})")
        if analysis.get("max_density", 0) > 0.7:
            reasons.append(f"High density ({analysis['max_density']:.2f})")
        if analysis.get("velocity_std", 0) > 2.0:
            reasons.append(f"Abnormal movement (std={analysis['velocity_std']:.2f})")
        if not reasons:
            reasons.append(f"Crowd not in control ({person_count} persons)")

        alert = {
            "timestamp": now,
            "area": area_name,
            "reason": "; ".join(reasons),
            "person_count": person_count,
            "status": "CRITICAL",
        }

        self._last_alert_time[area_name] = now
        self.history.append(alert)

        # Trim history
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

        return alert

    def get_history(self, limit: int = 50) -> list[dict]:
        return self.history[-limit:]
