from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List

from pydantic import BaseModel

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
CONFIG_PATH = DATA_DIR / "config.json"


class AreaConfig(BaseModel):
    name: str
    camera_index: int
    max_count: int = 50
    density_limit: float = 0.8
    chaos_threshold: float = 3.0


class DetectionConfig(BaseModel):
    confidence_threshold: float = 0.4
    model_path: str = "models/yolov8n.pt"
    input_size: int = 480


class AppConfig(BaseModel):
    areas: list[AreaConfig] = []
    detection: DetectionConfig = DetectionConfig()
    smoothing_window: int = 30
    hysteresis_frames: int = 5
    alert_cooldown_seconds: int = 30


def load_config() -> AppConfig:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r") as f:
            data = json.load(f)
        return AppConfig(**data)
    cfg = AppConfig(
        areas=[
            AreaConfig(name="Area 1", camera_index=0, max_count=50),
            AreaConfig(name="Area 2", camera_index=1, max_count=100),
        ]
    )
    save_config(cfg)
    return cfg


def save_config(cfg: AppConfig):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg.model_dump(), f, indent=2)
