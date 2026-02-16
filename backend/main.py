from __future__ import annotations

import asyncio
import base64
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Ensure project root is on path so imports work when run from any directory
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.config import load_config, save_config, AppConfig, AreaConfig, MODELS_DIR
from backend.camera.manager import CameraManager
from backend.inference.detector import PersonDetector
from backend.crowd.tracker import IOUTracker
from backend.crowd.analyzer import CrowdAnalyzer
from backend.crowd.alerts import AlertManager
from backend.perf.profiler import PipelineProfiler, StageTimer

app = FastAPI(title="Crowd Control Monitor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global state ---
config: AppConfig = None
camera_manager: CameraManager = None
detector: PersonDetector = None
trackers: dict[int, IOUTracker] = {}       # camera_index → tracker
analyzers: dict[int, CrowdAnalyzer] = {}   # camera_index → analyzer
alert_manager: AlertManager = None
pipeline_profiler: PipelineProfiler = None
connected_clients: list[WebSocket] = []
processing_active = False


@app.on_event("startup")
async def startup():
    global config, camera_manager, detector, alert_manager, pipeline_profiler, processing_active

    config = load_config()

    # Initialize camera manager
    camera_manager = CameraManager()
    print("[STARTUP] Discovering cameras...")
    available = camera_manager.discover_cameras()
    print(f"[STARTUP] Found cameras at indices: {available}")

    # Start cameras configured in areas
    needed = [area.camera_index for area in config.areas]
    results = camera_manager.start_cameras(needed)
    for idx, ok in results.items():
        status = "OK" if ok else "FAILED"
        print(f"[STARTUP] Camera {idx}: {status}")

    # Initialize detector
    model_name = config.detection.model_path.replace("models/", "")
    model_path = MODELS_DIR / model_name
    if not model_path.exists():
        # Fallback: let ultralytics auto-download yolov8n.pt
        print(f"[STARTUP] Model not found at {model_path}, will auto-download yolov8n.pt")
        model_path = "yolov8n.pt"
    else:
        model_path = str(model_path)
    print(f"[STARTUP] Loading model: {model_path}")
    detector = PersonDetector(
        model_path=str(model_path),
        confidence=config.detection.confidence_threshold,
        input_size=config.detection.input_size,
    )
    print("[STARTUP] Model loaded")

    # Initialize per-area trackers and analyzers
    for area in config.areas:
        trackers[area.camera_index] = IOUTracker()
        analyzers[area.camera_index] = CrowdAnalyzer(
            max_count=area.max_count,
            density_limit=area.density_limit,
            chaos_threshold=area.chaos_threshold,
            smoothing_window=config.smoothing_window,
            hysteresis_frames=config.hysteresis_frames,
        )

    alert_manager = AlertManager(cooldown_seconds=config.alert_cooldown_seconds)
    pipeline_profiler = PipelineProfiler(window_size=150)

    # Start processing loop
    processing_active = True
    asyncio.create_task(processing_loop())
    print("[STARTUP] Crowd control system ready")


@app.on_event("shutdown")
async def shutdown():
    global processing_active
    processing_active = False
    if camera_manager:
        camera_manager.stop_all()


def _process_all_areas_sync(areas_cfg, cam_mgr, det, trks, anlzs, alrt_mgr):
    """
    Run ALL heavy processing in a single thread — keeps the event loop 100% free.

    This avoids the GIL issue: YOLO inference, numpy tracking, and JPEG encoding
    all hold the GIL. By running everything in one thread call, the event loop
    can serve REST/WebSocket requests while this runs.
    """
    areas_data = []
    alerts = []
    timings = {"capture": 0, "detect": 0, "track": 0, "analyze": 0, "encode": 0}

    for area in areas_cfg:
        idx = area.camera_index

        try:
            # --- CAPTURE ---
            with StageTimer() as t:
                frame = cam_mgr.get_frame(idx)
            timings["capture"] += t.elapsed_ms
        except Exception as e:
            print(f"[ERROR] Camera {idx} capture failed: {e}")
            frame = None

        if frame is None:
            areas_data.append({
                "name": area.name, "camera_index": idx, "frame": None,
                "person_count": 0, "status": "OFFLINE", "detections": [],
                "density_grid": [], "avg_velocity": 0, "velocity_std": 0,
            })
            continue

        try:
            # --- DETECTION ---
            with StageTimer() as t:
                detections = det.detect(frame)
            timings["detect"] += t.elapsed_ms

            # --- TRACKING ---
            with StageTimer() as t:
                tracker = trks.get(idx)
                tracked = tracker.update(detections) if tracker else detections
            timings["track"] += t.elapsed_ms

            # --- ANALYSIS ---
            with StageTimer() as t:
                analyzer = anlzs.get(idx)
                analysis = analyzer.analyze(tracked) if analyzer else {
                    "person_count": 0, "status": "NORMAL", "density_grid": [],
                    "max_density": 0, "avg_velocity": 0, "velocity_std": 0,
                }
            timings["analyze"] += t.elapsed_ms

            # Check alerts
            alert = alrt_mgr.check(area.name, analysis["status"], analysis["person_count"], analysis)
            if alert:
                alerts.append(alert)

            # --- ENCODE (JPEG Q60 — 96.5% smaller than raw, fast enough for 15 FPS) ---
            with StageTimer() as t:
                _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
                frame_b64 = base64.b64encode(jpeg.tobytes()).decode("utf-8")
            timings["encode"] += t.elapsed_ms

            areas_data.append({
                "name": area.name, "camera_index": idx, "frame": frame_b64,
                "person_count": analysis["person_count"], "status": analysis["status"],
                "detections": [
                    {"bbox": d["bbox"], "confidence": d.get("confidence", 0), "track_id": d.get("track_id", -1)}
                    for d in tracked
                ],
                "density_grid": analysis.get("density_grid", []),
                "avg_velocity": analysis.get("avg_velocity", 0),
                "velocity_std": analysis.get("velocity_std", 0),
            })

        except Exception as e:
            print(f"[ERROR] Processing failed for area '{area.name}' (cam {idx}): {e}")
            areas_data.append({
                "name": area.name, "camera_index": idx, "frame": None,
                "person_count": 0, "status": "ERROR", "detections": [],
                "density_grid": [], "avg_velocity": 0, "velocity_std": 0,
            })

    return areas_data, alerts, timings


async def processing_loop():
    """Main loop: grab frames, detect, analyze, broadcast — with per-stage profiling."""
    target_interval = 1.0 / 15  # ~15 FPS

    while processing_active:
        prof = pipeline_profiler.new_frame()

        # Run ALL heavy work in a single thread — event loop stays free for REST/WS
        areas_data, alerts, timings = await asyncio.to_thread(
            _process_all_areas_sync,
            config.areas, camera_manager, detector, trackers, analyzers, alert_manager,
        )

        # Record stage times
        prof.capture_ms = timings["capture"]
        prof.detect_ms = timings["detect"]
        prof.track_ms = timings["track"]
        prof.analyze_ms = timings["analyze"]
        prof.encode_ms = timings["encode"]

        # --- BROADCAST (async, on event loop) ---
        with StageTimer() as t_broadcast:
            perf_latest = pipeline_profiler.get_latest()

            message = json.dumps({
                "type": "frame_update",
                "timestamp": time.time(),
                "areas": areas_data,
                "perf": perf_latest,
            })

            disconnected = []
            for ws in connected_clients:
                try:
                    await ws.send_text(message)
                except Exception:
                    disconnected.append(ws)

            for ws in disconnected:
                connected_clients.remove(ws)

            for alert in alerts:
                alert_msg = json.dumps({"type": "alert", **alert})
                for ws in connected_clients:
                    try:
                        await ws.send_text(alert_msg)
                    except Exception:
                        pass

        prof.broadcast_ms = t_broadcast.elapsed_ms

        # Finish profiling
        pipeline_profiler.finish_frame(prof)

        # Maintain target frame rate (minimum 5ms sleep for event loop breathing room)
        sleep_time = max(0.005, target_interval - (prof.total_ms / 1000))
        await asyncio.sleep(sleep_time)


# --- WebSocket ---

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    connected_clients.append(ws)
    print(f"[WS] Client connected ({len(connected_clients)} total)")
    try:
        while True:
            # Keep connection alive, handle incoming messages if needed
            data = await ws.receive_text()
            # Could handle config updates from dashboard here
    except WebSocketDisconnect:
        if ws in connected_clients:
            connected_clients.remove(ws)
        print(f"[WS] Client disconnected ({len(connected_clients)} total)")


# --- REST API ---

@app.get("/api/health")
async def health():
    active_cams = camera_manager.get_active_indices() if camera_manager else []
    return {
        "status": "running",
        "cameras": active_cams,
        "areas": [a.name for a in config.areas],
    }


@app.get("/api/config")
async def get_config():
    return config.model_dump()


@app.post("/api/config")
async def update_config(new_config: dict):
    global config
    try:
        # Merge with existing
        current = config.model_dump()
        current.update(new_config)
        config = AppConfig(**current)
        save_config(config)

        # Update analyzers with new thresholds
        for area in config.areas:
            idx = area.camera_index
            if idx in analyzers:
                analyzers[idx].max_count = area.max_count
                analyzers[idx].density_limit = area.density_limit
                analyzers[idx].chaos_threshold = area.chaos_threshold

        return {"status": "ok", "config": config.model_dump()}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@app.get("/api/alerts")
async def get_alerts():
    return alert_manager.get_history() if alert_manager else []


@app.get("/api/cameras")
async def get_cameras():
    if not camera_manager:
        return {"available": [], "active": []}
    available = camera_manager.discover_cameras()
    active = camera_manager.get_active_indices()
    return {"available": available, "active": active}


@app.get("/api/perf")
async def get_perf():
    """Full performance statistics — rolling averages, bottleneck analysis, efficiency."""
    if not pipeline_profiler:
        return {"error": "Profiler not initialized"}
    return pipeline_profiler.get_stats()


@app.get("/api/perf/latest")
async def get_perf_latest():
    """Latest single frame timing breakdown."""
    if not pipeline_profiler:
        return {"error": "Profiler not initialized"}
    return pipeline_profiler.get_latest()


# Serve React dashboard build
DASHBOARD_DIR = Path(__file__).parent.parent / "dashboard" / "dist"
if DASHBOARD_DIR.exists():
    app.mount("/", StaticFiles(directory=str(DASHBOARD_DIR), html=True), name="dashboard")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
