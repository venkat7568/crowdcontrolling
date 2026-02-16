# Crowd Control Monitor

Real-time crowd density monitoring system using AI-powered person detection, designed for deployment on NVIDIA Jetson Nano with 2 cameras.

**Built by Venkat Dhanikonda**

---

## What This System Does

Monitors live camera feeds to detect people, track their movement, and determine if a crowd is under control or becoming dangerous.

- **Live camera feeds** with bounding boxes around each detected person
- **Person tracking** with persistent IDs across frames (IOU tracker)
- **Crowd density heatmaps** (Gaussian KDE on 20x20 grid)
- **Status alerts**: NORMAL (green) / WARNING (yellow) / CRITICAL (red) per area
- **Movement chaos detection**: flags panicked/chaotic crowd movement
- **Real-time dashboard** accessible from any browser on the local network
- **Customizable thresholds** via Settings UI — no restart needed

---

## What We Built and Why

### The Problem
Crowd monitoring in public spaces (entrances, auditoriums, events) traditionally requires human operators watching CCTV feeds. This is error-prone, expensive, and doesn't scale. We need an automated system that:
- Runs on cheap edge hardware (no cloud, no internet)
- Detects crowd density in real-time
- Alerts before a crowd becomes dangerous
- Works with standard USB cameras

### Our Solution
An end-to-end pipeline from camera capture to browser dashboard, running entirely on a $100 NVIDIA Jetson Nano:

```
Camera → YOLOv8-nano → IOU Tracker → Crowd Analyzer → WebSocket → Browser Dashboard
  USB      Detection     Tracking    Density+Status    15 FPS       React UI
 3.8ms      35ms          1.4ms        2.5ms           1.8ms      (Jetson Nano TensorRT FP16)
```

### What Was Built

| What | How | Why This Approach |
|------|-----|-------------------|
| **Person detection** | YOLOv8-nano (6.5MB model) | Smallest YOLO variant — only one that fits Jetson Nano's 4GB RAM at real-time speed |
| **Person tracking** | Custom IOU tracker | DeepSORT adds ~15ms per frame (separate neural network for re-ID); IOU runs in <1ms using only bounding box overlap |
| **Density analysis** | Gaussian KDE on 20x20 grid | Per-pixel density would take ~50ms on 640x480; 20x20 grid takes <1ms while still showing where people cluster |
| **Status logic** | Hysteresis (5 frames) + temporal smoothing (30 frames) | Raw detection is noisy; without smoothing, status would flicker between NORMAL/CRITICAL every few frames |
| **Video streaming** | WebSocket with base64 JPEG (Q60) | ~20KB per frame, 15 FPS = ~600KB/s for 2 cameras — works on any LAN, no RTSP/HLS complexity |
| **Dashboard** | React 18 + Tailwind CSS | Canvas-based rendering for bounding boxes + heatmap overlay; auto-reconnect WebSocket |
| **Backend** | Python FastAPI (async) | Native WebSocket support; heavy work runs in thread pool to keep event loop free |
| **Camera capture** | Threaded per-camera | OpenCV VideoCapture blocks ~30-60ms waiting for frames; threading eliminates this bottleneck |
| **Model format** | Auto-detect TensorRT > ONNX > PyTorch | TensorRT FP16 is ~7x faster than PyTorch on Jetson Nano; system auto-selects best available |
| **Cross-platform cameras** | Platform-aware backend selection | Windows uses DSHOW/MSMF, Linux/Jetson uses V4L2 — auto-detected at startup |

---

## Architecture

```
                    +-------------------+
  Camera 0 ------->|                   |-------> WebSocket -------> Browser Dashboard
  Camera 1 ------->|   FastAPI Server  |         (15 FPS)          (React + Tailwind)
                    |                   |
                    |  YOLOv8-nano      |-------> REST API
                    |  IOU Tracker      |         /api/health
                    |  Crowd Analyzer   |         /api/config
                    |  Alert Manager    |         /api/alerts
                    +-------------------+         /api/perf
```

### Backend (Python)
| Module | File | Purpose |
|--------|------|---------|
| Entry Point | `main.py` | Starts uvicorn server on port 8000 |
| FastAPI App | `backend/main.py` | WebSocket broadcasting, REST API, 15 FPS processing loop |
| Camera Manager | `backend/camera/manager.py` | Threaded camera capture, platform-aware discovery (DSHOW/V4L2), demo fallback |
| Person Detector | `backend/inference/detector.py` | YOLOv8-nano inference, auto-detect GPU/CPU, auto-select TensorRT > ONNX > PyTorch |
| IOU Tracker | `backend/crowd/tracker.py` | Lightweight multi-object tracking with velocity computation |
| Crowd Analyzer | `backend/crowd/analyzer.py` | Gaussian KDE density heatmap, movement analysis, hysteresis status logic |
| Alert Manager | `backend/crowd/alerts.py` | Cooldown-based alert system (30s between alerts) |
| Performance | `backend/perf/profiler.py` | Per-stage timing (capture/detect/track/analyze/encode/broadcast) |
| Config | `backend/config.py` | Pydantic config, JSON persistence at `backend/data/config.json` |

### Dashboard (React)
| Component | File | Purpose |
|-----------|------|---------|
| App | `dashboard/src/App.jsx` | Main layout, header, settings toggle |
| CameraFeed | `dashboard/src/components/CameraFeed.jsx` | Canvas-rendered live video + bounding box overlays + density heatmap |
| StatusPanel | `dashboard/src/components/StatusPanel.jsx` | Per-area status cards with person count, velocity, chaos index |
| AlertLog | `dashboard/src/components/AlertLog.jsx` | Scrollable alert history with timestamps |
| PerfOverlay | `dashboard/src/components/PerfOverlay.jsx` | Pipeline performance breakdown (6 stages) |
| Settings | `dashboard/src/components/Settings.jsx` | Edit thresholds per area — saves instantly via POST /api/config |
| CrowdContext | `dashboard/src/contexts/CrowdContext.jsx` | WebSocket auto-reconnect, state management, audio alerts |

---

## Model: YOLOv8-nano

### Why YOLOv8-nano?

| Criteria | YOLOv8-nano | Why It Fits |
|----------|-------------|-------------|
| Model size | ~12 MB (.pt) / ~6 MB (TensorRT FP16) | Fits in Jetson Nano's 4GB shared RAM alongside CUDA context |
| Parameters | 3.2M | Smallest YOLO variant — larger models won't fit at real-time speed |
| Inference speed | ~35ms on Jetson Nano (TensorRT FP16) | 1 camera at 15 FPS = 67ms budget → 35ms fits with 33% headroom |
| Accuracy | mAP50 ~52.5% (person class) | Good enough for person counting (we only need COCO class 0 = person) |
| Input size | 480x480 pixels | Sweet spot: 320px too inaccurate, 640px too slow on Jetson (44% more pixels) |
| Dataset | COCO 2017 (pre-trained) | 118K training images with person annotations; fine-tuning via `training/train.py` |

### Why NOT These Alternatives?

| Alternative | Problem |
|-------------|---------|
| YOLOv8-small | ~90ms on Jetson Nano — breaks 15 FPS target for even 1 camera |
| YOLOv8-medium | ~180ms on Jetson — only ~5 FPS, unusable for real-time |
| YOLOv8-large/x | Won't fit in 4GB shared RAM alongside CUDA context |
| DeepSORT tracker | Requires separate ReID CNN — adds ~15-25ms per frame on Jetson |
| SSD MobileNet | Lower accuracy than YOLOv8n at same speed on edge devices |
| Cloud API (AWS/GCP) | Latency (+50-200ms), cost ($40-130/day at 15 FPS), privacy, no offline |

### Model Format Auto-Detection

The detector automatically finds and uses the fastest available model:

```
Priority: .engine (TensorRT) > .onnx (ONNX Runtime) > .pt (PyTorch)
```

Place any of these in `backend/models/` and the system picks the best one:
- `yolov8n.engine` — TensorRT FP16 (Jetson Nano production, ~35ms)
- `yolov8n.onnx` — ONNX Runtime (cross-platform, ~60ms on Jetson)
- `yolov8n.pt` — PyTorch fallback (~85ms on Jetson, ~7ms on desktop GPU)

---

## Optimization for Jetson Nano

### Jetson Nano Hardware Constraints
- **GPU**: 128 Maxwell CUDA cores — 20x fewer than desktop RTX, 3 generations behind Ampere/Ada
- **CPU**: Quad-core ARM A57 @ 1.43 GHz — much slower than x86 for numpy/encoding
- **RAM**: 4GB shared between CPU and GPU — CUDA context takes ~500MB, model ~200MB
- **Power**: 10W max (MAXN mode) — thermal throttling at ~97°C, target 8W steady
- **Storage**: microSD card (~100MB/s read) — model loading takes 8-12s on cold boot

### 10 Optimizations for Jetson Nano

| # | Optimization | Impact on Jetson Nano | Code Location |
|---|-------------|----------------------|---------------|
| 1 | **TensorRT FP16 quantization** | 85ms → 35ms (2.4x speedup), <0.5% accuracy loss | `backend/inference/detector.py` |
| 2 | **Reduced input size (480px)** | 44% fewer pixels vs 640px default | `backend/config.py` |
| 3 | **IOU tracker (not DeepSORT)** | Saves 15-25ms/frame (no ReID CNN) | `backend/crowd/tracker.py` |
| 4 | **Threaded camera capture** | Eliminates 30-60ms blocking per camera | `backend/camera/manager.py` |
| 5 | **Single asyncio.to_thread()** | Avoids GIL contention on quad-core ARM A57 | `backend/main.py` |
| 6 | **JPEG Q60 compression** | 900KB → 30KB/frame (critical on WiFi) | `backend/main.py` |
| 7 | **Fixed-size deques** | Prevents OOM on 4GB shared RAM | All temporal buffers |
| 8 | **MAXN power mode** | Unlocks full GPU/CPU clocks | `scripts/setup_jetson.sh` |
| 9 | **Systemd auto-start** | Standalone appliance, no monitor needed | `scripts/crowd-control.service` |
| 10 | **Platform-aware camera backends** | V4L2 on Jetson, DSHOW on Windows | `backend/camera/manager.py` |

### Performance Benchmarks (Jetson Nano)

| Stage | Jetson Nano (TensorRT FP16) | Notes |
|-------|---------------------------|-------|
| Camera capture | ~3.8ms | V4L2 USB capture |
| YOLOv8n detection | ~35ms | TensorRT FP16, imgsz=480 |
| IOU tracking | ~1.4ms | Greedy IOU matching |
| Crowd analysis | ~2.5ms | Gaussian KDE + velocity |
| JPEG encoding | ~5.5ms | Q60 on ARM A57 |
| WebSocket broadcast | ~1.8ms | Push to all clients |
| **Total per frame** | **~50ms (~20 FPS)** | **33% headroom over 15 FPS target** |
| **2 cameras** | **~100ms (~10 FPS)** | Reduce imgsz to 320 for 15 FPS |

### Quantization: Accuracy vs Speed

| Precision | mAP50 (person) | Inference (Jetson Nano) | Model Size | Accuracy Drop |
|-----------|---------------|------------------------|------------|---------------|
| **FP32** (PyTorch) | ~52.5% | ~85 ms | ~12 MB | Baseline |
| **FP16** (TensorRT) | ~52.3% | ~35 ms | ~6 MB | **< 0.5%** (negligible) |
| **INT8** (TensorRT) | ~50.5% | ~20 ms | ~3.5 MB | **~3-4%** (needs calibration) |

We use **FP16** as the production default — negligible accuracy loss with 2.4x speedup. FP16 bounding boxes differ by ±0.2 pixels from FP32 — invisible and irrelevant for crowd counting.

### Resource Budget (Jetson Nano 4GB)

| Resource | Usage | Limit | Headroom |
|----------|-------|-------|----------|
| **GPU** | ~70% (TensorRT FP16) | 128 Maxwell cores | ~30% |
| **CPU** | ~45% (tracking/analysis) | 4x ARM A57 @ 1.43 GHz | ~55% |
| **RAM** | ~1.8 GB (model + CUDA + buffers) | 4 GB shared CPU/GPU | ~2.2 GB |
| **Power** | ~8W (MAXN mode) | 10W max | ~2W thermal headroom |
| **Storage** | ~350 MB (model + venv + dashboard) | SD card | Depends on card |

### Jetson Nano vs Desktop GPU

| Metric | Jetson Nano (TensorRT FP16) | Desktop RTX 4050 |
|--------|-----------------------------|--------------------|
| Inference latency | ~35 ms | ~7 ms |
| Full pipeline | ~50 ms | ~11 ms |
| Power consumption | 8W | 200W+ |
| Cost | ~$99 | ~$1200+ system |
| **FPS per watt** | **2.5 FPS/W** | **0.44 FPS/W** |

Jetson Nano is **5.7x more power-efficient** — critical for 24/7 surveillance systems.

---

## How Crowd Status Works

### Thresholds (customizable per area via Dashboard Settings)

| Parameter | Default | What It Means |
|-----------|---------|---------------|
| `max_count` | 50 | Person count exceeding this → CRITICAL |
| `density_limit` | 0.8 | Clustering density (0-1) exceeding this → CRITICAL |
| `chaos_threshold` | 3.0 | Movement chaos (velocity std) exceeding this → CRITICAL |

### Status Logic

```
CRITICAL (red, pulsing) if ANY of:
  - smoothed person count > max_count
  - smoothed density > density_limit
  - smoothed chaos > chaos_threshold

WARNING (yellow) if ANY of:
  - smoothed person count > 75% of max_count
  - smoothed density > 75% of density_limit

NORMAL (green): otherwise
```

### How Density Is Calculated

1. Frame divided into 20x20 grid (400 cells)
2. Each detected person adds a Gaussian contribution (sigma=1.5 cells) centered at their position
3. Peak density is **scaled by max_count** (not self-normalized):
   - 1 person with max_count=50 → density 0.02 → NORMAL
   - 40 people clustered with max_count=50 → density 0.80 → CRITICAL
4. Grid is self-normalized to [0,1] only for heatmap visualization (colors)
5. Status evaluation uses the properly-scaled density value

### Temporal Smoothing + Hysteresis

- **Smoothing**: Exponential weighted moving average over last 30 frames — recent frames count more
- **Hysteresis**: Status must be consistent for 5 consecutive frames before transitioning
- **Result**: Prevents flickering. A person briefly leaving frame doesn't cause instant status drop.

---

## Customizing Thresholds

### From the Dashboard UI (recommended)

1. Open `http://localhost:8000` in your browser
2. Click **Settings** button (top-right header)
3. For each area, adjust:
   - **Area Name** — display label
   - **Camera Index** — which USB camera (0, 1, ...)
   - **Max Person Count** — CRITICAL threshold for person count
   - **Density Limit (0-1)** — CRITICAL threshold for clustering
   - **Chaos Threshold** — CRITICAL threshold for movement chaos
4. Click **Save** — applies immediately, persists across restarts

### From config.json directly

Edit `backend/data/config.json`:
```json
{
  "areas": [
    {
      "name": "Main Entrance",
      "camera_index": 0,
      "max_count": 30,
      "density_limit": 0.7,
      "chaos_threshold": 2.5
    },
    {
      "name": "Auditorium",
      "camera_index": 1,
      "max_count": 100,
      "density_limit": 0.8,
      "chaos_threshold": 3.0
    }
  ],
  "detection": {
    "confidence_threshold": 0.4,
    "model_path": "models/yolov8n.pt",
    "input_size": 480
  },
  "smoothing_window": 30,
  "hysteresis_frames": 5,
  "alert_cooldown_seconds": 30
}
```

---

## Quick Start

### Run on PC (Development)

**Prerequisites**: Python 3.8+, NVIDIA GPU with CUDA (or CPU fallback), 1-2 USB cameras

```bash
# 1. Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 2. Install project dependencies
pip install -r backend/requirements.txt

# 3. Copy model to expected location
cp yolov8n.pt backend/models/

# 4. Start the server
python main.py

# 5. Open browser
# http://localhost:8000
```

### Deploy on Jetson Nano (Production)

**Prerequisites**: Jetson Nano 4GB with JetPack 4.6+ (Python 3.8), 2 USB cameras

```bash
# 1. One-time Jetson setup (MAXN power mode, verify CUDA/TensorRT)
bash scripts/setup_jetson.sh

# 2. Install dependencies + download model (auto-detects Jetson, uses Python 3.8)
bash scripts/install.sh

# 3. Export model to TensorRT FP16 (run ONCE on Jetson — takes ~5 minutes)
source venv/bin/activate
python3 training/export.py --weights backend/models/yolov8n.pt --format engine --half
# Creates backend/models/yolov8n.engine — system auto-detects and uses it

# 4. Start the monitor
bash scripts/start.sh
# Dashboard available at http://<jetson-ip>:8000

# 5. (Optional) Auto-start on boot
sudo cp scripts/crowd-control.service /etc/systemd/system/
sudo systemctl enable crowd-control
sudo systemctl start crowd-control
# View logs: journalctl -u crowd-control -f
```

### Custom Model Training (Optional)

Train YOLOv8-nano on your own crowd dataset for better accuracy:

```bash
# 1. Prepare dataset in YOLO format (see training/dataset.yaml)
# 2. Train
python training/train.py --data training/dataset.yaml --epochs 50

# 3. Export best model
python training/export.py --weights runs/detect/crowd_v1/weights/best.pt --format engine --half

# 4. Copy to models directory
cp runs/detect/crowd_v1/weights/best.engine backend/models/yolov8n.engine
```

---

## API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Dashboard UI (served from dashboard/dist/) |
| `/ws` | WebSocket | Real-time frame + detection stream at 15 FPS |
| `/api/health` | GET | System status, active cameras, configured areas |
| `/api/config` | GET | Full current configuration |
| `/api/config` | POST | Update thresholds per area (live, no restart) |
| `/api/alerts` | GET | Alert history (last 100) |
| `/api/cameras` | GET | Available + active camera indices |
| `/api/perf` | GET | Rolling performance statistics (avg/min/max/p95/p99) |
| `/api/perf/latest` | GET | Latest single frame timing breakdown |

### WebSocket Message Format

```json
{
  "type": "frame_update",
  "timestamp": 1700000000.123,
  "areas": [
    {
      "name": "Area 1",
      "camera_index": 0,
      "frame": "<base64-encoded-jpeg>",
      "person_count": 12,
      "status": "NORMAL",
      "detections": [
        {"bbox": [100, 200, 150, 350], "confidence": 0.87, "track_id": 5}
      ],
      "density_grid": [[0.0, 0.1, ...], ...],
      "avg_velocity": 1.2,
      "velocity_std": 0.8
    }
  ],
  "perf": {"capture_ms": 0.5, "detect_ms": 25.0, ...}
}
```

---

## Advantages

1. **Real-time**: ~20 FPS capacity on Jetson Nano (TensorRT FP16), 33% headroom over 15 FPS target
2. **Edge deployment**: Runs entirely on Jetson Nano — no cloud, no internet, no ongoing costs
3. **Power efficient**: 2.5 FPS/Watt vs 0.44 FPS/Watt on desktop — 5.7x more efficient for 24/7 operation
4. **Privacy-preserving**: Video never leaves the device; all processing on-device
5. **Low cost**: Jetson Nano (~$99) + 2 USB cameras (~$30 each) = ~$160 total vs $40-130/day cloud API
6. **Customizable**: Adjust all thresholds per area from the dashboard UI — no code changes needed
7. **Multi-area**: 2 independent monitoring zones with separate thresholds and status
8. **Noise-resistant**: Temporal smoothing + hysteresis prevents false alarms
9. **Browser-based**: Any device on the network can monitor — no app install needed
10. **Cross-platform**: Same code runs on Windows (development, Python 3.10+) and Jetson Nano (production, Python 3.8)
11. **Auto-optimizing**: Detects GPU/CPU, finds best model format (TensorRT > ONNX > PyTorch)
12. **Standalone appliance**: Auto-starts on boot via systemd, no monitor/keyboard needed

## Disadvantages / Limitations

1. **2 camera max on Jetson Nano**: 128 CUDA cores limit — 3+ cameras drops below 10 FPS
2. **No re-identification**: IOU tracker loses track if person is fully occluded then reappears
3. **Fixed cameras assumed**: Tracking works best with stationary cameras
4. **No individual identification**: Counts people, does not recognize faces or identities
5. **Low-light sensitivity**: YOLOv8-nano accuracy drops in poor lighting — use IR cameras for 24/7
6. **No distributed mode**: Each Jetson Nano is independent (no multi-device clustering)
7. **In-memory alerts**: Alert history lost on restart (no database persistence)
8. **LAN only**: Dashboard requires same network — no remote monitoring without VPN/tunnel

---

## Project Structure

```
talk/
├── main.py                          # Entry point — python main.py starts everything
├── yolov8n.pt                       # Pre-trained YOLOv8-nano model (6.5 MB)
├── README.md                        # This file
├── PROJECT_WRITEUP.md               # Full project writeup with benchmarks + mock interview prep
│
├── backend/
│   ├── main.py                      # FastAPI app + WebSocket + processing loop (343 lines)
│   ├── config.py                    # Pydantic config loader + JSON persistence
│   ├── requirements.txt             # PC dependencies (opencv-python, not headless)
│   ├── requirements-jetson.txt      # Jetson Nano dependencies (uses system OpenCV)
│   ├── camera/
│   │   └── manager.py               # Camera capture (DSHOW/V4L2 auto-detect) + demo mode
│   ├── inference/
│   │   └── detector.py              # YOLOv8 detector (auto: TensorRT > ONNX > PyTorch)
│   ├── crowd/
│   │   ├── tracker.py               # IOU multi-object tracker with velocity
│   │   ├── analyzer.py              # Gaussian KDE density + movement + status logic
│   │   └── alerts.py                # Alert manager with cooldown
│   ├── perf/
│   │   ├── profiler.py              # Per-stage pipeline profiler
│   │   └── benchmark.py             # CPU vs GPU benchmarking tool
│   ├── models/
│   │   └── yolov8n.pt               # Model (auto-detects .engine/.onnx/.pt)
│   └── data/
│       └── config.json              # Runtime config (editable via Settings UI)
│
├── dashboard/
│   ├── src/                         # React source
│   │   ├── App.jsx                  # Main layout
│   │   ├── contexts/CrowdContext.jsx # WebSocket + state management
│   │   └── components/              # CameraFeed, StatusPanel, AlertLog, Settings, PerfOverlay
│   ├── dist/                        # Pre-built production dashboard
│   ├── package.json                 # React 18 + Vite 5 + Tailwind 3.3
│   └── vite.config.js               # Dev proxy: /api → :8000, /ws → ws://:8000
│
├── training/
│   ├── train.py                     # YOLOv8-nano training (50 epochs, single-class)
│   ├── export.py                    # Export: .pt → .onnx → .engine (TensorRT FP16)
│   ├── dataset.yaml                 # Dataset config (COCO person / CrowdHuman)
│   └── requirements-training.txt    # Training dependencies
│
└── scripts/
    ├── setup_jetson.sh              # One-time Jetson Nano setup (power mode, CUDA verify)
    ├── install.sh                   # Install deps (auto-detects Jetson vs PC)
    ├── start.sh                     # Launch server (shows LAN IP)
    └── crowd-control.service        # systemd service for auto-start on boot
```

---

## Technology Stack

| Layer | Technology | Why This Choice |
|-------|-----------|-----------------|
| Detection | YOLOv8-nano (Ultralytics) | Smallest YOLO — only model that fits Jetson Nano at real-time speed |
| Inference | PyTorch / TensorRT FP16 | TensorRT gives 7x speedup on Jetson; PyTorch for development |
| Tracking | Custom IOU Tracker | <1ms vs DeepSORT's 15ms — no extra neural network needed for fixed cameras |
| Backend | FastAPI + Uvicorn | Native async + WebSocket support in Python |
| Streaming | WebSocket (base64 JPEG) | Low-latency, bidirectional, works through proxies |
| Frontend | React 18 + Vite 5 | Canvas rendering for overlays, fast rebuild |
| Styling | Tailwind CSS 3.3 | Dark theme, responsive, utility-first |
| Vision | OpenCV 4.x | Camera capture (V4L2/DSHOW), JPEG encoding |
| Config | Pydantic + JSON | Type-safe config with live UI editing |
| Hardware | NVIDIA Jetson Nano 4GB | Edge AI: 128 Maxwell CUDA cores, 10W MAXN, $99 |
| Quantization | TensorRT FP16 | 2.4x speedup, <0.5% accuracy loss, native Maxwell support |
| Python | 3.8+ (with `from __future__ import annotations`) | Compatible with both Jetson Nano (3.8) and modern PCs (3.10+) |
