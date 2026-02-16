from __future__ import annotations

from typing import List

import numpy as np


def iou(box_a, box_b) -> float:
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0


class Track:
    def __init__(self, track_id: int, bbox: list, centroid: list):
        self.track_id = track_id
        self.bbox = bbox
        self.centroid = np.array(centroid, dtype=np.float32)
        self.prev_centroid = self.centroid.copy()
        self.velocity = np.array([0.0, 0.0], dtype=np.float32)
        self.frames_since_update = 0

    def update(self, bbox: list, centroid: list):
        self.prev_centroid = self.centroid.copy()
        self.bbox = bbox
        self.centroid = np.array(centroid, dtype=np.float32)
        self.velocity = self.centroid - self.prev_centroid
        self.frames_since_update = 0

    def mark_missed(self):
        self.frames_since_update += 1


class IOUTracker:
    """Simple IOU-based multi-object tracker. Lightweight for Jetson Nano."""

    def __init__(self, iou_threshold: float = 0.3, max_lost: int = 15):
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost
        self.tracks: list[Track] = []
        self._next_id = 1

    def update(self, detections: list[dict]) -> list[dict]:
        """
        Match detections to existing tracks using IOU.

        Args:
            detections: list of {"bbox": [x1,y1,x2,y2], "confidence": float, "centroid": [cx,cy]}

        Returns:
            list of {"bbox", "confidence", "centroid", "track_id", "velocity": [vx,vy]}
        """
        if not detections:
            for track in self.tracks:
                track.mark_missed()
            self.tracks = [t for t in self.tracks if t.frames_since_update <= self.max_lost]
            return []

        det_bboxes = [d["bbox"] for d in detections]
        matched_det = set()
        matched_track = set()

        # Compute IOU matrix
        if self.tracks:
            iou_matrix = np.zeros((len(self.tracks), len(detections)))
            for i, track in enumerate(self.tracks):
                for j, det_bbox in enumerate(det_bboxes):
                    iou_matrix[i, j] = iou(track.bbox, det_bbox)

            # Greedy matching: pick highest IOU pairs
            while True:
                if iou_matrix.size == 0:
                    break
                max_val = iou_matrix.max()
                if max_val < self.iou_threshold:
                    break
                idx = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
                ti, di = int(idx[0]), int(idx[1])

                self.tracks[ti].update(detections[di]["bbox"], detections[di]["centroid"])
                matched_det.add(di)
                matched_track.add(ti)

                iou_matrix[ti, :] = 0
                iou_matrix[:, di] = 0

        # Mark unmatched tracks as missed
        for i, track in enumerate(self.tracks):
            if i not in matched_track:
                track.mark_missed()

        # Create new tracks for unmatched detections
        for j, det in enumerate(detections):
            if j not in matched_det:
                new_track = Track(self._next_id, det["bbox"], det["centroid"])
                self.tracks.append(new_track)
                self._next_id += 1

        # Remove lost tracks
        self.tracks = [t for t in self.tracks if t.frames_since_update <= self.max_lost]

        # Build output
        results = []
        for track in self.tracks:
            if track.frames_since_update == 0:
                results.append({
                    "bbox": track.bbox,
                    "confidence": 0.0,
                    "centroid": track.centroid.tolist(),
                    "track_id": track.track_id,
                    "velocity": track.velocity.tolist(),
                })

        # Attach confidence from original detections where possible
        for r in results:
            for d in detections:
                if iou(r["bbox"], d["bbox"]) > 0.5:
                    r["confidence"] = d["confidence"]
                    break

        return results
