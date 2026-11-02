"""Decision stage and Warden (2018) §7.2 streaming metrics — a NumPy port of ``src/streaming_eval.py``.

Used by the operating-point simulation and, later, to re-score ``RAW`` logs from the board. The decision stage
mirrors the device: a queue of the last K probability vectors, arg-max over the keyword classes only, a
detection when the smoothed probability reaches the threshold and the suppression interval has elapsed.
The energy gate skips windows (no inference) and clears the queue.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class GTEvent:
    start_ms: float
    end_ms: float
    center_ms: float
    label: str


@dataclass
class Detection:
    fire_time_ms: float
    label: str
    confidence: float = 0.0


def load_gt(path: Path) -> list[GTEvent]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [GTEvent(e["start_ms"], e["end_ms"], e["center_ms"], e["label"]) for e in data]


def decide(times_ms: np.ndarray, probs: np.ndarray, classes: list[str], keyword_indices: list[int],
           k: int, threshold: float, suppression_ms: float, gate_pass: np.ndarray | None = None) -> list[Detection]:
    """Smoothing + threshold + global suppression, exactly as ``collect_raw_events``/``apply_detection_logic``.

    times_ms: fire time of each window (end of the 1-s window); probs: (n, n_classes) probabilities in window order;
    gate_pass: optional bool mask — False windows are skipped (no inference on the device) and reset the queue.
    """
    kw = np.asarray(keyword_indices)
    queue: list[np.ndarray] = []
    detections: list[Detection] = []
    last_fire = -np.inf
    for i in range(len(times_ms)):
        if gate_pass is not None and not gate_pass[i]:
            queue.clear()
            continue
        queue.append(probs[i])
        if len(queue) > k:
            queue.pop(0)
        if len(queue) < k:
            continue
        smoothed = np.mean(queue, axis=0)
        best_local = int(np.argmax(smoothed[kw]))
        best_idx = int(kw[best_local])
        best_prob = float(smoothed[best_idx])
        t = float(times_ms[i])
        if best_prob >= threshold and (t - last_fire) >= suppression_ms:
            detections.append(Detection(t, classes[best_idx], best_prob))
            last_fire = t
    return detections


def score(detections: list[Detection], gt_events: list[GTEvent], stream_duration_s: float,
          target_classes: list[str], tolerance_ms: float = 750.0) -> dict:
    """Greedy nearest-unclaimed matching within ±tolerance; Warden's matched/wrong/missed/FP metrics."""
    dets = sorted(detections, key=lambda d: d.fire_time_ms)
    gt = sorted(gt_events, key=lambda e: e.center_ms)
    claimed = [False] * len(gt)
    match: list[int | None] = [None] * len(dets)
    for di, det in enumerate(dets):
        best_j, best_dist = None, float("inf")
        for gj, ev in enumerate(gt):
            if claimed[gj]:
                continue
            dist = abs(det.fire_time_ms - ev.center_ms)
            if dist <= tolerance_ms and dist < best_dist:
                best_dist, best_j = dist, gj
        if best_j is not None:
            claimed[best_j] = True
            match[di] = best_j
    matched = sum(1 for di, d in enumerate(dets) if match[di] is not None and gt[match[di]].label == d.label)
    wrong = sum(1 for di, d in enumerate(dets) if match[di] is not None and gt[match[di]].label != d.label)
    false_pos = sum(1 for m in match if m is None)
    missed = sum(1 for c in claimed if not c)
    n_gt = len(gt)
    minutes = stream_duration_s / 60.0
    per_class = {}
    for cls in target_classes:
        idx_gt = [j for j, e in enumerate(gt) if e.label == cls]
        per_class[cls] = {
            "n_gt": len(idx_gt),
            "matched": sum(1 for di, d in enumerate(dets) if d.label == cls and match[di] is not None and gt[match[di]].label == cls),
            "missed": sum(1 for j in idx_gt if not claimed[j]),
            "false_positives": sum(1 for di, d in enumerate(dets) if d.label == cls and match[di] is None),
        }
    return {
        "n_gt_events": n_gt, "n_detections": len(dets), "matched": matched, "wrong": wrong, "missed": missed,
        "false_positives": false_pos,
        "matched_rate": matched / n_gt if n_gt else 0.0, "wrong_rate": wrong / n_gt if n_gt else 0.0,
        "miss_rate": missed / n_gt if n_gt else 0.0,
        "false_positive_rate": false_pos / len(dets) if dets else 0.0,
        "false_positives_per_minute": false_pos / minutes if minutes else 0.0,
        "stream_duration_s": stream_duration_s, "per_class": per_class,
    }
