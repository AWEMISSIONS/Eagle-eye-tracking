from __future__ import annotations

import json
import math
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


@dataclass
class AutoSpeedResult:
    mph: float
    confidence: float
    source: str = "AUTO"


class AdaptiveLearningEngine:
    """Small, transparent local learning layer.

    It does not rewrite application code. It learns operating parameters from feedback
    and persists them locally so detection becomes better tuned to one fixed camera.
    """

    DEFAULTS = {
        "version": 1,
        "feedback_count": 0,
        "false_alarm_count": 0,
        "missed_count": 0,
        "motion_sensitivity": 7.0,
        "confidence": 0.28,
        "night_threshold": 78.0,
        "vehicle_confirm_frames": 4,
        "person_confirm_frames": 3,
        "auto_speed_enabled": True,
        "auto_speed_scale": 1.0,
        "auto_speed_samples": 0,
        "headlight_assist": True,
        "auto_start_monitoring": False,
        "last_source": "0",
        "performance_mode": "Balanced",
        "learned_notes": [],
    }

    def __init__(self, path: Path):
        self.path = path
        self.profile = dict(self.DEFAULTS)
        self.load()

    def load(self):
        try:
            if self.path.exists():
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    self.profile.update(raw)
        except Exception:
            pass
        self._clamp()

    def save(self):
        self._clamp()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.profile, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def _clamp(self):
        p = self.profile
        p["motion_sensitivity"] = float(max(1.0, min(10.0, p.get("motion_sensitivity", 7.0))))
        p["confidence"] = float(max(0.15, min(0.65, p.get("confidence", 0.28))))
        p["night_threshold"] = float(max(35.0, min(140.0, p.get("night_threshold", 78.0))))
        p["vehicle_confirm_frames"] = int(max(3, min(9, p.get("vehicle_confirm_frames", 4))))
        p["person_confirm_frames"] = int(max(2, min(7, p.get("person_confirm_frames", 3))))
        p["auto_speed_scale"] = float(max(0.45, min(2.2, p.get("auto_speed_scale", 1.0))))

    def apply_to_engine(self, engine):
        engine.motion_sensitivity = float(self.profile["motion_sensitivity"])
        engine.confidence = float(self.profile["confidence"])
        engine.night_threshold = float(self.profile["night_threshold"])
        engine.vehicle_confirm_frames = int(self.profile["vehicle_confirm_frames"])
        engine.person_confirm_frames = int(self.profile["person_confirm_frames"])
        engine.auto_speed_enabled = bool(self.profile.get("auto_speed_enabled", True))
        engine.auto_speed_scale = float(self.profile.get("auto_speed_scale", 1.0))
        engine.headlight_assist = bool(self.profile.get("headlight_assist", True))

    def feedback_false_alarm(self, reason: str = "") -> str:
        p = self.profile
        p["feedback_count"] = int(p.get("feedback_count", 0)) + 1
        p["false_alarm_count"] = int(p.get("false_alarm_count", 0)) + 1
        # Become a little stricter, but only in small increments.
        p["motion_sensitivity"] = float(p.get("motion_sensitivity", 7.0)) - 0.30
        p["confidence"] = float(p.get("confidence", 0.28)) + 0.01
        p["vehicle_confirm_frames"] = int(p.get("vehicle_confirm_frames", 4)) + 1
        if reason:
            self._add_note("false alarm: " + reason)
        self.save()
        return self.summary(short=True)

    def feedback_missed(self, reason: str = "") -> str:
        p = self.profile
        p["feedback_count"] = int(p.get("feedback_count", 0)) + 1
        p["missed_count"] = int(p.get("missed_count", 0)) + 1
        # Become a little more sensitive, with conservative bounds.
        p["motion_sensitivity"] = float(p.get("motion_sensitivity", 7.0)) + 0.35
        p["confidence"] = float(p.get("confidence", 0.28)) - 0.012
        p["vehicle_confirm_frames"] = int(p.get("vehicle_confirm_frames", 4)) - 1
        if reason:
            self._add_note("missed detection: " + reason)
        self.save()
        return self.summary(short=True)

    def feedback_night_too_dark(self) -> str:
        self.profile["night_threshold"] = float(self.profile.get("night_threshold", 78.0)) + 5.0
        self._add_note("night scene needed earlier/stronger assist")
        self.save()
        return self.summary(short=True)

    def feedback_night_too_noisy(self) -> str:
        self.profile["night_threshold"] = float(self.profile.get("night_threshold", 78.0)) - 5.0
        self._add_note("night assist was too aggressive/noisy")
        self.save()
        return self.summary(short=True)

    def learn_speed_scale(self, auto_mph: float, stronger_mph: float):
        if auto_mph <= 1 or stronger_mph <= 1:
            return
        ratio = max(0.55, min(1.8, stronger_mph / auto_mph))
        old = float(self.profile.get("auto_speed_scale", 1.0))
        n = int(self.profile.get("auto_speed_samples", 0))
        alpha = 0.22 if n < 6 else 0.08
        self.profile["auto_speed_scale"] = old * (1.0 - alpha) + ratio * alpha
        self.profile["auto_speed_samples"] = n + 1
        self.save()

    def record_request(self, text: str):
        self._add_note("requested capability: " + text.strip())
        self.save()

    def _add_note(self, note: str):
        notes = list(self.profile.get("learned_notes", []))
        notes.append({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "note": note[:500]})
        self.profile["learned_notes"] = notes[-100:]

    def reset(self):
        self.profile = dict(self.DEFAULTS)
        self.save()

    def summary(self, short: bool = False) -> str:
        p = self.profile
        base = (
            f"Learning profile: motion {p['motion_sensitivity']:.1f}/10, "
            f"AI confidence {p['confidence']:.2f}, confirm {p['vehicle_confirm_frames']} frames, "
            f"night threshold {p['night_threshold']:.0f}, auto-speed scale {p['auto_speed_scale']:.2f}."
        )
        if short:
            return base
        return base + (
            f" Feedback: {p.get('feedback_count',0)} total "
            f"({p.get('false_alarm_count',0)} false alarms, {p.get('missed_count',0)} misses)."
        )


class AutoSpeedEstimator:
    """No-input monocular speed estimate.

    Uses movement relative to apparent object size. This is an estimate, not a calibrated
    measurement. It becomes more useful on a fixed camera and can learn a correction factor.
    """

    NOMINAL_LENGTH_FT = {
        "Car": 15.0,
        "Truck": 18.5,
        "Bus": 35.0,
        "Motorcycle": 7.0,
    }

    def __init__(self):
        self.hist: dict[int, deque] = {}
        self.ema: dict[int, float] = {}

    def reset(self):
        self.hist.clear()
        self.ema.clear()

    def forget(self, track_id: int):
        self.hist.pop(track_id, None)
        self.ema.pop(track_id, None)

    def update(self, track_id: int, label: str, center: tuple[int, int], xyxy, now: float,
               scale: float = 1.0, night: bool = False) -> Optional[AutoSpeedResult]:
        x1, y1, x2, y2 = xyxy
        bw = max(1.0, float(x2 - x1))
        bh = max(1.0, float(y2 - y1))
        size = max(bw, bh * 1.6)
        hist = self.hist.setdefault(track_id, deque(maxlen=28))
        hist.append((now, float(center[0]), float(center[1]), size))
        while hist and now - hist[0][0] > 1.20:
            hist.popleft()
        if len(hist) < 5:
            return None
        dt = hist[-1][0] - hist[0][0]
        if dt < 0.28:
            return None

        # Robust path length and object-size normalization. This handles perspective better
        # than raw px/s but still remains only an estimate with one uncalibrated camera.
        steps = []
        for a, b in zip(hist, list(hist)[1:]):
            d = math.hypot(b[1] - a[1], b[2] - a[2])
            dd = max(b[0] - a[0], 1e-4)
            mid_size = max((a[3] + b[3]) * 0.5, 1.0)
            steps.append((d / dd) / mid_size)
        if not steps:
            return None
        norm_lengths_per_s = float(np.median(steps))
        nominal = self.NOMINAL_LENGTH_FT.get(label, 15.0)
        mph = norm_lengths_per_s * nominal * 0.6818181818 * float(scale)

        # The apparent-size method tends to overreact to box jitter at long range.
        # Use a conservative range and smoothing.
        if not (1.0 <= mph <= 120.0):
            return None
        prev = self.ema.get(track_id)
        smooth = mph if prev is None else 0.72 * prev + 0.28 * mph
        self.ema[track_id] = smooth

        avg_size = float(np.mean([x[3] for x in hist]))
        samples = len(hist)
        stability = 1.0 - min(1.0, float(np.std(steps)) / max(norm_lengths_per_s, 0.001))
        size_factor = min(1.0, avg_size / 170.0)
        duration_factor = min(1.0, dt / 0.9)
        confidence = 0.25 + 0.28 * stability + 0.24 * size_factor + 0.23 * duration_factor
        if night:
            confidence *= 0.78
        confidence = max(0.20, min(0.88, confidence))
        return AutoSpeedResult(float(smooth), float(confidence))


class HeadlightAssist:
    """Conservative night hint detector for moving headlight pairs.

    It is intentionally only an assist. It does not pretend that two bright pixels are a
    confirmed vehicle; it returns a confidence and only looks for plausible paired blobs.
    """

    def __init__(self):
        self.prev_centers: deque = deque(maxlen=8)
        self.last_alert = 0.0

    def reset(self):
        self.prev_centers.clear()
        self.last_alert = 0.0

    def detect(self, frame: np.ndarray) -> tuple[Optional[tuple[int,int,int,int]], float]:
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Bright regions; use a high percentile to adapt to different cameras.
        thresh_val = max(205, int(np.percentile(gray, 99.35)))
        _, mask = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
        mask = cv2.dilate(mask, np.ones((5,5), np.uint8), iterations=1)
        n, labels, stats, cents = cv2.connectedComponentsWithStats(mask)
        blobs = []
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            if area < 8 or area > (w*h*0.012):
                continue
            if y < h * 0.20:  # usually sky/ceiling reflections
                continue
            blobs.append((x, y, bw, bh, area, cents[i][0], cents[i][1]))
        best = None
        best_score = 0.0
        for i in range(len(blobs)):
            for j in range(i+1, len(blobs)):
                a, b = blobs[i], blobs[j]
                dx = abs(a[5] - b[5])
                dy = abs(a[6] - b[6])
                avg_h = max((a[3] + b[3]) * 0.5, 1.0)
                if dx < avg_h * 1.0 or dx > w * 0.28:
                    continue
                if dy > max(18.0, avg_h * 1.8):
                    continue
                area_ratio = min(a[4], b[4]) / max(a[4], b[4])
                if area_ratio < 0.20:
                    continue
                score = 0.45 + 0.30 * area_ratio + 0.25 * (1.0 - min(1.0, dy / max(dx,1.0)))
                if score > best_score:
                    x1 = int(min(a[0], b[0]) - dx * 0.35)
                    x2 = int(max(a[0]+a[2], b[0]+b[2]) + dx * 0.35)
                    y1 = int(min(a[1], b[1]) - avg_h * 2.0)
                    y2 = int(max(a[1]+a[3], b[1]+b[3]) + avg_h * 2.4)
                    best = (max(0,x1), max(0,y1), min(w-1,x2), min(h-1,y2))
                    best_score = score
        if best is None:
            return None, 0.0
        cx = (best[0] + best[2]) * 0.5
        cy = (best[1] + best[3]) * 0.5
        now = time.time()
        self.prev_centers.append((now, cx, cy))
        while self.prev_centers and now - self.prev_centers[0][0] > 1.2:
            self.prev_centers.popleft()
        motion_bonus = 0.0
        if len(self.prev_centers) >= 3:
            _, x0, y0 = self.prev_centers[0]
            _, x1c, y1c = self.prev_centers[-1]
            move = math.hypot(x1c-x0, y1c-y0) / max(math.hypot(w,h), 1.0)
            # Require visible movement to suppress porch lights/reflections.
            if move < 0.004:
                return best, min(0.55, best_score * 0.62)
            motion_bonus = min(0.18, move * 8.0)
        return best, float(max(0.0, min(1.0, best_score + motion_bonus)))
