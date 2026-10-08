from __future__ import annotations

import csv
import json
import math
import os
import queue
import re
import sqlite3
import sys
import threading
import time
import shutil
import zipfile
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
from adaptive import AdaptiveLearningEngine, AutoSpeedEstimator, HeadlightAssist
from zones import ZoneManager

try:
    import winsound
except ImportError:  # pragma: no cover - Windows target
    winsound = None

try:
    import pyttsx3
except Exception:
    pyttsx3 = None


# -----------------------------------------------------------------------------
# Paths / model classes
# -----------------------------------------------------------------------------
if getattr(sys, "frozen", False):
    APP_DIR = Path(sys.executable).resolve().parent
else:
    APP_DIR = Path(__file__).resolve().parents[1]

os.chdir(APP_DIR)
DATA_DIR = APP_DIR / "data"
SNAPSHOT_DIR = DATA_DIR / "snapshots"
IDENTITY_DIR = DATA_DIR / "identities"
DB_PATH = DATA_DIR / "streetwatch.db"
MODEL_PATH = APP_DIR / "yolo26n.pt"
LEARNING_PATH = DATA_DIR / "learning_profile.json"
ZONES_PATH = DATA_DIR / "zones.json"

DATA_DIR.mkdir(parents=True, exist_ok=True)
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
IDENTITY_DIR.mkdir(parents=True, exist_ok=True)

VEHICLE_CLASSES = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}
PERSON_CLASSES = {0: "Person"}
ANIMAL_CLASSES = {
    14: "Bird", 15: "Cat", 16: "Dog", 17: "Horse", 18: "Sheep",
    19: "Cow", 20: "Elephant", 21: "Bear", 22: "Zebra", 23: "Giraffe",
}
TARGET_CLASSES = {**VEHICLE_CLASSES, **PERSON_CLASSES, **ANIMAL_CLASSES}

CATEGORY_BY_CLASS = {cid: "vehicle" for cid in VEHICLE_CLASSES}
CATEGORY_BY_CLASS.update({cid: "person" for cid in PERSON_CLASSES})
CATEGORY_BY_CLASS.update({cid: "animal" for cid in ANIMAL_CLASSES})

CATEGORY_COLORS = {
    "vehicle": (0, 220, 0),
    "person": (255, 180, 0),
    "animal": (0, 165, 255),
}


@dataclass
class Detection:
    track_id: int
    class_id: int
    category: str
    label: str
    confidence: float
    xyxy: tuple[int, int, int, int]
    center: tuple[int, int]
    color: str
    identity_code: str = ""
    identity_name: str = ""
    identity_match: float = 0.0
    speed_mph: Optional[float] = None
    speed_source: str = ""
    speed_confidence: float = 0.0
    zone: str = ""
    is_moving: bool = False
    motion_score: float = 0.0


# -----------------------------------------------------------------------------
# Offline speech worker
# -----------------------------------------------------------------------------
class SpeechWorker:
    def __init__(self):
        self.q: queue.Queue[str] = queue.Queue(maxsize=8)
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def say(self, text: str):
        if not text:
            return
        try:
            self.q.put_nowait(text)
        except queue.Full:
            pass

    def _run(self):
        engine = None
        if pyttsx3 is not None:
            try:
                engine = pyttsx3.init()
                engine.setProperty("rate", 185)
                engine.setProperty("volume", 0.9)
            except Exception:
                engine = None
        while self.running:
            try:
                text = self.q.get(timeout=0.4)
            except queue.Empty:
                continue
            if engine is not None:
                try:
                    engine.say(text)
                    engine.runAndWait()
                    continue
                except Exception:
                    engine = None
            if winsound is not None:
                try:
                    winsound.MessageBeep(winsound.MB_ICONASTERISK)
                except Exception:
                    pass

    def stop(self):
        self.running = False


# -----------------------------------------------------------------------------
# Fingerprinting for VEHICLES and ANIMALS only.
# People intentionally do not receive persistent visual identities.
# -----------------------------------------------------------------------------
def crop_with_pad(frame: np.ndarray, xyxy: tuple[int, int, int, int], pad_ratio: float = 0.08) -> np.ndarray:
    x1, y1, x2, y2 = xyxy
    pad_x = int((x2 - x1) * pad_ratio)
    pad_y = int((y2 - y1) * pad_ratio)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(frame.shape[1], x2 + pad_x)
    y2 = min(frame.shape[0], y2 + pad_y)
    return frame[y1:y2, x1:x2]


def dhash64(crop: np.ndarray) -> int:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]
    value = 0
    for bit in diff.flatten():
        value = (value << 1) | int(bit)
    return value


def make_fingerprint(crop: np.ndarray) -> Optional[dict]:
    if crop is None or crop.size == 0 or crop.shape[0] < 18 or crop.shape[1] < 18:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [12, 8], [0, 180, 0, 256])
    hist = cv2.normalize(hist, hist).flatten().astype(float)
    h, w = crop.shape[:2]
    return {
        "hist": hist.tolist(),
        "dhash": f"{dhash64(crop):016x}",
        "aspect": float(w / max(h, 1)),
    }


def hamming64(hex_a: str, hex_b: str) -> int:
    try:
        return (int(hex_a, 16) ^ int(hex_b, 16)).bit_count()
    except Exception:
        return 64


def fingerprint_similarity(a: dict, b: dict, color_a: str, color_b: str, type_a: str, type_b: str) -> float:
    try:
        ha = np.asarray(a.get("hist", []), dtype=np.float32)
        hb = np.asarray(b.get("hist", []), dtype=np.float32)
        if ha.shape != hb.shape or ha.size == 0:
            hist_sim = 0.0
        else:
            # Bhattacharyya in [0,1], converted to similarity.
            hist_sim = 1.0 - float(cv2.compareHist(ha, hb, cv2.HISTCMP_BHATTACHARYYA))
            hist_sim = max(0.0, min(1.0, hist_sim))
    except Exception:
        hist_sim = 0.0

    hash_sim = 1.0 - hamming64(str(a.get("dhash", "")), str(b.get("dhash", ""))) / 64.0
    ar_a = float(a.get("aspect", 1.0))
    ar_b = float(b.get("aspect", 1.0))
    aspect_sim = max(0.0, 1.0 - abs(math.log(max(ar_a, 0.01) / max(ar_b, 0.01))) / 1.2)

    if color_a == color_b:
        color_sim = 1.0
    elif {color_a, color_b} <= {"Gray/Silver", "White", "Black"}:
        color_sim = 0.55
    else:
        color_sim = 0.15

    if type_a == type_b:
        type_sim = 1.0
    elif {type_a, type_b} <= {"Car", "Truck", "Bus"}:
        type_sim = 0.55
    else:
        type_sim = 0.0

    return (
        0.36 * hist_sim
        + 0.34 * hash_sim
        + 0.10 * aspect_sim
        + 0.10 * color_sim
        + 0.10 * type_sim
    )


# -----------------------------------------------------------------------------
# Database
# -----------------------------------------------------------------------------
class StreetWatchDatabase:
    def __init__(self, path: Path):
        self.lock = threading.RLock()
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        with self.conn:
            self.conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS identities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    code TEXT UNIQUE,
                    category TEXT NOT NULL,
                    object_type TEXT NOT NULL,
                    color TEXT NOT NULL,
                    fingerprint_json TEXT NOT NULL,
                    name TEXT DEFAULT '',
                    first_seen TEXT NOT NULL,
                    last_seen TEXT NOT NULL,
                    seen_count INTEGER NOT NULL DEFAULT 1,
                    snapshot_path TEXT DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_code TEXT UNIQUE,
                    category TEXT NOT NULL,
                    happened_at TEXT NOT NULL,
                    object_type TEXT NOT NULL,
                    color TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    snapshot_path TEXT DEFAULT '',
                    track_id INTEGER,
                    identity_code TEXT DEFAULT '',
                    identity_match REAL DEFAULT 0,
                    display_name TEXT DEFAULT '',
                    speed_mph REAL,
                    speed_source TEXT DEFAULT '',
                    speed_confidence REAL DEFAULT 0,
                    bookmarked INTEGER DEFAULT 0,
                    note TEXT DEFAULT '',
                    zone TEXT DEFAULT ''
                );

                CREATE INDEX IF NOT EXISTS idx_events_category_time
                    ON events(category, happened_at DESC);
                CREATE INDEX IF NOT EXISTS idx_identities_category
                    ON identities(category);

                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    happened_at TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    event_code TEXT DEFAULT '',
                    note TEXT DEFAULT '',
                    zone TEXT DEFAULT ''
                );
                """
            )
            # Lightweight migrations for users upgrading an existing database.
            cols = {r[1] for r in self.conn.execute("PRAGMA table_info(events)").fetchall()}
            for name, ddl in [
                ("speed_source", "ALTER TABLE events ADD COLUMN speed_source TEXT DEFAULT ''"),
                ("speed_confidence", "ALTER TABLE events ADD COLUMN speed_confidence REAL DEFAULT 0"),
                ("bookmarked", "ALTER TABLE events ADD COLUMN bookmarked INTEGER DEFAULT 0"),
                ("note", "ALTER TABLE events ADD COLUMN note TEXT DEFAULT ''"),
                ("zone", "ALTER TABLE events ADD COLUMN zone TEXT DEFAULT ''"),
            ]:
                if name not in cols:
                    self.conn.execute(ddl)

    def find_identity(self, category: str, object_type: str, color: str, fingerprint: dict,
                      threshold: float) -> tuple[Optional[sqlite3.Row], float]:
        if category not in {"vehicle", "animal"}:
            return None, 0.0
        with self.lock:
            rows = self.conn.execute(
                """
                SELECT * FROM identities
                WHERE category = ?
                ORDER BY last_seen DESC
                LIMIT 350
                """,
                (category,),
            ).fetchall()
        best = None
        best_score = 0.0
        for row in rows:
            try:
                other = json.loads(row["fingerprint_json"])
            except Exception:
                continue
            score = fingerprint_similarity(
                fingerprint, other, color, row["color"], object_type, row["object_type"]
            )
            if score > best_score:
                best = row
                best_score = score
        if best is not None and best_score >= threshold:
            return best, best_score
        return None, best_score

    def create_identity(self, category: str, object_type: str, color: str,
                        fingerprint: dict, snapshot_path: str) -> sqlite3.Row:
        now = datetime.now().isoformat(timespec="seconds")
        prefix = "SWV" if category == "vehicle" else "SWA"
        with self.lock, self.conn:
            cur = self.conn.execute(
                """
                INSERT INTO identities
                (code, category, object_type, color, fingerprint_json, name,
                 first_seen, last_seen, seen_count, snapshot_path)
                VALUES (NULL, ?, ?, ?, ?, '', ?, ?, 1, ?)
                """,
                (category, object_type, color, json.dumps(fingerprint), now, now, snapshot_path),
            )
            row_id = cur.lastrowid
            code = f"{prefix}-{row_id:04d}"
            self.conn.execute("UPDATE identities SET code=? WHERE id=?", (code, row_id))
            return self.conn.execute("SELECT * FROM identities WHERE id=?", (row_id,)).fetchone()

    def touch_identity(self, code: str):
        now = datetime.now().isoformat(timespec="seconds")
        with self.lock, self.conn:
            self.conn.execute(
                "UPDATE identities SET last_seen=?, seen_count=seen_count+1 WHERE code=?",
                (now, code),
            )

    def set_identity_name(self, code: str, name: str) -> bool:
        with self.lock, self.conn:
            cur = self.conn.execute("UPDATE identities SET name=? WHERE code=?", (name.strip(), code.upper()))
            return cur.rowcount > 0

    def get_identity(self, code: str) -> Optional[sqlite3.Row]:
        with self.lock:
            return self.conn.execute("SELECT * FROM identities WHERE code=?", (code.upper(),)).fetchone()

    def add_event(self, category: str, object_type: str, color: str, direction: str,
                  confidence: float, snapshot_path: str, track_id: int,
                  identity_code: str = "", identity_match: float = 0.0,
                  speed_mph: Optional[float] = None, display_name: str = "",
                  speed_source: str = "", speed_confidence: float = 0.0, zone: str = "") -> sqlite3.Row:
        now = datetime.now().isoformat(timespec="seconds")
        with self.lock, self.conn:
            cur = self.conn.execute(
                """
                INSERT INTO events
                (event_code, category, happened_at, object_type, color, direction,
                 confidence, snapshot_path, track_id, identity_code, identity_match,
                 display_name, speed_mph, speed_source, speed_confidence, zone)
                VALUES (NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (category, now, object_type, color, direction, confidence,
                 snapshot_path, track_id, identity_code, identity_match,
                 display_name, speed_mph, speed_source, speed_confidence, zone),
            )
            row_id = cur.lastrowid
            prefix = {"vehicle": "V", "person": "P", "animal": "A"}[category]
            event_code = f"{prefix}-{row_id:06d}"
            self.conn.execute("UPDATE events SET event_code=? WHERE id=?", (event_code, row_id))
            return self.conn.execute("SELECT * FROM events WHERE id=?", (row_id,)).fetchone()

    def update_event_speed(self, event_id: int, speed_mph: float, source: str = "CALIBRATED", confidence: float = 0.95):
        with self.lock, self.conn:
            self.conn.execute(
                "UPDATE events SET speed_mph=?, speed_source=?, speed_confidence=? WHERE id=?",
                (speed_mph, source, confidence, event_id),
            )

    def add_feedback(self, kind: str, event_code: str = "", note: str = ""):
        with self.lock, self.conn:
            self.conn.execute(
                "INSERT INTO feedback(happened_at,kind,event_code,note) VALUES(?,?,?,?)",
                (datetime.now().isoformat(timespec="seconds"), kind, event_code, note),
            )

    def toggle_bookmark(self, event_code: str) -> bool:
        with self.lock, self.conn:
            row = self.conn.execute("SELECT bookmarked FROM events WHERE event_code=?", (event_code,)).fetchone()
            if not row:
                return False
            newv = 0 if int(row[0] or 0) else 1
            self.conn.execute("UPDATE events SET bookmarked=? WHERE event_code=?", (newv, event_code))
            return bool(newv)

    def set_event_note(self, event_code: str, note: str) -> bool:
        with self.lock, self.conn:
            cur = self.conn.execute("UPDATE events SET note=? WHERE event_code=?", (note.strip(), event_code))
            return cur.rowcount > 0

    def dashboard_stats(self) -> dict:
        today = datetime.now().date().isoformat()
        with self.lock:
            row = self.conn.execute(
                """SELECT COUNT(*) n, AVG(speed_mph) avg_speed, MAX(speed_mph) max_speed,
                          SUM(CASE WHEN identity_match>0 THEN 1 ELSE 0 END) repeats
                   FROM events WHERE category='vehicle' AND substr(happened_at,1,10)=?""", (today,)
            ).fetchone()
        return {
            "vehicles": int(row["n"] or 0),
            "avg_speed": float(row["avg_speed"] or 0.0),
            "max_speed": float(row["max_speed"] or 0.0),
            "repeats": int(row["repeats"] or 0),
        }

    def set_person_event_name(self, event_code: str, name: str) -> bool:
        with self.lock, self.conn:
            cur = self.conn.execute(
                "UPDATE events SET display_name=? WHERE event_code=? AND category='person'",
                (name.strip(), event_code.upper()),
            )
            return cur.rowcount > 0

    def get_event(self, event_code: str) -> Optional[sqlite3.Row]:
        with self.lock:
            return self.conn.execute("SELECT * FROM events WHERE event_code=?", (event_code.upper(),)).fetchone()

    def recent_identity_events(self, identity_code: str, limit: int = 5):
        if not identity_code:
            return []
        with self.lock:
            return self.conn.execute(
                "SELECT * FROM events WHERE identity_code=? ORDER BY id DESC LIMIT ?",
                (identity_code, limit),
            ).fetchall()

    def recent_events(self, categories: tuple[str, ...], limit: int = 1000):
        placeholders = ",".join("?" for _ in categories)
        with self.lock:
            return self.conn.execute(
                f"""
                SELECT e.*,
                       COALESCE(NULLIF(e.display_name,''), NULLIF(i.name,''), '') AS resolved_name,
                       COALESCE(i.seen_count, 0) AS seen_count
                FROM events e
                LEFT JOIN identities i ON i.code=e.identity_code
                WHERE e.category IN ({placeholders})
                ORDER BY e.id DESC
                LIMIT ?
                """,
                (*categories, limit),
            ).fetchall()

    def today_counts(self) -> dict[str, int]:
        today = datetime.now().date().isoformat()
        with self.lock:
            rows = self.conn.execute(
                """
                SELECT category, COUNT(*) AS n FROM events
                WHERE substr(happened_at,1,10)=?
                GROUP BY category
                """,
                (today,),
            ).fetchall()
        result = {"vehicle": 0, "person": 0, "animal": 0}
        for row in rows:
            result[row["category"]] = int(row["n"])
        return result

    def search_vehicle_events(self, text: str = "", limit: int = 12):
        q = text.strip().lower()
        clauses = ["e.category='vehicle'"]
        params = []
        if "today" in q:
            clauses.append("substr(e.happened_at,1,10)=?")
            params.append(datetime.now().date().isoformat())
        color_words = ["black","white","red","blue","green","yellow","orange","brown","gray","grey","silver"]
        for c in color_words:
            if c in q:
                clauses.append("lower(e.color) LIKE ?")
                params.append("%" + ("gray" if c == "grey" else c) + "%")
                break
        types = {"car":"Car","truck":"Truck","bus":"Bus","motorcycle":"Motorcycle","bike":"Motorcycle"}
        for word,val in types.items():
            if word in q:
                clauses.append("e.object_type=?")
                params.append(val)
                break
        if "driveway" in q:
            clauses.append("e.zone='driveway'")
        if "road" in q:
            clauses.append("e.zone='road'")
        sql = f"""SELECT e.*, COALESCE(NULLIF(i.name,''),'') AS resolved_name
                  FROM events e LEFT JOIN identities i ON i.code=e.identity_code
                  WHERE {' AND '.join(clauses)} ORDER BY e.id DESC LIMIT ?"""
        params.append(limit)
        with self.lock:
            return self.conn.execute(sql, tuple(params)).fetchall()

    def direction_stats_today(self) -> dict[str,int]:
        today=datetime.now().date().isoformat()
        with self.lock:
            rows=self.conn.execute(
                "SELECT direction,COUNT(*) n FROM events WHERE category='vehicle' AND substr(happened_at,1,10)=? GROUP BY direction",
                (today,),
            ).fetchall()
        return {str(r['direction']): int(r['n']) for r in rows}

    def clear_events(self, categories: tuple[str, ...]):
        placeholders = ",".join("?" for _ in categories)
        with self.lock, self.conn:
            self.conn.execute(f"DELETE FROM events WHERE category IN ({placeholders})", categories)

    def export_csv(self, path: str):
        with self.lock:
            rows = self.conn.execute(
                """
                SELECT e.event_code, e.category, e.happened_at, e.object_type, e.color,
                       e.direction, e.confidence, e.speed_mph, e.identity_code,
                       e.identity_match, e.speed_source, e.speed_confidence, e.bookmarked, e.note, e.zone,
                       COALESCE(NULLIF(e.display_name,''), NULLIF(i.name,''), '') AS name,
                       e.snapshot_path
                FROM events e
                LEFT JOIN identities i ON i.code=e.identity_code
                ORDER BY e.id DESC
                """
            ).fetchall()
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Event ID", "Category", "Time", "Type", "Color", "Direction",
                "AI confidence", "Speed MPH", "Speed source", "Speed confidence", "Persistent ID", "Repeat match",
                "Name/Label", "Bookmarked", "Note", "Zone", "Snapshot",
            ])
            for r in rows:
                writer.writerow([
                    r["event_code"], r["category"], r["happened_at"], r["object_type"],
                    r["color"], r["direction"], r["confidence"], r["speed_mph"], r["speed_source"], r["speed_confidence"],
                    r["identity_code"], r["identity_match"], r["name"], r["bookmarked"], r["note"], r["zone"], r["snapshot_path"],
                ])


# -----------------------------------------------------------------------------
# Vision engine
# -----------------------------------------------------------------------------
class StreetWatchEngine:
    def __init__(self, frame_queue: queue.Queue, event_queue: queue.Queue):
        self.frame_queue = frame_queue
        self.event_queue = event_queue
        self.db = StreetWatchDatabase(DB_PATH)
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.cap: Optional[cv2.VideoCapture] = None
        self.model: Optional[object] = None
        self.infer_imgsz = 640
        self.model_loading = False
        self.model_error = ""

        self.source: Union[int, str] = 0
        self.confidence = 0.28
        self.tripwire_orientation = "vertical"
        self.tripwire_position = 50
        self.speed_gate_a = 35
        self.speed_gate_b = 65
        self.speed_distance_ft = 30.0
        self.reid_threshold = 0.82
        self.monitor_vehicles = True
        self.monitor_people = True
        self.monitor_animals = True

        # Adaptive learning is local and transparent: it tunes thresholds from your feedback
        # and remembers them for this fixed camera. It never uploads video.
        self.learning = AdaptiveLearningEngine(LEARNING_PATH)
        self.zones = ZoneManager(ZONES_PATH)
        self.vehicle_confirm_frames = 4
        self.person_confirm_frames = 3
        self.auto_speed_enabled = True
        self.auto_speed_scale = 1.0
        self.calibrated_speed_enabled = False
        self.headlight_assist = True
        self.auto_speed = AutoSpeedEstimator()
        self.headlights = HeadlightAssist()
        self.speed_source_by_track: dict[int, str] = {}
        self.speed_conf_by_track: dict[int, float] = {}
        self.last_headlight_alert = 0.0
        self.learning.apply_to_engine(self)

        # Motion gating prevents parked vehicles from creating alerts/events.
        # sensitivity: 1 = strict (needs obvious motion), 10 = sensitive (far-road motion).
        self.ignore_stationary_vehicles = True
        self.motion_sensitivity = 7.0
        self.track_history: dict[int, deque[tuple[float, int, int]]] = {}
        self.moving_tracks: set[int] = set()
        self.ever_moving_tracks: set[int] = set()
        self.stopped_since: dict[int, float] = {}
        self.stop_alerted_tracks: set[int] = set()
        self.moving_confirm_count: dict[int, int] = {}
        self.person_entry_logged: set[int] = set()
        self.recent_identity_alert: dict[str, float] = {}

        # Automatic low-light enhancement for ordinary webcams and IP cameras.
        self.auto_night_assist = True
        self.night_threshold = 78.0
        self.scene_brightness = 255.0
        self.night_active = False
        self.night_strength = 0.0

        self.previous_side: dict[int, int] = {}
        self.logged_tracks: set[int] = set()
        self.last_seen: dict[int, float] = {}
        self.track_seen_count: dict[int, int] = {}
        self.alerted_tracks: set[int] = set()
        self.approach_alerted_tracks: set[int] = set()
        self.previous_center: dict[int, tuple[int, int]] = {}

        self.track_identity: dict[int, tuple[str, str, float]] = {}
        self.track_event_id: dict[int, int] = {}
        self.track_event_code: dict[int, str] = {}

        self.speed_prev_a: dict[int, int] = {}
        self.speed_prev_b: dict[int, int] = {}
        self.speed_first_cross: dict[int, tuple[str, float]] = {}
        self.speed_by_track: dict[int, float] = {}

        self.fps_ema = 0.0
        self.infer_ms_ema = 0.0
        self.last_loop_time = 0.0
        self.frame_count = 0

    def start(self, source: Union[int, str], confidence: float, orientation: str, position: int):
        if self.running:
            return
        self.source = source
        self.confidence = confidence
        self.tripwire_orientation = orientation
        self.tripwire_position = position
        self._reset_session_state()
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _reset_session_state(self):
        for collection in [
            self.previous_side, self.last_seen, self.track_seen_count, self.previous_center,
            self.track_identity, self.track_event_id, self.track_event_code,
            self.speed_prev_a, self.speed_prev_b, self.speed_first_cross,
            self.speed_by_track, self.speed_source_by_track, self.speed_conf_by_track, self.track_history, self.moving_confirm_count, self.recent_identity_alert,
        ]:
            collection.clear()
        self.logged_tracks.clear()
        self.alerted_tracks.clear()
        self.approach_alerted_tracks.clear()
        self.moving_tracks.clear()
        self.ever_moving_tracks.clear()
        self.stopped_since.clear()
        self.stop_alerted_tracks.clear()
        self.person_entry_logged.clear()
        self.auto_speed.reset()
        self.headlights.reset()
        self.last_headlight_alert = 0.0
        self.best_crop_by_track.clear()
        self.camera_fail_count = 0
        self.frame_count = 0
        self.camera_fail_count = 0
        self.best_crop_by_track: dict[int, tuple[float, np.ndarray]] = {}
        self.fps_ema = 0.0
        self.infer_ms_ema = 0.0
        self.last_loop_time = 0.0

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def prepare_ai_async(self):
        """Load the AI in the background so the camera/UI can appear immediately."""
        if self.model is not None or self.model_loading:
            return
        threading.Thread(target=self._load_model, daemon=True, name="StreetWatch-AI-Loader").start()

    def _load_model(self):
        if self.model is not None or self.model_loading:
            return
        self.model_loading = True
        self.model_error = ""
        try:
            self.event_queue.put(("status", "Preparing local AI in background..."))
            # Lazy import is intentional: importing Torch/Ultralytics can take several
            # seconds on Windows. Keeping it out of module startup lets the StreetWatch
            # window and camera selector appear much faster.
            from ultralytics import YOLO
            model_ref = str(MODEL_PATH) if MODEL_PATH.exists() else "yolo26n.pt"
            model = YOLO(model_ref)
            # Warm the model once off the camera thread. The first real vehicle should
            # not have to pay the large one-time inference startup cost.
            try:
                warm = np.zeros((480, 640, 3), dtype=np.uint8)
                model.predict(warm, verbose=False, imgsz=min(int(self.infer_imgsz), 640))
            except Exception:
                pass
            self.model = model
            self.event_queue.put(("status", "AI READY — choose/start camera" if not self.running else "LIVE — AI monitoring"))
        except Exception as exc:
            self.model_error = str(exc)
            self.event_queue.put(("error", f"AI could not load: {exc}"))
        finally:
            self.model_loading = False

    def _open_camera(self):
        if isinstance(self.source, int):
            cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(self.source, cv2.CAP_MSMF)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            cap = cv2.VideoCapture(self.source)
        return cap

    def _active_class_ids(self) -> list[int]:
        ids: list[int] = []
        if self.monitor_vehicles:
            ids.extend(VEHICLE_CLASSES.keys())
        if self.monitor_people:
            ids.extend(PERSON_CLASSES.keys())
        if self.monitor_animals:
            ids.extend(ANIMAL_CLASSES.keys())
        return ids or list(VEHICLE_CLASSES.keys())

    def _run(self):
        try:
            # Open the camera FIRST. AI loading happens in parallel so you get an
            # immediate live picture instead of staring at a Loading message.
            self.cap = self._open_camera()
            if not self.cap.isOpened():
                raise RuntimeError(f"Could not open camera/source: {self.source}")
            if self.model is None:
                self.event_queue.put(("status", "CAMERA LIVE — AI loading in background..."))
                self.prepare_ai_async()
            else:
                self.event_queue.put(("status", "LIVE — AI monitoring"))

            while self.running:
                loop_start = time.perf_counter()
                ok, frame = self.cap.read()
                if not ok:
                    self.camera_fail_count += 1
                    if self.camera_fail_count >= 30:
                        self.event_queue.put(("status", "Camera interrupted — reconnecting..."))
                        try:
                            self.cap.release()
                        except Exception:
                            pass
                        time.sleep(0.6)
                        self.cap = self._open_camera()
                        self.camera_fail_count = 0
                    else:
                        time.sleep(0.03)
                    continue
                self.camera_fail_count = 0

                self.frame_count += 1

                raw_frame = frame
                detection_frame, brightness, night_active, night_strength = self._prepare_frame(raw_frame)
                self.scene_brightness = brightness
                self.night_active = night_active
                self.night_strength = night_strength

                # While the local AI imports/warms up, keep the CAMERA LIVE. This makes
                # startup feel immediate and also proves the selected camera is correct.
                model = self.model
                if model is None:
                    annotated = detection_frame.copy()
                    cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 56), (20, 20, 20), -1)
                    msg = "CAMERA LIVE  •  LOCAL AI LOADING..." if not self.model_error else "CAMERA LIVE  •  AI LOAD ERROR"
                    cv2.putText(annotated, msg, (18, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 220, 255), 2)
                    self.zones.draw(annotated)
                    self._draw_lines(annotated)
                    if self.frame_count % 8 == 0:
                        self.event_queue.put(("stats", {
                            "fps": 0.0, "infer_ms": 0.0,
                            "counts": {"vehicle": 0, "moving_vehicle": 0, "person": 0, "animal": 0},
                            "brightness": self.scene_brightness,
                            "night_active": self.night_active,
                            "night_strength": self.night_strength,
                            "quality": self._camera_quality(detection_frame, self.scene_brightness),
                        }))
                    try:
                        while True:
                            self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                    self.frame_queue.put(annotated)
                    time.sleep(0.015)
                    continue

                infer_start = time.perf_counter()
                results = model.track(
                    detection_frame,
                    persist=True,
                    tracker=str(APP_DIR / "config" / "streetwatch_bytetrack.yaml") if (APP_DIR / "config" / "streetwatch_bytetrack.yaml").exists() else "bytetrack.yaml",
                    conf=self.confidence,
                    classes=self._active_class_ids(),
                    verbose=False,
                    imgsz=int(self.infer_imgsz),
                )
                infer_ms = (time.perf_counter() - infer_start) * 1000.0
                self.infer_ms_ema = infer_ms if self.infer_ms_ema <= 0 else 0.9 * self.infer_ms_ema + 0.1 * infer_ms

                # At night, show the same enhanced image the detector sees.
                annotated = detection_frame.copy()
                detections: list[Detection] = []
                now = time.time()

                if results and results[0].boxes is not None:
                    boxes = results[0].boxes
                    ids = boxes.id
                    if ids is not None:
                        for box, track_id_tensor in zip(boxes, ids):
                            cls_id = int(box.cls.item())
                            if cls_id not in TARGET_CLASSES:
                                continue
                            track_id = int(track_id_tensor.item())
                            conf = float(box.conf.item())
                            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                            x1 = max(0, min(x1, raw_frame.shape[1] - 1))
                            y1 = max(0, min(y1, raw_frame.shape[0] - 1))
                            x2 = max(0, min(x2, raw_frame.shape[1]))
                            y2 = max(0, min(y2, raw_frame.shape[0]))
                            if x2 <= x1 or y2 <= y1:
                                continue

                            category = CATEGORY_BY_CLASS[cls_id]
                            label = TARGET_CLASSES[cls_id]
                            # Keep color estimation on the untouched camera image.
                            raw_crop = raw_frame[y1:y2, x1:x2]
                            color = estimate_color(raw_crop) if category == "vehicle" else ""
                            if category == "vehicle" and night_active and crop_is_too_dark_for_color(raw_crop):
                                color = "Unknown"
                            center = ((x1 + x2) // 2, (y1 + y2) // 2)
                            det = Detection(
                                track_id=track_id,
                                class_id=cls_id,
                                category=category,
                                label=label,
                                confidence=conf,
                                xyxy=(x1, y1, x2, y2),
                                center=center,
                                color=color,
                            )
                            det.zone = self.zones.zone_at(center, raw_frame.shape)
                            if det.zone == "ignore":
                                continue

                            self._update_best_crop(detection_frame, det)
                            self.last_seen[track_id] = now
                            self.track_seen_count[track_id] = self.track_seen_count.get(track_id, 0) + 1

                            # Prove motion over multiple frames before treating a vehicle as traffic.
                            # A parked car may have a slightly wobbly detector box, so vehicles require
                            # several consecutive motion-positive frames before alerts/events are allowed.
                            moving_raw, motion_score = self._update_motion(track_id, center, raw_frame.shape, now)
                            det.motion_score = motion_score

                            if category == "vehicle":
                                count = self.moving_confirm_count.get(track_id, 0)
                                if moving_raw and motion_score >= 0.95:
                                    count = min(12, count + 1)
                                else:
                                    count = max(0, count - 2)
                                self.moving_confirm_count[track_id] = count
                                det.is_moving = moving_raw and count >= int(self.vehicle_confirm_frames)
                                if det.is_moving:
                                    self.ever_moving_tracks.add(track_id)
                                    self.stopped_since.pop(track_id, None)
                                elif track_id in self.ever_moving_tracks:
                                    self.stopped_since.setdefault(track_id, now)
                                if det.is_moving and self.track_seen_count[track_id] >= 4:
                                    self._ensure_identity(detection_frame, det)
                            else:
                                det.is_moving = moving_raw

                            if category == "animal" and self.track_seen_count[track_id] >= 2:
                                self._ensure_identity(detection_frame, det)
                            elif category == "person":
                                # Current-track-only label until a confirmed entry gets a P-event ID.
                                det.identity_code = f"LIVE-P{track_id}"

                            if track_id in self.track_identity:
                                code, name, score = self.track_identity[track_id]
                                det.identity_code = code
                                det.identity_name = name
                                det.identity_match = score
                            if track_id in self.speed_by_track:
                                det.speed_mph = self.speed_by_track[track_id]
                                det.speed_source = self.speed_source_by_track.get(track_id, "")
                                det.speed_confidence = self.speed_conf_by_track.get(track_id, 0.0)

                            detections.append(det)

                self.zones.draw(annotated)
                self._draw_lines(annotated)
                for det in detections:
                    self._process_auto_speed(det, now)
                    self._process_speed(det, raw_frame.shape)
                    if det.track_id in self.speed_by_track:
                        det.speed_mph = self.speed_by_track[det.track_id]
                        det.speed_source = self.speed_source_by_track.get(det.track_id, "")
                        det.speed_confidence = self.speed_conf_by_track.get(det.track_id, 0.0)

                    # People are logged on confirmed entry into view; they do NOT have to cross
                    # the vehicle traffic line to receive a P-event ID and snapshot.
                    if det.category == "person":
                        self._process_person_entry(detection_frame, det)

                    self._maybe_alert(detection_frame, det)
                    self._maybe_stopped(det, now)
                    if det.category != "person":
                        self._maybe_approach(det, raw_frame.shape)
                    # Night events keep the enhanced snapshot so the saved evidence is visible.
                    self._process_crossing(detection_frame, det)
                    self._draw_trail(annotated, det)
                    self._draw_detection(annotated, det)
                    self.previous_center[det.track_id] = det.center

                # Night-only secondary detector: if YOLO cannot see the body, look for a
                # genuinely moving pair of bright lights. It is an assist, not a fake class ID.
                if self.night_active and self.headlight_assist:
                    moving_vehicle_present = any(d.category == "vehicle" and d.is_moving for d in detections)
                    box, hconf = self.headlights.detect(detection_frame)
                    if box is not None and hconf >= 0.72 and not moving_vehicle_present:
                        x1,y1,x2,y2 = box
                        cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,200,255), 2)
                        cv2.putText(annotated, f"PROBABLE VEHICLE / HEADLIGHTS {hconf:.0%}",
                                    (x1, max(24,y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0,200,255), 2)
                        if now - self.last_headlight_alert > 7.0:
                            self.last_headlight_alert = now
                            self.event_queue.put(("alert", {
                                "category":"vehicle",
                                "text":f"NIGHT VEHICLE APPROACHING — moving headlights {hconf:.0%}",
                                "speech":"Possible vehicle approaching at night.",
                                "phase":"night_hint",
                            }))

                self._forget_stale(now)

                # Live performance / count telemetry.
                dt = max(time.perf_counter() - loop_start, 1e-6)
                instant_fps = 1.0 / dt
                self.fps_ema = instant_fps if self.fps_ema <= 0 else 0.90 * self.fps_ema + 0.10 * instant_fps
                if self.frame_count % 8 == 0:
                    counts = {
                        "vehicle": sum(d.category == "vehicle" for d in detections),
                        "moving_vehicle": sum(d.category == "vehicle" and d.is_moving for d in detections),
                        "person": sum(d.category == "person" for d in detections),
                        "animal": sum(d.category == "animal" for d in detections),
                    }
                    self.event_queue.put(("stats", {
                        "fps": self.fps_ema,
                        "infer_ms": self.infer_ms_ema,
                        "counts": counts,
                        "brightness": self.scene_brightness,
                        "night_active": self.night_active,
                        "night_strength": self.night_strength,
                        "quality": self._camera_quality(detection_frame, self.scene_brightness),
                    }))

                try:
                    while True:
                        self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
                self.frame_queue.put(annotated)

        except Exception as exc:
            self.event_queue.put(("error", str(exc)))
        finally:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self.running = False
            self.event_queue.put(("status", "Stopped"))

    def _update_best_crop(self, frame: np.ndarray, det: Detection):
        crop = crop_with_pad(frame, det.xyxy, 0.08)
        if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
            return
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        area_bonus = min(220.0, (crop.shape[0] * crop.shape[1]) / 4500.0)
        exposure = float(np.mean(gray))
        exposure_bonus = 40.0 if 35 <= exposure <= 215 else 0.0
        score = sharp + area_bonus + exposure_bonus
        old = self.best_crop_by_track.get(det.track_id)
        if old is None or score > old[0]:
            self.best_crop_by_track[det.track_id] = (score, crop.copy())

    @staticmethod
    def _camera_quality(frame: np.ndarray, brightness: float) -> str:
        small = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if brightness < 18:
            return "TOO DARK"
        if blur < 18:
            return "POOR / BLURRY"
        if brightness < 35 or blur < 35:
            return "FAIR"
        if brightness > 235:
            return "POOR / OVEREXPOSED"
        return "GOOD" if blur < 85 else "EXCELLENT"

    def _update_motion(self, track_id: int, center: tuple[int, int], shape, now: float) -> tuple[bool, float]:
        """Return whether a tracked object is genuinely moving, resisting detector jitter.

        Motion is measured over a short history and normalized to the camera frame size,
        so a fixed parked vehicle does not trigger just because its box wiggles by a few pixels.
        """
        h, w = shape[:2]
        diag = max(math.hypot(w, h), 1.0)
        hist = self.track_history.setdefault(track_id, deque(maxlen=24))
        hist.append((now, int(center[0]), int(center[1])))
        while hist and now - hist[0][0] > 1.25:
            hist.popleft()

        if len(hist) < 4 or (hist[-1][0] - hist[0][0]) < 0.22:
            return (track_id in self.moving_tracks), 0.0

        # Net displacement rejects random bounding-box jitter better than path length alone.
        _, x0, y0 = hist[0]
        _, x1, y1 = hist[-1]
        net_norm = math.hypot(x1 - x0, y1 - y0) / diag

        # Median step motion helps catch a far-away vehicle that moves steadily only a few px/frame.
        steps = []
        for a, b in zip(hist, list(hist)[1:]):
            steps.append(math.hypot(b[1] - a[1], b[2] - a[2]) / diag)
        median_step = float(np.median(steps)) if steps else 0.0

        # Higher UI sensitivity lowers the motion threshold. Default 7 is tuned for street cameras.
        sens = max(1.0, min(10.0, float(self.motion_sensitivity)))
        threshold = 0.0165 - (sens - 1.0) * (0.0125 / 9.0)  # ~1.65% down to ~0.40% of frame diagonal
        step_threshold = threshold / 7.0
        motion_score = max(net_norm / max(threshold, 1e-6), median_step / max(step_threshold, 1e-6))

        if net_norm >= threshold and median_step >= step_threshold * 0.45:
            self.moving_tracks.add(track_id)
        elif track_id in self.moving_tracks:
            # Hysteresis: once moving, keep that state through short pauses/occlusion jitter.
            if net_norm < threshold * 0.28 and median_step < step_threshold * 0.20:
                self.moving_tracks.discard(track_id)

        return (track_id in self.moving_tracks), motion_score

    def _prepare_frame(self, frame: np.ndarray) -> tuple[np.ndarray, float, bool, float]:
        """Measure ambient light and enhance dark frames before inference."""
        sample = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
        # Median resists headlights/street lamps dominating the scene estimate.
        brightness = float(np.median(gray))

        if not self.auto_night_assist or brightness >= float(self.night_threshold):
            return frame, brightness, False, 0.0

        threshold = max(float(self.night_threshold), 1.0)
        strength = max(0.0, min(1.0, (threshold - brightness) / threshold))

        # Lift shadows with gamma, then improve local contrast with CLAHE.
        gamma = max(0.48, 1.0 - 0.52 * strength)
        lut = np.array([((i / 255.0) ** gamma) * 255.0 for i in range(256)], dtype=np.uint8)
        lifted = cv2.LUT(frame, lut)
        lab = cv2.cvtColor(lifted, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.7 + 1.3 * strength, tileGridSize=(8, 8))
        l_chan = clahe.apply(l_chan)
        enhanced = cv2.cvtColor(cv2.merge((l_chan, a_chan, b_chan)), cv2.COLOR_LAB2BGR)

        # Blend progressively so dusk transitions do not suddenly change appearance.
        alpha = 0.35 + 0.55 * strength
        output = cv2.addWeighted(enhanced, alpha, frame, 1.0 - alpha, 0.0)
        return output, brightness, True, strength

    def _ensure_identity(self, frame: np.ndarray, det: Detection):
        if det.track_id in self.track_identity:
            return
        crop = crop_with_pad(frame, det.xyxy, 0.06)
        fp = make_fingerprint(crop)
        if fp is None:
            return
        row, score = self.db.find_identity(
            det.category, det.label, det.color, fp, float(self.reid_threshold)
        )
        if row is not None:
            self.db.touch_identity(row["code"])
            self.track_identity[det.track_id] = (
                row["code"], row["name"] or "", float(score)
            )
            return

        # New persistent vehicle/animal identity.
        tmp_path = ""
        if crop.size:
            tmp_path = str(IDENTITY_DIR / f"pending_{int(time.time()*1000)}.jpg")
            cv2.imwrite(tmp_path, crop)
        row = self.db.create_identity(det.category, det.label, det.color, fp, tmp_path)
        code = row["code"]
        if tmp_path:
            final_path = IDENTITY_DIR / f"{code}.jpg"
            try:
                Path(tmp_path).replace(final_path)
                with self.db.lock, self.db.conn:
                    self.db.conn.execute(
                        "UPDATE identities SET snapshot_path=? WHERE code=?",
                        (str(final_path), code),
                    )
            except Exception:
                final_path = Path(tmp_path)
        self.track_identity[det.track_id] = (code, "", 0.0)

    def _process_person_entry(self, frame: np.ndarray, det: Detection):
        """Create one person visit/event as soon as a person is stably in view.

        This intentionally does not create a persistent biometric identity. The P-code
        identifies this recorded visit/photo only.
        """
        if det.track_id in self.person_entry_logged:
            code = self.track_event_code.get(det.track_id)
            if code:
                det.identity_code = code
            return
        # Three tracked frames filters one-frame false positives while remaining immediate.
        if self.track_seen_count.get(det.track_id, 0) < int(self.person_confirm_frames):
            return

        timestamp = datetime.now()
        filename = f"{timestamp:%Y%m%d_%H%M%S}_person_{det.track_id}.jpg"
        snapshot_path = SNAPSHOT_DIR / filename
        crop = crop_with_pad(frame, det.xyxy, 0.12)
        if crop.size:
            cv2.imwrite(str(snapshot_path), crop)
        else:
            snapshot_path = Path("")

        row = self.db.add_event(
            category="person",
            object_type=det.label,
            color="",
            direction="Entered view",
            confidence=det.confidence,
            snapshot_path=str(snapshot_path),
            track_id=det.track_id,
            zone=det.zone,
        )
        self.person_entry_logged.add(det.track_id)
        self.track_event_id[det.track_id] = int(row["id"])
        self.track_event_code[det.track_id] = row["event_code"]
        det.identity_code = row["event_code"]
        self.event_queue.put(("event", dict(row)))

    def _maybe_alert(self, frame: np.ndarray, det: Detection):
        minimum_seen = 3 if det.category == "person" else 2
        if det.track_id in self.alerted_tracks or self.track_seen_count.get(det.track_id, 0) < minimum_seen:
            return
        if det.category == "person" and det.track_id not in self.person_entry_logged:
            return
        if det.category == "vehicle" and self.ignore_stationary_vehicles and not det.is_moving:
            return

        # If the tracker briefly changes IDs, a persistent vehicle identity prevents a second beep.
        identity_key = det.identity_code if det.category == "vehicle" else ""
        now = time.time()
        if identity_key and now - self.recent_identity_alert.get(identity_key, 0.0) < 8.0:
            self.alerted_tracks.add(det.track_id)
            return

        self.alerted_tracks.add(det.track_id)
        if identity_key:
            self.recent_identity_alert[identity_key] = now
        if det.category == "vehicle":
            title = "VEHICLE DETECTED"
            who = det.identity_name or det.identity_code or det.label
            zone_text = f" • {det.zone.upper()}" if det.zone else ""
            text = f"{title} — {who} • {det.color} {det.label}{zone_text}"
            speech = f"Vehicle detected. {det.color} {det.label}."
            if det.identity_name:
                speech = f"{det.identity_name} detected."
        elif det.category == "person":
            code = self.track_event_code.get(det.track_id, det.identity_code)
            zone_text = f" • {det.zone.upper()}" if det.zone else ""
            text = f"PERSON DETECTED — {code} • person in view{zone_text}"
            speech = "Person detected."
        else:
            who = det.identity_name or det.identity_code or det.label
            zone_text = f" • {det.zone.upper()}" if det.zone else ""
            text = f"ANIMAL DETECTED — {who} • {det.label}{zone_text}"
            speech = f"{det.label} detected."
            if det.identity_name:
                speech = f"{det.identity_name} detected."
        self.event_queue.put(("alert", {
            "category": det.category,
            "text": text,
            "speech": speech,
            "phase": "detected",
        }))

    def _maybe_stopped(self, det: Detection, now: float):
        if det.category != "vehicle" or det.track_id in self.stop_alerted_tracks:
            return
        since = self.stopped_since.get(det.track_id)
        if since is None or now - since < 8.0:
            return
        self.stop_alerted_tracks.add(det.track_id)
        who = det.identity_name or det.identity_code or det.label
        self.event_queue.put(("alert", {
            "category":"vehicle",
            "text":f"VEHICLE STOPPED — {who} has remained nearly stationary",
            "speech":"A vehicle has stopped in view.",
            "phase":"stopped",
        }))

    def _maybe_approach(self, det: Detection, shape):
        if det.track_id in self.approach_alerted_tracks:
            return
        if det.category == "vehicle" and self.ignore_stationary_vehicles and not det.is_moving:
            return
        h, w = shape[:2]
        cx, cy = det.center
        if self.tripwire_orientation == "vertical":
            line = int(w * self.tripwire_position / 100)
            distance = abs(cx - line) / max(w, 1)
            prev = self.previous_center.get(det.track_id)
            moving_toward = prev is not None and abs(cx - line) < abs(prev[0] - line)
        else:
            line = int(h * self.tripwire_position / 100)
            distance = abs(cy - line) / max(h, 1)
            prev = self.previous_center.get(det.track_id)
            moving_toward = prev is not None and abs(cy - line) < abs(prev[1] - line)
        if distance < 0.12 and moving_toward:
            self.approach_alerted_tracks.add(det.track_id)
            label = det.identity_name or det.identity_code or det.label
            self.event_queue.put(("alert", {
                "category": det.category,
                "text": f"APPROACHING LOG LINE — {label}",
                "speech": "Vehicle approaching." if det.category == "vehicle" else "",
                "phase": "approaching",
            }))

    def _process_auto_speed(self, det: Detection, now: float):
        """Estimate speed with no user calibration. Marked AUTO because one camera has no true scale."""
        if det.category != "vehicle" or not det.is_moving or not self.auto_speed_enabled:
            return
        result = self.auto_speed.update(
            det.track_id, det.label, det.center, det.xyxy, now,
            scale=float(self.auto_speed_scale), night=bool(self.night_active),
        )
        if result is None:
            return
        # Keep a calibrated result if we already have one; otherwise maintain the smoothed auto estimate.
        if self.speed_source_by_track.get(det.track_id) == "CALIBRATED":
            return
        self.speed_by_track[det.track_id] = result.mph
        self.speed_source_by_track[det.track_id] = "AUTO"
        self.speed_conf_by_track[det.track_id] = result.confidence
        event_id = self.track_event_id.get(det.track_id)
        if event_id:
            self.db.update_event_speed(event_id, result.mph, "AUTO", result.confidence)
            self.event_queue.put(("speed_update", {
                "event_id": event_id, "speed_mph": result.mph,
                "speed_source": "AUTO", "speed_confidence": result.confidence,
            }))

    def _process_speed(self, det: Detection, shape):
        if not self.calibrated_speed_enabled:
            return
        if det.category != "vehicle":
            return
        if self.ignore_stationary_vehicles and not det.is_moving:
            return
        h, w = shape[:2]
        coord = det.center[0] if self.tripwire_orientation == "vertical" else det.center[1]
        length = w if self.tripwire_orientation == "vertical" else h
        a = int(length * self.speed_gate_a / 100)
        b = int(length * self.speed_gate_b / 100)
        side_a = -1 if coord < a else 1
        side_b = -1 if coord < b else 1

        prev_a = self.speed_prev_a.get(det.track_id)
        prev_b = self.speed_prev_b.get(det.track_id)
        self.speed_prev_a[det.track_id] = side_a
        self.speed_prev_b[det.track_id] = side_b

        crossed_gate = None
        if prev_a is not None and prev_a != side_a:
            crossed_gate = "A"
        elif prev_b is not None and prev_b != side_b:
            crossed_gate = "B"
        if crossed_gate is None:
            return

        now = time.perf_counter()
        first = self.speed_first_cross.get(det.track_id)
        if first is None:
            self.speed_first_cross[det.track_id] = (crossed_gate, now)
            return
        first_gate, first_time = first
        if first_gate == crossed_gate:
            return
        elapsed = now - first_time
        if not (0.08 <= elapsed <= 12.0) or self.speed_distance_ft <= 0:
            return
        feet_per_second = float(self.speed_distance_ft) / elapsed
        mph = feet_per_second * 0.6818181818
        # Reject obviously bad calibration/tracking jumps.
        if 0.5 <= mph <= 160:
            auto_before = self.speed_by_track.get(det.track_id) if self.speed_source_by_track.get(det.track_id) == "AUTO" else None
            if auto_before:
                self.learning.learn_speed_scale(float(auto_before), float(mph))
                self.auto_speed_scale = float(self.learning.profile.get("auto_speed_scale", self.auto_speed_scale))
            self.speed_by_track[det.track_id] = mph
            self.speed_source_by_track[det.track_id] = "CALIBRATED"
            self.speed_conf_by_track[det.track_id] = 0.95
            event_id = self.track_event_id.get(det.track_id)
            if event_id:
                self.db.update_event_speed(event_id, mph, "CALIBRATED", 0.95)
                self.event_queue.put(("speed_update", {
                    "event_id": event_id,
                    "speed_mph": mph,
                    "speed_source": "CALIBRATED",
                    "speed_confidence": 0.95,
                }))

    def _process_crossing(self, frame: np.ndarray, det: Detection):
        # Person visits are recorded on entry into view, independent of the vehicle tripwire.
        if det.category == "person":
            return
        if det.category == "vehicle" and self.ignore_stationary_vehicles and not det.is_moving:
            return
        h, w = frame.shape[:2]
        cx, cy = det.center
        if self.tripwire_orientation == "vertical":
            line = int(w * self.tripwire_position / 100)
            side = -1 if cx < line else 1
            neg_dir, pos_dir = "Left → Right", "Right → Left"
        else:
            line = int(h * self.tripwire_position / 100)
            side = -1 if cy < line else 1
            neg_dir, pos_dir = "Top → Bottom", "Bottom → Top"

        prev = self.previous_side.get(det.track_id)
        self.previous_side[det.track_id] = side
        if prev is None or prev == side or det.track_id in self.logged_tracks:
            return

        direction = neg_dir if prev == -1 and side == 1 else pos_dir
        timestamp = datetime.now()
        filename = f"{timestamp:%Y%m%d_%H%M%S}_{det.category}_{det.track_id}.jpg"
        snapshot_path = SNAPSHOT_DIR / filename
        best = self.best_crop_by_track.get(det.track_id)
        crop = best[1] if best is not None else crop_with_pad(frame, det.xyxy, 0.10)
        if crop.size:
            cv2.imwrite(str(snapshot_path), crop)
        else:
            snapshot_path = Path("")

        identity_code = ""
        identity_name = ""
        identity_match = 0.0
        if det.category in {"vehicle", "animal"}:
            self._ensure_identity(frame, det)
            identity_code, identity_name, identity_match = self.track_identity.get(
                det.track_id, ("", "", 0.0)
            )

        speed_mph = self.speed_by_track.get(det.track_id) if det.category == "vehicle" else None
        row = self.db.add_event(
            category=det.category,
            object_type=det.label,
            color=det.color,
            direction=direction,
            confidence=det.confidence,
            snapshot_path=str(snapshot_path),
            track_id=det.track_id,
            identity_code=identity_code,
            identity_match=identity_match,
            speed_mph=speed_mph,
            speed_source=self.speed_source_by_track.get(det.track_id, ""),
            speed_confidence=self.speed_conf_by_track.get(det.track_id, 0.0),
            zone=det.zone,
        )
        self.track_event_id[det.track_id] = int(row["id"])
        self.track_event_code[det.track_id] = row["event_code"]
        self.logged_tracks.add(det.track_id)

        self.event_queue.put(("event", dict(row)))
        if det.category == "vehicle" and identity_code:
            passes = self.db.recent_identity_events(identity_code, 3)
            if len(passes) >= 2:
                try:
                    prev = passes[1]
                    age = (datetime.fromisoformat(row["happened_at"]) - datetime.fromisoformat(prev["happened_at"])).total_seconds()
                    if age <= 600:
                        same_dir = prev["direction"] == direction
                        phrase = "repeat pass" if same_dir else "possible turn-around / return"
                        self.event_queue.put(("alert", {
                            "category":"vehicle",
                            "text":f"REPEAT VEHICLE — {identity_name or identity_code} • {phrase}",
                            "speech":f"Repeat vehicle detected. {phrase}.",
                            "phase":"repeat",
                        }))
                except Exception:
                    pass
        if det.category == "vehicle":
            who = identity_name or identity_code or det.label
            speed_text = f" at {speed_mph:.0f} miles per hour" if speed_mph else ""
            speech = f"{who} passed{speed_text}."
            text = f"PASSED — {who} • {det.color} {det.label}"
            if speed_mph:
                text += f" • {speed_mph:.1f} MPH"
        elif det.category == "person":
            speech = "Person passed the monitoring line."
            text = f"PERSON PASSED — event {row['event_code']}"
        else:
            who = identity_name or identity_code or det.label
            speech = f"{who} passed."
            text = f"ANIMAL PASSED — {who} • {det.label}"
        self.event_queue.put(("alert", {
            "category": det.category,
            "text": text,
            "speech": speech,
            "phase": "passed",
        }))

    def _forget_stale(self, now: float):
        stale = [tid for tid, ts in self.last_seen.items() if now - ts > 20]
        for tid in stale:
            self.last_seen.pop(tid, None)
            self.previous_side.pop(tid, None)
            self.track_seen_count.pop(tid, None)
            self.previous_center.pop(tid, None)
            self.track_identity.pop(tid, None)
            self.track_event_id.pop(tid, None)
            self.track_event_code.pop(tid, None)
            self.speed_prev_a.pop(tid, None)
            self.speed_prev_b.pop(tid, None)
            self.speed_first_cross.pop(tid, None)
            self.speed_by_track.pop(tid, None)
            self.speed_source_by_track.pop(tid, None)
            self.speed_conf_by_track.pop(tid, None)
            self.auto_speed.forget(tid)
            self.best_crop_by_track.pop(tid, None)
            self.track_history.pop(tid, None)
            self.moving_confirm_count.pop(tid, None)
            self.moving_tracks.discard(tid)
            self.ever_moving_tracks.discard(tid)
            self.stopped_since.pop(tid, None)
            self.stop_alerted_tracks.discard(tid)
            self.person_entry_logged.discard(tid)
            self.logged_tracks.discard(tid)
            self.alerted_tracks.discard(tid)
            self.approach_alerted_tracks.discard(tid)

    def _draw_lines(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        vertical = self.tripwire_orientation == "vertical"
        length = w if vertical else h

        def draw_at(position_pct: int, color: tuple[int, int, int], text: str, thickness: int = 2):
            p = int(length * position_pct / 100)
            if vertical:
                cv2.line(frame, (p, 0), (p, h), color, thickness)
                cv2.putText(frame, text, (min(p + 6, w - 170), 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2)
            else:
                cv2.line(frame, (0, p), (w, p), color, thickness)
                cv2.putText(frame, text, (10, max(24, p - 7)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2)

        draw_at(self.speed_gate_a, (255, 130, 0), "SPEED A")
        draw_at(self.speed_gate_b, (255, 0, 180), "SPEED B")
        draw_at(self.tripwire_position, (0, 255, 255), "LOG LINE", 3)

    def _draw_trail(self, frame: np.ndarray, det: Detection):
        hist = self.track_history.get(det.track_id)
        if not hist or len(hist) < 2:
            return
        pts=[(int(x),int(y)) for _,x,y in hist]
        color = CATEGORY_COLORS.get(det.category, (0,220,0))
        for a,b in zip(pts,pts[1:]):
            cv2.line(frame,a,b,color,2)

    @staticmethod
    def _draw_detection(frame: np.ndarray, det: Detection):
        x1, y1, x2, y2 = det.xyxy
        color = CATEGORY_COLORS.get(det.category, (0, 220, 0))
        if det.category == "vehicle" and not det.is_moving:
            color = (145, 145, 145)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        if det.category == "vehicle":
            if not det.is_moving:
                label = f"PARKED / STATIONARY — ignored | {det.color} {det.label}"
            else:
                ident = det.identity_name or det.identity_code or f"Track {det.track_id}"
                label = f"{ident} | MOVING | {det.color} {det.label} | {det.confidence:.0%}"
            if det.is_moving and det.speed_mph is not None:
                prefix = "~" if det.speed_source == "AUTO" else ""
                conf_txt = f" {det.speed_confidence:.0%}" if det.speed_source == "AUTO" else ""
                label += f" | {prefix}{det.speed_mph:.1f} MPH {det.speed_source}{conf_txt}"
            if det.is_moving and det.identity_match and det.identity_match < 0.999:
                label += f" | repeat {det.identity_match:.0%}"
            if det.zone:
                label += f" | {det.zone.upper()}"
        elif det.category == "animal":
            ident = det.identity_name or det.identity_code or f"Track {det.track_id}"
            label = f"{ident} | {det.label} | {det.confidence:.0%}"
            if det.identity_match and det.identity_match < 0.999:
                label += f" | repeat {det.identity_match:.0%}"
        else:
            ident = det.identity_code or f"LIVE-P{det.track_id}"
            label = f"Person | {ident} | {det.confidence:.0%}"

        # Add a dark backing strip for readability.
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
        y_text = max(22, y1 - 8)
        cv2.rectangle(frame, (x1, y_text - th - 7), (min(frame.shape[1] - 1, x1 + tw + 8), y_text + 3), (25, 25, 25), -1)
        cv2.putText(frame, label, (x1 + 4, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2)
        cv2.circle(frame, det.center, 4, (255, 255, 255), -1)


# -----------------------------------------------------------------------------
# Color estimation (vehicles)
# -----------------------------------------------------------------------------
def crop_is_too_dark_for_color(crop: np.ndarray) -> bool:
    """Return True when there is not enough light to name paint color responsibly."""
    if crop is None or crop.size == 0:
        return True
    h, w = crop.shape[:2]
    if h < 4 or w < 4:
        return True
    sample = cv2.resize(crop, (min(96, w), min(64, h)), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2]
    return float(np.percentile(value, 65)) < 58.0


def estimate_color(crop: np.ndarray) -> str:
    if crop is None or crop.size == 0:
        return "Unknown"
    h, w = crop.shape[:2]
    x1, x2 = int(w * 0.2), int(w * 0.8)
    y1, y2 = int(h * 0.25), int(h * 0.75)
    roi = crop[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else crop
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape(-1, 3)
    mask = (pixels[:, 2] > 35) & (pixels[:, 2] < 245)
    if mask.any():
        pixels = pixels[mask]
    if len(pixels) == 0:
        return "Unknown"
    H = float(np.median(pixels[:, 0]))
    S = float(np.median(pixels[:, 1]))
    V = float(np.median(pixels[:, 2]))

    if V < 65:
        return "Black"
    if S < 28 and V > 190:
        return "White"
    if S < 38:
        return "Gray/Silver"
    if H < 8 or H >= 172:
        return "Red"
    if H < 18:
        return "Orange/Brown"
    if H < 34:
        return "Yellow/Gold"
    if H < 85:
        return "Green"
    if H < 132:
        return "Blue"
    if H < 160:
        return "Purple"
    return "Red"


# -----------------------------------------------------------------------------
# Desktop UI
# -----------------------------------------------------------------------------
class StreetWatchApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("StreetWatch Pro 3.1 — Fast Start + Camera Picker")
        root.geometry("1560x930")
        root.minsize(1180, 760)

        self.frame_queue: queue.Queue = queue.Queue(maxsize=1)
        self.event_queue: queue.Queue = queue.Queue()
        self.engine = StreetWatchEngine(self.frame_queue, self.event_queue)
        self.speech = SpeechWorker()
        self.photo: Optional[ImageTk.PhotoImage] = None

        self.snapshot_by_iid: dict[str, str] = {}
        self.dbid_to_vehicle_iid: dict[int, str] = {}

        learned = self.engine.learning.profile
        self.source_var = tk.StringVar(value=str(learned.get("last_source", "0")))
        self.camera_scan_running = False
        self.conf_var = tk.DoubleVar(value=float(learned.get("confidence", 0.28)))
        self.orientation_var = tk.StringVar(value="vertical")
        self.position_var = tk.IntVar(value=50)
        self.speed_a_var = tk.IntVar(value=35)
        self.speed_b_var = tk.IntVar(value=65)
        self.speed_distance_var = tk.DoubleVar(value=30.0)
        self.reid_var = tk.DoubleVar(value=0.82)
        self.sound_var = tk.BooleanVar(value=True)
        self.voice_var = tk.BooleanVar(value=True)
        self.monitor_vehicle_var = tk.BooleanVar(value=True)
        self.monitor_people_var = tk.BooleanVar(value=True)
        self.monitor_animal_var = tk.BooleanVar(value=True)
        self.night_assist_var = tk.BooleanVar(value=True)
        self.night_threshold_var = tk.IntVar(value=int(learned.get("night_threshold", 78)))
        self.ignore_parked_var = tk.BooleanVar(value=True)
        self.motion_sensitivity_var = tk.DoubleVar(value=float(learned.get("motion_sensitivity", 7.0)))
        self.auto_speed_var = tk.BooleanVar(value=bool(learned.get("auto_speed_enabled", True)))
        self.headlight_assist_var = tk.BooleanVar(value=bool(learned.get("headlight_assist", True)))
        self.calibrated_speed_var = tk.BooleanVar(value=False)
        self.auto_start_var = tk.BooleanVar(value=bool(learned.get("auto_start_monitoring", True)))
        self.performance_var = tk.StringVar(value=str(learned.get("performance_mode", "Balanced")))

        self.status_var = tk.StringVar(value="Ready")
        self.live_var = tk.StringVar(value="Live: 0 vehicles • 0 people • 0 animals")
        self.perf_var = tk.StringVar(value="AI: -- FPS • -- ms")
        self.light_var = tk.StringVar(value="Lighting: DAY / normal")
        self.today_var = tk.StringVar(value="Today: --")
        self.last_event_var = tk.StringVar(value="Last event: none")
        self.learning_var = tk.StringVar(value=self.engine.learning.summary())
        self.dashboard_var = tk.StringVar(value="Adaptive learning ready")
        self.last_event_code = ""

        self.alert_default_bg = "#111827"
        self.alert_flash_token = 0
        self.latest_frame: Optional[np.ndarray] = None
        self.fullscreen_window: Optional[tk.Toplevel] = None
        self.fullscreen_label: Optional[tk.Label] = None
        self.fullscreen_photo: Optional[ImageTk.PhotoImage] = None
        self.display_geometry = None
        self.zone_draw_mode: Optional[str] = None
        self.zone_drag_start: Optional[tuple[int,int]] = None

        self._build_ui()
        self._refresh_all_logs()
        self._refresh_today_counts()
        self.root.after(30, self._poll)
        # Let the window paint first, then prepare AI in the background.
        self.root.after(300, self.engine.prepare_ai_async)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.bind("<F11>", lambda _e: self.toggle_fullscreen_camera())
        self.root.bind("<Escape>", lambda _e: self.exit_fullscreen_camera())
        if self.auto_start_var.get():
            self.root.after(650, self.start)

    def _build_ui(self):
        # Alert banner
        self.alert_frame = tk.Frame(self.root, bg=self.alert_default_bg, height=58)
        self.alert_frame.pack(fill="x")
        self.alert_label = tk.Label(
            self.alert_frame,
            text="STREETWATCH READY — waiting for camera",
            bg=self.alert_default_bg,
            fg="white",
            font=("Segoe UI", 17, "bold"),
            pady=10,
        )
        self.alert_label.pack(fill="x")

        # Main controls
        top = ttk.Frame(self.root, padding=(9, 8, 9, 4))
        top.pack(fill="x")
        ttk.Label(top, text="Camera:").pack(side="left")
        # Editable picker: choose a discovered camera OR type an RTSP/HTTP stream URL.
        self.camera_combo = ttk.Combobox(
            top, textvariable=self.source_var, width=22, state="normal",
            values=("Camera 0", "Camera 1", "Camera 2", "Camera 3", "Camera 4", "Camera 5")
        )
        self.camera_combo.pack(side="left", padx=(5, 4))
        ttk.Button(top, text="🔎 FIND CAMERAS", command=self.scan_cameras).pack(side="left", padx=(0, 7))
        ttk.Button(top, text="▶ START LIVE", command=self.start).pack(side="left", padx=3)
        ttk.Button(top, text="■ Stop", command=self.stop).pack(side="left", padx=3)
        ttk.Button(top, text="⛶ FULL SCREEN CAMERA", command=self.toggle_fullscreen_camera).pack(side="left", padx=(8, 3))
        ttk.Button(top, text="📷 Snapshot", command=self.manual_snapshot).pack(side="left", padx=3)
        ttk.Button(top, text="★ Bookmark last", command=self.bookmark_last_event).pack(side="left", padx=3)
        ttk.Label(top, text="AI confidence").pack(side="left", padx=(14, 4))
        ttk.Scale(top, from_=0.15, to=0.75, variable=self.conf_var, orient="horizontal", length=130).pack(side="left")
        ttk.Checkbutton(top, text="Sound", variable=self.sound_var).pack(side="left", padx=(12, 3))
        ttk.Checkbutton(top, text="Voice", variable=self.voice_var).pack(side="left", padx=3)
        ttk.Label(top, text="LOCAL AI • NO CLOUD", foreground="#087f23", font=("Segoe UI",9,"bold")).pack(side="left", padx=(10,3))
        ttk.Label(top, textvariable=self.status_var, font=("Segoe UI", 10, "bold")).pack(side="right", padx=8)

        # Monitor toggles / tripwire
        config = ttk.Frame(self.root, padding=(9, 3, 9, 7))
        config.pack(fill="x")
        ttk.Checkbutton(config, text="Vehicles", variable=self.monitor_vehicle_var).pack(side="left")
        ttk.Checkbutton(config, text="Auto-start monitoring", variable=self.auto_start_var).pack(side="left", padx=(8,4))
        ttk.Label(config, text="Performance").pack(side="left", padx=(8,3))
        ttk.Combobox(config, textvariable=self.performance_var, values=("Fast","Balanced","Maximum Accuracy"), state="readonly", width=16).pack(side="left")
        ttk.Checkbutton(config, text="People", variable=self.monitor_people_var).pack(side="left", padx=5)
        ttk.Checkbutton(config, text="Animals", variable=self.monitor_animal_var).pack(side="left", padx=5)
        ttk.Checkbutton(config, text="Ignore parked vehicles", variable=self.ignore_parked_var).pack(side="left", padx=(12, 5))
        ttk.Label(config, text="Motion sensitivity").pack(side="left", padx=(8, 3))
        ttk.Scale(config, from_=1, to=10, variable=self.motion_sensitivity_var, orient="horizontal", length=95).pack(side="left")
        ttk.Separator(config, orient="vertical").pack(side="left", fill="y", padx=10)
        ttk.Radiobutton(config, text="Traffic left/right", variable=self.orientation_var, value="vertical").pack(side="left", padx=3)
        ttk.Radiobutton(config, text="Traffic toward/away", variable=self.orientation_var, value="horizontal").pack(side="left", padx=3)
        ttk.Label(config, text="Yellow log line").pack(side="left", padx=(12, 3))
        ttk.Scale(config, from_=10, to=90, variable=self.position_var, orient="horizontal", length=125).pack(side="left")

        # Speed calibration panel
        speed = ttk.LabelFrame(self.root, text="Speed — automatic estimate is ON; calibration below is optional for higher accuracy", padding=(8, 4))
        speed.pack(fill="x", padx=9, pady=(0, 7))
        ttk.Checkbutton(speed, text="AUTO speed (no setup)", variable=self.auto_speed_var).pack(side="left", padx=(0,10))
        ttk.Checkbutton(speed, text="Use calibrated gates", variable=self.calibrated_speed_var).pack(side="left", padx=(0,10))
        ttk.Label(speed, text="Gate A").pack(side="left")
        ttk.Scale(speed, from_=5, to=95, variable=self.speed_a_var, orient="horizontal", length=115).pack(side="left", padx=(3, 10))
        ttk.Label(speed, text="Gate B").pack(side="left")
        ttk.Scale(speed, from_=5, to=95, variable=self.speed_b_var, orient="horizontal", length=115).pack(side="left", padx=(3, 10))
        ttk.Label(speed, text="Actual distance A↔B (feet)").pack(side="left")
        ttk.Spinbox(speed, from_=1, to=500, increment=1, textvariable=self.speed_distance_var, width=7).pack(side="left", padx=(4, 12))
        ttk.Label(speed, text="Repeat-ID strictness").pack(side="left")
        ttk.Scale(speed, from_=0.70, to=0.95, variable=self.reid_var, orient="horizontal", length=120).pack(side="left", padx=(3, 8))
        ttk.Label(speed, text="Higher = fewer false repeat matches").pack(side="left")

        lighting = ttk.LabelFrame(self.root, text="Automatic lighting", padding=(8, 4))
        lighting.pack(fill="x", padx=9, pady=(0, 7))
        ttk.Checkbutton(lighting, text="Auto Night Assist", variable=self.night_assist_var).pack(side="left")
        ttk.Checkbutton(lighting, text="Moving-headlight assist", variable=self.headlight_assist_var).pack(side="left", padx=(12,2))
        ttk.Label(lighting, text="Night threshold").pack(side="left", padx=(14, 3))
        ttk.Scale(lighting, from_=35, to=130, variable=self.night_threshold_var, orient="horizontal", length=150).pack(side="left")
        ttk.Label(lighting, textvariable=self.light_var, font=("Segoe UI", 10, "bold")).pack(side="left", padx=(14, 8))
        ttk.Label(lighting, text="Dark frames are enhanced before AI detection.").pack(side="left")

        paned = ttk.Panedwindow(self.root, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=9, pady=(0, 8))
        camera_panel = ttk.Frame(paned)
        side_panel = ttk.Frame(paned, width=610)
        paned.add(camera_panel, weight=3)
        paned.add(side_panel, weight=2)

        self.video_label = ttk.Label(camera_panel, anchor="center", text="Camera preview will appear here")
        self.video_label.pack(fill="both", expand=True)
        self.video_label.bind("<Double-1>", lambda _e: self.toggle_fullscreen_camera())
        self.video_label.bind("<ButtonPress-1>", self._zone_mouse_down)
        self.video_label.bind("<ButtonRelease-1>", self._zone_mouse_up)

        dash = ttk.Frame(camera_panel, padding=(2, 7))
        dash.pack(fill="x")
        ttk.Label(dash, textvariable=self.live_var, font=("Segoe UI", 10, "bold")).pack(side="left")
        ttk.Label(dash, textvariable=self.perf_var).pack(side="left", padx=18)
        ttk.Label(dash, textvariable=self.today_var).pack(side="right")
        ttk.Label(camera_panel, textvariable=self.last_event_var, anchor="w").pack(fill="x", padx=2, pady=(0, 4))

        self.tabs = ttk.Notebook(side_panel)
        self.tabs.pack(fill="both", expand=True)
        vehicle_tab = ttk.Frame(self.tabs, padding=5)
        people_tab = ttk.Frame(self.tabs, padding=5)
        chat_tab = ttk.Frame(self.tabs, padding=8)
        learn_tab = ttk.Frame(self.tabs, padding=8)
        zones_tab = ttk.Frame(self.tabs, padding=8)
        self.tabs.add(vehicle_tab, text="🚗 Vehicles")
        self.tabs.add(people_tab, text="👤 People & Animals")
        self.tabs.add(chat_tab, text="💬 ID + Teach")
        self.tabs.add(learn_tab, text="🧠 Learning")
        self.tabs.add(zones_tab, text="▣ Zones")

        self._build_vehicle_tab(vehicle_tab)
        self._build_people_tab(people_tab)
        self._build_chat_tab(chat_tab)
        self._build_learning_tab(learn_tab)
        self._build_zones_tab(zones_tab)

    def _build_vehicle_tab(self, parent):
        buttons = ttk.Frame(parent)
        buttons.pack(fill="x", pady=(0, 6))
        ttk.Button(buttons, text="Open photo", command=lambda: self.open_snapshot(self.vehicle_tree)).pack(side="left", padx=2)
        ttk.Button(buttons, text="Vehicle history", command=self.open_vehicle_history).pack(side="left", padx=2)
        ttk.Button(buttons, text="Export CSV", command=self.export_csv).pack(side="left", padx=2)
        ttk.Button(buttons, text="Open data folder", command=lambda: os.startfile(str(DATA_DIR))).pack(side="left", padx=2)
        ttk.Button(buttons, text="Backup data", command=self.backup_data).pack(side="left", padx=2)
        ttk.Button(buttons, text="Note last", command=self.note_last_event).pack(side="left", padx=2)
        ttk.Button(buttons, text="Clear vehicle events", command=self.clear_vehicle_logs).pack(side="right", padx=2)

        cols = ("id", "time", "type", "color", "speed", "direction", "repeat", "name")
        self.vehicle_tree = ttk.Treeview(parent, columns=cols, show="headings", selectmode="browse")
        headings = {
            "id": "Vehicle ID", "time": "Time", "type": "Type", "color": "Color",
            "speed": "MPH", "direction": "Direction", "repeat": "Match", "name": "Name/Label",
        }
        widths = {"id": 86, "time": 145, "type": 70, "color": 78, "speed": 52, "direction": 92, "repeat": 58, "name": 105}
        for c in cols:
            self.vehicle_tree.heading(c, text=headings[c])
            self.vehicle_tree.column(c, width=widths[c], anchor="center" if c not in {"time", "name"} else "w")
        y = ttk.Scrollbar(parent, orient="vertical", command=self.vehicle_tree.yview)
        self.vehicle_tree.configure(yscrollcommand=y.set)
        self.vehicle_tree.pack(side="left", fill="both", expand=True)
        y.pack(side="right", fill="y")
        self.vehicle_tree.bind("<Double-1>", lambda _e: self.open_snapshot(self.vehicle_tree))

    def _build_people_tab(self, parent):
        info = ttk.Label(
            parent,
            text=("People get a visit/event ID and photo. StreetWatch does not use facial recognition or "
                  "persistent biometric person identification. Animals can keep a repeat SWA ID and name."),
            wraplength=535,
            justify="left",
        )
        info.pack(fill="x", pady=(0, 6))
        buttons = ttk.Frame(parent)
        buttons.pack(fill="x", pady=(0, 6))
        ttk.Button(buttons, text="Open photo", command=lambda: self.open_snapshot(self.people_tree)).pack(side="left", padx=2)
        ttk.Button(buttons, text="Clear people/animal events", command=self.clear_people_logs).pack(side="right", padx=2)

        cols = ("id", "time", "kind", "type", "direction", "name")
        self.people_tree = ttk.Treeview(parent, columns=cols, show="headings", selectmode="browse")
        headings = {"id": "ID", "time": "Time", "kind": "Kind", "type": "Type", "direction": "Direction", "name": "Name/Note"}
        widths = {"id": 90, "time": 145, "kind": 70, "type": 72, "direction": 100, "name": 120}
        for c in cols:
            self.people_tree.heading(c, text=headings[c])
            self.people_tree.column(c, width=widths[c], anchor="center" if c not in {"time", "name"} else "w")
        y = ttk.Scrollbar(parent, orient="vertical", command=self.people_tree.yview)
        self.people_tree.configure(yscrollcommand=y.set)
        self.people_tree.pack(side="left", fill="both", expand=True)
        y.pack(side="right", fill="y")
        self.people_tree.bind("<Double-1>", lambda _e: self.open_snapshot(self.people_tree))

    def _build_chat_tab(self, parent):
        intro = ttk.Label(
            parent,
            text=("Local ID helper. Examples:\n"
                  "  SWV-0004 is My Jeep\n"
                  "  SWA-0012 is Buddy\n"
                  "  P-000123 is John   (labels that recorded visit only)\n"
                  "  SWV-0004            (show what StreetWatch knows)\n"
                  "  you missed a vehicle / that was a false alarm\n"
                  "  learn feature: <idea>"),
            justify="left",
        )
        intro.pack(fill="x", pady=(0, 7))

        self.chat_text = tk.Text(parent, height=18, wrap="word", state="disabled", font=("Segoe UI", 10))
        self.chat_text.pack(fill="both", expand=True)
        entry_bar = ttk.Frame(parent)
        entry_bar.pack(fill="x", pady=(7, 0))
        self.chat_var = tk.StringVar()
        entry = ttk.Entry(entry_bar, textvariable=self.chat_var)
        entry.pack(side="left", fill="x", expand=True)
        entry.bind("<Return>", lambda _e: self._chat_send())
        ttk.Button(entry_bar, text="Send", command=self._chat_send).pack(side="left", padx=(5, 0))
        self._chat_append("StreetWatch", "Type an ID or tell me the name/label you want attached to it.")

    def _build_zones_tab(self, parent):
        ttk.Label(parent, text="Draw simple areas directly on the camera preview.",
                  font=("Segoe UI", 11, "bold")).pack(anchor="w")
        ttk.Label(parent, wraplength=535, justify="left", text=(
            "Click one of the buttons below, then drag a rectangle on the main camera image. "
            "IGNORE suppresses detections in a reflection/neighbor/parked area. ROAD, DRIVEWAY and "
            "SIDEWALK label events so StreetWatch understands where something is happening."
        )).pack(fill="x", pady=(4,10))
        for name, title in [("road","Draw ROAD zone"),("driveway","Draw DRIVEWAY zone"),
                            ("sidewalk","Draw SIDEWALK zone"),("ignore","Draw IGNORE zone")]:
            row=ttk.Frame(parent); row.pack(fill="x",pady=3)
            ttk.Button(row,text=title,command=lambda n=name:self._zone_begin(n)).pack(side="left",fill="x",expand=True)
            ttk.Button(row,text="Clear",command=lambda n=name:self._zone_clear(n)).pack(side="left",padx=(6,0))
        self.zone_status_var=tk.StringVar(value="Zones are saved automatically.")
        ttk.Label(parent,textvariable=self.zone_status_var,wraplength=535,justify="left").pack(fill="x",pady=(12,0))

    def _zone_begin(self, name: str):
        self.zone_draw_mode=name
        self.zone_drag_start=None
        self.zone_status_var.set(f"Draw {name.upper()}: click and drag a rectangle on the main camera view.")

    def _zone_clear(self, name: str):
        self.engine.zones.clear(name)
        self.zone_status_var.set(f"Cleared {name.upper()} zone.")

    def _zone_mouse_down(self, event):
        if not self.zone_draw_mode:
            return
        self.zone_drag_start=(event.x,event.y)

    def _zone_mouse_up(self, event):
        if not self.zone_draw_mode or self.zone_drag_start is None or self.display_geometry is None:
            return
        iw,ih,ox,oy,fw,fh=self.display_geometry
        def conv(px,py):
            x=max(0,min(iw,px-ox)); y=max(0,min(ih,py-oy))
            return x/max(iw,1), y/max(ih,1)
        x1,y1=conv(*self.zone_drag_start); x2,y2=conv(event.x,event.y)
        name=self.zone_draw_mode
        self.engine.zones.set_zone(name,(x1,y1,x2,y2))
        self.zone_draw_mode=None; self.zone_drag_start=None
        self.zone_status_var.set(f"Saved {name.upper()} zone. You can redraw it any time.")

    def _build_learning_tab(self, parent):
        ttk.Label(parent, text="StreetWatch learns this camera locally from your feedback.",
                  font=("Segoe UI", 11, "bold")).pack(anchor="w")
        ttk.Label(parent, textvariable=self.learning_var, wraplength=535, justify="left").pack(fill="x", pady=(5,10))
        grid = ttk.Frame(parent)
        grid.pack(fill="x")
        ttk.Button(grid, text="That was a FALSE ALARM", command=self._learn_false_alarm).grid(row=0,column=0,sticky="ew",padx=3,pady=3)
        ttk.Button(grid, text="You MISSED a vehicle", command=self._learn_missed).grid(row=0,column=1,sticky="ew",padx=3,pady=3)
        ttk.Button(grid, text="Night is TOO DARK", command=self._learn_night_dark).grid(row=1,column=0,sticky="ew",padx=3,pady=3)
        ttk.Button(grid, text="Night image is TOO NOISY", command=self._learn_night_noisy).grid(row=1,column=1,sticky="ew",padx=3,pady=3)
        ttk.Button(grid, text="Reset learned tuning", command=self._learn_reset).grid(row=2,column=0,columnspan=2,sticky="ew",padx=3,pady=7)
        grid.columnconfigure(0, weight=1); grid.columnconfigure(1, weight=1)
        ttk.Separator(parent).pack(fill="x", pady=8)
        ttk.Label(parent, text="How learning works", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        ttk.Label(parent, wraplength=535, justify="left", text=(
            "Feedback gently changes motion sensitivity, confirmation frames, AI confidence, and night thresholds. "
            "The settings are stored in data/learning_profile.json and reused next time. Automatic speed also "
            "learns a correction factor when a stronger calibrated measurement ever becomes available. The app "
            "does not rewrite its own program code or upload your camera feed."
        )).pack(fill="x", pady=(3,8))
        ttk.Label(parent, textvariable=self.dashboard_var, wraplength=535, justify="left").pack(fill="x", pady=(5,0))

    def _sync_learning_to_controls(self):
        p = self.engine.learning.profile
        self.conf_var.set(float(p.get("confidence", self.conf_var.get())))
        self.motion_sensitivity_var.set(float(p.get("motion_sensitivity", self.motion_sensitivity_var.get())))
        self.night_threshold_var.set(int(p.get("night_threshold", self.night_threshold_var.get())))
        self.auto_speed_var.set(bool(p.get("auto_speed_enabled", True)))
        self.headlight_assist_var.set(bool(p.get("headlight_assist", True)))
        self.learning_var.set(self.engine.learning.summary())
        self.engine.learning.apply_to_engine(self.engine)

    def _learn_false_alarm(self):
        self.engine.db.add_feedback("false_alarm")
        self.engine.learning.feedback_false_alarm("user button")
        self._sync_learning_to_controls()
        self.dashboard_var.set("Learned: becoming slightly stricter to reduce false alarms.")

    def _learn_missed(self):
        self.engine.db.add_feedback("missed_vehicle")
        self.engine.learning.feedback_missed("user button")
        self._sync_learning_to_controls()
        self.dashboard_var.set("Learned: becoming slightly more sensitive to moving vehicles.")

    def _learn_night_dark(self):
        self.engine.db.add_feedback("night_too_dark")
        self.engine.learning.feedback_night_too_dark()
        self._sync_learning_to_controls()
        self.dashboard_var.set("Learned: Night Assist will engage earlier/stronger.")

    def _learn_night_noisy(self):
        self.engine.db.add_feedback("night_too_noisy")
        self.engine.learning.feedback_night_too_noisy()
        self._sync_learning_to_controls()
        self.dashboard_var.set("Learned: Night Assist will be less aggressive.")

    def _learn_reset(self):
        if messagebox.askyesno("StreetWatch", "Reset all learned detection tuning to defaults?"):
            self.engine.learning.reset()
            self._sync_learning_to_controls()
            self.dashboard_var.set("Learning profile reset to defaults.")

    def _parse_camera_source(self) -> Union[int, str]:
        src_text = self.source_var.get().strip()
        match = re.fullmatch(r"(?:Camera\s*)?(\d+)", src_text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
        return src_text

    def scan_cameras(self):
        """Find working local webcams without freezing the StreetWatch window."""
        if self.camera_scan_running:
            return
        if self.engine.running:
            messagebox.showinfo("StreetWatch", "Stop monitoring first, then click FIND CAMERAS.")
            return
        self.camera_scan_running = True
        self.status_var.set("Finding cameras...")

        def worker():
            found: list[int] = []
            for idx in range(8):
                cap = None
                try:
                    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                    if not cap.isOpened():
                        cap.release()
                        cap = cv2.VideoCapture(idx, cv2.CAP_MSMF)
                    if cap.isOpened():
                        # A successful frame read weeds out stale/virtual camera entries.
                        ok, _frame = cap.read()
                        if ok:
                            found.append(idx)
                except Exception:
                    pass
                finally:
                    try:
                        if cap is not None:
                            cap.release()
                    except Exception:
                        pass
            self.event_queue.put(("camera_scan", found))

        threading.Thread(target=worker, daemon=True, name="StreetWatch-Camera-Scan").start()

    def start(self):
        if self.engine.running:
            return
        source = self._parse_camera_source()
        if isinstance(source, str) and not source:
            messagebox.showinfo("StreetWatch", "Pick a camera first, or type a camera stream address.")
            return
        # Save friendly picker text while passing a real camera index to OpenCV.
        self.status_var.set("Opening camera...")
        self.engine.start(source, float(self.conf_var.get()), self.orientation_var.get(), int(self.position_var.get()))

    def stop(self):
        self.status_var.set("Stopping...")
        self.engine.stop()

    def _poll(self):
        try:
            frame = self.frame_queue.get_nowait()
            self._show_frame(frame)
        except queue.Empty:
            pass

        try:
            while True:
                kind, payload = self.event_queue.get_nowait()
                if kind == "status":
                    self.status_var.set(str(payload))
                elif kind == "error":
                    self.status_var.set("Error")
                    messagebox.showerror("StreetWatch", str(payload))
                elif kind == "event":
                    self._handle_event(payload)
                elif kind == "alert":
                    self._handle_alert(payload)
                elif kind == "stats":
                    self._handle_stats(payload)
                elif kind == "speed_update":
                    self._handle_speed_update(payload)
                elif kind == "camera_scan":
                    self.camera_scan_running = False
                    cams = [f"Camera {int(i)}" for i in payload]
                    if cams:
                        self.camera_combo.configure(values=tuple(cams))
                        current = self.source_var.get().strip()
                        if not re.fullmatch(r"(?:Camera\s*)?\d+", current, flags=re.IGNORECASE):
                            pass  # keep a manually typed stream URL
                        elif current not in cams and current.replace("Camera ", "") not in [str(i) for i in payload]:
                            self.source_var.set(cams[0])
                        self.status_var.set(f"Found {len(cams)} camera{'s' if len(cams) != 1 else ''} — pick one and START LIVE")
                    else:
                        self.camera_combo.configure(values=("Camera 0", "Camera 1", "Camera 2", "Camera 3"))
                        self.status_var.set("No camera answered the scan — you can still try Camera 0/1 or a stream URL")
        except queue.Empty:
            pass

        # Live settings, no restart required.
        self.engine.confidence = float(self.conf_var.get())
        self.engine.tripwire_orientation = self.orientation_var.get()
        self.engine.tripwire_position = int(self.position_var.get())
        self.engine.speed_gate_a = int(self.speed_a_var.get())
        self.engine.speed_gate_b = int(self.speed_b_var.get())
        try:
            self.engine.speed_distance_ft = float(self.speed_distance_var.get())
        except Exception:
            pass
        self.engine.reid_threshold = float(self.reid_var.get())
        self.engine.monitor_vehicles = bool(self.monitor_vehicle_var.get())
        self.engine.monitor_people = bool(self.monitor_people_var.get())
        self.engine.monitor_animals = bool(self.monitor_animal_var.get())
        self.engine.ignore_stationary_vehicles = bool(self.ignore_parked_var.get())
        self.engine.motion_sensitivity = float(self.motion_sensitivity_var.get())
        self.engine.auto_night_assist = bool(self.night_assist_var.get())
        self.engine.night_threshold = float(self.night_threshold_var.get())
        self.engine.auto_speed_enabled = bool(self.auto_speed_var.get())
        self.engine.headlight_assist = bool(self.headlight_assist_var.get())
        self.engine.calibrated_speed_enabled = bool(self.calibrated_speed_var.get())
        mode = self.performance_var.get()
        self.engine.infer_imgsz = {"Fast":480, "Balanced":640, "Maximum Accuracy":960}.get(mode,640)

        self.root.after(30, self._poll)

    def _show_frame(self, frame: np.ndarray):
        self.latest_frame = frame.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        max_w = max(620, self.video_label.winfo_width())
        max_h = max(450, self.video_label.winfo_height())
        image.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
        iw, ih = image.size
        lw, lh = max(1, self.video_label.winfo_width()), max(1, self.video_label.winfo_height())
        self.display_geometry = (iw, ih, max(0,(lw-iw)//2), max(0,(lh-ih)//2), frame.shape[1], frame.shape[0])
        self.photo = ImageTk.PhotoImage(image=image)
        self.video_label.configure(image=self.photo, text="")

        if self.fullscreen_window is not None and self.fullscreen_window.winfo_exists() and self.fullscreen_label is not None:
            fs_img = Image.fromarray(rgb)
            sw = max(800, self.fullscreen_window.winfo_screenwidth())
            sh = max(600, self.fullscreen_window.winfo_screenheight())
            fs_img.thumbnail((sw, sh), Image.Resampling.LANCZOS)
            self.fullscreen_photo = ImageTk.PhotoImage(image=fs_img)
            self.fullscreen_label.configure(image=self.fullscreen_photo, text="")

    def manual_snapshot(self):
        if self.latest_frame is None:
            messagebox.showinfo("StreetWatch", "No camera frame is available yet.")
            return
        path = SNAPSHOT_DIR / f"manual_{datetime.now():%Y%m%d_%H%M%S}.jpg"
        cv2.imwrite(str(path), self.latest_frame)
        self.last_event_var.set(f"Manual snapshot saved: {path.name}")

    def bookmark_last_event(self):
        if not self.last_event_code:
            messagebox.showinfo("StreetWatch", "No event has been logged yet.")
            return
        state = self.engine.db.toggle_bookmark(self.last_event_code)
        self.last_event_var.set(f"{self.last_event_code} bookmark {'ON' if state else 'OFF'}")

    def note_last_event(self):
        if not self.last_event_code:
            return
        note = simpledialog.askstring("StreetWatch", f"Note for {self.last_event_code}:")
        if note is not None:
            self.engine.db.set_event_note(self.last_event_code, note)

    def backup_data(self):
        target = filedialog.asksaveasfilename(title="Backup StreetWatch data", defaultextension=".zip",
                                              initialfile=f"StreetWatch-backup-{datetime.now():%Y%m%d}.zip",
                                              filetypes=[("ZIP file","*.zip")])
        if not target:
            return
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
            for f in DATA_DIR.rglob("*"):
                if f.is_file() and f.name != "test_v3.db":
                    z.write(f, f.relative_to(APP_DIR))
        messagebox.showinfo("StreetWatch", f"Backup created:\n{target}")
        messagebox.showinfo("StreetWatch", f"Backup created:\n{target}")

    def toggle_fullscreen_camera(self):
        if self.fullscreen_window is not None and self.fullscreen_window.winfo_exists():
            self.exit_fullscreen_camera()
            return
        win = tk.Toplevel(self.root)
        self.fullscreen_window = win
        win.title("StreetWatch — Full Screen Camera")
        win.configure(bg="black")
        win.attributes("-fullscreen", True)
        win.bind("<Escape>", lambda _e: self.exit_fullscreen_camera())
        win.bind("<F11>", lambda _e: self.exit_fullscreen_camera())
        win.protocol("WM_DELETE_WINDOW", self.exit_fullscreen_camera)

        label = tk.Label(win, bg="black", fg="white", text="Waiting for camera...", font=("Segoe UI", 18, "bold"))
        label.pack(fill="both", expand=True)
        self.fullscreen_label = label

        exit_btn = tk.Button(
            win, text="EXIT FULL SCREEN  (Esc)", command=self.exit_fullscreen_camera,
            bg="#111827", fg="white", activebackground="#374151", activeforeground="white",
            font=("Segoe UI", 12, "bold"), relief="flat", padx=16, pady=8,
        )
        exit_btn.place(relx=0.985, rely=0.02, anchor="ne")

        if self.latest_frame is not None:
            self._show_frame(self.latest_frame)

    def exit_fullscreen_camera(self):
        win = self.fullscreen_window
        self.fullscreen_window = None
        self.fullscreen_label = None
        self.fullscreen_photo = None
        if win is not None:
            try:
                win.destroy()
            except Exception:
                pass

    def _handle_stats(self, payload: dict):
        c = payload.get("counts", {})
        total_v = c.get('vehicle', 0)
        moving_v = c.get('moving_vehicle', 0)
        self.live_var.set(
            f"Live: {total_v} vehicles ({moving_v} moving) • {c.get('person',0)} people • {c.get('animal',0)} animals"
        )
        self.perf_var.set(f"AI: {payload.get('fps',0):.1f} FPS • {payload.get('infer_ms',0):.0f} ms")
        brightness = float(payload.get("brightness", 255.0))
        if payload.get("night_active"):
            strength = float(payload.get("night_strength", 0.0))
            self.light_var.set(f"Lighting: NIGHT ASSIST ON • level {brightness:.0f} • boost {strength:.0%} • quality {payload.get('quality','--')}")
        else:
            self.light_var.set(f"Lighting: DAY / normal • level {brightness:.0f} • quality {payload.get('quality','--')}")

    def _handle_alert(self, payload: dict):
        category = payload.get("category", "vehicle")
        text = payload.get("text", "Detected")
        phase = payload.get("phase", "detected")
        palette = {
            "vehicle": ("#ff4d00", "white"),
            "person": ("#0066ff", "white"),
            "animal": ("#00a341", "white"),
        }
        if phase == "passed":
            bg, fg = "#ffd000", "black"
        elif phase == "approaching":
            bg, fg = "#b000ff", "white"
        elif phase == "stopped":
            bg, fg = "#ff1744", "white"
        elif phase == "repeat":
            bg, fg = "#7c3aed", "white"
        elif phase == "night_hint":
            bg, fg = "#ff8c00", "black"
        else:
            bg, fg = palette.get(category, ("#ff4d00", "white"))
        self._flash_banner(text, bg, fg)

        if self.sound_var.get() and winsound is not None:
            threading.Thread(target=self._beep_for, args=(category, phase), daemon=True).start()
        if self.voice_var.get() and payload.get("speech"):
            self.speech.say(payload["speech"])

    def _beep_for(self, category: str, phase: str):
        if winsound is None:
            return
        try:
            if phase == "passed":
                for f in (950, 1250):
                    winsound.Beep(f, 100)
            else:
                freq = {"vehicle": 1000, "person": 760, "animal": 860}.get(category, 900)
                winsound.Beep(freq, 140)
        except Exception:
            pass

    def _flash_banner(self, text: str, bg: str, fg: str):
        self.alert_flash_token += 1
        token = self.alert_flash_token
        self.alert_label.config(text=text, fg=fg)

        def step(i: int):
            if token != self.alert_flash_token:
                return
            if i >= 8:
                self.alert_frame.config(bg=self.alert_default_bg)
                self.alert_label.config(bg=self.alert_default_bg, fg="white")
                return
            use_bg = bg if i % 2 == 0 else self.alert_default_bg
            self.alert_frame.config(bg=use_bg)
            self.alert_label.config(bg=use_bg)
            self.root.after(180, lambda: step(i + 1))

        step(0)

    def _handle_event(self, row: dict):
        category = row["category"]
        event_id = int(row["id"])
        self.last_event_code = row["event_code"]
        self.last_event_var.set(
            f"Last event: {row['event_code']} • {row['object_type']} • {self._fmt_time(row['happened_at'])}"
        )
        if category == "vehicle":
            identity = row.get("identity_code") or row["event_code"]
            speed = row.get("speed_mph")
            iid = self.vehicle_tree.insert("", 0, values=(
                identity,
                self._fmt_time(row["happened_at"]),
                row["object_type"],
                row["color"],
                (("~" if row.get("speed_source") == "AUTO" else "") + f"{speed:.1f}" +
                 (f" ({row.get('speed_confidence',0):.0%})" if row.get("speed_source") == "AUTO" else "")) if speed is not None else "—",
                row["direction"],
                ((f"{row.get('identity_match',0):.0%}") if row.get("identity_match", 0) > 0 else "NEW") if row.get("identity_code") else "—",
                self._identity_name(row.get("identity_code", "")),
            ))
            self.dbid_to_vehicle_iid[event_id] = iid
        else:
            ident = row["event_code"] if category == "person" else (row.get("identity_code") or row["event_code"])
            name = row.get("display_name", "")
            if category == "animal" and row.get("identity_code"):
                name = self._identity_name(row["identity_code"])
            iid = self.people_tree.insert("", 0, values=(
                ident,
                self._fmt_time(row["happened_at"]),
                "Person" if category == "person" else "Animal",
                row["object_type"],
                row["direction"],
                name,
            ))
        self.snapshot_by_iid[iid] = row.get("snapshot_path", "") or ""
        self._refresh_today_counts()

    def _handle_speed_update(self, payload: dict):
        iid = self.dbid_to_vehicle_iid.get(int(payload.get("event_id", 0)))
        if not iid or not self.vehicle_tree.exists(iid):
            self._refresh_vehicle_logs()
            return
        vals = list(self.vehicle_tree.item(iid, "values"))
        if len(vals) >= 5:
            prefix = "~" if payload.get("speed_source") == "AUTO" else ""
            conf = f" ({float(payload.get('speed_confidence',0)):.0%})" if payload.get("speed_source") == "AUTO" else ""
            vals[4] = f"{prefix}{float(payload['speed_mph']):.1f}{conf}"
            self.vehicle_tree.item(iid, values=vals)

    def _refresh_all_logs(self):
        self._refresh_vehicle_logs()
        self._refresh_people_logs()

    def _refresh_vehicle_logs(self):
        for iid in self.vehicle_tree.get_children():
            self.vehicle_tree.delete(iid)
        self.dbid_to_vehicle_iid.clear()
        for row in self.engine.db.recent_events(("vehicle",), 1200):
            ident = row["identity_code"] or row["event_code"]
            speed = row["speed_mph"]
            iid = self.vehicle_tree.insert("", "end", values=(
                ident, self._fmt_time(row["happened_at"]), row["object_type"], row["color"],
                (("~" if row["speed_source"] == "AUTO" else "") + f"{speed:.1f}" +
                 (f" ({row['speed_confidence']:.0%})" if row["speed_source"] == "AUTO" else "")) if speed is not None else "—", row["direction"],
                ((f"{row['identity_match']:.0%}") if row["identity_match"] > 0 else "NEW") if row["identity_code"] else "—",
                row["resolved_name"] or "",
            ))
            self.snapshot_by_iid[iid] = row["snapshot_path"] or ""
            self.dbid_to_vehicle_iid[int(row["id"])] = iid

    def _refresh_people_logs(self):
        for iid in self.people_tree.get_children():
            self.people_tree.delete(iid)
        for row in self.engine.db.recent_events(("person", "animal"), 1200):
            ident = row["event_code"] if row["category"] == "person" else (row["identity_code"] or row["event_code"])
            iid = self.people_tree.insert("", "end", values=(
                ident, self._fmt_time(row["happened_at"]),
                "Person" if row["category"] == "person" else "Animal",
                row["object_type"], row["direction"], row["resolved_name"] or "",
            ))
            self.snapshot_by_iid[iid] = row["snapshot_path"] or ""

    def _refresh_today_counts(self):
        c = self.engine.db.today_counts()
        self.today_var.set(
            f"Today: {c['vehicle']} vehicles • {c['person']} people • {c['animal']} animals"
        )
        d = self.engine.db.dashboard_stats()
        self.dashboard_var.set(
            f"Today: {d['vehicles']} vehicle events • average speed {d['avg_speed']:.1f} MPH • "
            f"fastest {d['max_speed']:.1f} MPH • repeat matches {d['repeats']}."
        )

    @staticmethod
    def _fmt_time(iso_text: str) -> str:
        try:
            d = datetime.fromisoformat(iso_text)
            return d.strftime("%m/%d %I:%M:%S %p")
        except Exception:
            return iso_text

    def _identity_name(self, code: str) -> str:
        if not code:
            return ""
        row = self.engine.db.get_identity(code)
        return (row["name"] or "") if row else ""

    def open_vehicle_history(self):
        selected=self.vehicle_tree.selection()
        if not selected:
            return
        vals=self.vehicle_tree.item(selected[0],"values")
        if not vals:
            return
        code=str(vals[0])
        if not code.startswith("SWV-"):
            messagebox.showinfo("StreetWatch","That event does not have a persistent vehicle ID yet.")
            return
        rows=self.engine.db.recent_identity_events(code,100)
        ident=self.engine.db.get_identity(code)
        win=tk.Toplevel(self.root); win.title(f"StreetWatch — {code} history"); win.geometry("760x520")
        name=(ident['name'] if ident else '') or 'unnamed'
        ttk.Label(win,text=f"{code} — {name} — seen {len(rows)} logged pass(es)",font=("Segoe UI",12,"bold")).pack(anchor="w",padx=10,pady=10)
        box=tk.Text(win,wrap="word",font=("Consolas",10)); box.pack(fill="both",expand=True,padx=10,pady=(0,10))
        for r in rows:
            speed='—' if r['speed_mph'] is None else (("~" if r['speed_source']=='AUTO' else '') + f"{r['speed_mph']:.1f} MPH")
            box.insert('end',f"{r['event_code']}  {self._fmt_time(r['happened_at'])}  {r['color']} {r['object_type']}  {speed}  {r['direction']}  {r['zone'] or ''}\n")
        box.config(state='disabled')

    def open_snapshot(self, tree: ttk.Treeview):
        selected = tree.selection()
        if not selected:
            return
        path = self.snapshot_by_iid.get(selected[0], "")
        if path and Path(path).exists():
            os.startfile(path)
        else:
            messagebox.showinfo("StreetWatch", "No photo is available for this event.")

    def clear_vehicle_logs(self):
        if messagebox.askyesno("StreetWatch", "Clear vehicle event history? Vehicle identity memory and photos are kept."):
            self.engine.db.clear_events(("vehicle",))
            self._refresh_vehicle_logs()
            self._refresh_today_counts()

    def clear_people_logs(self):
        if messagebox.askyesno("StreetWatch", "Clear people and animal event history? Animal identity memory and photos are kept."):
            self.engine.db.clear_events(("person", "animal"))
            self._refresh_people_logs()
            self._refresh_today_counts()

    def export_csv(self):
        default = f"StreetWatch-events-{datetime.now():%Y%m%d}.csv"
        path = filedialog.asksaveasfilename(
            title="Export StreetWatch events",
            defaultextension=".csv",
            initialfile=default,
            filetypes=[("CSV file", "*.csv")],
        )
        if not path:
            return
        self.engine.db.export_csv(path)
        messagebox.showinfo("StreetWatch", f"Exported:\n{path}")

    def _chat_append(self, who: str, text: str):
        self.chat_text.config(state="normal")
        self.chat_text.insert("end", f"{who}: {text}\n\n")
        self.chat_text.see("end")
        self.chat_text.config(state="disabled")

    def _chat_send(self):
        raw = self.chat_var.get().strip()
        if not raw:
            return
        self.chat_var.set("")
        self._chat_append("You", raw)
        self._chat_append("StreetWatch", self._chat_answer(raw))

    def _chat_answer(self, raw: str) -> str:
        text = raw.strip()
        low = text.lower()
        if any(p in low for p in ["you missed a vehicle", "missed a car", "missed a truck", "missed vehicle"]):
            self.engine.db.add_feedback("missed_vehicle", note=text)
            msg = self.engine.learning.feedback_missed(text)
            self._sync_learning_to_controls()
            return "I learned from that miss. " + msg
        if any(p in low for p in ["false alarm", "wrong detection", "that was parked", "beeped at a parked"]):
            self.engine.db.add_feedback("false_alarm", note=text)
            msg = self.engine.learning.feedback_false_alarm(text)
            self._sync_learning_to_controls()
            return "I learned from that false alarm. " + msg
        if "night" in low and any(p in low for p in ["too dark", "cant see", "can't see"]):
            self.engine.learning.feedback_night_too_dark(); self._sync_learning_to_controls()
            return "I adjusted Night Assist to engage earlier. " + self.engine.learning.summary(short=True)
        if low.startswith("learn feature:") or low.startswith("feature request:"):
            idea = text.split(":",1)[1].strip() if ":" in text else text
            self.engine.learning.record_request(idea)
            return ("I saved that capability request in my local learning profile. I can learn settings/rules automatically, "
                    "but adding brand-new program code still requires an app update rather than silently rewriting myself.")
        if low in {"show learning", "what have you learned", "learning status"}:
            return self.engine.learning.summary()
        if low == "reset learning":
            self.engine.learning.reset(); self._sync_learning_to_controls()
            return "Learning profile reset to defaults."
        # Flexible: "SWV-0001 is My Jeep", "SWA-2 = Buddy", "P-000123 name is John"
        m = re.match(r"^\s*([A-Za-z]+-\d+)\s+(?:is|=|name\s+is)\s+(.+?)\s*$", text, flags=re.I)
        if m:
            code = m.group(1).upper()
            name = m.group(2).strip()
            if code.startswith(("SWV-", "SWA-")):
                if self.engine.db.set_identity_name(code, name):
                    self._refresh_all_logs()
                    kind = "vehicle" if code.startswith("SWV-") else "animal"
                    return f"Got it. {code} is now labeled “{name}”. When that {kind} is likely matched again, StreetWatch can show and announce that label."
                return f"I can't find {code}. Let it be detected/logged first, then try again."
            if code.startswith("P-"):
                if self.engine.db.set_person_event_name(code, name):
                    self._refresh_people_logs()
                    return (f"I labeled recorded person event {code} as “{name}”. This label stays on that visit/photo only; "
                            "StreetWatch does not use facial recognition to identify that person on future visits.")
                return f"I can't find person event {code}."

        if any(k in low for k in ["daily summary", "how many vehicles today", "traffic today", "fastest today"]):
            c=self.engine.db.today_counts(); d=self.engine.db.dashboard_stats(); dirs=self.engine.db.direction_stats_today()
            dir_text=", ".join(f"{k}: {v}" for k,v in dirs.items()) or "no direction data yet"
            return (f"Today StreetWatch logged {c['vehicle']} vehicle event(s), {c['person']} person visit(s), and {c['animal']} animal event(s). "
                    f"Vehicle average speed is {d['avg_speed']:.1f} MPH and fastest is {d['max_speed']:.1f} MPH. Directions: {dir_text}.")
        if low.startswith("show ") or low.startswith("find "):
            rows=self.engine.db.search_vehicle_events(low,10)
            if not rows:
                return "I could not find matching vehicle events in the local history."
            lines=[]
            for r in rows:
                speed='—' if r['speed_mph'] is None else (("~" if r['speed_source']=='AUTO' else '') + f"{r['speed_mph']:.0f} mph")
                lines.append(f"{r['event_code']} / {r['identity_code'] or 'new'} / {r['color']} {r['object_type']} / {speed} / {self._fmt_time(r['happened_at'])}")
            return "Matching events:\n" + "\n".join(lines)

        # ID lookup
        id_match = re.search(r"\b(SWV-\d+|SWA-\d+|[VPA]-\d+)\b", text, flags=re.I)
        if id_match:
            code = id_match.group(1).upper()
            if code.startswith(("SWV-", "SWA-")):
                row = self.engine.db.get_identity(code)
                if not row:
                    return f"I can't find {code}."
                name = row["name"] or "unnamed"
                return (f"{code}: {row['object_type']}, {row['color'] or 'color not used'}, label {name}, "
                        f"seen {row['seen_count']} time(s), last seen {self._fmt_time(row['last_seen'])}.")
            event = self.engine.db.get_event(code)
            if event:
                name = event["display_name"] or "no note"
                return (f"{code}: {event['category']} / {event['object_type']}, "
                        f"{self._fmt_time(event['happened_at'])}, {event['direction']}, note: {name}.")
            return f"I can't find {code}."

        if text.lower() in {"help", "?", "what can i type"}:
            return ("Try: “SWV-0004 is My Jeep”, “SWA-0012 is Buddy”, “P-000123 is John”, "
                    "or type an ID by itself to look it up.")
        return ("I work with StreetWatch IDs. Type an ID by itself, or say something like "
                "“SWV-0004 is My Jeep” or “SWA-0012 is Buddy”.")

    def _on_close(self):
        # Persist the final camera tuning so StreetWatch starts next time where you left it.
        try:
            prof = self.engine.learning.profile
            prof["confidence"] = float(self.conf_var.get())
            prof["motion_sensitivity"] = float(self.motion_sensitivity_var.get())
            prof["night_threshold"] = float(self.night_threshold_var.get())
            prof["auto_speed_enabled"] = bool(self.auto_speed_var.get())
            prof["headlight_assist"] = bool(self.headlight_assist_var.get())
            prof["auto_start_monitoring"] = bool(self.auto_start_var.get())
            prof["last_source"] = self.source_var.get().strip() or "0"
            prof["performance_mode"] = self.performance_var.get()
            self.engine.learning.save()
        except Exception:
            pass
        self.engine.stop()
        self.speech.stop()
        self.root.destroy()


def main():
    root = tk.Tk()
    style = ttk.Style()
    try:
        style.theme_use("vista")
    except tk.TclError:
        pass
    StreetWatchApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()