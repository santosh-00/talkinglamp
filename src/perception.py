"""Webcam perception: face presence, head orientation, and engagement with hysteresis.

Runs in its own thread. Publishes:
  - state:   latest FaceState snapshot (level state; consumers should reconcile against this)
  - events:  queue of (name, monotonic_time, info) for logging/metrics, always ordered
             face_appeared → engaged → disengaged → face_lost
Everything runs locally; no frames leave the machine.
"""
import os
import queue
import threading
import time
import urllib.request
from dataclasses import dataclass, replace

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")
MODEL_PATH = os.path.join(ROOT, "models", "face_landmarker.task")

# MediaPipe face-mesh landmark indices
NOSE, EYE_OUTER_A, EYE_OUTER_B, CHIN = 1, 33, 263, 152


@dataclass
class PerceptionConfig:
    camera_index: int = 0
    yaw_facing: float = 0.18           # |nose offset from eye midpoint| / eye distance
    pitch_band: tuple = (0.10, 0.65)   # nose position between eye line (0) and chin (1); tune to your camera
    min_eye_px: float = 35.0           # smaller faces are too far away to count
    engage_after: float = 0.6          # seconds of facing before engaging
    disengage_after: float = 2.0       # seconds of visibly looking away before disengaging
    disengage_no_face_after: float = 6.0   # longer when the face is hidden: an object may be blocking it
    lost_after: float = 0.5            # seconds without a face before "face_lost" (only when not engaged)
    smoothing: float = 0.35            # EMA factor for face position


@dataclass
class FaceState:
    present: bool = False
    facing: bool = False
    engaged: bool = False
    x: float = 0.0            # face centre, -1 (image left) .. +1 (image right), camera view, not mirrored
    y: float = 0.0            # -1 (top) .. +1 (bottom)
    yaw_ratio: float = 0.0
    pitch_ratio: float = 0.0
    eye_px: float = 0.0
    fps: float = 0.0


def _ensure_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        print("[perception] downloading face landmarker model…")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def _measure(landmarks, w, h):
    """Return (eye_px, yaw_ratio, pitch_ratio, x, y) for one face."""
    def pt(i):
        return np.array([landmarks[i].x * w, landmarks[i].y * h])
    nose, ea, eb, chin = pt(NOSE), pt(EYE_OUTER_A), pt(EYE_OUTER_B), pt(CHIN)
    eye_mid = (ea + eb) / 2
    eye_px = float(np.linalg.norm(ea - eb))
    yaw_ratio = float((nose[0] - eye_mid[0]) / max(eye_px, 1e-6))
    pitch_ratio = float((nose[1] - eye_mid[1]) / max(chin[1] - eye_mid[1], 1e-6))
    x = landmarks[NOSE].x * 2 - 1
    y = landmarks[NOSE].y * 2 - 1
    return eye_px, yaw_ratio, pitch_ratio, x, y


class Perception:
    def __init__(self, cfg=None, show_debug=True):
        self.cfg = cfg or PerceptionConfig()
        self.show_debug = show_debug
        self.events = queue.Queue()
        self._state = FaceState()
        self._frame = None
        self._hold = False         # stay engaged regardless of the face (set while inspecting an object)
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

    @property
    def state(self):
        with self._lock:
            return replace(self._state)

    def hold_engagement(self, on):
        """While on, an engaged person stays engaged. The disengage timer restarts on release."""
        self._hold = on

    def latest_frame(self):
        """Copy of the most recent camera frame (BGR), or None."""
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def _emit(self, name, **info):
        self.events.put((name, time.monotonic(), info))

    def _run(self):
        _ensure_model()
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=2,
        )
        landmarker = vision.FaceLandmarker.create_from_options(options)
        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_V4L2  # DSHOW opens fast on Windows
        cap = cv2.VideoCapture(self.cfg.camera_index, backend)
        if not cap.isOpened():
            print("[perception] could not open camera")
            return

        cfg = self.cfg
        engaged = present = False
        facing_since = away_since = None
        last_seen = None
        sx = sy = 0.0
        last_ts = 0
        fps, last_frame = 0.0, time.monotonic()

        while self._running:
            ok, frame = cap.read()
            if not ok:
                continue
            now = time.monotonic()
            fps = 0.9 * fps + 0.1 / max(now - last_frame, 1e-3)
            last_frame = now

            ts = max(int(now * 1000), last_ts + 1)   # VIDEO mode needs strictly increasing timestamps
            last_ts = ts
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)

            h, w = frame.shape[:2]
            faces = [_measure(lms, w, h) for lms in result.face_landmarks]
            face = max(faces, key=lambda f: f[0]) if faces else None   # largest = closest person
            seen = face is not None and face[0] >= cfg.min_eye_px

            # ---- presence: appears immediately ----
            if seen:
                last_seen = now
                if not present:
                    present = True
                    self._emit("face_appeared")

            # ---- facing ----
            facing = False
            eye_px, yaw_r, pitch_r = (face[0], face[1], face[2]) if face else (0.0, 0.0, 0.0)
            if seen:
                _, _, _, x, y = face
                facing = abs(yaw_r) < cfg.yaw_facing and cfg.pitch_band[0] < pitch_r < cfg.pitch_band[1]
                sx += cfg.smoothing * (x - sx)
                sy += cfg.smoothing * (y - sy)

            # ---- engagement hysteresis (a missing face gets longer: it may just be blocked) ----
            if facing:
                away_since = None
                facing_since = facing_since or now
                if not engaged and now - facing_since >= cfg.engage_after:
                    engaged = True
                    self._emit("engaged", dwell=round(now - facing_since, 2))
            else:
                facing_since = None
                if engaged and self._hold:
                    away_since = None
                elif engaged:
                    away_since = away_since or now
                    limit = cfg.disengage_after if seen else cfg.disengage_no_face_after
                    if now - away_since >= limit:
                        engaged = False
                        away_since = None
                        self._emit("disengaged")

            # ---- presence lost: only once not engaged, so order is always
            #      engaged → disengaged → face_lost (a brief dropout can't skip a step) ----
            if present and not seen and not engaged and now - (last_seen or now) > cfg.lost_after:
                present = False
                self._emit("face_lost")

            with self._lock:
                self._state = FaceState(present, facing, engaged, sx, sy, yaw_r, pitch_r, eye_px, fps)
                self._frame = frame

            if self.show_debug:
                self._draw_debug(frame)

        cap.release()
        landmarker.close()
        if self.show_debug:
            cv2.destroyAllWindows()

    def _draw_debug(self, frame):
        s = self._state
        view = cv2.flip(frame, 1)   # mirrored for display only
        color = (0, 200, 0) if s.engaged else (0, 200, 255) if s.facing else (160, 160, 160)
        lines = [
            f"present={s.present} facing={s.facing} ENGAGED={s.engaged}",
            f"yaw_ratio={s.yaw_ratio:+.2f} (|x|<{self.cfg.yaw_facing})  "
            f"pitch_ratio={s.pitch_ratio:.2f} {self.cfg.pitch_band}",
            f"eye_px={s.eye_px:.0f} (min {self.cfg.min_eye_px:.0f})  fps={s.fps:.0f}",
        ]
        for i, line in enumerate(lines):
            cv2.putText(view, line, (10, 25 + 25 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.imshow("perception (debug)", view)
        cv2.waitKey(1)


if __name__ == "__main__":
    p = Perception()
    p.start()
    print("Look at the camera, then away. Ctrl+C to quit.")
    try:
        while True:
            name, t, info = p.events.get()
            print(f"[{t:9.2f}] {name} {info if info else ''}")
    except KeyboardInterrupt:
        p.stop()