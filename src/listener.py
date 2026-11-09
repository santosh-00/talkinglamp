"""Microphone listener: VAD-segmented utterances → local Whisper transcription.

Events (queue of (name, monotonic_time, info)):
  speech_start                       person started talking
  utterance  {text, stt_ms, audio_s} final transcript
  discarded  {reason}                too short / empty / hallucination
Half-duplex: call mute(True) while the robot is speaking so it never hears itself.
Audio stays on the machine.

Calibrate the loudness gate with:  python src/listener.py --levels
"""
import os
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")   # must be set before HF imports
import sys

import collections
import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import sounddevice as sd
import webrtcvad
from faster_whisper import WhisperModel

SR = 16000
FRAME_MS = 30
FRAME = SR * FRAME_MS // 1000          # 480 samples; webrtcvad needs exactly 10/20/30 ms

# Whisper sometimes "hears" these in near-silence
HALLUCINATIONS = {"", "you", "thank you.", "thanks for watching!", "thank you for watching.", "bye."}


@dataclass
class ListenerConfig:
    model_size: str = "base.en"
    device: str = "cpu"                # "cuda" only if CUDA 12 + cuDNN are set up
    compute_type: str = "int8"
    vad_aggressiveness: int = 3        # 0 (lenient) .. 3 (strictest)
    start_frames: int = 6              # 180 ms of voice to start an utterance
    end_silence_ms: int = 700          # silence that ends an utterance
    preroll_ms: int = 300              # keep audio from just before speech started
    max_utterance_s: float = 12.0
    min_utterance_s: float = 0.4
    min_rms: float = 350.0             # loudness gate; tune with --levels
    mic_device: Optional[int] = None   # None = system default


class Listener:
    def __init__(self, cfg=None):
        self.cfg = cfg or ListenerConfig()
        self.events = queue.Queue()
        self._frames = queue.Queue()
        self._enabled = threading.Event()
        self._muted = threading.Event()
        self._running = False
        self._stream = None
        self._thread = None
        print(f"[listener] loading whisper {self.cfg.model_size} on {self.cfg.device}…")
        self.model = WhisperModel(self.cfg.model_size, device=self.cfg.device,
                                  compute_type=self.cfg.compute_type)
        self.vad = webrtcvad.Vad(self.cfg.vad_aggressiveness)

    # ---------- control ----------
    def enable(self, on=True):
        """Only listen while the person is engaged."""
        (self._enabled.set if on else self._enabled.clear)()

    def mute(self, on=True):
        """Half-duplex: mute while the robot speaks; drop anything captured meanwhile."""
        if on:
            self._muted.set()
        else:
            self._drain()
            self._muted.clear()

    def start(self):
        self._running = True
        self._stream = sd.InputStream(samplerate=SR, channels=1, dtype="int16",
                                      blocksize=FRAME, device=self.cfg.mic_device,
                                      callback=self._on_audio)
        self._stream.start()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()

    # ---------- internals ----------
    def _on_audio(self, indata, frames, t, status):
        self._frames.put(bytes(indata))

    def _drain(self):
        while not self._frames.empty():
            try:
                self._frames.get_nowait()
            except queue.Empty:
                break

    def _emit(self, name, **info):
        self.events.put((name, time.monotonic(), info))

    def _run(self):
        cfg = self.cfg
        preroll = collections.deque(maxlen=cfg.preroll_ms // FRAME_MS)
        in_speech, voiced_run, silence_ms, buf = False, 0, 0, []

        while self._running:
            try:
                frame = self._frames.get(timeout=0.1)
            except queue.Empty:
                continue

            if not self._enabled.is_set() or self._muted.is_set():
                in_speech, voiced_run, silence_ms, buf = False, 0, 0, []
                preroll.clear()
                continue

            # A frame counts as speech only if it is loud enough AND voice-like
            samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
            loud = float(np.sqrt(np.mean(samples * samples))) > cfg.min_rms
            speech = loud and self.vad.is_speech(frame, SR)

            if not in_speech:
                preroll.append(frame)
                voiced_run = voiced_run + 1 if speech else 0
                if voiced_run >= cfg.start_frames:
                    in_speech, silence_ms, buf = True, 0, list(preroll)
                    self._emit("speech_start")
            else:
                buf.append(frame)
                silence_ms = 0 if speech else silence_ms + FRAME_MS
                duration = len(buf) * FRAME_MS / 1000
                if silence_ms >= cfg.end_silence_ms or duration >= cfg.max_utterance_s:
                    in_speech, voiced_run = False, 0
                    preroll.clear()
                    self._transcribe(buf)
                    buf = []

    def _transcribe(self, frames):
        audio_s = len(frames) * FRAME_MS / 1000
        if audio_s < self.cfg.min_utterance_s:
            self._emit("discarded", reason="too short")
            return
        audio = np.frombuffer(b"".join(frames), dtype=np.int16).astype(np.float32) / 32768.0
        t0 = time.monotonic()
        segments, _ = self.model.transcribe(audio, language="en", beam_size=1,
                                            condition_on_previous_text=False)
        text = " ".join(s.text.strip() for s in segments).strip()
        stt_ms = (time.monotonic() - t0) * 1000
        if text.lower() in HALLUCINATIONS:
            self._emit("discarded", reason=f"empty/hallucination: {text!r}")
            return
        self._emit("utterance", text=text, stt_ms=round(stt_ms), audio_s=round(audio_s, 2))


def show_levels():
    """Calibrate min_rms: one summary line per second."""
    with sd.InputStream(samplerate=SR, channels=1, dtype="int16", blocksize=FRAME) as s:
        print("Stay quiet for ~5 s, then talk normally for ~5 s. Ctrl+C to stop.\n")
        print("  quiet   median    loud")
        try:
            while True:
                levels = []
                for _ in range(1000 // FRAME_MS):          # ~1 second of frames
                    x = s.read(FRAME)[0].astype(np.float32)
                    levels.append(float(np.sqrt(np.mean(x * x))))
                lo, med, hi = np.percentile(levels, [10, 50, 90])
                print(f"{lo:7.0f} {med:8.0f} {hi:7.0f}  " + "#" * int(hi / 100))
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    if "--levels" in sys.argv:
        show_levels()
        sys.exit()

    lis = Listener()
    lis.start()
    lis.enable(True)
    print("Speak a sentence, pause, repeat. Ctrl+C to quit.")
    try:
        while True:
            name, t, info = lis.events.get()
            print(f"[{t:9.2f}] {name} {info if info else ''}")
    except KeyboardInterrupt:
        lis.stop()