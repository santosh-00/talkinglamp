"""Sound effects and music, synthesized in code (no audio files).

One output stream mixes three layers: a looping sleep lullaby, one-shot effects,
and a servo whir driven by how fast the joints are really moving.
The voice plays on its own stream, so nothing here can cut it off.
If no audio device is available the character carries on silently.
"""
import queue
import time

import numpy as np
import sounddevice as sd

SR = 44100
BLOCK = 1024

MUSIC_GAIN = 0.22
SFX_GAIN = 0.35
WHIR_GAIN = 0.05
MUSIC_FADE_S = 2.0

WHIR_DEADBAND = 0.12       # rad/s; below this (breathing, small gaze drift) the lamp is silent
WHIR_FULL_SPEED = 1.3      # rad/s above the deadband that counts as full speed
WHIR_HZ = (180.0, 500.0)   # pitch at no motion and at full speed


def _midi(n):
    return 440.0 * 2 ** ((n - 69) / 12)


def _note(freq, dur, attack=0.01, decay=0.5, overtone=0.35):
    """A soft bell-like note that starts and ends at zero (no clicks)."""
    t = np.arange(int(SR * dur)) / SR
    env = (1 - np.exp(-t / attack)) * np.exp(-t / decay)
    tone = np.sin(2 * np.pi * freq * t) + overtone * np.exp(-t / (decay / 3)) * np.sin(4 * np.pi * freq * t)
    taper = min(len(t), int(SR * 0.01))
    env[-taper:] *= np.linspace(1, 0, taper)
    return tone * env


def _sequence(notes):
    """Mix (start_s, samples) pairs into one peak-normalized sound."""
    out = np.zeros(max(int(s * SR) + len(x) for s, x in notes))
    for s, x in notes:
        i = int(s * SR)
        out[i:i + len(x)] += x
    return out / np.max(np.abs(out))


def make_happy():
    """Quick rising chime."""
    return _sequence([(0.09 * i, _note(_midi(n), 0.40, decay=0.12)) for i, n in enumerate((79, 84, 88, 91))])


def make_thinking():
    """Soft wandering bubbles, each one gliding upward slightly."""
    notes = []
    for i, n in enumerate((69, 74, 71, 76, 72)):
        t = np.arange(int(SR * 0.14)) / SR
        f = _midi(n) * (1 + 0.5 * t / t[-1])
        x = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.sin(np.pi * t / t[-1]) ** 2
        notes.append((0.15 * i, x))
    return _sequence(notes)


def make_lullaby():
    """A slow, original arpeggio loop. Note tails wrap around so the loop is seamless."""
    beat = 0.55
    chords = [(60, 64, 67, 72), (57, 60, 64, 69), (53, 57, 60, 65), (55, 59, 62, 67)] * 2
    arpeggio = (0, 1, 2, 3, 2, 1)
    melody = [76, 79, 72, 76, 72, 69, 74, 71, 76, 79, 81, 76, 77, 72, 74, 72]

    length = int(SR * beat * len(arpeggio) * len(chords))
    out = np.zeros(length)

    def add(start_s, x):
        out[(int(start_s * SR) + np.arange(len(x))) % length] += x

    for bar, chord in enumerate(chords):
        bar_t = bar * beat * len(arpeggio)
        add(bar_t, 0.5 * _note(_midi(chord[0] - 12), 4.0, attack=0.08, decay=1.6, overtone=0.1))
        for i, k in enumerate(arpeggio):
            add(bar_t + i * beat, 0.30 * _note(_midi(chord[k]), 2.2, attack=0.02, decay=0.7))
        add(bar_t, 0.40 * _note(_midi(melody[2 * bar]), 3.0, attack=0.05, decay=1.0))
        add(bar_t + 3 * beat, 0.40 * _note(_midi(melody[2 * bar + 1]), 3.0, attack=0.05, decay=1.0))
    return out / np.max(np.abs(out))


class Audio:
    def __init__(self):
        self.sounds = {"happy": make_happy(), "thinking": make_thinking()}
        self.music = make_lullaby()
        self._new = queue.SimpleQueue()
        self._playing = []              # [samples, position]; touched only by the audio callback
        self._music_on = False
        self._music_pos = 0
        self._music_level = 0.0
        self._motion_target = 0.0
        self._motion = 0.0
        self._motion_level = 1.0
        self._whir_gain = 0.0
        self._whir_hz = WHIR_HZ[0]
        self._phase = 0.0
        self._stream = None

    # ---------- control (safe from any thread) ----------
    def start(self):
        try:
            self._stream = sd.OutputStream(samplerate=SR, channels=1, dtype="float32", blocksize=BLOCK,
                                           latency="high", callback=self._callback)
            self._stream.start()
        except Exception as e:
            self._stream = None
            print(f"[audio] no output stream ({e}); continuing without SFX and music")

    def stop(self):
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def play(self, name):
        self._new.put([self.sounds[name], 0])

    def set_music(self, on):
        self._music_on = on

    def set_motion(self, speed):
        """Feed the fastest joint speed (rad/s)."""
        self._motion_target = min(max(speed - WHIR_DEADBAND, 0.0) / WHIR_FULL_SPEED, 1.0)

    def set_motion_level(self, level):
        """Scale the whir: lower under the voice, 0 while the microphone is capturing."""
        self._motion_level = level

    # ---------- audio thread ----------
    def _callback(self, out, frames, t, status):
        mix = np.zeros(frames)

        # Music, with a gain ramp for fades
        step = frames / (SR * MUSIC_FADE_S)
        target = 1.0 if self._music_on else 0.0
        new_level = min(self._music_level + step, target) if target > self._music_level \
            else max(self._music_level - step, target)
        if new_level > 0 or self._music_level > 0:
            idx = (self._music_pos + np.arange(frames)) % len(self.music)
            mix += self.music[idx] * np.linspace(self._music_level, new_level, frames) * MUSIC_GAIN
            self._music_pos = (self._music_pos + frames) % len(self.music)
        self._music_level = new_level

        # One-shot effects
        while not self._new.empty():
            self._playing.append(self._new.get())
        alive = []
        for item in self._playing:
            sound, pos = item
            chunk = sound[pos:pos + frames]
            mix[:len(chunk)] += chunk * SFX_GAIN
            item[1] = pos + frames
            if item[1] < len(sound):
                alive.append(item)
        self._playing = alive

        # Servo whir: pitch and loudness follow joint speed
        self._motion += (self._motion_target - self._motion) * 0.25
        hz = WHIR_HZ[0] + (WHIR_HZ[1] - WHIR_HZ[0]) * self._motion
        gain = WHIR_GAIN * min(self._motion * 3, 1.0) * self._motion_level
        if gain > 1e-4 or self._whir_gain > 1e-4:
            phases = self._phase + 2 * np.pi * np.cumsum(np.linspace(self._whir_hz, hz, frames)) / SR
            wave = 0.6 * np.sin(phases) + 0.3 * np.sin(2 * phases) + 0.1 * np.sin(3.01 * phases)
            mix += wave * np.linspace(self._whir_gain, gain, frames)
            self._phase = float(phases[-1] % (2 * np.pi))
        self._whir_hz, self._whir_gain = hz, gain

        out[:, 0] = np.clip(mix, -1.0, 1.0)


if __name__ == "__main__":
    a = Audio()
    a.start()
    print("happy")
    a.play("happy")
    time.sleep(1.2)
    print("thinking")
    a.play("thinking")
    time.sleep(1.5)
    print("movement whir: slow, then fast")
    for speed in (0.4, 0.8, 1.4, 0.8, 0.0):
        a.set_motion(speed)
        time.sleep(0.7)
    print("lullaby for 15 s (Ctrl+C to stop)")
    a.set_music(True)
    try:
        time.sleep(15)
        a.set_music(False)
        time.sleep(MUSIC_FADE_S + 0.3)
    except KeyboardInterrupt:
        pass
    a.stop()
