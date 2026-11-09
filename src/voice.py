"""Text-to-speech, local. Piper (neural, offline) with Windows SAPI fallback.

Character voice: a small, young-sounding lamp. Piper has no child voices, so we
pitch-shift: Piper synthesizes slightly slower (length_scale), and we play the
audio back at a higher sample rate. Pitch goes up; speaking speed stays natural.
"""
import os
import sys
import time

import numpy as np
import sounddevice as sd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_VOICE = os.path.join(ROOT, "models", "voices", "en_US-lessac-medium.onnx")

PITCH = 1.25        # 1.0 = original; 1.2–1.35 = younger/cuter; >1.4 gets chipmunky
SPEED = 1.05        # >1 slightly faster/bouncier delivery after pitch compensation


class Voice:
    def __init__(self, voice_path=DEFAULT_VOICE, pitch=PITCH, speed=SPEED):
        self.pitch = pitch
        self.speed = speed
        self.backend = None
        self.syn_config = None
        try:
            from piper import PiperVoice
            self.piper = PiperVoice.load(voice_path)
            self.backend = "piper"
            try:
                from piper import SynthesisConfig
                # Synthesize slower by `pitch` so the faster playback lands back at normal speed
                self.syn_config = SynthesisConfig(length_scale=pitch / speed)
            except Exception as e:
                print(f"[voice] SynthesisConfig unavailable ({e}); voice will be pitched AND faster")
        except Exception as e:
            print(f"[voice] Piper unavailable ({e}); falling back to Windows voice")
            self.backend = "sapi"
        print(f"[voice] backend: {self.backend}  pitch: {self.pitch}")

    def speak(self, text):
        """Blocking. Returns synth_ms (time until audio starts playing)."""
        t0 = time.monotonic()
        if self.backend == "piper":
            if self.syn_config is not None:
                chunks = list(self.piper.synthesize(text, syn_config=self.syn_config))
            else:
                chunks = list(self.piper.synthesize(text))
            audio = np.concatenate([c.audio_int16_array for c in chunks])
            sr = chunks[0].sample_rate
            synth_ms = (time.monotonic() - t0) * 1000
            sd.play(audio, int(sr * self.pitch))     # higher playback rate = higher pitch
            sd.wait()
        else:
            import pyttsx3
            engine = pyttsx3.init()                  # created per call: SAPI is picky about threads
            synth_ms = (time.monotonic() - t0) * 1000
            voices = engine.getProperty("voices")
            female = [v for v in voices if "zira" in v.name.lower() or "female" in v.name.lower()]
            if female:
                engine.setProperty("voice", female[0].id)
            engine.setProperty("rate", 190)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        return synth_ms


if __name__ == "__main__":
    args = sys.argv[1:]
    pitch = PITCH
    if "--pitch" in args:
        i = args.index("--pitch")
        pitch = float(args[i + 1])
        args = args[:i] + args[i + 2:]
    text = " ".join(args) or "Oh, hello there! I'm Lumen, your little desk lamp. What's that you've got?"
    v = Voice(pitch=pitch)
    print(f"synth: {v.speak(text):.0f} ms")