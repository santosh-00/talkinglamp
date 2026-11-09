"""Character: the one place that decides behaviour.

Inputs:  perception state (absent/present/engaged), listener events, LLM replies, scene memory.
Output:  one coherent expression across motion, light and voice.

Pattern: the main thread reconciles the body against (mode, phase) every tick.
Slow work (LLM, VLM, TTS) runs in a worker thread that only sets phase/emotion/gesture,
so the simulated body is only ever touched from the main thread.
"""
import csv
import os
import random
import threading
import time

from body import COOL, WARM
from memory import is_seek_request, is_show_request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TURN_LOG = os.path.join(ROOT, "logs", "turns.csv")

EMOTION_LIGHT = {   # emotion → (colour, intensity)
    "happy":      ((1.00, 0.78, 0.40), 1.0),
    "curious":    ((0.75, 1.00, 0.65), 0.95),
    "thoughtful": (COOL, 0.8),
    "surprised":  ((1.00, 1.00, 1.00), 1.0),
    "sad":        ((0.45, 0.55, 0.95), 0.5),
    "neutral":    (WARM, 0.9),
}
INSPECT_LIGHT = ((1.0, 1.0, 0.95), 1.0)     # bright, neutral "let me see" light

GREETINGS = ["Oh! Hi there!", "Hello! I was hoping someone would stop by.", "Oh, hey! Nice to see you."]
WELCOME_BACK = ["Oh, you're back!", "Welcome back!"]
LOOK_FILLERS = ["Ooh, let me see!", "Ooh, what's that? Let me look!", "Hold it still, let me see!"]
CANT_SEE = "Hmm, I couldn't quite make it out. Can you hold it a bit closer?"
SEEK_FILLERS = ["Ooh, a hunt! Let me look around.", "Something new? I'm on it!"]
SEEK_LOOK_AGAIN = "Let me look again!"
SEEK_CONFIRM = "Wait, let me make sure."
SEEK_LOST = "Oh, where did it go? Let me look again!"
SEEK_GIVE_UP = "I can't spot anything new right now. Show me something later?"
SEEK_FAILED = "Hmm, my eyes went a little fuzzy. Can we try that again?"

GAZE_EXPRESSIONS = {"engaged", "listening", "speaking"}
LOOK_DOWN_AT_OBJECT = (0.0, -0.15)          # (yaw, pitch) while inspecting: straight ahead, slightly down
WHIR_LEVEL = {"speaking": 0.4, "looking": 0.4, "listening": 0.0}   # keep motor sound out of speech and the mic

YAW_GAIN, PITCH_GAIN, YAW_SIGN = 0.6, 0.35, -1
REGREET_AFTER_S = 60.0      # gone this long → full greeting again
WELCOME_BACK_AFTER_S = 8.0  # gone this long → "welcome back"; shorter → just perk up
CAPTURE_DELAY_S = 0.4       # let the person finish raising the object before grabbing the frame
MAX_SEEK_ROUNDS = 4         # observe-act attempts before giving up on a goal
MAX_SEEK_ACTIONS = 4        # longest action sequence accepted from the model
SEEK_WAIT_S = 3.0           # time for the person to bring something out before looking again
ACTION_S = 0.9              # time given to each lamp action
SCAN_YAW = 0.7              # how far a scan turns (rad); + is the lamp's left
SEEK_LIGHT = ((0.75, 1.00, 0.65), 0.9)      # curious green while hunting


class Character:
    def __init__(self, body, perception, listener, brain, voice, memory=None, audio=None):
        self.body, self.per, self.listener = body, perception, listener
        self.brain, self.voice, self.memory, self.audio = brain, voice, memory, audio
        self.mode = None            # absent | present | engaged   (from perception)
        self.phase = "idle"         # idle | listening | thinking | looking | speaking   (from conversation)
        self.emotion = "neutral"
        self._gesture = None        # one-shot gesture requested by the worker
        self._aim = (0.0, 0.0)      # (yaw, pitch) held during the seeking and spotting phases
        self._busy = False          # a greeting or turn is running
        self._lock = threading.Lock()
        self._expr = self._expr_emotion = None
        self._last_engaged_end = None
        os.makedirs(os.path.dirname(TURN_LOG), exist_ok=True)

    # ================= main thread =================
    def tick(self, dt):
        while not self.per.events.empty():          # perception events: logging only
            name, _, info = self.per.events.get_nowait()
            print(f"[perception] {name} {info or ''}")

        s = self.per.state
        mode = "engaged" if s.engaged else "present" if s.present else "absent"
        if mode != self.mode:
            self._on_mode_change(self.mode, mode)
            self.mode = mode

        self._poll_listener()

        with self._lock:
            phase, emotion, gesture, aim = self.phase, self.emotion, self._gesture, self._aim
            self._gesture = None

        expr = phase if phase != "idle" else mode
        if expr != self._expr or (expr == "speaking" and emotion != self._expr_emotion):
            self._express(expr, emotion, self._expr)
            self._express_audio(expr, emotion)
            self._expr, self._expr_emotion = expr, emotion
        if gesture:
            self.body.gesture(gesture)

        if expr == "looking":
            self.body.look_at(*LOOK_DOWN_AT_OBJECT)
        elif expr in ("seeking", "spotting"):
            self.body.look_at(*aim)
        elif expr in GAZE_EXPRESSIONS:
            if s.present:
                self.body.look_at(YAW_SIGN * s.x * YAW_GAIN, -s.y * PITCH_GAIN)
        else:
            self.body.clear_gaze()

        self.body.update(dt)
        if self.audio:
            self.audio.set_motion(self.body.speed)

    def _on_mode_change(self, prev, new):
        print(f"[character] mode {prev} → {new}")
        if new == "engaged":
            self.listener.enable(True)
            gap = None if self._last_engaged_end is None else time.monotonic() - self._last_engaged_end
            if gap is None or gap > REGREET_AFTER_S:
                self._start_job(self._say_line, random.choice(GREETINGS), "happy")
            elif gap > WELCOME_BACK_AFTER_S:
                self._start_job(self._say_line, random.choice(WELCOME_BACK), "happy")
            # brief glance away: no speech, the perk on re-engaging is enough
        elif prev == "engaged":
            self.listener.enable(False)
            self._last_engaged_end = time.monotonic()

    def _poll_listener(self):
        while not self.listener.events.empty():
            name, t, info = self.listener.events.get_nowait()
            if name == "speech_start":
                if self.mode == "engaged" and not self._busy:
                    self._set(phase="listening")
            elif name == "discarded":
                with self._lock:
                    if self.phase == "listening":
                        self.phase = "idle"
            elif name == "utterance":
                # When the person actually stopped talking: before endpointing silence and STT
                t_speech_end = t - info["stt_ms"] / 1000 - self.listener.cfg.end_silence_ms / 1000
                self._start_job(self._turn, info["text"], t_speech_end, info["stt_ms"])

    def _express(self, expr, emotion, prev):
        """Map an expression to pose + light (+ entrance gesture)."""
        b = self.body
        if expr == "absent":
            b.set_pose("sleep", 0.6)
            b.set_light(WARM, 0.15, 0.0)
        elif expr == "present":
            if prev in GAZE_EXPRESSIONS or prev in ("thinking", "looking", "seeking", "spotting"):
                b.gesture("droop")
            b.set_pose("idle", 0.7)
            b.set_light(WARM, 0.35, 0.0)
        elif expr == "engaged":
            if prev in (None, "absent", "present"):
                b.gesture("perk")
            b.set_pose("attentive", 1.3)
            b.set_light(WARM, 1.0, 0.0)
        elif expr == "listening":
            b.set_pose("lean_in", 1.0)
            b.set_light(WARM, 1.0, 0.0)
        elif expr == "thinking":
            b.set_pose("thinking", 0.9)
            b.set_light(COOL, 0.8, 1.0)
        elif expr in ("looking", "spotting"):
            b.set_pose("lean_in", 1.2)
            b.set_light(*INSPECT_LIGHT, 0.0)
        elif expr == "seeking":
            b.set_pose("attentive", 1.1)
            b.set_light(*SEEK_LIGHT, 0.6)
        elif expr == "speaking":
            color, intensity = EMOTION_LIGHT.get(emotion, EMOTION_LIGHT["neutral"])
            b.set_pose("attentive", 1.2)
            b.set_light(color, intensity, 0.0)

    def _express_audio(self, expr, emotion):
        """Music while asleep, SFX on entering a state, whir kept out of the way of speech."""
        if not self.audio:
            return
        self.audio.set_music(expr == "absent")
        self.audio.set_motion_level(WHIR_LEVEL.get(expr, 1.0))
        if expr == "thinking":
            self.audio.play("thinking")
        elif expr == "speaking" and emotion == "happy":
            self.audio.play("happy")

    # ================= worker thread =================
    def _set(self, phase=None, emotion=None, gesture=None, aim=None):
        with self._lock:
            if aim is not None:
                self._aim = aim
            if phase:
                self.phase = phase
            if emotion:
                self.emotion = emotion
            if gesture and gesture != "none":
                self._gesture = gesture

    def _start_job(self, fn, *args):
        if self._busy:
            return False
        self._busy = True

        def run():
            self.listener.mute(True)          # half-duplex: never hear ourselves
            try:
                fn(*args)
            except Exception as e:
                print(f"[character] job failed: {e}")
            finally:
                self._set(phase="idle")
                self.listener.mute(False)
                self._busy = False

        threading.Thread(target=run, daemon=True).start()
        return True

    def _say_line(self, line, emotion):
        self._set(phase="speaking", emotion=emotion)
        print(f"lumen: {line}")
        self.voice.speak(line)

    def _turn(self, text, t_speech_end, stt_ms):
        print(f"you: {text}")
        if self.memory and is_seek_request(text):
            self.per.hold_engagement(True)      # the person may step aside to fetch something
            try:
                self._seek_turn(text, t_speech_end, stt_ms)
            finally:
                self.per.hold_engagement(False)
            return
        if self.memory and is_show_request(text):
            self.per.hold_engagement(True)      # the object may be hiding the person's face
            try:
                self._look_turn(text, t_speech_end, stt_ms)
            finally:
                self.per.hold_engagement(False)
            return

        self._set(phase="thinking")
        notes = self.memory.notes() if self.memory else ""
        reply, llm_ms = self.brain.reply(text, scene_notes=notes)
        print(f"lumen [{reply.emotion}/{reply.gesture}]: {reply.say}")

        self._set(phase="speaking", emotion=reply.emotion, gesture=reply.gesture)
        t_speak = time.monotonic()
        tts_ms = self.voice.speak(reply.say)                 # blocks until finished
        total_ms = (t_speak + tts_ms / 1000 - t_speech_end) * 1000
        print(f"  latency: endpoint {self.listener.cfg.end_silence_ms} + stt {stt_ms:.0f} + "
              f"llm {llm_ms:.0f} + tts {tts_ms:.0f} → {total_ms:.0f} ms (stopped talking → voice)")
        self._log_turn("talk", stt_ms, llm_ms, tts_ms, total_ms, text, reply.say, reply.emotion, reply.gesture)

    def _look_turn(self, text, t_speech_end, stt_ms):
        """Person is showing something: inspect, describe via VLM, react, remember."""
        self._set(phase="looking", gesture="tilt")
        result = {}

        def observe():
            time.sleep(CAPTURE_DELAY_S)
            look, ms = self.memory.observe(self.per.latest_frame(), text)
            result.update(look=look, ms=ms)

        worker = threading.Thread(target=observe, daemon=True)
        worker.start()
        t_filler = time.monotonic()
        self.voice.speak(random.choice(LOOK_FILLERS))      # spoken while the VLM runs: hides latency
        filler_first_audio_ms = (time.monotonic() - t_filler) * 1000
        worker.join()

        look, vlm_ms = result.get("look"), result.get("ms", 0.0)
        if look and look.objects:
            say, emotion, gesture = look.comment, look.emotion, look.gesture
        else:
            say, emotion, gesture = CANT_SEE, "curious", "tilt"
        print(f"lumen [{emotion}/{gesture}]: {say}")
        self.brain.history += [("user", text), ("model", say)]   # keep the conversation coherent

        self._set(phase="speaking", emotion=emotion, gesture=gesture)
        t_speak = time.monotonic()
        tts_ms = self.voice.speak(say)
        total_ms = (t_speak + tts_ms / 1000 - t_speech_end) * 1000
        print(f"  look latency: vlm {vlm_ms:.0f} ms (masked by filler), "
              f"reaction voice at {total_ms:.0f} ms after you stopped talking")
        self._log_turn("look", stt_ms, vlm_ms, tts_ms, total_ms, text, say, emotion, gesture)

    def _speak_while(self, line, fn):
        """Say a filler line while fn runs in the background; return fn's result (None if it failed)."""
        out = {}
        worker = threading.Thread(target=lambda: out.update(result=fn()), daemon=True)
        worker.start()
        self.voice.speak(line)
        worker.join()
        return out.get("result")

    def _run_actions(self, actions, aim):
        """Execute the model's action list. The model chooses; the body's limits still apply."""
        scans = {"scan_left": SCAN_YAW, "scan_right": -SCAN_YAW, "scan_center": 0.0}
        for action in actions:
            if action in scans:
                self._set(phase="seeking", aim=(scans[action], 0.05))
            elif action == "aim_at_target":
                if aim:
                    self._set(phase="seeking", aim=aim)
            elif action == "spotlight_on":
                if aim:
                    self._set(phase="spotting", aim=aim)
            elif action == "spotlight_off":
                self._set(phase="seeking")
            else:
                self._set(gesture=action)
            time.sleep(ACTION_S)

    def _seek_turn(self, text, t_speech_end, stt_ms):
        """Goal: find something new. Observe, plan actions, act, then re-observe before finishing."""
        say, emotion, gesture = SEEK_GIVE_UP, "sad", "droop"
        model_ms = 0.0
        line = random.choice(SEEK_FILLERS)

        for attempt in range(1, MAX_SEEK_ROUNDS + 1):
            self._set(phase="seeking", emotion="curious", aim=(0.0, 0.05))
            planned = self._speak_while(line, lambda: self.memory.plan_seek(self.per.latest_frame(), text))
            plan, ms = planned or (None, 0.0)
            model_ms += ms
            if plan is None:
                say, emotion, gesture = SEEK_FAILED, "thoughtful", "tilt"
                break

            actions = list(plan.actions[:MAX_SEEK_ACTIONS])
            aim = None
            if plan.found_new:
                x, y = max(0, min(plan.x, 1000)) / 500 - 1, max(0, min(plan.y, 1000)) / 500 - 1
                aim = (YAW_SIGN * x * YAW_GAIN, -y * PITCH_GAIN)
                actions = ["aim_at_target"] * ("aim_at_target" not in actions) + actions
                actions += ["spotlight_on"] * ("spotlight_on" not in actions)
            print(f"[goal] round {attempt}: found_new={plan.found_new} "
                  f"target={plan.target.name!r} at ({plan.x}, {plan.y}) actions={actions} [{ms:.0f} ms]")
            self._run_actions(actions, aim)

            if not plan.found_new:
                print(f"lumen [{plan.emotion}]: {plan.say}")
                self._set(phase="speaking", emotion=plan.emotion)
                self.voice.speak(plan.say)
                self._set(phase="seeking", aim=(0.0, 0.05))
                time.sleep(SEEK_WAIT_S)
                line = SEEK_LOOK_AGAIN
                continue

            checked = self._speak_while(SEEK_CONFIRM, lambda: self.memory.check_seek(self.per.latest_frame(), plan.target))
            still_there, ms, small = checked or (None, 0.0, None)
            model_ms += ms
            print(f"[goal] re-observed {plan.target.name!r}: still_there={still_there} [{ms:.0f} ms]")
            if still_there:
                self.memory.remember([plan.target], small)
                say, emotion, gesture = plan.say, plan.emotion, "bounce"
                break
            line = SEEK_LOST

        print(f"lumen [{emotion}/{gesture}]: {say}")
        self.brain.history += [("user", text), ("model", say)]
        self._set(phase="speaking", emotion=emotion, gesture=gesture)
        t_speak = time.monotonic()
        tts_ms = self.voice.speak(say)
        total_ms = (t_speak + tts_ms / 1000 - t_speech_end) * 1000
        self._log_turn("seek", stt_ms, model_ms, tts_ms, total_ms, text, say, emotion, gesture)

    def _log_turn(self, kind, stt_ms, model_ms, tts_ms, total_ms, text, say, emotion, gesture):
        new = not os.path.exists(TURN_LOG)
        with open(TURN_LOG, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["time", "kind", "endpoint_ms", "stt_ms", "model_ms", "tts_ms", "total_ms",
                            "user", "reply", "emotion", "gesture"])
            w.writerow([time.strftime("%H:%M:%S"), kind, self.listener.cfg.end_silence_ms, round(stt_ms),
                        round(model_ms), round(tts_ms), round(total_ms), text, say, emotion, gesture])
