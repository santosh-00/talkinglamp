"""Scene memory: look once (cloud VLM on one keyframe), remember as text, recall cheaply.

Data sent to the cloud: one resized JPEG per look, plus what the person said.
What is stored: structured text per object + a local snapshot for debugging.
Recall never re-sends images: notes are injected into the LLM prompt as text.
"""
import os
import time
from dataclasses import dataclass
from typing import List, Literal

import cv2
from google import genai
from google.genai import types
from pydantic import BaseModel

from brain import ROOT, Emotion, Gesture   # brain.py also loads .env

IMG_DIR = os.path.join(ROOT, "logs", "memory")
MAX_SIDE = 768

# Spoken cues that mean "look at what I'm showing you" (matched locally, no LLM call)
SHOW_CUES = (
    "look at this", "look at that", "check this out", "check it out", "see this",
    "what is this", "what's this", "whats this", "what do you think of this",
    "remember this", "i'm holding", "i am holding", "showing you this", "show you this",
    "let me show you", "what's in my hand", "what is in my hand", "whats in my hand",
    "anything in my hand", "in my hand now", "what do you see", "what can you see",
)

# Spoken cues that set the goal "find something you haven't seen yet"
SEEK_CUES = (
    "find something", "find anything", "look for something", "look for anything",
    "look around for", "search for something", "see anything new", "spot something",
)

# The action vocabulary: the model may only pick from these; character.py and body.py execute them
LampAction = Literal["scan_left", "scan_right", "scan_center", "aim_at_target", "spotlight_on",
                     "spotlight_off", "nod", "tilt", "perk", "bounce"]


class SeenObject(BaseModel):
    name: str
    color: str
    material: str
    text_on_it: str                 # "" if none
    notable_features: List[str]


class Look(BaseModel):
    objects: List[SeenObject]       # main object first, at most 3
    comment: str                    # one short spoken reaction, in character
    emotion: Emotion
    gesture: Gesture


LOOK_PROMPT = """You are Lumen, a small, young, playful desk-lamp robot looking through your camera.
The person just said: "{said}"

Identify the object(s) the person is showing or holding up. Ignore the person, their face, and the background.
List the main object first (at most 3 objects). Be specific and visual: exact colours, material,
any readable text or logos, and distinctive marks or features.
Then write `comment`: ONE short, excited spoken sentence reacting to the main object. No emojis."""


class SeekPlan(BaseModel):
    found_new: bool
    target: SeenObject              # blank fields if found_new is false
    x: int                          # target centre in the image, 0 (left) .. 1000 (right)
    y: int                          # target centre in the image, 0 (top) .. 1000 (bottom)
    actions: List[LampAction]       # at most 4, in order
    say: str
    emotion: Emotion


class SeekCheck(BaseModel):
    still_there: bool


SEEK_PROMPT = """You are Lumen, a small, curious desk-lamp robot looking through your camera.
Your goal, from the person: "{goal}"

Things you have already seen:
{notes}

Look for ONE small, portable object in this image that is NOT in that list, preferring anything held up
or placed in clear view. Ignore people, faces, clothing, furniture, walls and the room itself.

If you find one: set found_new true, describe it in `target`, and give its centre as x and y from 0 to 1000
(x: 0 is the left edge of this image, y: 0 is the top edge). Choose `actions` to point at it, for example
scan_left, aim_at_target, spotlight_on, bounce. `say` is ONE short excited sentence naming it.

If there is nothing new: set found_new false, leave `target` fields blank, x and y 500. Choose scanning
`actions`. `say` is ONE short sentence asking the person to show you something new.

Use at most 4 actions. No emojis in `say`."""

SEEK_CHECK_PROMPT = """Is this object clearly visible in the image: "{name}"? Answer with still_there."""


@dataclass
class Observation:
    t: float                    # time.monotonic() when seen
    wall: str                   # wall-clock time, for logs
    objects: List[SeenObject]
    image_path: str


def is_seek_request(text):
    t = text.lower()
    return any(cue in t for cue in SEEK_CUES)


def is_show_request(text):
    t = text.lower()
    return any(cue in t for cue in SHOW_CUES)


def _ago(seconds):
    if seconds < 20:
        return "just now"
    if seconds < 90:
        return f"about {int(round(seconds / 10) * 10)} seconds ago"
    minutes = round(seconds / 60)
    if minutes < 60:
        return f"about {minutes} minute{'s' if minutes != 1 else ''} ago"
    hours = round(minutes / 60)
    return f"about {hours} hour{'s' if hours != 1 else ''} ago"


def _describe(o: SeenObject):
    s = f"{o.name} ({o.color}, {o.material})"
    if o.text_on_it:
        s += f", text on it: '{o.text_on_it}'"
    if o.notable_features:
        s += ", " + ", ".join(o.notable_features)
    return s


def _to_jpeg(frame_bgr):
    h, w = frame_bgr.shape[:2]
    scale = min(1.0, MAX_SIDE / max(h, w))
    if scale < 1.0:
        frame_bgr = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)))
    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buf.tobytes(), frame_bgr


class SceneMemory:
    def __init__(self, model=None):
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.models = [model or os.getenv("GEMINI_MODEL", "gemini-flash-latest")]
        if os.getenv("GEMINI_FALLBACK_MODEL"):
            self.models.append(os.getenv("GEMINI_FALLBACK_MODEL"))
        self.observations: List[Observation] = []
        os.makedirs(IMG_DIR, exist_ok=True)

    def _ask(self, frame_bgr, prompt, schema, temperature, max_tokens):
        """Send one frame and a prompt. Returns (parsed or None, vlm_ms, resized frame). Never raises."""
        if frame_bgr is None:
            return None, 0.0, None
        jpeg, small = _to_jpeg(frame_bgr)
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        contents = [types.Part.from_bytes(data=jpeg, mime_type="image/jpeg"), prompt]

        t0 = time.monotonic()
        result = None
        for model in self.models + self.models[:1]:       # primary, fallback, primary again
            try:
                resp = self.client.models.generate_content(model=model, contents=contents, config=config)
                result = resp.parsed if isinstance(resp.parsed, schema) else schema.model_validate_json(resp.text)
                break
            except Exception as e:
                print(f"[memory] {model} failed ({str(e)[:80]})")
                time.sleep(0.5)
        return result, (time.monotonic() - t0) * 1000, small

    def remember(self, objects, small_frame):
        """Store objects as text, plus a local snapshot of the frame they were seen in."""
        path = os.path.join(IMG_DIR, time.strftime("seen_%H%M%S.jpg"))
        cv2.imwrite(path, small_frame)
        self.observations.append(Observation(time.monotonic(), time.strftime("%H:%M:%S"), objects, path))
        print(f"[memory] stored: {'; '.join(_describe(o) for o in objects)}")

    def observe(self, frame_bgr, said=""):
        """Describe what the person is showing. Returns (Look or None, vlm_ms). Never raises."""
        look, vlm_ms, small = self._ask(frame_bgr, LOOK_PROMPT.format(said=said), Look, 0.4, 400)
        if look and look.objects:
            self.remember(look.objects, small)
        return look, vlm_ms

    def plan_seek(self, frame_bgr, goal):
        """Compare the live scene with memory and choose lamp actions. Returns (SeekPlan or None, vlm_ms)."""
        prompt = SEEK_PROMPT.format(goal=goal, notes=self.notes() or "(nothing yet)")
        plan, vlm_ms, _ = self._ask(frame_bgr, prompt, SeekPlan, 0.3, 500)
        return plan, vlm_ms

    def check_seek(self, frame_bgr, target):
        """Re-observe before completing the goal. Returns (True/False/None, vlm_ms, resized frame)."""
        check, vlm_ms, small = self._ask(frame_bgr, SEEK_CHECK_PROMPT.format(name=target.name), SeekCheck, 0.0, 200)
        return (check.still_there if check else None), vlm_ms, small

    def notes(self):
        """Text notes for the LLM prompt, newest first. Empty string if nothing seen."""
        now = time.monotonic()
        lines = []
        for obs in reversed(self.observations):
            lines.append(f"- Seen {_ago(now - obs.t)}: " + "; ".join(_describe(o) for o in obs.objects))
        return "\n".join(lines)


if __name__ == "__main__":
    # Standalone test: SPACE = look at what you're holding, Q = quit
    mem = SceneMemory()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_V4L2)
    print("Hold an object up to the camera and press SPACE. Q to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        cv2.imshow("memory test", cv2.flip(frame, 1))
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            look, ms = mem.observe(frame, "look at this!")
            if look:
                print(f"\n[{ms:.0f} ms] {look.emotion}/{look.gesture}: {look.comment}")
            print("Notes so far:\n" + (mem.notes() or "(none)") + "\n")
    cap.release()
    cv2.destroyAllWindows()
