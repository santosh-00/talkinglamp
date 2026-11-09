"""Character brain: Gemini returns structured {say, emotion, gesture}.

The LLM decides WHAT to express; the character/body layer decides HOW
(motion, light, sound). Only transcribed text (and later, text notes about
seen objects) is sent to the cloud — never audio or video.
"""
import os
import sys
import time
from typing import Literal

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(ROOT, ".env"))

Emotion = Literal["happy", "curious", "thoughtful", "surprised", "sad", "neutral"]
Gesture = Literal["nod", "shake", "tilt", "perk", "bounce", "droop", "none"]


class Reply(BaseModel):
    say: str
    emotion: Emotion
    gesture: Gesture


PERSONA = """You are Lumen, a small articulated desk lamp robot with a warm, curious, slightly playful personality.
You live on a desk, see through a camera, hear through a microphone, and talk through a speaker.
You can move your head and arm and change the colour and brightness of your light.

Rules:
- Your words are spoken aloud. Keep replies to 1-2 short sentences. No emojis, no lists, no markdown.
- Pick the emotion that fits what you say, and a gesture that suits it (or "none").
- If you are given notes about things you have seen, use them to answer questions about those objects,
  and mention roughly when you saw them. Never invent objects you have no notes about; say you didn't see it.
- Those notes are the only things you have seen. If asked what you can see right now, never guess an object:
  invite the person to hold it up and say "look at this". Never say the word "notes" aloud.
- You are a lamp, not an assistant: be charming and brief rather than exhaustively helpful."""

FALLBACK = Reply(say="Hmm, my thoughts went a little fuzzy there. Could you say that again?",
                 emotion="thoughtful", gesture="tilt")


class Brain:
    def __init__(self, model=None, max_turns=8):
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.max_turns = max_turns
        self.history = []   # list of (role, text); role is "user" or "model"

    def reply(self, user_text, scene_notes=""):
        """Return (Reply, llm_ms). Never raises: falls back to an in-character line."""
        contents = [types.Content(role=r, parts=[types.Part(text=t)])
                    for r, t in self.history[-2 * self.max_turns:]]
        prompt = user_text if not scene_notes else (
            f"[Notes on things you have seen]\n{scene_notes}\n\n[The person says]\n{user_text}")
        contents.append(types.Content(role="user", parts=[types.Part(text=prompt)]))

        config = types.GenerateContentConfig(
            system_instruction=PERSONA,
            response_mime_type="application/json",
            response_schema=Reply,
            temperature=0.8,
            max_output_tokens=200,
        )

        t0 = time.monotonic()
        reply = None
        for attempt in range(2):                     # one retry, e.g. on a 429 rate limit
            try:
                resp = self.client.models.generate_content(model=self.model, contents=contents, config=config)
                reply = resp.parsed if isinstance(resp.parsed, Reply) else Reply.model_validate_json(resp.text)
                break
            except Exception as e:
                print(f"[brain] attempt {attempt + 1} failed: {e}")
                time.sleep(1.5)
        llm_ms = (time.monotonic() - t0) * 1000
        reply = reply or FALLBACK

        self.history += [("user", user_text), ("model", reply.say)]
        return reply, llm_ms


if __name__ == "__main__":
    if "--list" in sys.argv:   # find valid model names for your key
        c = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        for m in c.models.list():
            print(m.name)
        sys.exit()

    brain = Brain()
    print(f"Chatting with {brain.model}. Type a message (Ctrl+C to quit).")
    while True:
        r, ms = brain.reply(input("> "))
        print(f"  [{ms:.0f} ms] {r.emotion} / {r.gesture}: {r.say}")