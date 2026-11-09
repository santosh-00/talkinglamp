# Lumen: a live character lamp robot

Lumen is an expressive desk-lamp character built around the supplied 5-DOF lamp
model. It sees you through the laptop camera, listens through the microphone,
talks through the speaker, and moves and lights up in a MuJoCo simulation.

## Current status

| Challenge moment | Status | Notes |
|---|---|---|
| 1. Engagement | Done | Face and head orientation with hysteresis: engages after 0.6 s of facing, disengages after 2 s of looking away or 6 s with the face hidden |
| 2. Character response | Done | Motion, light, voice, sound effects and a sleep lullaby |
| 3. Spoken interaction | Done | Local speech recognition, Gemini reply, local speech synthesis |
| 4. Scene memory | Done | A spoken cue triggers one Gemini vision look; the result is stored as text and used to answer later questions. Memory lasts for the current run only |
| 5. Goal-directed action | Done | "Find something new": Gemini compares the live view with memory and picks lamp actions, then a fresh frame confirms the result before the goal completes |

| Platform | Status |
|---|---|
| Windows 11, Python 3.12 | This is the development environment |
| Ubuntu 24.04 (the challenge target) | **Not tested yet.** The steps below are written for it |

## Requirements

- Python 3.12
- A webcam, microphone and speaker
- A desktop session with OpenGL, for the MuJoCo and camera windows
- Internet access for first-time downloads and for every LLM reply
- A Google Gemini API key from <https://aistudio.google.com/apikey>

## Setup on Ubuntu 24.04

Run all commands from the `msrc/` directory.

### 1. Install system packages and Python dependencies

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip curl \
    libportaudio2 libgl1 libegl1 libgles2 libglib2.0-0t64

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` pins the versions from the working Windows environment. The apt
list is the expected set of native libraries and has not been confirmed on a clean
machine.

### 2. Download the voice model

```bash
mkdir -p models/voices
BASE=https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium
curl -L -o models/voices/en_US-lessac-medium.onnx      "$BASE/en_US-lessac-medium.onnx"
curl -L -o models/voices/en_US-lessac-medium.onnx.json "$BASE/en_US-lessac-medium.onnx.json"
```

Skip this if the files are already in `models/voices/`. On Linux there is no
fallback voice, so Lumen cannot speak without them.

### 3. Add your API key

Create `msrc/.env`:

```ini
GEMINI_API_KEY=your-key-here
# Optional: model for chat and vision. If unset, chat uses
# gemini-flash-lite-latest and vision uses gemini-flash-latest.
GEMINI_MODEL=gemini-flash-lite-latest
# Optional: second vision model to try if the first one fails.
# GEMINI_FALLBACK_MODEL=gemini-flash-latest
```

Keep this file private. It is listed in `.gitignore` so it won't be committed.

### 4. Prepare the lamp mesh (only for a fresh copy of the robot files)

MuJoCo only reads binary STL, and the supplied `lamp_shade.stl` is ASCII. The
copy in `msrc/robot/assets/` is already converted. If you replace `msrc/robot/`
with the original files, run:

```bash
python tools/convert_stl.py
```

## Setup on Windows

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Then do steps 2 to 4 above, downloading the two voice files by opening their URLs
in a browser or using `curl.exe`. On Windows, `pyttsx3` provides a fallback voice
if Piper cannot load.

## Check each part

Run these one at a time before the full program, from `msrc/` with the virtual
environment active. Each one exercises a single part of the system. Only one
program can use the camera at a time, so close each camera test before the next.

| Part | Command | What you should see |
|---|---|---|
| Lamp body | `python src/body.py` | The lamp cycles through poses, gestures and light changes |
| Robot model | `python tools/inspect_urdf.py` | Joint table, then each joint sweeps in turn |
| Camera | `python src/perception.py` | Debug window; `engaged` prints when you face the camera |
| Mic levels | `python src/listener.py --levels` | Quiet and loud levels per second, for tuning `min_rms` |
| Speech recognition | `python src/listener.py` | Your sentences printed as text |
| Voice | `python src/voice.py "Hello there"` | Lumen speaks; add `--pitch 1.3` to try another pitch |
| Sounds | `python src/audio.py` | Happy chime, thinking sound, movement whir, then 15 s of the lullaby |
| LLM | `python src/brain.py` | Type a message and get a reply with emotion and gesture |
| Model names | `python src/brain.py --list` | Models available to your API key |
| Scene memory | `python src/memory.py` | Hold up an object and press Space for a comment and stored notes; Q quits |
| Camera + body | `python tools/step3_follow.py` | The simulated lamp follows your face |

## Run Lumen

```bash
python src/main.py
```

Startup takes a few seconds. The first run also downloads the Whisper `base.en`
speech model and, if it is missing, the MediaPipe face model. When you see
`Ready. Look at the camera to say hi.`, two windows are open: the MuJoCo lamp and
a camera debug view.

To stop, close the MuJoCo window.

### How to interact

1. **Look at the camera.** After about 0.6 s Lumen perks up, brightens, follows
   your face and greets you.
2. **Wait for the greeting to finish, then speak.** The microphone is off while
   Lumen speaks, so it won't hear anything you say during a reply. Pause briefly
   when you finish; 0.7 s of silence ends your turn.
3. **Watch the turn play out.** Lumen leans in while listening, tilts back with a
   pulsing cool light while thinking, then speaks with a light colour and gesture
   that match its emotion.
4. **Show it an object.** Hold the object up close to the camera and say
   something like "Hey, look at this!" Lumen says a quick filler line, leans in
   with a bright white light, then reacts to what it sees. Keep the object still
   until Lumen reacts, because the photo is taken about 2 s after you stop talking.
5. **Ask about it later.** Put the object away, chat for a while, then ask "What
   colour was the thing I showed you?" Lumen answers from memory, without looking
   again, and mentions roughly when it saw the object.
6. **Give it a goal.** Say "Lumen, find something new." Lumen hunts with a pulsing
   green light, turns toward a new object it finds in the camera view and lights
   it up, looks again to confirm it is still there, then reacts and remembers it.
   If there is nothing new, it asks you to show it something and looks again, up
   to 4 rounds.
7. **Look away for 2 s.** Lumen droops and dims, and stops listening. It only
   listens while you are engaged. If your face is hidden instead, for example
   behind an object you are holding up, Lumen waits 6 s, and it stays engaged for
   the whole of a look or a goal.
8. **Come back.** After a short glance away it just perks up. After 8 s it says
   "welcome back", and after 60 s it greets you in full again.
9. **Leave the frame.** Lumen goes to sleep and plays a quiet lullaby until someone
   comes back.

Lumen only uses the camera for objects when your sentence contains one of these
cues (`SHOW_CUES` and `SEEK_CUES` in `src/memory.py`), matched locally:

| To | Say something like |
|---|---|
| Trigger a look | "look at this", "check this out", "what is this", "remember this", "I'm holding", "let me show you", "what's in my hand", "what do you see" |
| Start the goal | "find something new", "find anything", "look for something", "look around for", "search for something", "see anything new", "spot something" |
| Recall from memory | "What colour was the thing I showed you?", "What did I show you earlier?", "What was in my hand earlier?" |

A vague follow-up such as "What about now?" does not trigger a look. Say "look at
this" again for each new object.

| State | Pose | Light |
|---|---|---|
| Nobody present | Sleep | Very dim warm, with the lullaby playing |
| Present, not facing | Idle | Low warm |
| Engaged | Attentive, following your face | Bright warm |
| Listening | Leaning in | Bright warm |
| Thinking | Looking up and aside | Cool, pulsing |
| Inspecting an object | Leaning in, looking slightly down | Bright white |
| Hunting for something new | Attentive, scanning | Green, pulsing |
| Spotlighting a found object | Leaning in, aimed at the object | Bright white |
| Speaking | Attentive | Set by emotion: happy, curious, thoughtful, surprised, sad or neutral |

All sounds are generated in code, so there are no audio files to install:

| Sound | When it plays |
|---|---|
| Servo whir | Whenever the lamp moves; pitch and volume follow joint speed. Quieter under speech and off while you are talking |
| Lullaby | While Lumen is asleep, fading in and out over 2 s |
| Rising chime | When Lumen starts speaking with a happy emotion |
| Bubbly sound | When Lumen starts thinking |

### Reading the output

The console prints perception events, the conversation, and a latency breakdown
for every turn. Each turn is also appended to `logs/turns.csv`:

| Column | Meaning |
|---|---|
| `kind` | `talk` for a normal reply, `look` when Lumen inspected an object, `seek` for a goal |
| `endpoint_ms` | Silence wait that ends your turn |
| `stt_ms` | Speech-to-text time |
| `model_ms` | Gemini time, including any retry: chat for `talk`, vision for `look`, all planning and checking calls added together for `seek` |
| `tts_ms` | Speech synthesis time |
| `total_ms` | From when you stopped talking to when Lumen's voice starts. For `look` and `seek` turns this is the final reaction, after the filler lines and any lamp actions |
| `user`, `reply`, `emotion`, `gesture` | What was said and how Lumen expressed it |

Each successful look also saves the image sent to Gemini to `logs/memory/`, so you
can see exactly what Lumen saw. A completed goal saves its confirming frame there.

During a goal, the console prints a `[goal]` line for each round with what Gemini
found, the chosen actions and the result of the recheck.

## How it fits together

```text
camera -> perception.py --(face state, latest frame)--+
                                                      v
mic -> listener.py --(utterances)-----------> character.py --> body.py  (motion, light in MuJoCo)
                                               |        |  \-> voice.py (Piper TTS -> speaker)
                                               |        |  \-> audio.py (SFX, music -> speaker)
                                               v        v
        (Gemini: say, emotion, gesture) brain.py <-text notes- memory.py (Gemini vision: objects, comment, goal plan)
```

| File | Responsibility |
|---|---|
| `src/main.py` | Builds the modules and runs the 60 Hz loop |
| `src/perception.py` | Camera thread: MediaPipe face landmarks and engagement |
| `src/listener.py` | Voice activity detection and faster-whisper transcription |
| `src/brain.py` | Gemini call that returns structured `{say, emotion, gesture}` |
| `src/memory.py` | Spoken-cue matching, one-frame Gemini vision calls, object notes, goal planning and rechecks |
| `src/audio.py` | Synthesized sound effects, lullaby and servo whir, mixed on their own output stream |
| `src/character.py` | The only place that decides behaviour |
| `src/body.py` | Joint motion within URDF limits, gestures, light |
| `src/voice.py` | Piper speech with a pitch shift |

The LLM chooses what to express, and `character.py` and `body.py` decide how to
perform it.

Scene memory follows a "perceive once, recall many times" design. A look is the
one expensive step: a single frame goes to Gemini vision, and the filler line
hides its latency. The result is stored as text with a timestamp and added to
later chat prompts, so recall needs no new image.

Goal-directed action runs as an observe, act, re-observe loop:

1. A local keyword match recognizes the spoken goal.
2. One frame, the goal and the object notes go to Gemini, which returns whether
   something new is visible, where it is in the image, and up to 4 actions.
3. Actions come from a fixed list: `scan_left`, `scan_right`, `scan_center`,
   `aim_at_target`, `spotlight_on`, `spotlight_off`, `nod`, `tilt`, `perk` and
   `bounce`.
4. `character.py` and `body.py` execute them. The model decides what to do; the
   body decides how, and clamps every motion to the joint limits.
5. A fresh frame confirms the object is still visible before the goal completes
   and the object is stored.

The camera is the fixed laptop webcam, so scanning is expressive and does not
change what Lumen sees. The feedback comes from re-observing the scene.

**Data sent to Google Gemini:**
- The text transcript of what you say, the recent conversation, and the text
  notes about objects Lumen has seen.
- One JPEG per look, resized to at most 768 px, sent only when your sentence
  contains a look cue. It can include you and your surroundings.
- During a goal, the same kind of JPEG for each planning round and each recheck:
  at least two per goal, and up to eight if it keeps retrying.

Microphone audio and all other camera frames stay on the machine.

## Configuration

These settings are edited in the source.

| Setting | Location | Default | Change it when |
|---|---|---|---|
| `GEMINI_MODEL` | `.env` | Chat: `gemini-flash-lite-latest`; vision: `gemini-flash-latest` | You want a different model. When set, it applies to both |
| `GEMINI_FALLBACK_MODEL` | `.env` | Not set | You want a second vision model to try on failure |
| `SHOW_CUES` | `src/memory.py` | See [How to interact](#how-to-interact) | A phrase you use doesn't trigger a look, or triggers one by mistake |
| `CAPTURE_DELAY_S` | `src/character.py` | `0.4` s | The photo is taken too early or too late |
| `camera_index` | `PerceptionConfig` | `0` | The wrong camera opens |
| `yaw_facing`, `pitch_band` | `PerceptionConfig` | `0.18`, `(0.10, 0.65)` | Engagement is too strict or too loose for your camera angle |
| `engage_after`, `disengage_after` | `PerceptionConfig` | `0.6`, `2.0` s | You want a different engagement feel |
| `disengage_no_face_after` | `PerceptionConfig` | `6.0` s | Lumen gives up too soon when your face is blocked, or is slow to notice you left |
| `SEEK_CUES` | `src/memory.py` | See [How to interact](#how-to-interact) | A goal phrase isn't recognized, or starts a goal by mistake |
| `MAX_SEEK_ROUNDS`, `SEEK_WAIT_S` | `src/character.py` | `4`, `3.0` s | Lumen gives up too early, or hunts for too long |
| `MUSIC_GAIN`, `SFX_GAIN`, `WHIR_GAIN` | `src/audio.py` | `0.22`, `0.35`, `0.05` | The sounds are too loud or too quiet against the voice |
| `min_rms` | `ListenerConfig` | `350` | Tune with `--levels`: set it between your quiet and speaking levels |
| `mic_device` | `ListenerConfig` | system default | The wrong mic is used; list devices with `python -m sounddevice` |
| `end_silence_ms` | `ListenerConfig` | `700` | Your turns get cut off, or replies start too slowly |
| `model_size` | `ListenerConfig` | `base.en` | Trading accuracy against CPU time |
| `PITCH`, `SPEED` | `src/voice.py` | `1.25`, `1.05` | You want a different voice character |
| `YAW_SIGN` | `src/character.py` | `-1` | The lamp looks the wrong way |

## Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| `could not open camera` | Close other apps using the camera or change `camera_index`. On Linux, check that `/dev/video0` exists and your user can access it |
| Lumen never engages | Check the debug window: `eye_px` must be at least 35, so move closer. If `yaw_ratio` or `pitch_ratio` are out of range, adjust `PerceptionConfig` |
| Lumen looks engaged but doesn't hear you | Your voice may be under the loudness gate. Run `listener.py --levels` and lower `min_rms`, and confirm the mic with `python -m sounddevice` |
| `KeyError: 'GEMINI_API_KEY'` | `msrc/.env` is missing or the variable name is wrong |
| Lumen says "my thoughts went a little fuzzy" | The Gemini call failed. Read the `[brain] attempt` line in the console; common causes are a bad key, a rate limit, no internet, or an invalid model name |
| Lumen says it can't see the object, or doesn't react to it | Check `logs/turns.csv`. If the turn is `talk`, no cue matched, so say "look at this". If it is `look`, open the newest image in `logs/memory/` and hold the object closer and steadier |
| Lumen names the wrong object | If the object is clear in `logs/memory/`, set a stronger vision-capable model with `GEMINI_MODEL` |
| Lumen looks again when you meant to ask about the past | Your sentence contains a look cue. Rephrase it in the past tense or adjust `SHOW_CUES` |
| Lumen picks a background item as "new" | On a fresh run it has no memories, so anything small in view can count. Clear the view or show it those items first |
| A goal never starts | Check that the turn is logged as `seek`. If it is `talk`, no cue matched, so say "find something new" |
| No sound effects or music | Look for `[audio] no output stream` in the console, then run `python src/audio.py` and check the default output device |
| Lumen disengages while your face is partly covered | A partly hidden face can read as looking away, which has the shorter limit. Say the look cue sooner, or raise `disengage_after` |
| `Piper unavailable` | The voice files are missing from `models/voices/`. On Linux, Lumen then has no working voice |
| `PortAudio library not found` | Install `libportaudio2` |
| Error loading `lamp_shade.stl` | The mesh is still ASCII. Run `python tools/convert_stl.py` |
| Crash from the camera debug window on Linux | Use `Perception(show_debug=False)` in `src/main.py` |

## Known limitations

- Scene memory is lost when the program stops.
- Looks and goals are triggered by keyword only, so unlisted phrasings are treated
  as chat.
- The goal is a single behaviour, finding something new, rather than arbitrary
  spoken goals.
- The camera is fixed, so the lamp's movement can't reveal anything new, and aiming
  at an object is approximate because the camera and lamp are not calibrated.
- Lumen can't tell a hidden face from one that has left, so it takes 6 s to notice
  someone has walked away.
- Lumen needs internet access for replies, looks and goals. Without it, it uses a
  fallback line.
- Replies are half-duplex, so you can't interrupt Lumen while it is speaking.
- The camera debug window always opens, because `main.py` uses the default
  `show_debug=True`.
- Motion is kinematic: joint angles are set directly, with no physics or torque
  simulation. Position stays within URDF soft limits. Base motion respects the
  velocity limits, but gestures are added on top and can exceed them (3.5 rad/s
  measured on waking, against a limit of 1.60 rad/s at most).
- CPU and memory use have not been measured on the 4-core, 8 GB target.
