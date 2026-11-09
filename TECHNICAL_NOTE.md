# Lumen: technical note

Lumen is one character process: laptop camera, microphone and speaker, with the
supplied 5-DOF lamp simulated in MuJoCo. Setup is in `README.md`.

## Architecture and data flow

```text
camera -> perception (thread) --face state, latest frame--+
                                                          v
mic -> listener (VAD + Whisper) --utterance events--> character (main thread, 60 Hz)
                                                       |      |       |        |
                  cloud: text + notes     cloud: 1 JPEG|      |       |        |
                         v                    v        |      v       v        v
                   brain (Gemini) <-notes- memory <----+    body    voice    audio
                   say/emotion/gesture     objects, plan   MuJoCo   Piper    SFX, music
```

## Protocol and ownership

All modules run in one Python process, so there is no network protocol between
character and body. `character.py` alone decides behaviour.

- **Perception to character:** a polled level-state snapshot (present, facing,
  engaged, position). Dropped events cannot desynchronize the lamp, because it
  reconciles against state on every tick.
- **Listener to character:** a queue of `speech_start`, `utterance`, `discarded`.
- **Character to body:** `set_pose`, `look_at`, `gesture`, `set_light`, called only
  from the main thread. LLM, vision and TTS run on a worker that sets phase, emotion
  and aim under a lock.
- **Half-duplex:** the microphone is muted during Lumen's turns, so it never hears
  itself. The cost is that it cannot be interrupted.

## Model to action

Models return schema-validated JSON and choose only from closed vocabularies. Chat
returns `{say, emotion, gesture}`. Character and body decide how that is performed.

The goal "find something new" is a closed vision-language loop:

1. A local keyword match recognizes the spoken goal.
2. One frame, the goal and the object notes go to Gemini, which returns
   `found_new`, the target, its image position (0-1000) and up to 4 actions from
   `scan_left/right/center`, `aim_at_target`, `spotlight_on/off`, `nod`, `tilt`,
   `perk`, `bounce`.
3. Character clamps the coordinates, converts them to yaw and pitch, and executes
   the actions, adding the aim and spotlight if the model omits them.
4. A fresh frame must confirm the target before the goal completes and it is
   stored. Otherwise Lumen looks again, up to 4 rounds.

**Scene memory is "perceive once, recall many times".** A spoken cue triggers one
vision call, masked by a filler line. Objects are stored as timestamped text and
added to later chat prompts, so recall needs no image.

**Cloud boundary.** Gemini receives transcripts, conversation, object notes, and
JPEGs of at most 768 px, sent only on a look or goal cue. Audio and all other frames
stay local. Engagement, speech recognition, TTS and all execution are local. This
keeps the always-on loops cheap on CPU-only hardware and pays for cloud vision only
when the person asks.

## Simulation and physical reasoning

The URDF is loaded into MuJoCo and driven kinematically: per-joint critically damped
springs, clamped to URDF soft limits, with velocity limits on the spring motion. The
base carries slow turns and the neck covers fast ones. Physics stepping is not used.

On real hardware this would not transfer directly. Gestures are added on top of the
limited motion and reached 3.5 rad/s, against URDF limits of at most 1.60 rad/s, so
they would need a trajectory limiter. Torque, backlash and cable routing are not
modelled. The camera is the fixed laptop webcam, so scanning is expressive and the
feedback comes from re-observing the scene. Aiming is approximate because camera and
lamp are not calibrated. A head-mounted camera would change this loop.

## Deployment

The target is a Python 3.12 virtual environment with pinned `requirements.txt` and
listed apt libraries on Ubuntu 24.04. Whisper `base.en` int8, MediaPipe and Piper all
run on CPU with no GPU requirement. Audio uses PortAudio; the camera backend switches
to V4L2 on Linux. **The Ubuntu path has not been run.** There is no service unit or
container yet.

## Key choices and tradeoffs

| Choice | Benefit | Cost |
|---|---|---|
| Local keyword cues | Deterministic, no extra LLM call | Unlisted phrasing becomes chat |
| Text memory | Fast, cheap recall | Lost on exit; limited to noted details |
| 6 s grace for a hidden face | Holding up an object doesn't disengage | Slower to notice someone left |
| Synthesized audio | No assets or licences | Simple timbre |
| Single process | Simple ownership, low latency | No crash isolation between modules |

## Measurements

One 156 s session on the Windows development laptop (16 logical cores, 16.6 GB RAM),
covering chat and object looks. These are not target-hardware figures.

| Resource | Mean | p95 | Peak |
|---|---:|---:|---:|
| CPU, logical cores used (after 20 s warmup) | 1.6 | 4.5 | 5.4 |
| Memory (RSS) | 857 MB | | 1094 MB |

Average load fits a 4-core target, and peak memory is about 14% of 8 GB. Bursts above
4 cores mean speech recognition will be slower there.

| Latency, stopped talking to voice | n | Median | Max | Model median |
|---|---:|---:|---:|---:|
| Chat | 9 | 2856 ms | 14012 ms | 844 ms |
| Object look (reaction after filler) | 3 | 6403 ms | 6621 ms | 4140 ms |
| Goal | 0 | Not measured | Not measured | Not measured |

A typical chat turn was 700 ms endpointing, about 1000 ms recognition, 844 ms LLM and
about 250 ms TTS. The 14 s maximum was one turn with 12.3 s of recognition on a
one-word utterance. Whisper's temperature-fallback retries on unclear audio are the
likely cause, but this is unconfirmed.

**Engagement reliability** (informal manual trials; approximate rates, exact trial
counts not recorded):

| Test | Result |
|---|---|
| Engages when facing the camera | About 80-90% |
| Disengages when looking away | About 80-90% |
| No false engagement when facing away | About 80-90% |
| Stays engaged with face blocked | About 80-90% |

Observed engage dwell was 0.61-0.66 s against a 0.6 s setting.

## Completed, left out and known limitations

All five demonstration moments are implemented. Intentionally left out: a physics
simulation, persistent memory, packaging, and arbitrary spoken goals (the goal is one
behaviour).

- The goal loop was tested with simulated inputs and one Gemini call on a synthetic
  image, but its live latency was not measured.
- Whisper mis-hears unclear speech, and those turns still produce replies.
- An utterance finishing just after disengagement is still answered.
- No barge-in. Replies and vision need internet, with an in-character fallback.
- A partly covered face can read as looking away, which has the shorter limit.
- On a fresh run, background objects can count as "new".
