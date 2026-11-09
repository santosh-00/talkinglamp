"""Step 3 check: the sim lamp reacts to engagement and follows your face.

The lamp reconciles its mode against perception's *state* every frame
(level-triggered), so dropped or reordered events can't leave it out of sync.
"""
import os
import sys
import time

import mujoco.viewer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from body import Body, WARM          # noqa: E402
from perception import Perception    # noqa: E402

YAW_GAIN = 0.6     # image half-width → radians of lamp yaw
PITCH_GAIN = 0.35  # image half-height → radians of lamp pitch
YAW_SIGN = -1      # image-left (your right) → lamp's left; flip to +1 if it looks the wrong way


def enter_mode(body, mode, previous):
    """Perform the transition into a new mode (one-shot gestures happen here)."""
    if mode == "engaged":
        body.set_pose("attentive", 1.3)
        body.gesture("perk")
        body.set_light(WARM, 1.0)
    elif mode == "present":
        body.clear_gaze()
        if previous == "engaged":
            body.gesture("droop")
        body.set_pose("idle", 0.7)
        body.set_light(WARM, 0.35)
    else:  # absent
        body.clear_gaze()
        body.set_pose("sleep", 0.6)
        body.set_light(WARM, 0.15)


def main():
    body = Body()
    per = Perception()
    per.start()

    with mujoco.viewer.launch_passive(body.m, body.d) as v:
        v.cam.lookat[:] = [-0.1, 0.0, 0.35]
        v.cam.distance, v.cam.elevation, v.cam.azimuth = 1.5, -12, 180

        current = None
        last = time.time()
        while v.is_running():
            # Events are for logging only; behaviour comes from state below
            while not per.events.empty():
                name, _, info = per.events.get_nowait()
                print(name, info or "")

            s = per.state
            mode = "engaged" if s.engaged else "present" if s.present else "absent"
            if mode != current:
                print(f"  lamp: {current} → {mode}")
                enter_mode(body, mode, current)
                current = mode

            if mode == "engaged":
                body.look_at(YAW_SIGN * s.x * YAW_GAIN, -s.y * PITCH_GAIN)

            now = time.time()
            with v.lock():
                body.update(min(now - last, 0.05))
            last = now
            v.sync()
            time.sleep(1 / 60)

    per.stop()


if __name__ == "__main__":
    main()