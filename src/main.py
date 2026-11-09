"""Entry point: build all modules, then run the character loop at ~60 Hz."""
import time

import mujoco.viewer

from audio import Audio
from body import Body
from brain import Brain
from character import Character
from listener import Listener
from memory import SceneMemory
from perception import Perception
from voice import Voice


def main():
    print("Starting up…")
    body = Body()
    voice = Voice()
    brain = Brain()
    memory = SceneMemory()
    listener = Listener()          # loads Whisper (a few seconds)
    perception = Perception()

    perception.start()
    listener.start()
    audio = Audio()
    audio.start()
    character = Character(body, perception, listener, brain, voice, memory, audio)
    print("Ready. Look at the camera to say hi.")

    with mujoco.viewer.launch_passive(body.m, body.d) as v:
        v.cam.lookat[:] = [-0.1, 0.0, 0.35]
        v.cam.distance, v.cam.elevation, v.cam.azimuth = 1.5, -12, 180
        last = time.time()
        while v.is_running():
            now = time.time()
            dt = min(now - last, 0.05)
            last = now
            with v.lock():
                character.tick(dt)
            v.sync()
            time.sleep(1 / 60)

    audio.stop()
    listener.stop()
    perception.stop()


if __name__ == "__main__":
    main()
