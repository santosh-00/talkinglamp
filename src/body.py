"""Simulated lamp body: layered motion (pose, gaze, gestures, breathing) and light, in MuJoCo.

Frame notes (from the URDF):
  - At zero pose the head faces world -x.
  - +head_pitch looks DOWN; +shoulder leans BACK; -elbow folds FORWARD.
  - +base_yaw / +neck_yaw turn toward the lamp's LEFT.
The body is driven kinematically: we set joint angles directly, no physics stepping.
"""
import glob
import math
import os
import time
from dataclasses import dataclass

import mujoco
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

JOINTS = ["base_yaw_joint", "shoulder_pitch_joint", "elbow_pitch_joint",
          "neck_yaw_joint", "head_pitch_joint"]
BASE, SHOULDER, ELBOW, NECK, HEAD = range(5)

# From the URDF: safety_controller soft limits and velocity limits
SOFT_LIMITS = np.array([(-2.45, 2.45), (-0.65, 0.95), (-1.70, 0.30), (-1.25, 1.25), (-0.80, 0.60)])
MAX_VEL = np.array([1.50, 0.95, 1.15, 1.60, 1.45])
# Spring stiffness per joint: base and arms slow and heavy, neck and head quick like eyes
STIFFNESS = np.array([3.0, 4.0, 4.0, 9.0, 9.0])


@dataclass
class Pose:
    shoulder: float
    elbow: float
    down: float        # head gaze angle below horizontal (rad); negative = look up
    yaw: float = 0.0   # default facing direction when there is no gaze target


POSES = {
    "sleep":     Pose(shoulder=0.15, elbow=-1.25, down=1.10),
    "idle":      Pose(shoulder=0.35, elbow=-0.90, down=0.35),
    "attentive": Pose(shoulder=0.25, elbow=-0.75, down=0.00),
    "lean_in":   Pose(shoulder=0.10, elbow=-0.60, down=0.10),
    "thinking":  Pose(shoulder=0.40, elbow=-0.70, down=-0.35, yaw=0.30),
}

# Keyframes: (time_s, {joint_index: offset_rad}); joints not listed are 0
GESTURES = {
    "nod":    [(0.0, {}), (0.18, {HEAD: 0.28}), (0.36, {HEAD: -0.05}), (0.54, {HEAD: 0.22}), (0.80, {})],
    "shake":  [(0.0, {}), (0.20, {NECK: 0.30}), (0.45, {NECK: -0.30}), (0.70, {NECK: 0.20}), (0.95, {})],
    "tilt":   [(0.0, {}), (0.35, {NECK: 0.30, HEAD: -0.15}), (1.40, {NECK: 0.30, HEAD: -0.15}), (1.90, {})],
    "perk":   [(0.0, {}), (0.15, {SHOULDER: -0.12, HEAD: -0.20}), (0.35, {SHOULDER: 0.03, HEAD: 0.05}), (0.60, {})],
    "droop":  [(0.0, {}), (0.60, {SHOULDER: 0.10, HEAD: 0.30}), (1.40, {SHOULDER: 0.10, HEAD: 0.30}), (2.00, {})],
    "bounce": [(0.0, {}), (0.15, {ELBOW: 0.12}), (0.30, {}), (0.45, {ELBOW: 0.12}), (0.60, {})],
}

WARM = (1.0, 0.85, 0.60)
COOL = (0.55, 0.75, 1.00)


def _sample(frames, t):
    """Smoothstep-interpolated joint offsets of a gesture at time t."""
    out = np.zeros(5)
    for (t0, f0), (t1, f1) in zip(frames, frames[1:]):
        if t0 <= t <= t1:
            a = (t - t0) / (t1 - t0)
            a = a * a * (3 - 2 * a)
            for k in set(f0) | set(f1):
                out[k] = (1 - a) * f0.get(k, 0.0) + a * f1.get(k, 0.0)
            break
    return out


def _build_model(urdf):
    """Load the URDF and add a desk plane plus a spotlight on the light emitter."""
    try:
        spec = mujoco.MjSpec.from_file(urdf)
        spec.worldbody.add_geom(name="desk", type=mujoco.mjtGeom.mjGEOM_PLANE,
                                size=[1.5, 1.5, 0.05], rgba=[0.45, 0.36, 0.28, 1])
        emitter = spec.body("light_emitter_link")
        emitter.add_light(name="lamp_spot", dir=[1, 0, 0], diffuse=list(WARM),
                          cutoff=45, exponent=8, castshadow=True)
        m = spec.compile()
    except Exception as e:
        print(f"[body] spotlight setup failed ({e}); using plain URDF without spotlight")
        m = mujoco.MjModel.from_xml_path(urdf)
    # Dim the default headlight so the lamp's own light reads clearly
    m.vis.headlight.diffuse[:] = 0.35
    m.vis.headlight.ambient[:] = 0.15
    return m


def _find_glow_geoms(m):
    """Geoms that should glow: the bulb sphere and the light disc (fixture_light colour)."""
    glow = []
    for g in range(m.ngeom):
        body = m.body(m.geom_bodyid[g]).name
        mat = m.geom_matid[g]
        rgba = m.mat_rgba[mat] if mat >= 0 else m.geom_rgba[g]
        if body in ("light_emitter_link", "lamp_head_link") and rgba[0] > 0.99 and abs(rgba[2] - 0.76) < 0.03:
            glow.append(g)
    if not glow:  # colour not preserved by the importer: fall back to the emitter sphere
        glow = [g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name == "light_emitter_link"]
    for g in glow:  # drive colour through geom_rgba
        if m.geom_matid[g] >= 0:
            m.geom_rgba[g] = m.mat_rgba[m.geom_matid[g]]
            m.geom_matid[g] = -1
    return glow


class Body:
    def __init__(self, urdf=None):
        urdf = urdf or glob.glob(os.path.join(ROOT, "robot", "*.urdf"))[0]
        self.m = _build_model(urdf)
        self.d = mujoco.MjData(self.m)
        self.qadr = np.array([self.m.jnt_qposadr[self.m.joint(n).id] for n in JOINTS])

        self.pose = POSES["sleep"]
        self.energy = 1.0          # scales spring stiffness: >1 snappy, <1 sluggish
        self.gaze = None           # (yaw, pitch) in radians; pitch > 0 = look up
        self.gestures = []         # [frames, start_time]
        self.t = 0.0

        p = self.pose
        self.q = np.array([0.0, p.shoulder, p.elbow, 0.0, p.down + p.shoulder + p.elbow])
        self.v = np.zeros(5)

        self.glow = _find_glow_geoms(self.m)
        self.spot = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_LIGHT, "lamp_spot")
        self.color = np.array(WARM)
        self.target_color = np.array(WARM)
        self.intensity = self.target_intensity = 0.15
        self.pulse = 0.0
        self.speed = 0.0           # fastest joint speed last frame (rad/s), used for movement sound
        self._last_qpos = None
        self.update(0.0)

    # ---------- commands (called by the character layer) ----------
    def set_pose(self, name, energy=1.0):
        self.pose = POSES[name]
        self.energy = energy

    def look_at(self, yaw, pitch):
        """yaw > 0 = lamp's left; pitch > 0 = up. Radians relative to facing straight ahead."""
        self.gaze = (yaw, pitch)

    def clear_gaze(self):
        self.gaze = None

    def gesture(self, name):
        self.gestures.append([GESTURES[name], self.t])

    def set_light(self, color=None, intensity=None, pulse=None):
        if color is not None:
            self.target_color = np.array(color, dtype=float)
        if intensity is not None:
            self.target_intensity = float(intensity)
        if pulse is not None:
            self.pulse = float(pulse)

    # ---------- per-frame update ----------
    def update(self, dt):
        self.t += dt
        p = self.pose
        yaw, down = (self.gaze[0], -self.gaze[1]) if self.gaze else (p.yaw, p.down)

        target = np.empty(5)
        target[BASE] = 0.6 * yaw                     # base carries most of a turn, slowly
        target[SHOULDER] = p.shoulder
        target[ELBOW] = p.elbow
        target[NECK] = yaw - self.q[BASE]            # neck covers the rest, fast; unwinds as base catches up
        target[HEAD] = down + self.q[SHOULDER] + self.q[ELBOW]   # cancel arm tilt so gaze holds steady
        target = np.clip(target, SOFT_LIMITS[:, 0], SOFT_LIMITS[:, 1])

        # Critically damped spring per joint, velocity-limited
        w = STIFFNESS * self.energy
        acc = w * w * (target - self.q) - 2 * w * self.v
        self.v = np.clip(self.v + acc * dt, -MAX_VEL, MAX_VEL)
        self.q += self.v * dt

        # Additive layers: gestures + breathing
        offset = np.zeros(5)
        alive = []
        for frames, start in self.gestures:
            gt = self.t - start
            if gt <= frames[-1][0]:
                offset += _sample(frames, gt)
                alive.append([frames, start])
        self.gestures = alive
        breath = math.sin(2 * math.pi * 0.22 * self.t)
        offset[SHOULDER] += 0.012 * breath
        offset[HEAD] -= 0.018 * breath

        qpos = np.clip(self.q + offset, SOFT_LIMITS[:, 0], SOFT_LIMITS[:, 1])
        if self._last_qpos is not None and dt > 0:
            self.speed = float(np.max(np.abs(qpos - self._last_qpos)) / dt)
        self._last_qpos = qpos
        self.d.qpos[self.qadr] = qpos
        self._update_light(dt)
        mujoco.mj_forward(self.m, self.d)

    def _update_light(self, dt):
        k = 1 - math.exp(-4.0 * dt)   # ~0.25 s fade
        self.color += (self.target_color - self.color) * k
        self.intensity += (self.target_intensity - self.intensity) * k
        wave = 0.5 + 0.5 * math.sin(2 * math.pi * 1.1 * self.t)
        eff = self.intensity * (1 - 0.45 * self.pulse * wave)

        off = np.array([0.35, 0.35, 0.35])           # unlit bulb colour
        rgb = off + (self.color - off) * eff
        for g in self.glow:
            self.m.geom_rgba[g, :3] = rgb
        if self.spot >= 0:
            self.m.light_diffuse[self.spot] = self.color * eff * 1.2


# ---------- standalone demo: python src/body.py ----------
def demo():
    import mujoco.viewer

    body = Body()
    script = [
        (0.0,  "sleep, dim",             lambda: (body.set_pose("sleep", 0.6), body.set_light(WARM, 0.15, 0.0))),
        (3.0,  "wake: perk + bright",    lambda: (body.set_pose("attentive", 1.3), body.gesture("perk"), body.set_light(WARM, 1.0))),
        (5.0,  "look to lamp's left",    lambda: body.look_at(0.6, 0.1)),
        (6.5,  "look to lamp's right",   lambda: body.look_at(-0.6, 0.0)),
        (8.0,  "centre + nod",           lambda: (body.clear_gaze(), body.gesture("nod"))),
        (9.5,  "curious tilt",           lambda: body.gesture("tilt")),
        (12.0, "thinking: cool pulse",   lambda: (body.set_pose("thinking"), body.set_light(COOL, 0.8, 1.0))),
        (15.0, "lean in, warm",          lambda: (body.set_pose("lean_in"), body.set_light(WARM, 1.0, 0.0))),
        (17.0, "happy bounce",           lambda: body.gesture("bounce")),
        (18.5, "shake (no)",             lambda: body.gesture("shake")),
        (20.0, "droop back to sleep",    lambda: (body.gesture("droop"), body.set_pose("sleep", 0.6), body.set_light(intensity=0.15))),
    ]
    period = 24.0

    with mujoco.viewer.launch_passive(body.m, body.d) as v:
        v.cam.lookat[:] = [-0.1, 0.0, 0.35]
        v.cam.distance = 1.5
        v.cam.elevation = -12
        v.cam.azimuth = 180
        start = last = time.time()
        cycle, fired = -1, set()
        while v.is_running():
            now = time.time()
            dt = min(now - last, 0.05)
            last = now
            elapsed = now - start
            if int(elapsed // period) != cycle:
                cycle, fired = int(elapsed // period), set()
            local = elapsed % period
            for i, (ts, label, action) in enumerate(script):
                if local >= ts and i not in fired:
                    fired.add(i)
                    print(f"[{local:5.1f}s] {label}")
                    action()
            with v.lock():
                body.update(dt)
            v.sync()
            time.sleep(1 / 60)


if __name__ == "__main__":
    demo()