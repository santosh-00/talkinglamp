"""Load the lamp URDF in MuJoCo, print its structure, and sweep each joint in turn."""
import math
import os
import time
import glob
import mujoco
import mujoco.viewer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import glob
URDF = glob.glob(os.path.join(ROOT, "robot", "*.urdf"))[0]
print(f"Loading {URDF}")
SECONDS_PER_JOINT = 4.0

def main():
    m = mujoco.MjModel.from_xml_path(URDF)
    d = mujoco.MjData(m)

    print(f"\n{'idx':>3}  {'joint':<26} {'lower':>7} {'upper':>7}  body")
    print("-" * 70)
    movable = []
    for j in range(m.njnt):
        lo, hi = m.jnt_range[j]
        print(f"{j:>3}  {m.joint(j).name:<26} {lo:>7.3f} {hi:>7.3f}  {m.body(m.jnt_bodyid[j]).name}")
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
            movable.append(j)

    print("\nBodies:")
    for b in range(m.nbody):
        print(f"  {b:>2}  {m.body(b).name or '(world)':<24} parent={m.body_parentid[b]}")

    print(f"\nSweeping {len(movable)} joints, {SECONDS_PER_JOINT:.0f}s each (watch the terminal for which one)\n")

    # Kinematic only: we set joint angles directly, no physics stepping.
    with mujoco.viewer.launch_passive(m, d) as viewer:
        start = time.time()
        current = -1
        while viewer.is_running():
            t = time.time() - start
            idx = int(t // SECONDS_PER_JOINT) % len(movable)
            j = movable[idx]
            if idx != current:
                current = idx
                lo, hi = m.jnt_range[j]
                print(f"→ now sweeping: {m.joint(j).name}  (range {lo:+.2f} … {hi:+.2f})")

            d.qpos[:] = m.qpos0
            lo, hi = m.jnt_range[j]
            mid, amp = (lo + hi) / 2, 0.45 * (hi - lo)
            phase = t % SECONDS_PER_JOINT
            # Start each sweep at the upper limit so you can see which sign goes where
            d.qpos[m.jnt_qposadr[j]] = mid + amp * math.cos(2 * math.pi * phase / SECONDS_PER_JOINT)

            mujoco.mj_forward(m, d)
            viewer.sync()
            time.sleep(1 / 60)

if __name__ == "__main__":
    main()