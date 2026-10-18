"""
Manual (keyboard) control of the cube in the test1 arena — a sanity check of the
physics and the action space before any learning is plugged in.

Usage (from the repo root):
    python test1/play.py

Controls:
    W/A/S/D or arrows   add force along +y / -x / -y / +x
    SPACE               zero the force (stop pushing)
    T                   place a new random target at one of the edges
    R                   reset the simulation
    mouse               camera
"""
import sys
import time
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except (AttributeError, ValueError):
        pass

import mujoco
import mujoco.viewer
import numpy as np

ARENA = Path(__file__).parent / "arena.xml"

K_SPACE, K_R, K_T = 32, 82, 84
K_W, K_S, K_A, K_D = 87, 83, 65, 68
K_UP, K_DOWN, K_LEFT, K_RIGHT = 265, 264, 263, 262

FORCE_STEP = 0.15  # how much the force changes per key press (ctrl in [-1, 1])
ARENA_HALF = 0.37  # where to put the target near the edge


def randomize_target(model, data, rng):
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "target")
    if sid < 0:
        return
    side = rng.integers(0, 4)
    along = rng.uniform(-0.35, 0.35)
    pos = [(ARENA_HALF, along), (-ARENA_HALF, along),
           (along, ARENA_HALF), (along, -ARENA_HALF)][side]
    model.site_pos[sid][:2] = pos
    mujoco.mj_forward(model, data)
    print(f"[target] edge {side}: ({pos[0]:+.2f}, {pos[1]:+.2f})")


def handle_key(key, model, data, rng):
    ctrl = data.ctrl
    if key in (K_D, K_RIGHT):
        ctrl[0] = np.clip(ctrl[0] + FORCE_STEP, -1, 1)
    elif key in (K_A, K_LEFT):
        ctrl[0] = np.clip(ctrl[0] - FORCE_STEP, -1, 1)
    elif key in (K_W, K_UP):
        ctrl[1] = np.clip(ctrl[1] + FORCE_STEP, -1, 1)
    elif key in (K_S, K_DOWN):
        ctrl[1] = np.clip(ctrl[1] - FORCE_STEP, -1, 1)
    elif key == K_SPACE:
        ctrl[:] = 0.0
    elif key == K_T:
        randomize_target(model, data, rng)
        return
    elif key == K_R:
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        return
    print(f"[force] fx={ctrl[0]:+.2f}  fy={ctrl[1]:+.2f}")


def main():
    try:
        import ctypes
        ctypes.windll.winmm.timeBeginPeriod(1)
    except (AttributeError, OSError):
        ctypes = None

    model = mujoco.MjModel.from_xml_path(str(ARENA))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    rng = np.random.default_rng()

    print("Physics is ON. Drive the cube: W/A/S/D or arrows, SPACE — stop, "
          "T — new target, R — reset. Camera — mouse.")

    pending = []
    substeps = max(1, round((1.0 / 60.0) / model.opt.timestep))

    with mujoco.viewer.launch_passive(
        model, data, key_callback=pending.append
    ) as viewer:
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -55
        viewer.cam.distance = 1.7
        viewer.cam.lookat[:] = (0.0, 0.0, 0.0)
        while viewer.is_running():
            t0 = time.time()
            with viewer.lock():
                while pending:
                    handle_key(pending.pop(0), model, data, rng)
                for _ in range(substeps):
                    mujoco.mj_step(model, data)
            viewer.sync()
            rem = (1.0 / 60.0) - (time.time() - t0)
            if rem > 0:
                time.sleep(rem)

    if ctypes is not None:
        try:
            ctypes.windll.winmm.timeEndPeriod(1)
        except OSError:
            pass


if __name__ == "__main__":
    main()
