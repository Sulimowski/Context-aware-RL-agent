"""
Viewer / pose editor for OpenArm MJCF models in the interactive MuJoCo viewer.

Usage:
    python view.py                # bimanual arm (v1/scene.xml)
    python view.py single         # single arm (v1/openarm.xml)
    python view.py <path.xml>     # any model

Two modes (toggle with SPACE):
    PAUSE (default) — kinematic posing: you set joint angles directly and the
        arm snaps to them instantly (no dynamics, nothing falls).
    PHYSICS — real dynamics; a PD controller holds the target pose,
        so the arm does not collapse under gravity.

Joints and grippers:
    1..9         quick joint selection (the index is printed to the console)
    ,  /  .      cycle through ALL joints (for the 18-actuator bimanual arm)
    [  /  ]      decrease / increase the selected joint angle
    O  /  C      open / close all grippers
    R            reset the pose and camera

Camera (always active, plus mouse: LMB rotate, RMB pan, wheel zoom):
    arrows ← →   orbit around the robot (azimuth)
    arrows ↑ ↓   tilt (elevation)
    W / S        zoom in / out
    A / D        pan the look-at point left / right
    Q / E        pan the look-at point down / up
"""
import math
import sys
import time
from pathlib import Path

# the Windows console defaults to a legacy code page and chokes on non-ASCII — force utf-8
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except (AttributeError, ValueError):
        pass

import mujoco
import mujoco.viewer
import numpy as np

HERE = Path(__file__).parent
MODELS = {
    "bimanual": HERE / "openarm_mujoco" / "v1" / "scene.xml",
    "single": HERE / "openarm_mujoco" / "v1" / "openarm.xml",
}

CAM_HOME = dict(azimuth=160.0, elevation=-20.0, distance=2.0, lookat=(0.0, 0.0, 0.6))

FPS_CAP = 60   # pacing cap (on Optimus laptops rendering tops out at ~40 anyway)

# GLFW key codes
K_SPACE, K_R = 32, 82
K_W, K_S, K_A, K_D, K_Q, K_E = 87, 83, 65, 68, 81, 69
K_RIGHT, K_LEFT, K_DOWN, K_UP = 262, 263, 264, 265
K_EQUAL, K_MINUS = 61, 45
K_LBRACKET, K_RBRACKET = 91, 93
K_COMMA, K_PERIOD = 44, 46
K_O, K_C = 79, 67

PAN_STEP = 0.05        # m per key press (camera)
ORBIT_STEP = 5.0       # degrees per key press (camera)
ZOOM_FACTOR = 0.9
JOINT_STEP = 0.10      # rad per key press (joint)

# PD controller that holds the pose (PHYSICS mode)
KP = 60.0
KD = 5.0


def resolve(arg):
    if arg is None:
        return MODELS["bimanual"]
    return MODELS.get(arg, Path(arg))


def apply_cam_home(cam):
    cam.azimuth = CAM_HOME["azimuth"]
    cam.elevation = CAM_HOME["elevation"]
    cam.distance = CAM_HOME["distance"]
    cam.lookat[:] = CAM_HOME["lookat"]


class Rig:
    """Maps actuators to joints and manages the target joint angles."""

    def __init__(self, model, data):
        self.model, self.data = model, data
        self.qadr, self.dadr, self.jid, self.names = [], [], [], []
        for a in range(model.nu):
            if model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT:
                j = int(model.actuator_trnid[a, 0])
                self.jid.append(j)
                self.qadr.append(int(model.jnt_qposadr[j]))
                self.dadr.append(int(model.jnt_dofadr[j]))
            else:
                self.jid.append(-1)
                self.qadr.append(-1)
                self.dadr.append(-1)
            nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
            self.names.append(nm or f"act{a}")
        # a position servo (<position>) treats ctrl as the target position,
        # while a <motor> treats it as torque: handle them differently in physics
        self.is_position = [
            model.actuator_biastype[a] == mujoco.mjtBias.mjBIAS_AFFINE
            for a in range(model.nu)
        ]
        # finger (gripper) actuators — matched by name
        self.finger_idx = [a for a, n in enumerate(self.names) if "finger" in n]
        self.reset_target()

    def reset_target(self):
        self.target = np.array(
            [self.data.qpos[q] if q >= 0 else 0.0 for q in self.qadr]
        )

    def nudge(self, a, delta):
        if not (0 <= a < len(self.qadr)) or self.qadr[a] < 0:
            return
        j = self.jid[a]
        t = self.target[a] + delta
        if self.model.jnt_limited[j]:
            lo, hi = self.model.jnt_range[j]
            t = min(hi, max(lo, t))
        self.target[a] = t

    def set_grippers(self, opened):
        """Open (opened=True) or close all grippers, using the joint limits."""
        for a in self.finger_idx:
            j = self.jid[a]
            if j >= 0 and self.model.jnt_limited[j]:
                lo, hi = self.model.jnt_range[j]
            else:
                lo, hi = 0.0, 0.044
            self.target[a] = hi if opened else lo

    def apply_kinematic(self):
        """PAUSE: write the target angles straight into qpos and run forward kinematics."""
        for a, q in enumerate(self.qadr):
            if q >= 0:
                self.data.qpos[q] = self.target[a]
        mujoco.mj_forward(self.model, self.data)

    def apply_pd(self):
        """PHYSICS: PD torque towards the target angles, respecting ctrlrange."""
        for a, (q, d) in enumerate(zip(self.qadr, self.dadr)):
            if q < 0:
                continue
            if self.is_position[a]:
                # <position>: ctrl is the target position itself
                self.data.ctrl[a] = self.target[a]
                continue
            err = self.target[a] - self.data.qpos[q]
            tau = KP * err - KD * self.data.qvel[d]
            lo, hi = self.model.actuator_ctrlrange[a]
            if self.model.actuator_ctrllimited[a]:
                tau = min(hi, max(lo, tau))
            self.data.ctrl[a] = tau


def handle_key(key, cam, state, rig):
    if key == K_SPACE:
        state["paused"] = not state["paused"]
        print(f"[mode] {'PAUSE (kinematic)' if state['paused'] else 'PHYSICS (PD)'}")
        return
    if key == K_R:
        apply_cam_home(cam)
        mujoco.mj_resetData(rig.model, rig.data)
        if rig.model.nkey > 0:
            mujoco.mj_resetDataKeyframe(rig.model, rig.data, 0)
        mujoco.mj_forward(rig.model, rig.data)
        rig.reset_target()
        return

    # grippers: open / close all
    if key == K_O:
        rig.set_grippers(True)
        print("[grippers] OPEN")
        return
    if key == K_C:
        rig.set_grippers(False)
        print("[grippers] CLOSE")
        return

    # joint selection: digits 1..9 (quick access) or ,/. (cycle through all)
    if 49 <= key <= 57:
        a = key - 49
        if 0 <= a < rig.model.nu:
            state["sel"] = a
            print(f"[joint {a + 1}] {rig.names[a]}  angle={rig.target[a]:+.3f} rad")
        return
    if key in (K_COMMA, K_PERIOD):
        step = -1 if key == K_COMMA else 1
        state["sel"] = (state["sel"] + step) % rig.model.nu
        a = state["sel"]
        print(f"[joint {a + 1}] {rig.names[a]}  angle={rig.target[a]:+.3f} rad")
        return
    if key == K_LBRACKET:
        rig.nudge(state["sel"], -JOINT_STEP)
        print(f"[joint {state['sel'] + 1}] {rig.names[state['sel']]}  "
              f"angle={rig.target[state['sel']]:+.3f} rad")
        return
    if key == K_RBRACKET:
        rig.nudge(state["sel"], +JOINT_STEP)
        print(f"[joint {state['sel'] + 1}] {rig.names[state['sel']]}  "
              f"angle={rig.target[state['sel']]:+.3f} rad")
        return

    # camera
    if key == K_LEFT:
        cam.azimuth -= ORBIT_STEP
    elif key == K_RIGHT:
        cam.azimuth += ORBIT_STEP
    elif key == K_UP:
        cam.elevation = min(89.0, cam.elevation + ORBIT_STEP)
    elif key == K_DOWN:
        cam.elevation = max(-89.0, cam.elevation - ORBIT_STEP)
    elif key in (K_W, K_EQUAL):
        cam.distance *= ZOOM_FACTOR
    elif key in (K_S, K_MINUS):
        cam.distance /= ZOOM_FACTOR
    elif key == K_Q:
        cam.lookat[2] -= PAN_STEP
    elif key == K_E:
        cam.lookat[2] += PAN_STEP
    elif key in (K_A, K_D):
        az = math.radians(cam.azimuth)
        rx, ry = math.sin(az), -math.cos(az)
        sign = -1.0 if key == K_A else 1.0
        cam.lookat[0] += sign * rx * PAN_STEP
        cam.lookat[1] += sign * ry * PAN_STEP


def main():
    # Windows: raise the timer resolution, otherwise time.sleep caps FPS at ~30
    try:
        import ctypes
        ctypes.windll.winmm.timeBeginPeriod(1)
    except (AttributeError, OSError):
        ctypes = None

    path = resolve(sys.argv[1] if len(sys.argv) > 1 else None)
    print(f"Loading model: {path}")
    model = mujoco.MjModel.from_xml_path(str(path))
    # scene.xml asks for 8192×8192 shadows — that kills FPS on a laptop GPU.
    # Lower the shadow quality in memory only (the file itself is untouched).
    model.vis.quality.shadowsize = 2048
    model.vis.quality.offsamples = 2   # less MSAA — every frame is cheaper
    data = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    rig = Rig(model, data)
    print(f"nq={model.nq} nv={model.nv} nu={model.nu} "
          f"nbody={model.nbody} ngeom={model.ngeom}  timestep={model.opt.timestep}")
    print("Joints:", ", ".join(f"{i + 1}:{n}" for i, n in enumerate(rig.names)))
    print("PAUSE (kinematic): joint — 1..9 or ,/. ; angle — [ and ] ; "
          "grippers — O/C ; physics — SPACE.")

    state = {"paused": True, "sel": 0}
    pending = []
    substeps = max(1, round((1.0 / 60.0) / model.opt.timestep))
    frames, t_report = 0, time.time()

    with mujoco.viewer.launch_passive(
        model, data, key_callback=pending.append
    ) as viewer:
        apply_cam_home(viewer.cam)
        while viewer.is_running():
            frame_start = time.time()
            with viewer.lock():
                while pending:
                    handle_key(pending.pop(0), viewer.cam, state, rig)
                if state["paused"]:
                    rig.apply_kinematic()
                else:
                    for _ in range(substeps):
                        rig.apply_pd()
                        mujoco.mj_step(model, data)
            viewer.sync()

            frames += 1
            now = time.time()
            if now - t_report >= 2.0:
                print(f"[FPS] {frames / (now - t_report):.1f}  "
                      f"(mode: {'pause' if state['paused'] else 'physics'})")
                frames, t_report = 0, now

            rem = (1.0 / FPS_CAP) - (now - frame_start)
            if rem > 0:
                time.sleep(rem)

    if ctypes is not None:
        try:
            ctypes.windll.winmm.timeEndPeriod(1)
        except OSError:
            pass


if __name__ == "__main__":
    main()
