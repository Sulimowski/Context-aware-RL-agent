"""
An LLM (qwen2.5:3b via Ollama) controls the cube — WITH A WINDOW (viewer).
Chain-of-thought version: the model REASONS first, then answers 'FORCE: fx, fy'.

The fast loop (physics + rendering, ~60 fps) runs in the main thread under the
current force. A separate "brain" thread queries the LLM and updates the force.

Usage (from the repo root):
    python test1/llm_control.py
Requires a running Ollama server with the qwen2.5:3b model.

WHERE TO EDIT THE PROMPT: the ask_llm function below, the prompt = (...) block.
"""
import json
import re
import sys
import threading
import time
import urllib.request
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
OLLAMA = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:3b"

DECISION_DT = 0.4     # how often we query the LLM (s)
REACH = 0.05          # "reached the target" distance


def parse_force(text):
    """Extract fx, fy from the reply. Expects a line like 'FORCE: 1, 0'."""
    idx = text.upper().rfind("FORCE")
    if idx != -1:
        nums = re.findall(r"-?\d+\.?\d*", text[idx:])
        if len(nums) >= 2:
            return float(nums[0]), float(nums[1])
    nums = re.findall(r"-?\d+\.?\d*", text)          # fallback
    if len(nums) >= 2:
        return float(nums[-2]), float(nums[-1])
    return 0.0, 0.0


def ask_llm(cube, target, dist, prev_action, prev_dist):
    """Brain: state -> force (fx, fy) via reasoning (chain-of-thought)."""
    prev = ("none" if prev_dist is None else
            f"fx={prev_action[0]:+.2f}, fy={prev_action[1]:+.2f}, "
            f"distance was {prev_dist:.3f}")

    # ↓↓↓↓↓↓↓↓↓↓  HERE IS THE PROMPT — EDIT IT HERE  ↓↓↓↓↓↓↓↓↓↓
    prompt = (
        "You control a cube on a 2D field (walls at ±0.40) with a force fx, fy in [-1, 1].\n"
        "fx>0 pushes RIGHT (increases x), fx<0 pushes left. "
        "fy>0 pushes FORWARD (increases y), fy<0 pushes backward.\n"
        f"Cube: x={cube[0]:.3f}, y={cube[1]:.3f}. "
        f"Target: x={target[0]:.3f}, y={target[1]:.3f}. Distance: {dist:.3f}.\n"
        f"Previous step: {prev}. "
        "(If the distance grew after the previous action, you pushed the wrong way — turn around.)\n"
        "Reason BRIEFLY step by step: compare the cube's x with the target's x → sign of fx; "
        "compare y → sign of fy. "
        "At the VERY END, on the last line, output exactly: FORCE: fx, fy"
    )
    # ↑↑↑↑↑↑↑↑↑↑  END OF PROMPT  ↑↑↑↑↑↑↑↑↑↑

    body = json.dumps({
        "model": MODEL, "prompt": prompt, "stream": False,
        "keep_alive": "10m", "options": {"temperature": 0},
    }).encode()
    req = urllib.request.Request(
        OLLAMA, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        out = json.loads(resp.read())
    fx, fy = parse_force(out["response"])
    return float(np.clip(fx, -1, 1)), float(np.clip(fy, -1, 1))


def main():
    try:
        import ctypes
        ctypes.windll.winmm.timeBeginPeriod(1)
    except (AttributeError, OSError):
        ctypes = None

    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5)
    except Exception:
        print("Ollama is not responding on :11434. Start the server and pull qwen2.5:3b.")
        return

    model = mujoco.MjModel.from_xml_path(str(ARENA))
    data = mujoco.MjData(model)
    ax = model.actuator("fx").id
    ay = model.actuator("fy").id
    sid = model.site("target").id
    target = model.site_pos[sid][:2].copy()
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    shared = {"cube": np.zeros(2), "force": np.zeros(2), "running": True}
    print(f"Target: ({target[0]:.2f}, {target[1]:.2f}). A window will open and the LLM "
          f"will steer the cube (chain-of-thought — slower, but correct).")

    def brain():
        prev_action, prev_dist = (0.0, 0.0), None
        while shared["running"]:
            cube = shared["cube"].copy()
            dist = float(np.linalg.norm(cube - target))
            if dist < REACH:
                shared["force"] = np.zeros(2)
                print(f"✅ REACHED! distance {dist:.3f}")
                time.sleep(DECISION_DT)
                continue
            t0 = time.time()
            fx, fy = ask_llm(cube, target, dist, prev_action, prev_dist)
            ms = (time.time() - t0) * 1000
            shared["force"] = np.array([fx, fy])
            print(f"cube=({cube[0]:+.2f},{cube[1]:+.2f}) dist={dist:.3f} "
                  f"→ LLM fx={fx:+.2f} fy={fy:+.2f} ({ms:.0f} ms)")
            prev_action, prev_dist = (fx, fy), dist

    substeps = max(1, round((1.0 / 60.0) / model.opt.timestep))
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -75
        viewer.cam.distance = 1.4
        viewer.cam.lookat[:] = (0.0, 0.0, 0.0)
        th = threading.Thread(target=brain, daemon=True)
        th.start()
        while viewer.is_running():
            t0 = time.time()
            with viewer.lock():
                shared["cube"] = data.qpos[:2].copy()
                data.ctrl[ax], data.ctrl[ay] = shared["force"]
                for _ in range(substeps):
                    mujoco.mj_step(model, data)
            viewer.sync()
            rem = (1.0 / 60.0) - (time.time() - t0)
            if rem > 0:
                time.sleep(rem)
        shared["running"] = False

    if ctypes is not None:
        try:
            ctypes.windll.winmm.timeEndPeriod(1)
        except OSError:
            pass


if __name__ == "__main__":
    main()
