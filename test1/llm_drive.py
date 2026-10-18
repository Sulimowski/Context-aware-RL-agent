"""
Interactive cube driving with PLAIN TEXT — EMBODIED + PLAN.

The model reasons in the FIRST PERSON ("I am the orange cube") and produces a PLAN —
a list of waypoints. The controller drives the agent through them ONE BY ONE.
There is a blue cube object on the field (passive, pushable); both positions go into the prompt.

IMPORTANT: run it in YOUR OWN PyCharm terminal (keyboard input is required):
    python test1/llm_drive.py

Examples: "go to the blue cube", "visit all the corners", "go to the center, then to the bottom-right".
Empty input or "quit" — exit.
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

FIELD = 0.375       # coordinate limit
KP = 6.0            # P-controller gain
REACH_WP = 0.04     # "reached the waypoint" radius
MAX_WP = 6          # cap on the plan length


def parse_plan(text):
    """Extract a list of points from a line like 'PLAN: (x1,y1); (x2,y2); ...'."""
    idx = text.upper().rfind("PLAN")
    line = text[idx:].split("\n")[0] if idx != -1 else text
    nums = re.findall(r"-?\d+\.?\d*", line)
    pts = []
    for i in range(0, len(nums) - 1, 2):
        pts.append(np.array([
            float(np.clip(float(nums[i]), -FIELD, FIELD)),
            float(np.clip(float(nums[i + 1]), -FIELD, FIELD)),
        ]))
        if len(pts) >= MAX_WP:
            break
    return pts


def ask_plan(instruction, agent, obj):
    """Brain (first person): instruction + both positions -> PLAN of points."""
    # ↓↓↓ PROMPT — EDIT IT HERE ↓↓↓
    prompt = (
        "You are the ORANGE cube on a 2D field. You decide where YOU go.\n"
        "Field: coordinates in [-0.375, 0.375]. x>0 is right, x<0 is left; "
        "y>0 is forward/up, y<0 is backward/down; center (0,0); corners (±0.35, ±0.35).\n"
        "There is a BLUE cube nearby — an object (you can drive to it or push it).\n"
        f"You (orange) are now at: x={agent[0]:.3f}, y={agent[1]:.3f}.\n"
        f"Blue cube: x={obj[0]:.3f}, y={obj[1]:.3f}.\n"
        f"You were told: \"{instruction}\".\n"
        "Think BRIEFLY in the first person: where you need to move and in what order. "
        "Make a PLAN — the points you will visit one by one (1–5 of them).\n"
        "At the VERY END, on the last line, output exactly: PLAN: (x1,y1); (x2,y2); ..."
    )
    # ↑↑↑ END OF PROMPT ↑↑↑
    body = json.dumps({
        "model": MODEL, "prompt": prompt, "stream": True,
        "keep_alive": "10m", "options": {"temperature": 0},
    }).encode()
    req = urllib.request.Request(
        OLLAMA, data=body, headers={"Content-Type": "application/json"})
    parts = []
    print("\n  ┌─ cube's reasoning ─────────────")
    sys.stdout.write("  │ ")
    with urllib.request.urlopen(req, timeout=120) as resp:
        for raw in resp:
            raw = raw.strip()
            if not raw:
                continue
            chunk = json.loads(raw)
            tok = chunk.get("response", "")
            parts.append(tok)
            sys.stdout.write(tok.replace("\n", "\n  │ "))
            sys.stdout.flush()
            if chunk.get("done"):
                break
    print("\n  └────────────────────────────────")
    return parse_plan("".join(parts))


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
    tid = model.site("target").id
    b_agent = model.body("cube").id
    b_obj = model.body("cube2").id
    model.site_pos[tid][:2] = (0.0, 0.0)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    shared = {"agent": np.zeros(2), "obj": np.zeros(2),
              "plan": None, "wp": 0, "pending": None, "running": True}

    print("You steer the ORANGE cube with text. +x is right, +y is forward/up.")
    print("The blue cube is an object. Corners are labeled. Green marker = current plan waypoint.")
    print("Examples: 'go to the blue cube', 'visit all the corners', "
          "'go to the center, then to the bottom-right corner'.")
    print("Type an instruction and press Enter. Empty input or 'quit' — exit.\n")

    def input_loop():
        while shared["running"]:
            try:
                txt = input("instruction> ").strip()
            except EOFError:
                shared["running"] = False
                break
            if txt.lower() in ("", "quit", "exit"):
                shared["running"] = False
                break
            shared["pending"] = txt
            print(f"  [accepted] {txt}")

    def brain():
        while shared["running"]:
            instr = shared["pending"]
            if instr is None:
                time.sleep(0.1)
                continue
            shared["pending"] = None
            agent = shared["agent"].copy()
            obj = shared["obj"].copy()
            t0 = time.time()
            plan = ask_plan(instr, agent, obj)      # one LLM call per instruction
            ms = (time.time() - t0) * 1000
            shared["plan"], shared["wp"] = plan, 0
            pretty = " → ".join(f"({p[0]:+.2f},{p[1]:+.2f})" for p in plan) or "empty"
            print(f"  [plan] \"{instr}\": {pretty} ({ms:.0f} ms)")

    substeps = max(1, round((1.0 / 60.0) / model.opt.timestep))
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -75
        viewer.cam.distance = 1.4
        viewer.cam.lookat[:] = (0.0, 0.0, 0.0)
        viewer.opt.label = mujoco.mjtLabel.mjLABEL_SITE
        threading.Thread(target=input_loop, daemon=True).start()
        threading.Thread(target=brain, daemon=True).start()
        while viewer.is_running() and shared["running"]:
            t0 = time.time()
            with viewer.lock():
                agent = data.xpos[b_agent][:2].copy()
                shared["agent"] = agent
                shared["obj"] = data.xpos[b_obj][:2].copy()

                force = np.zeros(2)
                plan = shared["plan"]
                if plan:
                    wp = shared["wp"]
                    # skip waypoints that are already reached
                    while wp < len(plan) and np.linalg.norm(plan[wp] - agent) < REACH_WP:
                        wp += 1
                    shared["wp"] = wp
                    if wp < len(plan):
                        tgt = plan[wp]
                        model.site_pos[tid][:2] = tgt
                        force = np.clip(KP * (tgt - agent), -1, 1)

                data.ctrl[ax], data.ctrl[ay] = force
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
