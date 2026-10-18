"""
Headless smoke tests: the arena loads, the actuators move the cube in the
expected directions, the walls keep it inside, and the LLM reply parsers work.

Usage (from the repo root):
    python -m pytest tests
"""
import importlib
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ARENA = ROOT / "test1" / "arena.xml"


def make_sim():
    model = mujoco.MjModel.from_xml_path(str(ARENA))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def run(model, data, fx, fy, seconds):
    data.ctrl[model.actuator("fx").id] = fx
    data.ctrl[model.actuator("fy").id] = fy
    for _ in range(int(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)
    return data.xpos[model.body("cube").id][:2].copy()


def test_arena_loads():
    model, _ = make_sim()
    assert model.nu == 2
    for body in ("cube", "cube2"):
        assert model.body(body).id >= 0
    for site in ("target", "RIGHT_UP", "LEFT_UP", "LEFT_DOWN", "RIGHT_DOWN"):
        assert model.site(site).id >= 0


@pytest.mark.parametrize("fx, fy, axis, sign", [
    (1, 0, 0, +1), (-1, 0, 0, -1), (0, 1, 1, +1), (0, -1, 1, -1),
])
def test_force_moves_cube(fx, fy, axis, sign):
    model, data = make_sim()
    start = data.xpos[model.body("cube").id][:2].copy()
    end = run(model, data, fx, fy, seconds=0.5)
    assert sign * (end[axis] - start[axis]) > 0.01


def test_walls_contain_cube():
    model, data = make_sim()
    end = run(model, data, 1, -1, seconds=5.0)
    assert np.all(np.abs(end) < 0.4)


def _load(module):
    sys.path.insert(0, str(ROOT / "test1"))
    try:
        return importlib.import_module(module)
    except Exception as exc:  # the viewer needs OpenGL, which a headless CI box may lack
        pytest.skip(f"cannot import {module}: {exc}")
    finally:
        sys.path.pop(0)


def test_parse_force():
    parse_force = _load("llm_control").parse_force
    assert parse_force("x is smaller → push right\nFORCE: 0.5, -1") == (0.5, -1.0)
    assert parse_force("I'd go with 1 and 0") == (1.0, 0.0)
    assert parse_force("no numbers here") == (0.0, 0.0)


def test_parse_plan():
    parse_plan = _load("llm_drive").parse_plan
    plan = parse_plan("first the center, then a corner\nPLAN: (0, 0); (0.9, -0.35)")
    assert [p.tolist() for p in plan] == [[0.0, 0.0], [0.375, -0.35]]
