"""
Check which GPU the OpenGL context actually uses (the same path as the
MuJoCo viewer), and whether CUDA is available to PyTorch (if installed).

Usage:
    python gpu_check.py
"""
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def check_opengl() -> None:
    import ctypes

    import glfw

    if not glfw.init():
        print("[OpenGL] failed to initialize GLFW")
        return

    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    win = glfw.create_window(64, 64, "gpu-probe", None, None)
    if not win:
        print("[OpenGL] failed to create a window/context")
        glfw.terminate()
        return

    glfw.make_context_current(win)

    gl = ctypes.windll.opengl32
    gl.glGetString.restype = ctypes.c_char_p
    GL_VENDOR, GL_RENDERER, GL_VERSION = 0x1F00, 0x1F01, 0x1F02

    def s(name):
        raw = gl.glGetString(name)
        return raw.decode(errors="replace") if raw else "?"

    print("[OpenGL] Vendor  :", s(GL_VENDOR))
    print("[OpenGL] Renderer:", s(GL_RENDERER))
    print("[OpenGL] Version :", s(GL_VERSION))

    glfw.destroy_window(win)
    glfw.terminate()


def check_cuda() -> None:
    try:
        import torch
    except ImportError:
        print("[CUDA]   torch is not installed (will be added for the training stage)")
        return
    print(f"[CUDA]   torch {torch.__version__}, "
          f"cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[CUDA]   device: {torch.cuda.get_device_name(0)}")


if __name__ == "__main__":
    check_opengl()
    check_cuda()
