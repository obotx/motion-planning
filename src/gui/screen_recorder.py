
import os
import time

import numpy as np
import glfw


class ScreenRecorder:

    def __init__(self, fps=30, output_dir=None):
        self.fps = fps
        self._writer = None
        self._path = None
        self._frame_count = 0
        self._size = None
        if output_dir is None:
            output_dir = os.path.abspath(
                os.path.join(os.path.dirname(__file__),
                             "..", "..", "output_videos"))
        self._output_dir = output_dir
        try:
            import cv2
            from OpenGL import GL
            self._cv2 = cv2
            self._GL = GL
            self._available = True
        except Exception as e:
            print(f"[REC] Recording disabled (missing dependency: {e})")
            self._cv2 = None
            self._GL = None
            self._available = False

    @property
    def available(self):
        return self._available

    @property
    def active(self):
        return self._writer is not None

    def start(self, label="play_m1_session"):
        if not self._available:
            return
        if self.active:
            return
        os.makedirs(self._output_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        self._path = os.path.join(self._output_dir, f"{label}_{ts}.mp4")
        self._frame_count = 0
        self._size = None
        print(f"[REC] Recording will be saved to {self._path}")
        self._writer = "pending"

    def _open_writer(self, w, h):
        if self._cv2 is None:
            return
        fourcc = self._cv2.VideoWriter_fourcc(*"mp4v")
        try:
            writer = self._cv2.VideoWriter(
                self._path, fourcc, float(self.fps), (int(w), int(h)))
            if not writer.isOpened():
                print(f"[REC] cv2.VideoWriter failed to open: {self._path}")
                self._writer = None
                return
            self._writer = writer
            self._size = (w, h)
        except Exception as e:
            print(f"[REC] open writer error: {e}")
            self._writer = None

    def capture_frame(self, window):
        if not self.active or self._GL is None or self._cv2 is None:
            return
        try:
            w, h = glfw.get_framebuffer_size(window)
            if w <= 0 or h <= 0:
                return
            if self._writer == "pending":
                self._open_writer(w, h)
                if not self.active:
                    return
            if (w, h) != self._size:
                return
            pixels = self._GL.glReadPixels(
                0, 0, w, h, self._GL.GL_RGB, self._GL.GL_UNSIGNED_BYTE)
            frame = np.frombuffer(pixels, dtype=np.uint8).reshape(h, w, 3)
            frame = np.flipud(frame)
            frame = self._cv2.cvtColor(frame, self._cv2.COLOR_RGB2BGR)
            self._writer.write(frame)
            self._frame_count += 1
        except Exception as e:
            print(f"[REC] capture_frame error: {e}")

    def stop(self):
        if not self.active:
            return
        if self._writer != "pending":
            try:
                self._writer.release()
                print(f"[REC] Saved {self._frame_count} frames to {self._path}")
            except Exception as e:
                print(f"[REC] release error: {e}")
        self._writer = None
        self._frame_count = 0
        self._size = None
        self._path = None
