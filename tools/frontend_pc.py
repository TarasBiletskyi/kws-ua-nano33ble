"""ctypes wrapper around the DLL of firmware/kws_nano33/frontend.c (tools/build_frontend_pc.py)."""
from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np

DLL = Path(__file__).resolve().parent / "out" / "frontend_pc" / "frontend.dll"
N_FRAMES_RAW, N_MELS, N_FRAMES, N_FEAT, FRAME = 97, 40, 98, 39, 512


class FrontendC:
    def __init__(self, dll: Path = DLL):
        self.lib = ctypes.CDLL(str(dll))
        self.lib.fe_state_size.restype = ctypes.c_uint32
        self.state = ctypes.create_string_buffer(self.lib.fe_state_size())
        self.reset()

    def reset(self) -> None:
        self.lib.fe_reset(self.state)

    def push_frame(self, samples: np.ndarray) -> None:
        x = np.asarray(samples, dtype=np.int16)
        assert x.shape == (FRAME,)
        self.lib.fe_push_frame(self.state, x.ctypes.data_as(ctypes.c_void_p))

    def window_ready(self) -> bool:
        return bool(self.lib.fe_window_ready(self.state))

    def finish_window(self) -> tuple[np.ndarray, np.ndarray]:
        """(int8 98×39, float32 98×39)."""
        out = np.empty((N_FRAMES, N_FEAT), dtype=np.int8)
        feat = np.empty((N_FRAMES, N_FEAT), dtype=np.float32)
        self.lib.fe_finish_window(self.state, out.ctypes.data_as(ctypes.c_void_p), feat.ctypes.data_as(ctypes.c_void_p))
        return out, feat

    def process_clip(self, pcm16: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(pcm16, dtype=np.int16)
        assert x.shape == (16000,)
        out = np.empty((N_FRAMES, N_FEAT), dtype=np.int8)
        feat = np.empty((N_FRAMES, N_FEAT), dtype=np.float32)
        self.lib.fe_process_clip(self.state, x.ctypes.data_as(ctypes.c_void_p), out.ctypes.data_as(ctypes.c_void_p),
                                 feat.ctypes.data_as(ctypes.c_void_p))
        return out, feat

    def logmel(self) -> np.ndarray:
        out = np.empty((N_FRAMES_RAW, N_MELS), dtype=np.float32)
        self.lib.fe_get_logmel(self.state, out.ctypes.data_as(ctypes.c_void_p))
        return out

    def power_spectrum(self, frame: np.ndarray) -> np.ndarray:
        x = np.asarray(frame, dtype=np.float32)
        assert x.shape == (FRAME,)
        p = np.empty(257, dtype=np.float32)
        self.lib.fe_power_spectrum(x.ctypes.data_as(ctypes.c_void_p), p.ctypes.data_as(ctypes.c_void_p))
        return p
