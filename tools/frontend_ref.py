"""NumPy reference of the device frontend, built only from the exported tables.

This is the executable specification of ``frontend.c`` (knowledge base 03): framing with the
zero-padded periodic Hann window, power spectrum, sparse mel filterbank, 10·log10 with the
per-window top-dB clip, orthonormal DCT-II, Savitzky–Golay deltas with ``interp`` edges, the zero
row, per-window CMVN and int8 quantization. It is compared against ``src.features.extract_mfcc``
(the librosa pipeline used for training) in ``tools/tests/test_frontend_ref.py``.
"""
from __future__ import annotations

import numpy as np


def log_mel_frames(audio: np.ndarray, t: dict) -> np.ndarray:
    """(n_frames_raw, n_mels) unclipped 10·log10 mel power — the per-frame part of the device pipeline."""
    clip, n_fft, hop = t["clip_samples"], t["n_fft"], t["hop_length"]
    x = np.asarray(audio, dtype=np.float32)
    x = np.pad(x, (0, clip - len(x))) if len(x) < clip else x[:clip]
    win = np.zeros(n_fft, dtype=np.float32)
    win[t["win_offset"]:t["win_offset"] + t["win_length"]] = t["window"]
    n_raw = t["n_frames_raw"]
    frames = np.stack([x[i * hop:i * hop + n_fft] for i in range(n_raw)]) * win          # (97, 512)
    spec = np.fft.rfft(frames.astype(np.float64), axis=1)
    power = (spec.real ** 2 + spec.imag ** 2).astype(np.float32)                           # (97, 257)
    mel = np.zeros((n_raw, t["n_mels"]), dtype=np.float32)
    for i in range(t["n_mels"]):
        s, n, o = int(t["mel_start"][i]), int(t["mel_len"][i]), int(t["mel_offset"][i])
        mel[:, i] = power[:, s:s + n] @ t["mel_weights"][o:o + n]
    return (10.0 * np.log10(np.maximum(mel, t["amin"]))).astype(np.float32)


def sg_filter(x: np.ndarray, center: np.ndarray, head: np.ndarray, tail: np.ndarray) -> np.ndarray:
    """Savitzky–Golay derivative along axis 0 of (frames, coeffs) with ``mode='interp'`` edge handling."""
    n, w = x.shape[0], len(center)
    half = w // 2
    out = np.empty_like(x)
    for i in range(half, n - half):
        out[i] = center @ x[i - half:i + half + 1]
    for k in range(half):
        out[k] = head[k] @ x[0:w]
        out[n - half + k] = tail[k] @ x[n - w:n]
    return out


def finish_window(logmel: np.ndarray, t: dict) -> np.ndarray:
    """Per-window steps: top-dB clip, DCT, deltas, zero row, CMVN → (n_frames, n_features) float32."""
    s = np.maximum(logmel, logmel.max() - t["top_db"])                                     # (97, 40)
    mfcc = s @ t["dct"].T                                                                   # (97, 13)
    d1 = sg_filter(mfcc, t["sg1_center"], t["sg1_head"], t["sg1_tail"])
    d2 = sg_filter(mfcc, t["sg2_center"], t["sg2_head"], t["sg2_tail"])
    feat = np.concatenate([mfcc, d1, d2], axis=1)                                           # (97, 39)
    feat = np.pad(feat, ((0, t["n_frames"] - feat.shape[0]), (0, 0)))                       # (98, 39)
    mean = feat.mean(axis=0)
    std = feat.std(axis=0) + t["cmvn_eps"]
    return ((feat - mean) / std).astype(np.float32)


def features(audio: np.ndarray, t: dict) -> tuple[np.ndarray, np.ndarray]:
    """(log-mel 97×40, features 98×39) for one clip."""
    lm = log_mel_frames(audio, t)
    return lm, finish_window(lm, t)


def quantize(feat: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    """Round-to-nearest-even and clip, as the device does (``rintf``)."""
    return np.clip(np.rint(feat / scale) + zero_point, -128, 127).astype(np.int8)
