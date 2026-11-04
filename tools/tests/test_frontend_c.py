"""frontend.c (compiled for the PC) against the NumPy reference. Needs tools/build_frontend_pc.py to have run."""
from __future__ import annotations

import numpy as np
import pytest

from frontend_pc import DLL, FrontendC
from frontend_ref import features, quantize

pytestmark = pytest.mark.skipif(not DLL.is_file(), reason="run tools/build_frontend_pc.py first")


@pytest.fixture(scope="module")
def fe():
    return FrontendC()


@pytest.fixture(scope="module")
def tables_q():
    from export_tables import build_tables
    from kws_paths import TWO_OUTPUT_MODEL, load_config
    return build_tables(load_config(), TWO_OUTPUT_MODEL)


def speech_like(seed=0, n=16000):
    """Noise bursts on a tone: enough spectral and temporal variety for the whole pipeline."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / 16000.0
    env = np.clip(np.sin(2 * np.pi * 1.5 * t), 0, None) ** 2
    x = 0.3 * env * np.sin(2 * np.pi * (300 + 900 * t) * t) + 0.05 * env * rng.standard_normal(n) + 1e-3 * rng.standard_normal(n)
    return (np.clip(x, -1, 1) * 32767).astype(np.int16)


def test_power_spectrum_matches_numpy(fe):
    rng = np.random.default_rng(1)
    for x in (rng.standard_normal(512), np.sin(2 * np.pi * 37 * np.arange(512) / 512), np.ones(512), np.eye(1, 512, 3).ravel()):
        x = x.astype(np.float32)
        ref = np.abs(np.fft.rfft(x.astype(np.float64))) ** 2
        p = fe.power_spectrum(x)
        assert np.abs(p - ref).max() <= 1e-3 * max(ref.max(), 1.0)
        assert p.shape == (257,)


def test_logmel_and_features_match_reference(fe, tables_q):
    pcm = speech_like()
    lm_ref, feat_ref = features(pcm.astype(np.float32) / 32768.0, tables_q)
    out, feat = fe.process_clip(pcm)
    assert np.abs(fe.logmel() - lm_ref).max() < 2e-2
    assert np.abs(feat - feat_ref).max() < 5e-3
    q = tables_q["quant"]["input"]
    diff = np.abs(out.astype(int) - quantize(feat_ref, q["scale"], q["zero_point"]).astype(int))
    assert diff.max() <= 1 and (diff == 0).mean() > 0.98


def test_ring_gives_the_same_window_as_a_fresh_clip(fe):
    """A window taken after the ring wrapped around equals the same 97 frames pushed from an empty state."""
    pcm = speech_like(seed=3, n=16000 + 160 * 250)
    fe.reset()
    for t in range(97 + 250):
        fe.push_frame(pcm[t * 160:t * 160 + 512])
    assert fe.window_ready()
    a, fa = fe.finish_window()
    fe.reset()
    for t in range(250, 97 + 250):
        fe.push_frame(pcm[t * 160:t * 160 + 512])
    b, fb = fe.finish_window()
    assert np.array_equal(a, b) and np.array_equal(fa, fb)


def test_window_not_ready_before_97_frames(fe):
    fe.reset()
    z = np.zeros(512, dtype=np.int16)
    for _ in range(96):
        fe.push_frame(z)
    assert not fe.window_ready()
    fe.push_frame(z)
    assert fe.window_ready()


def test_digital_silence_is_finite(fe):
    out, feat = fe.process_clip(np.zeros(16000, dtype=np.int16))
    assert np.isfinite(feat).all()
    assert out.min() >= -128 and out.max() <= 127
