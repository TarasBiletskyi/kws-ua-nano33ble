"""The table-driven NumPy frontend must reproduce the librosa pipeline used for training."""
import numpy as np
import pytest
import soundfile as sf

from kws_paths import default_corpus_root, list_test_clips

try:
    CORPUS = default_corpus_root()
except FileNotFoundError:
    CORPUS = None

pytestmark = pytest.mark.skipif(CORPUS is None, reason="corpus checkout not available")


def _sample_clips(cfg, per_class=1):
    clips = list_test_clips(CORPUS, cfg["classes"])
    out, seen = [], {}
    for p, cls, idx in clips:
        if seen.get(cls, 0) < per_class:
            out.append((p, cls, idx))
            seen[cls] = seen.get(cls, 0) + 1
    return out


def test_frontend_ref_matches_extract_mfcc(cfg, tables):
    from src.features import extract_mfcc
    from frontend_ref import features, quantize
    scale, zp = 0.06802022, -17
    worst = 0.0
    agree = []
    for p, cls, _ in _sample_clips(cfg, per_class=1):
        audio, sr = sf.read(str(p), dtype="float32", always_2d=False)
        assert sr == cfg["audio"]["sample_rate"]
        ref = extract_mfcc(audio, cfg)
        _, rep = features(audio, tables)
        assert rep.shape == ref.shape == (98, 39)
        d = float(np.abs(rep - ref).max())
        worst = max(worst, d)
        agree.append(float(np.mean(quantize(rep, scale, zp) == quantize(ref, scale, zp))))
    print(f"max |frontend_ref - extract_mfcc| over {len(agree)} clips = {worst:.2e}; int8 agreement mean = {np.mean(agree):.4f}")
    assert worst < 5e-4
    assert np.mean(agree) > 0.98


def test_int16_scaling_matches_soundfile_float(cfg):
    clips = _sample_clips(cfg, per_class=1)[:2]
    for p, _, _ in clips:
        a16, _ = sf.read(str(p), dtype="int16", always_2d=False)
        a32, _ = sf.read(str(p), dtype="float32", always_2d=False)
        np.testing.assert_array_equal(a16.astype(np.float32) / 32768.0, a32)
