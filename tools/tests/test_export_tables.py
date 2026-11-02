import numpy as np
import pytest
from scipy.fftpack import dct as scipy_dct
from scipy.signal import savgol_filter

import librosa

from export_tables import build_tables, load_json, write_header, write_json
from kws_paths import PUBLISHED_MODEL


def test_window_matches_librosa_padded_window(tables):
    """librosa.stft: get_window('hann', win_length, fftbins=True) padded centrally to n_fft."""
    w = librosa.filters.get_window("hann", tables["win_length"], fftbins=True)
    padded = librosa.util.pad_center(w, size=tables["n_fft"])
    off = tables["win_offset"]
    assert off == 56
    np.testing.assert_allclose(padded[off:off + tables["win_length"]], tables["window"], rtol=0, atol=1e-7)
    assert np.all(padded[:off] == 0) and np.all(padded[off + tables["win_length"]:] == 0)
    assert tables["window"][0] == 0.0 and tables["window"].max() <= 1.0


def test_sparse_mel_reconstructs_dense_filterbank(tables):
    dense = tables["mel_dense"]
    rec = np.zeros_like(dense)
    for i in range(tables["n_mels"]):
        s, n, o = tables["mel_start"][i], tables["mel_len"][i], tables["mel_offset"][i]
        rec[i, s:s + n] = tables["mel_weights"][o:o + n]
    np.testing.assert_array_equal(rec, dense)
    assert dense.shape == (40, 257)
    assert len(tables["mel_weights"]) == int(tables["mel_len"].sum())


def test_dct_matrix_equals_scipy_dct(tables):
    rng = np.random.default_rng(0)
    s = rng.standard_normal((40, 97)).astype(np.float32)
    expected = scipy_dct(s, type=2, norm="ortho", axis=0)[:13]
    np.testing.assert_allclose(tables["dct"] @ s, expected, rtol=0, atol=1e-4)


@pytest.mark.parametrize("order", [1, 2])
def test_savitzky_golay_kernels_reproduce_savgol_filter_interp(tables, order):
    from frontend_ref import sg_filter
    rng = np.random.default_rng(1)
    x = rng.standard_normal((97, 13)).astype(np.float32)
    expected = savgol_filter(x, 9, polyorder=order, deriv=order, axis=0, mode="interp")
    got = sg_filter(x, tables[f"sg{order}_center"], tables[f"sg{order}_head"], tables[f"sg{order}_tail"])
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-5)


def test_delta_center_kernel_is_j_over_60(tables):
    np.testing.assert_allclose(tables["sg1_center"], np.arange(-4, 5) / 60.0, atol=1e-7)
    j = np.arange(-4, 5)
    np.testing.assert_allclose(tables["sg2_center"], (j ** 2 - 20 / 3) / 154.0, atol=1e-7)


def test_header_and_json_round_trip(tables, tmp_path):
    h = tmp_path / "fe_tables.h"
    j = tmp_path / "fe_tables.json"
    write_header(tables, h)
    write_json(tables, j)
    text = h.read_text(encoding="utf-8")
    for sym in ("FE_WINDOW[400]", "MEL_START[40]", "MEL_W[", "DCT13x40[520]", "SG1_HEAD[36]", "SG2_TAIL[36]",
                "KWS_CLASS_NAMES_UTF8", "KWS_KEYWORD_INDICES[13]", "#define FE_N_FRAMES_RAW 97", "#define FE_WIN_OFFSET 56"):
        assert sym in text, sym
    assert "FE_IN_SCALE" not in text  # no model given → no quant params
    back = load_json(j)
    np.testing.assert_array_equal(back["window"], tables["window"])
    np.testing.assert_array_equal(back["mel_weights"], tables["mel_weights"])
    assert back["classes"] == tables["classes"]
    assert back["classes_ascii"][9] == "tak"  # так


@pytest.mark.skipif(not PUBLISHED_MODEL.exists(), reason="submodule model not present")
def test_quant_params_of_published_model(cfg):
    t = build_tables(cfg, PUBLISHED_MODEL)
    q = t["quant"]
    assert abs(q["input"]["scale"] - 0.06802022) < 1e-6 and q["input"]["zero_point"] == -17
    assert abs(q["softmax"]["scale"] - 1 / 256) < 1e-9 and q["softmax"]["zero_point"] == -128


def test_header_float_literals_are_valid_c(tables, tmp_path):
    """'80f' or '0f' would not compile; every float literal needs a '.' or an exponent before the f suffix."""
    import re
    from export_tables import cfloat
    assert cfloat(80.0) == "80.0f" and cfloat(0.0) == "0.0f" and cfloat(1e-10) == "1e-10f" and cfloat(0.5) == "0.5f"
    h = tmp_path / "fe.h"
    write_header(tables, h)
    text = h.read_text(encoding="utf-8")
    literals = re.findall(r"(?<![\w.])[-+]?\d[\d.eE+-]*f\b", text)
    assert len(literals) > 1000
    bad = [l for l in literals if not re.fullmatch(r"[-+]?(\d+\.\d*|\.\d+)([eE][-+]?\d+)?f|[-+]?\d+[eE][-+]?\d+f", l)]
    assert not bad, bad[:10]
