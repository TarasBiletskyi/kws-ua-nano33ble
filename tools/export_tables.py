"""Export the constant tables of the MFCC frontend to C (``fe_tables.h``) and JSON.

Every number the device needs for ``frontend.c`` is derived here from the same libraries the
training pipeline used (SciPy window, librosa mel filterbank, SciPy DCT and Savitzky–Golay
coefficients) so that the C code only multiplies and adds. Specification: knowledge base 03.

    python tools/export_tables.py [--model firmware/kws_nano33/model/with_aug_two_output_int8.tflite]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import librosa
import numpy as np
from scipy.fftpack import dct as scipy_dct
from scipy.signal import get_window, savgol_coeffs

from kws_paths import (BACKGROUND_CLASSES, FIRMWARE_DIR, PUBLISHED_MODEL, REPO_ROOT, TWO_OUTPUT_MODEL,
                       load_config, transliterate)

SG_WIDTH = 9
TOP_DB = 80.0
AMIN = 1e-10
CMVN_EPS = 1e-8


def cfloat(v: float) -> str:
    """Valid C float literal: '80f' is not C, '80.0f' is; keep 9 significant digits (float32-exact)."""
    s = f"{float(v):.9g}"
    if not any(ch in s for ch in ".e") and s not in ("inf", "-inf", "nan"):
        s += ".0"
    return s + "f"


def read_quant_params(model_path: Path) -> dict:
    """Input / output (and GAP, if the model has a 64-wide output) scale and zero-point via the LiteRT runtime."""
    from ai_edge_litert import interpreter as lr
    it = lr.Interpreter(model_path=str(model_path))
    it.allocate_tensors()
    inp = it.get_input_details()[0]
    q = {
        "model": model_path.name,
        "input": {"shape": [int(v) for v in inp["shape"]], "scale": float(inp["quantization"][0]),
                  "zero_point": int(inp["quantization"][1])},
    }
    for out in it.get_output_details():
        entry = {"shape": [int(v) for v in out["shape"]], "scale": float(out["quantization"][0]),
                 "zero_point": int(out["quantization"][1]), "index": int(out["index"])}
        width = int(out["shape"][-1])
        if width == 15:
            q["softmax"] = entry
        elif width == 64:
            q["gap"] = entry
        else:
            q[f"output_{width}"] = entry
    return q


def sparse_mel(mel: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Triangular filters are contiguous: store (start bin, length, offset into flat weights, weights)."""
    starts, lengths, offsets, weights = [], [], [], []
    off = 0
    for row in mel:
        nz = np.nonzero(row)[0]
        start, stop = int(nz[0]), int(nz[-1]) + 1
        assert np.all(row[start:stop] > 0), "mel filter is not contiguous"
        starts.append(start)
        lengths.append(stop - start)
        offsets.append(off)
        weights.append(row[start:stop])
        off += stop - start
    return (np.array(starts, np.int32), np.array(lengths, np.int32), np.array(offsets, np.int32),
            np.concatenate(weights).astype(np.float32))


def build_tables(cfg: dict, model_path: Path | None = None) -> dict:
    a, m = cfg["audio"], cfg["mfcc"]
    sr, n_fft, win_len, hop = int(a["sample_rate"]), int(m["n_fft"]), int(m["win_length"]), int(m["hop_length"])
    n_mels, n_mfcc, n_frames = int(m["n_mels"]), int(m["n_mfcc"]), int(m["n_frames"])
    clip = int(a["clip_samples"])
    n_raw = 1 + (clip - n_fft) // hop

    window = get_window("hann", win_len, fftbins=True).astype(np.float32)          # periodic Hann, as librosa
    mel = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=0.0, fmax=sr / 2.0,
                              htk=False, norm="slaney").astype(np.float32)          # (40, 257)
    mel_start, mel_len, mel_off, mel_w = sparse_mel(mel)
    dct = scipy_dct(np.eye(n_mels), type=2, norm="ortho", axis=0)[:n_mfcc].astype(np.float32)  # c = dct @ S

    half = SG_WIDTH // 2
    sg = {}
    for p in (1, 2):
        sg[p] = {
            "center": savgol_coeffs(SG_WIDTH, p, deriv=p, pos=half, use="dot"),
            "head": np.stack([savgol_coeffs(SG_WIDTH, p, deriv=p, pos=k, use="dot") for k in range(half)]),
            "tail": np.stack([savgol_coeffs(SG_WIDTH, p, deriv=p, pos=k, use="dot") for k in range(half + 1, SG_WIDTH)]),
        }

    classes = list(cfg["classes"])
    tables = {
        "sample_rate": sr, "clip_samples": clip, "n_fft": n_fft, "win_length": win_len, "hop_length": hop,
        "win_offset": (n_fft - win_len) // 2, "n_bins": n_fft // 2 + 1, "n_mels": n_mels, "n_mfcc": n_mfcc,
        "n_frames_raw": n_raw, "n_frames": n_frames, "n_features": int(m["n_features"]),
        "top_db": TOP_DB, "amin": AMIN, "cmvn_eps": CMVN_EPS, "sg_width": SG_WIDTH,
        "window": window, "mel_dense": mel, "mel_start": mel_start, "mel_len": mel_len, "mel_offset": mel_off,
        "mel_weights": mel_w, "dct": dct,
        "sg1_center": sg[1]["center"].astype(np.float32), "sg1_head": sg[1]["head"].astype(np.float32),
        "sg1_tail": sg[1]["tail"].astype(np.float32),
        "sg2_center": sg[2]["center"].astype(np.float32), "sg2_head": sg[2]["head"].astype(np.float32),
        "sg2_tail": sg[2]["tail"].astype(np.float32),
        "classes": classes, "classes_ascii": [transliterate(c) if c not in BACKGROUND_CLASSES else c for c in classes],
        "keyword_indices": [i for i, c in enumerate(classes) if c not in BACKGROUND_CLASSES],
        "quant": read_quant_params(model_path) if model_path is not None else None,
    }
    return tables


def _c_float_array(name: str, arr: np.ndarray, per_line: int = 8) -> str:
    flat = np.asarray(arr, dtype=np.float32).ravel()
    body = ",\n".join("  " + ", ".join(f"{cfloat(v)}" for v in flat[i:i + per_line]) for i in range(0, len(flat), per_line))
    return f"static const float {name}[{len(flat)}] = {{\n{body}\n}};\n"


def _c_int_array(name: str, arr: np.ndarray, ctype: str = "int16_t", per_line: int = 16) -> str:
    flat = np.asarray(arr).ravel()
    body = ",\n".join("  " + ", ".join(str(int(v)) for v in flat[i:i + per_line]) for i in range(0, len(flat), per_line))
    return f"static const {ctype} {name}[{len(flat)}] = {{\n{body}\n}};\n"


def write_header(t: dict, path: Path) -> None:
    q = t["quant"] or {}
    lines = [
        "// GENERATED by tools/export_tables.py — do not edit. Constant tables of the MFCC frontend (knowledge base 03).",
        f"// Source model for quantization parameters: {q.get('model', 'none')}",
        "#pragma once", "#include <stdint.h>", "",
        f"#define FE_SAMPLE_RATE {t['sample_rate']}", f"#define FE_CLIP_SAMPLES {t['clip_samples']}",
        f"#define FE_N_FFT {t['n_fft']}", f"#define FE_WIN_LEN {t['win_length']}", f"#define FE_WIN_OFFSET {t['win_offset']}",
        f"#define FE_HOP {t['hop_length']}", f"#define FE_N_BINS {t['n_bins']}", f"#define FE_N_MELS {t['n_mels']}",
        f"#define FE_N_MFCC {t['n_mfcc']}", f"#define FE_N_FRAMES_RAW {t['n_frames_raw']}",
        f"#define FE_N_FRAMES {t['n_frames']}", f"#define FE_N_FEAT {t['n_features']}",
        f"#define FE_SG_WIDTH {t['sg_width']}", f"#define FE_SG_HALF {t['sg_width'] // 2}",
        f"#define FE_TOP_DB {cfloat(t['top_db'])}", f"#define FE_AMIN {cfloat(t['amin'])}", f"#define FE_CMVN_EPS {cfloat(t['cmvn_eps'])}",
        f"#define KWS_N_CLASSES {len(t['classes'])}", f"#define KWS_N_KEYWORDS {len(t['keyword_indices'])}",
        f"#define KWS_MEL_WEIGHTS_TOTAL {len(t['mel_weights'])}",
    ]
    if q:
        lines += [f"#define FE_IN_SCALE {cfloat(q['input']['scale'])}", f"#define FE_IN_ZP ({q['input']['zero_point']})"]
        if "softmax" in q:
            lines += [f"#define FE_OUT_SCALE {cfloat(q['softmax']['scale'])}", f"#define FE_OUT_ZP ({q['softmax']['zero_point']})"]
        if "gap" in q:
            lines += [f"#define FE_GAP_SCALE {cfloat(q['gap']['scale'])}", f"#define FE_GAP_ZP ({q['gap']['zero_point']})"]
    lines.append("")
    lines.append(_c_float_array("FE_WINDOW", t["window"]))
    lines.append(_c_int_array("MEL_START", t["mel_start"]))
    lines.append(_c_int_array("MEL_LEN", t["mel_len"]))
    lines.append(_c_int_array("MEL_OFFSET", t["mel_offset"]))
    lines.append(_c_float_array("MEL_W", t["mel_weights"]))
    lines.append("// DCT-II (orthonormal), row-major [FE_N_MFCC][FE_N_MELS]: c[n] = sum_i DCT13x40[n*40+i] * S[i]")
    lines.append(_c_float_array("DCT13x40", t["dct"]))
    lines.append("// Savitzky–Golay kernels (width 9, polyorder = deriv = 1 for delta, 2 for delta-delta), applied as dot products")
    lines.append("// with the 9 data samples in time order. HEAD[k]: window = frames 0..8 evaluated at frame k (k = 0..3);")
    lines.append("// TAIL[k]: window = last 9 frames evaluated at position 5 + k inside it (k = 0..3).")
    lines.append(_c_float_array("SG1_CENTER", t["sg1_center"], 9))
    lines.append(_c_float_array("SG1_HEAD", t["sg1_head"], 9))
    lines.append(_c_float_array("SG1_TAIL", t["sg1_tail"], 9))
    lines.append(_c_float_array("SG2_CENTER", t["sg2_center"], 9))
    lines.append(_c_float_array("SG2_HEAD", t["sg2_head"], 9))
    lines.append(_c_float_array("SG2_TAIL", t["sg2_tail"], 9))
    lines.append("static const char* const KWS_CLASS_NAMES_UTF8[KWS_N_CLASSES] = {")
    lines.append("  " + ", ".join(f'u8"{c}"' for c in t["classes"]) + "\n};")
    lines.append("static const char* const KWS_CLASS_NAMES_ASCII[KWS_N_CLASSES] = {")
    lines.append("  " + ", ".join(f'"{c}"' for c in t["classes_ascii"]) + "\n};")
    lines.append(_c_int_array("KWS_KEYWORD_INDICES", np.array(t["keyword_indices"]), "uint8_t"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def write_json(t: dict, path: Path) -> None:
    def conv(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        return v
    data = {k: conv(v) for k, v in t.items() if k != "mel_dense"}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def load_json(path: Path) -> dict:
    t = json.loads(path.read_text(encoding="utf-8"))
    for k in ("window", "mel_weights", "dct", "sg1_center", "sg1_head", "sg1_tail", "sg2_center", "sg2_head", "sg2_tail"):
        t[k] = np.asarray(t[k], dtype=np.float32)
    for k in ("mel_start", "mel_len", "mel_offset"):
        t[k] = np.asarray(t[k], dtype=np.int32)
    return t


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_model = TWO_OUTPUT_MODEL if TWO_OUTPUT_MODEL.exists() else PUBLISHED_MODEL
    ap.add_argument("--model", type=Path, default=default_model, help="LiteRT model to read scale/zero-point from")
    ap.add_argument("--header", type=Path, default=FIRMWARE_DIR / "fe_tables.h")
    ap.add_argument("--json", type=Path, default=REPO_ROOT / "tools" / "fe_tables.json")
    args = ap.parse_args()
    cfg = load_config()
    t = build_tables(cfg, args.model)
    write_header(t, args.header)
    write_json(t, args.json)
    q = t["quant"]
    print(f"frames {t['n_frames_raw']}+1, mel weights {len(t['mel_weights'])}, window offset {t['win_offset']}")
    print(f"quant from {q['model']}: input scale {q['input']['scale']:.8f} zp {q['input']['zero_point']}; "
          + "; ".join(f"{k} scale {v['scale']:.8f} zp {v['zero_point']}" for k, v in q.items() if k not in ('model', 'input')))
    print(f"written {args.header} ({args.header.stat().st_size} B) and {args.json}")


if __name__ == "__main__":
    main()
