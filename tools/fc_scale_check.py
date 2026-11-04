"""What the single-scale defect of the 2023 LiteRT Micro library costs on the whole test set (found at M2 steps 3-4).

The model's dense layer has per-class weight scales (the TF 2.17 converter quantizes dense layers per axis). The library behind
``Arduino_TensorFlowLite`` commit 7948b85 reads only the first scale, so on the board logit i comes out multiplied by s0 / s_i.
This script runs the device pipeline on the PC (table-driven frontend, int8 model in LiteRT 2.3.0) and applies the dense layer
to each GAP vector in floating point, once with the correct scales and once with the board's single scale.
The float head reproduces the quantized one closely but not bit-exactly, so the int8 softmax of LiteRT is reported as the reference.

    python tools/fc_scale_check.py [--limit N] [--json report/results/m2_fc_scale_check.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from export_tables import build_tables
from frontend_ref import features, quantize
from kws_paths import TWO_OUTPUT_MODEL, default_corpus_root, list_test_clips, load_config


def dense_head(it) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
    det = {t["index"]: t for t in it.get_tensor_details()}
    fc = [o for o in it._get_ops_details() if o["op_name"] == "FULLY_CONNECTED"][0]
    gap_idx, w_idx, b_idx = fc["inputs"]
    wq = it.get_tensor(w_idx).astype(np.float64)
    bq = it.get_tensor(b_idx).astype(np.float64)
    scales = det[w_idx]["quantization_parameters"]["scales"].astype(np.float64)
    gs, gz = det[gap_idx]["quantization"]
    return wq, bq, scales, float(gs), int(gz)


def softmax(z: np.ndarray) -> np.ndarray:
    p = np.exp(z - z.max(axis=-1, keepdims=True))
    return p / p.sum(axis=-1, keepdims=True)


def metrics(top: np.ndarray, conf: np.ndarray, labels: np.ndarray, thresholds=(0.55, 0.85)) -> dict:
    out = {"accuracy_pct": float(100.0 * (top == labels).mean())}
    for thr in thresholds:
        out[f"correct_and_conf_ge_{thr}_pct"] = float(100.0 * ((top == labels) & (conf >= thr)).mean())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    from ai_edge_litert import interpreter as lr
    it = lr.Interpreter(model_path=str(TWO_OUTPUT_MODEL), experimental_op_resolver_type=lr.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    it.allocate_tensors()
    inp = it.get_input_details()[0]
    outs = {int(o["shape"][-1]): o["index"] for o in it.get_output_details()}

    cfg = load_config()
    classes = list(cfg["classes"])
    tables = build_tables(cfg, TWO_OUTPUT_MODEL)
    q = tables["quant"]
    scale, zp = q["input"]["scale"], q["input"]["zero_point"]
    clips = list_test_clips(default_corpus_root(cfg), classes)
    if args.limit:
        clips = clips[:args.limit]

    wq, bq, s, gs, gz = dense_head(it)
    labels, probs8, gaps = [], [], []
    for path, _cls, label in clips:
        a16, _sr = sf.read(str(path), dtype="int16", always_2d=False)
        a16 = np.pad(a16, (0, tables["clip_samples"] - len(a16)))[:tables["clip_samples"]]
        _lm, rep = features(a16.astype(np.float32) / 32768.0, tables)
        x = quantize(rep, scale, zp)
        it.set_tensor(inp["index"], x[np.newaxis, :, :, np.newaxis])
        it.invoke()
        probs8.append(it.get_tensor(outs[15])[0].copy())
        gaps.append(it.get_tensor(outs[64])[0].copy())
        labels.append(label)
    labels = np.array(labels)
    probs8 = np.array(probs8, dtype=np.int64)
    gaps = np.array(gaps, dtype=np.float64)

    raw = (gaps - gz) @ wq.T + bq
    p_ok = softmax(raw * gs * s)
    p_board = softmax(raw * gs * s[0])
    top_int8 = probs8.argmax(1)
    top_ok, top_board = p_ok.argmax(1), p_board.argmax(1)
    conf_int8 = (probs8.max(1) + 128) / 256.0

    per_class = []
    for i, name in enumerate(classes):
        m = labels == i
        per_class.append({"class": name, "weight_scale_rel_to_class0": float(s[i] / s[0]), "clips": int(m.sum()),
                          "recall_ok_pct": float(100.0 * (top_ok[m] == i).mean()),
                          "recall_board_pct": float(100.0 * (top_board[m] == i).mean())})
    res = {
        "clips": int(len(labels)),
        "weight_scale_rel_to_class0": [float(v) for v in s / s[0]],
        "pc_int8_softmax": metrics(top_int8, conf_int8, labels),
        "float_head_correct_scales": metrics(top_ok, p_ok.max(1), labels),
        "float_head_board_single_scale": metrics(top_board, p_board.max(1), labels),
        "float_head_vs_int8_argmax_agree_pct": float(100.0 * (top_ok == top_int8).mean()),
        "board_vs_correct_argmax_changed_clips": int((top_board != top_ok).sum()),
        "per_class": per_class,
    }
    print(json.dumps({k: v for k, v in res.items() if k != "per_class"}, indent=1))
    print("| class | scale / s0 | clips | recall correct % | recall board % |")
    print("|---|---|---|---|---|")
    for r in per_class:
        print(f"| {r['class']} | {r['weight_scale_rel_to_class0']:.3f} | {r['clips']} | {r['recall_ok_pct']:.1f} | {r['recall_board_pct']:.1f} |")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
        print("written", args.json)


if __name__ == "__main__":
    main()
