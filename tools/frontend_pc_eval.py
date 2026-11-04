"""Test V3: frontend.c, compiled for the PC, plus the int8 model in LiteRT 2.3.0 on the whole test set.

Three input paths go through the same model, clip by clip:
  c        frontend.c, the code of the board
  numpy    tools/frontend_ref.py, the executable specification
  librosa  src.features.extract_mfcc, the pipeline the model was trained on (round + clip)
The pass criterion of the plan: accuracy of "c" within 0.5 points of the int8 reference ("librosa").

    python tools/frontend_pc_eval.py [--limit N] [--json report/results/v3_frontend_pc.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from export_tables import build_tables
from frontend_pc import FrontendC
from frontend_ref import features, quantize
from kws_paths import TWO_OUTPUT_MODEL, add_model_repo_to_path, default_corpus_root, list_test_clips, load_config

GROUPS = {"mfcc": (0, 13), "delta": (13, 26), "delta2": (26, 39)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    add_model_repo_to_path()
    from src.features import extract_mfcc
    from ai_edge_litert import interpreter as lr

    it = lr.Interpreter(model_path=str(TWO_OUTPUT_MODEL), experimental_op_resolver_type=lr.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    it.allocate_tensors()
    inp = it.get_input_details()[0]
    out15 = [o["index"] for o in it.get_output_details() if o["shape"][-1] == 15][0]

    def run(x: np.ndarray) -> np.ndarray:
        it.set_tensor(inp["index"], x[np.newaxis, :, :, np.newaxis])
        it.invoke()
        return it.get_tensor(out15)[0].copy()

    cfg = load_config()
    classes = list(cfg["classes"])
    tables = build_tables(cfg, TWO_OUTPUT_MODEL)
    q = tables["quant"]["input"]
    clips = list_test_clips(default_corpus_root(cfg), classes)
    if args.limit:
        clips = clips[:args.limit]
    fe = FrontendC()

    labels, tops = [], {"c": [], "numpy": [], "librosa": []}
    prob_diff = []
    same_int8, total_int8 = 0, 0
    int8_max_diff = 0
    feat_err_librosa = {g: [] for g in GROUPS}
    feat_err_numpy = {g: [] for g in GROUPS}
    per_clip_int8_identical = []
    per_clip_int8_max = []
    per_clip_feat_err = []
    digital_zero = []
    worst = []
    for path, cls, label in clips:
        a16, _sr = sf.read(str(path), dtype="int16", always_2d=False)
        a16 = np.pad(a16, (0, tables["clip_samples"] - len(a16)))[:tables["clip_samples"]]
        a32 = a16.astype(np.float32) / 32768.0
        x_c, f_c = fe.process_clip(a16)
        _lm, f_np = features(a32, tables)
        f_lib = np.asarray(extract_mfcc(a32, cfg), dtype=np.float32)
        x_np = quantize(f_np, q["scale"], q["zero_point"])
        x_lib = quantize(f_lib, q["scale"], q["zero_point"])
        p_c, p_np, p_lib = run(x_c), run(x_np), run(x_lib)
        labels.append(label)
        tops["c"].append(int(p_c.argmax()))
        tops["numpy"].append(int(p_np.argmax()))
        tops["librosa"].append(int(p_lib.argmax()))
        prob_diff.append(int(np.abs(p_c.astype(int) - p_lib.astype(int)).max()))
        d = np.abs(x_c.astype(int) - x_lib.astype(int))
        same_int8 += int((d == 0).sum())
        total_int8 += d.size
        int8_max_diff = max(int8_max_diff, int(d.max()))
        per_clip_int8_identical.append(float((d == 0).mean()))
        per_clip_int8_max.append(int(d.max()))
        per_clip_feat_err.append(float(np.abs(f_c - f_lib).max()))
        digital_zero.append(bool(np.abs(a16).max() == 0))
        for g, (a, b) in GROUPS.items():
            feat_err_librosa[g].append(float(np.abs(f_c[:, a:b] - f_lib[:, a:b]).max()))
            feat_err_numpy[g].append(float(np.abs(f_c[:, a:b] - f_np[:, a:b]).max()))
        worst.append((float(np.abs(f_c - f_lib).max()), cls, Path(path).name[-40:]))

    labels = np.array(labels)
    acc = {k: float(100.0 * (np.array(v) == labels).mean()) for k, v in tops.items()}
    confusion = np.zeros((len(classes), len(classes)), dtype=int)
    for y, p in zip(labels, tops["c"]):
        confusion[y, p] += 1
    per_class = [{"class": c, "clips": int((labels == i).sum()),
                  "recall_c_pct": float(100.0 * (np.array(tops["c"])[labels == i] == i).mean()),
                  "recall_librosa_pct": float(100.0 * (np.array(tops["librosa"])[labels == i] == i).mean())}
                 for i, c in enumerate(classes)]
    # All-zero clips have constant MFCCs: CMVN then divides rounding noise by about 1e-8, and every implementation gives other numbers.
    keep = ~np.array(digital_zero)
    identical = np.array(per_clip_int8_identical)
    excluding_zero = {
        "clips": int(keep.sum()),
        "feature_max_abs_err_vs_librosa": float(np.array(per_clip_feat_err)[keep].max()),
        "int8_input_identical_to_librosa_pct": float(100.0 * identical[keep].mean()),
        "int8_input_max_abs_diff": int(np.array(per_clip_int8_max)[keep].max()),
        "softmax_max_abs_diff_vs_librosa": int(np.array(prob_diff)[keep].max()),
    }
    res = {
        "clips": int(len(labels)),
        "digital_zero_clips": int((~keep).sum()),
        "excluding_digital_zero_clips": excluding_zero,
        "accuracy_pct": acc,
        "passes_V3_within_0p5_points": bool(abs(acc["c"] - acc["librosa"]) <= 0.5),
        "argmax_c_equals_librosa_pct": float(100.0 * (np.array(tops["c"]) == np.array(tops["librosa"])).mean()),
        "softmax_max_abs_diff_vs_librosa": {"max": int(max(prob_diff)), "median": float(np.median(prob_diff))},
        "int8_input_identical_to_librosa_pct": float(100.0 * same_int8 / total_int8),
        "int8_input_max_abs_diff": int8_max_diff,
        "per_clip_int8_identical_min_pct": float(100.0 * min(per_clip_int8_identical)),
        "feature_max_abs_err_vs_librosa": {g: {"max": float(np.max(v)), "median": float(np.median(v))} for g, v in feat_err_librosa.items()},
        "feature_max_abs_err_vs_numpy": {g: {"max": float(np.max(v)), "median": float(np.median(v))} for g, v in feat_err_numpy.items()},
        "worst_feature_clips": [{"err": e, "class": c, "file": f} for e, c, f in sorted(worst, reverse=True)[:5]],
        "per_class": per_class,
        "confusion_matrix_c": confusion.tolist(),
        "classes": classes,
    }
    print(json.dumps({k: v for k, v in res.items() if k not in ("per_class", "confusion_matrix_c", "classes")}, indent=1, ensure_ascii=False))
    print("| class | clips | recall c % | recall librosa % |")
    print("|---|---|---|---|")
    for r in per_class:
        print(f"| {r['class']} | {r['clips']} | {r['recall_c_pct']:.1f} | {r['recall_librosa_pct']:.1f} |")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
        print("written", args.json)


if __name__ == "__main__":
    main()
