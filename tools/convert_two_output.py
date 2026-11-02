"""Two-output INT8 LiteRT model (softmax + GAP vector) with the calibration recipe of src/quantize.py, plus V0.

Requires TensorFlow (the LiteRT converter for TensorFlow models lives in ``tf.lite``): run it with the model
repository's environment. The calibration reuses the exact 100 cached training feature maps selected by
``np.random.default_rng(42)`` over ``src.dataset.load_splits`` order, so the quantization parameters should be
identical to the published model; V0 verifies this bit for bit with the LiteRT runtime.

    python tools/convert_two_output.py [--cache-dir ../model-evaluation/cache/mfcc]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from kws_paths import (CHECKPOINT, MODEL_REPO, PUBLISHED_MODEL, RESULTS_DIR, TWO_OUTPUT_MODEL,
                       add_model_repo_to_path, default_cache_dir, default_corpus_root, load_config)

add_model_repo_to_path()


def convert(checkpoint: Path, representative: list[np.ndarray], out: Path) -> bytes:
    import tensorflow as tf
    model = tf.keras.models.load_model(str(checkpoint))
    gap = next(l for l in model.layers if isinstance(l, tf.keras.layers.GlobalAveragePooling2D))
    two = tf.keras.Model(model.input, [model.output, gap.output], name="ds_cnn_two_output")
    converter = tf.lite.TFLiteConverter.from_keras_model(two)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def gen():
        for s in representative:
            yield [s[np.newaxis, :, :, np.newaxis].astype(np.float32)]

    converter.representative_dataset = gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    blob = converter.convert()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(blob)
    return blob


def _interp(path: Path, preserve: bool = False):
    from ai_edge_litert import interpreter as lr
    it = lr.Interpreter(model_path=str(path), experimental_op_resolver_type=lr.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
                        experimental_preserve_all_tensors=preserve)
    it.allocate_tensors()
    return it


def v0_check(published: Path, two_output: Path, test_feats: list[np.ndarray], labels: list[int]) -> dict:
    """Softmax of the two-output model vs the published model; GAP vs the published model's MEAN tensor."""
    pub = _interp(published, preserve=True)
    two = _interp(two_output)
    pin = pub.get_input_details()[0]
    tin = two.get_input_details()[0]
    scale, zp = pin["quantization"]
    assert tuple(tin["quantization"]) == tuple(pin["quantization"]), "input quantization differs"
    pub_out = pub.get_output_details()[0]
    two_outs = {int(o["shape"][-1]): o for o in two.get_output_details()}
    soft, gapo = two_outs[15], two_outs[64]
    mean_tensor = next(int(op["outputs"][0]) for op in pub._get_ops_details() if op["op_name"] == "MEAN")
    n = n_soft_eq = n_gap_eq = n_argmax_eq = 0
    y_true, y_two = [], []
    max_soft = max_gap = 0
    for feat, label in zip(test_feats, labels):
        x = np.clip(np.rint(feat[np.newaxis, :, :, np.newaxis] / scale) + zp, -128, 127).astype(np.int8)
        pub.set_tensor(pin["index"], x); pub.invoke()
        two.set_tensor(tin["index"], x); two.invoke()
        o_pub = pub.get_tensor(pub_out["index"])[0].astype(np.int32)
        o_soft = two.get_tensor(soft["index"])[0].astype(np.int32)
        o_gap = two.get_tensor(gapo["index"])[0].astype(np.int32)
        o_mean = pub.get_tensor(mean_tensor).reshape(-1).astype(np.int32)
        n += 1
        n_soft_eq += int(np.array_equal(o_pub, o_soft))
        n_gap_eq += int(np.array_equal(o_mean, o_gap))
        n_argmax_eq += int(o_pub.argmax() == o_soft.argmax())
        max_soft = max(max_soft, int(np.abs(o_pub - o_soft).max()))
        max_gap = max(max_gap, int(np.abs(o_mean - o_gap).max()))
        y_true.append(int(label)); y_two.append(int(o_soft.argmax()))
    gap_q = next(d for d in pub.get_tensor_details() if d["index"] == mean_tensor)["quantization"]
    return {
        "n_clips": n,
        "softmax_identical": n_soft_eq, "softmax_argmax_identical": n_argmax_eq, "softmax_max_abs_diff": max_soft,
        "gap_identical_to_published_mean_tensor": n_gap_eq, "gap_max_abs_diff": max_gap,
        "two_output_accuracy": float(np.mean(np.array(y_true) == np.array(y_two))),
        "published_mean_tensor_quant": {"scale": float(gap_q[0]), "zero_point": int(gap_q[1])},
        "two_output_gap_quant": {"scale": float(gapo["quantization"][0]), "zero_point": int(gapo["quantization"][1])},
        "two_output_softmax_quant": {"scale": float(soft["quantization"][0]), "zero_point": int(soft["quantization"][1])},
        "passed": n_soft_eq == n,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    ap.add_argument("--published", type=Path, default=PUBLISHED_MODEL)
    ap.add_argument("--cache-dir", type=Path, default=default_cache_dir(), help="MFCC cache of the model repo (calibration + V0 inputs)")
    ap.add_argument("--out", type=Path, default=TWO_OUTPUT_MODEL)
    ap.add_argument("--report", type=Path, default=RESULTS_DIR / "v0_two_output_equality.json")
    ap.add_argument("--skip-convert", action="store_true", help="only run V0 on an existing --out model")
    args = ap.parse_args()
    if args.cache_dir is None or not args.cache_dir.exists():
        raise SystemExit("feature cache not found; pass --cache-dir (needed for calibration-identical conversion)")

    cfg = load_config()
    corpus_root = default_corpus_root(cfg)
    from src.dataset import load_splits
    from src.features import load_cached
    train_s, _, test_s = load_splits(corpus_root / "splits", args.cache_dir, cfg["classes"])
    print(f"splits: train {len(train_s)} test {len(test_s)} (expected 9719 / 1204 for identical calibration)")
    if len(train_s) != 9719:
        print("WARNING: train list length differs from the published run — calibration may differ")

    if not args.skip_convert:
        rng = np.random.default_rng(42)
        rep_idx = rng.choice(len(train_s), size=min(int(cfg["quantization"]["n_representative_samples"]), len(train_s)), replace=False)
        representative = [load_cached(train_s[i][0]) for i in rep_idx]
        blob = convert(args.checkpoint, representative, args.out)
        print(f"converted {args.out} ({len(blob)} B, sha256 {hashlib.sha256(blob).hexdigest()[:16]}…)")

    feats = [load_cached(p) for p, _ in test_s]
    labels = [l for _, l in test_s]
    rep = v0_check(args.published, args.out, feats, labels)
    rep.update({
        "published_model": str(args.published.relative_to(MODEL_REPO.parent.parent)) if args.published.is_relative_to(MODEL_REPO.parent.parent) else str(args.published),
        "two_output_model": str(args.out.name),
        "two_output_sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
        "checkpoint": str(args.checkpoint.name),
        "model_repo_commit": subprocess.run(["git", "-C", str(MODEL_REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
    })
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if k not in ("two_output_sha256",)}, indent=2))
    print("V0:", "PASSED" if rep["passed"] else "FAILED (softmax outputs are not identical)")


if __name__ == "__main__":
    main()
