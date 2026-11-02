"""Simulate the device operating point on the saved isolated-clip test stream.

The board can only run the model every few hundred milliseconds, so the paper's decision parameters
(30 ms hop, K = 16) are unreachable. This script runs the stream exactly as the device will — table-driven
frontend (``frontend_ref``), int8 two-output model in the LiteRT runtime (reference kernels), one window per
hop — caches the per-hop probabilities, and then scores a grid of decision parameters (K, threshold,
suppression, energy gate) with the Warden metrics. Output: CSV with every configuration, JSON with the
recommended operating point, Markdown summary for the report.

    python tools/simulate_operating_point.py [--hops 400 500 600] [--max-fp-per-min 1.5]
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from kws_paths import BACKGROUND_CLASSES, MODEL_REPO, REPO_ROOT, RESULTS_DIR, TWO_OUTPUT_MODEL, load_config
from export_tables import build_tables
from frontend_ref import features, quantize
from stream_scoring import decide, load_gt, score

STREAM_CANDIDATES = [REPO_ROOT.parent / "model-evaluation" / "results" / "stream" / "warden_stream_v2.wav",
                     MODEL_REPO / "results" / "stream" / "warden_stream_v2.wav"]
GT_DEFAULT = MODEL_REPO / "results" / "stream" / "warden_stream_v2_gt.json"


def find_stream() -> Path | None:
    return next((p for p in STREAM_CANDIDATES if p.exists()), None)


def litert(path: Path):
    from ai_edge_litert import interpreter as lr
    it = lr.Interpreter(model_path=str(path), experimental_op_resolver_type=lr.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    it.allocate_tensors()
    return it


def run_stream(stream: np.ndarray, hop_ms: int, tables: dict, it, cache: Path) -> dict:
    """Per-hop probabilities of the device pipeline; cached in an .npz per hop."""
    if cache.exists():
        d = np.load(cache)
        return {k: d[k] for k in d.files}
    sr, clip = tables["sample_rate"], tables["clip_samples"]
    hop = int(hop_ms * sr / 1000)
    scale, zp = tables["quant"]["input"]["scale"], tables["quant"]["input"]["zero_point"]
    inp = it.get_input_details()[0]
    soft = next(o for o in it.get_output_details() if int(o["shape"][-1]) == len(tables["classes"]))
    o_scale, o_zp = soft["quantization"]
    n = (len(stream) - clip) // hop + 1
    times, probs, dyn, rms = np.zeros(n), np.zeros((n, len(tables["classes"])), np.float32), np.zeros(n, np.float32), np.zeros(n, np.float32)
    for i in range(n):
        seg = stream[i * hop:i * hop + clip]
        lm, feat = features(seg, tables)
        x = quantize(feat, scale, zp)
        it.set_tensor(inp["index"], x[np.newaxis, :, :, np.newaxis])
        it.invoke()
        q = it.get_tensor(soft["index"])[0].astype(np.float32)
        probs[i] = (q - o_zp) * o_scale
        times[i] = (i * hop + clip) * 1000.0 / sr                      # end of the window, as collect_raw_events
        dyn[i] = lm.max() - lm.min()                                     # log-mel dynamic range (dB) for the energy gate
        rms[i] = 20 * np.log10(np.sqrt(np.mean(seg.astype(np.float64) ** 2)) + 1e-12)
        if (i + 1) % 200 == 0:
            print(f"  hop {hop_ms} ms: {i + 1}/{n} windows")
    out = {"times_ms": times, "probs": probs, "dyn_db": dyn, "rms_dbfs": rms, "hop_ms": np.int32(hop_ms)}
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **out)
    return out


def gate_mask(r: dict, gate: str):
    """Energy gate per window: None (off), log-mel dynamic range, or RMS above the stream's quiet floor."""
    if gate == "none":
        return None
    if gate.startswith("dyn"):
        return r["dyn_db"] >= float(gate[3:])
    if gate.startswith("rms"):
        floor = float(np.percentile(r["rms_dbfs"], 5))
        return r["rms_dbfs"] >= floor + float(gate[3:])
    raise ValueError(f"unknown gate spec {gate!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stream", type=Path, default=find_stream())
    ap.add_argument("--gt", type=Path, default=GT_DEFAULT)
    ap.add_argument("--model", type=Path, default=TWO_OUTPUT_MODEL)
    ap.add_argument("--hops", type=int, nargs="+", default=[400, 500, 600])
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--thresholds", type=float, nargs="+", default=[round(x, 2) for x in list(np.arange(0.50, 0.96, 0.05)) + [0.96, 0.97, 0.98, 0.99]])
    ap.add_argument("--suppressions", type=float, nargs="+", default=[1000, 1500])
    ap.add_argument("--gates", nargs="+", default=["none", "dyn15", "dyn25", "rms6", "rms12"],
                    help="energy gate: none | dyn<dB> (log-mel dynamic range >= dB) | rms<dB> (window RMS >= stream 5th-percentile floor + dB)")
    ap.add_argument("--max-fp-per-min", type=float, default=1.5)
    ap.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "tools" / "out" / "opsim")
    ap.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    args = ap.parse_args()
    if args.stream is None or not args.stream.exists():
        raise SystemExit("stream WAV not found; rebuild it in the model repo with "
                         "`python -m src.streaming_eval --model results/checkpoints/with_aug/best_model.keras --warden --save-stream results/stream/warden_stream_v2.wav`")

    cfg = load_config()
    tables = build_tables(cfg, args.model)
    classes = tables["classes"]
    kw = tables["keyword_indices"]
    targets = [c for c in classes if c not in BACKGROUND_CLASSES]
    stream, sr = sf.read(str(args.stream), dtype="float32", always_2d=False)
    assert sr == tables["sample_rate"]
    gt = load_gt(args.gt)
    duration_s = len(stream) / sr
    print(f"stream {args.stream.name}: {duration_s:.1f} s, {len(gt)} GT events; model {args.model.name}")
    it = litert(args.model)

    rows = []
    raw = {}
    for hop in args.hops:
        print(f"running device pipeline at hop {hop} ms ...")
        raw[hop] = run_stream(stream, hop, tables, it, args.raw_dir / f"raw_hop{hop}.npz")
        r = raw[hop]
        for k, thr, supp, gate in itertools.product(args.ks, args.thresholds, args.suppressions, args.gates):
            gate_pass = gate_mask(r, gate)
            dets = decide(r["times_ms"], r["probs"], classes, kw, k, thr, supp, gate_pass)
            m = score(dets, gt, duration_s, targets)
            rows.append({"hop_ms": hop, "k": k, "threshold": thr, "suppression_ms": supp, "gate": gate,
                         "correct_pct": 100 * m["matched_rate"], "wrong_pct": 100 * m["wrong_rate"],
                         "missed_pct": 100 * m["miss_rate"], "fp_per_min": m["false_positives_per_minute"],
                         "n_detections": m["n_detections"],
                         "windows_gated_pct": 0.0 if gate_pass is None else 100 * float(np.mean(~gate_pass))})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "operating_point_sim.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def best(rows_, fp_limit):
        ok = [r for r in rows_ if r["fp_per_min"] <= fp_limit]
        return max(ok, key=lambda r: (r["correct_pct"], -r["fp_per_min"], -r["wrong_pct"])) if ok else None

    rec = best(rows, args.max_fp_per_min)
    rec_strict = best(rows, 0.5)
    per_hop = {hop: best([r for r in rows if r["hop_ms"] == hop], args.max_fp_per_min) for hop in args.hops}
    per_hop_fp1 = {hop: best([r for r in rows if r["hop_ms"] == hop], 1.0) for hop in args.hops}
    ref_path = MODEL_REPO / "results" / "tables" / "streaming_eval_with_aug_warden_thr0p70.json"
    ref = json.loads(ref_path.read_text(encoding="utf-8")) if ref_path.exists() else {}
    summary = {
        "stream": args.stream.name, "duration_s": duration_s, "n_gt_events": len(gt), "model": args.model.name,
        "grid": {"hops_ms": args.hops, "ks": args.ks, "thresholds": args.thresholds, "suppressions_ms": args.suppressions, "gates": args.gates},
        "n_configurations": len(rows),
        "recommended": rec, "recommended_fp_limit_per_min": args.max_fp_per_min,
        "best_at_0p5_fp_per_min": rec_strict, "best_per_hop": per_hop, "best_per_hop_fp_le_1": per_hop_fp1,
        "paper_reference_fp32_hop30_k16_thr0p70": {k: ref.get(k) for k in ("matched_rate", "wrong_rate", "miss_rate", "false_positives_per_minute")} if ref else None,
        "windows_per_hop": {hop: int(len(raw[hop]["times_ms"])) for hop in args.hops},
    }
    (args.out_dir / "operating_point_sim.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [f"# Operating-point simulation — {args.stream.name} ({duration_s:.0f} s, {len(gt)} events), int8 two-output model, device pipeline", "",
             f"Paper reference (FP32, hop 30 ms, K 16, thr 0.70): correct {100 * ref.get('matched_rate', 0):.1f} %, FP/min {ref.get('false_positives_per_minute', 0):.2f}" if ref else "",
             "", f"## Best configuration per hop (FP/min ≤ {args.max_fp_per_min})", "",
             "| hop ms | K | thr | supp ms | gate | correct % | wrong % | missed % | FP/min | detections |", "|---|---|---|---|---|---|---|---|---|---|"]
    for hop in args.hops:
        r = per_hop[hop]
        if r:
            lines.append(f"| {hop} | {r['k']} | {r['threshold']:.2f} | {int(r['suppression_ms'])} | {r['gate']} | {r['correct_pct']:.1f} | {r['wrong_pct']:.1f} | {r['missed_pct']:.1f} | {r['fp_per_min']:.2f} | {r['n_detections']} |")
    lines += ["", "## Best configuration per hop (FP/min ≤ 1.0)", "",
              "| hop ms | K | thr | supp ms | gate | correct % | wrong % | missed % | FP/min | detections |", "|---|---|---|---|---|---|---|---|---|---|"]
    for hop in args.hops:
        r = per_hop_fp1[hop]
        if r:
            lines.append(f"| {hop} | {r['k']} | {r['threshold']:.2f} | {int(r['suppression_ms'])} | {r['gate']} | {r['correct_pct']:.1f} | {r['wrong_pct']:.1f} | {r['missed_pct']:.1f} | {r['fp_per_min']:.2f} | {r['n_detections']} |")
    lines += ["", "## Top 12 configurations overall (FP/min ≤ limit)", "",
              "| hop ms | K | thr | supp ms | gate | correct % | wrong % | missed % | FP/min |", "|---|---|---|---|---|---|---|---|---|"]
    top, seen = [], set()
    for r in sorted([r for r in rows if r["fp_per_min"] <= args.max_fp_per_min], key=lambda r: (-r["correct_pct"], r["fp_per_min"])):
        key = (r["hop_ms"], r["k"], r["threshold"], r["suppression_ms"], r["correct_pct"], r["wrong_pct"], r["fp_per_min"])
        if key not in seen:
            seen.add(key); top.append(r)
        if len(top) == 12:
            break
    for r in top:
        lines.append(f"| {r['hop_ms']} | {r['k']} | {r['threshold']:.2f} | {int(r['suppression_ms'])} | {r['gate']} | {r['correct_pct']:.1f} | {r['wrong_pct']:.1f} | {r['missed_pct']:.1f} | {r['fp_per_min']:.2f} |")
    if rec_strict:
        lines += ["", f"Best at ≤ 0.5 FP/min: hop {rec_strict['hop_ms']} ms, K {rec_strict['k']}, thr {rec_strict['threshold']:.2f}, supp {int(rec_strict['suppression_ms'])} ms, gate {rec_strict['gate']} → correct {rec_strict['correct_pct']:.1f} %, FP/min {rec_strict['fp_per_min']:.2f}"]
    (args.out_dir / "operating_point_sim.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwritten {csv_path}, operating_point_sim.json, operating_point_sim.md")


if __name__ == "__main__":
    main()
