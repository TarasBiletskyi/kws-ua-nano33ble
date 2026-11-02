"""Microphone-noise-floor check on the PC: how much accuracy the int8 model loses when the digital silence
around every training/test clip is replaced by a realistic microphone floor.

Every keyword clip of the corpus is the word centred in exact zeros; a MEMS microphone never produces zeros.
This script adds white or pink noise at a fixed RMS level (dBFS) to the whole test clip, runs the device
pipeline (table-driven frontend → int8 two-output model in the LiteRT runtime) and reports accuracy and the
share of ``_silence`` predictions per condition.

    python tools/noise_floor_check.py [--levels -60 -50 -40] [--types white pink] [--limit N]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from kws_paths import BACKGROUND_CLASSES, RESULTS_DIR, TWO_OUTPUT_MODEL, default_corpus_root, list_test_clips, load_config
from export_tables import build_tables
from frontend_ref import features, quantize


def pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """1/f noise via spectral shaping of white noise; unit RMS."""
    white = rng.standard_normal(n)
    spec = np.fft.rfft(white)
    f = np.fft.rfftfreq(n)
    scale = np.zeros_like(f)
    scale[1:] = 1.0 / np.sqrt(f[1:])
    x = np.fft.irfft(spec * scale, n)
    return x / (np.sqrt(np.mean(x ** 2)) + 1e-12)


def white_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    x = rng.standard_normal(n)
    return x / (np.sqrt(np.mean(x ** 2)) + 1e-12)


def frame_gate(y: np.ndarray, floor_dbfs: float, margin_db: float, frame: int = 160) -> np.ndarray:
    """Device-side 're-silencing' emulation: 10 ms frames whose RMS is below floor + margin are set to digital zero."""
    thr = 10 ** ((floor_dbfs + margin_db) / 20.0)
    out = y.copy()
    for s in range(0, len(y) - frame + 1, frame):
        seg = y[s:s + frame]
        if np.sqrt(np.mean(seg.astype(np.float64) ** 2)) < thr:
            out[s:s + frame] = 0.0
    return out


def litert(path: Path):
    from ai_edge_litert import interpreter as lr
    it = lr.Interpreter(model_path=str(path), experimental_op_resolver_type=lr.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    it.allocate_tensors()
    return it


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, default=TWO_OUTPUT_MODEL)
    ap.add_argument("--levels", type=float, nargs="+", default=[-60, -50, -40], help="noise RMS in dBFS")
    ap.add_argument("--types", nargs="+", default=["white", "pink"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None, help="use only the first N test clips (smoke test)")
    ap.add_argument("--gate-margin-db", type=float, default=0.0,
                    help="also evaluate each noisy condition with frames below floor+margin zeroed (0 = off)")
    ap.add_argument("--out", type=Path, default=RESULTS_DIR / "noise_floor_check.json")
    args = ap.parse_args()

    cfg = load_config()
    tables = build_tables(cfg, args.model)
    classes = tables["classes"]
    scale, zp = tables["quant"]["input"]["scale"], tables["quant"]["input"]["zero_point"]
    sil_idx = classes.index("_silence")
    corpus = default_corpus_root(cfg)
    clips = list_test_clips(corpus, classes)
    if args.limit:
        clips = clips[:args.limit]
    print(f"{len(clips)} test clips; model {args.model.name}")

    audio = []
    labels = np.array([idx for _, _, idx in clips])
    for p, _, _ in clips:
        a, sr = sf.read(str(p), dtype="float32", always_2d=False)
        assert sr == tables["sample_rate"]
        a = np.pad(a, (0, tables["clip_samples"] - len(a)))[:tables["clip_samples"]]
        audio.append(a)
    clip_rms_db = np.array([20 * np.log10(np.sqrt(np.mean(a.astype(np.float64) ** 2)) + 1e-12) for a in audio])

    it = litert(args.model)
    inp = it.get_input_details()[0]
    soft = next(o for o in it.get_output_details() if int(o["shape"][-1]) == len(classes))

    def predict(a: np.ndarray) -> int:
        _, feat = features(a, tables)
        it.set_tensor(inp["index"], quantize(feat, scale, zp)[np.newaxis, :, :, np.newaxis])
        it.invoke()
        return int(it.get_tensor(soft["index"])[0].argmax())

    conditions = [("clean", None, False)] + [(t, lvl, g) for t in args.types for lvl in args.levels
                                             for g in ([False, True] if args.gate_margin_db > 0 else [False])]
    results = []
    for noise_type, level, gated in conditions:
        rng = np.random.default_rng(args.seed)
        preds = np.zeros(len(audio), int)
        for i, a in enumerate(audio):
            if level is None:
                y = a
            else:
                gen = pink_noise if noise_type == "pink" else white_noise
                noise = gen(len(a), rng) * (10 ** (level / 20.0))
                y = np.clip(a + noise, -1.0, 1.0).astype(np.float32)
                if gated:
                    y = frame_gate(y, level, args.gate_margin_db)
            preds[i] = predict(y)
        acc = float(np.mean(preds == labels))
        kw_mask = np.array([classes[l] not in BACKGROUND_CLASSES for l in labels])
        res = {
            "condition": "clean" if level is None else f"{noise_type} {level:+.0f} dBFS" + (f" + gate {args.gate_margin_db:g} dB" if gated else ""),
            "frame_gate_margin_db": args.gate_margin_db if gated else None,
            "noise_type": noise_type if level is not None else None, "level_dbfs": level,
            "accuracy": acc,
            "keyword_accuracy": float(np.mean(preds[kw_mask] == labels[kw_mask])),
            "silence_share_of_predictions": float(np.mean(preds == sil_idx)),
            "keywords_predicted_silence_pct": 100 * float(np.mean(preds[kw_mask] == sil_idx)),
            "silence_clips_recall": float(np.mean(preds[labels == sil_idx] == sil_idx)),
            "unknown_clips_recall": float(np.mean(preds[labels == classes.index("_unknown_")] == classes.index("_unknown_"))),
            "mean_clip_rms_dbfs": float(clip_rms_db.mean()),
            "mean_snr_whole_clip_db": None if level is None else float(np.mean(clip_rms_db - level)),
        }
        results.append(res)
        print(f"{res['condition']:>16}: acc {100 * acc:5.1f} %  keyword acc {100 * res['keyword_accuracy']:5.1f} %  "
              f"silence share {100 * res['silence_share_of_predictions']:5.1f} %  keywords→silence {res['keywords_predicted_silence_pct']:5.1f} %"
              + ("" if level is None else f"  mean SNR(whole clip) {res['mean_snr_whole_clip_db']:5.1f} dB"))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"model": args.model.name, "n_clips": len(clips), "seed": args.seed, "results": results},
                                   indent=2, ensure_ascii=False), encoding="utf-8")
    md = ["| Condition | Accuracy % | Keyword acc. % | `_silence` share % | Keywords → `_silence` % | Mean SNR (whole clip) dB |",
          "|---|---|---|---|---|---|"]
    for r in results:
        snr = "—" if r["mean_snr_whole_clip_db"] is None else f"{r['mean_snr_whole_clip_db']:.1f}"
        md.append(f"| {r['condition']} | {100 * r['accuracy']:.1f} | {100 * r['keyword_accuracy']:.1f} | "
                  f"{100 * r['silence_share_of_predictions']:.1f} | {r['keywords_predicted_silence_pct']:.1f} | {snr} |")
    args.out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"written {args.out} and {args.out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
