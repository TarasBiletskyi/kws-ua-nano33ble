"""Test V2 on the built-in golden audio: frontend.c on the board against the same code on the PC and against the references.

Parses a log of firmware/kws_nano33 in KWS_TEST_MODE 2 (written by tools/serial_log.ps1). The criteria of the plan: largest feature
difference ≤ 0.05, at least 98 % of the int8 inputs identical and the rest ±1, same top class, top probability within 0.05.

    python tools/m3_board_compare.py tools/out/m3/frontend_board.log --json report/results/m3_board_frontend.json
"""
from __future__ import annotations

import argparse
import json
import re
import struct
from pathlib import Path

import numpy as np

from frontend_pc import FrontendC

REPO = Path(__file__).resolve().parents[1]
GOLDEN = REPO / "tools" / "golden" / "golden_vectors.npz"
CLOCK_HZ = 64_000_000
KV_RE = re.compile(r"(\w+)=(\S+)")


def parse(path: Path) -> dict:
    feats: dict[int, np.ndarray] = {}
    fe, inv, probs, other = {}, {}, {}, {}
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = raw.split(" ", 2)
        if len(parts) < 3:
            continue
        tag, body = parts[1], parts[2]
        if tag == "FEATF":
            m = re.match(r"clip=(\d+) row=(\d+) (\S+)", body)
            clip, row = int(m.group(1)), int(m.group(2))
            values = [struct.unpack("<f", struct.pack("<I", int(h, 16)))[0] for h in m.group(3).split(",")]
            feats.setdefault(clip, np.full((98, 39), np.nan, dtype=np.float32))[row] = values
            continue
        kv = {k: (int(v) if re.fullmatch(r"-?\d+", v) else v) for k, v in KV_RE.findall(body)}
        if tag == "FE":
            fe[kv["clip"]] = kv
        elif tag == "INVOKE":
            inv[kv["clip"]] = kv
        elif tag == "PROB8":
            m = re.match(r"clip=(\d+) (\S+)", body)
            probs[int(m.group(1))] = np.array([int(x) for x in m.group(2).split(",")])
        elif tag in ("HELLO", "ALLOC", "MEM", "ERR"):
            other[tag.lower()] = kv
    return {"feats": feats, "fe": fe, "invoke": inv, "probs": probs, **other}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", type=Path)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    b = parse(args.log)
    g = np.load(GOLDEN, allow_pickle=True)
    scale, zp = float(g["input_scale"]), int(g["input_zero_point"])
    fe_pc = FrontendC()
    rows = []
    for clip in sorted(b["feats"]):
        f_board = b["feats"][clip]
        assert not np.isnan(f_board).any(), f"clip {clip}: incomplete FEATF lines"
        x_board = np.clip(np.rint(f_board / scale) + zp, -128, 127).astype(np.int8)
        x_c_pc, f_c_pc = fe_pc.process_clip(g["audio_int16"][clip])
        d_lib = np.abs(f_board - g["features_librosa"][clip])
        di = np.abs(x_board.astype(int) - g["input_int8"][clip].astype(int))
        dpc = np.abs(x_board.astype(int) - x_c_pc.astype(int))
        p, p_pc = b["probs"][clip], g["prob_int8"][clip].astype(int)
        fe = b["fe"][clip]
        rows.append({
            "clip": clip, "label": int(g["labels"][clip]),
            "feature_max_diff_vs_c_on_pc": float(np.abs(f_board - f_c_pc).max()),
            "feature_identical_bits_vs_c_on_pc_pct": float(100.0 * (f_board == f_c_pc).mean()),
            "feature_max_diff_vs_librosa": float(d_lib.max()),
            "int8_identical_vs_librosa_pct": float(100.0 * (di == 0).mean()), "int8_max_diff_vs_librosa": int(di.max()),
            "int8_identical_vs_c_on_pc_pct": float(100.0 * (dpc == 0).mean()), "int8_max_diff_vs_c_on_pc": int(dpc.max()),
            "top_board": int(p.argmax()), "top_pc": int(p_pc.argmax()),
            "softmax_max_diff": int(np.abs(p - p_pc).max()),
            "top_probability_diff": float(abs(int(p.max()) - int(p_pc.max())) / 256.0),
            "frame_us_min": fe["frame_cycles_min"] / CLOCK_HZ * 1e6, "frame_us_mean": fe["frame_cycles_mean"] / CLOCK_HZ * 1e6,
            "frame_us_max": fe["frame_cycles_max"] / CLOCK_HZ * 1e6, "finish_ms": fe["finish_cycles"] / CLOCK_HZ * 1e3,
            "invoke_ms": b["invoke"][clip]["cycles"] / CLOCK_HZ * 1e3,
        })
    v2 = all(r["feature_max_diff_vs_librosa"] <= 0.05 and r["int8_identical_vs_librosa_pct"] >= 98.0 and r["int8_max_diff_vs_librosa"] <= 1
             and r["top_board"] == r["top_pc"] and r["top_probability_diff"] <= 0.05 for r in rows)
    res = {"clips": len(rows), "passes_V2_criteria_on_these_clips": v2, "hello": b.get("hello"), "alloc": b.get("alloc"), "mem": b.get("mem"), "rows": rows}
    print(json.dumps(res, indent=1, ensure_ascii=False))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
        print("written", args.json)


if __name__ == "__main__":
    main()
