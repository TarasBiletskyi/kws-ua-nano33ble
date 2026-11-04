"""Parse serial logs of ``firmware/kws_nano33`` (M2 steps 3-4, written by ``tools/serial_log.ps1``).

For every build it reports the benchmark (microseconds, cycles, instruction-cache hit rate), the cycles per operator group
and test V1: the device's int8 outputs against the two PC runtimes stored in ``tools/golden/golden_vectors.npz``
(LiteRT 2.3.0 for softmax and GAP, tf.lite 2.17 reference kernels for softmax). All numbers come from the last report of a log.

    python tools/m2_bench_parse.py --log Os=tools/out/m2/bench_Os.log --log O3=tools/out/m2/bench_O3.log \
        --json report/results/m2_bench.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

KV_RE = re.compile(r"(\w+)=(\S+)")
REPO = Path(__file__).resolve().parents[1]
GOLDEN = REPO / "tools" / "golden" / "golden_vectors.npz"
GOLDEN_H = REPO / "firmware" / "kws_nano33" / "golden_set.h"
MACS = 25.4e6
# MACs per operator from the model structure (98x39 input, 64 channels, feature map 49x20)
OP_MACS = {"first CONV_2D 10x4": 2.5e6, "5 x DEPTHWISE 3x3": 5 * 0.56e6, "5 x CONV_2D 1x1": 5 * 4.0e6}


def number(v: str):
    try:
        return float(v) if "." in v else int(v)
    except ValueError:
        return v


def parse(path: Path) -> dict:
    out: dict = {"log": str(path), "runs": [], "v1": {}, "probs": {}, "gap": {}, "ops": {}, "errors": []}
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = raw.split(" ", 2)
        if len(parts) < 2:
            continue
        tag, body = parts[1], parts[2] if len(parts) > 2 else ""
        kv = {k: number(v) for k, v in KV_RE.findall(body)}
        if tag == "RUN":
            out["runs"].append(kv)
        elif tag == "HELLO":
            out["hello"] = kv
        elif tag in ("SYS", "ALLOC", "QUANT", "BENCH", "MEM"):
            out[tag.lower()] = kv
        elif tag == "V1":
            out["v1"][kv["vec"]] = kv
        elif tag in ("PROB8", "GAP8"):
            m = re.match(r"vec=(\d+) (\S+)", body)
            out["probs" if tag == "PROB8" else "gap"][int(m.group(1))] = [int(x) for x in m.group(2).split(",")]
        elif tag == "OP":
            out["ops"][kv["i"]] = kv
        elif tag == "ERR" or "STALL" in raw:
            out["errors"].append(raw)
    return out


def op_groups(b: dict) -> list[dict]:
    ops = [b["ops"][i] for i in sorted(b["ops"])]
    total = b["bench"]["cycles_mean"]
    # graph order: conv, then (depthwise, conv 1x1) x 5, then mean, fully connected, softmax
    groups = {
        "first CONV_2D 10x4": [0],
        "5 x DEPTHWISE 3x3": [1, 3, 5, 7, 9],
        "5 x CONV_2D 1x1": [2, 4, 6, 8, 10],
        "MEAN": [11],
        "FULLY_CONNECTED": [12],
        "SOFTMAX": [13],
    }
    rows = []
    for name, idx in groups.items():
        cyc = sum(ops[i]["cycles_mean"] for i in idx)
        macs = OP_MACS.get(name)
        rows.append({"group": name, "cycles": cyc, "ms": cyc / 64e3, "share_pct": 100.0 * cyc / total,
                     "cycles_per_mac": cyc / macs if macs else None})
    return rows


def header_vectors(tag: str, n: int) -> list[np.ndarray]:
    text = GOLDEN_H.read_text(encoding="utf-8")
    out = []
    for i in range(n):
        m = re.search(rf"GOLDEN_{tag}_{i}\[\d+\] = \{{(.*?)\}};", text, re.S)
        out.append(np.array([int(x) for x in m.group(1).replace("\n", " ").split(",") if x.strip()], dtype=np.int8))
    return out


def v1(b: dict) -> dict:
    data = np.load(GOLDEN, allow_pickle=True)
    n = len(b["probs"])
    # The on-board vectors are a subset of the PC set: find each by its int8 input.
    inputs = header_vectors("INPUT", n)
    all_inputs = data["input_int8"].reshape(len(data["input_int8"]), -1)
    rows = []
    for i in range(n):
        j = int(np.flatnonzero((all_inputs == inputs[i]).all(axis=1))[0])
        dev_p = np.array(b["probs"][i], dtype=int)
        dev_g = np.array(b["gap"][i], dtype=int)
        row = {"vec": i, "pc_index": j, "label": int(data["labels"][j]), "top_device": int(dev_p.argmax()),
               "top_litert": int(data["prob_int8"][j].argmax()),
               "prob_max_diff_litert": int(np.abs(dev_p - data["prob_int8"][j]).max()),
               "prob_diff_n_litert": int((dev_p != data["prob_int8"][j]).sum()),
               "gap_max_diff_litert": int(np.abs(dev_g - data["gap_int8"][j]).max()),
               "gap_diff_n_litert": int((dev_g != data["gap_int8"][j]).sum())}
        if "prob_int8_tflite217" in data:
            ref = data["prob_int8_tflite217"][j]
            row["prob_max_diff_tflite217"] = int(np.abs(dev_p - ref).max())
            row["prob_diff_n_tflite217"] = int((dev_p != ref).sum())
        rows.append(row)
    return {"vectors": rows, "argmax_ok": all(r["top_device"] == r["top_litert"] for r in rows),
            "prob_max_diff": max(r["prob_max_diff_litert"] for r in rows),
            "gap_max_diff": max(r["gap_max_diff_litert"] for r in rows)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", action="append", required=True, metavar="LABEL=PATH", help="repeatable")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    result = {}
    for item in args.log:
        label, _, path = item.partition("=")
        b = parse(Path(path))
        if "bench" not in b:
            raise SystemExit(f"{path}: no BENCH line")
        ben = b["bench"]
        hits, misses = ben["icache_hit"], ben["icache_miss"]
        summary = {
            "hello": b["hello"], "alloc": b["alloc"], "mem": b.get("mem"), "errors": b["errors"], "bench": ben,
            "cycles_per_mac": ben["cycles_mean"] / MACS, "icache_hit_pct": 100.0 * hits / (hits + misses),
            "icache_miss_per_kcycle": 1000.0 * misses / ben["cycles_mean"],
            "x_over_300ms": ben["us_mean"] / 300e3, "op_groups": op_groups(b), "v1": v1(b),
        }
        result[label] = summary
        print(f"== {label}: opt={b['hello'].get('opt')} us min/mean/max {ben['us_min']}/{ben['us_mean']}/{ben['us_max']} "
              f"cycles_mean {ben['cycles_mean']} ({summary['cycles_per_mac']:.2f}/MAC, {summary['x_over_300ms']:.2f}x over 300 ms) "
              f"icache hit {summary['icache_hit_pct']:.1f} %, arena {b['alloc']['arena_used']}, errors {len(b['errors'])}")
        print("| group | ms | share % | cycles/MAC |")
        print("|---|---|---|---|")
        for r in summary["op_groups"]:
            cpm = f"{r['cycles_per_mac']:.1f}" if r["cycles_per_mac"] else "—"
            print(f"| {r['group']} | {r['ms']:.1f} | {r['share_pct']:.1f} | {cpm} |")
        print("V1:", {k: summary["v1"][k] for k in ("argmax_ok", "prob_max_diff", "gap_max_diff")})
        for r in summary["v1"]["vectors"]:
            print("  ", r)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
        print("written", args.json)


if __name__ == "__main__":
    main()
