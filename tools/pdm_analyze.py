"""Analyse serial logs of the ``firmware/bringup/pdm_rms`` sketch written by ``tools/serial_log.ps1``.

Log lines look like
    12.345 RMS t=47213 g=20 n=1536 cb=6 rms=-61.8 ac=-61.9 pk=-47.2 dc=-13 short=0
    61.002 SUM t=95870 run_s=14.5 samples=231936 expected=232000 loss_pct=0.028 cb=906 short=0 stalls=0 min_ac=-62.4 max_pk=-40.1 gain=40
    16.500 >> G 40

Per gain segment (a segment = consecutive RMS lines with the same g) the script reports the noise floor as the
5th percentile / median / minimum of ``ac`` (DC-free RMS, dBFS) and the loudest ``pk``; the first ``--settle`` seconds
of a segment are dropped. For the whole log it reports sample continuity: sum of n vs. 16 kHz × elapsed time, STALL
lines and short reads.

    python tools/pdm_analyze.py tools/out/pdm/sweep.log [--settle 2] [--json report/results/x.json] [--label sweep]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

RMS_RE = re.compile(r"^(?P<host>[\d.]+) RMS t=(?P<t>\d+) g=(?P<g>\d+) n=(?P<n>\d+) cb=(?P<cb>\d+) rms=(?P<rms>-?[\d.]+) "
                    r"ac=(?P<ac>-?[\d.]+) pk=(?P<pk>-?[\d.]+) dc=(?P<dc>-?[\d.]+) short=(?P<short>\d+)")
SUM_RE = re.compile(r"^(?P<host>[\d.]+) SUM (?P<body>.*)$")
SAMPLE_RATE = 16000


def parse(path: Path) -> tuple[list[dict], list[dict], list[str]]:
    rms, sums, events = [], [], []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = RMS_RE.match(line)
        if m:
            d = {k: float(v) for k, v in m.groupdict().items()}
            rms.append(d)
            continue
        m = SUM_RE.match(line)
        if m:
            d = {"host": float(m.group("host"))}
            for kv in m.group("body").split():
                k, v = kv.split("=", 1)
                d[k] = float(v)
            sums.append(d)
            continue
        if "STALL" in line or "ERR" in line or ">>" in line or "HELLO" in line or line.rstrip().endswith("OK zeroed") or " OK " in line:
            events.append(line.strip())
    return rms, sums, events


def segments(rms: list[dict], settle_s: float) -> list[dict]:
    out = []
    i = 0
    while i < len(rms):
        g = rms[i]["g"]
        j = i
        while j < len(rms) and rms[j]["g"] == g:
            j += 1
        seg = rms[i:j]
        t0 = seg[0]["t"]
        kept = [r for r in seg if (r["t"] - t0) / 1000.0 >= settle_s and r["n"] > 0]
        if kept:
            ac = np.array([r["ac"] for r in kept])
            pk = np.array([r["pk"] for r in kept])
            dc = np.array([r["dc"] for r in kept])
            out.append({
                "gain_reg": int(g), "gain_db": (g - 40) * 0.5, "n_intervals": len(kept),
                "duration_s": (kept[-1]["t"] - kept[0]["t"]) / 1000.0 + 0.1,
                "ac_p5_dbfs": float(np.percentile(ac, 5)), "ac_median_dbfs": float(np.median(ac)),
                "ac_min_dbfs": float(ac.min()), "ac_max_dbfs": float(ac.max()),
                "pk_max_dbfs": float(pk.max()), "dc_mean_counts": float(dc.mean()),
            })
        i = j
    return out


def continuity(rms: list[dict]) -> dict:
    valid = [r for r in rms if r["n"] > 0]
    if len(valid) < 2:
        return {"intervals": len(valid)}
    elapsed_ms = valid[-1]["t"] - valid[0]["t"] + 100.0
    samples = sum(r["n"] for r in valid)
    expected = elapsed_ms * SAMPLE_RATE / 1000.0
    n = np.array([r["n"] for r in valid])
    return {
        "intervals": len(valid), "elapsed_s": elapsed_ms / 1000.0, "samples": int(samples),
        "expected_samples": int(expected), "loss_pct": 100.0 * (expected - samples) / expected,
        "n_per_100ms_min": int(n.min()), "n_per_100ms_max": int(n.max()), "n_per_100ms_mean": float(n.mean()),
        "short_reads": int(sum(r["short"] for r in valid)),
        "empty_intervals": len(rms) - len(valid),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", type=Path)
    ap.add_argument("--settle", type=float, default=2.0, help="seconds dropped at the start of every gain segment")
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    rms, sums, events = parse(args.log)
    segs = segments(rms, args.settle)
    cont = continuity(rms)
    stalls = [e for e in events if "STALL" in e]

    print(f"{args.log}: {len(rms)} RMS lines, {len(sums)} SUM lines, {len(stalls)} STALL lines")
    print("| gain reg | gain dB | s | ac p5 dBFS | ac median | ac min | ac max | pk max dBFS | dc |")
    print("|---|---|---|---|---|---|---|---|---|")
    for s in segs:
        print(f"| {s['gain_reg']} | {s['gain_db']:+.1f} | {s['duration_s']:.1f} | {s['ac_p5_dbfs']:.1f} | {s['ac_median_dbfs']:.1f} | "
              f"{s['ac_min_dbfs']:.1f} | {s['ac_max_dbfs']:.1f} | {s['pk_max_dbfs']:.1f} | {s['dc_mean_counts']:.0f} |")
    if "elapsed_s" in cont:
        print(f"continuity: {cont['elapsed_s']:.1f} s, {cont['samples']} samples vs {cont['expected_samples']} expected "
              f"({cont['loss_pct']:+.3f} % loss), n/100ms min/mean/max {cont['n_per_100ms_min']}/{cont['n_per_100ms_mean']:.0f}/"
              f"{cont['n_per_100ms_max']}, short reads {cont['short_reads']}, empty intervals {cont['empty_intervals']}")
    for e in events:
        print("  ", e)
    if sums:
        last = sums[-1]
        print("last SUM:", {k: v for k, v in last.items() if k != "host"})

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"label": args.label, "log": str(args.log), "settle_s": args.settle,
                                         "segments": segs, "continuity": cont, "stalls": len(stalls),
                                         "events": events, "last_sum": sums[-1] if sums else None},
                                        indent=2, ensure_ascii=False), encoding="utf-8")
        print("written", args.json)


if __name__ == "__main__":
    main()
