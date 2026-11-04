"""Parse serial logs of ``firmware/bringup/ram_baseline`` (written by ``tools/serial_log.ps1``) into one JSON summary.

The sketch prints one block every 2 s (tags LNK, SYS, HEAP0, THR..., STK, PROBE, HEAP, KWS, SUM), for example
    1.145 LNK iter=2 data_start=0x20000200 ... static=44552 heap_region=216056 isr_stack=1024 main_stack_addr=0x20002B00
    1.155 THR name=main id=0x200021E8 state=2 prio=24 stack=32768 used=824 free=31944
    1.169 PROBE largest_malloc=113671 fill_1k_total=0 fill_pieces=0
    1.179 SUM ram_total=262144 reserved_low=512 static=44552 kws_buffers=0 main_stack=32768 ... heap_in_use=976 ...
The last complete block (LNK ... SUM) of every log is kept; the script also checks that all SUM lines of a run agree
(the values are constant over a run) and prints the builds side by side.

    python tools/ram_baseline_parse.py --log A=tools/out/ram/ram_baseline_a.log --log B=tools/out/ram/ram_baseline_b.log \
        --json report/results/m1_ram_baseline.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

LINE_RE = re.compile(r"^(?P<host>[\d.]+) (?P<tag>[A-Z0-9]+)(?P<body>(?: \S+=\S*)*)\s*$")
HEAP_RULE_BYTES = 8192  # minimum heap region the KWS firmware keeps (see report/results/m1_bringup.md, step 4c)


def parse_value(v: str):
    if v.startswith("0x"):
        return int(v, 16)
    try:
        return int(v)
    except ValueError:
        return v


def parse_body(body: str) -> dict:
    out = {}
    for tok in body.split():
        k, _, v = tok.partition("=")
        out[k] = parse_value(v)
    return out


def blocks(path: Path) -> list[dict]:
    """Split a log into complete LNK ... SUM blocks (a block is dropped if the host missed its start or end)."""
    result, cur = [], None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = LINE_RE.match(line)
        if not m:
            continue
        tag, kv = m.group("tag"), parse_body(m.group("body"))
        if tag == "LNK":
            cur = {"lnk": kv, "threads": []}
            continue
        if cur is None:
            continue
        if tag == "THR":
            cur["threads"].append(kv)
        elif tag == "SUM":
            cur["sum"] = kv
            result.append(cur)
            cur = None
        else:
            cur[tag.lower()] = kv
    return result


def summarize(label: str, path: Path) -> dict:
    bl = blocks(path)
    if not bl:
        raise SystemExit(f"{path}: no complete LNK...SUM block")
    last = bl[-1]
    sums = {json.dumps(b["sum"], sort_keys=True) for b in bl}
    out = {"label": label, "log": str(path), "blocks": len(bl), "sum_lines_identical": len(sums) == 1}
    for k in ("lnk", "sys", "heap0", "threads", "stk", "probe", "heap", "kws", "sum"):
        if k in last:
            out[k] = last[k]
    return out


ROWS = [
    ("static = .data + .bss (B)", lambda b: b["sum"]["static"]),
    ("main-thread stack, static in .bss (B)", lambda b: b["sum"]["main_stack"]),
    ("core static without main stack and KWS buffers (B)", lambda b: b["sum"]["core_static_wo_main_stack"]),
    ("KWS buffers (B)", lambda b: b["sum"]["kws_buffers"]),
    ("heap region __end__..__HeapLimit (B)", lambda b: b["sum"]["heap_region"]),
    ("heap in use at run time (B)", lambda b: b["sum"]["heap_in_use"]),
    ("heap allocations", lambda b: b["heap"]["allocs"]),
    ("largest single malloc (B)", lambda b: b["sum"]["heap_largest_free"]),
    ("1 KB fill total (B)", lambda b: b["sum"].get("heap_fill_total", "n/a")),
    ("RTOS threads", lambda b: ", ".join(f"{t['name']} {t['stack']}/{t['used']}" for t in b["threads"])),
    ("stacks reserved / max used (B)", lambda b: f"{b['stk']['reserved']} / {b['stk']['max_used']}"),
    ("ISR stack (B)", lambda b: b["sum"]["isr_stack"]),
]


def derived(builds: dict) -> dict:
    a = next((b for b in builds.values() if b["sum"]["kws_buffers"] == 0), None)
    if a is None:
        return {}
    s = a["sum"]
    ram_for_sections = s["ram_total"] - s["reserved_low"] - s["isr_stack"]
    available = ram_for_sections - s["static"]
    out = {
        "ram_for_data_bss_heap_bytes": ram_for_sections,
        "core_static_incl_main_stack_bytes": s["static"],
        "available_for_kws_statics_and_heap_bytes": available,
        "heap_rule_bytes": HEAP_RULE_BYTES,
        "safe_static_budget_for_kws_bytes": available - HEAP_RULE_BYTES,
    }
    for b in builds.values():
        if b["sum"]["kws_buffers"]:
            out["tested_kws_buffers_bytes"] = b["sum"]["kws_buffers"]
            out["heap_region_with_tested_buffers_bytes"] = b["sum"]["heap_region"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", action="append", required=True, metavar="LABEL=PATH", help="repeatable")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    builds = {}
    for item in args.log:
        label, _, path = item.partition("=")
        builds[label] = summarize(label, Path(path))

    labels = list(builds)
    print("| item | " + " | ".join(f"build {l}" for l in labels) + " |")
    print("|---|" + "---|" * len(labels))
    for name, fn in ROWS:
        print(f"| {name} | " + " | ".join(str(fn(builds[l])) for l in labels) + " |")
    for l in labels:
        b = builds[l]
        print(f"build {l}: {b['blocks']} complete blocks, SUM lines identical: {b['sum_lines_identical']}")
    d = derived(builds)
    for k, v in d.items():
        print(f"{k}: {v}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"builds": builds, "derived": d}, indent=2), encoding="utf-8")
        print("written", args.json)


if __name__ == "__main__":
    main()
