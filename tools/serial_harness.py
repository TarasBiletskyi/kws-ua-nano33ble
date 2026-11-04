"""Drive firmware/kws_nano33 in KWS_TEST_MODE 3 over USB serial: tests V1, V2 and V4 of the plan.

    python tools/serial_harness.py --ping
    python tools/serial_harness.py --v1 [--limit N]        50 golden int8 features into the model     -> report/results/device_v1.json
    python tools/serial_harness.py --v2 [--limit N]        50 golden audio clips through the frontend -> report/results/device_v2.json
    python tools/serial_harness.py --v4 [--limit N]        whole test set, about 45 min               -> report/results/device_v4.json

V4 appends every clip to tools/out/m3/v4_clips.jsonl and continues from there when started again.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "report" / "results"
GOLDEN = REPO / "tools" / "golden" / "golden_vectors.npz"
V4_CLIPS = REPO / "tools" / "out" / "m3" / "v4_clips.jsonl"
CLOCK_HZ = 64_000_000
BLOCK_BYTES = 2048  # kBlockBytes in serial_proto.cpp


def crc16(data: bytes) -> int:
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


class Board:
    def __init__(self, port: str):
        import serial
        self.s = serial.Serial(port, 115200, timeout=15, write_timeout=15)
        self.s.dtr = True
        time.sleep(0.5)
        self.s.reset_input_buffer()

    def _lines_until_end(self) -> dict:
        out: dict = {}
        while True:
            raw = self.s.readline()
            if not raw:
                raise TimeoutError(f"no answer from the board, got {sorted(out)}")
            line = raw.decode("ascii", errors="replace").strip()
            if line == "END":
                return out
            if not line:
                continue
            tag, _, rest = line.partition(" ")
            if tag in ("FEAT8", "FEATF"):
                n, crc = rest.split()
                data = self.s.read(int(n))
                if len(data) != int(n) or crc16(data) != int(crc, 16):
                    raise IOError(f"{tag}: {len(data)} of {n} bytes or bad CRC")
                out[tag] = data
            elif tag == "ERR":
                out.setdefault("ERR", []).append(rest)
            elif tag in ("STEP",):
                continue
            else:
                out[tag] = rest

    def request(self, line: str, payload: bytes | None = None, retries: int = 2) -> dict:
        for attempt in range(retries + 1):
            self.s.write((line + "\n").encode("ascii"))
            try:
                if payload is not None:
                    ready = self.s.readline().decode("ascii", errors="replace").strip()
                    if ready != "READY":
                        raise IOError(f"expected READY, got {ready!r}")
                    for off in range(0, len(payload), BLOCK_BYTES):
                        self.s.write(payload[off:off + BLOCK_BYTES])
                        ack = self.s.readline().decode("ascii", errors="replace").strip()
                        if ack != "K":
                            raise IOError(f"expected K after block at {off}, got {ack!r}")
                    self.s.write(crc16(payload).to_bytes(2, "big"))
                ans = self._lines_until_end()
            except (TimeoutError, IOError) as e:
                if attempt == retries:
                    raise
                print(f"  retry after: {e}", file=sys.stderr)
                time.sleep(6)
                self.s.reset_input_buffer()
                continue
            if "ERR" in ans and attempt < retries:
                print(f"  retry after ERR: {ans['ERR']}", file=sys.stderr)
                continue
            return ans
        raise RuntimeError("unreachable")


def ints(text: str) -> np.ndarray:
    return np.array([int(x) for x in text.split(",")], dtype=int)


def timing(ans: dict) -> dict:
    return {k: int(v) for k, v in re.findall(r"(\w+)=(\d+)", ans["TIME"])}


def write_json(name: str, res: dict) -> None:
    path = RESULTS / name
    path.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n")
    print("written", path)


def run_v1(b: Board, limit: int) -> None:
    g = np.load(GOLDEN, allow_pickle=True)
    n = min(limit or len(g["labels"]), len(g["labels"]))
    rows, t0 = [], time.time()
    for i in range(n):
        a = b.request("FEATIN 3822", g["input_int8"][i].tobytes())
        p, gap = ints(a["PROB8"]), ints(a["GAP8"])
        lite, ref = g["prob_int8"][i].astype(int), g["prob_int8_tflite217"][i].astype(int)
        rows.append({"clip": i, "label": int(g["labels"][i]), "top_board": int(p.argmax()), "top_litert": int(lite.argmax()), "top_tflite217": int(ref.argmax()),
                     "prob_max_diff_litert": int(np.abs(p - lite).max()), "prob_max_diff_tflite217": int(np.abs(p - ref).max()),
                     "gap_max_diff_litert": int(np.abs(gap - g["gap_int8"][i].astype(int)).max())})
    exact_ref = sum(r["prob_max_diff_tflite217"] == 0 for r in rows)
    exact_lite = sum(r["prob_max_diff_litert"] == 0 for r in rows)
    res = {"clips": n, "classes_covered": len({r["label"] for r in rows}),
           "top_equals_litert": sum(r["top_board"] == r["top_litert"] for r in rows), "top_equals_tflite217": sum(r["top_board"] == r["top_tflite217"] for r in rows),
           "softmax_bit_exact_tflite217": exact_ref, "softmax_bit_exact_litert": exact_lite,
           "softmax_max_diff_tflite217": max(r["prob_max_diff_tflite217"] for r in rows), "softmax_max_diff_litert": max(r["prob_max_diff_litert"] for r in rows),
           "gap_max_diff_litert": max(r["gap_max_diff_litert"] for r in rows), "elapsed_s": time.time() - t0, "rows": rows}
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))
    write_json("device_v1.json", res)


def run_v2(b: Board, limit: int) -> None:
    g = np.load(GOLDEN, allow_pickle=True)
    n = min(limit or len(g["labels"]), len(g["labels"]))
    rows, t0 = [], time.time()
    for i in range(n):
        a = b.request("AUDIO 32000 F", g["audio_int16"][i].astype("<i2").tobytes())
        x = np.frombuffer(a["FEAT8"], dtype=np.int8).reshape(98, 39)
        f = np.frombuffer(a["FEATF"], dtype="<f4").reshape(98, 39)
        p, ref_p = ints(a["PROB8"]), g["prob_int8"][i].astype(int)
        d = np.abs(x.astype(int) - g["input_int8"][i].astype(int))
        tm = timing(a)
        rows.append({"clip": i, "label": int(g["labels"][i]), "feature_max_diff_vs_librosa": float(np.abs(f - g["features_librosa"][i]).max()),
                     "int8_identical_pct": float(100.0 * (d == 0).mean()), "int8_max_diff": int(d.max()), "top_board": int(p.argmax()), "top_pc": int(ref_p.argmax()),
                     "top_probability_diff": abs(int(p.max()) - int(ref_p.max())) / 256.0, "softmax_max_diff": int(np.abs(p - ref_p).max()),
                     "frame_ms": tm["frames_cycles"] / 97 / CLOCK_HZ * 1e3, "finish_ms": tm["finish_cycles"] / CLOCK_HZ * 1e3, "invoke_ms": tm["invoke_cycles"] / CLOCK_HZ * 1e3})
    ok = {"feature_max_diff_le_0p05": all(r["feature_max_diff_vs_librosa"] <= 0.05 for r in rows),
          "int8_identical_ge_98pct_rest_1": all(r["int8_identical_pct"] >= 98 and r["int8_max_diff"] <= 1 for r in rows),
          "top_same_at_least_49_of_50": sum(r["top_board"] == r["top_pc"] for r in rows) >= n - 1,
          "top_probability_diff_le_0p05": all(r["top_probability_diff"] <= 0.05 for r in rows)}
    res = {"clips": n, "classes_covered": len({r["label"] for r in rows}), "criteria": ok, "passes_V2": all(ok.values()),
           "top_same": sum(r["top_board"] == r["top_pc"] for r in rows), "feature_max_diff": max(r["feature_max_diff_vs_librosa"] for r in rows),
           "int8_identical_min_pct": min(r["int8_identical_pct"] for r in rows), "int8_max_diff": max(r["int8_max_diff"] for r in rows),
           "top_probability_diff_max": max(r["top_probability_diff"] for r in rows), "frame_ms_mean": float(np.mean([r["frame_ms"] for r in rows])),
           "finish_ms_mean": float(np.mean([r["finish_ms"] for r in rows])), "invoke_ms_mean": float(np.mean([r["invoke_ms"] for r in rows])), "elapsed_s": time.time() - t0, "rows": rows}
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))
    write_json("device_v2.json", res)


def run_v4(b: Board, limit: int) -> None:
    import soundfile as sf

    from export_tables import build_tables
    from frontend_pc import FrontendC
    from kws_paths import TWO_OUTPUT_MODEL, default_corpus_root, list_test_clips, load_config
    from ai_edge_litert import interpreter as lr

    it = lr.Interpreter(model_path=str(TWO_OUTPUT_MODEL), experimental_op_resolver_type=lr.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    it.allocate_tensors()
    inp = it.get_input_details()[0]
    out15 = [o["index"] for o in it.get_output_details() if o["shape"][-1] == 15][0]
    fe = FrontendC()
    cfg = load_config()
    classes = list(cfg["classes"])
    clips = list_test_clips(default_corpus_root(cfg), classes)
    if limit:
        clips = clips[:limit]

    V4_CLIPS.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if V4_CLIPS.exists():
        for line in V4_CLIPS.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done[r["file"]] = r
    t0 = time.time()
    with V4_CLIPS.open("a", encoding="utf-8") as log:
        for i, (path, cls, label) in enumerate(clips):
            name = Path(path).name
            if name in done:
                continue
            a16, _sr = sf.read(str(path), dtype="int16", always_2d=False)
            a16 = np.pad(a16, (0, 16000 - len(a16)))[:16000].astype("<i2")
            a = b.request("AUDIO 32000", a16.tobytes())
            x = np.frombuffer(a["FEAT8"], dtype=np.int8).reshape(98, 39)
            p = ints(a["PROB8"])
            x_pc, _f = fe.process_clip(a16)
            it.set_tensor(inp["index"], x_pc[np.newaxis, :, :, np.newaxis])
            it.invoke()
            p_pc = it.get_tensor(out15)[0].astype(int)
            d = np.abs(x.astype(int) - x_pc.astype(int))
            tm = timing(a)
            r = {"file": name, "cls": cls, "label": int(label), "top_board": int(p.argmax()), "conf_board": (int(p.max()) + 128) / 256.0, "top_pc": int(p_pc.argmax()),
                 "softmax_max_diff": int(np.abs(p - p_pc).max()), "int8_identical_pct": float(100.0 * (d == 0).mean()), "int8_max_diff": int(d.max()), **tm}
            log.write(json.dumps(r, ensure_ascii=False) + "\n")
            log.flush()
            done[name] = r
            if len(done) % 50 == 0:
                print(f"{len(done)}/{len(clips)} clips, {time.time() - t0:.0f} s this run", flush=True)

    rows = [done[Path(p).name] for p, _c, _l in clips]
    labels = np.array([r["label"] for r in rows])
    top = np.array([r["top_board"] for r in rows])
    top_pc = np.array([r["top_pc"] for r in rows])
    confusion = np.zeros((len(classes), len(classes)), dtype=int)
    for y, t in zip(labels, top):
        confusion[y, t] += 1
    conf = np.array([r["conf_board"] for r in rows])
    res = {"clips": len(rows), "accuracy_board_pct": float(100.0 * (top == labels).mean()), "accuracy_pc_c_frontend_pct": float(100.0 * (top_pc == labels).mean()),
           "passes_V4_within_0p5_points": bool(abs((top == labels).mean() - (top_pc == labels).mean()) * 100.0 <= 0.5),
           "top_differs_from_pc_clips": int((top != top_pc).sum()),
           "correct_and_conf_ge_0p55_pct": float(100.0 * ((top == labels) & (conf >= 0.55)).mean()), "correct_and_conf_ge_0p85_pct": float(100.0 * ((top == labels) & (conf >= 0.85)).mean()),
           "int8_identical_pct_mean": float(np.mean([r["int8_identical_pct"] for r in rows])), "int8_max_diff": max(r["int8_max_diff"] for r in rows), "softmax_max_diff_vs_pc": max(r["softmax_max_diff"] for r in rows),
           "invoke_ms_mean": float(np.mean([r["invoke_cycles"] for r in rows])) / CLOCK_HZ * 1e3, "invoke_ms_max": float(max(r["invoke_cycles"] for r in rows)) / CLOCK_HZ * 1e3,
           "frame_ms_mean": float(np.mean([r["frames_cycles"] for r in rows])) / 97 / CLOCK_HZ * 1e3, "finish_ms_mean": float(np.mean([r["finish_cycles"] for r in rows])) / CLOCK_HZ * 1e3,
           "per_class": [{"class": c, "clips": int((labels == i).sum()), "recall_board_pct": float(100.0 * (top[labels == i] == i).mean()), "recall_pc_pct": float(100.0 * (top_pc[labels == i] == i).mean())} for i, c in enumerate(classes)],
           "confusion_matrix_board": confusion.tolist(), "classes": classes}
    print(json.dumps({k: v for k, v in res.items() if k not in ("confusion_matrix_board", "classes")}, indent=1, ensure_ascii=False))
    write_json("device_v4.json", res)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="COM4")
    ap.add_argument("--ping", action="store_true")
    ap.add_argument("--v1", action="store_true")
    ap.add_argument("--v2", action="store_true")
    ap.add_argument("--v4", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    b = Board(args.port)
    print(b.request("PING").get("PONG"))
    if args.ping:
        print(b.request("MEM").get("MEM"))
    if args.v1:
        run_v1(b, args.limit)
    if args.v2:
        run_v2(b, args.limit)
    if args.v4:
        run_v4(b, args.limit)
    print(b.request("MEM").get("MEM"))


if __name__ == "__main__":
    main()
