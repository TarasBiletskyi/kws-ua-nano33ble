import numpy as np

from stream_scoring import Detection, GTEvent, decide, score

CLASSES = ["далі", "два", "добре", "зараз", "ні", "один", "ось", "потім", "сьогодні", "так", "там", "три", "тут", "_silence", "_unknown_"]
KW = list(range(13))
TARGETS = CLASSES[:13]


def _gt(center_ms, label):
    return GTEvent(center_ms - 500, center_ms + 500, center_ms, label)


def test_score_perfect():
    gt = [_gt(1000, "так"), _gt(5000, "ні")]
    dets = [Detection(1000, "так"), Detection(5000, "ні")]
    m = score(dets, gt, 60.0, TARGETS)
    assert (m["matched"], m["wrong"], m["missed"], m["false_positives"]) == (2, 0, 0, 0)
    assert m["matched_rate"] == 1.0 and m["false_positives_per_minute"] == 0.0


def test_score_missed_and_false_positive():
    gt = [_gt(1000, "так")]
    m = score([Detection(30000, "два")], gt, 120.0, TARGETS)
    assert (m["matched"], m["missed"], m["false_positives"]) == (0, 1, 1)
    assert m["false_positive_rate"] == 1.0 and m["false_positives_per_minute"] == 0.5


def test_score_wrong_class_claims_event():
    gt = [_gt(1000, "так")]
    m = score([Detection(1200, "там")], gt, 60.0, TARGETS)
    assert (m["matched"], m["wrong"], m["missed"], m["false_positives"]) == (0, 1, 0, 0)
    assert m["per_class"]["так"]["missed"] == 0 and m["per_class"]["там"]["matched"] == 0


def test_score_tolerance_boundary():
    gt = [_gt(1000, "так")]
    assert score([Detection(1750, "так")], gt, 60.0, TARGETS)["matched"] == 1
    m = score([Detection(1751, "так")], gt, 60.0, TARGETS)
    assert m["matched"] == 0 and m["false_positives"] == 1 and m["missed"] == 1


def test_score_no_double_claim():
    gt = [_gt(1000, "так")]
    m = score([Detection(900, "так"), Detection(1100, "так")], gt, 60.0, TARGETS)
    assert m["matched"] == 1 and m["false_positives"] == 1


def _probs(n, hot=None):
    p = np.full((n, 15), 0.02, dtype=np.float32)
    p[:, 13] = 0.72  # silence dominates background windows
    if hot:
        for i, idx, val in hot:
            p[i] = 0.02
            p[i, idx] = val
    return p


def test_decide_k2_threshold_and_suppression():
    times = np.arange(8) * 500.0 + 1000.0
    probs = _probs(8, hot=[(2, 9, 0.9), (3, 9, 0.9), (4, 9, 0.9)])
    dets = decide(times, probs, CLASSES, KW, k=2, threshold=0.7, suppression_ms=1500)
    # window 2 alone averages with window 1 (0.02) -> 0.46 < 0.7; windows 2+3 -> 0.9 -> fire at t[3]
    assert [d.fire_time_ms for d in dets] == [times[3]]
    assert dets[0].label == "так" and abs(dets[0].confidence - 0.9) < 1e-6
    # suppression of 1.5 s blocks the repeat at t[4] (500 ms later)
    dets_short = decide(times, probs, CLASSES, KW, k=2, threshold=0.7, suppression_ms=400)
    assert [d.fire_time_ms for d in dets_short] == [times[3], times[4]]


def test_decide_ignores_background_classes():
    times = np.arange(3) * 500.0
    probs = _probs(3)  # silence 0.72 everywhere, keywords 0.02
    assert decide(times, probs, CLASSES, KW, k=1, threshold=0.5, suppression_ms=0) == []


def test_decide_gate_skips_windows_and_resets_queue():
    times = np.arange(5) * 500.0
    probs = _probs(5, hot=[(2, 4, 0.9), (3, 4, 0.9)])
    # without a gate, K=2 averages windows 2+3 -> 0.9 -> detection at t[3]
    dets = decide(times, probs, CLASSES, KW, k=2, threshold=0.5, suppression_ms=0)
    assert [d.fire_time_ms for d in dets] == [times[2 + 1]]
    # gate rejects window 2 (no inference) and clears the queue: window 3 starts a new queue, 3+4 average to 0.46 < 0.5
    gate = np.array([True, True, False, True, True])
    assert decide(times, probs, CLASSES, KW, k=2, threshold=0.5, suppression_ms=0, gate_pass=gate) == []
