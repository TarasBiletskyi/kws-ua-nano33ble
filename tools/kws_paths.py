"""Shared paths, configuration and corpus helpers for the PC-side tools.

Nothing here imports TensorFlow. The model repository is the git submodule
``external/kws-model-evaluation``; its ``config.yaml`` is the single source of
feature/model parameters and of the class order.
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_REPO = REPO_ROOT / "external" / "kws-model-evaluation"
FIRMWARE_DIR = REPO_ROOT / "firmware" / "kws_nano33"
MODEL_DIR = FIRMWARE_DIR / "model"
GOLDEN_DIR = REPO_ROOT / "tools" / "golden"
RESULTS_DIR = REPO_ROOT / "report" / "results"
PUBLISHED_MODEL = MODEL_REPO / "results" / "tflite" / "with_aug_int8.tflite"
TWO_OUTPUT_MODEL = MODEL_DIR / "with_aug_two_output_int8.tflite"
CHECKPOINT = MODEL_REPO / "results" / "checkpoints" / "with_aug" / "best_model.keras"

BACKGROUND_CLASSES = {"_silence", "_unknown_"}

# Ukrainian -> ASCII transliteration for serial output (national standard 2010, simplified).
_TRANSLIT = {
    "а": "a", "б": "b", "в": "v", "г": "h", "ґ": "g", "д": "d", "е": "e", "є": "ie", "ж": "zh", "з": "z",
    "и": "y", "і": "i", "ї": "i", "й": "i", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o", "п": "p",
    "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f", "х": "kh", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "shch",
    "ь": "", "ю": "iu", "я": "ia", "'": "",
}


def add_model_repo_to_path() -> None:
    """Make ``src.features`` etc. of the model repository importable."""
    if str(MODEL_REPO) not in sys.path:
        sys.path.insert(0, str(MODEL_REPO))


def load_config() -> dict:
    return yaml.safe_load((MODEL_REPO / "config.yaml").read_text(encoding="utf-8"))


def default_corpus_root(cfg: dict | None = None) -> Path:
    """Corpus checkout: sibling of this repository, else the model repo's relative setting."""
    candidates = [REPO_ROOT.parent / "common-voice-ua-kws-corpus"]
    if cfg is not None:
        candidates.append((MODEL_REPO / cfg["corpus_root"]).resolve())
    for c in candidates:
        if (c / "splits").exists():
            return c
    raise FileNotFoundError(f"corpus not found; tried {[str(c) for c in candidates]}")


def default_cache_dir() -> Path | None:
    """The MFCC feature cache of the model repository (needed only for calibration-identical conversion)."""
    for c in (REPO_ROOT.parent / "model-evaluation" / "cache" / "mfcc", MODEL_REPO / "cache" / "mfcc"):
        if c.exists():
            return c
    return None


def transliterate(word: str) -> str:
    return "".join(_TRANSLIT.get(ch, ch) for ch in word.lower())


def read_split(corpus_root: Path, classes: list[str], split: str = "test") -> list[tuple[str, str, int]]:
    """Return (stem, class_name, class_index) for every entry of ``splits/<class>/<split>.txt``.

    Mirrors ``src.dataset.load_splits`` (class order of config.yaml, file order inside each list),
    but without the feature-cache dependency.
    """
    out: list[tuple[str, str, int]] = []
    for idx, cls in enumerate(classes):
        cls_dir = next((d for d in (corpus_root / "splits" / cls, corpus_root / "splits" / (cls + "_")) if d.exists()), None)
        if cls_dir is None:
            raise FileNotFoundError(f"splits dir not found for {cls}")
        for line in (cls_dir / f"{split}.txt").read_text(encoding="utf-8").strip().splitlines():
            stem = Path(line.strip()).stem
            if stem:
                out.append((stem, cls, idx))
    return out


def wav_path(corpus_root: Path, cls: str, stem: str) -> Path | None:
    """Keyword clips live only in ``dataset/cleaned`` (``raw`` still holds the clips the cleanup removed, so it must not
    be used as a fallback); the auxiliary classes live only in ``dataset/raw``. None if the clip was removed."""
    sub = "raw" if cls in BACKGROUND_CLASSES else "cleaned"
    p = corpus_root / "dataset" / sub / cls / f"{stem}.wav"
    return p if p.exists() else None


def list_test_clips(corpus_root: Path, classes: list[str]) -> list[tuple[Path, str, int]]:
    """(wav_path, class, index) for the test split, in the loader's order, skipping cleanup-removed clips."""
    clips = []
    for stem, cls, idx in read_split(corpus_root, classes, "test"):
        p = wav_path(corpus_root, cls, stem)
        if p is not None:
            clips.append((p, cls, idx))
    return clips
