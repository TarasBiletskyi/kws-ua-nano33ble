import pytest

from kws_paths import BACKGROUND_CLASSES, default_corpus_root, list_test_clips, read_split

try:
    CORPUS = default_corpus_root()
except FileNotFoundError:
    CORPUS = None

pytestmark = pytest.mark.skipif(CORPUS is None, reason="corpus checkout not available")


def test_test_split_counts_match_the_article(cfg):
    listed = read_split(CORPUS, cfg["classes"], "test")
    clips = list_test_clips(CORPUS, cfg["classes"])
    assert len(listed) == 1223          # listed in splits/*/test.txt
    assert len(clips) == 1204           # minus the 19 test clips removed by the corpus cleanup


def test_keyword_clips_come_from_cleaned_and_aux_from_raw(cfg):
    for p, cls, _ in list_test_clips(CORPUS, cfg["classes"]):
        expected = "raw" if cls in BACKGROUND_CLASSES else "cleaned"
        assert p.parent.parent.name == expected, (cls, p)
