import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1]
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from kws_paths import add_model_repo_to_path, load_config  # noqa: E402

add_model_repo_to_path()


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def tables(cfg):
    from export_tables import build_tables
    return build_tables(cfg, model_path=None)
