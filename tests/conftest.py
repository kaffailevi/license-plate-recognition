import sys
from pathlib import Path

SRC_DIR = Path(__file__).parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from inference import ensure_dummy_assets_ready


def pytest_configure():
    ensure_dummy_assets_ready()
