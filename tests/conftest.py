import os
import sys
import tempfile
from pathlib import Path

# Isolate every test session in a throwaway data dir and use the deterministic fake embedder,
# so unit tests never touch data/wardrobe.db or need model weights.
if "WARDROBE_DATA_DIR" not in os.environ:
    os.environ["WARDROBE_DATA_DIR"] = tempfile.mkdtemp(prefix="wardrobe-test-")
os.environ.setdefault("WARDROBE_FAKE_EMBEDDER", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
