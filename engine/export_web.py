"""Export the taxonomy (single source of truth) for the browser build.

    uv run python -m engine.export_web      -> web-static/src/taxonomy.json
"""
import json

from engine import config
from engine.taxonomy import SEASONS, VIBES, anchor_rows, taxonomy_hash

OUT = config.ROOT / "web-static" / "src" / "taxonomy.json"


def main():
    rows = anchor_rows()
    OUT.write_text(json.dumps({"hash": taxonomy_hash(rows), "vibes": VIBES, "seasons": SEASONS, "rows": rows},
                              indent=1))
    print(f"Wrote {len(rows)} anchor rows -> {OUT}")


if __name__ == "__main__":
    main()
