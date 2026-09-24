"""Reading the per-app JSON files under data/raw."""
import json
from pathlib import Path


def iter_json(directory: Path):
    """Yield (path, parsed) for every *.json file, skipping (and reporting) unreadable ones —
    e.g. a file cut short when a fetch was interrupted — instead of failing the whole build."""
    bad = []
    for p in directory.glob("*.json"):
        try:
            yield p, json.loads(p.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            bad.append(p.name)
    if bad:
        print(f"skipped {len(bad)} unreadable file(s) in {directory.name}: {bad[:5]}")
