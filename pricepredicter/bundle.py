"""Pack / unpack the raw data (the only state the daily job keeps; everything in
data/processed is rebuilt from it) as one archive, stored as a GitHub release asset.

    python -m pricepredicter.bundle pack           # data/raw -> data-bundle.tar.gz
    python -m pricepredicter.bundle unpack         # data-bundle.tar.gz -> data/raw
    python -m pricepredicter.bundle merge <file>   # add another bundle's data to data/raw
"""
import json
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

from .config import RAW, ROOT

ARCHIVE = ROOT / "data-bundle.tar.gz"
SKIP = {"steamspy"}  # SteamSpy catalog pages: already folded into catalog.json


def pack():
    with tarfile.open(ARCHIVE, "w:gz", compresslevel=6) as tar:
        for p in sorted(RAW.iterdir()):
            if p.name not in SKIP:
                tar.add(p, arcname=f"raw/{p.name}")
    print(f"{ARCHIVE.name}: {ARCHIVE.stat().st_size / 1e6:.0f} MB")


def unpack():
    RAW.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(ARCHIVE, "r:gz") as tar:
        tar.extractall(RAW.parent, filter="data")
    print(f"unpacked into {RAW}")


def merge(archive: str):
    """Fold another copy of the data (e.g. what a CI run collected while a local run was also
    collecting) into data/raw: per-app files that are missing or older here are copied;
    top-level JSON maps (catalog, ids, names) are unioned, keeping local entries."""
    added = replaced = 0
    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(tmp, filter="data")
        other = Path(tmp) / "raw"
        for src in other.rglob("*.json"):
            dst = RAW / src.relative_to(other)
            if src.parent == other:  # top-level map
                if dst.exists():
                    mine = json.loads(dst.read_text(encoding="utf-8"))
                    theirs = json.loads(src.read_text(encoding="utf-8"))
                    dst.write_text(json.dumps(theirs | mine, ensure_ascii=False), encoding="utf-8")
                else:
                    shutil.copy2(src, dst)
                continue
            if not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                added += 1
            elif src.stat().st_mtime > dst.stat().st_mtime + 1:
                shutil.copy2(src, dst)
                replaced += 1
    print(f"merged: {added} new files, {replaced} newer files")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "merge":
        merge(sys.argv[2])
    else:
        {"pack": pack, "unpack": unpack}[cmd]()
