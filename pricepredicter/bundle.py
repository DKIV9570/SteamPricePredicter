"""Pack / unpack the raw data (the only state the daily job keeps; everything in
data/processed is rebuilt from it) as one archive, stored as a GitHub release asset.

    python -m pricepredicter.bundle pack     # data/raw -> data-bundle.tar.gz
    python -m pricepredicter.bundle unpack   # data-bundle.tar.gz -> data/raw
"""
import sys
import tarfile

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


if __name__ == "__main__":
    {"pack": pack, "unpack": unpack}[sys.argv[1]]()
