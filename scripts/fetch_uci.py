"""Download UCI dataset 395 into data/raw/uci and verify its checksum.

Isenkul et al. 2014, "Parkinson Disease Spiral Drawings Using Digitized Graphics
Tablet", CC BY 4.0. https://archive.ics.uci.edu/dataset/395
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = (
    "https://archive.ics.uci.edu/static/public/395/"
    "parkinson+disease+spiral+drawings+using+digitized+graphics+tablet.zip"
)
SHA256 = "4181243d64c29382a0035a9bca3ebcf446079b65c70a0b343fe7975372dc92e6"  # fetched 2026-11-16
DEST = Path(__file__).resolve().parent.parent / "data" / "raw" / "uci"


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    zpath = DEST / "uci395.zip"
    if not zpath.exists():
        print(f"downloading {URL}")
        with urllib.request.urlopen(URL, timeout=60) as r:
            zpath.write_bytes(r.read())
    digest = hashlib.sha256(zpath.read_bytes()).hexdigest()
    if digest != SHA256:
        print(f"checksum mismatch: got {digest}, expected {SHA256}. Delete {zpath} and retry.")
        return 1
    with zipfile.ZipFile(zpath) as z:
        for name in z.namelist():
            # Refuse paths that would escape the destination folder.
            target = (DEST / name).resolve()
            if not str(target).startswith(str(DEST.resolve())):
                print(f"refusing unsafe path in zip: {name}")
                return 1
        z.extractall(DEST)
    print(f"ok: {DEST} (sha256 {digest[:12]}…)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
