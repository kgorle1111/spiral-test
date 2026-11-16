"""Loader for UCI dataset 395 (Isenkul et al. 2014, CC BY 4.0).

Format: ';'-separated, no header: X;Y;Z;Pressure;GripAngle;Timestamp;TestID.
TestID 0 = static spiral, 1 = dynamic spiral, 2 = circular motion (not a spiral).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# kn: assumed scale. The dataset gives tablet pixels with no calibration. A Wacom
# Cintiq 12WX shows 1280x800 px over 261.1x163.2 mm, i.e. 0.204 mm/px. Frequency
# results do not depend on this; every mm value on UCI data carries it.
MM_PER_PX = 0.204

ROOT = Path(__file__).resolve().parent.parent / "data" / "raw" / "uci"


@dataclass
class Drawing:
    subject: str  # file stem, e.g. C_0001
    group: str  # "control" or "parkinson"
    source: str  # "hw_dataset" or "new_dataset"
    test: int
    t_ms: np.ndarray
    x_mm: np.ndarray
    y_mm: np.ndarray


def _read(path: Path) -> np.ndarray:
    rows = []
    for line in path.read_text().splitlines():
        parts = line.strip().split(";")
        if len(parts) != 7:
            continue
        try:
            rows.append([float(p) for p in parts])
        except ValueError:
            continue
    return np.asarray(rows, dtype=float)


def load_all(root: Path = ROOT, tests=(0, 1)) -> list[Drawing]:
    if not root.exists():
        raise FileNotFoundError(f"{root} missing. Run: uv run python scripts/fetch_uci.py")
    out = []
    for source in ("hw_dataset", "new_dataset"):
        for group in ("control", "parkinson"):
            d = root / source / group
            if not d.exists():
                continue
            for f in sorted(d.glob("*.txt")):
                arr = _read(f)
                for tid in tests:
                    sel = arr[arr[:, 6] == tid]
                    if sel.shape[0] < 50:
                        continue
                    out.append(
                        Drawing(
                            subject=f.stem,
                            group=group,
                            source=source,
                            test=int(tid),
                            t_ms=sel[:, 5],
                            x_mm=sel[:, 0] * MM_PER_PX,
                            # Screen y grows downward; flip so angles run the math way.
                            y_mm=-sel[:, 1] * MM_PER_PX,
                        )
                    )
    return out
