"""
Layer 2 -- Drone Data Augmentation Pipeline
============================================
Module : src/datasets/validate_frames.py

PURPOSE
-------
Read-only inspection of downloaded drone frames.

Reports:
  - total image files found
  - readable vs. corrupted/unreadable images
  - zero-byte files
  - image dimensions and their frequencies
  - sorted filename list
  - missing sequence gaps (for DJI-style sequential names)

POLICY
------
This script is STRICTLY READ-ONLY.
It does NOT delete, rename, overwrite, move, or download any files.

DEPENDENCY
----------
Pillow (pip install Pillow) — pure image I/O, no GPU required.
"""

import argparse
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from PIL import Image, UnidentifiedImageError
except ImportError:
    print("[ERROR] Pillow is not installed. Run:  pip install Pillow")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass
class FrameInfo:
    path: Path
    filename: str
    size_bytes: int
    is_zero_byte: bool
    is_readable: bool
    width: Optional[int] = None
    height: Optional[int] = None
    error: Optional[str] = None

    @property
    def dimensions(self) -> Optional[Tuple[int, int]]:
        if self.width is not None and self.height is not None:
            return (self.width, self.height)
        return None


@dataclass
class ValidationReport:
    directory: Path
    total_files: int = 0
    readable: int = 0
    corrupted: int = 0
    zero_byte: int = 0
    frames: List[FrameInfo] = field(default_factory=list)
    dimension_counts: Dict[Tuple[int, int], int] = field(default_factory=dict)
    sequence_gaps: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core validation logic
# ---------------------------------------------------------------------------

def _try_open_image(path: Path) -> Tuple[bool, Optional[int], Optional[int], Optional[str]]:
    """
    Attempt to open and verify an image file.

    Returns
    -------
    (is_readable, width, height, error_message)
    """
    try:
        with Image.open(path) as img:
            img.verify()          # detect truncation / corruption
        # Re-open after verify (verify() closes the stream)
        with Image.open(path) as img:
            w, h = img.size
        return True, w, h, None
    except UnidentifiedImageError as e:
        return False, None, None, f"UnidentifiedImage: {e}"
    except Exception as e:
        return False, None, None, f"{type(e).__name__}: {e}"


def _detect_sequence_gaps(filenames: List[str]) -> List[str]:
    """
    For filenames following the pattern PREFIX_NNNN.EXT,
    detect numeric gaps in the sequence.

    Returns a list of human-readable gap descriptions.
    """
    # Extract numeric index from each filename
    pattern = re.compile(r"^(.+?)_(\d+)(\.[^.]+)$")
    entries: List[Tuple[str, int, str]] = []  # (prefix, number, ext)

    for fn in filenames:
        m = pattern.match(fn)
        if m:
            entries.append((m.group(1), int(m.group(2)), m.group(3)))

    if not entries:
        return []

    # Group by (prefix, ext) — only look for gaps within the same series
    from collections import defaultdict
    groups: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for prefix, num, ext in entries:
        groups[(prefix, ext)].append(num)

    gaps = []
    for (prefix, ext), numbers in groups.items():
        if len(numbers) < 2:
            continue
        numbers_sorted = sorted(numbers)
        for i in range(len(numbers_sorted) - 1):
            a = numbers_sorted[i]
            b = numbers_sorted[i + 1]
            if b - a > 1:
                missing = list(range(a + 1, b))
                if len(missing) == 1:
                    gaps.append(f"{prefix}_{missing[0]:04d}{ext}")
                else:
                    gaps.append(
                        f"{prefix}_{missing[0]:04d}{ext} … {prefix}_{missing[-1]:04d}{ext} "
                        f"({len(missing)} missing)"
                    )
    return gaps


def validate_frames(directory: Path) -> ValidationReport:
    """
    Inspect all image files in ``directory`` and return a ValidationReport.

    Parameters
    ----------
    directory : Path
        The folder to scan. Must exist.

    Returns
    -------
    ValidationReport
    """
    if not directory.exists():
        raise ValueError(f"Directory does not exist: {directory}")
    if not directory.is_dir():
        raise ValueError(f"Path is not a directory: {directory}")

    report = ValidationReport(directory=directory)

    # Collect candidate files
    candidates = sorted(
        (f for f in directory.iterdir()
         if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS),
        key=lambda f: f.name
    )

    report.total_files = len(candidates)

    dim_counter: Counter = Counter()

    for path in candidates:
        size = path.stat().st_size
        is_zero = size == 0

        if is_zero:
            frame = FrameInfo(
                path=path,
                filename=path.name,
                size_bytes=0,
                is_zero_byte=True,
                is_readable=False,
                error="Zero-byte file"
            )
            report.zero_byte += 1
            report.corrupted += 1
        else:
            ok, w, h, err = _try_open_image(path)
            frame = FrameInfo(
                path=path,
                filename=path.name,
                size_bytes=size,
                is_zero_byte=False,
                is_readable=ok,
                width=w,
                height=h,
                error=err
            )
            if ok:
                report.readable += 1
                dim_counter[(w, h)] += 1
            else:
                report.corrupted += 1

        report.frames.append(frame)

    report.dimension_counts = dict(dim_counter)

    # Sequence gap detection
    all_filenames = [f.filename for f in report.frames]
    report.sequence_gaps = _detect_sequence_gaps(all_filenames)

    return report


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(report: ValidationReport) -> None:
    W = 60
    print("\n" + "=" * W)
    print("  Layer 2 -- Frame Validation Report")
    print("=" * W)
    print(f"  Directory : {report.directory}")
    print(f"  Total files found : {report.total_files}")
    print(f"  Readable          : {report.readable}")
    print(f"  Corrupted         : {report.corrupted}")
    print(f"  Zero-byte         : {report.zero_byte}")
    print("=" * W)

    # Dimension frequencies
    print("\n--- Dimension Frequencies ---")
    if report.dimension_counts:
        for (w, h), count in sorted(report.dimension_counts.items()):
            print(f"  {w} × {h}  :  {count} file(s)")
    else:
        print("  (no readable images)")

    # Filenames
    print(f"\n--- Filenames (sorted, {len(report.frames)} total) ---")
    for frame in report.frames:
        status = "OK" if frame.is_readable else ("ZERO" if frame.is_zero_byte else "CORRUPT")
        dim_str = f"  [{frame.width}×{frame.height}]" if frame.dimensions else ""
        print(f"  [{status:6s}]  {frame.filename}{dim_str}")

    # Corrupted detail
    corrupted = [f for f in report.frames if not f.is_readable]
    if corrupted:
        print("\n--- Corrupted / Unreadable Files ---")
        for f in corrupted:
            print(f"  {f.filename}  →  {f.error}")

    # Sequence gaps
    print("\n--- Filename Sequence Gaps ---")
    if report.sequence_gaps:
        for gap in report.sequence_gaps:
            print(f"  MISSING: {gap}")
    else:
        print("  No gaps detected.")

    print("\n" + "=" * W)
    result = "PASS" if report.corrupted == 0 else "WARN"
    print(f"  Validation Result: [{result}]")
    print("=" * W + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _resolve_default_dir() -> Path:
    script_dir = Path(__file__).resolve().parent       # src/datasets/
    return script_dir.parent.parent / "data" / "frames" / "drive_raw"


def main():
    parser = argparse.ArgumentParser(
        description="Read-only validation of downloaded drone frames."
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=_resolve_default_dir(),
        help="Directory to validate (default: data/frames/drive_raw/)"
    )
    args = parser.parse_args()

    report = validate_frames(args.dir)
    print_report(report)

    # Exit non-zero if any corruption found
    sys.exit(1 if report.corrupted > 0 else 0)


if __name__ == "__main__":
    main()
