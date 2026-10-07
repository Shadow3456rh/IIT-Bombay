"""
Layer 2 -- Drone Data Augmentation Pipeline
============================================
File : src/test_validate_frames.py

PURPOSE
-------
Deterministic unit tests for validate_frames logic using synthetic temp files.
Does NOT touch data/frames/drive_raw/ -- all tests use temporary directories.
"""

import sys
import io
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from datasets.validate_frames import (
    validate_frames,
    _detect_sequence_gaps,
    _try_open_image,
    FrameInfo,
    ValidationReport,
    SUPPORTED_EXTENSIONS,
)

try:
    from PIL import Image
except ImportError:
    print("[ERROR] Pillow not installed. Run: pip install Pillow")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_failures = 0

def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def check(label: str, condition: bool, detail: str = "") -> None:
    global _failures
    tag = "[PASS]" if condition else "[FAIL]"
    print(f"  {tag}  {label}")
    if detail:
        print(f"         {detail}")
    if not condition:
        _failures += 1

def expect_error(label: str, fn, exc=(ValueError, TypeError)) -> None:
    global _failures
    try:
        fn()
        print(f"  [FAIL]  {label}  -- no exception raised")
        _failures += 1
    except exc as e:
        print(f"  [PASS]  {label}")
        print(f"         {type(e).__name__}: {e}")


def _make_valid_jpeg(path: Path, width: int = 100, height: int = 80) -> None:
    """Create a real, valid JPEG at the given path."""
    img = Image.new("RGB", (width, height), color=(128, 64, 32))
    img.save(path, format="JPEG")


def _make_valid_png(path: Path, width: int = 50, height: int = 50) -> None:
    img = Image.new("RGB", (width, height), color=(0, 128, 255))
    img.save(path, format="PNG")


def _make_corrupt_file(path: Path) -> None:
    """Write random bytes that are not a valid image."""
    path.write_bytes(b"\xFF\xD8\xFF\xE0" + b"\x00" * 50)  # broken JPEG header


def _make_zero_byte(path: Path) -> None:
    path.write_bytes(b"")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_1_empty_directory() -> None:
    section("1. Empty Directory")
    with tempfile.TemporaryDirectory() as tmp:
        report = validate_frames(Path(tmp))
        check("Total files = 0", report.total_files == 0)
        check("Readable = 0", report.readable == 0)
        check("Corrupted = 0", report.corrupted == 0)
        check("No sequence gaps", report.sequence_gaps == [])


def test_2_single_valid_jpeg() -> None:
    section("2. Single Valid JPEG")
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "DJI_0001.jpg"
        _make_valid_jpeg(p, 100, 80)
        report = validate_frames(Path(tmp))
        check("Total = 1", report.total_files == 1)
        check("Readable = 1", report.readable == 1)
        check("Corrupted = 0", report.corrupted == 0)
        check("Dimensions captured", (100, 80) in report.dimension_counts)


def test_3_single_valid_png() -> None:
    section("3. Single Valid PNG")
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "frame_001.png"
        _make_valid_png(p, 640, 480)
        report = validate_frames(Path(tmp))
        check("Total = 1", report.total_files == 1)
        check("Readable = 1", report.readable == 1)
        check("Dimensions captured", (640, 480) in report.dimension_counts)


def test_4_zero_byte_file() -> None:
    section("4. Zero-Byte File")
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "bad.jpg"
        _make_zero_byte(p)
        report = validate_frames(Path(tmp))
        check("Total = 1", report.total_files == 1)
        check("Zero byte counted", report.zero_byte == 1)
        check("Corrupted = 1", report.corrupted == 1)
        check("Readable = 0", report.readable == 0)


def test_5_corrupt_file() -> None:
    section("5. Corrupt File (Invalid Image Data)")
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "corrupt.jpg"
        _make_corrupt_file(p)
        report = validate_frames(Path(tmp))
        check("Total = 1", report.total_files == 1)
        check("Corrupted = 1", report.corrupted == 1)
        check("Readable = 0", report.readable == 0)
        check("Error string populated", report.frames[0].error is not None)


def test_6_mixed_valid_and_invalid() -> None:
    section("6. Mixed: Valid + Corrupt + Zero-Byte")
    with tempfile.TemporaryDirectory() as tmp:
        _make_valid_jpeg(Path(tmp) / "a.jpg", 100, 80)
        _make_corrupt_file(Path(tmp) / "b.jpg")
        _make_zero_byte(Path(tmp) / "c.jpg")
        report = validate_frames(Path(tmp))
        check("Total = 3", report.total_files == 3)
        check("Readable = 1", report.readable == 1)
        check("Corrupted = 2", report.corrupted == 2)
        check("Zero byte = 1", report.zero_byte == 1)


def test_7_dimension_frequency() -> None:
    section("7. Dimension Frequency Counting")
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(3):
            _make_valid_jpeg(Path(tmp) / f"img_{i:03d}.jpg", 1920, 1080)
        _make_valid_jpeg(Path(tmp) / "small.jpg", 640, 480)
        report = validate_frames(Path(tmp))
        check("1920×1080 counted 3", report.dimension_counts.get((1920, 1080)) == 3)
        check("640×480 counted 1", report.dimension_counts.get((640, 480)) == 1)


def test_8_sorted_filename_order() -> None:
    section("8. Filenames Sorted Alphabetically")
    with tempfile.TemporaryDirectory() as tmp:
        for name in ["c.jpg", "a.jpg", "b.jpg"]:
            _make_valid_jpeg(Path(tmp) / name)
        report = validate_frames(Path(tmp))
        names = [f.filename for f in report.frames]
        check("Filenames sorted A->Z", names == sorted(names))


def test_9_to_12_sequence_gaps() -> None:
    section("9-12. Sequence Gap Detection")

    # No gaps
    gaps = _detect_sequence_gaps(["DJI_0336.jpg", "DJI_0337.jpg", "DJI_0338.jpg"])
    check("No gaps when sequence is continuous", gaps == [])

    # Single gap
    gaps = _detect_sequence_gaps(["DJI_0336.jpg", "DJI_0338.jpg"])
    check("Single gap detected", len(gaps) == 1 and "0337" in gaps[0])

    # Multiple gaps
    gaps = _detect_sequence_gaps(["DJI_0336.jpg", "DJI_0340.jpg"])
    check("Multi-gap range described", any("0337" in g or "missing" in g for g in gaps))

    # Non-sequential filenames: no crash
    gaps = _detect_sequence_gaps(["photo.jpg", "image.png", "scan.webp"])
    check("Non-sequential filenames produce no crash", isinstance(gaps, list))


def test_13_supported_extensions() -> None:
    section("13. Only Supported Extensions Collected")
    with tempfile.TemporaryDirectory() as tmp:
        _make_valid_jpeg(Path(tmp) / "frame.jpg")
        (Path(tmp) / "notes.txt").write_text("not an image")
        (Path(tmp) / "model.ply").write_bytes(b"\x00" * 10)
        report = validate_frames(Path(tmp))
        check("Only .jpg counted", report.total_files == 1)


def test_14_invalid_directory() -> None:
    section("14. Invalid Directory Raises Error")
    expect_error("Non-existent directory raises ValueError",
                 lambda: validate_frames(Path("/nonexistent/path/xyz")))


def test_15_frameinfo_fields() -> None:
    section("15. FrameInfo Fields Correct")
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "test.jpg"
        _make_valid_jpeg(p, 200, 150)
        report = validate_frames(Path(tmp))
        fi = report.frames[0]
        check("filename set", fi.filename == "test.jpg")
        check("size_bytes > 0", fi.size_bytes > 0)
        check("is_readable True", fi.is_readable is True)
        check("is_zero_byte False", fi.is_zero_byte is False)
        check("width = 200", fi.width == 200)
        check("height = 150", fi.height == 150)
        check("dimensions property", fi.dimensions == (200, 150))
        check("error is None", fi.error is None)


def test_16_webp_supported() -> None:
    section("16. WEBP Extension Supported")
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "frame.webp"
        img = Image.new("RGB", (100, 100), color=(255, 0, 0))
        img.save(p, format="WEBP")
        report = validate_frames(Path(tmp))
        check("WEBP counted in total", report.total_files == 1)
        check("WEBP readable", report.readable == 1)


def test_17_multiple_same_dimensions() -> None:
    section("17. Multiple Files Same Dimension")
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(5):
            _make_valid_jpeg(Path(tmp) / f"dji_{i:04d}.jpg", 4000, 3000)
        report = validate_frames(Path(tmp))
        check("All 5 at same resolution counted", report.dimension_counts.get((4000, 3000)) == 5)


def main() -> int:
    print("=" * 60)
    print("  Layer 2 -- Frame Validator Unit Tests")
    print("=" * 60)

    test_1_empty_directory()
    test_2_single_valid_jpeg()
    test_3_single_valid_png()
    test_4_zero_byte_file()
    test_5_corrupt_file()
    test_6_mixed_valid_and_invalid()
    test_7_dimension_frequency()
    test_8_sorted_filename_order()
    test_9_to_12_sequence_gaps()
    test_13_supported_extensions()
    test_14_invalid_directory()
    test_15_frameinfo_fields()
    test_16_webp_supported()
    test_17_multiple_same_dimensions()

    section("RESULT")
    if _failures == 0:
        print("  [PASS]  Frame Validator: ALL TESTS PASSED")
    else:
        print(f"  [FAIL]  Frame Validator: {_failures} test(s) failed")
    print()

    return 0 if _failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
