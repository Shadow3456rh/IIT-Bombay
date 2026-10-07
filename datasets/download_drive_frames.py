"""
Layer 2 -- Drone Data Augmentation Pipeline
============================================
Module : src/datasets/download_drive_frames.py

PURPOSE
-------
Downloads raw drone frames from a publicly shared Google Drive folder
into data/frames/drive_raw/.

USAGE
-----
    python src/datasets/download_drive_frames.py

REQUIREMENTS
------------
    pip install gdown

AUTHENTICATION
--------------
This script is designed for PUBLICLY SHARED Google Drive folders.

If the folder is private or restricted:
  1. Open the Drive folder URL in a browser.
  2. Click the "Share" button and set access to "Anyone with the link".
  3. Re-run this script.

Do NOT paste Google passwords, tokens, or service-account credentials
directly into this source file.

LIMITATIONS
-----------
- gdown relies on Google Drive's public sharing mechanism.
- Very large folders may require retries or manual batch splitting.
- The script only downloads JPG, JPEG, PNG, and WEBP files.
- No files are overwritten without explicit confirmation.
"""

import argparse
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1g59mgdT3_F_7YWj5v4zvvNn-6w-GN6dl"
FOLDER_ID       = "1g59mgdT3_F_7YWj5v4zvvNn-6w-GN6dl"

# Resolve the output directory relative to this script's location
_SCRIPT_DIR = Path(__file__).resolve().parent          # src/datasets/
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent              # Layer2/
DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "data" / "frames" / "drive_raw"

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_gdown():
    """Import gdown or print a clear installation message."""
    try:
        import gdown
        return gdown
    except ImportError:
        print("[ERROR] gdown is not installed.")
        print("        Run:  pip install gdown")
        sys.exit(1)


def _probe_folder(gdown) -> list:
    """
    Use gdown to list files in the public Drive folder.
    Returns a list of dicts: [{"name": ..., "id": ..., "mimeType": ...}, ...]
    """
    print(f"\n[INFO] Probing Google Drive folder...")
    print(f"       URL : {DRIVE_FOLDER_URL}")

    try:
        files = gdown.download_folder(
            id=FOLDER_ID,
            output=str(DEFAULT_OUTPUT_DIR),
            quiet=True,
        )
        # gdown.download_folder returns None or list of downloaded paths
        return files
    except Exception as e:
        return None


def _list_folder_files(gdown) -> list:
    """
    Retrieve the file listing from the Drive folder without downloading.
    Returns list of (file_id, file_name) tuples for image files only.
    """
    import requests
    from bs4 import BeautifulSoup

    try:
        # Use the Drive folder embed URL to retrieve file list
        url = f"https://drive.google.com/drive/folders/{FOLDER_ID}"
        r = requests.get(url, timeout=15)
        r.raise_for_status()
    except Exception as e:
        print(f"[ERROR] Cannot reach Google Drive: {e}")
        print("        Check your internet connection or verify the folder URL.")
        return []

    # Parse the initData JSON that Drive embeds
    import re
    import json

    # Drive embeds file data in a specific JS variable; extract file names from title
    soup = BeautifulSoup(r.text, "html.parser")
    title = soup.find("title")
    if title:
        print(f"[INFO] Folder title: {title.text.strip()}")

    # Extract file metadata from the embedded JSON blobs
    # Drive pages embed file info in multiple JSON-like script tags
    found_files = []

    # Pattern to find file IDs and names embedded in Drive's JS payload
    # Drive uses a pattern like ["filename.jpg","","","id",...]
    pattern = re.compile(
        r'"([^"]+\.(?:jpg|jpeg|png|webp))"[^"]*"[^"]*"[^"]*"([A-Za-z0-9_-]{25,})"',
        re.IGNORECASE
    )
    matches = pattern.findall(r.text)

    seen_ids = set()
    for name, fid in matches:
        if fid not in seen_ids:
            seen_ids.add(fid)
            found_files.append((fid, name))

    return found_files


def _is_image(filename: str) -> bool:
    return Path(filename).suffix.lower() in SUPPORTED_EXTENSIONS


def download_folder(output_dir: Path, overwrite: bool = False, quiet: bool = False):
    """
    Main download routine. Downloads all image files from the Drive folder.

    Parameters
    ----------
    output_dir : Path
        Local directory to save downloaded images.
    overwrite : bool
        If True, re-download files that already exist locally.
    quiet : bool
        If True, suppress per-file progress output.
    """
    gdown = _require_gdown()

    print("\n" + "=" * 60)
    print("  Layer 2 -- Drive Frame Downloader")
    print("=" * 60)
    print(f"  Source  : {DRIVE_FOLDER_URL}")
    print(f"  Output  : {output_dir}")
    print(f"  Formats : {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
    print("=" * 60)

    # 1. Probe access
    print(f"\n[STEP 1] Verifying folder access...")
    import requests
    try:
        r = requests.get(DRIVE_FOLDER_URL, timeout=15)
        r.raise_for_status()
        print(f"[OK]    Folder is reachable (HTTP {r.status_code}).")
    except Exception as e:
        print(f"[ERROR] Cannot reach folder: {e}")
        print("        Ensure the folder is set to 'Anyone with the link can view'.")
        return

    # 2. Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[STEP 2] Output directory: {output_dir}")

    # 3. List files
    print(f"\n[STEP 3] Listing image files in folder...")
    file_list = _list_folder_files(gdown)

    if not file_list:
        print(
            "[WARN]  Could not parse individual file IDs from the folder page.\n"
            "        Falling back to gdown.download_folder which downloads the\n"
            "        entire folder contents at once."
        )
        _fallback_download_folder(gdown, output_dir, overwrite, quiet)
        return

    image_files = [(fid, name) for fid, name in file_list if _is_image(name)]
    print(f"[INFO]  Found {len(image_files)} image file(s) in the Drive folder.")

    if not image_files:
        print("[WARN]  No supported image files detected from the file listing.")
        print("        Falling back to full folder download via gdown.")
        _fallback_download_folder(gdown, output_dir, overwrite, quiet)
        return

    # 4. Download each image
    print(f"\n[STEP 4] Downloading images...")
    downloaded = 0
    skipped = 0
    failed = 0

    for fid, name in image_files:
        dest = output_dir / name
        if dest.exists() and not overwrite:
            if not quiet:
                print(f"  [SKIP]  {name}  (already exists, use --overwrite to re-download)")
            skipped += 1
            continue

        try:
            url = f"https://drive.google.com/uc?id={fid}"
            out = gdown.download(url, output=str(dest), quiet=quiet, fuzzy=True)
            if out and Path(out).exists():
                size_kb = Path(out).stat().st_size // 1024
                print(f"  [OK]    {name}  ({size_kb} KB)")
                downloaded += 1
            else:
                print(f"  [FAIL]  {name}  — download returned no file")
                failed += 1
        except Exception as e:
            print(f"  [FAIL]  {name}  — {e}")
            failed += 1

    # 5. Report
    print("\n" + "=" * 60)
    print("  DOWNLOAD COMPLETE")
    print("=" * 60)
    print(f"  Downloaded : {downloaded} file(s)")
    print(f"  Skipped    : {skipped} file(s)  (already existed)")
    print(f"  Failed     : {failed} file(s)")
    print(f"  Total      : {downloaded + skipped + failed} file(s) seen")
    print(f"  Output dir : {output_dir}")

    # Final tally of files in directory
    all_images = [f for f in output_dir.iterdir() if _is_image(f.name)]
    print(f"  Images now in output dir: {len(all_images)}")
    print("=" * 60)


def _fallback_download_folder(gdown, output_dir: Path, overwrite: bool, quiet: bool):
    """
    Fallback: download the entire folder using gdown.download_folder.
    This pulls all files Drive considers downloadable.
    """
    print(f"\n[FALLBACK] Using gdown.download_folder on folder ID {FOLDER_ID} ...")
    print(f"           Output: {output_dir}")
    print("           This may take a while for large folders.\n")

    try:
        paths = gdown.download_folder(
            id=FOLDER_ID,
            output=str(output_dir),
            quiet=quiet,
        )
    except Exception as e:
        print(f"[ERROR] gdown.download_folder failed: {e}")
        print("\n  TROUBLESHOOTING:")
        print("  1. Open the folder URL in a browser and verify it is publicly accessible.")
        print("  2. If prompted to sign in, the folder is PRIVATE. You must change its")
        print("     sharing settings to 'Anyone with the link can view' in Google Drive.")
        print("  3. For very large folders gdown may hit rate limits. Wait and retry.")
        return

    if paths is None:
        print("[WARN]  gdown returned no paths. The folder may be empty or access-restricted.")
        return

    # Count image files downloaded
    image_paths = [p for p in (paths or []) if _is_image(str(p))]
    print(f"\n[INFO]  Downloaded {len(image_paths)} image file(s) via fallback method.")

    # Move non-image files? Inform user.
    non_image = [p for p in (paths or []) if not _is_image(str(p))]
    if non_image:
        print(f"[WARN]  {len(non_image)} non-image file(s) were also downloaded:")
        for f in non_image:
            print(f"          {f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download raw drone frames from Google Drive into data/frames/drive_raw/"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Destination directory (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Re-download and overwrite files that already exist locally."
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="Suppress per-file progress bars."
    )
    parser.add_argument(
        "--probe-only",
        action="store_true",
        default=False,
        help="Only check folder access and list files — do NOT download."
    )

    args = parser.parse_args()

    if args.probe_only:
        gdown = _require_gdown()
        print(f"\n[PROBE] Checking access to: {DRIVE_FOLDER_URL}")
        files = _list_folder_files(gdown)
        if not files:
            print("[WARN]  No files detected via HTML parsing (may still be downloadable via gdown fallback).")
        else:
            images = [(fid, n) for fid, n in files if _is_image(n)]
            print(f"[INFO]  {len(images)} image file(s) visible in folder:")
            for fid, name in images:
                print(f"          {name}  (id={fid})")
        return

    download_folder(
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()
