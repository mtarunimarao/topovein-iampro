"""
TopoVein — File 1: Dataset Explorer
=====================================
Task: Dataset exploration — understand the available finger-vein dataset
layout before batch preprocessing.

It auto-detects one of the supported dataset layouts and prints:
  - How many subjects exist
  - How many sessions
  - How many fingers per subject
  - How many images per finger
  - Image sizes and any corrupted files

Usage:
    python 01_explore_dataset.py

Output:
    dataset_report.txt  — full text report
    dataset_index.csv   — one row per image (for batch processing)
"""

import os
import cv2
import csv
import json
from pathlib import Path
from collections import defaultdict

# ─────────────────────────────────────────
# CONFIGURE THIS — supported dataset roots
# ─────────────────────────────────────────
DATASET_CANDIDATES = (
    Path("Dataset/Published_database_FV-USM_Dec2013"),
    Path("Dataset/Published_database_UTFVP"),
    Path("Dataset/20151211-utwente-vingervein/dataset"),
    Path("Dataset/dataset"),
)
OUTPUT_REPORT = "results/dataset_report.txt"
OUTPUT_CSV    = "results/dataset_index.csv"
OUTPUT_JSON   = "results/dataset_stats.json"
IMAGE_PATTERNS = ("*.jpg", "*.png", "*.bmp")


def _contains_images(root: Path) -> bool:
    for pattern in IMAGE_PATTERNS:
        if next(root.rglob(pattern), None) is not None:
            return True
    return False


def _looks_like_fv_usm(root: Path) -> bool:
    if not root.exists() or not root.is_dir():
        return False

    for session_dir in root.iterdir():
        if not session_dir.is_dir():
            continue
        raw_data_dir = session_dir / "raw_data"
        subject_dirs = []

        if raw_data_dir.exists():
            subject_dirs = [child for child in raw_data_dir.iterdir() if child.is_dir()]
        else:
            subject_dirs = [child for child in session_dir.iterdir() if child.is_dir()]

        if not subject_dirs:
            continue

        # FV-USM uses folders like 001_1, 001_2, ...
        if not any("_" in child.name for child in subject_dirs):
            continue

        if _contains_images(root):
            return True
    return False


def _looks_like_utfvp(root: Path) -> bool:
    data_dir = root / "data"
    return root.exists() and data_dir.exists() and next(data_dir.glob("*/*.png"), None) is not None


def detect_dataset_root():
    for candidate in DATASET_CANDIDATES:
        if _looks_like_utfvp(candidate):
            return candidate, "UTFVP"
        if _looks_like_fv_usm(candidate):
            return candidate, "FV-USM"

    looked_for = "\n".join(f"  - {candidate}" for candidate in DATASET_CANDIDATES)
    raise FileNotFoundError(
        "No supported dataset root was found.\n"
        "Looked for:\n"
        f"{looked_for}"
    )


def explore_fv_usm_dataset(root):
    """
    Walk the FV-USM folder tree and collect metadata for every image.

    FV-USM structure:
        <root>/
          <session>/          e.g. 1st_session, 2nd_session
            raw_data/
              <subject_finger>/   e.g. 001_1, 001_2, 002_1 ...
                01.jpg ... 06.jpg
    """
    records   = []   # one dict per image
    errors    = []   # corrupted or unreadable files
    stats     = defaultdict(set)

    root_path = Path(root)
    print(f"\nScanning FV-USM dataset: {root}\n{'─'*55}")

    for session_dir in sorted(root_path.iterdir()):
        if not session_dir.is_dir():
            continue
        session_name = session_dir.name

        raw_data_dir = session_dir / "raw_data"
        if not raw_data_dir.exists():
            # try direct children
            raw_data_dir = session_dir

        for subject_finger_dir in sorted(raw_data_dir.iterdir()):
            if not subject_finger_dir.is_dir():
                continue

            folder_name = subject_finger_dir.name  # e.g. "001_1"

            # Parse subject ID and finger ID from folder name
            parts = folder_name.split("_")
            if len(parts) >= 2:
                subject_id = parts[0]           # "001"
                finger_id  = parts[1]           # "1"
            else:
                subject_id = folder_name
                finger_id  = "?"

            stats["sessions"].add(session_name)
            stats["subjects"].add(subject_id)
            stats["fingers"].add(f"{subject_id}_{finger_id}")

            image_files = sorted(subject_finger_dir.glob("*.jpg")) + \
                          sorted(subject_finger_dir.glob("*.png")) + \
                          sorted(subject_finger_dir.glob("*.bmp"))

            for img_path in image_files:
                img = cv2.imread(str(img_path))

                if img is None:
                    errors.append(str(img_path))
                    records.append({
                        "path":       str(img_path),
                        "session":    session_name,
                        "subject_id": subject_id,
                        "finger_id":  finger_id,
                        "image_name": img_path.name,
                        "height":     -1,
                        "width":      -1,
                        "channels":   -1,
                        "size_bytes": img_path.stat().st_size,
                        "dataset_name": "FV-USM",
                        "source_image_name": img_path.name,
                        "status":     "CORRUPTED",
                    })
                    continue

                h, w = img.shape[:2]
                ch   = img.shape[2] if len(img.shape) == 3 else 1

                stats["heights"].add(h)
                stats["widths"].add(w)
                stats["total_images"] = stats.get("total_images", 0) + 1

                records.append({
                    "path":       str(img_path),
                    "session":    session_name,
                    "subject_id": subject_id,
                    "finger_id":  finger_id,
                    "image_name": img_path.name,
                    "height":     h,
                    "width":      w,
                    "channels":   ch,
                    "size_bytes": img_path.stat().st_size,
                    "dataset_name": "FV-USM",
                    "source_image_name": img_path.name,
                    "status":     "OK",
                })

    return records, errors, stats


def _map_utfvp_capture(capture_id: str) -> tuple[str, str]:
    capture_number = int(capture_id)
    if capture_number in (1, 2):
        return "1st_session", f"{capture_number:02d}.png"
    if capture_number in (3, 4):
        return "2nd_session", f"{capture_number - 2:02d}.png"
    return f"session_{capture_number}", f"{capture_number:02d}.png"


def explore_utfvp_dataset(root):
    """
    Walk the UTFVP dataset tree and map it into the same logical fields the
    rest of the TopoVein pipeline already expects.

    UTFVP structure:
        <root>/
          data/
            <subject_id>/          e.g. 0001
              0001_1_1_<timestamp>.png
              0001_1_2_<timestamp>.png
              0001_1_3_<timestamp>.png
              0001_1_4_<timestamp>.png

    The third token indicates capture order. Captures 1-2 are mapped to
    `1st_session` and captures 3-4 to `2nd_session`.
    """
    records   = []
    errors    = []
    stats     = defaultdict(set)

    root_path = Path(root)
    data_dir = root_path / "data"
    print(f"\nScanning UTFVP dataset: {root}\n{'─'*55}")

    for subject_dir in sorted(data_dir.iterdir()):
        if not subject_dir.is_dir():
            continue

        for img_path in sorted(subject_dir.glob("*.png")):
            parts = img_path.stem.split("_")
            if len(parts) < 4:
                continue

            subject_id = parts[0]
            finger_id = parts[1]
            capture_id = parts[2]
            session_name, normalized_image_name = _map_utfvp_capture(capture_id)

            stats["sessions"].add(session_name)
            stats["subjects"].add(subject_id)
            stats["fingers"].add(f"{subject_id}_{finger_id}")

            img = cv2.imread(str(img_path))

            if img is None:
                errors.append(str(img_path))
                records.append({
                    "path":       str(img_path),
                    "session":    session_name,
                    "subject_id": subject_id,
                    "finger_id":  finger_id,
                    "image_name": normalized_image_name,
                    "height":     -1,
                    "width":      -1,
                    "channels":   -1,
                    "size_bytes": img_path.stat().st_size,
                    "dataset_name": "UTFVP",
                    "source_image_name": img_path.name,
                    "source_capture_id": capture_id,
                    "status":     "CORRUPTED",
                })
                continue

            h, w = img.shape[:2]
            ch   = img.shape[2] if len(img.shape) == 3 else 1

            stats["heights"].add(h)
            stats["widths"].add(w)
            stats["total_images"] = stats.get("total_images", 0) + 1

            records.append({
                "path":       str(img_path),
                "session":    session_name,
                "subject_id": subject_id,
                "finger_id":  finger_id,
                "image_name": normalized_image_name,
                "height":     h,
                "width":      w,
                "channels":   ch,
                "size_bytes": img_path.stat().st_size,
                "dataset_name": "UTFVP",
                "source_image_name": img_path.name,
                "source_capture_id": capture_id,
                "status":     "OK",
            })

    return records, errors, stats


def explore_dataset(root, dataset_name):
    if dataset_name == "FV-USM":
        return explore_fv_usm_dataset(root)
    if dataset_name == "UTFVP":
        return explore_utfvp_dataset(root)
    raise ValueError(f"Unsupported dataset type: {dataset_name}")


def print_report(records, errors, stats, dataset_name, dataset_root):
    lines = []
    lines.append("=" * 60)
    lines.append("  TopoVein — Finger-Vein Dataset Exploration Report")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Dataset name : {dataset_name}")
    lines.append(f"Dataset root : {dataset_root}")
    lines.append("")

    # Summary
    ok_records = [r for r in records if r["status"] == "OK"]
    lines.append("SUMMARY")
    lines.append("─" * 40)
    lines.append(f"  Total sessions  : {len(stats['sessions'])}")
    lines.append(f"  Total subjects  : {len(stats['subjects'])}")
    lines.append(f"  Unique fingers  : {len(stats['fingers'])}")
    lines.append(f"  Total images    : {len(ok_records)}")
    lines.append(f"  Corrupted files : {len(errors)}")
    lines.append("")

    # Image dimensions
    lines.append("IMAGE DIMENSIONS")
    lines.append("─" * 40)
    if ok_records:
        heights = [r["height"] for r in ok_records]
        widths  = [r["width"]  for r in ok_records]
        lines.append(f"  Height range : {min(heights)} – {max(heights)} px")
        lines.append(f"  Width  range : {min(widths)}  – {max(widths)} px")
        if len(set(heights)) == 1 and len(set(widths)) == 1:
            lines.append(f"  All images   : SAME SIZE ✓  ({widths[0]} × {heights[0]})")
        else:
            lines.append(f"  WARNING: images have different sizes — resize needed before batch processing")
    lines.append("")

    # Per-session breakdown
    lines.append("PER-SESSION BREAKDOWN")
    lines.append("─" * 40)
    sessions = sorted(set(r["session"] for r in ok_records))
    for s in sessions:
        sess_imgs = [r for r in ok_records if r["session"] == s]
        subjs     = set(r["subject_id"] for r in sess_imgs)
        lines.append(f"  {s:<20} : {len(sess_imgs):>4} images  |  {len(subjs)} subjects")
    lines.append("")

    # Per-subject sample
    lines.append("PER-SUBJECT SAMPLE (first 10 subjects)")
    lines.append("─" * 40)
    subjects = sorted(set(r["subject_id"] for r in ok_records))[:10]
    for subj in subjects:
        subj_imgs   = [r for r in ok_records if r["subject_id"] == subj]
        fingers     = set(r["finger_id"] for r in subj_imgs)
        sessions_s  = set(r["session"]   for r in subj_imgs)
        lines.append(f"  Subject {subj} : {len(subj_imgs):>3} images | fingers={sorted(fingers)} | sessions={len(sessions_s)}")
    lines.append("")

    # Errors
    if errors:
        lines.append("CORRUPTED / UNREADABLE FILES")
        lines.append("─" * 40)
        for e in errors:
            lines.append(f"  ✗ {e}")
        lines.append("")

    lines.append("=" * 60)
    lines.append("Files saved:")
    lines.append(f"  {OUTPUT_CSV}   — full image index for batch processing")
    lines.append(f"  {OUTPUT_JSON}  — stats summary")
    lines.append("=" * 60)

    report_text = "\n".join(lines)
    print(report_text)

    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\nReport saved → {OUTPUT_REPORT}")

    return report_text


def save_csv_index(records):
    """Save full image index as CSV — used by preprocessing scripts."""
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print(f"CSV index saved → {OUTPUT_CSV}  ({len(records)} rows)")


def save_json_stats(stats, dataset_name, dataset_root):
    """Save stats as JSON for later use."""
    serializable = {k: list(v) if isinstance(v, set) else v
                    for k, v in stats.items()}
    serializable["dataset_name"] = dataset_name
    serializable["dataset_root"] = str(dataset_root)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"JSON stats saved → {OUTPUT_JSON}")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    dataset_root, dataset_name = detect_dataset_root()
    records, errors, stats = explore_dataset(dataset_root, dataset_name)

    if not records:
        print("No images found. Check the dataset folder.")
    else:
        print_report(records, errors, stats, dataset_name, dataset_root)
        save_csv_index(records)
        save_json_stats(stats, dataset_name, dataset_root)
