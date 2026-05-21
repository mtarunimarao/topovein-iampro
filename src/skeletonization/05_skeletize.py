import os
import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage.util import img_as_ubyte
import glob
from pathlib import Path

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
PREPROCESSED_ROOT = "results/preprocessed"   # output of 02_clahe_pipeline.py
BINARY_ROOT       = "results/binary"         # where binary images are saved
SKELETON_ROOT     = "results/skeleton"
# ─────────────────────────────────────────
# SKELETIZATION FUNCTION
# ─────────────────────────────────────────
def skeletize_image(binary_path, skeleton_root):
    binary_img = cv2.imread(binary_path, cv2.IMREAD_GRAYSCALE)
    if binary_img is None:
        print(f"Failed to read {binary_path}")
        return None

    # Convert to boolean array
    img_bool = binary_img > 0

    # Apply Zhang-Suen thinning
    skeleton = skeletonize(img_bool)

    # Convert back to uint8 for display
    skeleton_uint8 = (skeleton * 255).astype(np.uint8)

    # Save the skeleton image with the same directory structure
    relative_path = Path(binary_path).relative_to(BINARY_ROOT)
    out_path = Path(skeleton_root) / relative_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the skeleton image
    cv2.imwrite(str(out_path), skeleton_uint8)
    print(f"Saved {out_path}")

    return out_path

def run_skeletization(binary_root=BINARY_ROOT, skeleton_root=SKELETON_ROOT):
    if not os.path.exists(skeleton_root):
        os.makedirs(skeleton_root)

    pattern = os.path.join(binary_root, "**", "*_binary.png")
    binary_files = glob.glob(pattern, recursive=True)
    
    total = len(binary_files)
    print("=" * 60)
    print("  TopoVein — File 5: Skeletonization")
    print("=" * 60)
    print(f"  Images to skeletize : {total}")
    print()

    for i, binary_path in enumerate(binary_files, 1):
        if i % 20 == 0 or i == 1 or i == total:
            print(f"  [{i:>4}/{total}]  {Path(binary_path).name}")

        skeletize_image(binary_path, skeleton_root)

    print("\n  Skeletonization complete. Results saved in", skeleton_root)

if __name__ == "__main__":
    run_skeletization()