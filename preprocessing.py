"""
TopoVein — Phase 1: Data Preprocessing Pipeline
================================================

"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from skimage.util import img_as_ubyte
import os

# ──────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────
INPUT_PATH   = "01.jpg"          # your NIR image
OUTPUT_DIR   = "output_stages"   # folder to save each stage
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save(name, img, cmap="gray"):
    """Save image to output folder and return it."""
    path = os.path.join(OUTPUT_DIR, name)
    cv2.imwrite(path, img)
    print(f"  Saved: {path}")
    return img


# ──────────────────────────────────────────────────────────────
# STEP 1 — Load raw NIR image
# ──────────────────────────────────────────────────────────────
def step1_load(path):
    """Load the NIR image as-is (BGR)."""
    raw = cv2.imread(path)
    if raw is None:
        raise FileNotFoundError(f"Cannot open image: {path}")
    print(f"[1] Loaded '{path}'  shape={raw.shape}  dtype={raw.dtype}")
    save("01_raw.png", raw)
    return raw


# ──────────────────────────────────────────────────────────────
# STEP 2 — Grayscale conversion
# ──────────────────────────────────────────────────────────────
def step2_grayscale(raw):
    """
    NIR images are captured in IR; colour channels carry the same
    luminance signal. Converting to grayscale collapses the 3-channel
    redundancy into a single intensity map.
    """
    gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    print(f"[2] Grayscale  shape={gray.shape}  min={gray.min()}  max={gray.max()}")
    save("02_grayscale.png", gray)
    return gray


# ──────────────────────────────────────────────────────────────
# STEP 3 — ROI Crop (isolate finger, remove black borders)
# ──────────────────────────────────────────────────────────────
def step3_roi_crop(gray):
    """
    Your image has a large black border. We threshold at a very low
    value to find the bright finger region, then crop a tight bounding
    box around it.  This prevents the background from corrupting CLAHE
    tile statistics.
    """
    # Binary mask of non-black pixels (threshold = 15 to ignore noise)
    _, mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)

    # Find bounding box of the lit region
    coords = cv2.findNonZero(mask)
    if coords is None:
        print("[3] No bright region found — returning full image")
        return gray

    x, y, w, h = cv2.boundingRect(coords)
    # Add a small padding so we don't clip the edges
    pad = 10
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(gray.shape[1] - x, w + 2 * pad)
    h = min(gray.shape[0] - y, h + 2 * pad)

    cropped = gray[y:y+h, x:x+w]
    print(f"[3] ROI crop  box=({x},{y},{w},{h})  new_shape={cropped.shape}")
    save("03_roi_crop.png", cropped)
    return cropped


# ──────────────────────────────────────────────────────────────
# STEP 4 — CLAHE (Contrast Limited Adaptive Histogram Equalization)
# ──────────────────────────────────────────────────────────────
def step4_clahe(gray_roi):
    """
    Your NIR image has:
      • Uneven illumination (bright center, dim edges)
      • Low-contrast vein shadows

    CLAHE divides the image into small tiles (8×8 pixels each) and
    equalises the histogram of each tile independently. The clipLimit
    cap prevents noise amplification in near-uniform tiles.

    Tuning guide:
      clipLimit  : 2.0–3.0  (higher → more contrast, more noise risk)
      tileGridSize: (8,8)   (smaller tiles → more local, can over-segment)
    """
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray_roi)
    print(f"[4] CLAHE applied  min={enhanced.min()}  max={enhanced.max()}")
    save("04_clahe.png", enhanced)
    return enhanced


# ──────────────────────────────────────────────────────────────
# STEP 5 — Gaussian Blur (noise suppression)
# ──────────────────────────────────────────────────────────────
def step5_denoise(enhanced):
    """
    NIR sensors introduce shot noise and speckle artefacts.
    A mild Gaussian blur smooths these before thresholding
    so random bright pixels don't become false vein branches.

    kernel (5,5) is a good starting point; increase to (7,7)
    if skeletonization produces too many spurious spurs.
    """
    blurred = cv2.GaussianBlur(enhanced, (5, 5), sigmaX=0)
    print(f"[5] Gaussian blur (5,5) applied")
    save("05_denoised.png", blurred)
    return blurred


# ──────────────────────────────────────────────────────────────
# STEP 6 — Binarization (adaptive threshold)
# ──────────────────────────────────────────────────────────────
def step6_binarize(blurred):
    """
    Veins appear DARKER than surrounding tissue in transmitted NIR light
    because deoxygenated haemoglobin absorbs 850 nm radiation.

    We use ADAPTIVE thresholding (not global Otsu) because:
      • Illumination varies across the finger
      • A global threshold would miss dim vein regions

    THRESH_BINARY_INV flips dark veins → white foreground so that
    skeletonize() treats them as the object of interest.

    blockSize (11): neighbourhood size for local threshold calculation
    C (2)         : constant subtracted from local mean — fine-tune this
                    up (more background excluded) or down (more veins kept)
    """
    binary = cv2.adaptiveThreshold(
        blurred,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=11,
        C=2
    )
    print(f"[6] Binarized  white_px={np.sum(binary == 255)}")
    save("06_binary.png", binary)
    return binary


# ──────────────────────────────────────────────────────────────
# STEP 7 — Morphological cleanup
# ──────────────────────────────────────────────────────────────
def step7_morph_clean(binary):
    """
    After binarization, the vein mask often has:
      • Isolated salt noise (single white pixels)
      • Small holes inside thick vein segments
      • Thin disconnected fragments

    Morphological Opening (erosion then dilation) removes small isolated
    white blobs (noise) without destroying the vein structure.
    Closing (dilation then erosion) fills small holes inside veins.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # Opening: removes tiny noise blobs
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    # Closing: fills small holes in thick vein regions
    cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)

    print(f"[7] Morphological cleaning done  white_px={np.sum(cleaned == 255)}")
    save("07_morph_clean.png", cleaned)
    return cleaned


# ──────────────────────────────────────────────────────────────
# STEP 8 — Skeletonization (Zhang-Suen / Lee thinning)
# ──────────────────────────────────────────────────────────────
def step8_skeletonize(cleaned):
    """
    Skeletonization reduces every vein to a 1-pixel-wide medial axis.
    This is the critical step for graph construction:
      • Each pixel in the skeleton = a potential node or edge pixel
      • Pixels with 3+ neighbours = bifurcation nodes
      • Pixels with 1  neighbour  = endpoint nodes

    skimage.morphology.skeletonize uses the Lee (1994) 3D thinning
    algorithm, which is equivalent to Zhang-Suen in 2D and is more
    robust to thick blobs.

    Input must be a boolean array (True = foreground vein).
    """
    bool_img = cleaned > 0        # convert to boolean
    skeleton = skeletonize(bool_img)
    skel_uint8 = img_as_ubyte(skeleton)   # back to 0/255 uint8

    print(f"[8] Skeleton  vein_pixels={skeleton.sum()}")
    save("08_skeleton.png", skel_uint8)
    return skel_uint8, skeleton


# ──────────────────────────────────────────────────────────────
# STEP 9 — Bifurcation detection (node candidates for Phase 2)
# ──────────────────────────────────────────────────────────────
def step9_detect_nodes(skel_uint8, skeleton):
    """
    Counts the 8-connected neighbours of every skeleton pixel.
    A pixel with ≥ 3 neighbours is a bifurcation (branch point).
    A pixel with exactly 1 neighbour is an endpoint.

    These points become the VERTICES of the graph G = (V, E)
    built in Phase 2 using NetworkX.
    """
    from scipy.ndimage import convolve

    # Kernel sums the 8-neighbourhood
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)

    # neighbour count at each skeleton pixel (non-skeleton = 0)
    neighbour_count = convolve(skeleton.astype(np.uint8), kernel, mode='constant', cval=0)
    neighbour_count = neighbour_count * skeleton  # mask to skeleton only

    bifurcations = np.argwhere(neighbour_count >= 3)   # branch points
    endpoints    = np.argwhere(neighbour_count == 1)   # terminations

    print(f"[9] Nodes detected: {len(bifurcations)} bifurcations, {len(endpoints)} endpoints")

    # Visualise nodes on skeleton (BGR for colour overlay)
    vis = cv2.cvtColor(skel_uint8, cv2.COLOR_GRAY2BGR)
    for (r, c) in bifurcations:
        cv2.circle(vis, (c, r), 3, (0, 0, 255), -1)   # red = bifurcation
    for (r, c) in endpoints:
        cv2.circle(vis, (c, r), 3, (0, 255, 0), -1)   # green = endpoint
    save("09_nodes.png", vis)

    return bifurcations, endpoints, vis


# ──────────────────────────────────────────────────────────────
# VISUALISATION — side-by-side comparison plot
# ──────────────────────────────────────────────────────────────
def visualise_all(stages: dict):
    """Plot all pipeline stages in a single figure."""
    n = len(stages)
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    for i, (title, img) in enumerate(stages.items()):
        ax = axes[i]
        if len(img.shape) == 3:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            ax.imshow(img, cmap="gray")
        ax.set_title(title, fontsize=11, pad=6)
        ax.axis("off")

    # Hide unused subplots
    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle("TopoVein — Preprocessing Pipeline (Phase 1)", fontsize=14, y=1.01)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "00_pipeline_overview.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n  Pipeline overview saved → {out_path}")
    plt.show()


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
def run_pipeline(input_path=INPUT_PATH):
    print("=" * 55)
    print("  TopoVein — Phase 1: Preprocessing Pipeline")
    print("=" * 55)

    raw         = step1_load(input_path)
    gray        = step2_grayscale(raw)
    roi         = step3_roi_crop(gray)
    enhanced    = step4_clahe(roi)
    blurred     = step5_denoise(enhanced)
    binary      = step6_binarize(blurred)
    cleaned     = step7_morph_clean(binary)
    skel_u8, sk = step8_skeletonize(cleaned)
    bifs, ends, node_vis = step9_detect_nodes(skel_u8, sk)

    print("\n✓ All stages complete. Outputs in:", OUTPUT_DIR)
    print(f"  → {len(bifs)} bifurcation nodes  (graph vertices for Phase 2)")
    print(f"  → {len(ends)} endpoint nodes")

    stages = {
        "1. Raw NIR":        cv2.cvtColor(raw, cv2.COLOR_BGR2RGB),
        "2. Grayscale":      gray,
        "3. ROI Crop":       roi,
        "4. CLAHE":          enhanced,
        "5. Gaussian Blur":  blurred,
        "6. Binary":         binary,
        "7. Morph Clean":    cleaned,
        "8. Skeleton":       skel_u8,
        "9. Node Overlay":   node_vis,
    }
    visualise_all(stages)

    return {
        "skeleton":      sk,
        "bifurcations":  bifs,
        "endpoints":     ends,
        "skeleton_img":  skel_u8,
    }


if __name__ == "__main__":
    result = run_pipeline(r"E:\topoVein-Iampro\Published_database_FV-USM_Dec2013\Published_database_FV-USM_Dec2013\1st_session\raw_data\001_1\01.jpg")