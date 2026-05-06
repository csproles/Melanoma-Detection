"""
asymmetry.py
============
Criterion A — measures how asymmetric the lesion shape is.

Finds the lesion's centroid using image moments, crops a square centred
exactly on that point, then flips horizontally and vertically and measures
the IoU (intersection over union) overlap with the original.

Centering on the centroid is the critical fix over naive image-centre
flipping — a round mole sitting off to one side of the image would score
as highly asymmetric if flipped around the image centre, even though the
shape itself is perfectly symmetric.

Score range: 0.0 = perfectly symmetric, 1.0 = completely asymmetric
Concern threshold: > 0.20
"""

import cv2
import numpy as np


def score_asymmetry(mask: np.ndarray) -> dict:
    """
    Compute the asymmetry score for a lesion mask.

    Args:
        mask: binary mask — 255 = lesion pixels, 0 = background

    Returns:
        dict with value, concern flag, and display label
    """
    h, w = mask.shape

    # Find centroid using image moments
    M = cv2.moments(mask)
    if M["m00"] > 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = w // 2, h // 2

    # Crop a square centred on the centroid
    coords = np.argwhere(mask > 0)
    if len(coords) > 0:
        r_min, c_min = coords.min(axis=0)
        r_max, c_max = coords.max(axis=0)
        half = max(r_max - r_min, c_max - c_min) // 2 + 10

        r0 = max(cy - half, 0); r1 = min(cy + half, h)
        c0 = max(cx - half, 0); c1 = min(cx + half, w)
        crop = (mask[r0:r1, c0:c1] // 255).astype(np.uint8)
    else:
        crop = np.zeros((1, 1), np.uint8)

    if crop.size == 0:
        asymmetry = 0.0
    else:
        # Flip horizontally — measures left/right symmetry
        flip_h    = np.fliplr(crop)
        overlap_h = np.sum(crop & flip_h) / (np.sum(crop | flip_h) + 1e-6)

        # Flip vertically — measures top/bottom symmetry
        flip_v    = np.flipud(crop)
        overlap_v = np.sum(crop & flip_v) / (np.sum(crop | flip_v) + 1e-6)

        # Average both axes: 0 = symmetric, 1 = asymmetric
        asymmetry = 1 - (overlap_h + overlap_v) / 2

    concern = asymmetry > 0.20
    print(f"[✔] Asymmetry: {asymmetry:.3f}  (centroid={cx},{cy})  {'⚠' if concern else '✔'}")

    return {
        "value":   round(asymmetry, 3),
        "concern": concern,
        "label":   f"Asymmetry score: {asymmetry:.3f} {'⚠ Irregular' if concern else '✔ Regular'}",
    }