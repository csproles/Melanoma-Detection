"""
border.py
=========
Criterion B — measures how irregular the lesion border is.

Uses the circularity index: (4π × area) / perimeter²

A perfect circle scores 1.0. As the border becomes more jagged and
irregular the perimeter grows much faster than the area, pushing the
circularity score toward 0.

Border irregularity = 1 − circularity
So 0.0 = perfectly round border, 1.0 = maximally irregular border.

Concern threshold: > 0.50
"""

import cv2
import numpy as np


def score_border(mask: np.ndarray) -> dict:
    """
    Compute the border irregularity score for a lesion mask.

    Args:
        mask: binary mask — 255 = lesion pixels, 0 = background

    Returns:
        dict with value, concern flag, and display label
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if contours:
        contour     = max(contours, key=cv2.contourArea)
        perimeter   = cv2.arcLength(contour, True)
        area        = cv2.contourArea(contour)
        circularity = (4 * np.pi * area) / (perimeter ** 2 + 1e-6)
        border_irreg = 1 - circularity
    else:
        border_irreg = 0.0

    concern = border_irreg > 0.50
    print(f"[✔] Border irregularity: {border_irreg:.3f}  {'⚠' if concern else '✔'}")

    return {
        "value":   round(border_irreg, 3),
        "concern": concern,
        "label":   f"Border irregularity: {border_irreg:.3f} {'⚠ Irregular' if concern else '✔ Regular'}",
    }