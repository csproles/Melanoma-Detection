"""
bilateral_filter.py
===================
Smooths skin texture while keeping lesion edges sharp.

The bilateral filter considers both spatial closeness AND color similarity
when blending pixels. Nearby pixels with similar colors get blended
together, smoothing out fine skin texture like pores and skin grain.
But pixels across a color boundary (like the lesion edge) are too
different in color to blend — so the edge stays sharp.

This is what makes it ideal for skin lesion imaging compared to a
standard Gaussian blur, which would smooth everything including the
edges we need for border analysis.
"""

import cv2
import numpy as np


def apply_bilateral_filter(
    img: np.ndarray,
    diameter: int = 9,
    sigma_color: float = 75,
    sigma_space: float = 75,
) -> np.ndarray:
    """
    Apply bilateral filter to smooth skin texture while preserving edges.

    Args:
        img:         BGR image
        diameter:    pixel neighborhood diameter
        sigma_color: larger = more colors blended together
        sigma_space: larger = farther pixels influence each other
    """
    filtered = cv2.bilateralFilter(img, diameter, sigma_color, sigma_space)
    print(f"[✔] Bilateral filter applied (d={diameter}, σ_color={sigma_color}, σ_space={sigma_space})")
    return filtered