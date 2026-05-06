"""
segment_edges.py
================
Detects the precise border of the lesion using Canny edge detection.

Canny works in four stages:
  1. Gaussian smoothing — reduces noise before gradient calculation
  2. Sobel operators    — compute gradient strength and direction
  3. Non-maximum suppression — thins edges to single-pixel width
  4. Hysteresis thresholding — keeps strong edges and any weak edges
     connected to them, discards isolated weak edges

The result is a clean single-pixel outline of the lesion boundary,
used in border.py to compute the border irregularity score.
"""

import cv2
import numpy as np


def detect_edges(
    img: np.ndarray,
    low_threshold: int = 50,
    high_threshold: int = 150,
) -> np.ndarray:
    """
    Apply Canny edge detection to find the lesion border.

    Args:
        img: BGR image (the segmented lesion)
        low_threshold:  edges below this strength are discarded
        high_threshold: edges above this strength are always kept
                        edges between the two thresholds are kept only if connected to a strong edge
    """
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    #print(f"Canny edge detection done (low={low_threshold}, high={high_threshold})")
    return edges
#