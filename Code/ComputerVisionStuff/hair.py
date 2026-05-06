"""
hair.py
=======
Two hair operations — calibration must run before removal.

  measure_hair_width_px — measures hair width in pixels to derive
                          a real-world mm/pixel scale for the image
  remove_hair           — erases hair strands using inpainting

The calibration uses the distance transform to find each hair strand's
centreline, then measures the half-width at every centreline pixel.
The median of those measurements times 2 gives the full hair width.

Since vellus body hair averages ~70 micrometers wide:
    mm_per_pixel = 0.070 / hair_width_px
"""

import cv2
import numpy as np


def measure_hair_width_px(
    img: np.ndarray,
    kernel_size: int = 17,
    threshold: int = 10,
):
    """
    Measure hair strand width in pixels using the distance transform.

    Args:
        img:         BGR image — must still have hairs (run before remove_hair)
        kernel_size: blackhat structuring element size
        threshold:   blackhat intensity cutoff

    Returns:
        median hair width in pixels, or None if no usable hair found
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, hair_mask = cv2.threshold(blackhat, threshold, 255, cv2.THRESH_BINARY)

    if np.sum(hair_mask > 0) < 50:
        print("[!] Hair calibration: no hair detected — skipping")
        return None

    # Distance transform: each pixel = distance to nearest non-hair pixel
    # At the centreline of a hair strand this equals half the hair width
    dist = cv2.distanceTransform(hair_mask, cv2.DIST_L2, 5)

    # Skeleton = local maxima of the distance transform (the centreline)
    kernel_sk    = np.ones((3, 3), np.uint8)
    dist_dilated = cv2.dilate(dist, kernel_sk)
    skeleton     = (dist == dist_dilated) & (dist > 0)

    # Collect and filter half-width values
    half_widths = dist[skeleton]
    half_widths = half_widths[(half_widths >= 0.5) & (half_widths <= 8.0)]

    if len(half_widths) < 10:
        print(f"[!] Hair calibration: too few points ({len(half_widths)}) — skipping")
        return None

    hair_width_px = float(np.median(half_widths) * 2)
    print(f"[✔] Hair calibration: median width = {hair_width_px:.2f}px  ({len(half_widths):,} points)")
    return hair_width_px


def remove_hair(
    img: np.ndarray,
    kernel_size: int = 17,
    threshold: int = 10,
) -> np.ndarray:
    """
    Detect and remove hair strands using blackhat morphology and inpainting.

    Blackhat highlights thin dark structures (hair) smaller than the
    structuring element. Those pixels are masked and filled in using
    the Telea inpainting algorithm, which reconstructs them from the
    surrounding skin pixels.

    Args:
        img:         BGR image
        kernel_size: size of the structuring element
        threshold:   intensity cutoff for the hair mask
    """
    gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    _, hair_mask = cv2.threshold(blackhat, threshold, 255, cv2.THRESH_BINARY)
    result = cv2.inpaint(img, hair_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    print(f"[✔] Hair removal complete  |  Hair pixels masked: {np.sum(hair_mask > 0):,}")
    return result