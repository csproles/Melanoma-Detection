"""
diameter.py
===========
Criterion D — estimates the real-world lesion diameter in millimeters.

Uses the hair width calibration scale (mm_per_px) derived in hair.py.
Since vellus body hair averages ~70 micrometers wide, measuring hair
width in pixels gives a real-world scale without needing an external
ruler or device specification.

Finds the minimum enclosing circle of the lesion contour and converts
its diameter from pixels to millimeters using that scale.

The threshold is 10mm rather than the clinical 6mm because the 70µm
hair assumption tends to overestimate slightly — validated against the
test dataset.

Skipped (concern=False, value=None) if no hair was detected in the image.
"""

import cv2
import numpy as np


def score_diameter(
    mask:      np.ndarray,
    mm_per_px: float = None,
) -> dict:
    """
    Estimate the lesion diameter in millimeters using hair calibration.

    Args:
        mask:      binary lesion mask
        mm_per_px: scale factor from hair width calibration, or None

    Returns:
        dict with value (mm), concern flag, and display label
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if contours and mm_per_px is not None:
        contour = max(contours, key=cv2.contourArea)
        (_, _), radius_px = cv2.minEnclosingCircle(contour)
        diameter_mm = radius_px * 2 * mm_per_px
        concern     = diameter_mm > 10.0

        print(f"[✔] Diameter: {diameter_mm:.1f}mm  (scale={mm_per_px*1000:.3f}µm/px)  {'⚠' if concern else '✔'}")

        return {
            "value":   round(diameter_mm, 2),
            "concern": concern,
            "label":   f"Est. diameter: {diameter_mm:.1f}mm (hair-calibrated) {'⚠ >10mm' if concern else '✔ ≤10mm'}",
        }
    else:
        reason = "no hair detected" if mm_per_px is None else "no lesion contour"
        print(f"[!] Diameter: skipped ({reason})")
        return {
            "value":   None,
            "concern": False,
            "label":   f"Diameter: N/A ({reason})",
        }