"""
vignette_remover.py
===================
Removes the dark circular border that surrounds dermoscopy images.

Dermoscope lenses produce a circular lit window surrounded by black.
If left in, that black border gets mistaken for the lesion during
segmentation and produces wildly incorrect results.

The detected circle center and radius are returned alongside the cleaned
image because they get reused downstream:
  - color.py    uses the circle to define the skin color sampling ring
  - diameter.py uses it as a reference size for the lesion
"""

import cv2
import numpy as np


def remove_vignette(img: np.ndarray, shrink: float = 0.95):
    """
    Detect and remove the dark circular vignette border.

    Method:
      1. Threshold grayscale image to find all non-black pixels
      2. Find the largest contour — the lit dermoscope window
      3. Fit a minimum enclosing circle to that contour
      4. Build a circular mask shrunk slightly inward to trim the
         discolored ring just inside the border
      5. Fill everything outside the mask with the average skin color
         sampled from inside the circle

    Args:
        img:    BGR image
        shrink: fraction of detected radius to use (0.95 clips the
                slightly darker ring just inside the vignette edge)

    Returns:
        result:      cleaned image with border filled
        circle_info: (cx, cy, radius) or None if no circle found
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bright = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("[!] Vignette removal: no circle found, skipping")
        return img, None

    largest = max(contours, key=cv2.contourArea)
    (cx, cy), radius = cv2.minEnclosingCircle(largest)

    # Skip if the detected circle is too small — likely not a real vignette
    img_area    = img.shape[0] * img.shape[1]
    circle_area = np.pi * radius ** 2
    if circle_area / img_area < 0.3:
        print("[!] Vignette removal: detected circle too small, skipping")
        return img, None

    circle_mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.circle(circle_mask, (int(cx), int(cy)), int(radius * shrink), 255, -1)

    skin_color = cv2.mean(img, mask=circle_mask)[:3]
    result     = img.copy()
    result[circle_mask == 0] = [int(c) for c in skin_color]

    print(f"[✔] Vignette removed  |  Circle center=({cx:.0f},{cy:.0f}), radius={radius:.0f}px")
    return result, (int(cx), int(cy), int(radius))