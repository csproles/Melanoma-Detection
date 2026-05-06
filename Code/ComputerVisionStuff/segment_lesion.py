"""
segment_lesion.py
=================
Isolates the skin lesion from the surrounding healthy skin.
"""

import cv2
import numpy as np


def segment_lesion(img: np.ndarray):
    """
    Produce a binary mask of the lesion.

    Returns:
        mask: binary mask — 255 = lesion pixels, 0 = background skin
        masked: original image with background blacked out
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = img.shape[:2]

    # Sample skin color from a border ring (avoids the centrally-placed lesion)
    border_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(
        border_mask,
        (int(w*0.05), int(h*0.05)),
        (int(w*0.95), int(h*0.95)),
        255,
        int(min(h, w) * 0.12)
    )
    border_pixels = lab[border_mask > 0]
    skin_color    = np.median(border_pixels, axis=0)

    # Distance from skin color in LAB space
    dist      = np.sqrt(np.sum((lab - skin_color) ** 2, axis=2))
    dist_norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask   = cv2.threshold(dist_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological cleanup — close fills holes, open removes noise blobs
    kernel = np.ones((7, 7), np.uint8)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=2)

    # Keep only the largest connected blob
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if num_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask    = np.uint8(labels == largest) * 255
    #

    # Fallback to grayscale Otsu if LAB found too little
    lesion_area = np.sum(mask > 0)
    total_area  = h * w
    if lesion_area / total_area < 0.005:
        print("[!] LAB segmentation found too little — falling back to Otsu")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        if num_labels > 1:
            largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask    = np.uint8(labels == largest) * 255
        #
        lesion_area = np.sum(mask > 0)
    #

    masked = cv2.bitwise_and(img, img, mask=mask)
    #print(f"Lesion segmented  |  Area: {lesion_area:,} px ({100*lesion_area/total_area:.1f}% of image)")
    return mask, masked
#