"""
median_filter.py
================
Removes salt & pepper noise from dermoscopy images.
"""

import cv2
import numpy as np


def remove_salt_pepper_noise(img: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Apply a median filter to remove salt & pepper noise.

    Args:
        img: BGR image
        kernel_size: size of the filter window — must be odd (3, 5, 7...) -- larger values remove bigger noise clusters
    """
    denoised = cv2.medianBlur(img, kernel_size)
    print(f"[✔] Salt & pepper noise removed (kernel={kernel_size}x{kernel_size})")
    return denoised
#