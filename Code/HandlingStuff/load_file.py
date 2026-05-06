"""
load_file.py
============
Loads an image from disk into OpenCV BGR format.

OpenCV reads images in BGR (blue, green, red) channel order rather than
the more common RGB. All other functions in the pipeline expect BGR input.
"""

import cv2
import numpy as np
import os


def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from disk and return it as a BGR numpy array.

    Args:
        image_path: full path to the image file

    Raises:
        FileNotFoundError: if the path does not exist
        ValueError:        if OpenCV cannot read the file
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    print(f"[✔] Loaded image: {image_path}  |  Size: {img.shape[1]}x{img.shape[0]}")
    return img