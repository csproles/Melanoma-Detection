"""
score.py
========
Combines the individual ABCD criterion scores into a final risk level.

Scoring rules:
  - HIGH if 2 or more criteria flag as concerning
  - LOW  if 0 or 1 criteria flag

Critical color override:
  If any single dangerous color (pink/red, blue-gray, white, black)
  covers more than 50% of the lesion, the risk is forced HIGH regardless
  of how A, B, and D scored. A lesion that is 50%+ a dangerous color is
  clinically alarming on its own and does not need corroboration.
"""

import numpy as np

from MelanomaDeterminingStuff.asymmetry import score_asymmetry
from MelanomaDeterminingStuff.border    import score_border
from MelanomaDeterminingStuff.color     import score_color
from MelanomaDeterminingStuff.diameter  import score_diameter


def analyze_abcde(
    mask:        np.ndarray,
    original:    np.ndarray,
    circle_info: tuple = None,
    mm_per_px:   float = None,
) -> dict:
    """
    Run all ABCD criteria and return a combined risk assessment.

    Args:
        mask:        binary lesion mask (255 = lesion, 0 = background)
        original:    original BGR image (after preprocessing, before inpaint)
        circle_info: (cx, cy, radius) from vignette removal, or None
        mm_per_px:   real-world scale from hair calibration, or None

    Returns:
        dict with keys A_asymmetry, B_border, C_color, D_diameter,
        and _summary containing the overall risk level
    """
    results = {}

    # Score each criterion independently
    results["A_asymmetry"] = score_asymmetry(mask)
    results["B_border"]    = score_border(mask)

    color_result, pink_red_px, blue_gray_px, white_px, black_px = score_color(
        mask, original, circle_info
    )
    results["C_color"]    = color_result
    results["D_diameter"] = score_diameter(mask, mm_per_px)

    # Count how many criteria are flagged
    concerns = sum(1 for v in results.values() if v["concern"])

    # Critical override — force HIGH if any dangerous color dominates
    critical_color    = max(pink_red_px, blue_gray_px, white_px, black_px)
    critical_override = critical_color > 0.50

    if critical_override:
        risk_level = "HIGH"
        print(f"[!] Critical color override triggered ({critical_color:.0%} dominant dangerous color)")
    else:
        risk_level = "LOW" if concerns <= 1 else "HIGH"

    n_scored = "4" if results["D_diameter"]["value"] is not None else "3"
    results["_summary"] = {
        "concerns":   concerns,
        "risk_level": risk_level,
        "label":      f"Overall Risk: {risk_level}  ({concerns}/{n_scored} criteria flagged)",
    }

    return results