"""
color.py
========
Criterion C — measures color variation and detects dangerous colors.

Two checks run together:

Check 1 — Color spread (CV):
  Pixelates the image to kill fine texture, samples the skin baseline
  color from the dermoscope ring, computes each lesion pixel's LAB
  distance from that baseline, then measures the coefficient of variation
  (std / mean) of those distances. A benign mole has tightly clustered
  distances (low CV). Melanoma with mixed colors has a wide spread.

Check 2 — Dangerous colors:
  Looks for clinically alarming colors relative to the patient's own
  skin tone. All thresholds are derived from the sampled skin baseline
  so they automatically adapt to different skin tones.

  Pink/red  — vascularization (new blood vessel growth by the tumor)
  Blue-gray — melanin migrated deep into the dermis
  White     — regression zones (immune system attacking the lesion)
  Black     — dense melanin, aggressive pigmentation

C flags if EITHER check triggers.
Critical override: if any dangerous color covers >50% of the lesion,
the overall risk is forced HIGH regardless of A, B, and D.

Concern threshold (CV): > 0.35
Concern threshold (dangerous color): > 8% of lesion pixels
"""

import cv2
import numpy as np


# ── Helpers ───────────────────────────────────────────────────────────────────

def sample_skin_color(
    img:         np.ndarray,
    circle_info: tuple,
    lesion_mask: np.ndarray,
    inner_frac:  float = 0.60,
    outer_frac:  float = 0.80,
) -> np.ndarray:
    """
    Sample the baseline skin color from a clean ring inside the dermoscope circle.

    The ring sits between inner_frac and outer_frac of the detected radius:
      - outer_frac (0.80) keeps us clear of the discolored vignette edge
      - inner_frac (0.60) keeps us away from the lesion

    Lesion pixels are excluded from sampling. Falls back to the full ring
    if too few clean pixels remain.

    Returns the median LAB color of the sampled pixels.
    """
    cx, cy, radius = circle_info
    h, w = img.shape[:2]

    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

    ring_mask = (
        (dist >= radius * inner_frac) &
        (dist <= radius * outer_frac)
    ).astype(np.uint8) * 255

    ring_mask[lesion_mask > 0] = 0

    n_pixels = int(np.sum(ring_mask > 0))
    if n_pixels < 100:
        ring_mask = (
            (dist >= radius * inner_frac) &
            (dist <= radius * outer_frac)
        ).astype(np.uint8) * 255
        print("[!] Skin sampling: few clean pixels, using full ring")

    lab         = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    skin_pixels = lab[ring_mask > 0]
    skin_color  = np.median(skin_pixels, axis=0)

    # Sanity check — if L is extremely low the ring has drifted into the
    # black vignette border rather than actual skin. Fall back to corners.
    if skin_color[0] < 60:
        print("[!] Skin sampling: baseline too dark (L={:.0f}) — falling back to corners".format(skin_color[0]))
        h_img, w_img = img.shape[:2]
        cs = min(h_img, w_img) // 8
        corners = np.vstack([
            lab[:cs,      :cs     ].reshape(-1, 3),
            lab[:cs,      -cs:    ].reshape(-1, 3),
            lab[-cs:,     :cs     ].reshape(-1, 3),
            lab[-cs:,     -cs:    ].reshape(-1, 3),
        ])
        # Filter out very dark corner pixels (part of the vignette border)
        corners = corners[corners[:, 0] > 60]
        if len(corners) > 10:
            skin_color = np.median(corners, axis=0)
        else:
            # Last resort: sample the brightest 20% of the ring
            bright_pixels = skin_pixels[skin_pixels[:, 0] > np.percentile(skin_pixels[:, 0], 80)]
            if len(bright_pixels) > 10:
                skin_color = np.median(bright_pixels, axis=0)

    print(f"[✔] Skin color sampled  |  {n_pixels:,} pixels  |  LAB=({skin_color[0]:.1f}, {skin_color[1]:.1f}, {skin_color[2]:.1f})")
    return skin_color


def pixelate(img: np.ndarray, block_size: int = 12) -> np.ndarray:
    """
    Block-average the image to remove fine texture noise.

    Each block_size × block_size region becomes a single flat color —
    more aggressive than Gaussian blur for removing pores and skin lines
    because no detail smaller than block_size survives at all.
    """
    h, w  = img.shape[:2]
    small = cv2.resize(img, (w // block_size, h // block_size), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


# ── Main scoring function ──────────────────────────────────────────────────────

def score_color(
    mask:        np.ndarray,
    original:    np.ndarray,
    circle_info: tuple = None,
) -> tuple[dict, float, float, float, float]:
    """
    Score criterion C and return dangerous color pixel fractions.

    Returns:
        result:          dict with value, concern flag, and display label
        pink_red_pixels: fraction of lesion that is pink/red
        blue_gray_pixels:fraction of lesion that is blue-gray
        white_pixels:    fraction of lesion that is white
        black_pixels:    fraction of lesion that is black
    """
    # Step 1: pixelate to kill texture
    pixelated = pixelate(original, block_size=12)

    # Step 2: get skin baseline
    if circle_info is not None:
        skin_lab = sample_skin_color(original, circle_info, mask)
    else:
        lab_img = cv2.cvtColor(original, cv2.COLOR_BGR2LAB).astype(np.float32)
        h_img, w_img = original.shape[:2]
        cs = min(h_img, w_img) // 8
        corners = np.vstack([
            lab_img[:cs, :cs].reshape(-1, 3),
            lab_img[:cs, -cs:].reshape(-1, 3),
            lab_img[-cs:, :cs].reshape(-1, 3),
            lab_img[-cs:, -cs:].reshape(-1, 3),
        ])
        skin_lab = np.median(corners, axis=0)
        print("[!] No circle info — using corners for skin baseline")

    # Step 3: per-pixel LAB distance from skin (using block_size=12 pixelation)
    pix_lab    = cv2.cvtColor(pixelated, cv2.COLOR_BGR2LAB).astype(np.float32)
    lesion_lab = pix_lab[mask > 0]
    distances  = np.sqrt(np.sum((lesion_lab - skin_lab) ** 2, axis=1))

    # Step 4: coefficient of variation (uses heavily pixelated image)
    color_cv   = float(np.std(distances)) / (float(np.mean(distances)) + 1e-6)
    cv_concern = color_cv > 0.35

    # Step 5: dangerous color detection
    # Uses block_size=4 to preserve concentrated dark patches.
    # All thresholds use BOTH absolute (skin-relative) and relative
    # (lesion-internal) checks because dermoscopy images under polarized
    # oil immersion light have darker-looking tissue that is much lighter
    # in LAB space than expected — fixed offsets from skin alone miss them.
    pix_small     = pixelate(original, block_size=4)
    pix_small_lab = cv2.cvtColor(pix_small, cv2.COLOR_BGR2LAB).astype(np.float32)
    lesion_lab_dc = pix_small_lab[mask > 0]

    skin_L, skin_A, skin_B = skin_lab

    L_ch = lesion_lab_dc[:, 0]
    A_ch = lesion_lab_dc[:, 1]
    B_ch = lesion_lab_dc[:, 2]

    lesion_L_median = float(np.median(L_ch))
    lesion_L_p10    = float(np.percentile(L_ch, 10))
    lesion_B_p10    = float(np.percentile(B_ch, 10))

    # Pink/red: redder AND light enough to not just be dark brown
    pink_red_pixels = float(np.mean(
        (A_ch > skin_A + 10) & (L_ch > skin_L - 30)
    ))

    # Blue-gray: absolute check OR relative check
    # Relative: darkest 10% of lesion are substantially bluer than lesion median
    blue_gray_abs    = float(np.mean((L_ch < skin_L - 60) & (B_ch < skin_B - 15)))
    blue_gray_rel    = (lesion_L_p10 < lesion_L_median - 20 and
                        lesion_B_p10 < float(np.median(B_ch)) - 8)
    blue_gray_pixels = max(blue_gray_abs, 0.15 if blue_gray_rel else 0.0)

    # White: substantially brighter than skin
    white_pixels = float(np.mean(L_ch > skin_L + 75))

    # Black/dark: absolute check OR relative check
    # Relative: darkest 10% are substantially darker than lesion median
    black_abs    = float(np.mean(L_ch < skin_L - 130))
    black_rel    = lesion_L_p10 < lesion_L_median - 35
    black_pixels = max(black_abs, 0.12 if black_rel else 0.0)

    print(f"[✔] Skin baseline: L={skin_L:.0f} A={skin_A:.0f} B={skin_B:.0f}")
    print(f"[✔] Lesion L: median={lesion_L_median:.0f}  p10={lesion_L_p10:.0f}  B_p10={lesion_B_p10:.0f}")

    danger_threshold = 0.08
    dangerous_colors = []
    if pink_red_pixels  > danger_threshold: dangerous_colors.append(f"pink/red({pink_red_pixels:.0%})")
    if blue_gray_pixels > danger_threshold: dangerous_colors.append(f"blue-gray({blue_gray_pixels:.0%})")
    if white_pixels     > danger_threshold: dangerous_colors.append(f"white({white_pixels:.0%})")
    if black_pixels     > danger_threshold: dangerous_colors.append(f"black({black_pixels:.0%})")

    danger_concern = len(dangerous_colors) > 0
    c_concern      = cv_concern or danger_concern

    detail = []
    if cv_concern:     detail.append(f"CV={color_cv:.3f}")
    if danger_concern: detail.append(", ".join(dangerous_colors))
    detail_str = " | ".join(detail) if detail else "clean"

    print(f"[✔] Color: CV={color_cv:.3f} {'⚠' if cv_concern else '✔'}  |  Dangerous: {dangerous_colors or 'none'}")

    result = {
        "value":   round(color_cv, 3),
        "concern": c_concern,
        "label":   f"Color: {detail_str} {'⚠ Concerning' if c_concern else '✔ Normal'}",
    }

    return result, pink_red_pixels, blue_gray_pixels, white_pixels, black_pixels