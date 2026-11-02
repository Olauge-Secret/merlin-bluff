"""A measured image-plane direction for elongated initial drafts only.

This supplies an observation, never rotates generated geometry or changes
later rewrite instructions. Unstable/no mask and near-round objects get no hint.
"""
from __future__ import annotations
import math
import numpy as np
from sn17gen import measure_own

MIN_ELONGATION = 3.0


def axis(png: bytes) -> dict[str, float] | None:
    segmented = measure_own.photo_mask(png)
    if segmented is None:
        return None
    y, x = np.nonzero(segmented[0])
    if len(x) < 80:
        return None
    points = np.column_stack((x-x.mean(), y-y.mean()))
    values, vectors = np.linalg.eigh(points.T @ points / len(points))
    if values[-1] <= 0:
        return None
    elongation = math.sqrt(float(values[-1]) / max(float(values[0]), 1e-9))
    if elongation < MIN_ELONGATION:
        return None
    direction = vectors[:, -1]
    angle = (math.degrees(math.atan2(float(direction[1]), float(direction[0]))) + 90) % 180 - 90
    return {"clockwise_degrees": float(round(angle)), "elongation": round(elongation, 2)}


def note(png: bytes) -> str:
    measured = axis(png)
    if measured is None:
        return ""
    return (
        "\nPhoto silhouette measurement for initial construction: its long axis is approximately "
        f"{measured['clockwise_degrees']:.0f} degrees clockwise from image horizontal "
        "(0 is horizontal; either 90 direction is vertical). Preserve this projected direction "
        "in the presentation to the +Z front camera rather than making the object horizontal or upright by default. "
        "This is an image-plane direction, not a Y-axis rotation. Confirm it against the photo; perspective can affect it."
    )
