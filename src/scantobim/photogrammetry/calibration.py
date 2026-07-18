"""Scanner camera calibration files (``info/calibration.yaml`` & Co.).

SLAM scanners ship the factory calibration of their cameras — real focal
length, principal point, distortion. Until now that file sat unused while
the focal length was self-calibrated from photo colors. Reading it gives
the photo projection exact intrinsics, which sharpens the photo texture.

The parser is deliberately tolerant and dependency-free (no PyYAML): it
extracts intrinsics from the two spellings seen in the wild —

* flat keys::

      fx: 1234.5
      fy: 1236.1
      cx: 2027.3
      cy: 1519.8

* OpenCV-style matrices::

      camera_matrix:
        rows: 3
        cols: 3
        data: [1234.5, 0., 2027.3, 0., 1236.1, 1519.8, 0., 0., 1.]
      distortion_coefficients:
        data: [-0.04, 0.01, 0., 0., 0.]
"""

from __future__ import annotations

import re
from pathlib import Path

_NUM = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"


def read_camera_calibration(path: str | Path) -> dict | None:
    """Extract camera intrinsics from a calibration file.

    Returns ``{"fx", "fy", "cx", "cy", "k1", "width", "height"}`` (missing
    entries are None) or None when nothing camera-like is found.
    """
    try:
        text = Path(path).read_text(errors="replace")
    except OSError:
        return None
    if len(text) > 2_000_000:  # calibration files are small
        return None

    out: dict = {
        "fx": None, "fy": None, "cx": None, "cy": None,
        "k1": None, "width": None, "height": None,
    }

    # OpenCV-style: first camera_matrix data block with ≥9 numbers.
    m = re.search(r"camera_matrix.{0,200}?data\s*:\s*\[([^\]]+)\]",
                  text, re.DOTALL | re.IGNORECASE)
    if m:
        nums = [float(x) for x in re.findall(_NUM, m.group(1))]
        if len(nums) >= 9:
            out["fx"], out["cx"] = nums[0], nums[2]
            out["fy"], out["cy"] = nums[4], nums[5]
    d = re.search(r"distortion[_a-z]*.{0,200}?data\s*:\s*\[([^\]]+)\]",
                  text, re.DOTALL | re.IGNORECASE)
    if d:
        nums = [float(x) for x in re.findall(_NUM, d.group(1))]
        if nums:
            out["k1"] = nums[0]

    # Flat keys fill whatever the matrix form did not provide.
    for key, pattern in (
        ("fx", r"\bfx\b"), ("fy", r"\bfy\b"),
        ("cx", r"\bcx\b"), ("cy", r"\bcy\b"),
        ("k1", r"\bk1\b"),
        ("width", r"\b(?:image_)?width\b"),
        ("height", r"\b(?:image_)?height\b"),
    ):
        if out[key] is not None:
            continue
        m = re.search(pattern + r"\s*[:=]\s*(" + _NUM + ")", text, re.IGNORECASE)
        if m:
            out[key] = float(m.group(1))

    fx = out["fx"]
    if fx is None or not (50.0 < fx < 50_000.0):  # sanity: plausible pixels
        return None
    if out["fy"] is None:
        out["fy"] = fx
    if out["width"] is not None:
        out["width"] = int(out["width"])
    if out["height"] is not None:
        out["height"] = int(out["height"])
    return out
