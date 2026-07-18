"""Scanner camera calibration files (``info/calibration.yaml`` & Co.).

SLAM scanners ship the factory calibration of ALL their cameras — the
SHARE S20 e.g. carries a 640×480 navigation camera (fisheye_middle,
fx≈548) **and** the two 3504×4672 photo cameras (fisheye_left/right,
model POLYFISHEYE, A11/A22≈1480, distortion k2..k7). Picking the wrong
entry poisons the photo projection, so cameras are parsed per block and
selected by IMAGE SIZE match against the actual photos.

The parser is deliberately tolerant and dependency-free (no PyYAML): it
extracts intrinsics from the spellings seen in the wild —

* flat keys: ``fx/fy/cx/cy`` or ``A11/A22/u0/v0`` (+ ``k2..k7``)
* OpenCV-style matrices: ``camera_matrix: {data: [fx,0,cx, 0,fy,cy, …]}``
"""

from __future__ import annotations

import re
from pathlib import Path

_NUM = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"


def _parse_block(text: str) -> dict | None:
    """Intrinsics from one text block; None when nothing camera-like."""
    out: dict = {
        "fx": None, "fy": None, "cx": None, "cy": None,
        "k1": None, "a12": None, "max_theta_deg": None,
        "width": None, "height": None,
        "model": None, "poly": None,
    }

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

    flat = {
        "fx": r"\b(?:fx|A11)\b", "fy": r"\b(?:fy|A22)\b",
        "cx": r"\b(?:cx|u0)\b", "cy": r"\b(?:cy|v0)\b",
        "k1": r"\bk1\b", "a12": r"\bA12\b",
        "max_theta_deg": r"\bmaxIncidentAngle\b",
        "width": r"\b(?:image_)?width\b",
        "height": r"\b(?:image_)?height\b",
    }
    for key, pattern in flat.items():
        if out[key] is not None:
            continue
        m = re.search(pattern + r"\s*[:=]\s*(" + _NUM + ")", text, re.IGNORECASE)
        if m:
            out[key] = float(m.group(1))

    mm = re.search(r"(?:camera_)?model(?:_type)?\s*[:=]\s*([A-Za-z_]+)",
                   text, re.IGNORECASE)
    if mm:
        out["model"] = mm.group(1).upper()
    poly = []
    for i in range(2, 10):
        km = re.search(rf"\bk{i}\b\s*[:=]\s*({_NUM})", text, re.IGNORECASE)
        if km:
            poly.append(float(km.group(1)))
        else:
            break
    if poly:
        out["poly"] = poly
        if out["model"] is None:
            out["model"] = "POLYFISHEYE"

    fx = out["fx"]
    if fx is None or not (50.0 < fx < 50_000.0):
        return None
    if out["fy"] is None:
        out["fy"] = fx
    if out["width"] is not None:
        out["width"] = int(out["width"])
    if out["height"] is not None:
        out["height"] = int(out["height"])
    return out


def read_camera_calibration(path: str | Path) -> list[dict] | None:
    """All cameras found in a calibration file, in file order.

    Each entry: ``{"name", "fx", "fy", "cx", "cy", "k1", "width",
    "height", "model", "poly"}`` (missing → None). None when nothing
    camera-like is found at all.
    """
    try:
        text = Path(path).read_text(errors="replace")
    except OSError:
        return None
    if len(text) > 2_000_000:
        return None

    # Segment at camera-ish section names; fall back to one whole-file block.
    markers = list(re.finditer(
        r"(?m)^\s*\"?([A-Za-z_][\w]*(?:cam|fisheye|left|right|middle|rgb)[\w]*)\"?\s*:",
        text, re.IGNORECASE,
    ))
    blocks: list[tuple[str | None, str]] = []
    if markers:
        for i, m in enumerate(markers):
            end = markers[i + 1].start() if i + 1 < len(markers) else len(text)
            blocks.append((m.group(1), text[m.start():end]))
    if not blocks:
        blocks = [(None, text)]

    cameras = []
    for name, block in blocks:
        cam = _parse_block(block)
        if cam is not None:
            cam["name"] = name
            cameras.append(cam)
    if not cameras:
        cam = _parse_block(text)
        if cam is not None:
            cam["name"] = None
            cameras.append(cam)
    return cameras or None


def pick_calibration(
    calibration, width: int, height: int
) -> dict | None:
    """The calibration entry matching the actual photo size — or None.

    Guards against grabbing the navigation camera (640×480, fx≈548) for
    the 3504×4672 photo cameras: without a size match (±2%, portrait/
    landscape tolerant) NO calibration is returned and the caller falls
    back to full self-calibration.
    """
    if calibration is None:
        return None
    cams = calibration if isinstance(calibration, list) else [calibration]

    def _matches(cam: dict) -> bool:
        w, h = cam.get("width"), cam.get("height")
        if not w or not h:
            return False
        for cw, ch in ((w, h), (h, w)):
            if abs(cw - width) <= 0.02 * width and abs(ch - height) <= 0.02 * height:
                return True
        return False

    sized = [c for c in cams if _matches(c)]
    if sized:
        return sized[0]
    return _pick_unsized(cams)


def pick_calibrations(calibration, width: int, height: int) -> list[dict]:
    """ALL size-matching calibration entries (e.g. fisheye_left AND _right).

    Lets stereo rigs use each photo's own camera: images under ``left/``
    get fisheye_left's intrinsics, ``right/`` fisheye_right's.
    """
    single = pick_calibration(calibration, width, height)
    if single is None:
        return []
    cams = calibration if isinstance(calibration, list) else [calibration]
    sized = []
    for c in cams:
        w, h = c.get("width"), c.get("height")
        if not w or not h:
            continue
        for cw, ch in ((w, h), (h, w)):
            if abs(cw - width) <= 0.02 * width and abs(ch - height) <= 0.02 * height:
                sized.append(c)
                break
    return sized or [single]


def calibration_for_name(sized: list[dict], name: str | None) -> dict:
    """The entry whose camera name matches the photo path (left/right/…)."""
    low = (name or "").lower()
    for c in sized:
        cname = (c.get("name") or "").lower()
        for side in ("left", "right", "middle"):
            if side in cname and side in low:
                return c
    return sized[0]


def _pick_unsized(cams: list[dict]) -> dict | None:
    # No sizes recorded anywhere → a single entry may still be right;
    # multiple entries without sizes are ambiguous → refuse.
    unsized = [c for c in cams if not c.get("width")]
    if len(cams) == 1 and len(unsized) == 1:
        return unsized[0]
    return None

