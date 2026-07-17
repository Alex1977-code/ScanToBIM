"""Texture image encoding: JPEG for photo atlases, PNG as safe fallback.

A photographic 8192² atlas is ~180 MB as PNG but ~10-20 MB as JPEG at
quality 90 with no visible loss — glTF and browsers accept both. Pillow is
an optional dependency (the ``photos`` extra), and photo atlases can only
exist when it is installed, so the PNG fallback effectively only serves
small synthetic textures.
"""

from __future__ import annotations

import numpy as np


def encode_texture(image: np.ndarray, quality: int = 90) -> tuple[bytes, str]:
    """Encode an (H, W, 3) uint8 texture. Returns ``(bytes, mime_type)``."""
    if max(image.shape[:2]) >= 1024:
        try:
            import io

            from PIL import Image

            buf = io.BytesIO()
            Image.fromarray(np.ascontiguousarray(image)).save(
                buf, "JPEG", quality=quality, subsampling=1
            )
            return buf.getvalue(), "image/jpeg"
        except Exception:  # noqa: BLE001 — Pillow missing/failed → PNG
            pass
    from scantobim.io.png import encode_png

    return encode_png(image), "image/png"
