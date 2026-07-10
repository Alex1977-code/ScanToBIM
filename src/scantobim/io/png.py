"""Minimal dependency-free PNG encoder (RGB8, no interlacing).

Used to embed baked textures into GLB/HTML exports and to write OBJ/MTL
texture files without pulling in an imaging library.
"""

from __future__ import annotations

import struct
import zlib

import numpy as np


def encode_png(image: np.ndarray) -> bytes:
    """Encode an ``(H, W, 3)`` uint8 array as a PNG byte string."""
    img = np.ascontiguousarray(image, dtype=np.uint8)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"expected (H, W, 3) uint8 image, got {img.shape}")
    h, w = img.shape[:2]

    # Filter type 0 (None) per scanline.
    raw = np.empty((h, 1 + w * 3), dtype=np.uint8)
    raw[:, 0] = 0
    raw[:, 1:] = img.reshape(h, w * 3)
    compressed = zlib.compress(raw.tobytes(), 6)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)  # 8-bit RGB
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", compressed)
        + chunk(b"IEND", b"")
    )
