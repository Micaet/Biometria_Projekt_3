import numpy as np

import os
import glob
import cv2
import matplotlib.pyplot as plt

A0 = frozenset({
    3, 6, 7, 12, 14, 15, 24, 28, 30, 31, 48, 56, 60, 62, 63, 96, 112,
    120, 124, 126, 127, 129, 131, 135, 143, 159, 191, 192, 193, 195,
    199, 207, 223, 224, 225, 227, 231, 239, 240, 241, 243, 247, 248,
    249, 251, 252, 253, 254,
})

A1 = frozenset({7, 14, 28, 56, 112, 131, 193, 224})

A2 = frozenset({
    7, 14, 15, 28, 30, 56, 60, 112, 120, 131, 135,
    193, 195, 224, 225, 240,
})

A3 = frozenset({
    7, 14, 15, 28, 30, 31, 56, 60, 62, 112, 120, 124, 131, 135, 143,
    193, 195, 199, 224, 225, 227, 240, 241, 248,
})

A4 = frozenset({
    7, 14, 15, 28, 30, 31, 56, 60, 62, 63, 112, 120, 124, 126, 131,
    135, 143, 159, 193, 195, 199, 207, 224, 225, 227, 231, 240, 241,
    243, 248, 249, 252,
})

A5 = frozenset({
    7, 14, 15, 28, 30, 31, 56, 60, 62, 63, 112, 120, 124, 126, 131,
    135, 143, 159, 191, 193, 195, 199, 207, 224, 225, 227, 231, 239,
    240, 241, 243, 248, 249, 251, 252, 254,
})

A1pix = frozenset({
    3, 6, 7, 12, 14, 15, 24, 28, 30, 31, 48, 56, 60, 62, 63, 96, 112,
    120, 124, 126, 127, 129, 131, 135, 143, 159, 191, 192, 193, 195,
    199, 207, 223, 224, 225, 227, 231, 239, 240, 241, 243, 247, 248,
    249, 251, 252, 253, 254,
})

NEIGHBOURS = (
    (-1,  0,   1),
    (-1,  1,   2),
    ( 0,  1,   4),
    ( 1,  1,   8),
    ( 1,  0,  16),
    ( 1, -1,  32),
    ( 0, -1,  64),
    (-1, -1, 128),
)

def weight(img, r, c):
    h, w = img.shape
    val = 0
    for dr, dc, bit in NEIGHBOURS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w and img[nr, nc]:
            val += bit
    return val


def k3m(img_binary, max_iter=1000):
    img = (img_binary > 0).astype(np.uint8)
    h, w = img.shape

    for _ in range(max_iter):

        border = []
        for r in range(1, h - 1):
            for c in range(1, w - 1):
                if img[r, c] == 1 and weight(img, r, c) in A0:
                    border.append((r, c))

        if not border:
            break

        modified = False

        for phase_set in (A1, A2, A3, A4, A5):
            for r, c in border:
                if img[r, c] == 0:
                    continue
                if weight(img, r, c) in phase_set:
                    img[r, c] = 0
                    modified = True

        if not modified:
            break

    changed = True
    while changed:
        changed = False
        for r in range(1, h - 1):
            for c in range(1, w - 1):
                if img[r, c] == 1 and weight(img, r, c) in A1pix:
                    img[r, c] = 0
                    changed = True

    return (img * 255).astype(np.uint8)

