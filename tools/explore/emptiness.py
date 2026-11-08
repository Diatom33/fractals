"""Detail coverage: reject frames that are mostly empty with detail peeking in from an edge.

coverage(img) splits the image into a 12x9 grid and counts tiles with real local detail
(luminance gradient energy). Also reports how much of the detail sits in the central region.
Usage as a script: emptiness.py img1.jpg img2.jpg ...  (prints coverage, center share)
"""
import sys

import numpy as np
from PIL import Image

GX, GY = 12, 9


def coverage(path):
    im = Image.open(path).convert("L").resize((480, 360), Image.LANCZOS)
    L = np.asarray(im).astype(np.float32) / 255
    th, tw = L.shape[0] // GY, L.shape[1] // GX
    T = L[:th * GY, :tw * GX].reshape(GY, th, GX, tw)
    E = T.std(axis=(1, 3))  # within-tile variation: catches soft shapes as well as fine detail
    busy = E > 0.035
    cov = float(busy.mean())
    inner = busy[2:-2, 3:-3]
    center = float(inner.mean())
    return cov, center


def is_empty(path, max_cov=0.35, min_center=0.2):
    """Calibrated on rated gallery images (Nov 2026): the rejected 'entrance' frames (mostly
    empty, detail peeking in at an edge) all scored cov <= 0.20 with center <= 0.10; nothing
    rated like/love fell below center 0.27 at that coverage."""
    cov, center = coverage(path)
    return cov < max_cov and center < min_center, cov, center


if __name__ == "__main__":
    for p in sys.argv[1:]:
        e, c, m = is_empty(p)
        print(f"{'EMPTY' if e else 'ok   '} cov={c:.2f} center={m:.2f} {p}")
