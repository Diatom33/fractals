"""Find the zoom where each hexmorph candidate's 2q-fold ring is strongest.

Sweeps the log range [10 x size, 2 x branch scale] finely at 256 px, scores each frame by the share
of angular power at harmonic 2q (rings at 15-45% of the frame), keeps the best two fractions and
renders 768x576 glade previews there.

Usage: hexpick.py hex_dir [--steps 28]
Writes hex_dir/picks.json [{id, frac, width, score}] and hex_dir/pick_<id>_<j>.png.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

import fr


def harmonic_share(path, n):
    a = np.asarray(Image.open(path).convert("L")).astype(np.float32) / 255
    gy, gx = np.gradient(a)
    g = np.hypot(gx, gy)
    h, w = g.shape
    cy, cx = (h - 1) / 2, (w - 1) / 2
    th = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    P = np.zeros(37)
    for rf in np.linspace(0.15, 0.45, 16):
        r = rf * min(h, w)
        ring = g[np.clip((cy + r * np.sin(th)).astype(int), 0, h - 1), np.clip((cx + r * np.cos(th)).astype(int), 0, w - 1)]
        if ring.std() < 1e-4:
            continue
        F = np.abs(np.fft.rfft(ring - ring.mean())) ** 2
        P += F[:37] / F[1:].sum()
    tot = P[1:].sum()
    return float(P[n] / tot) if tot > 0 else 0.0


def main():
    d = Path(sys.argv[1])
    steps = int(sys.argv[sys.argv.index("--steps") + 1]) if "--steps" in sys.argv else 28
    tmp = d / "sweep"; tmp.mkdir(exist_ok=True)
    picks = []
    for line in open(d / "candidates.jsonl"):
        c = json.loads(line)
        n = 2 * c.get("q", 3)
        lo, hi = math.log(c["mini_size"] * 10), math.log(c["feat"] * 2)
        scores = []
        for j in range(steps):
            f = 0.42 + 0.4 * j / (steps - 1)
            w = math.exp(lo + (hi - lo) * f)
            p = tmp / f"{c['id']}_{j:02d}.png"
            if not p.exists():
                fr.render(p, palette="glade", cre=c["cre"], cim=c["cim"], zoom=w, iters=c["iters"], w=256, h=256,
                          pparam=7, pparam2=24)
            scores.append((harmonic_share(p, n), f, w))
        best = []
        for sc in sorted(scores, reverse=True):
            if all(abs(sc[1] - b[1]) > 0.04 for b in best):
                best.append(sc)
            if len(best) == 2:
                break
        for k, (sc, f, w) in enumerate(best):
            fr.render(d / f"pick_{c['id']}_{k}.png", palette="glade", cre=c["cre"], cim=c["cim"], zoom=w,
                      iters=c["iters"], w=768, h=576, pparam=7, pparam2=24)
            picks.append(dict(id=c["id"], k=k, frac=f, width=w, score=round(sc, 3)))
        print(c["id"], [(round(f, 3), round(sc, 3)) for sc, f, _ in best], flush=True)
    json.dump(picks, open(d / "picks.json", "w"), indent=0)


if __name__ == "__main__":
    main()
