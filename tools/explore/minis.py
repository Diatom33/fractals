"""Find framings around good frames that show many visible minibrots.

For each seed frame, render small GPU previews of ~16 framings (wider zooms,
offsets), find interior components (exact-black pixels: fully interior with
the smooth palette at 1 sample), and score: several minibrots, a spread of
sizes, each big enough to read as a minibrot, little black overall.

Usage: minis.py out_dir seeds.json   (seeds: [{id, cre, cim, zoom, iters}...])
Writes out_dir/candidates.jsonl (best framings, standard orientation).
"""
import json
import math
import sys
from pathlib import Path

import gmpy2
import numpy as np
from gmpy2 import mpc, mpfr
from PIL import Image
from scipy import ndimage

import fr

W, H = 240, 180
MIN_PX = 5  # component area at preview scale; ~170 px^2 at 1600 wide


def score(path):
    a = np.asarray(Image.open(path).convert("RGB"))
    black = a.max(2) == 0
    frac = float(black.mean())
    lab, n = ndimage.label(black)
    sizes = np.bincount(lab.ravel())[1:] if n else np.array([])
    # ignore components touching the border that are huge (main-body edges)
    vis = sizes[sizes >= MIN_PX]
    k = len(vis)
    rng = float(np.log10(vis.max() / vis.min())) if k >= 2 else 0.0
    s = min(k, 14) * 1.0 + rng * 2.0
    if frac > 0.30:
        s -= (frac - 0.30) * 40
    if frac < 0.004:
        s -= 3
    return s, dict(minis=int(k), size_range=round(rng, 2), black=round(frac, 3))


def main():
    od = Path(sys.argv[1]); od.mkdir(parents=True, exist_ok=True)
    seeds = json.load(open(sys.argv[2]))
    pv = od / "preview"; pv.mkdir(exist_ok=True)
    rng = np.random.default_rng(3)
    with open(od / "candidates.jsonl", "a") as f:
        for sd in seeds:
            bits = int(max(128, -math.log2(sd["zoom"]) * 2.2 + 96))
            gmpy2.get_context().precision = bits
            base = mpc(mpfr(sd["cre"], bits), mpfr(sd["cim"], bits))
            digits = int(bits * 0.302) + 2
            # Morph frames are offset from their minibrot: recover the nucleus
            # (Newton at the stored period) and frame relative to its size.
            size = None
            if sd.get("period"):
                import morph
                N = morph.nucleus(base, sd["period"], bits + 64)
                if N is not None:
                    s = morph.mini_size(N, sd["period"], bits + 64)
                    if s and abs(math.log10(s / sd.get("mini_size", s))) < 0.5:
                        base, size = N, s
            if size is None:
                print(sd["id"], "no nucleus", flush=True)
                continue
            trials = []
            for j in range(16):
                k = [0.6, 1.0, 1.5, 2.3][j % 4] * rng.uniform(0.85, 1.2)
                w = size * k * 4
                off = complex(rng.choice([-1, 1]) * rng.uniform(0.15, 0.45), rng.uniform(-0.3, 0.3)) * w if j >= 2 else 0j
                c = base + mpc(off)
                cand = dict(sd, id=f"{sd['id']}_m{j:02d}", cre=format(c.real, f".{digits}e"),
                            cim=format(c.imag, f".{digits}e"), zoom=w, minis_of=sd["id"], type="mandelbrot")
                p = pv / f"{cand['id']}.png"
                try:
                    if not p.exists():
                        fr.render(p, palette="smooth", cre=cand["cre"], cim=cand["cim"], zoom=w,
                                  iters=sd["iters"], w=W, h=H)
                    s, m = score(p)
                except Exception as e:
                    print("FAIL", cand["id"], str(e)[:120], flush=True)
                    continue
                cand["metrics"] = m
                trials.append((s, cand))
            trials.sort(key=lambda t: -t[0])
            kept = [c for s, c in trials[:2] if c["metrics"]["minis"] >= 4]
            for c in kept:
                f.write(json.dumps(c) + "\n")
            f.flush()
            best = trials[0][1]["metrics"] if trials else None
            print(sd["id"], "kept", len(kept), "best", best, flush=True)


if __name__ == "__main__":
    main()
