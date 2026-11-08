"""Measure the rotational symmetry order (n-fold) of the decorations around each frame's minibrot.

For each morph candidate: Newton to its nucleus, render a small view centered on it, take the
luminance gradient in rings around the center, and find the dominant angular harmonic. Reports the
fundamental: the smallest n whose power is at least half the strongest harmonic's (a 3-fold shape
also has power at 6, 9, ...).

Usage: symmetry.py round_dir [k]   (k = view width in minibrot sizes, default 40)
Writes round_dir/symmetry.json {id: {"n": n, "strength": s}}.
"""
import json
import math
import sys
from pathlib import Path

import gmpy2
import numpy as np
from PIL import Image

import fr
import morph


def order(path):
    a = np.asarray(Image.open(path).convert("L")).astype(np.float32) / 255
    gy, gx = np.gradient(a)
    g = np.hypot(gx, gy)
    h, w = g.shape
    cy, cx = (h - 1) / 2, (w - 1) / 2
    th = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    P = np.zeros(25)
    for rf in np.linspace(0.10, 0.46, 24):
        r = rf * min(h, w)
        ys = np.clip((cy + r * np.sin(th)).astype(int), 0, h - 1)
        xs = np.clip((cx + r * np.cos(th)).astype(int), 0, w - 1)
        ring = g[ys, xs]
        if ring.std() < 1e-4:
            continue
        F = np.abs(np.fft.rfft(ring - ring.mean())) ** 2
        P += F[:25] / F[1:].sum()
    P[0] = P[1] = 0
    top = float(P.max())
    if top <= 0:
        return 0, 0.0
    for n in range(2, 25):
        if P[n] >= 0.5 * top:
            return n, round(float(P[n] / P[2:].sum()), 3)
    return 0, 0.0


def main():
    rd = Path(sys.argv[1])
    k = float(sys.argv[2]) if len(sys.argv) > 2 else 40.0
    od = rd / "sym"; od.mkdir(exist_ok=True)
    res = json.load(open(rd / "symmetry.json")) if (rd / "symmetry.json").exists() else {}
    for l in open(rd / "candidates.jsonl"):
        c = json.loads(l)
        if c["id"] in res or not c.get("period"):
            continue
        s0, p = c["mini_size"], c["period"]
        bits = morph.prec_for(s0 * 1e-3)
        gmpy2.get_context().precision = bits
        N = morph.nucleus(morph.mpc_s(c["cre"], c["cim"], bits), p, bits + 32)
        if N is None:
            continue
        digits = int(bits * 0.302) + 2
        out = od / f"{c['id']}.png"
        try:
            fr.render(out, palette="smooth", cre=format(N.real, f".{digits}e"), cim=format(N.imag, f".{digits}e"),
                      zoom=s0 * k, iters=c["iters"], w=256, h=256)
        except Exception as e:
            print("FAIL", c["id"], str(e)[:80], flush=True)
            continue
        n, st = order(out)
        res[c["id"]] = {"n": n, "strength": st, "nucleus": [format(N.real, f".{digits}e"), format(N.imag, f".{digits}e")]}
        print(c["id"], n, st, flush=True)
        json.dump(res, open(rd / "symmetry.json", "w"), indent=0)


if __name__ == "__main__":
    main()
