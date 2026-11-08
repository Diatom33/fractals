"""Nested-minibrot walks in z^d + c, to test the symmetry-multiplication rule.

In z^2 + c each nested minibrot doubles the rotational symmetry of its surroundings (the squares
and octagons). Near a minibrot of z^d + c the local map is z^d, so each nesting level should
multiply the symmetry by d instead: d = 3 -> triangles then 9-gons, d = 7 -> 7-gons.

Uses the GPU renderer's Multibrot type (no perturbation, so moderate depths only) for views and
gmpy2 for the minibrot search: ball period with dz = d z^(d-1) dz + 1, Newton to the nucleus,
size from the isolated black component in a render. Each level must have a higher period than
the last and lie inside the previous minibrot's decorated zone, as in morph.py.

Usage: multimorph.py d walks out_dir [--seed S]
Writes out_dir/d<d>.json and centered renders out_dir/d<d>_w<walk>_l<level>_k<k>.png.
"""
import json
import sys
from pathlib import Path

import gmpy2
import numpy as np
from gmpy2 import mpc, mpfr
from PIL import Image
from scipy import ndimage

import fr
from symmetry import order

D = int(sys.argv[1])
BITS = 160
gmpy2.get_context().precision = BITS


def fmt(x):
    return format(x, ".40e")


def render(path, c, w, iters, n=256, palette="smooth"):
    # n must be a multiple of 64: the renderer pads widths up to one (align_width)
    fr.render(path, type="multibrot", power=D, palette=palette, cre=fmt(c.real), cim=fmt(c.imag), zoom=w,
              iters=iters, w=n, h=n)
    return np.asarray(Image.open(path).convert("RGB"))


def ball_period(c, R, maxit):
    z = mpc(0); dz = mpc(0)
    for n in range(1, maxit):
        dz = D * z ** (D - 1) * dz + 1
        z = z ** D + c
        az = abs(complex(z))
        if az > 1e6:
            return None
        if az < R * abs(complex(dz)):
            return n
    return None


def nucleus(c, p, it=80):
    for _ in range(it):
        z = mpc(0); dz = mpc(0)
        for _ in range(p):
            dz = D * z ** (D - 1) * dz + 1
            z = z ** D + c
        if dz == 0:
            return None
        step = z / dz
        c = c - step
        if abs(complex(step)) < 1e-35:
            return c
    return c


def mini_size(N, guess, iters, tmp):
    """Span of the isolated black component at the center, searching scales upward."""
    for i in range(20):
        w = guess * 1e-4 * 3 ** i
        a = render(tmp, N, w, iters, n=128)
        black = a.max(2) == 0
        if not black[64, 64]:
            continue
        lab, _ = ndimage.label(black)
        comp = lab == lab[64, 64]
        ys, xs = np.nonzero(comp)
        if min(ys.min(), xs.min()) == 0 or max(ys.max(), xs.max()) == 127:
            continue
        span = (max(xs.max() - xs.min(), ys.max() - ys.min()) + 1) * w / 127
        if span < w * 0.4:
            return span
    return None


def busy_pick(rng, tmp, center, width, iters):
    a = render(tmp, center, float(width), iters, n=128)
    g = np.asarray(Image.open(tmp).convert("L")).astype(float)
    sc = np.hypot(*np.gradient(g))
    # stay well away from interior: the lowest-period components next to black regions are
    # bulbs attached to bigger bodies, not isolated minibrots
    sc[ndimage.binary_dilation(a.max(2) == 0, iterations=6)] = 0
    sc[:10] = sc[-10:] = 0; sc[:, :10] = sc[:, -10:] = 0
    p = sc.ravel() ** 2
    if not p.sum() > 0:
        return None
    k = rng.choice(p.size, p=p / p.sum()); yy, xx = divmod(k, 128)
    return center + mpc(complex((xx - 63.5) * float(width) / 127, -(yy - 63.5) * float(width) / 127))


def main():
    walks = int(sys.argv[2]); od = Path(sys.argv[3]); od.mkdir(parents=True, exist_ok=True)
    seed = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 1
    rng = np.random.default_rng(seed)
    tmp = od / "tmp.png"
    rows = []
    for wk in range(walks):
        center, width = mpc(0), mpfr(2.6)
        for _ in range(3):   # descend to a busy boundary view first
            t = busy_pick(rng, tmp, center, width, 400)
            if t is None:
                break
            center, width = t, width / 6
        last_p = 0
        last_s = None
        for lvl in range(1, 4):
            iters = 800 + 1500 * lvl
            found = None
            for _attempt in range(14 if D <= 3 else 30):
                T = busy_pick(rng, tmp, center, width, iters)
                if T is None:
                    break
                per = ball_period(T, float(width) * rng.uniform(0.01, 0.06), 6000)
                if not per or per <= last_p:
                    continue
                N = nucleus(T, per)
                if N is None or abs(complex(N - T)) > float(width) * 0.3:
                    continue
                s = mini_size(N, float(width) * 1e-3, iters + 30 * per, tmp)
                if not s or not (float(width) * 1e-7 < s < float(width) * 0.05) or (last_s and s > last_s * (0.05 if D <= 3 else 0.3)):
                    continue
                found = (N, per, s)
                break
            if not found:
                print(wk, lvl, "no minibrot", flush=True)
                break
            N, per, s = found
            orders = []
            for kk in (8, 25, 80, 250):
                img = od / f"d{D}_w{wk:03d}_l{lvl}_k{kk}.png"
                render(img, N, s * kk, iters + 30 * per)
                orders.append(order(img)[0])
            rows.append(dict(walk=wk, level=lvl, period=per, size=s, orders=orders, cre=fmt(N.real), cim=fmt(N.imag)))
            print(f"walk {wk} level {lvl} period {per} size {s:.1e} n(k=8,25,80,250)={orders}", flush=True)
            last_p = per
            last_s = s
            # high degrees keep their decoration close to the minibrot
            center, width = N, mpfr(s * (rng.uniform(150, 500) if D <= 3 else rng.uniform(25, 120)))
            if s * 300 < 1e-28:
                break
    json.dump(rows, open(od / f"d{D}.json", "w"), indent=0)


if __name__ == "__main__":
    main()
