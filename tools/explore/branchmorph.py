"""Test: ring symmetry around a deep minibrot = 2 x (filament arms meeting at the point zoomed toward).

In the decorated zone around a parent minibrot, find filament points by arm count (1 = tip,
2 = plain filament, 3+ = branch point), aim at each, find the nearest deeper minibrot, and
measure the rotational order of its surroundings at several widths.

Usage: branchmorph.py cre cim period out_dir [--width-k 200] [--per-class 4] [--seed 0]
Writes out_dir/candidates.jsonl (with "arms") and prints arms -> measured n.
"""
import json
import sys
from pathlib import Path

import gmpy2
import numpy as np
from gmpy2 import mpc
from scipy import ndimage
from skimage.morphology import skeletonize

import fr
import morph
from symmetry import order

GW, GH = 384, 288


def arg(name, default):
    return type(default)(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


def arms_at(mask, y, x, r):
    th = np.linspace(0, 2 * np.pi, 96, endpoint=False)
    ys = np.clip(np.round(y + r * np.sin(th)).astype(int), 0, mask.shape[0] - 1)
    xs = np.clip(np.round(x + r * np.cos(th)).astype(int), 0, mask.shape[1] - 1)
    ring = mask[ys, xs].astype(int)
    if ring.all():
        return 0
    return int(((ring - np.roll(ring, 1)) == 1).sum())


def misiurewicz(T, bits, step, maxn=600):
    """Snap a filament pixel to the preperiodic (Misiurewicz) point its orbit shadows:
    find (m, n) with z_{m+n} ~ z_m before escape, then Newton on z_{m+n}(c) - z_m(c)."""
    gmpy2.get_context().precision = bits
    z = mpc(0); orb = [0j]
    for _ in range(200000):
        z = z * z + T
        zc = complex(z)
        if abs(zc) > 4:
            break
        orb.append(zc)
    o = np.array(orb)
    E = len(o)
    best = None
    for n in range(1, min(maxn, E // 2)):
        d = np.abs(o[n:] - o[:-n]) / (np.abs(o[:-n]) + 0.05)
        m = int(np.argmin(d[1:])) + 1
        if best is None or d[m] < best[0] * 0.5:
            best = (d[m], m, n)
    if best is None:
        return None
    _, m, n = best
    c = T
    for _ in range(60):
        z = mpc(0); dz = mpc(0)
        zm = dzm = None
        for k in range(1, m + n + 1):
            dz = 2 * z * dz + 1
            z = z * z + c
            if k == m:
                zm, dzm = z, dz
        f, df = z - zm, dz - dzm
        if df == 0:
            return None
        dc = f / df
        c = c - dc
        if abs(complex(dc)) < step * 1e-9:
            break
    if abs(complex(c - T)) > 3 * step:
        return None
    return c, m, n


def main():
    cre, cim, p0, od = sys.argv[1], sys.argv[2], int(sys.argv[3]), Path(sys.argv[4])
    od.mkdir(parents=True, exist_ok=True)
    wk, per_class, rng = arg("--width-k", 200.0), arg("--per-class", 4), np.random.default_rng(arg("--seed", 0))
    bits = morph.prec_for(1e-30)
    gmpy2.get_context().precision = bits
    N0 = morph.nucleus(morph.mpc_s(cre, cim, bits), p0, bits)
    s0 = morph.mini_size(N0, p0, bits)
    W = s0 * wk
    bits = morph.prec_for(W * 1e-12)
    gmpy2.get_context().precision = bits
    it = int(min(80000, 40 * p0 + 4000))
    Z = morph.ref_orbit(N0, it)
    S = morph.perturb_grid(N0, N0, Z, W, GW, GH, it)
    L = np.log(np.where(np.isnan(S), np.nanmax(S) * 1.5, S) + 1)
    mask = (L - ndimage.gaussian_filter(L, 6)) > 0.04
    mask = ndimage.binary_opening(mask, iterations=1)
    sk = skeletonize(mask)
    np.save(od / "mask.npy", mask)
    pts = {2: [], 3: [], 4: []}
    ys, xs = np.nonzero(sk)
    for y, x in zip(ys, xs):
        if not (GH * 0.15 < y < GH * 0.85 and GW * 0.15 < x < GW * 0.85):
            continue
        if np.hypot(y - (GH - 1) / 2, x - (GW - 1) / 2) < GH * 0.12:
            continue
        a = [arms_at(mask, y, x, r) for r in (4, 6, 8, 10)]
        m = max(set(a), key=a.count)
        if a.count(m) >= 3 and m in pts:
            pts[m].append((y, x))
    step = W / (GW - 1)
    cands = []
    for arms, P in pts.items():
        rng.shuffle(P)
        got = 0
        for y, x in P:
            if got >= per_class:
                break
            T = N0 + mpc(complex((x - (GW - 1) / 2) * step, -(y - (GH - 1) / 2) * step))
            ms = misiurewicz(T, bits, step)
            if ms is None:
                continue
            T, mm, nn = ms
            found = None
            R = step * 0.002
            for _ in range(10):
                q = morph.ball_period(T, R, 200000, bits)
                if q and q > p0:
                    N = morph.nucleus(T, q, bits + 32)
                    if N is not None and abs(complex(N - T)) < 3 * R:
                        s = morph.mini_size(N, q, bits + 32)
                        if W * 1e-12 < s < step:
                            found = (N, q, s, abs(complex(N - T)))
                            break
                R *= 2
            if not found:
                continue
            N, q, s, dist = found
            got += 1
            digits = int(morph.prec_for(s * 1e-4) * 0.302) + 2
            c = dict(type="mandelbrot", cre=format(N.real, f".{digits}e"), cim=format(N.imag, f".{digits}e"),
                     period=q, mini_size=s, arms=arms, dist_px=dist / step, misiu=[mm, nn], parent=p0,
                     iters=int(min(400000, 60 * q + 20000)), julia=None,
                     id=f"bm{p0}_a{arms}_p{q}_{x}_{y}")
            # The target's surroundings at radius rho reappear around the deeper minibrot near
            # the geometric mean of its size and rho (pullback through the critical point).
            rho = 10 * step
            ns = []
            for a in (0.3, 0.45, 0.6, 0.75):
                w = s ** (1 - a) * rho ** a
                out = od / f"{c['id']}_a{int(a * 100)}.png"
                fr.render(out, palette="smooth", cre=c["cre"], cim=c["cim"], zoom=w, iters=c["iters"], w=256, h=256)
                ns.append(order(out)[0])
            c["n"] = ns
            cands.append(c)
            print(f"arms {arms} M{mm},{nn} period {q} dist {dist / step:.2f}px size {s / step:.1e}px  n(a=.3,.45,.6,.75) = {ns}", flush=True)
    with open(od / "candidates.jsonl", "w") as f:
        for c in cands:
            f.write(json.dumps(c) + "\n")


if __name__ == "__main__":
    main()
