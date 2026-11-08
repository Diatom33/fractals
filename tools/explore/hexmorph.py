"""Hexagon hunt: a minibrot sitting exactly on a q-arm branch point of a parent minibrot's decorated
zone should get a 2q-fold ring (q = 3 -> hexagon) between its own halo and the inherited rings.

1. Render the parent's zone, skeletonize the filaments, pick q-arm branch nodes.
2. Refine each node by repeated zoom + re-detection (8x per step) so the target sits on the
   branch point to ~1e-5 of the zone width.
3. Find the lowest-period minibrot within the final precision, and render it at widths
   spanning its size up to the branch point's feature scale.

Usage: hexmorph.py cre cim period out_dir [--q 3] [--n 6] [--width-k 200] [--seed 0]
       period 0 = plain-set region of width --width-k centred at cre, cim (no inherited rings)
Writes out_dir/candidates.jsonl and out_dir/<id>_w<j>.png (smooth, for symmetry measurement).
"""
import json
import math
import sys
from pathlib import Path

import gmpy2
import numpy as np
from gmpy2 import mpc
from scipy import ndimage
from skimage.morphology import skeletonize

import fr
import morph
from branchmorph import arms_at


def arg(name, default):
    return type(default)(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


def filament_mask(D):
    """D = distance estimate in pixels (0 = interior). Filaments: within ~0.7 px of the set."""
    m = D < 0.7
    return ndimage.binary_closing(m, iterations=1)


def de_grid(center, ref, Z, width, n, maxiter):
    """Distance estimate (in pixels) per pixel; perturbation with rebasing, dz/dc tracked on the full orbit."""
    step = width / (n - 1)
    d0 = complex(center - ref)
    xs = (np.arange(n) - (n - 1) / 2) * step
    dc = (d0 + xs[None, :] - 1j * xs[:, None]).ravel().astype(np.complex128)
    out = np.zeros(dc.size)
    dz = np.zeros_like(dc); der = np.zeros_like(dc)
    r = np.zeros(dc.size, np.int64)
    idx = np.arange(dc.size)
    L = len(Z)
    for i in range(1, maxiter + 1):
        zf_old = Z[r] + dz
        der = 2 * zf_old * der + 1
        dz = 2 * Z[r] * dz + dz * dz + dc
        r = r + 1
        zf = Z[r] + dz
        m2 = zf.real ** 2 + zf.imag ** 2
        esc = m2 > 1e8
        if esc.any():
            a = np.sqrt(m2[esc])
            out[idx[esc]] = a * np.log(a) / np.abs(der[esc]) / step
            keep = ~esc
            dz, der, r, zf, m2, dc, idx = dz[keep], der[keep], r[keep], zf[keep], m2[keep], dc[keep], idx[keep]
            if idx.size == 0:
                break
        reb = (m2 < dz.real ** 2 + dz.imag ** 2) | (r >= L - 1)
        if reb.any():
            dz[reb] = zf[reb]
            r[reb] = 0
    return out.reshape(n, n)


def nodes(mask, q, margin=0.15):
    gh, gw = mask.shape
    sk = skeletonize(mask)
    out = []
    rs = [gh * f for f in (0.015, 0.022, 0.03, 0.037)]
    for y, x in zip(*np.nonzero(sk)):
        if not (gh * margin < y < gh * (1 - margin) and gw * margin < x < gw * (1 - margin)):
            continue
        a = [arms_at(mask, y, x, r) for r in rs]
        if a.count(q) >= 3:
            out.append((y, x))
    return out


def grid(center, ref, Z, width, n, it):
    return de_grid(center, ref, Z, width, n, it)


def refine(T, w, q, ref, Z, it, steps=6, n=160):
    """Zoom on the branch node nearest the center, 8x per step."""
    for _ in range(steps):
        S = grid(T, ref, Z, w, n, it)
        P = nodes(filament_mask(S), q, margin=0.25)
        if not P:
            return None, w
        c = (n - 1) / 2
        y, x = min(P, key=lambda p: (p[0] - c) ** 2 + (p[1] - c) ** 2)
        if np.hypot(y - c, x - c) > n * 0.2:
            return None, w
        step = w / (n - 1)
        T = T + mpc(complex((x - c) * step, -(y - c) * step))
        w /= 8
    return T, w


def main():
    cre, cim, p0, od = sys.argv[1], sys.argv[2], int(sys.argv[3]), Path(sys.argv[4])
    od.mkdir(parents=True, exist_ok=True)
    q, nwant, wk = arg("--q", 3), arg("--n", 6), arg("--width-k", 200.0)
    rng = np.random.default_rng(arg("--seed", 0))
    bits = morph.prec_for(1e-30)
    gmpy2.get_context().precision = bits
    if p0 > 0:
        N0 = morph.nucleus(morph.mpc_s(cre, cim, bits), p0, bits)
        s0 = morph.mini_size(N0, p0, bits)
        W = s0 * wk
    else:
        # plain-set region: period 0, --width-k is the region width itself
        N0 = morph.mpc_s(cre, cim, bits)
        W = wk
    bits = morph.prec_for(W * 1e-14)
    gmpy2.get_context().precision = bits
    it = int(min(80000, 40 * p0 + 4000))
    Z = morph.ref_orbit(N0, it)
    S = grid(N0, N0, Z, W, 384, it)
    P = nodes(filament_mask(S), q)
    P = [p for p in P if np.hypot(p[0] - 191.5, p[1] - 191.5) > 384 * 0.1]
    rng.shuffle(P)
    print(len(P), "branch nodes with", q, "arms", flush=True)
    step = W / 383
    out = open(od / "candidates.jsonl", "a")
    got, seen = 0, set()
    for y, x in P[:60]:
        if got >= nwant:
            break
        T0 = N0 + mpc(complex((x - 191.5) * step, -(y - 191.5) * step))
        feat = step * 6  # branch feature scale in the zone render
        T, wfin = refine(T0, feat * 4, q, N0, Z, it)
        if T is None:
            print("  refine lost the node", flush=True)
            continue
        found = None
        R = wfin * 0.5
        for _ in range(12):
            per = morph.ball_period(T, R, 300000, bits)
            if per and per > p0:
                N = morph.nucleus(T, per, bits + 32)
                if N is not None and abs(complex(N - T)) < 3 * R:
                    s = morph.mini_size(N, per, bits + 32)
                    if 0 < s < R:
                        found = (N, per, s, abs(complex(N - T)))
                        break
            R *= 2
        if not found:
            print("  no minibrot near the refined node", flush=True)
            continue
        N, per, s, dist = found
        if per in seen:
            continue
        seen.add(per)
        got += 1
        digits = int(morph.prec_for(s * 1e-4) * 0.302) + 2
        c = dict(type="mandelbrot", cre=format(N.real, f".{digits}e"), cim=format(N.imag, f".{digits}e"),
                 period=per, mini_size=s, q=q, parent=p0, feat=feat, dist_over_feat=dist / feat,
                 iters=int(min(400000, 60 * per + 20000)), julia=None, id=f"hx{p0 or 'r'}_q{q}_p{per}_{x}_{y}")
        # The 2q-fold ring (spiral clusters around the minibrot) shows ~60% of the way up the log
        # range from 10x its size to the branch feature scale (6-fold star below, the plain branch above).
        lo, hi = math.log(s * 10), math.log(feat * 2)
        c["fracs"] = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75]
        c["widths"] = [math.exp(lo + (hi - lo) * f) for f in c["fracs"]]
        for j, w in enumerate(c["widths"]):
            fr.render(od / f"{c['id']}_f{j}.png", palette="glade", cre=c["cre"], cim=c["cim"], zoom=w,
                      iters=c["iters"], w=512, h=384, pparam=7, pparam2=24)
        out.write(json.dumps(c) + "\n"); out.flush()
        print(f"q{q} period {per} dist/feat {dist / feat:.1e}", flush=True)


if __name__ == "__main__":
    main()
