"""Near-regular q-fold branch points: Misiurewicz points landing on the alpha fixed point, deep in the
valley at the root of the 1/q bulb, where |alpha multiplier| -> 1 so the q arms are nearly congruent
(elsewhere successive arms differ in scale by |lambda|, which makes doubled rings pair up).

For each preperiod m, Newton-solve z_m(c) = alpha(c) from seeds just outside the bulb root, keep
solutions in the valley, then find the minibrot sitting on each point and sweep frames around it.

Usage: valley.py q out_dir [--mmax 60] [--n 8]
"""
import cmath
import json
import math
import sys
from pathlib import Path

import gmpy2
from gmpy2 import mpc

import fr
import morph


def arg(name, default):
    return type(default)(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


def alpha(c):
    """Repelling fixed point with the q rays (multiplier 2*alpha near e^{2 pi i p/q})."""
    r = gmpy2.sqrt(1 - 4 * c)
    return (1 - r) / 2


def solve(c, m, bits, iters=80):
    gmpy2.get_context().precision = bits
    for _ in range(iters):
        z = mpc(0); dz = mpc(0)
        for _ in range(m):
            dz = 2 * z * dz + 1
            z = z * z + c
        a = alpha(c)
        da = 1 / gmpy2.sqrt(1 - 4 * c)
        f, df = z - a, dz - da
        if df == 0:
            return None
        step = f / df
        c = c - step
        if abs(complex(step)) < 2.0 ** (-bits * 0.8):
            return c
    return None


def main():
    q, od = arg("--q", 3) if "--q" in sys.argv else int(sys.argv[1]), Path(sys.argv[2])
    od.mkdir(parents=True, exist_ok=True)
    mmax, nwant = arg("--mmax", 60), arg("--n", 8)
    th = 2 * math.pi / q
    root = cmath.exp(1j * th) / 2 - cmath.exp(2j * th) / 4
    normal = cmath.exp(1j * th)  # outward from the cardioid at the root
    bits = 256
    gmpy2.get_context().precision = bits
    pts = []
    import numpy as np
    from scipy import ndimage
    for width in (0.06, 0.012, 0.0025):
        ctr = root
        n = 400
        xs = (np.arange(n) - (n - 1) / 2) * width / (n - 1)
        C = ctr + xs[None, :] - 1j * xs[:, None]
        A = (1 - np.sqrt(1 - 4 * C)) / 2
        dA = 1 / np.sqrt(1 - 4 * C)
        # exterior only: interior orbits near the root all drift onto alpha
        Ze = np.zeros_like(C); ext = np.zeros(C.shape, bool)
        for _ in range(3000):
            Ze = np.where(ext, Ze, Ze * Ze + C)
            ext |= np.abs(Ze) > 4
        Z = np.zeros_like(C); dZ = np.zeros_like(C)
        px = width / (n - 1)
        for m in range(1, mmax):
            dZ = 2 * Z * dZ + 1
            Z = Z * Z + C
            big = np.abs(Z) > 4
            Z[big] = 4; dZ[big] = 1
            if m < q + 2:
                continue
            E = np.abs(Z - A) / np.maximum(np.abs(dZ - dA), 1e-300)
            E[~ext] = 1e9
            lm = (E == ndimage.minimum_filter(E, size=7)) & (E < 2 * px)
            for y, x in zip(*np.nonzero(lm)):
                c = solve(mpc(complex(C[y, x])), m, bits)
                if c is None:
                    continue
                cc = complex(c)
                d = abs(cc - root)
                if abs(cc - C[y, x]) > 4 * px or d > 0.06:
                    continue
                if any(abs(cc - complex(p[0])) < 1e-12 for p in pts):
                    continue
                pts.append((c, m, abs(2 * complex(alpha(c))), d))
    # keep the valley: along the common tangent of cardioid and bulb at the root
    tang = 1j * cmath.exp(1j * th) * (1 - cmath.exp(1j * th))
    tang /= abs(tang)
    def in_valley(c):
        v = complex(c) - root
        a = abs(math.degrees(cmath.phase(v / tang)))
        return min(a, 180 - a) < 12
    pts = [p for p in pts if in_valley(p[0]) and 5e-4 < p[3] < 0.03]
    pts.sort(key=lambda p: p[3])
    print(len(pts), "valley Misiurewicz points; |lambda| range",
          f"{pts[0][2]:.4f}..{pts[-1][2]:.4f}" if pts else "-", flush=True)
    out = open(od / "candidates.jsonl", "a")
    got = 0
    for c, m, lam, d in pts:
        if got >= nwant:
            break
        scale = d * 0.05  # branch feature scale ~ a fraction of the distance to the root
        found = None
        R = scale * 1e-6
        for _ in range(9):
            per = morph.ball_period(c, R, 300000, bits)
            if per:
                N = morph.nucleus(c, per, bits + 64)
                if N is not None and abs(complex(N - c)) < 3 * R:
                    s = morph.mini_size(N, per, bits + 64)
                    if 0 < s < R:
                        found = (N, per, s)
                        break
            R *= 2
        if not found:
            print(f"m {m} |lambda| {lam:.4f}: no minibrot", flush=True)
            continue
        N, per, s = found
        got += 1
        digits = int(morph.prec_for(s * 1e-4) * 0.302) + 2
        cid = f"vq{q}_m{m}_p{per}"
        cand = dict(type="mandelbrot", cre=format(N.real, f".{digits}e"), cim=format(N.imag, f".{digits}e"),
                    period=per, mini_size=s, q=q, misiu_m=m, lam=lam, feat=scale,
                    misiu=[format(c.real, ".40e"), format(c.imag, ".40e")],
                    iters=int(min(400000, 60 * per + 20000)), julia=None, id=cid)
        out.write(json.dumps(cand) + "\n"); out.flush()
        sw = od / "sweep"; sw.mkdir(exist_ok=True)
        lo, hi = math.log(s * 10), math.log(scale * 4)
        for j in range(24):
            w = math.exp(lo + (hi - lo) * j / 23)
            fr.render(sw / f"{cid}_{j:02d}.png", palette="glade", cre=cand["cre"], cim=cand["cim"], zoom=w,
                      iters=cand["iters"], w=256, h=256, pparam=7, pparam2=24)
        print(f"m {m} |lambda| {lam:.4f} dist-to-root {d:.1e} period {per} size {s:.1e}", flush=True)


if __name__ == "__main__":
    main()
