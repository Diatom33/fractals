"""Seeds for n-fold symmetric structures.

The principal Misiurewicz point of the p/q limb is where q spokes of the limb's antenna meet
(the orbit of 0 lands on the alpha fixed point after q+1 steps). Structures near it are q-fold stars; a minibrot
near it is wrapped in decorations with 2q-fold symmetry (each minibrot doubles the local
symmetry, which is why walks along ordinary 2-sided filaments gave 4- and 8-fold shapes).

Usage: polygon.py  -> prints [q, p, cre, cim, scale] per limb, standard orientation.
"""
import cmath
import json
import math
import sys

import gmpy2
from gmpy2 import mpc, mpfr

gmpy2.get_context().precision = 200


def f_iter(c, n):
    z = mpc(0); dz = mpc(0)
    for _ in range(n):
        dz = 2 * z * dz + 1
        z = z * z + c
    return z, dz


def misiurewicz(c0, q, iters=60):
    """Newton on g(c) = f^(q+2)(0) - f^(q+1)(0): the orbit lands on a fixed point after q+1 steps."""
    c = mpc(c0)
    for _ in range(iters):
        zq, dq = f_iter(c, q + 1)
        zq1 = zq * zq + c
        dq1 = 2 * zq * dq + 1
        g = zq1 - zq
        dg = dq1 - dq
        if dg == 0:
            return None
        step = g / dg
        c -= step
        if abs(complex(step)) < 1e-40:
            return c
    return None


def limb(p, q):
    t = 2 * math.pi * p / q
    root = cmath.exp(1j * t) / 2 - cmath.exp(2j * t) / 4
    r = math.sin(math.pi * p / q) / q ** 2  # approximate bulb radius
    out = root / abs(root) if abs(root) > 0 else 1
    # cardioid outward normal at the root ~ direction of d/dt of root rotated
    d = 1j * cmath.exp(1j * t) / 2 - 1j * cmath.exp(2j * t) / 2
    n = -1j * d / abs(d)
    if (n.conjugate() * out).real < 0:
        n = -n
    return root, r, n


def main():
    out = []
    for p, q in [(1, 3), (1, 4), (1, 5), (2, 5), (1, 6), (1, 7), (2, 7), (3, 7), (1, 8)]:
        root, r, n = limb(p, q)
        best = None
        for k in [1.4 + 0.15 * i for i in range(30)]:
            for side in (0.0, 0.3, -0.3):
                m = misiurewicz(root + n * r * k * cmath.exp(1j * side), q)
                if m is None:
                    continue
                # principal Misiurewicz point: f^q(0) lands on the alpha fixed point
                # (rotation number p/q, where the q spokes meet), not on beta (antenna tips)
                zq, _ = f_iter(m, q + 1)
                sq = gmpy2.sqrt(1 - 4 * m)
                alpha = (1 - sq) / 2
                if abs(complex(zq - alpha)) > 1e-20:
                    continue
                mc = complex(m)
                d = abs(mc - root)
                if d < r * 1.1 or d > r * 8:
                    continue
                if best is None or d < best[1]:
                    best = (m, d)
        if best is None:
            print("no point", p, q, file=sys.stderr)
            continue
        m = best[0]
        out.append([q, p, format(m.real, ".40e"), format(m.imag, ".40e"), r])
        print(q, p, complex(m), "bulb r", round(r, 5), file=sys.stderr)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
