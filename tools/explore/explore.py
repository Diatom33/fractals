"""Find candidate fractal views by random zoom walks toward busy boundary regions.

Usage: explore.py N out.jsonl [--types mandelbrot,burningship,...] [--seed S]
Each line: {id, type, cre, cim, zoom, iters, julia, metrics}. zoom = x-extent.
CPU numpy (float64), so depth is limited to ~1e-11 of the full view.
"""
import json
import math
import sys
import time
from multiprocessing import Pool

import numpy as np

FULL = {  # center, x-extent of the starting view
    "mandelbrot": (-0.75 + 0j, 3.5),
    "burningship": (-0.25 - 0.75j, 3.5),
    "tricorn": (-0.5 + 0j, 4.0),
    "celtic": (-0.5 + 0j, 4.0),
    "perpendicular": (-0.5 + 0j, 4.0),
    "buffalo": (-0.5 + 0j, 4.0),
    "julia": (0j, 4.0),
}


def step(kind, z, c):
    if kind in ("mandelbrot", "julia"):
        return z * z + c
    zr, zi = z.real, z.imag
    if kind == "burningship":
        zr, zi = np.abs(zr), np.abs(zi)
        return (zr * zr - zi * zi + c.real) + 1j * (2 * zr * zi + c.imag)
    if kind == "tricorn":
        return (zr * zr - zi * zi + c.real) + 1j * (-2 * zr * zi + c.imag)
    if kind == "celtic":
        return (np.abs(zr * zr - zi * zi) + c.real) + 1j * (2 * zr * zi + c.imag)
    if kind == "perpendicular":
        return (zr * zr - zi * zi + c.real) + 1j * (-2 * np.abs(zr) * zi + c.imag)
    if kind == "buffalo":
        return (np.abs(zr * zr - zi * zi) + c.real) + 1j * (-2 * np.abs(zr) * np.abs(zi) + c.imag)
    raise ValueError(kind)


def escape_grid(kind, center, width, gw, gh, max_iter, jc=None):
    """Smooth iteration grid (NaN = interior). Row 0 = top = smaller im (renderer convention)."""
    xs = (np.arange(gw) - (gw - 1) / 2) * (width / (gw - 1))
    ys = (np.arange(gh) - (gh - 1) / 2) * (width / (gw - 1))
    C = (center.real + xs[None, :]) + 1j * (center.imag + ys[:, None])
    C = C.ravel()
    if kind == "julia":
        z = C.copy(); c = np.full_like(C, jc)
    else:
        z = np.zeros_like(C); c = C
    out = np.full(C.size, np.nan)
    idx = np.arange(C.size)
    for i in range(1, max_iter + 1):
        z = step(kind, z, c)
        m2 = z.real ** 2 + z.imag ** 2
        esc = m2 > 256.0
        if esc.any():
            out[idx[esc]] = i + 1 - np.log2(0.5 * np.log2(m2[esc]))
            keep = ~esc
            z, c, idx = z[keep], c[keep], idx[keep]
            if idx.size == 0:
                break
    return out.reshape(gh, gw)


def busy_score(S):
    """Per-pixel 'interest': local spread of log-iterations, boosted next to interior."""
    L = np.log(np.nan_to_num(S, nan=np.nanmax(S) * 1.5 if np.isfinite(S).any() else 1.0) + 1)
    pad = np.pad(L, 2, mode="edge")
    win = np.stack([pad[dy:dy + L.shape[0], dx:dx + L.shape[1]] for dy in range(5) for dx in range(5)])
    sd = win.std(0)
    interior = np.isnan(S)
    pi = np.pad(interior, 2, mode="edge")
    near_in = np.stack([pi[dy:dy + L.shape[0], dx:dx + L.shape[1]] for dy in range(5) for dx in range(5)]).any(0)
    score = sd * (1 + 2 * (near_in & ~interior))
    score[interior] = 0
    return score


def frame_metrics(S):
    interior = np.isnan(S)
    f_in = float(interior.mean())
    E = S[~interior]
    if E.size < 100:
        return dict(f_in=f_in, ok=False)
    L = np.log(np.where(interior, np.nanmax(S), S) + 1)
    gy, gx = np.gradient(L)
    g = np.hypot(gx, gy)[~interior]
    flat = float((g < 0.002).mean())
    noise = float((g > 0.35).mean())
    spread = float(np.percentile(np.log(E + 1), 95) - np.percentile(np.log(E + 1), 5))
    ok = f_in < 0.5 and flat < 0.6 and noise < 0.5 and spread > 0.4
    return dict(f_in=round(f_in, 4), flat=round(flat, 3), noise=round(noise, 3),
                spread=round(spread, 3), p99=float(np.percentile(E, 99)), ok=bool(ok))


def composition(S):
    """Score a frame like the poster spot: several minibrots of different sizes,
    some black but not much, rich depth range, limited noise."""
    from scipy import ndimage
    met = frame_metrics(S)
    if met.get("p99") is None:
        return -1.0, met
    interior = np.isnan(S)
    lab, n = ndimage.label(interior)
    sizes = np.bincount(lab.ravel())[1:] if n else np.array([])
    sizes = sizes[sizes >= 2]
    k = len(sizes)
    size_range = float(np.log10(sizes.max() / sizes.min())) if k >= 2 else 0.0
    f_in = met["f_in"]
    s = 0.0
    s += min(k, 8) * 0.5 + size_range * 1.0
    s += 1.5 if 0.02 < f_in < 0.3 else (-1.0 if f_in > 0.45 else 0.0)
    s += min(met["spread"], 3.0) * 0.6
    s -= max(0.0, met["noise"] - 0.3) * 8
    s -= max(0.0, met["flat"] - 0.3) * 4
    met.update(blobs=int(k), size_range=round(size_range, 2), comp=round(s, 2))
    return s, met


def walk(args):
    seed, kind = args
    rng = np.random.default_rng(seed)
    center, w0 = FULL[kind]
    jc = None
    if kind == "julia":
        # constant near the Mandelbrot boundary: short walk on the M-set
        mc, mw = FULL["mandelbrot"]
        for _ in range(int(rng.integers(1, 4))):
            S = escape_grid("mandelbrot", mc, mw, 96, 72, 600)
            sc = busy_score(S); sc[:, :10] = sc[:, -10:] = 0; sc[:10] = sc[-10:] = 0
            p = sc.ravel() ** 2
            if p.sum() == 0:
                break
            k = rng.choice(p.size, p=p / p.sum())
            yy, xx = divmod(k, 96)
            mc = mc + ((xx - 47.5) + 1j * (yy - 35.5)) * (mw / 95)
            mw /= rng.uniform(3, 8)
        jc = complex(mc)
    depth = 10 ** rng.uniform(-10, -2) if kind != "julia" else 10 ** rng.uniform(-4, 0)
    width = w0
    gw, gh = 160, 120
    while width > w0 * depth * 1.0001:
        it = int(min(20000, 500 + 400 * math.log10(w0 / width) ** 2))
        S = escape_grid(kind, center, width, gw, gh, it, jc)
        sc = busy_score(S)
        m = 18
        sc[:m, :] = sc[-m:, :] = 0; sc[:, :m] = sc[:, -m:] = 0
        p = sc.ravel() ** 2
        if p.sum() == 0:
            return None
        k = rng.choice(p.size, p=p / p.sum())
        yy, xx = divmod(k, gw)
        center = center + ((xx - (gw - 1) / 2) + 1j * (yy - (gh - 1) / 2)) * (width / (gw - 1))
        width = max(width / rng.uniform(3, 12), w0 * depth)
    it = int(min(30000, 800 + 600 * math.log10(w0 / width) ** 2))
    if COMPOSE:
        best = None
        for j in range(14):
            if j == 0:
                cc, ww = center, width
            else:
                ww = width * 10 ** rng.uniform(-0.6, 0.6)
                cc = center + complex(rng.uniform(-0.6, 0.6), rng.uniform(-0.45, 0.45)) * ww
            S = escape_grid(kind, cc, ww, 160, 120, it, jc)
            sc, met = composition(S)
            if best is None or sc > best[0]:
                best = (sc, cc, ww, met)
        sc, center, width, met = best
        if sc < 4.0:
            return None
        S = escape_grid(kind, center, width, 240, 180, it, jc)
        met = {**frame_metrics(S), **{k: met[k] for k in ("blobs", "size_range", "comp")}}
        met["ok"] = True
    else:
        S = escape_grid(kind, center, width, 240, 180, it, jc)
        met = frame_metrics(S)
    if not met.get("ok"):
        return None
    iters = int(min(200000, max(1000, 6 * met["p99"])))
    return dict(type=kind, cre=repr(center.real), cim=repr(center.imag), zoom=width,
                iters=iters, julia=[jc.real, jc.imag] if jc is not None else None, metrics=met)


COMPOSE = "--compose" in sys.argv


def main():
    n = int(sys.argv[1]); out = sys.argv[2]
    types = "mandelbrot"
    seed = int(time.time())
    if "--types" in sys.argv:
        types = sys.argv[sys.argv.index("--types") + 1]
    if "--seed" in sys.argv:
        seed = int(sys.argv[sys.argv.index("--seed") + 1])
    kinds = types.split(",")
    jobs = [(seed * 1000 + i, kinds[i % len(kinds)]) for i in range(n)]
    found = 0
    with Pool(10) as pool, open(out, "a") as f:
        for i, r in enumerate(pool.imap_unordered(walk, jobs)):
            if r:
                found += 1
                r["id"] = f"{r['type'][:4]}{seed % 100000:05d}_{i:04d}"
                f.write(json.dumps(r) + "\n"); f.flush()
    print(f"{found}/{n} candidates passed", flush=True)


if __name__ == "__main__":
    main()
