"""Searches inspired by Claude 3 Opus's description of its favorite Mandelbrot places.

atrium: "the faintest glimmer ... from an unremarkable cranny" that opens into "a cavernous
        atrium of scintillating fractals". A calm wide frame whose detail is concentrated in one
        small spot, paired with a zoom into that spot that turns out rich.
eye:    "rare islands of simplicity amidst the churning mathematical sea". A clean minibrot in
        a calm field, with the busy structure pushed to the periphery.
garden: "a hidden garden of recursion ... a stream, its flow following an endlessly
        self-referential pattern". A chain of minibrots running across the frame like a stream,
        with branching structure around it (rendered in canopy's pink light).

Usage: opus.py mode n out.jsonl [--seed S]
Output is in the standard orientation (screen up = +im).
"""
import json
import math
import sys
from multiprocessing import Pool

import numpy as np
from scipy import ndimage

from explore import FULL, busy_score, escape_grid, frame_metrics

GW, GH = 160, 120
DEBUG = "--debug" in sys.argv


def local_sd(S):
    L = np.log(np.nan_to_num(S, nan=np.nanmax(S) * 1.5 if np.isfinite(S).any() else 1.0) + 1)
    pad = np.pad(L, 2, mode="edge")
    win = np.stack([pad[dy:dy + L.shape[0], dx:dx + L.shape[1]] for dy in range(5) for dx in range(5)])
    return win.std(0)


def iters_for(width):
    return int(min(30000, 800 + 600 * math.log10(FULL["mandelbrot"][1] / width) ** 2))


def descend(rng, depth):
    """Random busy-weighted zoom walk (as explore.walk) down to width ~ w0*depth."""
    center, w0 = FULL["mandelbrot"]
    width = w0
    while width > w0 * depth * 1.0001:
        S = escape_grid("mandelbrot", center, width, GW, GH, iters_for(width))
        sc = busy_score(S)
        m = 18
        sc[:m, :] = sc[-m:, :] = 0; sc[:, :m] = sc[:, -m:] = 0
        p = sc.ravel() ** 2
        if p.sum() == 0:
            return None, None
        k = rng.choice(p.size, p=p / p.sum())
        yy, xx = divmod(k, GW)
        center = center + ((xx - (GW - 1) / 2) + 1j * (yy - (GH - 1) / 2)) * (width / (GW - 1))
        width = max(width / rng.uniform(3, 12), w0 * depth)
    return center, width


def px_to_c(center, width, xx, yy):
    return center + ((xx - (GW - 1) / 2) + 1j * (yy - (GH - 1) / 2)) * (width / (GW - 1))


def comps(S):
    interior = np.isnan(S)
    lab, n = ndimage.label(interior)
    if not n:
        return interior, lab, np.array([]), np.zeros((0, 2))
    sizes = np.bincount(lab.ravel())[1:]
    cents = np.array(ndimage.center_of_mass(interior, lab, range(1, n + 1)))
    return interior, lab, sizes, cents


# --- atrium -----------------------------------------------------------------

def atrium_entrance(S):
    """Calm frame with detail concentrated in one small spot. Returns (score, (x, y))."""
    interior = np.isnan(S)
    if interior.mean() > 0.15:
        return -1, None
    sd = local_sd(S)
    sd[interior] = 0
    w = sd ** 2
    m = 10
    w[:m] = w[-m:] = 0; w[:, :m] = w[:, -m:] = 0
    tot = w.sum()
    if tot <= 0:
        return -1, None
    sm = ndimage.uniform_filter(w, size=17)
    yy, xx = np.unravel_index(np.argmax(sm), sm.shape)
    Y, X = np.mgrid[:GH, :GW]
    r = np.hypot(X - xx, Y - yy)
    conc = w[r < 0.09 * GW].sum() / tot
    ext = ~interior
    calm = float((sd[ext] < np.percentile(sd[ext], 50) * 1.0 + 0.02).mean())
    return conc * 4 + calm, (xx, yy)


def atrium(seed):
    rng = np.random.default_rng(seed)
    center, width = descend(rng, 10 ** rng.uniform(-7, -2))
    if center is None:
        return None
    best = None
    for j in range(16):
        ww = width * 10 ** rng.uniform(-0.3, 0.9)
        cc = center + complex(rng.uniform(-0.5, 0.5), rng.uniform(-0.4, 0.4)) * ww
        S = escape_grid("mandelbrot", cc, ww, GW, GH, iters_for(ww))
        s, spot = atrium_entrance(S)
        if spot and (best is None or s > best[0]):
            best = (s, cc, ww, spot)
    if best is None or (best[0] < 2.0 and not DEBUG):
        return None
    s, cc, ww, (xx, yy) = best
    hot = px_to_c(cc, ww, xx, yy)
    # The glimmer: a minibrot hiding at the hot spot, far too small to read at this scale.
    import gmpy2
    import morph
    bits = morph.prec_for(ww * 1e-4)
    gmpy2.get_context().precision = bits
    T = morph.mpc_s(hot.real, hot.imag, bits)
    p = morph.ball_period(T, ww * 0.04, 30000, bits)
    if p is None:
        return None
    N = morph.nucleus(T, p, bits + 32)
    if N is None:
        return None
    size = morph.mini_size(N, p, bits + 32)
    Nc = complex(N)
    if not (ww * 1e-7 < size < ww / 300) or abs(Nc - cc) > ww * 0.6:
        return None
    frames = []
    for k in (40, 150):
        iw = size * k
        if iw < 2e-12:
            continue
        S2 = escape_grid("mandelbrot", Nc, iw, GW, GH, iters_for(iw) + 4 * p)
        met = frame_metrics(S2)
        if met.get("p99"):
            frames.append((iw, met, k))
    if not frames:
        return None
    Sw = escape_grid("mandelbrot", cc, ww, 240, 180, iters_for(ww))
    mw = frame_metrics(Sw)
    p99 = max([mw.get("p99") or 1000] + [m["p99"] for _, m, _ in frames])
    it = int(min(200000, max(2000, 6 * p99, 4 * p)))
    digits = int(bits * 0.302) + 2
    out = [dict(type="mandelbrot", cre=repr(float(cc.real)), cim=repr(float(-cc.imag)), zoom=ww, iters=it,
                julia=None, opus="atrium", role="entrance", score=round(float(s), 3), period=p, mini_size=size)]
    for iw, met, k in frames:
        out.append(dict(type="mandelbrot", cre=format(N.real, f".{digits}e"), cim=format(-N.imag, f".{digits}e"),
                        zoom=iw, iters=it, julia=None, opus="atrium", role=f"a{k}", score=round(float(s), 3),
                        period=p, mini_size=size, k=k, mag=round(ww / iw, 1), metrics=met))
    return out


# --- island -----------------------------------------------------------------

def openness(S, cy, cx, r0, r1):
    sd = local_sd(S)
    Y, X = np.mgrid[:S.shape[0], :S.shape[1]]
    r = np.hypot(X - cx, Y - cy)
    ring = (r > r0) & (r < r1)
    if ring.sum() < 30:
        return 0.0
    ext = ring & ~np.isnan(S)
    return float((sd[ext] < 0.06).sum() / ring.sum())


def island_score(S):
    interior, lab, sizes, cents = comps(S)
    if len(sizes) == 0:
        return -1
    h, w = S.shape
    cy, cx = (h - 1) / 2, (w - 1) / 2
    d = np.hypot(cents[:, 0] - cy, cents[:, 1] - cx)
    k = int(np.argmax(sizes * (d < 0.15 * w)))
    if d[k] >= 0.15 * w:
        return -1
    frac = sizes[k] / S.size
    if not (0.003 < frac < 0.05) or interior.mean() > frac * 2 + 0.01:
        return -1
    rm = math.sqrt(sizes[k] / math.pi)
    op = openness(S, cents[k][0], cents[k][1], rm + 3, rm + 0.22 * w)
    sd = local_sd(S); sd[interior] = 0
    struct = float((sd > 0.12).mean())
    return op * 4 + min(struct, 0.25) * 4


def island(seed):
    rng = np.random.default_rng(seed)
    center, width = descend(rng, 10 ** rng.uniform(-6, -1.5))
    if center is None:
        return None
    best = None
    for j in range(5):
        ww = width * 10 ** rng.uniform(0, 1.2)
        cc = center + complex(rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3)) * ww
        S = escape_grid("mandelbrot", cc, ww, GW, GH, iters_for(ww))
        interior, lab, sizes, cents = comps(S)
        for k in np.argsort(-sizes)[:6]:
            if sizes[k] < 1:
                continue
            rm = math.sqrt(sizes[k] / math.pi)
            op = openness(S, cents[k][0], cents[k][1], rm + 2, rm + 3 + 6 * max(rm, 1))
            if op < 0.4:
                continue
            mc = px_to_c(cc, ww, cents[k][1], cents[k][0])
            msz = max(math.sqrt(sizes[k]), 1.0) * ww / (GW - 1)
            for f in (6, 10, 16, 26):
                fw = msz * f
                S2 = escape_grid("mandelbrot", mc, fw, GW, GH, iters_for(fw))
                i2, _, s2, c2 = comps(S2)
                if len(s2):
                    jj = int(np.argmax(s2))
                    mc = px_to_c(mc, fw, c2[jj][1], c2[jj][0])
                    S2 = escape_grid("mandelbrot", mc, fw, GW, GH, iters_for(fw))
                sc = island_score(S2)
                if best is None or sc > best[0]:
                    best = (sc, mc, fw)
    if best is None or (best[0] < 3.2 and not DEBUG):
        return None
    sc, mc, fw = best
    S = escape_grid("mandelbrot", mc, fw, 240, 180, iters_for(fw))
    met = frame_metrics(S)
    return [dict(type="mandelbrot", cre=repr(float(mc.real)), cim=repr(float(-mc.imag)), zoom=fw,
                 iters=int(min(200000, max(2000, 6 * (met.get("p99") or 1000)))), julia=None,
                 opus="island", score=round(float(sc), 3), metrics=met)]


# --- garden -----------------------------------------------------------------

def garden_score(S):
    interior, lab, sizes, cents = comps(S)
    f_in = interior.mean()
    keep = sizes >= 2
    if keep.sum() < 6 or f_in > 0.2:
        return -1
    P = cents[keep][:, ::-1]  # x, y
    wts = np.sqrt(sizes[keep])
    mu = (P * wts[:, None]).sum(0) / wts.sum()
    Q = P - mu
    # fit y = a x^2 + b x + c in the principal frame, to allow a curving stream
    cov = (Q * wts[:, None]).T @ Q / wts.sum()
    ev, evec = np.linalg.eigh(cov)
    U = Q @ evec[:, ::-1]
    span = U[:, 0].max() - U[:, 0].min()
    if span < 0.55 * GW:
        return -1
    A = np.stack([U[:, 0] ** 2, U[:, 0], np.ones(len(U))], 1)
    coef, *_ = np.linalg.lstsq(A * wts[:, None], U[:, 1] * wts, rcond=None)
    res = U[:, 1] - A @ coef
    rms = math.sqrt(float((res ** 2 * wts).sum() / wts.sum())) / GW
    met = frame_metrics(S)
    if not met.get("p99"):
        return -1
    s = min(int(keep.sum()), 14) * 0.4 + span / GW * 2 - rms * 40
    s -= max(0, met["noise"] - 0.3) * 8 + max(0, met["flat"] - 0.3) * 4
    return s


def garden(seed):
    """Shallow frames only (canopy needs dark exterior); the stream is judged later on the canopy render."""
    rng = np.random.default_rng(seed)
    center, width = descend(rng, 10 ** rng.uniform(-3.6, -1.0))
    if center is None:
        return None
    out = []
    for j in range(3):
        ww = width * 10 ** rng.uniform(-0.2, 0.5)
        cc = center + complex(rng.uniform(-0.5, 0.5), rng.uniform(-0.4, 0.4)) * ww
        S = escape_grid("mandelbrot", cc, ww, GW, GH, iters_for(ww))
        met = frame_metrics(S)
        if not met.get("ok") or met["f_in"] > 0.2:
            continue
        out.append(dict(type="mandelbrot", cre=repr(float(cc.real)), cim=repr(float(-cc.imag)), zoom=ww,
                        iters=int(min(200000, max(2000, 6 * met["p99"]))), julia=None, opus="garden",
                        role=f"f{j}", score=0, metrics=met))
    return out


MODES = {"atrium": atrium, "island": island, "garden": garden}


def main():
    mode, n, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    seed = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 1
    fn = MODES[mode]
    found = 0
    with Pool(11) as pool, open(out, "a") as f:
        for i, rs in enumerate(pool.imap_unordered(fn, [seed * 10000 + i for i in range(n)])):
            if not rs:
                continue
            found += 1
            for j, r in enumerate(rs):
                r["id"] = f"{mode[:3]}{seed % 1000:03d}_{i:04d}" + (f"_{r['role']}" if "role" in r else "")
                f.write(json.dumps(r) + "\n")
            f.flush()
    print(mode, f"{found}/{n}", flush=True)


if __name__ == "__main__":
    main()
