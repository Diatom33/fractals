"""Julia-morphing walks: pass through successive minibrots so each level's
decorations inherit the region they came from (seahorse-ish antennae,
evolving trees).

Arbitrary depth: reference orbits in gmpy2, per-pixel perturbation in float64
with Zhuoran rebasing. Minibrots: ball-method period detection at a target
point, Newton for the nucleus, Heiland-Allen size estimate.

Coordinates are STANDARD orientation (row 0 = top = +im), matching the
renderer since commit c9905fc.

Usage: morph.py N out.jsonl [--seed S] [--levels L]
"""
import json
import math
import sys
import time
from multiprocessing import Pool

import gmpy2
import numpy as np
from gmpy2 import mpc, mpfr

# Starting regions: (re, im, width, label, target mode). Width ~ region scale.
SEEDS = [
    ("-0.7436447860", "0.1318252536", 2e-3, "seahorse", "busy"),
    ("-0.7453", "0.1127", 4e-3, "seahorse", "busy"),
    ("0.2925", "0.0149", 4e-3, "elephant", "busy"),
    ("0.2550", "0.0005", 2e-3, "elephant", "busy"),
    ("-0.0883", "0.6545", 4e-3, "triple-spiral", "busy"),
    ("-1.7700", "0.0020", 6e-3, "antenna-trees", "ridge"),
    ("-1.9400", "0.0010", 4e-3, "antenna-trees", "ridge"),
    ("-1.6250", "0.0030", 6e-3, "antenna-trees", "ridge"),
    ("-0.1560", "1.0320", 4e-3, "top-antenna-trees", "ridge"),
    ("-1.25066", "0.02012", 2e-3, "period-2-bulb", "busy"),
    ("0.3750", "0.2160", 4e-3, "east-valley", "busy"),
    ("-0.5600", "0.6400", 4e-3, "north-spirals", "busy"),
    # Spiral Canopy, level 1 (round-12 loves): walk the zone around its minibrot
    ("-1.5397164778102671669261739145e-01", "1.0306485936161868994812085884665e+00", 1.8e-7, "spiral-canopy", "ridge"),
    # q-spoke junctions (principal Misiurewicz points of the p/q limbs, polygon.py): minibrots
    # near them carry 2q-fold decorations (triangle limb -> hexagons), the junction itself q-fold
    ("-1.0109636384562216102578544573862256546381e-01", "9.5628651080914150077109605772997743580983e-01", 5.774e-03, "poly3-1", "busy"),
    ("3.6636298342276434132893273079784905804985e-01", "5.9153377326144522757749349371654198115355e-01", 2.652e-03, "poly4-1", "busy"),
    ("4.3792424135946277735156981251927540730828e-01", "3.4189208433811610960124343625073791201972e-01", 1.411e-03, "poly5-1", "busy"),
    ("-5.6220262152303721274496959626296192623234e-01", "6.4281714907277524889962465662783094147300e-01", 2.283e-03, "poly5-2", "busy"),
    ("4.2451271905003964244247221417189023743288e-01", "2.0753022816674530250607348224391828017800e-01", 8.333e-04, "poly6-1", "busy"),
    ("3.9739182229654117498429436868857991699248e-01", "1.3351120487187763265769951707176891661615e-01", 5.313e-04, "poly7-1", "busy"),
    ("1.4355621452983444961420213158120240119546e-01", "6.5207264731080062833687427941092822585755e-01", 9.573e-04, "poly7-2", "busy"),
    ("-6.7020918790325372409934023384598640090189e-01", "4.5806097529694587290921367610631399623824e-01", 1.194e-03, "poly7-3", "busy"),
    ("3.7213770544957648814439927939265822078817e-01", "9.0398233158172876680286484175130248354997e-02", 3.588e-04, "poly8-1", "busy"),
]
# The user's loved/liked spots from rounds 1-2, im already negated into the
# standard orientation (favorites.json, built from the gallery ratings).
try:
    import os as _os
    FAVORITES = [tuple(x) for x in json.load(open(_os.path.join(_os.path.dirname(__file__), "favorites.json")))]
except FileNotFoundError:
    FAVORITES = []


KS = (25, 80, 250)
ONLY = ""  # --only <substring>: restrict seeds to regions whose label contains it


def prec_for(width):
    return int(max(128, -math.log2(width) * 2.2 + 96))


def mpc_s(re, im, bits):
    return mpc(mpfr(re, bits), mpfr(im, bits))


def ref_orbit(c, maxiter):
    z = mpc(0)
    out = np.empty(maxiter + 2, np.complex128)
    out[0] = 0
    n = 1
    for _ in range(maxiter):
        z = z * z + c
        zc = complex(z)
        out[n] = zc
        n += 1
        if zc.real * zc.real + zc.imag * zc.imag > 1e10:
            break
    return out[:n]


def perturb_grid(center, ref, Z, width, gw, gh, maxiter):
    """Smooth iteration counts (NaN = interior). Pixel deltas are float64 offsets from ref."""
    step = width / (gw - 1)
    d0 = complex(center - ref)
    xs = (np.arange(gw) - (gw - 1) / 2) * step
    ys = -(np.arange(gh) - (gh - 1) / 2) * step
    dc = (d0 + xs[None, :] + 1j * ys[:, None]).ravel().astype(np.complex128)
    out = np.full(dc.size, np.nan)
    dz = np.zeros_like(dc)
    r = np.zeros(dc.size, np.int64)
    idx = np.arange(dc.size)
    L = len(Z)
    for i in range(1, maxiter + 1):
        dz = 2 * Z[r] * dz + dz * dz + dc
        r = r + 1
        zf = Z[r] + dz
        m2 = zf.real ** 2 + zf.imag ** 2
        esc = m2 > 256.0
        if esc.any():
            out[idx[esc]] = i + 1 - np.log2(0.5 * np.log2(m2[esc]))
            keep = ~esc
            dz, r, zf, m2, dc, idx = dz[keep], r[keep], zf[keep], m2[keep], dc[keep], idx[keep]
            if idx.size == 0:
                break
        reb = (m2 < dz.real ** 2 + dz.imag ** 2) | (r >= L - 1)
        if reb.any():
            dz[reb] = zf[reb]
            r[reb] = 0
    return out.reshape(gh, gw)


def ball_period(T, R, maxiter, bits):
    """Lowest period p whose minibrot lies (roughly) within radius R of T:
    first n with |z_n| < R * |dz_n/dc|. Exact mp orbit (fine for p <~ 1e5)."""
    gmpy2.get_context().precision = bits
    z = mpc(0); d = mpc(0)
    for n in range(1, maxiter + 1):
        d = 2 * z * d + 1
        z = z * z + T
        az = abs(complex(z))
        if az > 1e6:
            return None
        if az < R * abs(complex(d)):
            return n
    return None


def nucleus(c0, p, bits, iters=40):
    gmpy2.get_context().precision = bits
    c = c0
    for _ in range(iters):
        z = mpc(0); d = mpc(0)
        for _ in range(p):
            d = 2 * z * d + 1
            z = z * z + c
        if d == 0:
            return None
        dc = z / d
        c = c - dc
        if abs(complex(dc)) < 1e-6 * 2.0 ** (-bits / 2.4):
            break
    return c


def mini_size(c, p, bits):
    gmpy2.get_context().precision = bits
    z = mpc(0); l = mpc(1); b = mpc(1)
    for _ in range(1, p):
        z = z * z + c
        l = 2 * z * l
        b = b + 1 / l
    s = 1 / (b * l * l)
    return abs(complex(s)) if complex(s) != 0 else float(abs(s))


def busy_target(S, rng, mode):
    """Pick a pixel in the central region: 'busy' = varied surroundings,
    'ridge' = thin filaments (dendrite branches) for evolving trees."""
    gh, gw = S.shape
    interior = np.isnan(S)
    L = np.log(np.where(interior, np.nanmax(S) * 1.2 if np.isfinite(S).any() else 1, S) + 1)
    pad = np.pad(L, 2, mode="edge")
    win = np.stack([pad[dy:dy + gh, dx:dx + gw] for dy in range(5) for dx in range(5)])
    if mode == "ridge":
        score = np.clip(L - win.mean(0), 0, None)
    else:
        score = win.std(0)
    score[interior] = 0
    m_y, m_x = gh // 6, gw // 6
    score[:m_y] = score[-m_y:] = 0
    score[:, :m_x] = score[:, -m_x:] = 0
    p = score.ravel() ** 2
    if p.sum() <= 0:
        return None
    k = rng.choice(p.size, p=p / p.sum())
    return divmod(k, gw)


def walk(args):
    seed, levels = args
    rng = np.random.default_rng(seed)
    seeds = [s for s in SEEDS if ONLY in s[3]] if ONLY else SEEDS
    if FAVORITES and not ONLY and rng.random() < 0.35:
        sre, sim, w0, label, mode = FAVORITES[rng.integers(len(FAVORITES))]
    else:
        sre, sim, w0, label, mode = seeds[rng.integers(len(seeds))]
    if not ONLY and rng.random() < (0.45 if mode == "busy" else 0.2):
        mode = "ridge" if mode == "busy" else "busy"
    bits = prec_for(w0 * 1e-6)
    gmpy2.get_context().precision = bits
    center = mpc_s(sre, sim, bits)
    # jitter the seed so walks differ
    center += mpc(complex(rng.uniform(-0.3, 0.3) * w0, rng.uniform(-0.3, 0.3) * w0))
    width = w0
    frames = []
    path = []
    for lvl in range(levels):
        bits = prec_for(width * 1e-12)
        gmpy2.get_context().precision = bits
        it = 3000 if not path else int(min(60000, 40 * path[-1]["p"] + 4000))
        Z = ref_orbit(center, it)
        S = perturb_grid(center, center, Z, width, 96, 72, it)
        found = None
        for _attempt in range(6):
            t = busy_target(S, rng, mode)
            if t is None:
                break
            yy, xx = t
            step = width / 95
            T = center + mpc(complex((xx - 47.5) * step, -(yy - 35.5) * step))
            R = width * rng.uniform(0.01, 0.08)
            p = ball_period(T, R, 60000, bits)
            if not p or (path and p <= path[-1]["p"]):
                continue
            N = nucleus(T, p, bits)
            if N is None or abs(complex(N - T)) > 4 * R:
                continue
            s = mini_size(N, p, bits)
            # Degenerate Newton results report absurd sizes (1e-101 at period 36).
            if not (width * 1e-9 < s < width * 0.05):
                continue
            # Bulb nuclei (attached to a bigger component) frame as mostly
            # interior: render a cheap preview of the k~60 frame and reject.
            pw = s * 60
            Zp = ref_orbit(N, min(60000, 40 * p + 4000))
            Sp = perturb_grid(N, N, Zp, pw, 48, 36, min(60000, 40 * p + 4000))
            if np.isnan(Sp).mean() > 0.4:
                continue
            found = (N, p, s)
            break
        if not found:
            break
        N, p, s = found
        path.append({"p": p, "size": s})
        # Frames at k x the minibrot size: ~25x shows the minibrot with its
        # echo structure, ~250x the wider decorated zone around it. Offset
        # the center so the minibrot isn't always dead center.
        for k in KS:
            fw = s * k * rng.uniform(0.8, 1.25)
            off = complex(rng.uniform(-0.25, 0.25), rng.uniform(-0.18, 0.18)) * fw if k > 25 else 0j
            fc = N + mpc(off)
            frames.append({"level": lvl + 1, "f": k, "cre": fc.real, "cim": fc.imag, "zoom": fw, "p": p, "size": s})
        # Next view: the decorated zone around this minibrot; the next target
        # (and its deeper minibrot) comes from there.
        center = N
        width = s * rng.uniform(120, 500)
    out = []
    for fr in frames:
        bits = prec_for(fr["zoom"] * 1e-4)
        digits = int(bits * 0.302) + 2
        out.append({
            "type": "mandelbrot", "seed_region": label, "mode": mode,
            "cre": format(fr["cre"], f".{digits}e"), "cim": format(fr["cim"], f".{digits}e"),
            "zoom": fr["zoom"], "iters": int(min(400000, 60 * fr["p"] + 20000)),
            "julia": None, "level": fr["level"], "f": fr["f"], "period": fr["p"],
            "mini_size": fr["size"], "path": [x["p"] for x in path],
        })
    return out


def main():
    n = int(sys.argv[1]); out = sys.argv[2]
    seed = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else int(time.time())
    levels = int(sys.argv[sys.argv.index("--levels") + 1]) if "--levels" in sys.argv else 3
    global KS, ONLY
    if "--only" in sys.argv:
        ONLY = sys.argv[sys.argv.index("--only") + 1]
    if "--ks" in sys.argv:
        KS = tuple(int(k) for k in sys.argv[sys.argv.index("--ks") + 1].split(","))
    jobs = [(seed * 1000 + i, levels) for i in range(n)]
    cnt = 0
    with Pool(10) as pool, open(out, "a") as f:
        for i, frames in enumerate(pool.imap_unordered(walk, jobs)):
            for j, r in enumerate(frames):
                r["id"] = f"morph{seed % 100000:05d}_{i:03d}_{r['level']}k{int(r['f']):03d}"
                f.write(json.dumps(r) + "\n"); cnt += 1
            f.flush()
            print(f"walk {i}: levels {max([x['level'] for x in frames], default=0)} periods {frames[-1]['path'] if frames else []}", flush=True)
    print(cnt, "frames", flush=True)


if __name__ == "__main__":
    main()
