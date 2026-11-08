"""Find embedded Julia sets: zoom sequences toward a minibrot, scored for 'the Julia set is the first
thing you see' in a sparse palette.

Zooming from a view of width W toward a minibrot of size s, an embedded Julia set opens up around
the log-midpoint. In palettes that don't draw every sub-pixel strand (obsidian's black field), the
surrounding tree fades while the Julia copy, dense with boundary, reads as one bright shape. For
each minibrot we render widths log-spaced between W and 40 s, centered on the nucleus, and score
each frame: bright structure concentrated in a central blob of decent size, little elsewhere.

Usage: juliaseq.py out_dir sources.json [palette]
sources.json: [{"id", "cre", "cim", "period", "mini_size", "start_w"}]  (standard orientation;
cre/cim near the nucleus, refined by Newton here)
Writes out_dir/seq_<id>.png strips, out_dir/candidates.jsonl (best frames), out_dir/sheet_NN.jpg.
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

STEPS = 9


def salience(path):
    a = np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255
    L = a @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    h, w = L.shape
    bright = L > max(0.25, float(np.percentile(L, 60)))
    Y, X = np.mgrid[:h, :w]
    r = np.hypot(X - (w - 1) / 2, Y - (h - 1) / 2) / w
    tot = bright.sum()
    if tot < 50:
        return -1.0, 0.0
    best = -1.0
    best_r = 0.0
    for rc in (0.12, 0.18, 0.25, 0.32):
        inside = bright[r < rc].mean()
        outside = bright[r > rc + 0.08].mean() if (r > rc + 0.08).any() else 0
        s = (inside - outside * 1.5) * min(1.0, rc / 0.18)
        if s > best:
            best, best_r = s, rc
    interior = (a.max(2) == 0).mean()
    return float(best - max(0, interior - 0.08) * 3), best_r


def main():
    od = Path(sys.argv[1]); od.mkdir(parents=True, exist_ok=True)
    srcs = json.load(open(sys.argv[2]))
    pal = sys.argv[3] if len(sys.argv) > 3 else "obsidian"
    outc = []
    for src in srcs:
        s0 = src["mini_size"]
        bits = morph.prec_for(s0 * 1e-3)
        gmpy2.get_context().precision = bits
        T = morph.mpc_s(src["cre"], src["cim"], bits)
        p = src["period"]
        N = morph.nucleus(T, p, bits + 32)
        if N is None:
            print("no nucleus", src["id"], flush=True)
            continue
        s = morph.mini_size(N, p, bits + 32)
        if abs(math.log10(s / s0)) > 0.7:
            print("size mismatch", src["id"], s, s0, flush=True)
            continue
        digits = int(bits * 0.302) + 2
        w_hi, w_lo = src["start_w"], s * 40
        if w_hi <= w_lo * 3:
            continue
        frames = []
        for i in range(STEPS):
            t = i / (STEPS - 1)
            wdt = math.exp(math.log(w_hi) * (1 - t) + math.log(w_lo) * t)
            c = dict(type="mandelbrot", cre=format(N.real, f".{digits}e"), cim=format(N.imag, f".{digits}e"),
                     zoom=wdt, iters=int(min(400000, 60 * p + 20000)), julia=None, period=p, mini_size=s,
                     id=f"{src['id']}_j{i}", jseq_of=src["id"], jt=round(t, 3))
            out = od / f"{c['id']}_{pal}.png"
            try:
                if not out.exists():
                    fr.render(out, palette=pal, cre=c["cre"], cim=c["cim"], zoom=wdt, iters=c["iters"], w=320, h=240)
            except Exception as e:
                print("FAIL", c["id"], str(e)[:100], flush=True)
                continue
            sc, rc = salience(out)
            c["jscore"] = round(sc, 3)
            frames.append((c, out))
        if not frames:
            continue
        fr.contact_sheet([o for _, o in frames], od / f"seq_{src['id']}.jpg",
                         labels=[f"t={c['jt']} {c['jscore']} {c['zoom']:.0e}" for c, _ in frames],
                         cols=STEPS, tile=(200, 150))
        for c, _ in frames:
            outc.append(c)
        print(src["id"], " ".join(f"{c['jscore']:.2f}" for c, _ in frames), flush=True)
    with open(od / "candidates.jsonl", "w") as f:
        for c in outc:
            f.write(json.dumps(c) + "\n")


if __name__ == "__main__":
    main()
