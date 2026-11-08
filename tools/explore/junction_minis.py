"""Minibrots sitting right at a q-spoke junction: their halos should be 2q-fold (q=3 -> hexagons).

For each junction (polygon.py output), search with shrinking radius R for the lowest-period
minibrot within R (ball period), Newton to its nucleus, and write frames at k x its size.
Also writes frames of the junction star itself (q-fold) at a few widths.

Usage: junction_minis.py misiu.json out_dir/candidates.jsonl
"""
import json
import sys

import gmpy2

import morph


def main():
    M = json.load(open(sys.argv[1]))
    out = open(sys.argv[2], "a")
    seen = set()
    for q, p, cre, cim, r in M:
        if q not in (3, 5, 6, 7):
            continue
        for e in range(2, 13):
            R = r * 10 ** (-e / 2)
            bits = morph.prec_for(R * 1e-6)
            gmpy2.get_context().precision = bits
            T = morph.mpc_s(cre, cim, bits)
            per = morph.ball_period(T, R, 200000, bits)
            if not per or (q, p, per) in seen:
                continue
            N = morph.nucleus(T, per, bits + 32)
            if N is None:
                continue
            s = morph.mini_size(N, per, bits + 32)
            if not (R * 1e-6 < s < R):
                continue
            seen.add((q, p, per))
            digits = int(morph.prec_for(s * 1e-4) * 0.302) + 2
            for k in (25, 60, 150):
                c = dict(type="mandelbrot", cre=format(N.real, f".{digits}e"), cim=format(N.imag, f".{digits}e"),
                         zoom=s * k, iters=int(min(400000, 60 * per + 20000)), julia=None, period=per, mini_size=s,
                         level=1, seed_region=f"junction{q}-{p}", mode="junction", f=k,
                         id=f"jn{q}{p}_p{per}_k{k:03d}")
                out.write(json.dumps(c) + "\n")
            print(q, p, "R", f"{R:.1e}", "period", per, "size", f"{s:.2e}", flush=True)
        # the junction star itself
        for w in (r * 0.05, r * 0.005, r * 5e-4, r * 5e-5):
            c = dict(type="mandelbrot", cre=cre, cim=cim, zoom=w, iters=20000, julia=None, seed_region=f"junction{q}-{p}",
                     mode="star", id=f"star{q}{p}_{w:.0e}")
            out.write(json.dumps(c) + "\n")
        out.flush()


if __name__ == "__main__":
    main()
