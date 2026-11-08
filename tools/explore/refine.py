"""Refine loved frames: reframes (zoom/offset variants) plus deeper morph
continuations seeded at the loved frame.

Usage: refine.py out_dir id1 [id2 ...]   (ids from any round's candidates.jsonl)
Writes out_dir/candidates.jsonl. Standard orientation (morph rounds only).
"""
import json
import sys
from pathlib import Path

import gmpy2
import numpy as np
from gmpy2 import mpc, mpfr

import morph

ROOT = Path.home() / "Pictures" / "fractals" / "explore"


def load_all():
    C = {}
    for f in ROOT.glob("round*/candidates.jsonl"):
        for l in open(f):
            d = json.loads(l)
            C[d["id"]] = d
    return C


def reframes(c, rng):
    bits = morph.prec_for(c["zoom"] * 1e-4)
    gmpy2.get_context().precision = bits
    base = mpc(mpfr(c["cre"], bits), mpfr(c["cim"], bits))
    digits = int(bits * 0.302) + 2
    out = []
    for tag, zf, ox, oy in [("zin", 0.6, 0, 0), ("zin2", 0.35, 0, 0), ("zout", 1.7, 0, 0),
                            ("L", 1.0, -0.22, 0), ("R", 1.0, 0.22, 0), ("U", 1.0, 0, 0.18),
                            ("D", 1.0, 0, -0.18), ("thirds", 1.25, rng.choice([-0.17, 0.17]), rng.choice([-0.13, 0.13]))]:
        w = c["zoom"] * zf
        ctr = base + mpc(complex(ox * c["zoom"], oy * c["zoom"]))
        d = dict(c)
        d.update(id=f"{c['id']}_{tag}", cre=format(ctr.real, f".{digits}e"),
                 cim=format(ctr.imag, f".{digits}e"), zoom=w, refine_of=c["id"], variant=tag)
        out.append(d)
    return out


def continuation(c, seed, levels=2):
    """Deeper morph walk starting from the loved frame's view."""
    morph.SEEDS[:] = [(c["cre"], c["cim"], c["zoom"], f"from-{c['id'][10:]}", c.get("mode", "busy"))]
    morph.FAVORITES[:] = []
    morph.KS = (60,)
    frames = morph.walk((seed, levels))
    out = []
    for f in frames:
        f["id"] = f"{c['id']}_deep{seed % 1000:03d}_{f['level']}"
        f["refine_of"] = c["id"]
        f["variant"] = f"deeper {f['level']}"
        out.append(f)
    return out


def main():
    od = Path(sys.argv[1]); od.mkdir(parents=True, exist_ok=True)
    C = load_all()
    rng = np.random.default_rng(7)
    with open(od / "candidates.jsonl", "w") as f:
        for cid in sys.argv[2:]:
            c = C[cid]
            items = reframes(c, rng)
            for s in range(3):
                items += continuation(c, 9000 + s * 17 + len(cid))
            for d in items:
                f.write(json.dumps(d) + "\n")
            print(cid, len(items), flush=True)


if __name__ == "__main__":
    main()
