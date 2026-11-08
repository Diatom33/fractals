"""Render one morph frame per zoom decade in given palettes to see where a palette breaks down.

Usage: depthtest.py out_dir palette1,palette2 [decades e.g. 2-20]
"""
import json
import math
import sys
from pathlib import Path

import fr
from screen import render_cand

EX = fr.OUT


def main():
    od = Path(sys.argv[1]); od.mkdir(parents=True, exist_ok=True)
    pals = sys.argv[2].split(",")
    lo, hi = map(int, (sys.argv[3] if len(sys.argv) > 3 else "2-20").split("-"))
    C = [json.loads(l) for r in ("round3", "round7") for l in open(EX / r / "candidates.jsonl")]
    pick = {}
    for c in C:
        d = -int(math.floor(math.log10(c["zoom"])))
        if lo <= d <= hi and d not in pick:
            pick[d] = c
    paths, labels = [], []
    for d in sorted(pick):
        c = pick[d]
        for p in pals:
            out = od / f"d{d:02d}_{p.replace('@', '-')}.png"
            if not out.exists():
                render_cand(c, out, p, 480, 360)
            paths.append(out); labels.append(f"1e-{d} {p}")
        print(d, flush=True)
    fr.contact_sheet(paths, od / "depth.jpg", labels=labels, cols=len(pals) * 2, tile=(320, 240))


if __name__ == "__main__":
    main()
