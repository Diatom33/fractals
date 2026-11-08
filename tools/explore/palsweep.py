"""Sweep favorite frames through a set of palettes: one row per frame.

Usage: palsweep.py out_dir pal1,pal2,... [w h] [--ss N]
"""
import json
import sys
from pathlib import Path

import fr
from screen import render_cand

FAV = [("round12", "morph00051_008_1k060_zin", "Spiral Canopy"), ("round3", "morph00041_045_1k060", "Palm Wheel"),
       ("round3", "morph00041_066_2k060", "Snow Star"), ("round3", "morph00041_033_1k060", "Night Dendrite"),
       ("round8", "mand00031_0076", "Twin Pools"), ("round9", "m00071_040_1k060_j3", "Julia opening")]


def main():
    od = Path(sys.argv[1]); od.mkdir(parents=True, exist_ok=True)
    pals = sys.argv[2].split(",")
    w, h = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 and not sys.argv[3].startswith("--") else (640, 480)
    extra = ["--ss", sys.argv[sys.argv.index("--ss") + 1]] if "--ss" in sys.argv else []
    paths, labels = [], []
    for rnd, cid, name in FAV:
        c = next(json.loads(l) for l in open(fr.OUT / rnd / "candidates.jsonl") if json.loads(l)["id"] == cid)
        for p in pals:
            out = od / f"{cid}_{p.replace('@', '-')}.png"
            if not out.exists():
                try:
                    render_cand(c, out, p, w, h, extra=extra)
                except Exception as e:
                    print("FAIL", cid, p, str(e)[-300:], flush=True)
                    continue
            paths.append(out); labels.append(f"{name} {p}")
        print(name, flush=True)
    fr.contact_sheet(paths, od / "sweep.jpg", labels=labels, cols=len(pals), tile=(320, 240))
    print(od / "sweep.jpg")


if __name__ == "__main__":
    main()
