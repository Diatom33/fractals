"""Palette sweep: render picked locations in many palettes, one labeled sheet per location.

Usage: sweep.py round_dir id1 id2 ...   (ids from round_dir/candidates.jsonl)
Writes round_dir/sweep/<id>/<palette>.png and round_dir/sweep/<id>.jpg
"""
import json
import sys
from pathlib import Path

import fr
from screen import render_cand

# thinfilm excluded on the user's request; mono/inverted/storm rarely work.
SWEEP = ["biolum", "canopy", "obsidian", "corona", "lichtenberg", "noctilucent",
         "slotcanyon", "steve", "aurora", "smooth", "oklab", "classic"]


def main():
    rd = Path(sys.argv[1])
    ids = [a for a in sys.argv[2:] if not a.startswith("--")]
    for a in sys.argv[2:]:
        if a.startswith("--ids-file="):
            ids += open(a.split("=", 1)[1]).read().split()
    pal = SWEEP
    for a in sys.argv[2:]:
        if a.startswith("--palettes="):
            pal = a.split("=", 1)[1].split(",")
    cands = {c["id"]: c for c in map(json.loads, open(rd / "candidates.jsonl"))}
    for cid in ids:
        c = cands[cid]
        od = rd / "sweep" / cid
        od.mkdir(parents=True, exist_ok=True)
        paths = []
        for p in pal:
            out = od / f"{p}.png"
            try:
                if not out.exists():
                    render_cand(c, out, p, 640, 480)
                paths.append(out)
            except Exception as e:
                print("FAIL", cid, p, str(e)[:160], flush=True)
        fr.contact_sheet(paths, rd / "sweep" / f"{cid}.jpg", labels=[f"{cid} {x.stem}" for x in paths],
                         cols=4 if len(pal) > 6 else 3, tile=(400, 300))
        print("swept", cid, flush=True)


if __name__ == "__main__":
    main()
