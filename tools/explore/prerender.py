"""Render picks into gallery/img (same names publish.py uses) without touching gallery.json,
so they can be QA'd before a round is published. publish.py then skips the render.

Usage: prerender.py round_dir picks.json
"""
import json
import sys
from pathlib import Path

from PIL import Image

from emptiness import is_empty
from publish import GAL, MEAN_PALETTES
from screen import render_cand


def main():
    rd = Path(sys.argv[1])
    picks = json.load(open(sys.argv[2]))
    cands = {c["id"]: c for c in map(json.loads, open(rd / "candidates.jsonl"))}
    tmp = rd / "render_tmp.png"
    for p in picks:
        key = f"{p['id']}_{p['palette'].replace('@', '-')}"
        big, thumb = GAL / "img" / f"{key}.jpg", GAL / "img" / f"t_{key}.jpg"
        if not big.exists():
            ss = "3" if p["palette"].split("@")[0] in MEAN_PALETTES else "2"
            render_cand(cands[p["id"]], tmp, p["palette"], 1600, 1200, extra=["--ss", ss], pparam=p.get("pparam"))
            im = Image.open(tmp).convert("RGB")
            im.save(big, quality=90)
            im.resize((800, 600), Image.LANCZOS).save(thumb, quality=86)
        empty, cov, center = is_empty(big)
        print(f"{'EMPTY' if empty else 'ok   '} {key}", flush=True)


if __name__ == "__main__":
    main()
