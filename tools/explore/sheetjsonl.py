"""Render every candidate in a jsonl small (sorted by score) and tile a labeled sheet.

Usage: sheetjsonl.py in.jsonl out.jpg [palette]
"""
import json
import sys
from pathlib import Path

import fr
from screen import render_cand

src, out = Path(sys.argv[1]), Path(sys.argv[2])
pal = sys.argv[3] if len(sys.argv) > 3 else "smooth"
C = [json.loads(l) for l in open(src)]
if not any("role" in c for c in C):
    C.sort(key=lambda c: -c.get("score", 0))
d = out.parent / (out.stem + "_tiles"); d.mkdir(parents=True, exist_ok=True)
paths, labels = [], []
for c in C:
    p = d / f"{c['id']}_{pal.replace('@', '-')}.png"
    try:
        if not p.exists():
            render_cand(c, p, pal, 320, 240)
    except Exception as e:
        print("FAIL", c["id"], str(e)[:120]); continue
    paths.append(p); labels.append(f"{c['id']} {c.get('score', '')} {c['zoom']:.0e}")
fr.contact_sheet(paths, out, labels=labels, cols=6, tile=(260, 195))
print(out, len(paths))
