"""Screen refinement candidates in the palette their parent was loved in.

Usage: screen_loved.py round_dir  (parent palette map is hardcoded below)
"""
import json
import sys
from pathlib import Path

import fr
from screen import render_cand

PARENT_PAL = {
    "morph00041_011_4k060": "corona", "morph00041_014_3k060": "corona",
    "morph00041_045_1k060": "lichtenberg", "morph00051_008_2k060": "corona",
    "morph00051_030_3k060": "inverted@0.08", "morph00041_006_1k060": "lichtenberg",
    "morph00041_002_1k060": "obsidian",
}

rd = Path(sys.argv[1])
C = [json.loads(l) for l in open(rd / "candidates.jsonl")]
sd = rd / "screen"; sd.mkdir(exist_ok=True)
done = []
for c in C:
    out = sd / f"{c['id']}.png"
    try:
        if not out.exists():
            render_cand(c, out, PARENT_PAL[c["refine_of"]], 400, 300)
        done.append(c)
    except Exception as e:
        print("FAIL", c["id"], str(e)[:150], flush=True)
sh = rd / "sheets"; sh.mkdir(exist_ok=True)
by = {}
for c in done:
    by.setdefault(c["refine_of"], []).append(c)
for k, (parent, cs) in enumerate(by.items()):
    fr.contact_sheet([sd / f"{c['id']}.png" for c in cs], sh / f"r{k}.jpg",
                     labels=[f"{parent[10:]} {c['variant']}" + (f" p{c['period']}" if 'period' in c and c['variant'].startswith('deeper') else '') for c in cs],
                     cols=5, tile=(320, 240))
print("done", len(done), flush=True)
