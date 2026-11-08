"""Grid of palette-param values for one candidate.

Usage: paramtest.py cand_round cand_id palette p1a,p1b,.. [p2a,p2b,..]
"""
import json
import sys

import fr
from screen import render_cand

r, cid, pal = sys.argv[1:4]
p1 = [float(x) for x in sys.argv[4].split(",")]
p2 = [float(x) for x in sys.argv[5].split(",")] if len(sys.argv) > 5 else [None]
c = next(json.loads(l) for l in open(fr.OUT / r / "candidates.jsonl") if json.loads(l)["id"] == cid)
od = fr.OUT / "round8" / "params"; od.mkdir(parents=True, exist_ok=True)
paths, labels = [], []
for b in p2:
    for a in p1:
        out = od / f"{cid}_{pal}_{a}_{b}.png"
        if not out.exists():
            fr.render(out, type=c["type"], palette=pal, cre=c["cre"], cim=c["cim"], zoom=c["zoom"],
                      iters=c["iters"], w=640, h=480, julia=c.get("julia"), pparam=a, pparam2=b)
        paths.append(out); labels.append(f"{a} / {b}")
fr.contact_sheet(paths, od / f"{cid}_{pal}.jpg", labels=labels, cols=len(p1), tile=(320, 240))
print(od / f"{cid}_{pal}.jpg")
