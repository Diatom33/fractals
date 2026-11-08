"""Round 17 candidates: (d-1)-gon minibrot bodies/halos and d^k-armed rings from z^d + c walks.

Usage: mkpicks17.py d:w,w,w [d:w,w ...]   (walk ids to use from multi<d>/d<d>.json, deepest level)
Writes round17/candidates.jsonl and round17/picks_s.json.
"""
import json
import sys
from pathlib import Path

EX = Path("/home/diatom/Pictures/fractals/explore")
POLY = {3: "oval", 4: "triangle", 5: "square", 6: "pentagon", 7: "hexagon", 8: "heptagon", 9: "octagon"}
PALS = ("glade@7@24", "hoarfrost", "halo")
C, P = {}, []

for spec in sys.argv[1:]:
    d, ws = spec.split(":")
    d = int(d)
    rows = json.load(open(EX / f"multi{d}" / f"d{d}.json"))
    for w in map(int, ws.split(",")):
        r = max((x for x in rows if x["walk"] == w), key=lambda x: x["level"])
        for k in (8, 25):
            cid = f"mb{d}_w{w}l{r['level']}_k{k:03d}"
            C[cid] = dict(type="multibrot", power=d, cre=r["cre"], cim=r["cim"], zoom=r["size"] * k,
                          iters=int(8000 + 30 * r["period"]), julia=None, id=cid, pool="r17")
            shape = (f"{POLY[d]} minibrot in its {POLY[d]} halo" if k == 8
                     else f"{d}-fold arms around a {POLY[d]} minibrot")
            for pal in PALS:
                P.append(dict(id=cid, palette=pal, title=f"z^{d} + c: {shape}, {pal.split('@')[0]}",
                              blurb=f"z^{d} + c minibrots have {d - 1} lobes; nested rings multiply by {d}"))

out = EX / "round17"
out.mkdir(exist_ok=True)
with open(out / "candidates.jsonl", "w") as f:
    for c in C.values():
        f.write(json.dumps(c) + "\n")
json.dump(P, open(out / "picks_s.json", "w"), indent=0)
print(len(C), len(P))
