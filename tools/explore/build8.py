"""Round-8 pool: shallow frames for canopy, sparse trees for noctilucent. All standard orientation."""
import json
from pathlib import Path

import fr

EX = fr.OUT
RD = EX / "round8"


def load(r):
    return [json.loads(l) for l in open(EX / r / "candidates.jsonl")]


def std(c):
    c = dict(c)
    if not c["id"].startswith(("morph", "mini")):
        if c["type"] not in ("mandelbrot", "julia"):
            return None
        c["cim"] = str(-float(c["cim"]))
        if c.get("julia"):
            c["julia"] = [c["julia"][0], -c["julia"][1]]
    return c


seen = set()
out = []


def add(c, pool):
    c = std(c)
    if c is None or c["id"] in seen:
        return
    seen.add(c["id"]); c["pool"] = pool; out.append(c)


for r in ("round1", "round2"):
    for c in load(r):
        if c["type"] == "mandelbrot" and c["zoom"] >= 2e-5:
            add(c, "canopy")
for r in ("round3", "round4", "round5", "round6", "round7"):
    for c in load(r):
        if c["zoom"] >= 2e-5:
            add(c, "canopy")
sparse = set(open(EX / "round7" / "sparse_ids.txt").read().split())
for c in load("round7"):
    if c["id"] in sparse:
        add(dict(c), "nocti") if c["id"] not in seen else None
for r in ("round3", "round4", "round5"):
    for c in load(r):
        if c.get("seed_region", "").startswith(("antenna", "top-antenna")) or c.get("refine_of"):
            add(c, "nocti")
RD.mkdir(exist_ok=True)
with open(RD / "candidates.jsonl", "w") as f:
    for c in out:
        f.write(json.dumps(c) + "\n")
from collections import Counter
print(len(out), Counter(c["pool"] for c in out))
