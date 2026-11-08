"""Noctilucent screen of non-antenna morph frames (rounds 3-4) and liked round 1-2 spots."""
import glob
import json

import fr
from screen import render_cand

RD = fr.OUT / "round8"


def std(c):
    if c["type"] not in ("mandelbrot", "julia"):
        return None
    c = dict(c, cim=str(-float(c["cim"])))
    if c.get("julia"):
        c["julia"] = [c["julia"][0], -c["julia"][1]]
    return c

liked = set()
for f in glob.glob(str(fr.OUT / "ratings9" / "ratings" / "*.json")):
    d = json.load(open(f))
    if d.get("r") in ("like", "love"):
        liked.add("_".join(f.split("/")[-1].split("_")[:2]))
have = {json.loads(l)["id"] for l in open(RD / "candidates.jsonl")}
new = []
for r in ("round3", "round4"):
    for l in open(fr.OUT / r / "candidates.jsonl"):
        c = json.loads(l)
        if "antenna" not in c.get("seed_region", "") and c["id"] not in have:
            c["pool"] = "nocti"; new.append(c)
for r in ("round1", "round2"):
    for l in open(fr.OUT / r / "candidates.jsonl"):
        c = json.loads(l)
        if c["id"] in liked and c["id"] not in have:
            c = std(c)
            if c:
                c["pool"] = "nocti"; new.append(c)
with open(RD / "candidates.jsonl", "a") as f:
    for c in new:
        f.write(json.dumps(c) + "\n")
print("added", len(new), flush=True)
sd = RD / "screen_noctilucent"
for c in new:
    out = sd / f"{c['id']}.png"
    try:
        if not out.exists():
            render_cand(c, out, "noctilucent", 400, 300)
    except Exception as e:
        print("FAIL", c["id"], str(e)[:150], flush=True)
print("done", flush=True)
