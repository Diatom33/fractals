"""Screen the round-8 pool: canopy pool in canopy, nocti pool (and round-7 sparse) in noctilucent."""
import json
import sys

import fr
from screen import render_cand

RD = fr.OUT / "round8"
sparse = set(open(fr.OUT / "round7" / "sparse_ids.txt").read().split())
C = [json.loads(l) for l in open(RD / "candidates.jsonl")]
jobs = []
for c in C:
    if c["pool"] == "canopy":
        jobs.append((c, "canopy"))
    if c["pool"] == "nocti" or c["id"] in sparse:
        jobs.append((c, "noctilucent"))
only = sys.argv[1] if len(sys.argv) > 1 else None
for pal in ("noctilucent", "canopy"):
    if only and pal != only:
        continue
    sd = RD / f"screen_{pal}"; sd.mkdir(exist_ok=True)
    done = []
    for c, p in jobs:
        if p != pal:
            continue
        out = sd / f"{c['id']}.png"
        try:
            if not out.exists():
                render_cand(c, out, p, 400, 300)
            done.append(c["id"])
        except Exception as e:
            print("FAIL", c["id"], str(e)[:150], flush=True)
    sh = RD / f"sheets_{pal}"; sh.mkdir(exist_ok=True)
    for k in range(0, len(done), 30):
        ids = done[k:k + 30]
        fr.contact_sheet([sd / f"{x}.png" for x in ids], sh / f"sheet_{k // 30:02d}.jpg",
                         labels=ids, cols=6, tile=(260, 195))
    print(pal, len(done), flush=True)
