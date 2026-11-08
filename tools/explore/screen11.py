"""Round 11 pool: deep morph frames (not main-antenna, not yet shown in glade) screened in glade@7.

Writes round11/candidates.jsonl, round11/screen/<id>.png, and sheets of the non-empty ones.
"""
import json

import fr
from emptiness import is_empty
from screen import render_cand

EX = fr.OUT
RD = EX / "round11"; RD.mkdir(exist_ok=True)
shown = {it["id"].rsplit("_", 1)[0] for r in json.load(open("gallery/gallery.json"))["rounds"]
         for it in r["items"] if it["palette"].startswith("glade")}
pool = {}
for r in ("round3", "round4", "round5", "round7"):
    for l in open(EX / r / "candidates.jsonl"):
        c = json.loads(l)
        reg = c.get("seed_region", "")
        if reg == "antenna-trees" or c["id"] in shown or c["zoom"] > 1e-4:
            continue
        pool.setdefault(c["id"], c)
with open(RD / "candidates.jsonl", "w") as f:
    for c in pool.values():
        f.write(json.dumps(c) + "\n")
print("pool", len(pool), flush=True)
sd = RD / "screen"; sd.mkdir(exist_ok=True)
keep = []
for i, c in enumerate(pool.values()):
    out = sd / f"{c['id']}.png"
    try:
        if not out.exists():
            render_cand(c, out, "glade@7@24", 400, 300)
    except Exception as e:
        print("FAIL", c["id"], str(e)[:100], flush=True)
        continue
    e, cov, cen = is_empty(out)
    if not e:
        keep.append(c["id"])
    if i % 50 == 0:
        print(i, flush=True)
sh = RD / "sheets"; sh.mkdir(exist_ok=True)
for k in range(0, len(keep), 30):
    ids = keep[k:k + 30]
    fr.contact_sheet([sd / f"{x}.png" for x in ids], sh / f"sheet_{k // 30:02d}.jpg", labels=ids, cols=6, tile=(260, 195))
print("kept", len(keep), "sheets", (len(keep) + 29) // 30, flush=True)
