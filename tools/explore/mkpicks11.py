"""Round 11: glade@7@24 (the loved dense-bokeh setting) on new deep morph frames and reframes of loves."""
import json
from pathlib import Path

RD = "/home/diatom/Pictures/fractals/explore/round11"
GAL = json.load(open(Path(__file__).resolve().parent / "gallery" / "gallery.json"))
known = {}
for r in GAL["rounds"]:
    for it in r["items"]:
        loc = it["id"].rsplit("_", 1)[0]
        if it.get("title") and loc not in known:
            known[loc] = it["title"].split(",")[0].split(":")[0].split(" (")[0]
NEW = "002_2k 006_2k 011_1k 011_3k 019_2k 021_2k 022_1k 023_2k 028_3k 033_1k 034_2k 037_1k 044_1k 052_1k 054_1k 057_3k 060_1k 062_1k 066_2k 068_1k 068_3k 081_1k 081_3k 084_1k".split()
ids = [f"morph00041_{x}060" for x in NEW] + ["morph00051_008_1k060"]
REFINE = {"morph00051_008_2k060_deep020_1": "one minibrot deeper", "morph00051_008_2k060_thirds": "off-center",
          "morph00041_045_1k060_L": "shifted", "morph00041_045_1k060_thirds": "off-center",
          "morph00071_040_2k060": "next minibrot", "morph00071_040_3k060": "two minibrots down",
          "morph00041_006_1k060_D": "shifted", "morph00041_011_4k060_zin2": "much closer",
          "morph00041_002_1k060_zin2": "much closer"}
C = {json.loads(l)["id"]: json.loads(l) for l in open(f"{RD}/candidates.jsonl")}
out = []
for cid in ids + list(REFINE):
    assert cid in C, cid
    base = cid
    while base not in known and "_" in base:
        base = base.rsplit("_", 1)[0]
    name = known.get(base)
    if cid in REFINE:
        title = f"{name or 'Top Antenna Tree'} ({REFINE[cid]}), glade"
    elif name:
        title = f"{name}, glade"
    else:
        title = f"Glade study {cid[5:].replace('_', '.')}"
    out.append({"id": cid, "palette": "glade@7@24", "title": title, "blurb": "glade, dense bokeh (7 / 24)"})
json.dump(out, open(f"{RD}/picks_l.json", "w"), indent=0)
for o in out:
    print(o["id"], "|", o["title"])
