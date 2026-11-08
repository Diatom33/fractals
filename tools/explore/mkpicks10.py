"""Round 10: the new Glade palette on favorites, plus Rose Spirals reframes."""
import json

EX = "/home/diatom/Pictures/fractals/explore/"
C = {}
for r in ("round3", "round4", "round5", "round7", "round8", "round9"):
    for l in open(EX + r + "/candidates.jsonl"):
        c = json.loads(l)
        if r in ("round8", "round9") or c["id"].startswith(("morph", "mini")):
            C.setdefault(c["id"], c)
out = {}


def use(cid):
    c = dict(C[cid]); c["pool"] = c.get("pool", "r10"); out[cid] = c


# Rose Spirals (mand00031_0290) reframes
base = C["mand00031_0290"]
cre, cim, z = float(base["cre"]), float(base["cim"]), base["zoom"]
for tag, (dx, dy, f) in {"zin": (0, 0, 0.6), "zout": (0, 0, 1.6), "L": (-0.2, 0, 0.85), "R": (0.2, 0, 0.85),
                         "U": (0, 0.15, 0.85), "D": (0, -0.15, 0.85), "zin2": (0.05, 0.05, 0.4)}.items():
    cid = f"mand00031_0290_{tag}"
    out[cid] = dict(base, id=cid, cre=repr(cre + dx * z), cim=repr(cim + dy * z), zoom=z * f, pool="r10",
                    refine_of="mand00031_0290", variant=tag)
P = []
G = [("morph00041_045_1k060", "Palm Wheel"), ("morph00051_008_2k060", "Spiral Canopy"),
     ("morph00051_008_2k060_zin2", "Spiral Canopy, much closer"), ("morph00051_030_3k060_zin2", "Root System II"),
     ("morph00041_011_4k060", "Spiral Star IV"), ("morph00041_014_3k060", "Snowflake"),
     ("morph00041_002_1k060", "Elephant Star"), ("morph00071_040_1k060", "Top Antenna Tree"),
     ("mand00031_0076", "Twin Pools"), ("mand00031_0290", "Rose Spirals"), ("mand00031_0373", "Green Fern Bay"),
     ("gar093_0225_f0", "Hidden Garden 225"), ("mand00031_0099", "Spiral Shelf"), ("m00071_040_1k060_j3", "Julia opening, Top Antenna"),
     ("mini00061_0094", "Island Chain"), ("morph00041_006_1k060", "Spiral Crossing"), ("morph00041_045_1k060_zin2", "Palm Wheel, much closer")]
for cid, name in G:
    use(cid)
    P.append(dict(id=cid, palette="glade", title=f"{name}, glade", blurb="new palette: canopy bokeh over relief-lit violet, structure at every scale"))
for pal, lab in (("glade@2.5@24", "softer bokeh"), ("glade@7@24", "dense bokeh"), ("glade@4@8", "short bokeh memory"), ("glade@4@80", "long bokeh memory")):
    P.append(dict(id="morph00041_045_1k060", palette=pal, title=f"Palm Wheel, glade ({lab})", blurb=f"glade sliders {pal.split('@', 1)[1].replace('@', ' / ')}"))
for tag in ("zin", "zout", "L", "R", "U", "D", "zin2"):
    cid = f"mand00031_0290_{tag}"
    P.append(dict(id=cid, palette="canopy", title=f"Rose Spirals ({tag}), canopy", blurb="reframe toward a wall piece"))
    P.append(dict(id=cid, palette="glade", title=f"Rose Spirals ({tag}), glade", blurb="reframe toward a wall piece"))
P.append(dict(id="mand00031_0290", palette="canopy@2", title="Rose Spirals, canopy trap 2", blurb="canopy trap scale 2"))
with open(EX + "round10/candidates.jsonl", "w") as f:
    for c in out.values():
        f.write(json.dumps(c) + "\n")
json.dump(P, open(EX + "round10/picks_k.json", "w"), indent=0)
print(len(out), len(P))
