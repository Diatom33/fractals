"""Round-8 picks: canopy on shallow frames, noctilucent on lace/trees, Palm Wheel lichtenberg variants."""
import json
from pathlib import Path

RD = "/home/diatom/Pictures/fractals/explore/round8"
GAL = json.load(open(Path(__file__).resolve().parent / "gallery" / "gallery.json"))
known = {}
for r in GAL["rounds"]:
    for it in r["items"]:
        loc = it["id"].rsplit("_", 1)[0] if not it["id"].endswith(("-0.04", "-0.08", "-0.15")) else it["id"].rsplit("_", 1)[0]
        if it.get("title") and loc not in known:
            known[loc] = it["title"].split(",")[0].split(":")[0]

P = [
    # requested: Palm Wheel + Twin Pools in noctilucent, Palm Wheel lichtenberg params
    ("morph00041_045_1k060", "noctilucent", "Palm Wheel, noctilucent", "default wisps (5 / 1.4)"),
    ("morph00041_045_1k060", "noctilucent@5@0.8", "Palm Wheel, noctilucent bright", "wisp contrast 0.8"),
    ("morph00041_045_1k060", "noctilucent@9@1.4", "Palm Wheel, noctilucent fine", "wisp density 9"),
    ("morph00041_045_1k060_zin2", "noctilucent@5@0.8", "Palm Wheel: much closer, noctilucent", "wisp contrast 0.8"),
    ("mand00031_0076", "noctilucent", "Twin Pools, noctilucent", "default wisps (5 / 1.4)"),
    ("mand00031_0076", "noctilucent@2.5@0.8", "Twin Pools, noctilucent bright", "wisp density 2.5, contrast 0.8"),
    ("mand00031_0076", "canopy@6", "Twin Pools, canopy", "trap scale 6"),
    ("morph00041_045_1k060", "lichtenberg@2@10", "Palm Wheel, lichtenberg 2 / 10", "branch density 2, sharpness 10: clean fronds on black"),
    ("morph00041_045_1k060", "lichtenberg@4@3", "Palm Wheel, lichtenberg 4 / 3", "branch density 4, sharpness 3: warm glow"),
    ("morph00041_045_1k060", "lichtenberg@6@30", "Palm Wheel, lichtenberg 6 / 30", "branch density 6, sharpness 30: faint filaments"),
    ("morph00041_045_1k060", "lichtenberg@10@30", "Palm Wheel, lichtenberg 10 / 30", "branch density 10, sharpness 30: dark streamlines"),
    ("morph00041_045_1k060", "lichtenberg@10@10", "Palm Wheel, lichtenberg 10 / 10", "branch density 10, sharpness 10: bright feathers"),
    ("morph00041_045_1k060", "lichtenberg@16@30", "Palm Wheel, lichtenberg 16 / 30", "branch density 16, sharpness 30"),
    # canopy (shallow frames only)
    ("mand00031_0099", "canopy@6", "Spiral Shelf, canopy", "trap scale 6"),
    ("mand00031_0099", "canopy", "Spiral Shelf, canopy pink", "trap scale 3"),
    ("morph00041_017_1k060", "canopy@6", "Seahorse Drift, canopy", "trap scale 6"),
    ("mand00031_0118", "canopy@6", None, "trap scale 6"),
    ("mand00031_0416", "canopy@6", "Firefly Bank, canopy", "trap scale 6"),
    ("mand00031_0290", "canopy", "Rose Spirals, canopy", "trap scale 3"),
    ("mand00011_0199", "canopy@6", "Nebula Branch, canopy", "trap scale 6"),
    ("mand00011_0267", "canopy", "Sunburst Reef, canopy", "trap scale 3"),
    ("mand00031_0535", "canopy@6", "Ember Wheel, canopy", "trap scale 6"),
    ("morph00041_012_1k060", "canopy@6", "Curl Garden, canopy", "trap scale 6"),
    ("mand00031_0373", "canopy", "Green Fern Bay, canopy", "trap scale 3"),
    ("mand00031_0299", "canopy@6", "Seahorse Ribbon, canopy", "trap scale 6"),
    ("mand00031_0081", "canopy@6", "Green Glow, canopy", "trap scale 6"),
    ("mini00061_0094", "canopy@6", None, "trap scale 6"),
    # noctilucent on lace and trees
    ("morph00051_008_2k060_deep020_1", "noctilucent", None, "one minibrot deeper, halo"),
    ("morph00051_008_2k060_zin2", "noctilucent", None, "much closer"),
    ("morph00041_011_4k060_zin2", "noctilucent", None, "much closer"),
    ("morph00041_011_4k060_U", "noctilucent", None, "shifted"),
    ("morph00041_045_1k060_deep037_1", "noctilucent", None, "one minibrot deeper"),
    ("morph00041_002_1k060_deep054_2", "noctilucent", None, "two minibrots deeper"),
    ("morph00041_002_1k060_thirds", "noctilucent", None, "off-center"),
    ("morph00041_002_2k060", "noctilucent", None, "next minibrot"),
    ("morph00041_006_1k060_deep054_2", "noctilucent", None, "two minibrots deeper"),
    ("morph00051_038_3k060", "noctilucent", None, ""),
    ("morph00051_056_4k060", "noctilucent", None, ""),
    ("morph00071_040_3k060", "noctilucent", "Top Antenna Lace", "two minibrots below the round-7 love"),
    ("morph00071_029_3k060", "noctilucent", "Comb", ""),
    ("morph00071_024_1k060", "noctilucent", "Frost I", "antenna frost dendrites"),
    ("morph00071_104_1k060", "noctilucent", "Frost II", "antenna frost dendrites"),
    ("morph00071_006_2k060", "noctilucent", "Frost Star", "antenna frost dendrites"),
    ("morph00041_054_1k060", "noctilucent", "Flower Ring", ""),
    ("morph00041_052_1k060", "noctilucent", "Flower Wheel", ""),
    ("morph00041_019_3k060", "noctilucent", "Pale Lace", ""),
    ("morph00041_037_2k060", "noctilucent", "Spiral Chain", ""),
    ("juli00012_0058", "noctilucent", None, "Julia set"),
]
cands = {json.loads(l)["id"]: json.loads(l) for l in open(f"{RD}/candidates.jsonl")}
out = []
for cid, pal, title, blurb in P:
    assert cid in cands, cid
    if title is None:
        base = cid
        while base not in known and "_" in base:
            base = base.rsplit("_", 1)[0]
        name = known.get(base, cid)
        title = f"{name}{' (' + blurb + ')' if blurb else ''}, {pal.split('@')[0]}"
    out.append({"id": cid, "palette": pal, "title": title, "blurb": blurb})
json.dump(out, open(f"{RD}/picks_i.json", "w"), indent=0)
for o in out:
    print(o["id"], "|", o["palette"], "|", o["title"])
