"""Round 12: reframes/deeper continuations of round-11 loves, and 062 explored in other palettes."""
import json

RD = "/home/diatom/Pictures/fractals/explore/round12"
NAME = {"morph00041_006_2k060": "Cyan Spirals", "morph00041_033_1k060": "Night Dendrite",
        "morph00041_066_2k060": "Snow Star", "morph00051_008_1k060": "Spiral Canopy (one minibrot up)",
        "morph00041_062_1k060": "Paisley Stream"}
VAR = {"zin": "closer", "zin2": "much closer", "zout": "wider", "L": "shifted left", "R": "shifted right",
       "U": "shifted up", "D": "shifted down", "thirds": "off-center",
       "deep020_1": "one minibrot deeper", "deep020_2": "two minibrots deeper", "deep037_1": "one minibrot deeper",
       "deep037_2": "two minibrots deeper", "deep054_1": "one minibrot deeper", "deep054_2": "two minibrots deeper"}
PICK = {"morph00041_006_2k060": "zout U R deep054_1", "morph00041_033_1k060": "zout R D thirds zin2",
        "morph00041_066_2k060": "zout L U D deep037_1", "morph00051_008_1k060": "zin zin2 L deep020_1 deep037_2",
        "morph00041_062_1k060": "zin2 zout L U deep020_1 deep054_2"}
C = {json.loads(l)["id"]: json.loads(l) for l in open(f"{RD}/candidates.jsonl")}
out = []
for base, vs in PICK.items():
    for v in vs.split():
        cid = f"{base}_{v}"
        assert cid in C, cid
        out.append({"id": cid, "palette": "glade@7@24", "title": f"{NAME[base]} ({VAR[v]}), glade",
                    "blurb": "refinement of a round-11 love"})
# Paisley Stream (062) in other palettes: the user asked for "more interesting things" with this spot
src = json.loads(next(l for l in open("/home/diatom/Pictures/fractals/explore/round11/candidates.jsonl")
                      if json.loads(l)["id"] == "morph00041_062_1k060"))
C["morph00041_062_1k060"] = src
for pal, lab in (("lichtenberg@16@30", "lichtenberg 16 / 30"), ("lichtenberg@4@3", "lichtenberg 4 / 3"),
                 ("slotcanyon", "slotcanyon"), ("noctilucent", "noctilucent with relief"),
                 ("corona", "corona"), ("obsidian", "obsidian")):
    out.append({"id": "morph00041_062_1k060", "palette": pal, "title": f"Paisley Stream, {lab}",
                "blurb": "the round-11 'interesting spot' in other palettes"})
with open(f"{RD}/candidates.jsonl", "w") as f:
    for c in C.values():
        f.write(json.dumps(c) + "\n")
json.dump(out, open(f"{RD}/picks_m.json", "w"), indent=0)
print(len(out))
