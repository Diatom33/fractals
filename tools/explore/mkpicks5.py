"""Round-5 picks: refinements of loved frames."""
import json

RD = "/home/diatom/Pictures/fractals/explore/round5"
PARENT_TITLE = {
    "morph00041_011_4k060": "Spiral Star IV", "morph00041_014_3k060": "Snowflake",
    "morph00041_045_1k060": "Palm Wheel", "morph00051_008_2k060": "Spiral Canopy",
    "morph00051_030_3k060": "Root System II", "morph00041_006_1k060": "Spiral Crossing",
    "morph00041_002_1k060": "Elephant Star",
}
VARIANT = {"zin": "closer", "zin2": "much closer", "zout": "wider", "U": "shifted", "D": "shifted",
           "L": "shifted", "R": "shifted", "thirds": "off-center"}
P = [
    ("morph00041_011_4k060_zin2", "obsidian"), ("morph00041_011_4k060_zin2", "corona"),
    ("morph00041_011_4k060_zin", "slotcanyon"), ("morph00041_011_4k060_thirds", "corona"),
    ("morph00041_014_3k060_zin2", "corona"), ("morph00041_014_3k060_deep054_1", "obsidian"),
    ("morph00041_014_3k060_deep020_1", "corona"),
    ("morph00041_045_1k060_zin2", "slotcanyon"), ("morph00041_045_1k060_zin2", "obsidian"),
    ("morph00041_045_1k060_zout", "lichtenberg"), ("morph00041_045_1k060_U", "corona"),
    ("morph00041_045_1k060_deep020_1", "obsidian"),
    ("morph00051_008_2k060_zin2", "obsidian"), ("morph00051_008_2k060_zin2", "inverted@0.04"),
    ("morph00051_008_2k060_D", "obsidian"), ("morph00051_008_2k060_deep020_1", "corona"),
    ("morph00051_008_2k060_deep054_1", "inverted@0.04"),
    ("morph00051_030_3k060_zout", "inverted@0.04"), ("morph00051_030_3k060_zout", "slotcanyon"),
    ("morph00051_030_3k060_zin2", "obsidian"), ("morph00051_030_3k060_deep054_1", "corona"),
    ("morph00041_006_1k060_zin", "obsidian"), ("morph00041_006_1k060_zin", "smooth"),
    ("morph00041_006_1k060_zin2", "slotcanyon"), ("morph00041_006_1k060_D", "obsidian"),
    ("morph00041_006_1k060_D", "corona"),
    ("morph00041_002_1k060_D", "lichtenberg"), ("morph00041_002_1k060_deep054_1", "lichtenberg"),
    ("morph00041_002_1k060_deep054_2", "obsidian"), ("morph00041_002_1k060_deep020_2", "corona"),
]
C = {json.loads(l)["id"]: json.loads(l) for l in open(f"{RD}/candidates.jsonl")}
out = []
for cid, pal in P:
    c = C[cid]
    parent = c["refine_of"]
    v = c["variant"]
    if v.startswith("deeper"):
        desc = f"one minibrot deeper (period {c['period']})" if v.endswith("1") else f"two minibrots deeper (period {c['period']})"
    else:
        desc = VARIANT.get(v, v)
    pname = pal.replace("@", " ")
    out.append({"id": cid, "palette": pal, "title": f"{PARENT_TITLE[parent]}: {desc}, {pname}",
                "blurb": f"refinement of {PARENT_TITLE[parent]}"})
json.dump(out, open(f"{RD}/picks_f.json", "w"), indent=0)
print(len(out))
