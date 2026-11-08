"""Round-4 (trees) picks with morph-path blurbs."""
import json

RD = "/home/diatom/Pictures/fractals/explore/round4"
P = [
    ("003_3", "obsidian", "Burning Thicket"), ("003_3", "corona", "Burning Thicket, neon"),
    ("004_4", "biolum", "Kelp Forest"),
    ("008_2", "corona", "Spiral Canopy"), ("008_2", "obsidian", "Spiral Canopy, ember"),
    ("009_1", "inverted@0.15", "Seed Heads"), ("009_1", "biolum", "Seed Heads, deep sea"),
    ("009_2", "obsidian", "Dandelion"),
    ("009_3", "corona", "Dandelion II"), ("009_3", "inverted@0.08", "Dandelion II, inverted"),
    ("009_4", "lichtenberg", "Dandelion III"), ("009_4", "obsidian", "Dandelion III, ember"),
    ("030_2", "inverted@0.08", "Root System"),
    ("030_3", "inverted@0.08", "Root System II"), ("030_3", "corona", "Root System II, neon"),
    ("031_1", "inverted@0.08", "Lone Tree"), ("031_1", "lichtenberg", "Lone Tree, gold"),
    ("031_2", "inverted@0.15", "Lone Tree II"), ("031_2", "obsidian", "Lone Tree II, ember"),
    ("031_3", "corona", "Lone Tree III"),
    ("036_1", "corona", "Compass Rose"), ("036_1", "inverted@0.08", "Compass Rose, inverted"),
    ("036_2", "lichtenberg", "Compass Rose II"), ("036_2", "inverted@0.15", "Compass Rose II, inverted"),
    ("038_3", "obsidian", "Palm Ring"),
    ("056_2", "inverted@0.08", "Palm Clock"),
    ("056_3", "corona", "Palm Clock II"), ("056_3", "obsidian", "Palm Clock II, ember"),
    ("056_4", "biolum", "Palm Clock III"), ("056_4", "inverted@0.08", "Palm Clock III, inverted"),
]
C = {json.loads(l)["id"]: json.loads(l) for l in open(f"{RD}/candidates.jsonl")}
out = []
for k, pal, title in P:
    cid = f"morph00051_{k}k060"
    c = C[cid]
    lvl = c["level"]
    path = c["path"][:lvl]
    blurb = (f"{lvl} minibrot{'s' if lvl > 1 else ''} deep from the {c['seed_region'].replace('-', ' ')} "
             f"(periods {' → '.join(str(p) for p in path)})")
    out.append({"id": cid, "palette": pal, "title": title, "blurb": blurb})
json.dump(out, open(f"{RD}/picks_e.json", "w"), indent=0)
print(len(out))
