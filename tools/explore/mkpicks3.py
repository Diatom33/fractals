"""Write round-3 picks with titles and morph-path blurbs."""
import json

RD = "/home/diatom/Pictures/fractals/explore/round3"
P = [
    ("002_1", "inverted@0.15", "Elephant Star"), ("002_1", "lichtenberg", "Elephant Star, gold"),
    ("002_2", "biolum", "Elephant Star II"),
    ("003_3", "inverted@0.08", "Tree Sun"), ("003_3", "corona", "Tree Sun, neon"),
    ("003_4", "inverted@0.08", "Tree Sun II"), ("003_4", "corona", "Tree Sun II, neon"), ("003_4", "lichtenberg", "Tree Sun II, gold"),
    ("006_1", "corona", "Spiral Crossing"), ("006_1", "lichtenberg", "Spiral Crossing, gold"),
    ("011_2", "lichtenberg", "Spiral Star"), ("011_2", "biolum", "Spiral Star, deep sea"),
    ("011_4", "corona", "Spiral Star IV"), ("011_4", "obsidian", "Spiral Star IV, ember"),
    ("014_3", "corona", "Snowflake"), ("014_3", "biolum", "Snowflake, deep sea"),
    ("019_4", "lichtenberg", "Elephant Pinwheel"), ("019_4", "corona", "Elephant Pinwheel, neon"),
    ("041_2", "corona", "Frost Veins"),
    ("044_1", "lichtenberg", "Fern Mandala"),
    ("044_2", "corona", "Fern Mandala II"), ("044_2", "lichtenberg", "Fern Mandala II, gold"),
    ("045_1", "lichtenberg", "Palm Wheel"),
    ("045_3", "obsidian", "Palm Wheel III"), ("045_3", "inverted@0.08", "Palm Wheel III, inverted"),
    ("048_2", "obsidian", "Ember Star"),
    ("049_1", "corona", "Lantern Star"),
    ("052_2", "lichtenberg", "Fern Crown"),
    ("054_2", "corona", "Fern Rosette"), ("054_2", "inverted@0.08", "Fern Rosette, inverted"),
    ("057_3", "obsidian", "Spiral Galaxy"),
    ("060_3", "biolum", "Fern Bloom"),
    ("061_4", "corona", "Ringed Island"),
    ("064_2", "obsidian", "Fern Wheel"), ("064_2", "inverted@0.08", "Fern Wheel, inverted"),
    ("066_2", "corona", "Snow Star"),
    ("079_3", "biolum", "Coral Ring"),
    ("081_1", "corona", "Spiral Cross"),
    ("081_3", "biolum", "Spiral Halo"), ("081_3", "inverted@0.15", "Spiral Halo, inverted"),
]
C = {json.loads(l)["id"]: json.loads(l) for l in open(f"{RD}/candidates.jsonl")}
out = []
for k, pal, title in P:
    cid = f"morph00041_{k}k060"
    c = C[cid]
    lvl = c["level"]
    path = c["path"][:lvl]
    region = c["seed_region"].replace("fav-", "your favorite ")
    blurb = (f"{lvl} minibrot{'s' if lvl > 1 else ''} deep from {region} "
             f"(periods {' → '.join(str(p) for p in path)})")
    out.append({"id": cid, "palette": pal, "title": title, "blurb": blurb})
json.dump(out, open(f"{RD}/picks_d.json", "w"), indent=0)
print(len(out))
