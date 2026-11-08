"""Round 16: odd/other-degree polygons from z^d + c, plus the 6-spoke pinwheel and the loved
heptagon rosette in other palettes.

Usage: mkpicks16.py [multi7.json multi6.json ...]  (extra multimorph results to include)
"""
import json
import sys

EX = "/home/diatom/Pictures/fractals/explore/"
J = {json.loads(l)["id"]: json.loads(l) for l in open(EX + "round15j/candidates.jsonl")}
C = {}
P = []


def add(c, pal, title, blurb):
    C[c["id"]] = dict(c, pool="r16")
    P.append(dict(id=c["id"], palette=pal, title=title, blurb=blurb))


for pal, lab in (("hoarfrost", "hoarfrost"), ("lichtenberg@16@30", "lichtenberg 16 / 30"), ("kintsugi", "kintsugi"),
                 ("corona", "corona"), ("verdigris", "verdigris"), ("halo", "halo")):
    add(J["jn61_p62_k060"], pal, f"Hexagonal pinwheel, {lab}", "the 6-spoke junction pinwheel in other palettes")
for pal, lab in (("hoarfrost", "hoarfrost"), ("lichtenberg@16@30", "lichtenberg 16 / 30"), ("kintsugi", "kintsugi"),
                 ("verdigris", "verdigris"), ("halo", "halo")):
    add(J["jn71_p48_k060"], pal, f"Heptagon rosette I, {lab}", "the loved 7-fold rosette in other palettes")
add(J["jn71_p48_k025"], "glade@7@24", "Heptagon rosette I, closer, glade", "loved 7-fold rosette, reframed")
add(J["jn71_p48_k150"], "glade@7@24", "Heptagon rosette I, wider, glade", "loved 7-fold rosette, reframed")

# z^3 + c: nested minibrots, symmetry x3 per level (triangles -> 9-fold)
cubic = json.load(open(EX + "multi3c/d3.json"))
NAMES = {(5, 2): "Cubic triangle", (5, 3): "Cubic nine-star", (1, 3): "Cubic nine-ring", (3, 3): "Cubic lace"}
for (w, l), name in NAMES.items():
    r = next(x for x in cubic if x["walk"] == w and x["level"] == l)
    for k in ((6, 15, 40) if (w, l) == (5, 2) else (6, 15)):
        c = dict(type="multibrot", power=3, cre=r["cre"], cim=r["cim"], zoom=r["size"] * k,
                 iters=int(8000 + 30 * r["period"]), julia=None, id=f"mb3_w{w}l{l}_k{k:03d}")
        add(c, "glade@7@24", f"{name} (z^3 + c, level {l}), {k} minibrot widths, glade",
            "z^3 + c: each nested minibrot multiplies the ring symmetry by 3")
    c = dict(type="multibrot", power=3, cre=r["cre"], cim=r["cim"], zoom=r["size"] * 15,
             iters=int(8000 + 30 * r["period"]), julia=None, id=f"mb3_w{w}l{l}_k015")
    P.append(dict(id=c["id"], palette="hoarfrost", title=f"{name} (z^3 + c, level {l}), hoarfrost",
                  blurb="z^3 + c nested minibrot"))

for path in sys.argv[1:]:
    rows = json.load(open(path))
    d = int(path.split("/d")[-1].split(".")[0])
    best = sorted(rows, key=lambda r: -r["level"])[:4]
    for r in best:
        for k in (8, 25):
            c = dict(type="multibrot", power=d, cre=r["cre"], cim=r["cim"], zoom=r["size"] * k,
                     iters=int(8000 + 30 * r["period"]), julia=None, id=f"mb{d}_w{r['walk']}l{r['level']}_k{k:03d}")
            add(c, "glade@7@24", f"z^{d} + c minibrot, level {r['level']}, {k} widths, glade",
                f"z^{d} + c: rings with {d}-fold symmetry")

with open(EX + "round16/candidates.jsonl", "w") as f:
    for c in C.values():
        f.write(json.dumps(c) + "\n")
json.dump(P, open(EX + "round16/picks_s.json", "w"), indent=0)
print(len(P))
