"""Prepare a gallery publish: archive "no"-rated items (into gallery_archive_no.json, their files
removed from the artifact), and write the Artifact `files` map for a round.

Usage: galfiles.py ratings_dir "round title prefix" out_files.json
Prints the number of files the published artifact will reference (cap 512 incl. index.html).
"""
import json
import sys
from pathlib import Path

GAL = Path(__file__).resolve().parent / "gallery"


def ratings(d):
    docs = {}
    for p in list((Path(d) / "ratings").glob("*.json")) + list(Path(d).glob("*.json")):
        r = json.load(open(p))
        if p.stem not in docs or r.get("t", 0) > docs[p.stem].get("t", 0):
            docs[p.stem] = r
    return docs


def main():
    R = ratings(sys.argv[1])
    prefix, out = sys.argv[2], sys.argv[3]
    g = json.load(open(GAL / "gallery.json"))
    arch = json.load(open(GAL / "gallery_archive_no.json"))
    files = {"gallery.json": "gallery.json"}
    for rnd in g["rounds"]:
        keep, gone = [], []
        for it in rnd["items"]:
            (gone if R.get(it["id"], {}).get("r") == "no" else keep).append(it)
        if gone:
            ar = next((r for r in arch["rounds"] if r["title"] == rnd["title"]), None)
            if ar is None:
                ar = {"title": rnd["title"], "items": []}
                arch["rounds"].append(ar)
            ar["items"] += gone
            for it in gone:
                for f in {it["src"], it["thumb"]}:
                    files[f] = None
            print(f"archived {len(gone)} from {rnd['title'][:40]}")
        rnd["items"] = keep
    new = next(r for r in g["rounds"] if r["title"].startswith(prefix))
    for it in new["items"]:
        for f in {it["src"], it["thumb"]}:
            files[f] = f
    json.dump(g, open(GAL / "gallery.json", "w"), indent=1)
    json.dump(arch, open(GAL / "gallery_archive_no.json", "w"), indent=1)
    json.dump(files, open(out, "w"), indent=0)
    refs = {f for r in g["rounds"] for it in r["items"] for f in (it["src"], it["thumb"])}
    print("referenced files:", len(refs) + 2)


if __name__ == "__main__":
    main()
