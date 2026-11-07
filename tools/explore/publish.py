"""Render chosen (candidate, palette) pairs for the gallery and update gallery.json.

Usage: publish.py round_dir round_title picks.json
picks.json: [{"id": "...", "palette": "biolum", "title": "...", "blurb": "...", "pparam": null}, ...]
Images: gallery/img/<id>_<palette>.jpg (1600x1200) and gallery/img/t_<id>_<palette>.jpg (800x600).
"""
import json
import sys
from pathlib import Path

from PIL import Image

from screen import render_cand

GAL = Path(__file__).resolve().parent / "gallery"


def main():
    rd = Path(sys.argv[1]); round_title = sys.argv[2]
    picks = json.load(open(sys.argv[3]))
    cands = {c["id"]: c for c in map(json.loads, open(rd / "candidates.jsonl"))}
    (GAL / "img").mkdir(parents=True, exist_ok=True)
    gj = GAL / "gallery.json"
    gallery = json.load(open(gj)) if gj.exists() else {"rounds": []}
    rnd = next((r for r in gallery["rounds"] if r["title"] == round_title), None)
    if rnd is None:
        rnd = {"title": round_title, "items": []}
        gallery["rounds"].append(rnd)
    have = {it["id"] for it in rnd["items"]}
    tmp = rd / "render_tmp.png"
    for p in picks:
        c = cands[p["id"]]
        key = f"{p['id']}_{p['palette']}"
        big, thumb = GAL / "img" / f"{key}.jpg", GAL / "img" / f"t_{key}.jpg"
        if not big.exists():
            render_cand(c, tmp, p["palette"], 1600, 1200, extra=["--ss", "2"], pparam=p.get("pparam"))
            im = Image.open(tmp).convert("RGB")
            im.save(big, quality=90)
            im.resize((800, 600), Image.LANCZOS).save(thumb, quality=86)
        if key in have:
            continue
        rnd["items"].append({
            "id": key, "src": f"img/{key}.jpg", "thumb": f"img/t_{key}.jpg",
            "type": c["type"], "palette": p["palette"], "cre": c["cre"], "cim": c["cim"],
            "zoom": c["zoom"], "iters": c["iters"], "julia": c.get("julia"),
            "pparam": p.get("pparam"), "title": p.get("title"), "blurb": p.get("blurb"),
        })
        print("added", key, flush=True)
    json.dump(gallery, open(gj, "w"), indent=1)


if __name__ == "__main__":
    main()
