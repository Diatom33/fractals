"""Make per-image QA sheets for a gallery round: the full frame downscaled to 800x600 plus two 1:1
crops from the 1600x1200 render (busiest region: grain/noise/aliasing; smoothest region: banding,
seams). Usage: qa.py round_title_prefix|picks.json out_dir   -> out_dir/<item id>.png, prints the list.
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

GAL = Path(__file__).resolve().parent / "gallery"
CW, CH = 400, 300


def crops(im):
    L = np.asarray(im.convert("L")).astype(np.float32) / 255
    gy, gx = np.gradient(L)
    g = np.hypot(gx, gy)
    best, worst = None, None
    for y in range(0, im.height - CH + 1, 150):
        for x in range(0, im.width - CW + 1, 200):
            e = float(g[y:y + CH, x:x + CW].mean())
            if best is None or e > best[0]:
                best = (e, x, y)
            if worst is None or e < worst[0]:
                worst = (e, x, y)
    return [im.crop((x, y, x + CW, y + CH)) for _, x, y in (best, worst)]


def main():
    prefix, od = sys.argv[1], Path(sys.argv[2])
    od.mkdir(parents=True, exist_ok=True)
    if prefix.endswith(".json"):
        keys = [f"{p['id']}_{p['palette'].replace('@', '-')}" for p in json.load(open(prefix))]
        items = [{"id": k, "src": f"img/{k}.jpg"} for k in keys]
    else:
        g = json.load(open(GAL / "gallery.json"))
        items = next(r for r in g["rounds"] if r["title"].startswith(prefix))["items"]
    for it in items:
        im = Image.open(GAL / it["src"]).convert("RGB")
        if im.width < 1600:
            print("SKIP (thumbnail only)", it["id"]); continue
        sheet = Image.new("RGB", (800 + 6 + CW, 600), (40, 40, 40))
        sheet.paste(im.resize((800, 600), Image.LANCZOS), (0, 0))
        busy, calm = crops(im)
        sheet.paste(busy, (806, 0)); sheet.paste(calm, (806, 300))
        d = ImageDraw.Draw(sheet)
        d.text((812, 4), "1:1 busiest", fill=(255, 255, 0)); d.text((812, 304), "1:1 smoothest", fill=(255, 255, 0))
        out = od / f"{it['id']}.png"
        sheet.save(out)
        print(out)


if __name__ == "__main__":
    main()
