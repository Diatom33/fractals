"""One contact sheet per hexmorph candidate: its glade previews at the six log-range fractions.

Usage: hexsheets.py hex_dir   -> hex_dir/sheets/<id>.png
"""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

d = Path(sys.argv[1])
(d / "sheets").mkdir(exist_ok=True)
for line in open(d / "candidates.jsonl"):
    c = json.loads(line)
    fs = sorted(d.glob(f"{c['id']}_f*.png"))
    if not fs:
        continue
    T = 400
    S = Image.new("RGB", (3 * T, 2 * (300 + 16)), "white")
    for i, f in enumerate(fs):
        im = Image.open(f).convert("RGB").resize((T, 300), Image.LANCZOS)
        x, y = (i % 3) * T, (i // 3) * 316
        S.paste(im, (x, y + 16))
        ImageDraw.Draw(S).text((x + 4, y + 2), f"f{i} frac {c['fracs'][i]}", fill="black")
    S.save(d / "sheets" / f"{c['id']}.png")
    print(c["id"])
