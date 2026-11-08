"""Filter round-8 screens by simple image stats and re-sheet the survivors.

Usage: filter8.py palette
"""
import sys

import numpy as np
from PIL import Image

import fr

pal = sys.argv[1]
RD = fr.OUT / "round8"
sd = RD / f"screen_{pal}"
rows = []
for p in sorted(sd.glob("*.png")):
    a = np.asarray(Image.open(p).convert("RGB")).astype(np.float32) / 255
    L = a @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    interior = float((a.max(2) == 0).mean())
    dark = float(((L < (0.35 if pal == "canopy" else 0.18)) & (a.max(2) > 0)).mean())
    std = float(L.std())
    gy, gx = np.gradient(L)
    edge = float((np.hypot(gx, gy) > 0.08).mean())
    rows.append((p.stem, interior, dark, std, edge))
keep = []
for cid, interior, dark, std, edge in rows:
    if interior > 0.3 or edge > 0.35:
        continue
    if pal == "canopy" and not (interior < 0.2 and dark > 0.12):
        continue
    if pal == "noctilucent" and not (std > 0.07):
        continue
    keep.append((cid, interior, dark, std, edge))
keep.sort(key=lambda r: -(r[2] if pal == "canopy" else r[3]))
print(len(rows), "->", len(keep))
sh = RD / f"filt_{pal}"; sh.mkdir(exist_ok=True)
for f in sh.glob("*.jpg"):
    f.unlink()
for k in range(0, len(keep), 30):
    ids = [r[0] for r in keep[k:k + 30]]
    fr.contact_sheet([sd / f"{x}.png" for x in ids], sh / f"sheet_{k // 30:02d}.jpg",
                     labels=ids, cols=6, tile=(260, 195))
open(sh / "ids.txt", "w").write("\n".join(r[0] for r in keep))
