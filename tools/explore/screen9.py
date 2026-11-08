"""Screen round-9 (Opus-inspired) candidates.

Usage: screen9.py garden|atrium|island [palette]
garden: renders canopy (trap 3, the pink one) and scores a long dark winding channel (the stream).
atrium: renders entrance + atrium frames side by side, in pair order.
island: renders and sorts by search score.
Writes round9/candidates.jsonl (all modes merged) and round9/sheets_<mode>/.
"""
import json
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

import fr
from screen import render_cand

RD = fr.OUT / "round9"
mode = sys.argv[1]
pal = sys.argv[2] if len(sys.argv) > 2 else {"garden": "canopy", "atrium": "smooth", "island": "smooth"}[mode]
C = [json.loads(l) for l in open(RD / f"{mode}.jsonl")]
sd = RD / f"screen_{mode}_{pal.replace('@', '-')}"; sd.mkdir(exist_ok=True)


def stream_score(p):
    a = np.asarray(Image.open(p).convert("RGB")).astype(np.float32) / 255
    L = a @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    interior = a.max(2) == 0
    dark = (L < np.percentile(L[~interior], 30)) & ~interior
    dark = ndimage.binary_opening(dark, iterations=1)
    lab, n = ndimage.label(dark)
    if not n:
        return 0.0
    sizes = np.bincount(lab.ravel())[1:]
    k = int(np.argmax(sizes)) + 1
    ys, xs = np.nonzero(lab == k)
    P = np.stack([xs, ys], 1).astype(float); P -= P.mean(0)
    ev, evec = np.linalg.eigh(np.cov(P.T))
    u = P @ evec[:, -1]
    extent = (u.max() - u.min()) / a.shape[1]
    area = sizes[k - 1] / dark.size
    thin = area / max(extent, 1e-3)
    s = extent * 3 - abs(thin - 0.12) * 6 - interior.mean() * 4
    return float(s)


rows = []
for c in C:
    p = sd / f"{c['id']}.png"
    try:
        if not p.exists():
            render_cand(c, p, pal, 320, 240)
    except Exception as e:
        print("FAIL", c["id"], str(e)[:120], flush=True)
        continue
    if mode == "garden":
        c["stream"] = round(stream_score(p), 3)
    rows.append(c)
if mode == "garden":
    rows.sort(key=lambda c: -c["stream"])
elif mode == "island":
    rows.sort(key=lambda c: -c["score"])
cols = 3 if mode == "atrium" else 6
sh = RD / f"sheets_{mode}"; sh.mkdir(exist_ok=True)
for f in sh.glob("*.jpg"):
    f.unlink()
per = 30
for k in range(0, len(rows), per):
    ch = rows[k:k + per]
    fr.contact_sheet([sd / f"{c['id']}.png" for c in ch], sh / f"sheet_{k // per:02d}.jpg",
                     labels=[f"{c['id']} {c.get('stream', c.get('score', ''))} {c['zoom']:.0e}" for c in ch],
                     cols=cols, tile=(260, 195))
allc = {}
if (RD / "candidates.jsonl").exists():
    allc = {json.loads(l)["id"]: json.loads(l) for l in open(RD / "candidates.jsonl")}
for c in rows:
    c["pool"] = "opus"
    allc[c["id"]] = c
with open(RD / "candidates.jsonl", "w") as f:
    for c in allc.values():
        f.write(json.dumps(c) + "\n")
print(mode, len(rows), flush=True)
