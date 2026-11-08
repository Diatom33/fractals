"""Render each candidate small in one palette and tile into labeled contact sheets.

Usage: screen.py round_dir [palette] [w h]
Reads round_dir/candidates.jsonl, writes round_dir/screen/<id>.png and sheets/sheet_NN.jpg.
"""
import json
import sys
from pathlib import Path

import fr

TYPE_FLAG = {"mandelbrot": "mandelbrot", "multibrot": "multibrot", "julia": "julia", "burningship": "burningship",
             "tricorn": "tricorn", "celtic": "celtic", "perpendicular": "perpendicular",
             "buffalo": "buffalo"}


def render_cand(c, out, palette, w, h, extra=(), pparam=None, iters=None, pparam2=None):
    # "palette@p1[@p2]" selects palette slider values, e.g. "inverted@0.15", "lichtenberg@10@30"
    if "@" in palette:
        palette, *vs = palette.split("@")
        pparam = float(vs[0])
        if len(vs) > 1:
            pparam2 = float(vs[1])
    return fr.render(out, type=TYPE_FLAG[c["type"]], palette=palette, cre=c["cre"], cim=c["cim"],
                     zoom=c["zoom"], iters=iters or c["iters"], w=w, h=h,
                     julia=c.get("julia"), power=c.get("power"), extra=extra, pparam=pparam, pparam2=pparam2)


def main():
    rd = Path(sys.argv[1])
    palette = sys.argv[2] if len(sys.argv) > 2 else "smooth"
    w, h = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (512, 384)
    cands = [json.loads(l) for l in open(rd / "candidates.jsonl")]
    sd = rd / "screen"; sd.mkdir(exist_ok=True)
    done = []
    for i, c in enumerate(cands):
        out = sd / f"{c['id']}.png"
        try:
            if not out.exists():
                render_cand(c, out, palette, w, h)
            done.append(c["id"])
        except Exception as e:
            print("FAIL", c["id"], str(e)[:200], flush=True)
        if i % 25 == 0:
            print(f"{i}/{len(cands)}", flush=True)
    sh = rd / "sheets"; sh.mkdir(exist_ok=True)
    for k in range(0, len(done), 20):
        ids = done[k:k + 20]
        fr.contact_sheet([sd / f"{x}.png" for x in ids], sh / f"sheet_{k // 20:02d}.jpg",
                         labels=ids, cols=5, tile=(320, 240))
    print("sheets", (len(done) + 19) // 20, flush=True)


if __name__ == "__main__":
    main()
