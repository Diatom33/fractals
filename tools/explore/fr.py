"""Thin driver around the fractals export CLI."""
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BIN = ROOT / "target" / "release" / "fractals"
OUT = Path.home() / "Pictures" / "fractals" / "explore"

PALETTES = ["classic", "oklab", "smooth", "mono", "thinfilm", "aurora", "storm", "canopy",
            "biolum", "steve", "inverted", "obsidian", "noctilucent", "lichtenberg",
            "corona", "slotcanyon"]


def render(out, *, type="mandelbrot", palette="biolum", cre=None, cim=None, zoom=None,
           bounds=None, iters=5000, w=640, h=480, julia=None, relax=None, degree=None,
           power=None, pparam=None, pparam2=None, extra=(), timeout=600):
    cmd = [str(BIN), "--export", str(out), "--type", type, "--palette", palette,
           "--iter", str(int(iters)), "--width", str(w), "--height", str(h)]
    if cre is not None:
        cmd += ["--center-re", str(cre), "--center-im", str(cim), "--zoom", str(zoom)]
    if bounds is not None:
        cmd += ["--bounds", ",".join(str(b) for b in bounds)]
    if julia is not None:
        cmd += ["--julia-re", str(julia[0]), "--julia-im", str(julia[1])]
    if relax is not None:
        cmd += ["--relaxation", str(relax)]
    if degree is not None:
        cmd += ["--degree", str(degree)]
    if power is not None:
        cmd += ["--power", str(power)]
    if pparam is not None:
        cmd += ["--palette-param", str(pparam)]
    if pparam2 is not None:
        cmd += ["--palette-param-2", str(pparam2)]
    cmd += list(extra)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0 or not os.path.exists(out):
        raise RuntimeError(f"render failed: {' '.join(cmd)}\n{r.stdout[-500:]}\n{r.stderr[-500:]}")
    return out


def contact_sheet(paths, out, labels=None, cols=4, tile=(400, 300)):
    from PIL import Image, ImageDraw
    rows = (len(paths) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tile[0] + (cols - 1) * 6, rows * tile[1] + (rows - 1) * 6), (255, 0, 255))
    d = ImageDraw.Draw(sheet)
    for i, p in enumerate(paths):
        im = Image.open(p).convert("RGB").resize(tile, Image.LANCZOS)
        x, y = (i % cols) * (tile[0] + 6), (i // cols) * (tile[1] + 6)
        sheet.paste(im, (x, y))
        lab = labels[i] if labels else Path(p).stem
        d.rectangle([x, y, x + 7 * len(lab) + 10, y + 16], fill=(0, 0, 0))
        d.text((x + 5, y + 3), lab, fill=(255, 255, 255))
    sheet.save(out, quality=88)
    return out
