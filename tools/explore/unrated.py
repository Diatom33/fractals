"""Count gallery items with no rating yet.

Usage: unrated.py ratings_dir   (an ArtifactData out_dir dump: ratings_dir/ratings/<id>.json)
Prints "unrated N of M" and the unrated ids by round; exit code 0 when everything is rated, 1 otherwise.
"""
import json
import sys
from pathlib import Path

GAL = Path(__file__).resolve().parent / "gallery" / "gallery.json"
rd = Path(sys.argv[1]) / "ratings"
rated = {p.stem for p in rd.glob("*.json") if json.load(open(p)).get("r")}
g = json.load(open(GAL))
missing = [(r["title"], it["id"]) for r in g["rounds"] for it in r["items"] if it["id"] not in rated]
total = sum(len(r["items"]) for r in g["rounds"])
print(f"unrated {len(missing)} of {total}")
for t, i in missing[:40]:
    print(f"  {t[:30]} | {i}")
sys.exit(0 if not missing else 1)
