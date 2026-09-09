"""Inline viewer_data/*.json into the viewer template and write index.html.

The app must run from file:// with no network, and fetch() is blocked there,
so the data travels inside the document. Images stay as files under
viewer_data/img and load through <img src> with relative paths.

Run:
    python viewer/build_index.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "viewer_data"
TEMPLATE = ROOT / "viewer" / "index.template.html"
OUT = ROOT / "index.html"

FILES = ["manifest", "subjects", "views", "parts", "bmw_sequence", "bmw_views",
         "ps_captures", "measurements", "sensitivity", "compare",
         "geometry_first", "extras", "acquisition"]


def main() -> None:
    bundle = {}
    for name in FILES:
        path = DATA / f"{name}.json"
        bundle[name] = json.loads(path.read_text(encoding="utf-8"))

    # Withhold empty map images. After the refinement pass a filtered map can be
    # genuinely empty, because the gate removed every pixel. Showing black would
    # read as a broken image rather than as the abstention it is, so the layer is
    # withheld and the capture records it. The unfiltered variant is unaffected
    # and remains selectable. Nothing is recomputed.
    from PIL import Image, ImageStat
    withheld = 0
    for c in bundle["ps_captures"]:
        c["withheld_maps"] = {}
        for variant, maps in c["maps"].items():
            for key in list(maps):
                p = DATA / maps[key]
                if (not p.exists()) or ImageStat.Stat(Image.open(p).convert("L")).mean[0] < 3.0:
                    c["withheld_maps"].setdefault(key, []).append(variant)
                    del maps[key]
                    withheld += 1
        if c["withheld_maps"]:
            print(f"  empty after refinement: {c['capture']} {c['camera']} "
                  f"{ {k: v for k, v in c['withheld_maps'].items()} }")
    print(f"  withheld {withheld} empty map images (mean grey < 3/255)")

    # Diagnostics on the joins the front end relies on. Reported, never fixed.
    ps_keys = {f"{c['capture']}::{c['camera']}" for c in bundle["ps_captures"]}
    for p in bundle["parts"]:
        ps = p["photometric_stereo"]
        if ps.get("state") in ("valid", "partial", "invalid"):
            key = f"{ps.get('source_capture')}::{ps.get('source_camera')}"
            print(f"  part PS  {p['vehicle']:8s} {p['part']:18s} {ps['state']:8s} "
                  f"{key[:60]:60s} maps={'yes' if key in ps_keys else 'NO'}")

    payload = json.dumps(bundle, separators=(",", ":"), ensure_ascii=False)
    payload = payload.replace("</", "<\\/")   # never terminate the script tag

    html = TEMPLATE.read_text(encoding="utf-8")
    marker = "/*__DATA__*/"
    assert marker in html, "template is missing the data marker"
    html = html.replace(marker, payload, 1)
    OUT.write_text(html, encoding="utf-8")

    html_bytes = OUT.stat().st_size
    img_bytes = sum(f.stat().st_size for f in (DATA / "img").rglob("*") if f.is_file())
    print(f"\nindex.html      {html_bytes/1e6:6.2f} MB  (inlined JSON {len(payload)/1e6:.2f} MB)")
    print(f"viewer_data/img {img_bytes/1e6:6.2f} MB")
    print(f"total payload   {(html_bytes+img_bytes)/1e6:6.2f} MB")


if __name__ == "__main__":
    main()
