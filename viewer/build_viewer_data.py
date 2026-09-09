"""Build the frozen data bundle for the SmartGate inspection viewer.

Reads the completed experiment outputs and emits `viewer_data/`, a self
contained folder of JSON plus web-sized JPEG that the front end codes against.
Nothing here recomputes a result. Every number is copied from a stored output
or transcribed from a table in the writeups, and the transcribed values carry
their source in a `source` field so they can be checked.

Run:
    python viewer/build_viewer_data.py
"""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np
from matplotlib import colormaps
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent

# The indexed RGB sequences are the all-LED, 30-degree acquisition described in
# writeups/Experiment_AllLED_RGB_Reevaluation.md. The earlier externally relit
# 45-degree run remains on disk under report_figures/multiview_fusion and
# report_figures/bmw_sequence as the illumination-limited baseline; it is not
# deleted. It is carried here as measured numbers rather than as a second set
# of images, because two full sequences of imagery do not fit the payload
# ceiling and the report now reports the all-LED sequence.
MVF = ROOT / "report_figures" / "multiview_fusion_allled_rr30"
BMW30 = ROOT / "report_figures" / "multiview_fusion_allled_bmw30"
CMP = ROOT / "report_figures" / "multiview_fusion_allled_comparison"
LEGACY_MVF = ROOT / "report_figures" / "multiview_fusion"
PSFIG = ROOT / "Photometric Stereo" / "report_figures"
GEO = ROOT / "report_figures" / "geometry_first_proposals"
OUT = ROOT / "viewer_data"
IMG = OUT / "img"

# Sizes are tuned per category. The photometric stereo maps and the BMW panels
# are what a reader zooms into, so they keep full resolution. The 144 indexed
# sequence views are navigational and carry a lower budget; the part overlay is
# flat colour and tolerates heavier compression than the photographs do.
#
# The bundle no longer targets the old 16 MB hosting ceiling. It runs from
# file:// off local disk, so folder size costs nothing at runtime, and the
# all-LED sequence is both 50% larger in view count and genuinely brighter,
# carrying detail where the relit frames carried black. Holding 16 MB would
# now mean compressing the figures the report is going to print. The images
# are instead kept at a quality that survives a report figure, and the bundle
# lands near 21 MB.
STAGE_MAX = 1400          # longest edge for a stage image
STAGE_Q = 80
THUMB_MAX = 150
THUMB_Q = 62
RR_MAX, RR_Q = 820, 65           # indexed sequence photograph and damage overlay
RR_PARTS_MAX, RR_PARTS_Q = 700, 55  # flat-colour part overlay
ZOOM_MAX, ZOOM_Q = 1500, 84       # normal maps and the registered photograph
MAP_Q = 78                        # derived colour fields, smooth, compress well
BMW_MAX, BMW_Q = 1250, 80         # BMW close-up frames and their overlays

written: list[tuple[str, int]] = []
missing: list[str] = []


def content_box(src: Path, thresh: int = 10, pad_frac: float = 0.02):
    """Bounding box of the non-background content in a photometric stereo map.

    The solver writes its output into a full-frame image with everything
    outside the solved region left black, so a panel can occupy a fifth of the
    frame. Cropping to the content is what makes the map worth zooming into.
    """
    if not src.exists():
        return None
    with Image.open(src) as im:
        g = im.convert("L")
        bbox = g.point(lambda v: 255 if v > thresh else 0).getbbox()
    if not bbox:
        return None
    x0, y0, x1, y1 = bbox
    pad = round(pad_frac * max(x1 - x0, y1 - y0))
    return (max(0, x0 - pad), max(0, y0 - pad),
            min(g.width, x1 + pad), min(g.height, y1 + pad))


def emit(src: Path, rel: str, max_edge: int = STAGE_MAX, q: int = STAGE_Q,
         crop=None) -> str | None:
    """Downscale `src` to `IMG/rel` as JPEG. Returns the relative path or None.

    `crop` is applied before scaling. Every map belonging to one capture is
    given the same box, so switching layer holds the viewport steady.
    """
    if not src.exists():
        missing.append(str(src.relative_to(ROOT)) if src.is_relative_to(ROOT) else str(src))
        return None
    dst = IMG / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        im = im.convert("RGB")
        if crop:
            im = im.crop(crop)
        if max(im.size) > max_edge:
            scale = max_edge / max(im.size)
            im = im.resize((round(im.width * scale), round(im.height * scale)),
                           Image.LANCZOS)
        im.save(dst, "JPEG", quality=q, optimize=True, progressive=True)
    written.append((f"img/{rel}", dst.stat().st_size))
    return f"img/{rel}"


def jdump(obj, name: str) -> None:
    path = OUT / name
    path.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    written.append((name, path.stat().st_size))


# --------------------------------------------------------------------------
# 1. Range Rover multi-view sequence
# --------------------------------------------------------------------------

def read_view_records(root: Path, prefix: str, eligible_by_view=None):
    """Map one `view_records.jsonl` onto the viewer's view schema.

    The all-LED Range Rover and damaged BMW runs write the same record shape,
    so both indexed sequences are read through here and the interface treats
    them alike. `eligible_by_view` is supplied only where a part-level summary
    exists to say which parts a view contributed to.
    """
    eligible_by_view = eligible_by_view or {}
    records = root / "results" / "view_records.jsonl"
    if not records.exists():
        records = root / "segmentation" / "view_records.jsonl"

    views = []
    for line in records.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        vid = r["view_id"]
        crop_ok = r.get("crop_source") == "auto"

        dets = []
        for d in r.get("damage_instances", []):
            dets.append({
                "class": d.get("damage_class"),
                "confidence": round(float(d.get("rgb_confidence") or 0.0), 4),
                "area_px": d.get("mask_area_px"),
                "part": d.get("assigned_part"),
                "attributed": d.get("attribution_status") == "attributed",
                "reason": d.get("reason"),
                "bbox": d.get("bbox_crop_xyxy"),
            })

        q = r.get("quality", {})
        views.append({
            "id": vid,
            "vehicle": r["vehicle"],
            "camera": r["camera"],
            "angle": r["angle_deg"],
            "crop_ok": crop_ok,
            "eligible": crop_ok,
            "crop_xyxy": r.get("crop_xyxy"),
            "exposure_scale": r.get("exposure_scale"),
            "quality_raw": {
                "median_brightness": q.get("median_brightness"),
                "saturated_fraction": q.get("saturated_fraction"),
                "sharpness": q.get("sharpness_laplacian_variance"),
                "crop_width": q.get("crop_width"),
                "crop_height": q.get("crop_height"),
            },
            "eligible_parts": sorted(set(eligible_by_view.get(vid, []))),
            "n_parts_detected": len(r.get("parts", [])),
            "part_classes": sorted({p["part"] for p in r.get("parts", [])}),
            "detections": dets,
            "img": {
                # The photograph is cropped to the box the two models actually
                # ran on. The overlays are already written at that crop, so
                # this is what puts all three frame layers on one geometry:
                # switching rgb to damage now holds the viewport steady instead
                # of jumping from the full frame to the crop. It also spends
                # the pixel budget on the vehicle rather than on chamber wall.
                "rgb": emit(root / "rgb" / f"{vid}_rgb.png",
                            f"{prefix}/{vid}_rgb.jpg", RR_MAX, RR_Q,
                            crop=r.get("crop_xyxy")),
                "damage": emit(root / "segmentation" / f"{vid}_damage.png",
                               f"{prefix}/{vid}_damage.jpg", RR_MAX, RR_Q),
                "parts": emit(root / "segmentation" / f"{vid}_parts.png",
                              f"{prefix}/{vid}_parts.jpg", RR_PARTS_MAX, RR_PARTS_Q),
                "thumb": emit(root / "rgb" / f"{vid}_rgb.png",
                              f"{prefix}/{vid}_thumb.jpg", THUMB_MAX, THUMB_Q,
                              crop=r.get("crop_xyxy")),
            },
        })
    return views


def build_range_rover():
    summary = json.loads((MVF / "results" / "part_level_summary.json").read_text())
    part_records = summary["parts"]

    eligible_by_view: dict[str, list[str]] = {}
    for p in part_records:
        for vid in p.get("eligible_view_ids", []):
            eligible_by_view.setdefault(vid, []).append(p["part"])

    views = read_view_records(MVF, "views", eligible_by_view)

    # Where each part is best seen, as a navigation target. This is a second
    # measured quantity, kept separate from `representative_view`: the fusion
    # run picks the representative view on a quality score that weighs exposure,
    # sharpness, area and boundary contact, and the highest-scoring view is
    # sometimes a steep close-up where the part model has collapsed the whole
    # body into one or two classes. That view is a legitimate quality winner and
    # a poor thing to open when a reader clicks a windscreen.
    #
    # The target is therefore the view where the part covers the most pixels
    # among views in which the part model resolved a plausible number of
    # distinct classes. A view carrying eight or more classes has not collapsed.
    # The threshold relaxes if no view clears it, so every part still resolves.
    # Both quantities are stored and both are shown in the record.
    RESOLVED_CLASSES = (8, 6, 0)
    seen_by_part: dict[tuple[str, str], list[tuple[int, int, str]]] = {}
    for line in (MVF / "results" / "view_records.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        n_classes = len({p["part"] for p in r.get("parts", [])})
        for part in r.get("parts", []):
            key = (r["vehicle"], part["part"])
            seen_by_part.setdefault(key, []).append(
                (n_classes, int(part.get("area_px") or 0), r["view_id"]))

    # Trim the part records to the fields the interface uses, keeping the
    # verbatim state_reason strings.
    parts = []
    for p in part_records:
        ps = p.get("photometric_stereo", {})
        eligible = set(p.get("eligible_view_ids", []))
        seen = [t for t in seen_by_part.get((p["vehicle"], p["part"]), [])
                if t[2] in eligible]
        best, min_classes = None, None
        for floor in RESOLVED_CLASSES:
            pool = [t for t in seen if t[0] >= floor]
            if pool:
                best = max(pool, key=lambda t: t[1])
                min_classes = floor
                break
        parts.append({
            "part": p["part"],
            "vehicle": p["vehicle"],
            "decision": p["decision"],
            "review_required": p.get("review_required"),
            "eligible_views": p.get("eligible_views"),
            "eligible_view_ids": p.get("eligible_view_ids", []),
            "representative_view": p.get("representative_view"),
            "representative_view_quality": p.get("representative_view_quality"),
            "representative_view_components": p.get("representative_view_components"),
            "best_seen_view": best[2] if best else None,
            "best_seen_area_px": best[1] if best else None,
            "best_seen_classes_in_view": best[0] if best else None,
            "best_seen_class_floor": min_classes,
            "fixed_view_id": p.get("fixed_view_id"),
            "fixed_view_eligible": p.get("fixed_view_eligible"),
            "angles_seeing_part": p.get("angles_seeing_part", []),
            "cameras_seeing_part": p.get("cameras_seeing_part", []),
            "rgb": p.get("rgb", {}),
            "photometric_stereo": ps,
        })

    return views, parts, summary.get("unattributed_by_vehicle", {})


# --------------------------------------------------------------------------
# 2. Photometric stereo captures, all subjects
# --------------------------------------------------------------------------

# Maps are rendered from the solver's own `.npz` arrays rather than lifted from
# the matplotlib PNGs beside them. The PNGs are inconsistent: `normal_color` is
# written at full frame while the rest are small figure renders with their own
# padding, so no single crop box can serve both and cropping the small ones to
# the full-frame box empties them. The arrays carry `bbox_y0_y1_x0_x1`, which
# gives an exact shared geometry for every layer including the photograph.
#
# key -> (npz field, colormap, range) where range is None for a robust
# percentile stretch, a (lo, hi) pair for a fixed scale, or "signed" for a
# diverging map centred on zero.
# The field name is a list of candidates: the earliest captures predate the
# illumination_support rename and carry `confidence` instead.
PS_ARRAYS = [
    ("support", ["illumination_support", "confidence"], "viridis", (0.0, 1.0)),
    ("usable", ["usable_lights"], "viridis", (0.0, 12.0)),
    ("depth", ["depth"], "RdBu_r", "signed"),
    ("elevation", ["elevation_deg"], "viridis", (0.0, 90.0)),
]
UNSOLVED = (16, 16, 16)   # matches the viewport ground, so masked pixels read as absent

# Captures the viewer shows, with the camera used. Grades come from
# ps_evidence_records.json where present; the state here is the documented
# capture-level verdict from PS_Results.md.
PS_SHOWN = [
    # capture, camera, subject, panel label, state
    ("ps_bmw_white_multidamage_hood_cam3_150k_repeat01", "cam3", "bmw_white",
     "Bonnet, three dents and three scratches", "valid"),
    ("ps_bmw_white_damaged_hood_r000_150k", "cam3", "bmw_white",
     "Bonnet, one dent and one scratch", "valid"),
    ("ps_bmw_white_damaged_doors_r270_150k", "cam3", "bmw_white",
     "Rear door, one dent", "valid"),
    ("ps_bmw_white_r000", "cam3", "bmw_white",
     "Undamaged bonnet, positive control", "valid"),
    ("ps_sg_fdm_tilt_cam3_150k_repeat01", "cam3", "sg_fdm",
     "Tilted artefact, repeat 01", "valid"),
    ("ps_sg_fdm_bare_r000", "cam3", "sg_fdm",
     "Bare artefact, grazing view", "valid"),
    ("ps_sg_fdm_valid_r000", "cam3", "sg_fdm",
     "Same artefact, under-exposed", "invalid"),
    ("ps_sheetmetal_grey_dents_cam3_150k_repeat01", "cam3", "sheet_metal",
     "Painted steel strip, four formed dents", "valid"),
    ("ps_rr_black_damaged_side_r020_600k", "cam2", "range_rover",
     "Damaged front door, gloss black", "partial"),
    ("ps_rr_black_damaged_side_r020_600k", "cam3", "range_rover",
     "Damaged front door, second camera", "invalid"),
    ("ps_rr_black_clean_side_r020_600k", "cam2", "range_rover",
     "Matched clean front door", "invalid"),
    ("ps_rr_black_damaged_cam3_tilt_r000_600k", "cam3", "range_rover",
     "Damaged door, tilted pose", "invalid"),
    # Both of these are cited as the source capture by a Range Rover part
    # record, so the record cannot show its maps unless they are carried here.
    ("ps_rr_black_clean_side_r020_600k", "cam3", "range_rover",
     "Matched clean front door, second camera", "invalid"),
    ("ps_rr_black_clean_cam3_tilt_r000_600k", "cam3", "range_rover",
     "Clean door, tilted pose", "invalid"),
]


def _load_npz(base: Path, cam: str, filtered: bool):
    p = base / f"{cam}_l1{'_filtered' if filtered else ''}_data.npz"
    if not p.exists():
        return None
    try:
        return np.load(p)
    except Exception as exc:                       # corrupt archive, say so
        missing.append(f"{p.name}: {exc}")
        return None


def _render(field: np.ndarray, mask: np.ndarray, cmap_name: str, rng,
            canvas_hw, offset) -> Image.Image:
    """Colour one solver array and place it on the shared canvas."""
    valid = mask & np.isfinite(field)
    if rng == "signed":
        if valid.any():
            lim = float(np.nanpercentile(np.abs(field[valid]), 98)) or 1.0
        else:
            lim = 1.0
        lo, hi = -lim, lim
    elif rng is None:
        lo, hi = (float(np.nanpercentile(field[valid], 2)),
                  float(np.nanpercentile(field[valid], 98))) if valid.any() else (0.0, 1.0)
    else:
        lo, hi = rng
    if hi <= lo:
        hi = lo + 1e-6

    norm = np.clip((field.astype(np.float64) - lo) / (hi - lo), 0.0, 1.0)
    norm[~valid] = 0.0
    rgb = (colormaps[cmap_name](norm)[..., :3] * 255).astype(np.uint8)
    rgb[~valid] = UNSOLVED

    canvas = np.full((canvas_hw[0], canvas_hw[1], 3), UNSOLVED, dtype=np.uint8)
    oy, ox = offset
    h, w = rgb.shape[:2]
    canvas[oy:oy + h, ox:ox + w] = rgb
    return Image.fromarray(canvas)


def build_ps_captures():
    ev_raw = json.loads((MVF / "results" / "ps_evidence_records.json").read_text())
    # Graded records are nested under roi_comparison. The predicted-part region
    # is the production condition, so it is preferred over the manual one.
    ev_index = {}
    for r in ev_raw.get("records", []):
        roi = r.get("roi_comparison") or {}
        graded = roi.get("predicted_roi_evidence") or roi.get("manual_roi_evidence")
        if not graded:
            continue
        key = (r.get("capture_set"), r.get("camera"), bool(r.get("filtered")))
        ev_index[key] = {**graded,
                         "roi_source": "predicted part mask, eroded 2 px"
                         if roi.get("predicted_roi_evidence") else "manual mask",
                         "roi_agreement_iou": roi.get("iou_manual_predicted"),
                         "predicted_part": roi.get("predicted_part")}

    out = []
    for capture, cam, subject, label, state in PS_SHOWN:
        base = PSFIG / capture
        zu, zf = _load_npz(base, cam, False), _load_npz(base, cam, True)
        ref = zu if zu is not None else zf
        if ref is None:
            missing.append(f"{capture}/{cam}_l1[_filtered]_data.npz")
            continue

        # The unfiltered solve covers the wider region, so its bounding box is
        # the shared canvas. Every layer, both variants and the photograph are
        # placed in it, which is what makes the shared zoom viewport exact.
        y0, y1, x0, x1 = (int(v) for v in ref["bbox_y0_y1_x0_x1"])
        canvas_hw = (y1 - y0, x1 - x0)

        maps: dict[str, dict[str, str]] = {"filtered": {}, "unfiltered": {}}
        for variant, z in (("unfiltered", zu), ("filtered", zf)):
            if z is None:
                continue
            by0, by1, bx0, bx1 = (int(v) for v in z["bbox_y0_y1_x0_x1"])
            offset = (by0 - y0, bx0 - x0)
            mask = z["mask"].astype(bool)

            # Normals keep the solver's own colouring, which is the headline
            # result and already reads well, cropped to the shared box.
            src = base / f"{cam}_l1{'_filtered' if variant == 'filtered' else ''}_normal_color.png"
            if src.exists():
                with Image.open(src) as im:
                    full = tuple(int(v) for v in z["full_shape_hw"])
                    if im.size == (full[1], full[0]):
                        nc = im.convert("RGB").crop((x0, y0, x1, y1))
                    else:                       # already a figure render
                        nc = im.convert("RGB").resize((canvas_hw[1], canvas_hw[0]),
                                                      Image.LANCZOS)
                tag = "f" if variant == "filtered" else "u"
                rel = f"ps/{capture}_{cam}_{tag}_normal.jpg"
                (IMG / rel).parent.mkdir(parents=True, exist_ok=True)
                nc.save(IMG / rel, "JPEG", quality=ZOOM_Q, optimize=True)
                written.append((f"img/{rel}", (IMG / rel).stat().st_size))
                maps[variant]["normal"] = f"img/{rel}"

            for key, field_names, cmap_name, rng in PS_ARRAYS:
                field_name = next((f for f in field_names if f in z.files), None)
                if field_name is None:
                    missing.append(f"{capture}/{cam} {variant}: no {field_names[0]}")
                    continue
                im = _render(z[field_name], mask, cmap_name, rng, canvas_hw, offset)
                tag = "f" if variant == "filtered" else "u"
                rel = f"ps/{capture}_{cam}_{tag}_{key}.jpg"
                im.save(IMG / rel, "JPEG", quality=MAP_Q, optimize=True)
                written.append((f"img/{rel}", (IMG / rel).stat().st_size))
                maps[variant][key] = f"img/{rel}"

        # The prepared photograph, cropped to the same box. This is what makes
        # an RGB against normals comparison possible under one viewport.
        photo = PSFIG / "rgb_frames" / f"{cam}_{capture}_rgb.png"
        if photo.exists():
            rel = emit(photo, f"ps/{capture}_{cam}_photo.jpg", ZOOM_MAX, ZOOM_Q,
                       crop=(x0, y0, x1, y1))
            if rel:
                maps["filtered"]["photo"] = maps["unfiltered"]["photo"] = rel
        else:
            missing.append(f"rgb_frames/{cam}_{capture}_rgb.png")

        native = [canvas_hw[1], canvas_hw[0]]

        rec = {
            "capture": capture,
            "native_px": native,
            "camera": cam,
            "subject": subject,
            "label": label,
            "state": state,
            "algorithm": "l1_filtered",
            "maps": maps,
        }
        # attach the graded evidence record when the fusion run produced one
        for cand in ((capture, cam, True), (capture, cam, False)):
            if cand in ev_index:
                rec["evidence"] = ev_index[cand]
                break
        out.append(rec)
    return out


# --------------------------------------------------------------------------
# 3. Ground-truth measurements, transcribed from PS_Results.md
# --------------------------------------------------------------------------

MEASUREMENTS = {
  "sg_fdm": {
    "source": "PS_Results.md section 2.3, three fixed-pose repeats, cam3",
    "reference": "nominal CAD, not measured geometry",
    "summary": {"mae_mm": 0.604, "median_abs_error_mm": 0.545, "rmse_mm": 0.709,
                "mean_per_feature_sd_mm": 0.0043, "max_across_repeat_range_mm": 0.0149,
                "sign_agreement": "7 of 8"},
    "features": [
      {"name": "FDM-C08", "nominal_mm": -0.5594, "recovered_mm": -0.1630, "sd_mm": 0.0008, "abs_error_mm": 0.3964, "sign_correct": True,  "released": True},
      {"name": "FDM-C16", "nominal_mm": -1.1243, "recovered_mm": -0.6712, "sd_mm": 0.0030, "abs_error_mm": 0.4532, "sign_correct": True,  "released": True},
      {"name": "FDM-C30", "nominal_mm": -2.1436, "recovered_mm": -1.5072, "sd_mm": 0.0078, "abs_error_mm": 0.6364, "sign_correct": True,  "released": True},
      {"name": "FDM-X30", "nominal_mm":  2.1436, "recovered_mm":  2.1441, "sd_mm": 0.0046, "abs_error_mm": 0.0005, "sign_correct": True,  "released": True},
      {"name": "FDM-D25", "nominal_mm": -0.8741, "recovered_mm": -0.1470, "sd_mm": 0.0080, "abs_error_mm": 0.7271, "sign_correct": True,  "released": True},
      {"name": "FDM-D30", "nominal_mm": -1.0489, "recovered_mm":  0.0573, "sd_mm": 0.0051, "abs_error_mm": 1.1062, "sign_correct": False, "released": False,
       "rejection_reason": "reconstructed with the wrong sign; concavity not recovered"},
      {"name": "FDM-W0",  "nominal_mm":  1.0000, "recovered_mm":  0.6637, "sd_mm": 0.0042, "abs_error_mm": 0.3363, "sign_correct": True,  "released": True},
      {"name": "FDM-W1",  "nominal_mm":  2.0000, "recovered_mm":  0.8203, "sd_mm": 0.0013, "abs_error_mm": 1.1797, "sign_correct": True,  "released": True}
    ]
  },
  "sheet_metal": {
    "source": "PS_Results.md section 2.4, three fixed-pose repeats, cam3",
    "reference": "calliper depth on hand-formed dents in primed steel",
    "summary": {"valid_only_mae_mm": 0.38, "valid_only_range_mm": [0.29, 0.47],
                "valid_coverage": "3 of 4",
                "mae_if_rejected_dent_forced_mm": 0.94,
                "note": "Forcing dent 4 into the average despite its failed local reconstruction nearly triples the error. Abstention is part of the measurement result."},
    "features": [
      {"name": "Dent 1", "ball_bearing_mm": 8, "calliper_mm": 2.3, "inner_diameter_mm": 5.6, "response_px": 20.13, "response_sd_px": 0.15, "recovered_mm": [2.10, 2.28], "abs_error_mm": [0.02, 0.20], "released": True,  "note": "1% to 9% low"},
      {"name": "Dent 2", "ball_bearing_mm": 7, "calliper_mm": 2.7, "inner_diameter_mm": 5.6, "response_px": 21.87, "response_sd_px": 0.46, "recovered_mm": [2.19, 2.37], "abs_error_mm": [0.33, 0.51], "released": True,  "note": "12% to 19% low, compressed"},
      {"name": "Dent 3", "ball_bearing_mm": 6, "calliper_mm": 2.8, "inner_diameter_mm": 5.0, "response_px": 19.95, "response_sd_px": 0.28, "recovered_mm": [2.09, 2.27], "abs_error_mm": [0.53, 0.71], "released": True,  "note": "19% to 25% low, compressed"},
      {"name": "Dent 4", "ball_bearing_mm": 5, "calliper_mm": 2.4, "inner_diameter_mm": 4.0, "response_px": -1.88, "response_sd_px": 0.13, "recovered_mm": None, "abs_error_mm": None, "released": False,
       "rejection_reason": "integrated response has the wrong sign, about -0.20 to -0.22 mm. The normal anomaly localises the dent but no severity value is released."}
    ]
  },
  "bmw_white": {
    "source": "PS_Results.md sections 3.3 and 3.4",
    "reference": "calliper depth on hand-formed dents in a die-cast bonnet and door",
    "summary": {"multidamage_mae_mm": 0.42, "fixed_pose_sd_mm": 0.01,
                "note": "All three depression polarities are correct and all three are compressed. The dominant limitation is systematic depth compression rather than acquisition noise."},
    "features": [
      {"name": "Bonnet, top left",     "calliper_mm": 0.60, "footprint": "2.2 mm diameter", "local_usable_lights": 11, "normal_departure_deg": [6.30, 8.11], "recovered_mm": 0.39, "sd_mm": 0.01, "abs_error_mm": 0.21, "released": True, "note": "36% low, exploratory"},
      {"name": "Bonnet, top right",    "calliper_mm": 0.90, "footprint": "7.8 by 2.0 mm",   "local_usable_lights": 12, "normal_departure_deg": [5.50, 7.22], "recovered_mm": 0.50, "sd_mm": 0.01, "abs_error_mm": 0.40, "released": True, "note": "44% low, exploratory"},
      {"name": "Bonnet, middle right", "calliper_mm": 0.90, "footprint": "1.9 mm diameter", "local_usable_lights": 10, "normal_departure_deg": [3.84, 6.93], "recovered_mm": 0.25, "sd_mm": 0.01, "abs_error_mm": 0.65, "released": True, "note": "72% low, exploratory"},
      {"name": "Rear door dent",       "calliper_mm": 1.00, "footprint": "5 mm impression", "local_usable_lights": 8,  "normal_departure_deg": [8.10, 11.10], "recovered_mm": 0.068, "sd_mm": None, "abs_error_mm": 0.93, "released": False,
       "rejection_reason": "strict refinement removes the exact centre of the dent. The panel passes the capture-level gate but depth is withheld for this instance."}
    ]
  }
}

# Capture-level support figures, PS_Results.md sections 2.4, 3.3 and 3.4.
CAPTURE_SUPPORT = {
  "ps_bmw_white_damaged_hood_r000_150k":                {"requested_px": 162431, "usable_lights_p10_med_p90": [11, 12, 12], "region_at_least_six": 0.99998, "retained": 0.9764, "state": "valid"},
  "ps_bmw_white_damaged_doors_r270_150k":               {"requested_px": 37485,  "usable_lights_p10_med_p90": [6, 8, 8],    "region_at_least_six": 0.9882,  "retained": 0.8004, "state": "valid"},
  "ps_bmw_white_multidamage_hood_cam3_150k_repeat01":   {"requested_px": 139069, "usable_lights_p10_med_p90": [10, 11, 12], "region_at_least_six": 0.99935, "retained": 0.97166, "state": "valid"},
  "ps_sheetmetal_grey_dents_cam3_150k_repeat01":        {"requested_px": 286481, "usable_lights_p10_med_p90": [9, 10, 10],  "region_at_least_six": 0.99938, "retained": 0.9773, "state": "valid"},
}

# RGB segmentation run on the BMW panel captures, measured 2026-09-06.
BMW_SEGMENTATION = {
  "source": "segment_chamber.py, run 2026-09-06 for the viewer",
  "finding": "The two perception streams want different framing. A photometric stereo capture fills the frame with one panel, which is the framing that breaks the RGB stream: the COCO vehicle crop finds no car in a bonnet close-up, so the part model runs at the wrong object-to-frame ratio and misattributes. Damage detection itself transfers, but only with the site-adapted checkpoint.",
  "runs": [
    {"capture": "ps_bmw_white_multidamage_hood_cam3_150k_repeat01", "damage_model": "CarDD 6-class, best_imgsz1024.pt", "crop": "none, no vehicle found", "detections": [], "note": "Three dents and three scratches that photometric stereo localises are not detected."},
    {"capture": "ps_bmw_white_multidamage_hood_cam3_150k_repeat01", "damage_model": "site-adapted YOLO11s 2-class", "crop": "none, no vehicle found", "detections": [{"class": "dent", "confidence": 0.85, "assigned_part": "trunk_tailgate", "attribution_correct": False}], "note": "Damage found. Part attribution wrong because the part model cannot parse a bonnet close-up."},
    {"capture": "ps_bmw_white_damaged_hood_r000_150k", "damage_model": "CarDD 6-class, best_imgsz1024.pt", "crop": "none, no vehicle found", "detections": [{"class": "glass shatter", "confidence": 0.39, "assigned_part": "trunk_tailgate", "attribution_correct": False}], "note": "False positive class on an undamaged region."},
    {"capture": "ps_bmw_white_damaged_hood_r000_150k", "damage_model": "site-adapted YOLO11s 2-class", "crop": "none, no vehicle found", "detections": [{"class": "dent", "confidence": 0.83, "assigned_part": "side_window", "attribution_correct": False}], "note": "Damage found, attribution wrong."},
    {"capture": "ps_bmw_white_damaged_doors_r270_150k", "damage_model": "CarDD 6-class, best_imgsz1024.pt", "crop": "auto, [513,377,1833,1033]", "detections": [{"class": "dent", "confidence": 0.40, "assigned_part": "front_door_left", "attribution_correct": True}], "note": "The one BMW capture framed wide enough for the crop to succeed. Damage and attribution both correct."}
  ]
}

SUBJECTS = [
  {"id": "range_rover", "name": "Range Rover", "scale": "1:18 die-cast",
   "finish": "gloss black", "geometry_outcome": "abstains",
   "has_multiview": True,
   "headline": "96 indexed views, 42 part records. Photometric stereo reaches partial on one region and invalid everywhere else. No metric depth released.",
   "why_it_matters": "The hardest surface in the study. Gloss black under chamber lighting leaves three to four usable lights of twelve, and the quality gate refuses to release geometry from it."},
  {"id": "bmw_white", "name": "White BMW", "scale": "1:24 die-cast",
   "finish": "white painted, diffuse", "geometry_outcome": "valid",
   "has_multiview": True,
   "headline": "Bonnet capture retains 97.2% of its region at 11 usable lights. Three imposed dents localised with a fixed-pose standard deviation of 0.01 mm.",
   "why_it_matters": "The strongest applied vehicle-panel evidence in the project. Where illumination support exists, geometry measures."},
  {"id": "sg_fdm", "name": "SG-FDM artefact", "scale": "printed PLA plate",
   "finish": "matte", "geometry_outcome": "valid",
   "has_multiview": False,
   "headline": "Mean absolute error 0.604 mm against nominal CAD across eight features, with a mean per-feature standard deviation of 0.0043 mm.",
   "why_it_matters": "High precision with moderate absolute accuracy. Seven of eight signs correct, several concave depths systematically compressed."},
  {"id": "sheet_metal", "name": "Painted steel strip", "scale": "65 mm wide strip",
   "finish": "grey etching primer", "geometry_outcome": "valid",
   "has_multiview": False,
   "headline": "Three of four dents yield valid depth at about 0.38 mm mean absolute error. The fourth is rejected for wrong sign.",
   "why_it_matters": "Forcing the rejected dent into the average would raise the error to 0.94 mm. This is abstention paying for itself."}
]

GEOMETRY_FIRST = {
  "status": "Research preview: validated on a machined artefact only",
  "claim": "Photometric stereo can propose damage regions with no RGB segmentation in the loop. Its detection limit is a limit on surface slope, not on depth.",
  "slope_not_depth": [
    {"feature": "C08", "rim_slope_deg": 8,  "nominal_depth_mm": 0.56, "detected": "0 of 6"},
    {"feature": "D25", "rim_slope_deg": 8,  "nominal_depth_mm": 0.87, "detected": "1 of 6"},
    {"feature": "D30", "rim_slope_deg": 8,  "nominal_depth_mm": 1.05, "detected": "1 of 6"},
    {"feature": "C16", "rim_slope_deg": 16, "nominal_depth_mm": 1.12, "detected": "0 of 6"},
    {"feature": "C30", "rim_slope_deg": 30, "nominal_depth_mm": 2.14, "detected": "5 of 6"},
    {"feature": "X30", "rim_slope_deg": 30, "nominal_depth_mm": 2.14, "detected": "5 of 6"},
    {"feature": "W0",  "rim_slope_deg": 90, "nominal_depth_mm": 1.00, "detected": "4 of 6"},
    {"feature": "W1",  "rim_slope_deg": 90, "nominal_depth_mm": 2.00, "detected": "5 of 6"}
  ],
  "deciding_comparison": "W0 is a 1.00 mm boss found in 4 captures of 6. C16 is a deeper 1.12 mm cap found in none. W0 has a vertical wall and C16 a 16 degree rim. Depth does not order the outcomes; slope does.",
  "negative_control": "Both views of the undamaged BMW bonnet produced zero proposals. The designed styling channels are visible in the departure map and sit below threshold throughout.",
  "range_rover": "Zero proposals on all six gloss-black Range Rover captures, including panels where RGB detects scratches at 0.9 confidence. At three to four usable lights of twelve there is no surface measurement to propose from.",
  "not_supported": "Nothing here is a damage detector evaluated on damage. Every positive is a machined spherical cap or a rectangular boss on a matte printed plate. This must not be merged into the fusion results until it has been evaluated on real damage."
}


def build_bmw_views():
    """Stage imagery for the BMW panel captures, both segmentation and PS."""
    seg = PSFIG / "segmentation"
    rgbdir = PSFIG / "rgb_frames"
    out = []
    for cap in ("ps_bmw_white_multidamage_hood_cam3_150k_repeat01",
                "ps_bmw_white_damaged_hood_r000_150k",
                "ps_bmw_white_damaged_doors_r270_150k",
                "ps_bmw_white_r000"):
        stem = f"cam3_{cap}_rgb"
        out.append({
            "id": cap,
            "subject": "bmw_white",
            "camera": "cam3",
            "support": CAPTURE_SUPPORT.get(cap),
            "img": {
                "rgb": emit(rgbdir / f"{stem}.png", f"bmw/{cap}_rgb.jpg",
                            BMW_MAX, BMW_Q),
                "damage": emit(seg / f"{stem}_damage.png", f"bmw/{cap}_damage.jpg",
                               BMW_MAX, BMW_Q),
                "parts": emit(seg / f"{stem}_parts.png", f"bmw/{cap}_parts.jpg",
                              RR_PARTS_MAX, RR_PARTS_Q),
                "thumb": emit(rgbdir / f"{stem}.png", f"bmw/{cap}_thumb.jpg",
                              THUMB_MAX, THUMB_Q),
            },
        })
    return out


def build_bmw_sequence():
    """The indexed 360-degree damaged BMW: 4 cameras, 12 angles, 48 views.

    This is what lets the turntable dial work on the subject where photometric
    stereo succeeds. Only the damaged vehicle is carried: the clean-against-
    damaged comparison already exists on the Range Rover, and a second 48-view
    sequence would push the bundle past the hosting ceiling.

    The records are the all-LED run, which writes the same schema as the Range
    Rover run, so both are read through `read_view_records`.
    """
    if not BMW30.exists():
        missing.append(str(BMW30.relative_to(ROOT)))
        return []

    out = read_view_records(BMW30, "bmwseq")
    for v in out:
        v["subject"] = "bmw_white"
        v["vehicle"] = "damaged"
    return out


def build_extra_figures():
    """Report figures the viewer cites directly."""
    figs = {}
    for key, src in {
        "fdm_evaluation": PSFIG / "ps_sg_fdm_tilt_cam3_150k_repeat01" / "cam3_l1_filtered_fdm_evaluation.png",
        "coverage_matrix": MVF / "figures" / "fig1_multiview_coverage.png",
        "single_vs_multiview": MVF / "figures" / "fig2_single_vs_multiview.png",
        "camera_arrangement": MVF / "figures" / "fig3_camera_arrangement.png",
        "quality_gated_evidence": MVF / "figures" / "fig5_quality_gated_evidence.png",
        "geo_detection_limit": GEO / "figures" / "figA_detection_limit.png",
        "geo_proposal_maps": GEO / "figures" / "figB_proposal_maps.png",
    }.items():
        figs[key] = emit(src, f"figures/{key}.jpg")
    return figs


def main():
    # This check must stay the first statement in main(). The next thing this
    # function does is delete viewer_data/, so without it a reader who runs the
    # script in the public repository destroys the published bundle and only
    # then discovers that the inputs are absent.
    if not MVF.exists():
        raise SystemExit(
            "This script rebuilds viewer_data/ from the full experiment "
            "outputs, which are not part of the public repository (several "
            "gigabytes of captures and solver results). It is published so "
            "that the provenance of every value shown in the viewer can be "
            "read. To rebuild only the page from the published bundle, run "
            "viewer/build_index.py instead.")

    if OUT.exists():
        shutil.rmtree(OUT)
    IMG.mkdir(parents=True)

    print("Range Rover sequence ...")
    views, parts, unattributed = build_range_rover()

    print("Photometric stereo captures ...")
    ps_captures = build_ps_captures()

    print("BMW panel captures ...")
    bmw_views = build_bmw_views()

    print("BMW indexed sequence ...")
    bmw_seq = build_bmw_sequence()

    print("Figures ...")
    figures = build_extra_figures()

    sensitivity = []
    with open(MVF / "results" / "aggregation_sensitivity.csv", newline="") as fh:
        for row in csv.DictReader(fh):
            sensitivity.append({k: (float(v) if "." in v or k.endswith("threshold")
                                    else int(v)) if v else None
                                for k, v in row.items()})

    compare = []
    with open(MVF / "results" / "clean_vs_damaged.csv", newline="") as fh:
        for row in csv.DictReader(fh):
            compare.append({k: (v if v else None) for k, v in row.items()})

    n_det = sum(len(v["detections"]) for v in views)
    n_attr = sum(1 for v in views for d in v["detections"] if d["attributed"])

    # The legacy 45-degree sequence is kept as measured numbers rather than as a
    # second set of images. Copied verbatim from the comparison run; nothing is
    # recomputed here.
    acquisition = json.loads((CMP / "metrics.json").read_text())
    acquisition["note"] = (
        "The externally relit 45-degree sequence is the illumination-limited "
        "baseline. Its imagery stays on disk under report_figures/multiview_fusion "
        "and report_figures/bmw_sequence. The viewer carries the all-LED sequence "
        "because that is what the report now reports.")
    acquisition["source"] = "report_figures/multiview_fusion_allled_comparison/metrics.json"
    acquisition["figure"] = emit(CMP / "fig_allled_rgb_comparison.png",
                                 "figures/acquisition_comparison.jpg")

    jdump({
        "generated": "2026-09-07",
        "vehicles": ["clean", "damaged"],
        "cameras": ["cam0", "cam1", "cam2", "cam3"],
        "angles": [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330],
        "acquisition": {
            "id": "allled_30deg",
            "label": "All-LED, 30 degree indexing",
            "illumination": "all twelve chamber white LED dies lit simultaneously, no external source",
            "increment_deg": 30,
            "exposure": {"range_rover": "100,000 us at gain 1.0",
                         "bmw_white": "60,000 us at gain 1.0"},
            "white_point": "one fixed per-camera white point estimated from the white BMW under the twelve-LED illuminant, applied unchanged to all three vehicles",
            "capture_sets": ["rgb_rr_black_clean_360_30deg_allled_r000",
                             "rgb_rr_black_damaged_360_30deg_allled_r000",
                             "rgb_bmw_white_damaged_360_30deg_allled_r000"],
            "supersedes": "the externally relit 45 degree sequences, retained on disk as the illumination-limited baseline",
            "source": "writeups/Experiment_AllLED_RGB_Reevaluation.md",
        },
        "confidence_threshold": 0.30,
        "damage_classes": ["scratch", "dent"],
        "damage_model": "site-adapted YOLO11s-seg, 1024 px, confidence 0.30",
        "part_model": "SegFormer-B2, 512 px letterboxed, 23-class taxonomy",
        "crop_model": "COCO-pretrained YOLO11x-seg, 1280 px, confidence 0.15",
        "parts": sorted({p["part"] for p in parts}),
        "counts": {
            "views_total": len(views),
            "views_eligible": sum(1 for v in views if v["eligible"]),
            "views_crop_failed": sum(1 for v in views if not v["crop_ok"]),
            "part_records": len(parts),
            "damage_instances": n_det,
            "attributed": n_attr,
            "unattributed": n_det - n_attr,
            "ps_captures_shown": len(ps_captures),
            "bmw_sequence_views": len(bmw_seq),
            "bmw_sequence_crop_ok": sum(1 for b in bmw_seq if b["crop_ok"]),
            "bmw_sequence_detections": sum(len(b["detections"]) for b in bmw_seq),
        },
        "caveats": {
            "sample": "Two physical vehicles. Every count is a case-study description, not an accuracy estimate. There is no per-instance ground truth on this sequence, so no precision, recall or F1 is reported.",
            "confound": "The damage model was fine-tuned on 22 images of the damaged car and none of a comparable clean one. Its false-positive behaviour on gloss black is a property of that training set as much as of the surface.",
            "scale": "Every measurement is on 1:18 and 1:24 scale die-cast subjects. A millimetre on a model is not a millimetre on a full-size vehicle, and depth does not scale by direct multiplication.",
        },
    }, "manifest.json")

    jdump(SUBJECTS, "subjects.json")
    jdump(views, "views.json")
    jdump(parts, "parts.json")
    jdump(ps_captures, "ps_captures.json")
    jdump(bmw_views, "bmw_views.json")
    jdump(bmw_seq, "bmw_sequence.json")
    jdump({"measurements": MEASUREMENTS,
           "capture_support": CAPTURE_SUPPORT,
           "bmw_segmentation": BMW_SEGMENTATION}, "measurements.json")
    jdump(sensitivity, "sensitivity.json")
    jdump(compare, "compare.json")
    jdump(GEOMETRY_FIRST, "geometry_first.json")
    jdump(acquisition, "acquisition.json")
    jdump({"unattributed_by_vehicle": unattributed, "figures": figures},
          "extras.json")

    total = sum(sz for _, sz in written)
    print(f"\n{len(written)} files, {total / 1e6:.2f} MB total")
    for name, sz in sorted(written, key=lambda t: -t[1])[:6]:
        print(f"   {sz/1e6:6.2f} MB  {name}")
    if missing:
        print(f"\n{len(missing)} missing source files:")
        for m in sorted(set(missing))[:25]:
            print("   ", m)


if __name__ == "__main__":
    main()
