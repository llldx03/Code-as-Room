"""
Stage 3 — Deterministic geometric verifier & repair-feeder (IDEA 1).

Self-contained add-on. The Stage 3 Code Critic only checks `compile()` syntax,
so it misses two error classes that weaker models (e.g. Qwen) produce far more
than strong ones (e.g. Gemini):
  1. runtime Blender-API misuse (already surfaced by the render step), and
  2. **geometric** defects — furniture-vs-furniture overlaps and out-of-bounds
     objects — which a *visual* critic routinely accepts.

A free, deterministic checker over the live-scene layout (`_layout.json`, which
the render step already produces) catches these. This module:
  * verifies the layout (overlaps, out-of-bounds),
  * injects the violations into the existing VLM `analysis` dict so the existing
    LLM fixer repairs them (no new fix logic anywhere else),
  * gates the score so the iteration loop cannot "pass" while hard geometric
    violations remain (floored well above the Stage-3 abort threshold so it can
    never cause a false abort),
  * writes a `geometry_report.json` artifact for transparency.

NOTHING here is imported at module load by the runner; the runner calls
`augment_analysis_with_geometry(...)` from a single guarded call site. Pure
stdlib only — safe to import inside Blender or the pipeline.

Standalone use (e.g. to reproduce the pilot):
    python geometry_verifier.py path/to/_layout.json [SCENE_W SCENE_D]
"""
from __future__ import annotations

import json
import os
import re
from itertools import combinations
from typing import Any, Callable, Dict, List, Optional, Tuple

# --------------------------------------------------------------------------- #
# Tunables (conservative on purpose: false positives trigger bad LLM fixes)
# --------------------------------------------------------------------------- #
MIN_OVERLAP_AREA = 0.06       # m^2 — ignore tiny touches
MIN_OVERLAP_FRAC = 0.12       # fraction of the smaller footprint that must overlap
MIN_Z_OVERLAP = 0.05          # m — require true 3D intersection (filters "on-top")
FLAT_HEIGHT = 0.12            # m — objects thinner than this are rugs/mats/decor
OOB_MARGIN = 0.40             # m — only flag *gross* wall penetration (no rotation data)
SCORE_PENALTY_PER_VIOLATION = 0.06
SCORE_FLOOR = 0.40            # stays above the Stage-3 abort threshold (~0.05)

# Small things that legitimately sit ON furniture — never count as collisions.
_DECOR_TOKENS = (
    "pillow", "cushion", "bolster", "blanket", "throw", "duvet", "bedding",
    "sheet", "linen", "rug", "mat", "runner", "carpet", "lamp", "book",
    "vase", "plant", "pot", "planter", "bowl", "candle", "tray", "frame",
    "clock", "decor", "ornament", "cloth", "towel", "basket", "bottle",
    "cup", "mug", "remote", "magazine", "sculpture", "figurine", "art",
    "mirror", "picture", "painting", "tv", "screen", "monitor", "keyboard",
    "mouse", "speaker", "phone", "tablet", "laptop", "box", "bag",
)
_ARCH_TOKENS = ("wall", "floor", "ceiling", "window", "door", "boundary")


def _is_decor_or_flat(o: Dict[str, Any]) -> bool:
    name = str(o.get("name", "")).lower()
    if o.get("height", 1.0) < FLAT_HEIGHT:
        return True
    norm = name.replace("_", " ").replace("-", " ")
    return any(tok in norm for tok in _DECOR_TOKENS)


def _is_arch(o: Dict[str, Any]) -> bool:
    norm = str(o.get("name", "")).lower().replace("_", " ").replace("-", " ")
    return any(tok in norm for tok in _ARCH_TOKENS)


def _footprint(o: Dict[str, Any]) -> float:
    return max(0.0, float(o.get("width", 0))) * max(0.0, float(o.get("depth", 0)))


def _xy_overlap(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    ax1, ax2 = a["x"] - a["width"] / 2, a["x"] + a["width"] / 2
    ay1, ay2 = a["y"] - a["depth"] / 2, a["y"] + a["depth"] / 2
    bx1, bx2 = b["x"] - b["width"] / 2, b["x"] + b["width"] / 2
    by1, by2 = b["y"] - b["depth"] / 2, b["y"] + b["depth"] / 2
    ox = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    oy = max(0.0, min(ay2, by2) - max(ay1, by1))
    return ox * oy


def _z_overlap(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    a1, a2 = a["z"] - a["height"] / 2, a["z"] + a["height"] / 2
    b1, b2 = b["z"] - b["height"] / 2, b["z"] + b["height"] / 2
    return max(0.0, min(a2, b2) - max(a1, b1))


def _required(o: Dict[str, Any]) -> bool:
    return all(k in o for k in ("name", "x", "y", "z", "width", "depth", "height"))


# --------------------------------------------------------------------------- #
# Core verification (pure function — used by the runner AND the standalone test)
# --------------------------------------------------------------------------- #
def verify_layout(
    layout: List[Dict[str, Any]],
    scene_w: Optional[float] = None,
    scene_d: Optional[float] = None,
) -> Dict[str, Any]:
    """Return {'overlaps': [...], 'out_of_bounds': [...]} for a layout list.

    Overlap entries: {a, b, area, frac}. OOB entries: {name, axis, over}.
    Conservative: skips decor/flat objects, requires true 3D intersection,
    and only flags gross wall penetration (OOB) since rotation isn't available.
    """
    objs = [o for o in (layout or []) if isinstance(o, dict) and _required(o)
            and not _is_arch(o)]

    overlaps: List[Dict[str, Any]] = []
    for a, b in combinations(objs, 2):
        if _is_decor_or_flat(a) or _is_decor_or_flat(b):
            continue
        area = _xy_overlap(a, b)
        if area < MIN_OVERLAP_AREA:
            continue
        if _z_overlap(a, b) < MIN_Z_OVERLAP:
            continue
        smaller = min(_footprint(a), _footprint(b)) or 1e-6
        frac = area / smaller
        if frac < MIN_OVERLAP_FRAC:
            continue
        overlaps.append({
            "a": a["name"], "b": b["name"],
            "area": round(area, 3), "frac": round(frac, 2),
            "a_footprint": round(_footprint(a), 3),
            "b_footprint": round(_footprint(b), 3),
        })
    overlaps.sort(key=lambda v: -v["area"])

    out_of_bounds: List[Dict[str, Any]] = []
    if scene_w and scene_d and scene_w > 0 and scene_d > 0:
        hx, hy = scene_w / 2.0, scene_d / 2.0
        for o in objs:
            if _is_decor_or_flat(o):
                continue
            over_x = max((o["x"] + o["width"] / 2) - hx, (-hx) - (o["x"] - o["width"] / 2))
            over_y = max((o["y"] + o["depth"] / 2) - hy, (-hy) - (o["y"] - o["depth"] / 2))
            over = max(over_x, over_y)
            if over > OOB_MARGIN:
                out_of_bounds.append({
                    "name": o["name"],
                    "axis": "x" if over_x >= over_y else "y",
                    "over": round(over, 2),
                })
        out_of_bounds.sort(key=lambda v: -v["over"])

    return {"overlaps": overlaps, "out_of_bounds": out_of_bounds}


def parse_room_dims(
    scene_code: Optional[str] = None,
    stage1_json: Optional[Any] = None,
) -> Tuple[Optional[float], Optional[float]]:
    """Best-effort (SCENE_W, SCENE_D). Returns (None, None) if unknown."""
    if scene_code:
        mw = re.search(r"SCENE_W\s*=\s*([\d.]+)", scene_code)
        md = re.search(r"SCENE_D\s*=\s*([\d.]+)", scene_code)
        if mw and md:
            try:
                return float(mw.group(1)), float(md.group(1))
            except ValueError:
                pass
    # Fallback: Stage 1 estimated_dimensions (dict or "6.5m x 5.5m" string)
    try:
        dims = None
        if isinstance(stage1_json, dict):
            dims = (stage1_json.get("estimated_dimensions")
                    or stage1_json.get("room_dimensions"))
        if isinstance(dims, dict):
            w = dims.get("width") or dims.get("x") or dims.get("length_x")
            d = dims.get("depth") or dims.get("y") or dims.get("length") or dims.get("length_y")
            if w and d:
                return float(w), float(d)
        if isinstance(dims, str):
            nums = re.findall(r"([\d.]+)", dims)
            if len(nums) >= 2:
                return float(nums[0]), float(nums[1])
    except Exception:
        pass
    return None, None


# --------------------------------------------------------------------------- #
# Runner integration: augment the VLM analysis + gate the score (single entry)
# --------------------------------------------------------------------------- #
def augment_analysis_with_geometry(
    analysis: Dict[str, Any],
    layout: List[Dict[str, Any]],
    scene_code: Optional[str] = None,
    stage1_json: Optional[Any] = None,
    score: float = 1.0,
    output_dir: Optional[str] = None,
    log: Optional[Callable[[str, str], None]] = None,
) -> float:
    """Mutate `analysis` in place with deterministic geometric violations and
    return a (possibly reduced) score. Designed to be the ONLY thing the runner
    calls; everything else stays unchanged.
    """
    def _say(msg: str, level: str = "info") -> None:
        if log:
            try:
                log(msg, level)
            except Exception:
                pass

    scene_w, scene_d = parse_room_dims(scene_code, stage1_json)
    result = verify_layout(layout, scene_w, scene_d)
    overlaps = result["overlaps"]
    oob = result["out_of_bounds"]
    n_viol = len(overlaps) + len(oob)

    # Persist a report regardless (transparency / pilot reproduction).
    if output_dir:
        try:
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "geometry_report.json"), "w",
                      encoding="utf-8") as f:
                json.dump({
                    "scene_w": scene_w, "scene_d": scene_d,
                    "vlm_score": score,
                    "n_violations": n_viol,
                    "overlaps": overlaps, "out_of_bounds": oob,
                }, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            _say(f"geometry_report not written: {exc}", "warning")

    if n_viol == 0:
        _say("Geometry verifier: no overlaps / out-of-bounds detected", "info")
        return score

    # ---- Inject into analysis so the EXISTING fixer repairs them ----
    analysis.setdefault("overlapping_pairs", [])
    analysis.setdefault("out_of_bounds", [])
    analysis.setdefault("objects_to_fix", [])

    existing_pairs = {
        frozenset((str(p.get("object_a", "")), str(p.get("object_b", ""))))
        for p in analysis["overlapping_pairs"] if isinstance(p, dict)
    }
    fp = {o.get("name", ""): _footprint(o) for o in layout if isinstance(o, dict)}

    for ov in overlaps:
        key = frozenset((ov["a"], ov["b"]))
        if key not in existing_pairs:
            analysis["overlapping_pairs"].append({
                "object_a": ov["a"], "object_b": ov["b"], "severity": "major",
                "description": (f"deterministic geometric overlap "
                                f"{ov['area']} m^2 ({int(ov['frac']*100)}% of smaller "
                                f"footprint); move the smaller item to clear it"),
            })
        # Move the smaller-footprint object (least disruptive).
        mover = ov["a"] if fp.get(ov["a"], 1e9) <= fp.get(ov["b"], 1e9) else ov["b"]
        other = ov["b"] if mover == ov["a"] else ov["a"]
        analysis["objects_to_fix"].append({
            "object_id": mover, "action": "move",
            "reason": f"resolve geometric overlap with {other} (verifier)",
        })

    for o in oob:
        analysis["out_of_bounds"].append({
            "object_id": o["name"],
            "description": (f"extends {o['over']} m past the {o['axis']} wall "
                            f"(deterministic verifier); pull it back inside the room"),
        })
        analysis["objects_to_fix"].append({
            "object_id": o["name"], "action": "move",
            "reason": f"pull inside room outline ({o['over']} m out, verifier)",
        })

    # ---- Gate the score (never below SCORE_FLOOR, so no false abort) ----
    penalty = min(0.5, SCORE_PENALTY_PER_VIOLATION * n_viol)
    new_score = max(SCORE_FLOOR, min(1.0, score) - penalty)

    _say(f"Geometry verifier: {len(overlaps)} overlap(s), {len(oob)} out-of-bounds "
         f"-> score {score:.0%} -> {new_score:.0%} (fed to fixer)", "warning")
    for ov in overlaps[:6]:
        _say(f"    overlap: {ov['a']} x {ov['b']}  {ov['area']} m^2 "
             f"({int(ov['frac']*100)}%)", "info")
    for o in oob[:4]:
        _say(f"    out-of-bounds: {o['name']}  {o['over']} m past {o['axis']} wall", "info")

    return new_score


# --------------------------------------------------------------------------- #
# Standalone CLI (pilot reproduction): python geometry_verifier.py x.json [W D]
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python geometry_verifier.py path/to/_layout.json [SCENE_W SCENE_D]")
        raise SystemExit(2)
    layout = json.load(open(sys.argv[1], encoding="utf-8"))
    w = float(sys.argv[2]) if len(sys.argv) > 2 else None
    d = float(sys.argv[3]) if len(sys.argv) > 3 else None
    res = verify_layout(layout, w, d)
    print(f"{len(layout)} objects | {len(res['overlaps'])} overlaps | "
          f"{len(res['out_of_bounds'])} out-of-bounds")
    for ov in res["overlaps"]:
        print(f"  overlap: {ov['a']:<28} x {ov['b']:<28} {ov['area']} m^2 ({int(ov['frac']*100)}%)")
    for o in res["out_of_bounds"]:
        print(f"  OOB: {o['name']:<28} {o['over']} m past {o['axis']} wall")
