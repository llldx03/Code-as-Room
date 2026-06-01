"""
Stage Small Objects - Surface-driven small-object placement (Stage 7)
========================================================================

Runs AFTER Stage 6 (stage6_geometry). Replaces the old "Stage 4 add small
items everywhere" approach with a surface-aware pipeline:

  1. Plane Discovery (rule-based): walk DETAILED_GEOMETRY from Stage 6,
     detect placement planes (table tops, cabinet tops, open-shelf layers,
     seat surfaces) with world-space size / center / orientation.

  2. Open-vs-closed furniture filter: only enumerate internal shelves of
     furniture whose object_type is in an open-furniture whitelist.

  3. LLM placement (per parent object, image-grounded): ask the LLM to
     decide which small objects to put on each plane, returning plane-local
     UV positions + bbox sizes.

  4. Code generation: append `create_box` / `create_cylinder` calls for the
     new small objects to the end of the Stage 6 geometry_output.py. The
     output file can be rendered directly in Blender.

The module is intentionally self-contained so it can be invoked as a CLI
without the rest of the pipeline, which makes iteration fast.

Usage:
    cd agent_utils
    python stage7_small_objects.py \
        --image ../agent_input/room.png \
        --geometry-json pipeline_output/stage6_geometry/geometry_progress.json \
        --base-code   pipeline_output/stage6_geometry/geometry_output.py \
        --output-dir  pipeline_output/stage7_small_objects

P1 scope: plane types = {top, shelf, seat}. The "floor_near" plane type is
intentionally deferred to a later phase.
"""

import argparse
import ast
import base64
import json
import math
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# Path setup identical to sibling stage modules.
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, "stage3"))

from stage3.core import LLMClient, extract_json_from_response  # noqa: E402
from memory import Memory  # noqa: E402
from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402


# ==============================================================================
# Configuration: object-type classification
# ==============================================================================

# object_type substrings that indicate the object has a usable top surface for
# decoration. Matching is lowercase substring — keep tokens short and specific.
TOP_SURFACE_TYPES: Tuple[str, ...] = (
    "table", "desk", "console", "counter", "countertop",
    "nightstand", "bedside",
    "dresser", "sideboard", "buffet", "credenza",
    "cabinet", "wardrobe",  # closed cabinets still have a usable top
    "bookshelf", "bookcase", "shelving", "shelf_unit",
    "tv_console", "tv_stand", "media_console", "media_unit",
    "coffee_table", "side_table", "end_table", "accent_table",
    "vanity",
    # Wall-mounted / freestanding shelves and hutches also have a usable
    # horizontal surface. Broad "shelf" token intentionally catches plain
    # "shelf", "floating shelf", "wall shelf", "display shelf", etc.
    "shelf", "ledge",
    "hutch", "armoire", "pantry",
    # Lab-specific: a fume hood's interior work surface is the primary
    # placement plane for reagents, beakers, hotplates etc.
    "fume_hood",
    # --- Lab benches & worktops (CRITICAL for lab scenes) ---
    # Stage 4 splits a double-deck island bench into a separate `*_worktop`
    # bbox; describe LLM tags it with phrases like "laboratory island
    # workbench", "wall-mounted workbench", "stainless steel worktop", or
    # "preparation bench". None of those contain "table" / "counter", so
    # before this list was extended the worktop produced ZERO planes and
    # nothing was placed on the bench. Tokens are intentionally specific
    # ("workbench", "lab_bench", ...) so we never collide with seating
    # benches like an entryway bench.
    "workbench", "work_bench", "worktop", "work_surface",
    "lab_bench", "laboratory_bench", "lab_workbench",
    "island_bench", "prep_bench", "preparation_bench",
    "wall_bench", "perimeter_bench", "side_bench", "south_bench",
    "north_bench", "east_bench", "west_bench",
    "biosafety_cabinet", "bio_safety_cabinet", "fume_cabinet",
    "optical_table", "lab_table",
)

# Part-name keywords that, when present on at least one bbox part of a
# parent, force the parent to be treated as having a top surface even
# when its `object_type` text does not match TOP_SURFACE_TYPES. This is
# the safety net for cases where Stage 7 paraphrases the object type
# vaguely ("support post", "work area") but Stage 6 still emits a part
# clearly labelled as a worktop / counter / desktop slab. Match is
# substring, lowercase.
TOP_SURFACE_PART_KEYWORDS: Tuple[str, ...] = (
    "worktop", "work_top", "tabletop", "table_top",
    "bench_top", "benchtop", "counter_top", "countertop",
    "top_panel", "top_surface", "main_top", "primary_top",
    "desktop", "desk_top", "work_surface", "work_area",
    "slab",  # "stainless_steel_worktop_slab", "granite_slab", ...
    "deck",  # "deck_top", "back_deck"
)

# object_type substrings that indicate internal shelves are open / visible and
# can be populated with items. Everything else is treated as closed — only its
# top is decorated, internal shelves are skipped.
OPEN_SHELF_TYPES: Tuple[str, ...] = (
    "bookshelf", "bookcase", "shelving", "shelf_unit", "open_shelf",
    "etagere", "etagère",
    "display_cabinet", "display_case",
    "tv_console", "tv_stand", "media_console", "media_unit",
    "sideboard", "credenza",        # often have open compartments
    "nightstand", "bedside",        # frequently have one open shelf below
    "coffee_table", "side_table",   # lower tier is almost always open
    "console",                       # console tables often have lower shelf
    # Glass-front / open-front multi-tier storage pieces. Items placed on
    # internal shelves are visible through the glass doors, so they count
    # as "open" for placement purposes.
    "hutch", "china_cabinet", "china_hutch",
    "glass_cabinet", "glass_front_cabinet", "glass-front",
    "curio", "curio_cabinet",
    "armoire", "pantry",
    "shelf",  # generic — covers "floating shelf", "wall shelf", etc.
    # Lab-specific: fume hoods typically have an open base cabinet shelf
    # under the work surface for storing solvent bottles / waste jugs.
    "fume_hood",
)

# object_type substrings that indicate a seat surface (chairs/sofas/benches).
SEAT_SURFACE_TYPES: Tuple[str, ...] = (
    "chair", "armchair", "sofa", "couch", "loveseat",
    "bench", "ottoman", "pouf", "stool", "footstool",
    "settee", "daybed",
)

# ------------------------------------------------------------------
# Wall-decoration whitelist (v3, 2026-05-02)
# ------------------------------------------------------------------
# Objects added by Stage 4 as wall-mounted (e.g. floating shelves, wall
# cabinets, spice racks) are *intentionally* skipped by Stage 6
# (stage6_geometry) and therefore never enter DETAILED_GEOMETRY. The
# regular PlaneFinder cannot see them, so historically NO small objects
# were ever placed on those Stage 4 wall items.
#
# The whitelist below describes the subset of wall items that DO carry
# a usable horizontal top surface (boards / cabinet tops / racks with a
# ledge). Pure decorations (art, mirror, clock, sconce, mounted TV,
# pot rack with hooks down) are deliberately EXCLUDED — there is no
# meaningful "place a small object on top of a painting".
#
# Matching is lowercase substring on the stage5_describe `object_type`
# string. Tokens use spaces (not underscores) because describe LLM
# emits phrases like "floating shelves" / "wall cabinet".
WALL_DECOR_TOP_TYPES: Tuple[str, ...] = (
    "floating shelf", "floating shelves",
    "wall shelf", "wall shelves", "wall-shelf", "wall-shelves",
    "wall ledge", "picture ledge", "plate ledge", "display ledge",
    "wall cabinet", "wall-mounted cabinet", "wall mounted cabinet",
    "wall cupboard", "wall-mounted cupboard",
    "spice rack", "spice shelf",
    "wall storage", "wall-mounted storage",
    "open wall rack", "wall mounted rack", "wall-mounted rack",
    "open shelving",
    "wall mounted bookshelf", "wall-mounted bookshelf", "wall bookshelf",
    "hutch",            # often wall-mounted
    "medicine cabinet", # bathroom
    "wall-mounted dish rack", "dish rack",
    # Stage 4 sometimes labels a decorative ledge as just "ledge":
    "ledge",
)

# Wall items we MUST NOT synthesise a top plane for. These usually
# have no flat top surface (hooks, picture face) and any "plane on top"
# would make the LLM place vases / books mid-air. Matching is the same
# substring rule as above.
WALL_DECOR_BLACKLIST_TYPES: Tuple[str, ...] = (
    "art", "artwork", "painting", "poster", "picture", "photo",
    "mirror", "clock", "tapestry", "mural", "canvas", "print",
    "sconce", "wall light", "wall lamp", "wall lantern",
    "wall-mounted tv", "wall mounted tv", "wall tv", "mounted tv",
    "pot rack",         # hooks pointing down, no top to place on
    "hook rail", "coat rack", "coat hook", "key holder",
    "curtain rod", "blind",
)


# Geometric thresholds.
MIN_PLANE_EDGE_M = 0.20        # ignore planes whose width or depth is below this
SHELF_MAX_THICKNESS_M = 0.08   # box parts thinner than this may qualify as a shelf
SHELF_MIN_COVERAGE = 0.50      # shelf dimensions must be at least this fraction of bbox
SEAT_Z_WORLD_MIN = 0.25        # seat top is typically above this world-space z
SEAT_Z_WORLD_MAX = 0.65        # ...and below this
TOP_PANEL_MAX_THICKNESS_M = 0.10


# ==============================================================================
# Data structures
# ==============================================================================
@dataclass
class Plane:
    """A single placement plane on a parent object.

    All spatial values are already in WORLD space. `size_wd` is given in the
    plane's LOCAL frame (i.e. before `orientation_rad` is applied), matching
    how the LLM will reason about (u, v) coordinates on the plane.
    """

    plane_id: str                        # unique within its parent
    parent_name: str                     # DETAILED_GEOMETRY key
    plane_type: str                      # "top" | "shelf" | "seat"
    world_center: Tuple[float, float, float]
    size_wd: Tuple[float, float]
    orientation_rad: float               # Z-axis rotation of the plane (world)
    source_part_name: str                # which part of the parent produced this
    parent_note: str = ""                # human-readable note (e.g. "shelf 2/4")

    def to_dict(self) -> dict:
        return {
            "plane_id": self.plane_id,
            "parent_name": self.parent_name,
            "plane_type": self.plane_type,
            "world_center": list(self.world_center),
            "size_wd": list(self.size_wd),
            "orientation_rad": self.orientation_rad,
            "source_part_name": self.source_part_name,
            "parent_note": self.parent_note,
        }


@dataclass
class SmallObjectItem:
    """A single small object decided by the LLM and resolved to world space.

    `stack_index` supports the "vertical stack of atomic units" pattern
    (plates on top of plates, books piled on a coffee table). Siblings with
    the same (plane_id, local_uv_rounded) and increasing `stack_index` have
    their world Z raised by the accumulated height of the lower siblings,
    so the stack still looks like a physical pile while each unit remains
    an independent primitive in Blender.
    """

    name: str
    item_type: str
    shape: str                           # "box" | "cylinder"
    parent_name: str
    plane_id: str
    plane_type: str
    world_location: Tuple[float, float, float]   # center of the bbox in world space
    size: Tuple[float, float, float]              # width, depth, height (meters)
    rotation_z: float                             # world-space Z rotation (radians)
    color_hint: str = ""
    description: str = ""
    stack_index: int = 0                          # 0 = base, 1+ = stacked above

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "item_type": self.item_type,
            "shape": self.shape,
            "parent_name": self.parent_name,
            "plane_id": self.plane_id,
            "plane_type": self.plane_type,
            "world_location": list(self.world_location),
            "size": list(self.size),
            "rotation_z": self.rotation_z,
            "color_hint": self.color_hint,
            "description": self.description,
            "stack_index": self.stack_index,
        }


# ==============================================================================
# PlaneFinder: rule-based plane discovery from DETAILED_GEOMETRY
# ==============================================================================
class PlaneFinder:
    """Walk DETAILED_GEOMETRY and emit placement planes.

    The finder is intentionally conservative:
      * Top planes are only generated for objects whose `object_type` matches
        TOP_SURFACE_TYPES.
      * Shelf planes are only generated for objects whose `object_type` matches
        OPEN_SHELF_TYPES (closed cabinets/wardrobes are skipped internally).
      * Seat planes are only generated for SEAT_SURFACE_TYPES objects.

    This way we never invent a plane on top of e.g. a lamp or a plant pot.
    """

    def __init__(self, detailed_geometry: Dict[str, Any],
                 describe_objects: Optional[List[Dict[str, Any]]] = None,
                 verbose: bool = False):
        self.detailed_geometry = detailed_geometry
        self.verbose = verbose
        # Build a name -> describe-entry map for fast object_type lookup. Names
        # coming from stage5_describe sometimes differ slightly from
        # DETAILED_GEOMETRY keys; we match case-insensitively and tolerate
        # underscore/space swaps.
        self.describe_by_name: Dict[str, Dict[str, Any]] = {}
        for entry in describe_objects or []:
            nm = str(entry.get("name", ""))
            if nm:
                self.describe_by_name[self._norm(nm)] = entry

    # ---------- helpers ----------

    @staticmethod
    def _norm(name: str) -> str:
        return name.lower().replace(" ", "_").strip("_")

    def _get_object_type(self, parent_name: str, fallback: str = "") -> str:
        """Look up `object_type` via describe data, fallback to parent_name."""
        key = self._norm(parent_name)
        entry = self.describe_by_name.get(key)
        if entry:
            ot = str(entry.get("object_type", "")).lower().strip()
            if ot:
                return ot
        return fallback or parent_name.lower()

    def _get_describe_bbox(self, parent_name: str) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
        """Return (center, dimensions) from Stage 7 describe data when present."""
        entry = self.describe_by_name.get(self._norm(parent_name))
        if not entry:
            return None
        center = entry.get("location") or entry.get("center_location")
        dims = entry.get("dimensions") or entry.get("bounding_dimensions")
        if not (
            isinstance(center, (list, tuple)) and len(center) >= 3 and
            isinstance(dims, (list, tuple)) and len(dims) >= 3
        ):
            return None
        try:
            c = (float(center[0]), float(center[1]), float(center[2]))
            d = (float(dims[0]), float(dims[1]), float(dims[2]))
        except (TypeError, ValueError):
            return None
        return c, d

    @staticmethod
    def _matches_any(needle: str, tokens: Tuple[str, ...]) -> bool:
        n = needle.lower()
        # We match on both the raw string and an underscore-normalized copy so
        # that "coffee table" matches both "coffee_table" and "coffee table".
        n_norm = n.replace(" ", "_")
        for t in tokens:
            if t in n or t in n_norm:
                return True
        return False

    def _has_top_part_signal(self, parts: List[Dict[str, Any]]) -> bool:
        """Return True if any part name contains a strong top-surface keyword.

        This is used as a safety net for objects whose `object_type` is too
        vague to match TOP_SURFACE_TYPES (e.g. Stage 7 returning "support
        post" or "work area"). When at least one Stage 6 part is named
        `*_worktop`, `*_tabletop`, `*_bench_top`, `*_slab`, `*_top_panel`,
        ... we trust the part naming and treat the parent as having a top
        plane.
        """
        for p in parts:
            pname = str(p.get("name", "")).lower()
            if not pname:
                continue
            for kw in TOP_SURFACE_PART_KEYWORDS:
                if kw in pname:
                    return True
        return False

    @staticmethod
    def _rotate_z(x: float, y: float, yaw: float) -> Tuple[float, float]:
        c, s = math.cos(yaw), math.sin(yaw)
        return x * c - y * s, x * s + y * c

    # ---------- main entry ----------

    def find_planes(self) -> List[Plane]:
        planes: List[Plane] = []
        for parent_name, obj in self.detailed_geometry.items():
            try:
                planes.extend(self._planes_for_object(parent_name, obj))
            except Exception as exc:  # noqa: BLE001
                if self.verbose:
                    print(f"   ! plane detection failed for {parent_name}: {exc}")
        return planes

    # ---------- per-object ----------

    def _planes_for_object(self, parent_name: str, obj: Dict[str, Any]) -> List[Plane]:
        parts = obj.get("parts", []) or []
        if not parts:
            return []

        center = obj.get("center", obj.get("center_location", [0.0, 0.0, 0.0]))
        rot = obj.get("rotation", obj.get("base_rotation", [0.0, 0.0, 0.0]))
        cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
        yaw = float(rot[2]) if len(rot) >= 3 else 0.0

        object_type = self._get_object_type(parent_name)
        parent_norm = self._norm(parent_name)

        # `_matches_any` is run against BOTH the describe-derived
        # object_type AND the normalized parent name. The parent-name
        # fallback rescues lab benches whose Stage 4 split parts carry
        # explicit names like `CentralIslandBench_worktop` even when the
        # describe LLM returned a vague type.
        has_top = (
            self._matches_any(object_type, TOP_SURFACE_TYPES)
            or self._matches_any(parent_norm, TOP_SURFACE_TYPES)
        )
        has_open_shelves = (
            self._matches_any(object_type, OPEN_SHELF_TYPES)
            or self._matches_any(parent_norm, OPEN_SHELF_TYPES)
        )
        has_seat = (
            self._matches_any(object_type, SEAT_SURFACE_TYPES)
            or self._matches_any(parent_norm, SEAT_SURFACE_TYPES)
        )

        # Strongest signal: a Stage 6 part name explicitly says "worktop"
        # / "tabletop" / "bench_top" / "slab" / "deck". Trust it and force
        # a top plane regardless of the textual object_type.
        if not has_top and self._has_top_part_signal(parts):
            has_top = True
            if self.verbose:
                print(
                    f"   + top plane forced by part-name signal for "
                    f"{parent_name} (type='{object_type}')"
                )

        # Lab benches frequently get classified as "bench" via parent name
        # alone, which would falsely trigger seat detection on a 0.9m-tall
        # workbench. If we already detected a top via lab-specific tokens,
        # turn off the seat path to avoid placing a "seat" plane on a
        # workbench.
        if has_top and has_seat:
            lab_bench_tokens = (
                "workbench", "work_bench", "worktop", "lab_bench",
                "laboratory_bench", "island_bench", "prep_bench",
                "preparation_bench", "wall_bench", "perimeter_bench",
                "lab_workbench",
            )
            if (
                self._matches_any(object_type, lab_bench_tokens)
                or self._matches_any(parent_norm, lab_bench_tokens)
            ):
                has_seat = False

        if not (has_top or has_open_shelves or has_seat):
            return []

        # Normalize parts into a common structure with world-space top-face info.
        processed = []
        for p in parts:
            ptype = str(p.get("type", p.get("shape_type", "box"))).lower()
            if ptype not in ("box", "cylinder"):
                continue  # spheres / cones cannot host flat decor
            pname = str(p.get("name", ""))
            loc = p.get("loc", p.get("relative_location", [0, 0, 0]))
            dim = p.get("dim", p.get("dimensions", [0, 0, 0]))
            try:
                rx, ry, rz = float(loc[0]), float(loc[1]), float(loc[2])
                dx, dy, dz = float(dim[0]), float(dim[1]), float(dim[2])
            except (TypeError, ValueError, IndexError):
                continue
            if dx <= 0 or dy <= 0 or dz <= 0:
                continue
            top_local_z = rz + dz / 2.0
            top_world_z = cz + top_local_z
            wx_off, wy_off = self._rotate_z(rx, ry, yaw)
            top_world_xy = (cx + wx_off, cy + wy_off)
            processed.append({
                "name": pname,
                "type": ptype,
                "rel_loc": (rx, ry, rz),
                "dim": (dx, dy, dz),
                "top_local_z": top_local_z,
                "top_world_z": top_world_z,
                "top_world_xy": top_world_xy,
            })

        planes: List[Plane] = []
        top_source_part: Optional[str] = None

        if has_top:
            plane = self._detect_top_plane(parent_name, processed, yaw)
            if plane is not None:
                plane = self._lift_lab_work_surface_to_bbox_top(
                    parent_name, obj, object_type, plane
                )
                planes.append(plane)
                top_source_part = plane.source_part_name

        if has_open_shelves:
            # Pass the top plane's source part so we never double-count the
            # tabletop as "shelf 1". Also require at least one additional
            # qualifying shelf part: a single-tier table is NOT multi-level.
            planes.extend(
                self._detect_shelf_planes(parent_name, processed, yaw,
                                           exclude_part=top_source_part)
            )

        if has_seat:
            seat_plane = self._detect_seat_plane(parent_name, processed, yaw)
            if seat_plane is not None:
                planes.append(seat_plane)

        return planes

    def _lift_lab_work_surface_to_bbox_top(
        self,
        parent_name: str,
        obj: Dict[str, Any],
        object_type: str,
        plane: Plane,
    ) -> Plane:
        """Prevent lab bench planes from following an under-modeled top slab.

        Stage 6 sometimes decomposes a 0.9 m high lab bench into a low cabinet
        body and places the named countertop at ~0.65-0.77 m. Small objects then
        correctly sit on that *wrong* plane and visibly intersect the furniture
        volume. For lab benches/counters, Stage 3/4 bbox top is the stable
        invariant: horizontal surface props should rest at bbox_top.
        """
        lab_surface_tokens = (
            "lab_bench", "laboratory_bench", "island_bench", "wall_bench",
            "bench", "workbench", "worktop", "counter", "countertop",
            "prep_bench", "preparation_bench",
        )
        parent_norm = self._norm(parent_name)
        if not (
            self._matches_any(object_type, lab_surface_tokens)
            or self._matches_any(parent_norm, lab_surface_tokens)
        ):
            return plane

        center = obj.get("center", obj.get("center_location", [0, 0, 0]))
        dims = obj.get("bounding_dimensions") or obj.get("dimensions")
        if not (
            isinstance(center, (list, tuple)) and len(center) >= 3 and
            isinstance(dims, (list, tuple)) and len(dims) >= 3
        ):
            described = self._get_describe_bbox(parent_name)
            if described:
                center, dims = described
            else:
                return plane

        try:
            bbox_top = float(center[2]) + float(dims[2]) / 2.0
        except (TypeError, ValueError, IndexError):
            return plane

        # Only lift upward. If the detailed part is above the bbox top (e.g.
        # real raised shelf), keep the detailed plane.
        if bbox_top <= plane.world_center[2] + 0.03:
            return plane

        if self.verbose:
            print(
                f"   ^ lift lab surface plane {plane.plane_id}: "
                f"{plane.world_center[2]:.3f} -> {bbox_top:.3f}"
            )
        return Plane(
            plane_id=plane.plane_id,
            parent_name=plane.parent_name,
            plane_type=plane.plane_type,
            world_center=(plane.world_center[0], plane.world_center[1], bbox_top),
            size_wd=plane.size_wd,
            orientation_rad=plane.orientation_rad,
            source_part_name=plane.source_part_name,
            parent_note=f"{plane.parent_note}; lifted to bbox top",
        )

    # ---------- plane type detectors ----------

    def _detect_top_plane(self, parent_name: str, processed: List[dict],
                          yaw: float) -> Optional[Plane]:
        """Pick the single flat part with the highest top face."""
        if not processed:
            return None

        # Soft preference: any of these in a part name boosts ranking when
        # multiple thin candidates compete.
        preferred_keywords = (
            "tabletop", "top_panel", "top_surface", "main_top",
            "desktop", "counter_top", "counter", "surface", "top",
        )

        # Strict whitelist that lets a part BYPASS the 0.10 m thickness
        # gate. Only unambiguous "this part IS the work surface" tokens go
        # here. We explicitly do NOT include the generic word "top" — that
        # also matches `hood_top_cap`, `lid_top`, `cover_top` which can be
        # 0.2-0.3 m thick housings that sit at 2 m world Z and must NOT be
        # selected as a placement plane. Real-world worktops can be 0.15-
        # 0.25 m thick (slab over a cabinet body), and optical breadboards
        # range up to 0.2 m, hence the bypass for `slab` / `breadboard` /
        # `worktop` / `tabletop` / `bench_top` / `countertop` / etc.
        thickness_bypass_keywords = (
            "worktop", "work_top",
            "tabletop", "table_top",
            "bench_top", "benchtop",
            "counter_top", "countertop",
            "desktop", "desk_top",
            "work_surface", "work_area",
            "slab",
            "breadboard",
        )
        # Anti-keywords: even if a strong token matches, parts whose name
        # also contains one of these are clearly housings/caps/covers and
        # should NEVER bypass the thickness gate.
        bypass_blocklist = (
            "cap", "cover", "housing", "casing", "enclosure", "roof",
            "lid", "shroud", "hood",
        )

        # Hard disqualifiers: parts whose name marks them as plumbing /
        # vent / cooktop features. They satisfy the geometric tests
        # (thin, horizontal, large enough) but are NOT valid placement
        # surfaces. The motivating bug: on a lab `Wall_Bench`, Stage 6
        # emits both a 3.6m `benchtop` (real worktop) AND a 0.55m
        # `sink_rim` sitting 2 cm higher; the rim wrongly beat the
        # benchtop in the top-plane ranking and the bench rendered
        # almost empty. Match is lowercase substring.
        plane_disqualifiers = (
            "sink", "basin", "drain", "drainboard",
            "faucet", "spout",
            "grille", "burner", "cooktop", "stovetop",
        )

        candidates = []
        for p in processed:
            if p["type"] != "box":
                continue
            dx, dy, dz = p["dim"]
            name_lower = p["name"].lower()
            # Hard reject sink rims / drain plates / burner caps / vent
            # grilles. These are sibling features built into the SAME
            # parent as the real worktop; they must never claim the
            # parent's top plane.
            if any(dq in name_lower for dq in plane_disqualifiers):
                continue
            name_score = 0
            for kw in preferred_keywords:
                if kw in name_lower:
                    name_score = 2
                    break
            allow_thick = (
                any(kw in name_lower for kw in thickness_bypass_keywords)
                and not any(bk in name_lower for bk in bypass_blocklist)
            )
            if not allow_thick and dz > TOP_PANEL_MAX_THICKNESS_M:
                continue
            if dx < MIN_PLANE_EDGE_M or dy < MIN_PLANE_EDGE_M:
                continue
            # Ranking key (highest first):
            #   1) name_score  — strong "this IS the work surface"
            #                    keyword (worktop / tabletop / top / ...)
            #   2) area        — larger top face wins among equal names
            #   3) top_world_z — only as a final tiebreaker
            #
            # The previous order was (top_world_z, name_score), which
            # let a 0.55x0.45 `sink_rim` sitting 2 cm above a 3.6x0.75
            # `benchtop` win the selection. Promoting name_score and
            # area above z fixes that without regressing the common
            # case (a single named worktop on a console / fridge top).
            area = dx * dy
            candidates.append((name_score, area, p["top_world_z"], p))

        if not candidates:
            return None

        candidates.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
        _ns, _area, best_z, best = candidates[0]

        wx, wy = best["top_world_xy"]
        return Plane(
            plane_id=f"{parent_name}__top",
            parent_name=parent_name,
            plane_type="top",
            world_center=(wx, wy, best_z),
            size_wd=(best["dim"][0], best["dim"][1]),
            orientation_rad=yaw,
            source_part_name=best["name"],
            parent_note="primary top surface",
        )

    def _detect_shelf_planes(self, parent_name: str, processed: List[dict],
                             yaw: float,
                             exclude_part: Optional[str] = None) -> List[Plane]:
        """Emit one plane per internal shelf layer of an open-shelf object.

        `exclude_part` is the name of the part already reported as the object's
        top plane — we never want to double-count the tabletop as shelf 1.
        """
        if not processed:
            return []

        # Infer the object's bounding box from parts (largest extents).
        max_dx = max(p["dim"][0] for p in processed)
        max_dy = max(p["dim"][1] for p in processed)
        if max_dx <= 0 or max_dy <= 0:
            return []

        shelf_like: List[dict] = []
        for p in processed:
            if p["type"] != "box":
                continue
            if exclude_part and p["name"] == exclude_part:
                continue
            dx, dy, dz = p["dim"]
            if dz > SHELF_MAX_THICKNESS_M:
                continue
            # A shelf must span most of the bounding box footprint, otherwise
            # it is likely an apron / stretcher / slat.
            if dx < SHELF_MIN_COVERAGE * max_dx:
                continue
            if dy < SHELF_MIN_COVERAGE * max_dy:
                continue
            if dx < MIN_PLANE_EDGE_M or dy < MIN_PLANE_EDGE_M:
                continue
            name_lower = p["name"].lower()
            # Reject things that are clearly not a shelf even if they pass the
            # geometric filter (for example bottom base plates).
            if any(k in name_lower for k in ("base_plate", "bottom", "floor", "apron",
                                              "stretcher", "slat", "kickplate",
                                              "backpanel", "back_panel")):
                continue
            shelf_like.append(p)

        # A single-tier table / console has exactly ONE shelf-like part, which
        # is just the tabletop we already emitted as the "top" plane. Require
        # at least one shelf part to survive AFTER excluding the top part,
        # otherwise the object is not multi-level and has no internal shelves.
        if not shelf_like:
            return []

        shelf_like.sort(key=lambda p: p["top_world_z"])

        planes: List[Plane] = []
        total = len(shelf_like)
        for idx, p in enumerate(shelf_like, start=1):
            wx, wy = p["top_world_xy"]
            planes.append(Plane(
                plane_id=f"{parent_name}__shelf_{idx}",
                parent_name=parent_name,
                plane_type="shelf",
                world_center=(wx, wy, p["top_world_z"]),
                size_wd=(p["dim"][0], p["dim"][1]),
                orientation_rad=yaw,
                source_part_name=p["name"],
                parent_note=f"shelf level {idx} of {total}",
            ))
        return planes

    def _detect_seat_plane(self, parent_name: str, processed: List[dict],
                           yaw: float) -> Optional[Plane]:
        """Pick a single 'seat' surface for chairs / sofas / benches."""
        if not processed:
            return None

        # Preferred: a part whose name screams "seat".
        preferred: List[dict] = []
        for p in processed:
            if p["type"] != "box":
                continue
            name_lower = p["name"].lower()
            if any(k in name_lower for k in ("seat_cushion", "cushion_seat",
                                              "seat_pad", "seat_base",
                                              "seat", "cushion")):
                # A backrest is often named 'back_cushion' — exclude it.
                if any(b in name_lower for b in ("back_cushion", "backrest",
                                                  "back_pillow", "lumbar")):
                    continue
                preferred.append(p)

        candidates: List[dict] = []
        if preferred:
            candidates = preferred
        else:
            # Fallback: any reasonably large horizontal box whose top lies in a
            # plausible seat-height range.
            for p in processed:
                if p["type"] != "box":
                    continue
                dx, dy, dz = p["dim"]
                if dx < MIN_PLANE_EDGE_M or dy < MIN_PLANE_EDGE_M:
                    continue
                if dz > 0.20:
                    continue  # likely the whole body, not a seat cushion
                if SEAT_Z_WORLD_MIN <= p["top_world_z"] <= SEAT_Z_WORLD_MAX:
                    candidates.append(p)

        if not candidates:
            return None

        # Prefer the largest-area candidate.
        best = max(candidates, key=lambda p: p["dim"][0] * p["dim"][1])
        wx, wy = best["top_world_xy"]
        return Plane(
            plane_id=f"{parent_name}__seat",
            parent_name=parent_name,
            plane_type="seat",
            world_center=(wx, wy, best["top_world_z"]),
            size_wd=(best["dim"][0], best["dim"][1]),
            orientation_rad=yaw,
            source_part_name=best["name"],
            parent_note="seat surface",
        )


# ==============================================================================
# Color palette used when the LLM gives only a qualitative color_hint.
# Stage 10 (stage10_material) will later rewrite these with proper PBR materials;
# we just want the preview render to not be monochrome.
# ==============================================================================
_COLOR_PALETTE: Dict[str, Tuple[float, float, float, float]] = {
    "white":        (0.92, 0.92, 0.90, 1.0),
    "cream":        (0.94, 0.90, 0.80, 1.0),
    "beige":        (0.88, 0.80, 0.68, 1.0),
    "gray":         (0.60, 0.60, 0.62, 1.0),
    "grey":         (0.60, 0.60, 0.62, 1.0),
    "black":        (0.10, 0.10, 0.10, 1.0),
    "brown":        (0.42, 0.28, 0.18, 1.0),
    "dark_brown":   (0.26, 0.16, 0.10, 1.0),
    "wood":         (0.55, 0.36, 0.22, 1.0),
    "oak":          (0.70, 0.52, 0.32, 1.0),
    "walnut":       (0.32, 0.20, 0.12, 1.0),
    "red":          (0.70, 0.15, 0.15, 1.0),
    "terracotta":   (0.78, 0.38, 0.22, 1.0),
    "orange":       (0.88, 0.52, 0.18, 1.0),
    "yellow":       (0.90, 0.78, 0.28, 1.0),
    "gold":         (0.80, 0.65, 0.28, 1.0),
    "green":        (0.30, 0.55, 0.35, 1.0),
    "olive":        (0.48, 0.48, 0.22, 1.0),
    "sage":         (0.62, 0.70, 0.55, 1.0),
    "blue":         (0.25, 0.40, 0.65, 1.0),
    "navy":         (0.10, 0.18, 0.38, 1.0),
    "teal":         (0.22, 0.52, 0.55, 1.0),
    "pink":         (0.92, 0.70, 0.75, 1.0),
    "purple":       (0.50, 0.35, 0.62, 1.0),
    "copper":       (0.72, 0.42, 0.22, 1.0),
    "brass":        (0.78, 0.62, 0.32, 1.0),
    "silver":       (0.78, 0.80, 0.82, 1.0),
    "ceramic":      (0.90, 0.88, 0.82, 1.0),
    "glass":        (0.80, 0.86, 0.90, 0.6),
    "plant":        (0.30, 0.50, 0.30, 1.0),
    "leaf":         (0.30, 0.55, 0.30, 1.0),
    "fabric":       (0.82, 0.76, 0.66, 1.0),
    "linen":        (0.88, 0.82, 0.72, 1.0),
    "velvet":       (0.35, 0.22, 0.38, 1.0),
    "default":      (0.70, 0.62, 0.52, 1.0),
}


def _color_for_hint(hint: str) -> Tuple[float, float, float, float]:
    """Pick a palette entry whose keyword appears in the hint string."""
    if not hint:
        return _COLOR_PALETTE["default"]
    h = hint.lower()
    # Longest-key-first so "dark_brown" beats "brown".
    for key in sorted(_COLOR_PALETTE.keys(), key=len, reverse=True):
        if key in h:
            return _COLOR_PALETTE[key]
    return _COLOR_PALETTE["default"]


# ==============================================================================
# Runner
# ==============================================================================
class StageSmallObjectsRunner:
    """Orchestrate plane discovery -> LLM decoration -> code generation."""

    def __init__(
        self,
        image_path: Optional[str] = None,
        geometry_json_path: Optional[str] = None,
        describe_json_path: Optional[str] = None,
        base_code_path: Optional[str] = None,
        stage1_json_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        use_memory: bool = True,
        verbose: bool = True,
        memory_file: str = "agent_memory.jsonl",
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        skip_llm: bool = False,
        parallel: int = 4,
    ):
        self.image_path = image_path
        self.geometry_json_path = geometry_json_path
        self.describe_json_path = describe_json_path
        self.base_code_path = base_code_path
        self.stage1_json_path = stage1_json_path
        self.output_dir = output_dir or os.path.join(
            current_dir, "pipeline_output", "stage7_small_objects"
        )
        self.use_memory = use_memory
        self.verbose = verbose
        self.skip_llm = skip_llm
        # Number of parent objects to process in parallel. Each parent object
        # triggers one LLM call; since calls take ~10–30s each, going from
        # sequential to 4-way parallel typically cuts Stage 7 runtime by
        # 3–4x on a typical room (~8–15 parents).
        self.parallel = max(1, int(parallel))

        self.memory = (
            Memory(workspace_dir=current_dir, memory_file=memory_file)
            if use_memory else None
        )
        # Only instantiate the LLM when we will actually need it; this keeps
        # `--dry-run` (plane-discovery only) fast and offline.
        self._llm_kwargs = dict(model=model, base_url=base_url, api_key=api_key)
        self._llm: Optional[LLMClient] = None
        # Guards the first-time init of ``self._llm`` when parallel workers
        # may race on the lazy property below.
        import threading as _threading
        self._llm_lock = _threading.Lock()
        # Serializes `print` so interleaved logs from parallel LLM workers
        # don't tear multi-line messages.
        self._log_lock = _threading.Lock()

        # Loaded state.
        self.detailed_geometry: Dict[str, Any] = {}
        self.describe_objects: List[Dict[str, Any]] = []
        self.room_style: Dict[str, Any] = {}
        self.stage1_data: Dict[str, Any] = {}
        self.base_code: str = ""
        self.planes: List[Plane] = []
        self.items: List[SmallObjectItem] = []
        # Walls parsed from the Stage 6 base code via AST. Each entry is
        # {"name": "Wall_North"|"Window_East"|..., "world_center": [x,y,z],
        #  "world_size": [dx,dy,dz]}. Used by `_walls_for_plane` to expose
        # nearby walls in plane-local coordinates so the LLM can infer which
        # side of a desk/bench is "back-against-wall" vs "user-facing".
        self.walls: List[Dict[str, Any]] = []
        # Per-plane "obstacle" list — existing objects already resting on that
        # plane (e.g. a Monitor / Keyboard / Lamp already placed on a desk top
        # by Stage 3). Populated after plane discovery and consumed by both
        # the LLM prompt and the resolver to avoid overlapping placements.
        # Key: plane_id, Value: list of dicts with keys
        #   {name, object_type, local_uv, size_wd, uv_half_extents, bottom_z}
        self.plane_occupants: Dict[str, List[Dict[str, Any]]] = {}
        # v3 (2026-05-01): minor placeholders pre-placed by Stage 4 Phase B.
        # Populated by `_load_stage4_minor_placed`. Each item is a dict with
        # keys {obj_id, label, parent_id, parent_label, placement_type,
        # block_name, ...}. Consumed by `_build_user_payload` so the LLM
        # avoids placing duplicate items (e.g. a second table-lamp on the
        # same parent).
        self.stage4_minor_placed: List[Dict[str, Any]] = []
        # Coarse de-dup helpers: lowercase label tokens of placed minors.
        # Used in prompt to tell the LLM "skip these item-types".
        self.stage4_placed_labels: List[str] = []
        # Per-parent-label set of placed minor labels (lowercase) — lets us
        # tell the LLM "on parent X, lamp & vase already exist; do not add
        # another one."
        self.stage4_placed_by_parent: Dict[str, List[str]] = {}

    # ---------------- logging ----------------

    def _log(self, msg: str, level: str = "info") -> None:
        if not self.verbose:
            return
        prefix = {
            "info": "ℹ️", "success": "✅", "warning": "⚠️",
            "error": "❌", "step": "📋", "plane": "🟩",
            "item": "🧸", "save": "💾",
        }.get(level, "")
        # Hold a lock so concurrent workers don't interleave fragments of the
        # same line. print() itself isn't reentrant-safe w.r.t. buffered
        # output on all platforms; this keeps console output readable.
        with self._log_lock:
            print(f"{prefix} {msg}")

    # ---------------- lazy LLM ----------------

    @property
    def llm(self) -> LLMClient:
        # Double-checked locking: in parallel mode multiple workers may hit
        # this property before the singleton is created. Without the lock
        # we'd spin up several LLMClient instances simultaneously, each of
        # which opens its own HTTP pool.
        if self._llm is None:
            with self._llm_lock:
                if self._llm is None:
                    self._llm = LLMClient(**self._llm_kwargs)
        return self._llm

    # ---------------- data loading ----------------

    def _candidate_stage_dirs(self, stage_folder: str) -> List[str]:
        """Return possible directories holding `<stage_folder>` outputs, in
        priority order that prefers the *current run*.

        Priority:
        1. Dir recorded in Memory metadata (`output_file`) of that stage —
           this is the strongest signal because it was written by the most
           recent run that actually produced the file.
        2. Sibling of `self.output_dir` (i.e. `<run_dir>/<stage_folder>`) —
           this works when Memory metadata is missing or when Memory is
           disabled but the run directory layout is conventional.
        3. Global `agent_utils/pipeline_output/<stage_folder>` — legacy
           fallback for ad-hoc runs outside a run directory.

        Duplicates are removed while preserving order.
        """
        candidates: List[str] = []
        stage_key = stage_folder  # e.g. "stage6_geometry"

        if self.use_memory and self.memory is not None:
            try:
                entry = self.memory.get_latest(stage=stage_key, type="result")
            except Exception:
                entry = None
            if entry and getattr(entry, "metadata", None):
                out_file = entry.metadata.get("output_file")
                if isinstance(out_file, str) and out_file:
                    candidates.append(os.path.dirname(out_file))

        if self.output_dir:
            sibling = os.path.join(
                os.path.dirname(os.path.abspath(self.output_dir)),
                stage_folder,
            )
            candidates.append(sibling)

        candidates.append(
            os.path.join(current_dir, "pipeline_output", stage_folder)
        )

        seen = set()
        unique: List[str] = []
        for d in candidates:
            if d and d not in seen:
                seen.add(d)
                unique.append(d)
        return unique

    def _load_geometry(self) -> bool:
        data: Optional[Dict[str, Any]] = None
        if self.geometry_json_path and os.path.exists(self.geometry_json_path):
            with open(self.geometry_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._log(f"Geometry JSON: {self.geometry_json_path}", "success")
        elif self.use_memory:
            # `geometry_progress.json` is the rich parts data (Memory stores
            # only the integrated code, not the progress file), so we must
            # resolve its path from the run layout. `_candidate_stage_dirs`
            # prefers the run directory that actually produced Stage 6.
            for d in self._candidate_stage_dirs("stage6_geometry"):
                candidate = os.path.join(d, "geometry_progress.json")
                if os.path.exists(candidate):
                    with open(candidate, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    self._log(f"Geometry JSON: {candidate}", "success")
                    break
            if data is None:
                self._log(
                    "Geometry JSON not found in any candidate stage6_geometry "
                    "directory (run-dir sibling / Memory metadata / global).",
                    "error",
                )
                return False
        else:
            self._log("Geometry JSON not supplied and Memory disabled.", "error")
            return False

        # Convert `detailed_objects` list to DETAILED_GEOMETRY-style dict.
        detailed_objs = data.get("detailed_objects", [])
        dg: Dict[str, Any] = {}
        for obj in detailed_objs:
            if not obj.get("generated", False):
                continue
            nm = obj.get("name")
            if not nm:
                continue
            dg[nm] = {
                "center": obj.get("center_location", [0, 0, 0]),
                "rotation": obj.get("base_rotation", [0, 0, 0]),
                "bounding_dimensions": obj.get("bounding_dimensions", [0, 0, 0]),
                "parts": obj.get("parts", []),
                "object_type": obj.get("object_type", ""),
                "description": obj.get("description", ""),
                "material_description": obj.get("material_description", ""),
                "color_description": obj.get("color_description", ""),
            }
        self.detailed_geometry = dg
        self.room_style = data.get("room_style", {})
        self._log(f"Loaded {len(dg)} generated objects.", "success")
        return len(dg) > 0

    def _load_describe(self) -> None:
        """Optional: richer per-object descriptions for LLM grounding."""
        data: Optional[Dict[str, Any]] = None
        if self.describe_json_path and os.path.exists(self.describe_json_path):
            with open(self.describe_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._log(f"Describe JSON: {self.describe_json_path}", "info")
        elif self.use_memory:
            entry = self.memory.get_latest(stage="stage5_describe", type="result")
            if entry:
                try:
                    data = json.loads(entry.content)
                    self._log("Describe JSON: from Memory", "info")
                except json.JSONDecodeError:
                    data = None
        if data:
            self.describe_objects = data.get("objects", [])
            if not self.room_style:
                self.room_style = data.get("room_style", {})

    def _load_base_code(self) -> bool:
        if self.base_code_path and os.path.exists(self.base_code_path):
            with open(self.base_code_path, "r", encoding="utf-8") as f:
                self.base_code = f.read()
            self._log(f"Base code: {self.base_code_path}", "success")
            return True
        # Resolve `geometry_output.py` against the same run directory that
        # produced Stage 6; see `_candidate_stage_dirs` for the priority
        # order. This is what makes `--run-dir` resumption pick up the
        # correct integrated code instead of stale global leftovers.
        for d in self._candidate_stage_dirs("stage6_geometry"):
            candidate = os.path.join(d, "geometry_output.py")
            if os.path.exists(candidate):
                with open(candidate, "r", encoding="utf-8") as f:
                    self.base_code = f.read()
                self.base_code_path = candidate
                self._log(f"Base code: {candidate}", "success")
                return True
        self._log(
            "Base code (geometry_output.py) not found in any candidate "
            "stage6_geometry directory.",
            "error",
        )
        return False

    def _load_image(self) -> bool:
        if not self.image_path and self.use_memory:
            entry = self.memory.get_latest(stage="stage1", type="result")
            if entry:
                self.image_path = entry.metadata.get("image_path")
        if self.image_path and os.path.exists(self.image_path):
            self._log(f"Reference image: {self.image_path}", "success")
            return True
        self._log("Reference image missing — LLM step will be skipped.", "warning")
        return False

    def _load_stage1_hints(self) -> None:
        """Pull Stage 1/2 small-object candidates as weak hints for the LLM."""
        data: Optional[Dict[str, Any]] = None
        if self.stage1_json_path and os.path.exists(self.stage1_json_path):
            with open(self.stage1_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        elif self.use_memory:
            entry = self.memory.get_latest(stage="stage1", type="result")
            if entry:
                if isinstance(entry.content, dict):
                    data = entry.content
                elif isinstance(entry.content, str):
                    try:
                        data = json.loads(entry.content)
                    except json.JSONDecodeError:
                        data = None
        self.stage1_data = data or {}

    def _stage1_hint_list(self) -> List[str]:
        """Extract `category == "minor"` object names from Stage 1 hierarchy."""
        if not self.stage1_data:
            return []
        hints: List[str] = []
        for zone in self.stage1_data.get("decoupled_zones", []) or []:
            for obj in zone.get("object_hierarchy", []) or []:
                if str(obj.get("category", "")).lower() == "minor":
                    nm = obj.get("name")
                    if nm:
                        hints.append(str(nm))
        return hints

    # ---------------- v3: stage4 minor avoidance ----------------

    def _load_stage4_minor_placed(self) -> None:
        """Load list of minor obj_ids already placed (as bbox) by Stage 4 Phase B.

        Lookup priority:
          1. Memory: stage="stage4", type="result" — metadata fields
             ``minor_placed_obj_ids`` and the richer ``minor_placed_json``.
          2. Disk: ``stage4_minor_placed.json`` next to the active Stage 4
             output directory (resolved via the same `_candidate_stage_dirs`
             logic used for `geometry_output.py`).

        Failure is silent (empty list) — if Stage 4 didn't run with v3 we
        simply have no avoid-list, which matches pre-v3 behaviour.
        """
        items: List[Dict[str, Any]] = []
        # 1) Memory
        if self.use_memory and self.memory is not None:
            try:
                entry = self.memory.get_latest(stage="stage4", type="result")
                if entry:
                    meta = entry.metadata or {}
                    json_path = meta.get("minor_placed_json")
                    if json_path and os.path.exists(json_path):
                        with open(json_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        items = data.get("items") or []
                    elif meta.get("minor_placed_obj_ids"):
                        # Fallback: only ids in metadata, no rich items.
                        items = [
                            {"obj_id": oid}
                            for oid in meta.get("minor_placed_obj_ids", [])
                        ]
            except Exception:  # noqa: BLE001
                pass
        # 2) Disk fallback (when no memory or memory missed)
        if not items:
            for d in self._candidate_stage_dirs("stage4"):
                cand = os.path.join(d, "stage4_minor_placed.json")
                if not os.path.exists(cand):
                    continue
                try:
                    with open(cand, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    items = data.get("items") or []
                    if items:
                        break
                except Exception:  # noqa: BLE001
                    continue

        self.stage4_minor_placed = items or []
        # Build de-dup helpers (label-only — obj_id matching is unreliable
        # because LLM in this stage decides item names from scratch).
        labels: List[str] = []
        by_parent: Dict[str, List[str]] = {}
        for it in self.stage4_minor_placed:
            lab = (it.get("label") or "").strip().lower()
            if not lab:
                continue
            labels.append(lab)
            parent = (it.get("parent_label") or "").strip().lower()
            if parent:
                by_parent.setdefault(parent, []).append(lab)
        self.stage4_placed_labels = labels
        self.stage4_placed_by_parent = by_parent

        if self.stage4_minor_placed:
            self._log(
                f"Stage4 minor avoid-list: {len(self.stage4_minor_placed)} "
                f"placeholders already placed (will be skipped here).",
                "info",
            )

    # ============================================================
    # v3 (2026-05-02): Stage 4 wall-decor plane synthesis
    # ============================================================
    # Stage 6 deliberately skips Stage 4 wall-mounted objects (they stay
    # as simple bbox in the rendered scene). That means floating shelves /
    # wall cabinets / spice racks never enter DETAILED_GEOMETRY, so the
    # regular PlaneFinder finds 0 planes on them and 0 small objects get
    # placed. The two methods below patch that:
    #
    #   _load_stage4_wall_object_geoms  -- read Stage 4 wall_objects.json
    #     and parse each Wall_* `create_box(...)` call out of self.base_code
    #     to recover (location, dim, rotation).
    #
    #   _synthesize_wall_decor_planes   -- whitelist filter via the
    #     stage5_describe object_type, compute the world-space TOP face of
    #     each whitelisted wall item (handles rotations like (pi/2, 0, *)
    #     correctly via 8-corner AABB), and emit a Plane object. Also
    #     injects a single-part DETAILED_GEOMETRY-style entry into
    #     self.detailed_geometry so downstream methods that iterate it
    #     (e.g. occupant detection, nearby-bbox payload) keep working.
    #
    # NOTE: pure decorations (wall art, mirror, clock, sconce, mounted TV,
    # pot rack, hook rail) are filtered out by WALL_DECOR_BLACKLIST_TYPES
    # — there is no usable surface on those.

    def _load_stage4_wall_object_geoms(self) -> Dict[str, Dict[str, Any]]:
        """Resolve {wall_obj_name -> {location, dim, rotation, created_via}}
        for every Wall_* item Stage 4 added.

        Source of truth for the *names* is `wall_objects.json` (Memory
        metadata or disk). The geometry comes from a regex over
        `self.base_code` because Stage 4 doesn't persist loc/dim/rot
        directly in the json — it just lists names.
        """
        # 1) Resolve the name list (mirrors stage6_geometry._load_wall_object_names)
        names: set = set()
        if self.use_memory and self.memory is not None:
            try:
                entry = self.memory.get_latest(stage="stage4", type="result")
                if entry:
                    meta = entry.metadata or {}
                    direct = meta.get("wall_object_names")
                    if direct:
                        names = set(direct)
                    elif meta.get("wall_objects_json") and \
                            os.path.exists(meta["wall_objects_json"]):
                        with open(meta["wall_objects_json"], "r",
                                  encoding="utf-8") as f:
                            names = set(json.load(f).get(
                                "wall_object_names", []))
            except Exception:  # noqa: BLE001
                pass
        if not names:
            for d in self._candidate_stage_dirs("stage4"):
                cand = os.path.join(d, "wall_objects.json")
                if not os.path.exists(cand):
                    continue
                try:
                    with open(cand, "r", encoding="utf-8") as f:
                        names = set(json.load(f).get("wall_object_names", []))
                    if names:
                        break
                except Exception:  # noqa: BLE001
                    continue
        if not names:
            return {}
        if not self.base_code:
            self._log(
                "Wall-decor: have wall_object_names but no base_code yet "
                "(skip).",
                "warning",
            )
            return {}

        # 2) For each name, find its create_box / create_cylinder call in
        #    base_code and pull location, dim, rotation tuples. We rely on
        #    the same numeric-tuple extractor as the wall-parser path.
        out: Dict[str, Dict[str, Any]] = {}
        import re as _re
        # Pattern grabs: shape, name, then a parenthesised body up to the
        # closing `)`. We deliberately use a NON-greedy body and match a
        # name exactly (escape) to avoid collisions.
        for nm in names:
            pat = _re.compile(
                r'create_(box|cylinder)\(\s*"' + _re.escape(nm) + r'"'
                r'\s*,\s*\(([^)]*)\)\s*'
                r',\s*\(([^)]*)\)'
                r'(?:\s*,\s*rotation\s*=\s*\(([^)]*)\))?',
                _re.DOTALL,
            )
            m = pat.search(self.base_code)
            if not m:
                continue
            shape = m.group(1)
            loc = self._safe_eval_tuple(m.group(2), arity=3)
            dim = self._safe_eval_tuple(m.group(3), arity=3)
            rot_raw = m.group(4) or "0, 0, 0"
            rot = self._safe_eval_tuple(rot_raw, arity=3)
            if loc is None or dim is None or rot is None:
                continue
            out[nm] = {
                "location": loc,
                "dim": dim,
                "rotation": rot,
                "shape": shape,
            }
        return out

    @staticmethod
    def _safe_eval_tuple(src: str, arity: int = 3) -> Optional[Tuple[float, ...]]:
        """Eval a comma-separated numeric expression like '1.0, 0.05, 0.25'
        or '-SCENE_W/2 + 0.1, 0, math.pi/2' to a float tuple. Limited to
        constants, +-*/, math.pi etc.
        """
        import ast
        import math as _math
        try:
            tree = ast.parse(src.strip(), mode="eval")
        except SyntaxError:
            try:
                # Wrap as a tuple expression; covers the bare comma list
                # inside `(...)` we already stripped one layer of parens off.
                tree = ast.parse("(" + src.strip() + ")", mode="eval")
            except SyntaxError:
                return None

        allowed = {
            "math": _math, "pi": _math.pi, "tau": _math.tau,
            "sin": _math.sin, "cos": _math.cos, "sqrt": _math.sqrt,
        }

        def _ev(node):
            if isinstance(node, ast.Expression):
                return _ev(node.body)
            if isinstance(node, ast.Num):  # py<3.8
                return node.n
            if isinstance(node, ast.Constant):
                return node.value
            if isinstance(node, ast.UnaryOp) and isinstance(
                    node.op, (ast.USub, ast.UAdd)):
                v = _ev(node.operand)
                return -v if isinstance(node.op, ast.USub) else +v
            if isinstance(node, ast.BinOp):
                l = _ev(node.left); r = _ev(node.right)
                if isinstance(node.op, ast.Add): return l + r
                if isinstance(node.op, ast.Sub): return l - r
                if isinstance(node.op, ast.Mult): return l * r
                if isinstance(node.op, ast.Div): return l / r
                if isinstance(node.op, ast.Mod): return l % r
                if isinstance(node.op, ast.Pow): return l ** r
                raise ValueError("op")
            if isinstance(node, ast.Name):
                if node.id in allowed:
                    return allowed[node.id]
                # SCENE_W / SCENE_D / WALL_T are unknown here; treat as 0
                # so we still get a (possibly imprecise) numeric. Caller
                # will fall back to 0 in those rare cases.
                return 0.0
            if isinstance(node, ast.Attribute):
                obj = _ev(node.value)
                return getattr(obj, node.attr)
            if isinstance(node, ast.Tuple):
                return tuple(_ev(e) for e in node.elts)
            if isinstance(node, ast.Call):
                fn = _ev(node.func)
                args = [_ev(a) for a in node.args]
                return fn(*args)
            raise ValueError(f"unsupported node {type(node).__name__}")

        try:
            v = _ev(tree)
        except Exception:  # noqa: BLE001
            return None
        if isinstance(v, (int, float)):
            v = (v,)
        if not isinstance(v, tuple):
            return None
        if len(v) != arity:
            return None
        try:
            return tuple(float(x) for x in v)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _box_world_aabb(loc: Tuple[float, float, float],
                        dim: Tuple[float, float, float],
                        rot: Tuple[float, float, float]
                        ) -> Tuple[float, float, float, float, float, float]:
        """8-corner world AABB of a box with XYZ Euler rotation (Blender).

        Returns (x_min, x_max, y_min, y_max, z_min, z_max).
        """
        dx, dy, dz = dim
        # Local-frame corner offsets (half-extents).
        corners = [
            (sx * dx / 2.0, sy * dy / 2.0, sz * dz / 2.0)
            for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)
        ]
        rx, ry, rz = rot
        cx, sx_ = math.cos(rx), math.sin(rx)
        cy, sy_ = math.cos(ry), math.sin(ry)
        czr, szr = math.cos(rz), math.sin(rz)
        # XYZ Euler (Blender default): R = Rz @ Ry @ Rx
        R = [
            [cy * czr,
             sx_ * sy_ * czr - cx * szr,
             cx * sy_ * czr + sx_ * szr],
            [cy * szr,
             sx_ * sy_ * szr + cx * czr,
             cx * sy_ * szr - sx_ * czr],
            [-sy_, sx_ * cy, cx * cy],
        ]
        wxs, wys, wzs = [], [], []
        for px, py, pz in corners:
            wx = R[0][0] * px + R[0][1] * py + R[0][2] * pz + loc[0]
            wy = R[1][0] * px + R[1][1] * py + R[1][2] * pz + loc[1]
            wz = R[2][0] * px + R[2][1] * py + R[2][2] * pz + loc[2]
            wxs.append(wx); wys.append(wy); wzs.append(wz)
        return (min(wxs), max(wxs), min(wys), max(wys), min(wzs), max(wzs))

    @staticmethod
    def _wall_decor_object_type(name: str,
                                describe_by_name: Dict[str, Dict[str, Any]]
                                ) -> str:
        """Lookup object_type via stage5_describe; fallback to lowercased name."""
        # describe_by_name uses normalized keys (lowercase, '_'-separated).
        nm_key = name.lower().replace(" ", "_").strip("_")
        entry = describe_by_name.get(nm_key)
        if entry:
            ot = str(entry.get("object_type", "")).lower().strip()
            if ot:
                return ot
        return name.lower().replace("_", " ")

    def _synthesize_wall_decor_planes(self) -> List[Plane]:
        """Discover top planes on Stage 4 wall items (shelf / cabinet / rack).

        Side-effects:
          - Appends a single-part DETAILED_GEOMETRY-style entry for each
            whitelisted wall item to ``self.detailed_geometry`` so any
            downstream code that iterates the dict (occupant detection,
            nearby-bbox payload) treats these items as real parents.
            We never overwrite an existing entry — if the same name
            already came from Stage 6 (rare for wall items, but possible
            if Stage 6 ever stops skipping them) we leave it alone.

        Returns:
            List of newly-synthesised Plane objects to append to
            ``self.planes``.
        """
        geoms = self._load_stage4_wall_object_geoms()
        if not geoms:
            return []

        # Build a normalized describe lookup once.
        describe_by_name: Dict[str, Dict[str, Any]] = {}
        for entry in self.describe_objects or []:
            nm = str(entry.get("name", ""))
            if nm:
                key = nm.lower().replace(" ", "_").strip("_")
                describe_by_name[key] = entry

        new_planes: List[Plane] = []
        skipped_blacklist: List[str] = []
        skipped_unknown: List[str] = []
        skipped_too_small: List[str] = []

        for name, geom in geoms.items():
            if name in self.detailed_geometry:
                # Stage 6 produced its own DETAILED_GEOMETRY for this name
                # — let the regular PlaneFinder handle it; do nothing here.
                continue

            object_type = self._wall_decor_object_type(name, describe_by_name)

            # Blacklist check first (cheap, conservative): wall art / mirror /
            # clock / sconce / TV / pot rack -> SKIP.
            if any(tok in object_type for tok in WALL_DECOR_BLACKLIST_TYPES):
                skipped_blacklist.append(f"{name} (type={object_type!r})")
                continue
            if not any(tok in object_type for tok in WALL_DECOR_TOP_TYPES):
                skipped_unknown.append(f"{name} (type={object_type!r})")
                continue

            loc = geom["location"]
            dim = geom["dim"]
            rot = geom["rotation"]
            x_min, x_max, y_min, y_max, _z_min, z_max = self._box_world_aabb(
                loc, dim, rot)
            top_w = x_max - x_min
            top_d = y_max - y_min
            if top_w < MIN_PLANE_EDGE_M or top_d < MIN_PLANE_EDGE_M:
                skipped_too_small.append(
                    f"{name} (top={top_w:.2f}x{top_d:.2f}m)")
                continue

            top_cx = (x_min + x_max) / 2.0
            top_cy = (y_min + y_max) / 2.0
            # The plane's own yaw is the box's world Z rotation; the LLM
            # uses this to map plane-local (u, v) to world XY. For wall
            # boards mounted on E/W walls the Stage 4 code rotates by
            # math.pi/2 around Z so the board's long axis runs along Y;
            # respecting that lets the LLM put items "along the shelf".
            yaw = float(rot[2]) if len(rot) >= 3 else 0.0
            # Plane size is reported in plane-local frame: (width along +u,
            # depth along +v). The world AABB is axis-aligned, so when
            # yaw is ±pi/2 we swap to keep (u=long_axis, v=short_axis).
            if abs(math.cos(yaw)) >= abs(math.sin(yaw)):
                # cos dominates → world X aligns with plane +u.
                size_wd = (top_w, top_d)
            else:
                size_wd = (top_d, top_w)

            plane = Plane(
                plane_id=f"{name}__top",
                parent_name=name,
                plane_type="top",
                world_center=(top_cx, top_cy, z_max),
                size_wd=size_wd,
                orientation_rad=yaw,
                source_part_name=name,
                parent_note=f"Stage 4 wall decor (type={object_type!r})",
            )
            new_planes.append(plane)

            # Also inject a virtual DETAILED_GEOMETRY entry so downstream
            # iterators (occupant scan, nearby-bbox payload) see a parent.
            self.detailed_geometry[name] = {
                "center": list(loc),
                "rotation": list(rot),
                "object_type": object_type,
                "description": (
                    "Stage 4 wall-mounted decor (synthesized for plane "
                    "discovery; geometry is a single bbox part)."
                ),
                "material_description": "",
                "color_description": "",
                "_synthesized_wall_decor": True,
                "parts": [{
                    "name": f"{name}_body",
                    "type": "box",
                    "rel_loc": [0.0, 0.0, 0.0],
                    "dim": list(dim),
                }],
            }

        if new_planes:
            self._log(
                f"Wall-decor planes: synthesized {len(new_planes)} top "
                f"plane(s) on Stage 4 wall items "
                f"(parents: {[p.parent_name for p in new_planes]})",
                "success",
            )
        if skipped_blacklist:
            self._log(
                f"Wall-decor: blacklist skip ({len(skipped_blacklist)}): "
                f"{skipped_blacklist[:4]}"
                + ("..." if len(skipped_blacklist) > 4 else ""),
                "info",
            )
        if skipped_unknown:
            self._log(
                f"Wall-decor: unknown object_type, no plane "
                f"({len(skipped_unknown)}): {skipped_unknown[:4]}"
                + ("..." if len(skipped_unknown) > 4 else ""),
                "info",
            )
        if skipped_too_small:
            self._log(
                f"Wall-decor: top face below {MIN_PLANE_EDGE_M:.2f}m "
                f"({len(skipped_too_small)}): {skipped_too_small[:4]}",
                "info",
            )
        return new_planes

    # ---------------- LLM placement ----------------

    @staticmethod
    def _encode_image(path: str) -> Tuple[str, str]:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        ext = os.path.splitext(path)[1].lower()
        mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".png": "image/png"}.get(ext, "image/png")
        return b64, mime

    def _build_planes_payload(self, planes: List[Plane]) -> List[dict]:
        payload = []
        for pl in planes:
            occupants = self.plane_occupants.get(pl.plane_id, [])
            # Trim the occupant payload to only the fields the LLM needs to
            # avoid overlap. We keep `name` + `object_type` for grounding so
            # the model can describe its placement decisions relative to
            # known landmarks ("to the right of the Monitor").
            existing_items = [
                {
                    "name": o["name"],
                    "object_type": o.get("object_type", ""),
                    "local_uv": o["local_uv"],
                    "uv_half_extents": o["uv_half_extents"],
                    "size_wd": o["size_wd"],
                }
                for o in occupants
            ]
            walls = self._walls_for_plane(pl)
            nearby_bboxes = self._nearby_bboxes_for_plane(pl)
            payload.append({
                "plane_id": pl.plane_id,
                "plane_type": pl.plane_type,
                "world_center": list(pl.world_center),
                "size_wd": list(pl.size_wd),
                "orientation_rad": pl.orientation_rad,
                "parent_note": pl.parent_note,
                "existing_items": existing_items,
                "walls": walls,
                "nearby_bboxes": nearby_bboxes,
            })
        return payload

    def _build_user_payload(self, parent_name: str, planes: List[Plane]) -> str:
        parent_entry = self.detailed_geometry.get(parent_name, {})
        object_type = parent_entry.get("object_type", "")
        description = parent_entry.get("description", "")
        material_desc = parent_entry.get("material_description", "")
        color_desc = parent_entry.get("color_description", "")
        hints = self._stage1_hint_list()

        # v3: stage4 has already placed placeholder bbox for some minor objects on this parent.
        # Tell the LLM, to avoid repeating the same type of small object on the same parent.
        # Matching is lowercase substring — parent_name may be "Coffee_Table" while stage4
        # parent_label is "coffee table"; we split both into tokens and compare.
        parent_norm = parent_name.lower().replace("_", " ").strip()
        already_on_parent: List[str] = []
        for placed_parent, labels in self.stage4_placed_by_parent.items():
            if not placed_parent:
                continue
            # bidirectional substring match, tolerates naming differences
            if (placed_parent in parent_norm
                    or parent_norm in placed_parent):
                already_on_parent.extend(labels)

        body = {
            "room_style": self.room_style,
            "parent_object": {
                "name": parent_name,
                "object_type": object_type,
                "description": description,
                "material": material_desc,
                "color": color_desc,
            },
            "planes": self._build_planes_payload(planes),
            "stage1_small_object_hints": hints,
        }
        if already_on_parent:
            body["already_placed_by_stage4"] = sorted(set(already_on_parent))

        prefix = (
            "Decorate the following parent object. Base every decision on the "
            "reference image and room style. Return JSON only.\n"
        )
        if already_on_parent:
            prefix += (
                "\nIMPORTANT: Stage 4 already placed coarse bbox placeholders "
                "for these item-types ON THIS PARENT: "
                f"{sorted(set(already_on_parent))}. Do NOT add another item of "
                "the same type on this parent — pick complementary smaller "
                "items instead (e.g. if a 'table lamp' is already there, you "
                "may add a coaster, a small book, or a tray, but NOT another "
                "lamp).\n"
            )
        return prefix + "\n" + json.dumps(body, ensure_ascii=False, indent=2)

    def _call_llm_for_parent(self, parent_name: str, planes: List[Plane],
                             system_prompt: str) -> Optional[Dict[str, Any]]:
        if not self.image_path or not os.path.exists(self.image_path):
            self._log(f"   skip {parent_name}: no image available", "warning")
            return None

        b64, mime = self._encode_image(self.image_path)
        user_text = self._build_user_payload(parent_name, planes)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=[
                {"type": "image_url",
                 "image_url": {"url": f"data:{mime};base64,{b64}"}},
                {"type": "text", "text": user_text},
            ]),
        ]

        try:
            raw = self.llm.invoke(messages)
        except Exception as exc:  # noqa: BLE001
            self._log(f"   LLM error for {parent_name}: {exc}", "error")
            return None

        # Persist raw response for debugging.
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", parent_name)
            raw_path = os.path.join(self.output_dir, f"raw_{safe}.txt")
            with open(raw_path, "w", encoding="utf-8") as f:
                f.write(raw or "(empty response)")
        except OSError:
            pass

        if not raw:
            return None

        json_str = extract_json_from_response(raw)
        if json_str:
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
        # Fallback: some models wrap JSON in prose.
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            self._log(f"   failed to parse JSON for {parent_name}", "warning")
            return None

    # ---------------- wall + nearby-bbox parsing (orientation hints) ------

    def _parse_walls_from_basecode(self) -> None:
        """Extract Wall_*/Window_* AABBs from the Stage 6 base code via AST.

        Stage 3/4 emits walls as e.g.::

            create_box("Wall_North",
                       (0, SCENE_D/2 + WALL_T/2, WALL_H/2),
                       (SCENE_W + 2*WALL_T, WALL_T, WALL_H),
                       material=mat_wall, show_direction=False)

        The position/size tuples reference local variables (SCENE_W,
        WALL_T, ...) defined inside ``run_layout_engine``. We walk every
        ``Assign`` in the AST, eagerly evaluate the RHS in a shared
        namespace (math available), then evaluate every wall call's
        position + size tuple. Failures are silent — a wall we can't
        resolve is simply dropped (still better than no wall info at all).
        """
        self.walls = []
        if not self.base_code:
            return
        src = self._sanitize_basecode(self.base_code)
        try:
            tree = ast.parse(src)
        except SyntaxError as exc:
            self._log(
                f"walls: base_code parse failed ({exc.msg} at line "
                f"{exc.lineno}); falling back to regex.",
                "warning",
            )
            self._parse_walls_via_regex(src)
            return

        namespace: Dict[str, Any] = {"math": math}
        # Whitelist a few common helpers Stage 3 might use in wall exprs.
        for name in ("pi", "sin", "cos", "tan", "sqrt", "radians", "degrees"):
            namespace[name] = getattr(math, name)

        def _try_eval(node: ast.AST) -> Tuple[bool, Any]:
            try:
                value = eval(  # noqa: S307 - sandboxed namespace
                    compile(ast.Expression(body=node), "<basecode-expr>", "eval"),
                    namespace,
                )
                return True, value
            except Exception:  # noqa: BLE001
                return False, None

        for sub in ast.walk(tree):
            if not isinstance(sub, ast.Assign):
                continue
            ok, value = _try_eval(sub.value)
            if not ok:
                continue
            for tgt in sub.targets:
                if isinstance(tgt, ast.Name):
                    namespace[tgt.id] = value
                elif isinstance(tgt, ast.Tuple) and isinstance(value, (tuple, list)):
                    if len(tgt.elts) == len(value):
                        for elt, v in zip(tgt.elts, value):
                            if isinstance(elt, ast.Name):
                                namespace[elt.id] = v

        seen_names: set = set()
        for sub in ast.walk(tree):
            if not isinstance(sub, ast.Call):
                continue
            if not (isinstance(sub.func, ast.Name) and sub.func.id == "create_box"):
                continue
            if len(sub.args) < 3:
                continue
            first = sub.args[0]
            if not isinstance(first, ast.Constant) or not isinstance(first.value, str):
                continue
            nm = first.value
            if not self._wall_name_looks_like_wall(nm):
                continue
            if nm in seen_names:
                continue
            ok_loc, loc = _try_eval(sub.args[1])
            ok_dim, dim = _try_eval(sub.args[2])
            if not (ok_loc and ok_dim):
                continue
            if not (isinstance(loc, (tuple, list)) and len(loc) == 3):
                continue
            if not (isinstance(dim, (tuple, list)) and len(dim) == 3):
                continue
            try:
                wc = [float(loc[0]), float(loc[1]), float(loc[2])]
                ws = [float(dim[0]), float(dim[1]), float(dim[2])]
            except (TypeError, ValueError):
                continue
            if not self._wall_dims_look_like_wall(ws):
                continue
            self.walls.append({
                "name": nm,
                "world_center": wc,
                "world_size": ws,
            })
            seen_names.add(nm)
        self._log(
            f"Walls parsed from base_code: {len(self.walls)}",
            "info",
        )

    # Name prefixes that LOOK like walls but aren't (wall-mounted furniture,
    # wall-mounted decor objects, etc.). These are filtered out so they
    # don't pollute the walls payload.
    _NON_WALL_NAME_PATTERNS = (
        "Wall_Mounted_",
        "Wall_Clock",
        "Wall_Sign",
        "Wall_Lamp",
        "Wall_Sconce",
        "Wall_Shelf",
        "Wall_TV",
        "Wall_Art",
        "Wall_Painting",
        "Wall_Mirror",
        "Wall_Pegboard",
        "Wall_Decor",
    )
    # Minimum thresholds for an AABB to count as a real architectural wall.
    # Walls are big (long) and tall; wall-mounted cabinets/decor are short.
    _WALL_MIN_LONG_EDGE_M = 1.0   # at least one of width/length >= 1.0 m
    _WALL_MIN_HEIGHT_M = 1.4      # height >= 1.4 m (windows can be lower
                                  # than full walls but still much taller
                                  # than a wall-mounted cabinet ~ 0.6 m)

    @classmethod
    def _wall_name_looks_like_wall(cls, name: str) -> bool:
        if not (name.startswith("Wall_") or name.startswith("Window_")):
            return False
        for pat in cls._NON_WALL_NAME_PATTERNS:
            if name.startswith(pat):
                return False
        return True

    @classmethod
    def _wall_dims_look_like_wall(cls, size: List[float]) -> bool:
        try:
            dx, dy, dz = float(size[0]), float(size[1]), float(size[2])
        except (TypeError, ValueError, IndexError):
            return False
        if dz < cls._WALL_MIN_HEIGHT_M:
            return False
        if max(dx, dy) < cls._WALL_MIN_LONG_EDGE_M:
            return False
        return True

    @staticmethod
    def _sanitize_basecode(src: str) -> str:
        """Patch known Stage 3/4 emit bugs that break ast.parse.

        Only fixes well-defined patterns; unknown SyntaxErrors are left
        alone so the regex fallback can take over.

        Currently handled:
          * Stage 4 sometimes strips the body out of the
            ``if show_direction and ...:`` block in create_box but forgets
            to insert ``pass``, leaving an empty if body followed by
            ``return obj``. Insert a ``pass`` so AST parsing succeeds.
        """
        return re.sub(
            r'(\n[ \t]+if\s+show_direction[^\n]+:\s*\n)(?:[ \t]*\n)+([ \t]+return\s+obj)',
            r'\1        pass\n\2',
            src,
        )

    def _parse_walls_via_regex(self, src: str) -> None:
        """Fallback wall extraction when ``ast.parse`` fails.

        Scans the source for top-level constants (SCENE_W, SCENE_D,
        WALL_T, WALL_H, side_len, etc.), then matches each
        ``create_box("Wall_X" / "Window_X", (loc), (dim), ...)`` call
        with paren-balanced extraction of the location / dimension
        tuples and evaluates them in a sandboxed namespace.

        Less robust than the AST path (e.g. multi-line tuples will be
        skipped), but recovers most of the wall list from buggy emits.
        """
        namespace: Dict[str, Any] = {"math": math}
        for nm in ("pi", "sin", "cos", "tan", "sqrt", "radians", "degrees"):
            namespace[nm] = getattr(math, nm)
        # Single-name simple assignments first:  NAME = expr
        for m in re.finditer(
            r'^[ \t]*([A-Za-z_]\w*)\s*=\s*([^\n#]+?)\s*(?:#.*)?$',
            src, re.MULTILINE,
        ):
            name, expr = m.group(1), m.group(2).strip()
            try:
                namespace[name] = eval(expr, namespace)  # noqa: S307
            except Exception:  # noqa: BLE001
                continue
        # Tuple-unpacking assignments:  A, B = expr1, expr2
        for m in re.finditer(
            r'^[ \t]*([A-Za-z_]\w*(?:\s*,\s*[A-Za-z_]\w*)+)\s*=\s*([^\n#]+?)\s*(?:#.*)?$',
            src, re.MULTILINE,
        ):
            names = [n.strip() for n in m.group(1).split(",")]
            expr = m.group(2).strip()
            try:
                value = eval(expr, namespace)  # noqa: S307
            except Exception:  # noqa: BLE001
                continue
            if isinstance(value, (tuple, list)) and len(value) == len(names):
                for n, v in zip(names, value):
                    namespace[n] = v

        seen_names: set = set()
        for m in re.finditer(
            r'create_box\(\s*"((?:Wall|Window)_[A-Za-z0-9_]+)"',
            src,
        ):
            name = m.group(1)
            if not self._wall_name_looks_like_wall(name):
                continue
            if name in seen_names:
                continue
            # Walk forward from the opening "(" of the call to extract the
            # next two paren-balanced tuples (location, dimension). This
            # is robust to nested function calls (e.g. math.cos(...)) inside
            # the tuple expressions.
            i = m.end()
            # Skip over the closing quote + optional whitespace + ","
            while i < len(src) and src[i] != ",":
                i += 1
            i += 1  # past the comma after the name
            tuples: List[str] = []
            while len(tuples) < 2 and i < len(src):
                while i < len(src) and src[i] in " \t\n":
                    i += 1
                if i >= len(src) or src[i] != "(":
                    break
                start = i
                depth = 1
                i += 1
                while i < len(src) and depth > 0:
                    ch = src[i]
                    if ch == "(":
                        depth += 1
                    elif ch == ")":
                        depth -= 1
                    i += 1
                if depth == 0:
                    tuples.append(src[start:i])
                    while i < len(src) and src[i] in " \t":
                        i += 1
                    if i < len(src) and src[i] == ",":
                        i += 1
            if len(tuples) < 2:
                continue
            try:
                loc = eval(tuples[0], namespace)  # noqa: S307
                dim = eval(tuples[1], namespace)  # noqa: S307
            except Exception:  # noqa: BLE001
                continue
            if not (isinstance(loc, (tuple, list)) and len(loc) == 3):
                continue
            if not (isinstance(dim, (tuple, list)) and len(dim) == 3):
                continue
            try:
                wc = [float(loc[0]), float(loc[1]), float(loc[2])]
                ws = [float(dim[0]), float(dim[1]), float(dim[2])]
            except (TypeError, ValueError):
                continue
            if not self._wall_dims_look_like_wall(ws):
                continue
            self.walls.append({
                "name": name,
                "world_center": wc,
                "world_size": ws,
            })
            seen_names.add(name)
        self._log(
            f"Walls parsed via regex fallback: {len(self.walls)}",
            "info",
        )

    # ---------------- existing-object (obstacle) discovery ----------------

    # Vertical filter for "is this PART resting on this plane?". We work at
    # part granularity rather than object granularity because:
    #   * A Keyboard has a wire / wrist-rest part whose bbox bottom drops
    #     well below the desk top — using `min(parts).z` would falsely
    #     classify the keyboard as "below" the desk.
    #   * A FloorLamp has a tall pole whose bbox top rises above many seat
    #     planes — using `max(parts).z` alone would falsely classify the
    #     lamp as "on" the sofa seat.
    # A part is "on" the plane iff its bottom z is in [plane_top - BELOW,
    # plane_top + ABOVE], its footprint is non-trivial, AND the object's
    # plane-local XY footprint overlaps the plane by at least
    # _OCCUPANT_MIN_OVERLAP_RATIO of its OWN footprint area.
    #
    # Asymmetric Z range on purpose: Stage 3 occasionally floats Keyboards/
    # Mice/Monitors 20–30 cm above the desk top (they get re-anchored later),
    # and tall items (PC tower, small incubator, mini centrifuge) rise up to
    # ~120 cm above the surface. We must catch all of them so the planes
    # payload reports them as obstacles to the LLM and Stage 7 doesn't
    # duplicate them.
    _OCCUPANT_PART_Z_BELOW_M = 0.05  # part bottom max 5 cm below plane top
    _OCCUPANT_PART_Z_ABOVE_M = 1.20  # part bottom max 1.2 m above plane top
    _OCCUPANT_PART_MIN_FOOTPRINT_M = 0.02  # ignore needle-thin parts (wires)
    # An object is only counted as an occupant if its plane-local XY
    # footprint (union over qualifying parts) overlaps the plane by at
    # least this fraction of its OWN footprint. Catches "tall furniture
    # standing next to a desk" (overlap ratio ~ 0%) vs "monitor genuinely
    # on the desk" (overlap ratio ~ 100%).
    _OCCUPANT_MIN_OVERLAP_RATIO = 0.5
    # Minimum height an occupant must rise ABOVE the plane top to count as a
    # physical obstacle. Anything flatter than this (placemats, doormats,
    # tablecloths, very thin trays) does NOT block placement — small
    # objects can sit on top of it with only a negligible z offset. Without
    # this filter, a Table_Runner laid on a Dining_Table would mistakenly
    # be reported as occupying the runner's OWN top plane (because the
    # dining table below the runner qualifies for the plane-proximity test
    # even though it's physically underneath).
    #
    # Tuned to 1.2 cm: a real keyboard on a desk has stickup ~ 2 cm and
    # MUST count as an obstacle. A standard table runner (~ 2.5 cm) will
    # also pass, which is fine — you don't pile decorations on a runner
    # in practice anyway.
    _OCCUPANT_MIN_STICKUP_M = 0.012  # 1.2 cm

    @staticmethod
    def _rotate_xy(x: float, y: float, yaw: float) -> Tuple[float, float]:
        c, s = math.cos(yaw), math.sin(yaw)
        return x * c - y * s, x * s + y * c

    def _find_objects_on_plane(self, plane: Plane) -> List[Dict[str, Any]]:
        """Find DETAILED_GEOMETRY objects currently resting on `plane`.

        Used so the LLM (and the resolver) avoid placing new small objects
        where Stage 3 already put a Monitor / Keyboard / Lamp / Printer / etc.

        We iterate parts (not whole objects) because real-world geometry is
        full of "honorary" parts — wires, brackets, base plates — that drag
        an object's bbox far above or below where the user perceives it.
        Per-part filtering is robust to those outliers.

        An object qualifies as an occupant iff at least one of its parts:
          * has its world bottom z within ``_OCCUPANT_PART_Z_BOT_M`` of
            the plane's top z (i.e. the part actually sits on the plane);
          * is at least ``_OCCUPANT_PART_MIN_FOOTPRINT_M`` square (wires
            and trim are filtered out);
          * has an XY footprint that overlaps the plane (in plane-local
            coordinates).

        The returned dicts use plane-local UV coordinates (u, v in [0, 1])
        and uv half-extents so both the prompt and the resolver can do
        purely UV-space overlap math without re-deriving frames.
        """
        pcx, pcy, pcz = plane.world_center
        pw, pd = plane.size_wd
        if pw <= 0 or pd <= 0:
            return []
        plane_yaw = plane.orientation_rad
        # Rotation that maps world -> plane-local is rot(-plane_yaw).
        cos_inv = math.cos(-plane_yaw)
        sin_inv = math.sin(-plane_yaw)
        half_w = pw / 2.0
        half_d = pd / 2.0

        results: List[Dict[str, Any]] = []
        for other_name, other_obj in self.detailed_geometry.items():
            if other_name == plane.parent_name:
                continue
            parts = other_obj.get("parts", []) or []
            if not parts:
                continue
            try:
                ocx = float(other_obj.get("center", [0, 0, 0])[0])
                ocy = float(other_obj.get("center", [0, 0, 0])[1])
                ocz = float(other_obj.get("center", [0, 0, 0])[2])
            except (TypeError, ValueError, IndexError):
                continue
            rot = other_obj.get("rotation", other_obj.get("base_rotation",
                                                          [0, 0, 0]))
            try:
                obj_yaw = float(rot[2]) if len(rot) >= 3 else 0.0
            except (TypeError, ValueError):
                obj_yaw = 0.0

            # Collect plane-local XY corners ONLY from qualifying parts.
            local_xs: List[float] = []
            local_ys: List[float] = []
            # Track the lowest qualifying part bottom z for debug output
            # AND the highest qualifying part top z so we can decide whether
            # this object actually sticks up above the plane (= real
            # obstacle) or merely rests flush / below (= not an obstacle).
            min_qual_bot_z = float("inf")
            max_qual_top_z = float("-inf")

            for p in parts:
                loc = p.get("loc", p.get("relative_location", [0, 0, 0]))
                dim = p.get("dim", p.get("dimensions", [0, 0, 0]))
                try:
                    rx, ry, rz = float(loc[0]), float(loc[1]), float(loc[2])
                    dx, dy, dz = float(dim[0]), float(dim[1]), float(dim[2])
                except (TypeError, ValueError, IndexError):
                    continue
                if dx <= 0 or dy <= 0 or dz <= 0:
                    continue
                # Skip wires / trim with negligible footprint.
                if (dx < self._OCCUPANT_PART_MIN_FOOTPRINT_M
                        or dy < self._OCCUPANT_PART_MIN_FOOTPRINT_M):
                    continue

                # World-space part center + bottom z.
                wx_off, wy_off = self._rotate_xy(rx, ry, obj_yaw)
                wcx = ocx + wx_off
                wcy = ocy + wy_off
                wcz = ocz + rz
                part_bot_z = wcz - dz / 2.0
                part_top_z = wcz + dz / 2.0

                # Vertical filter: part bottom must lie inside the
                # asymmetric [plane_top - BELOW, plane_top + ABOVE] band.
                if part_bot_z < pcz - self._OCCUPANT_PART_Z_BELOW_M:
                    continue
                if part_bot_z > pcz + self._OCCUPANT_PART_Z_ABOVE_M:
                    continue

                # XY corners (apply object yaw), projected to plane-local.
                part_local_xs: List[float] = []
                part_local_ys: List[float] = []
                for sx in (-dx / 2.0, dx / 2.0):
                    for sy in (-dy / 2.0, dy / 2.0):
                        rrx, rry = self._rotate_xy(sx, sy, obj_yaw)
                        wx = wcx + rrx
                        wy = wcy + rry
                        rel_x = wx - pcx
                        rel_y = wy - pcy
                        lx = rel_x * cos_inv - rel_y * sin_inv
                        ly = rel_x * sin_inv + rel_y * cos_inv
                        part_local_xs.append(lx)
                        part_local_ys.append(ly)

                # Per-part XY bbox, then check overlap with the plane.
                pxmin, pxmax = min(part_local_xs), max(part_local_xs)
                pymin, pymax = min(part_local_ys), max(part_local_ys)
                if pxmin > half_w or pxmax < -half_w:
                    continue
                if pymin > half_d or pymax < -half_d:
                    continue
                local_xs.extend(part_local_xs)
                local_ys.extend(part_local_ys)
                if part_bot_z < min_qual_bot_z:
                    min_qual_bot_z = part_bot_z
                if part_top_z > max_qual_top_z:
                    max_qual_top_z = part_top_z

            if not local_xs:
                continue

            # Hard filter: an occupant only blocks new items if it physically
            # protrudes above the plane by at least a small threshold. A
            # table runner (2 cm thick, top ≈ plane_top + 2 cm) is NOT an
            # obstacle for items placed on top of it — they'll sit slightly
            # higher but won't clash. This also drops the symmetric false
            # positive where querying Table_Runner__top detects the
            # Dining_Table underneath as an "occupant", even though the
            # table is entirely below the runner's surface.
            stickup = max_qual_top_z - pcz
            if stickup <= self._OCCUPANT_MIN_STICKUP_M:
                continue

            lx_min, lx_max = min(local_xs), max(local_xs)
            ly_min, ly_max = min(local_ys), max(local_ys)
            # Clip union of qualifying parts to the plane footprint.
            cx_local = (max(lx_min, -half_w) + min(lx_max, half_w)) / 2.0
            cy_local = (max(ly_min, -half_d) + min(ly_max, half_d)) / 2.0
            obj_w = min(lx_max, half_w) - max(lx_min, -half_w)
            obj_d = min(ly_max, half_d) - max(ly_min, -half_d)
            if obj_w <= 0 or obj_d <= 0:
                continue

            # Reject objects that only graze the plane in XY: a tall
            # cabinet pulled up next to a desk passes the Z filter at
            # multiple part levels, but its footprint barely touches the
            # desk top. Real occupants overlap the plane by most of their
            # own footprint area.
            obj_full_w = lx_max - lx_min
            obj_full_d = ly_max - ly_min
            if obj_full_w > 0 and obj_full_d > 0:
                overlap_ratio = (obj_w * obj_d) / (obj_full_w * obj_full_d)
                if overlap_ratio < self._OCCUPANT_MIN_OVERLAP_RATIO:
                    continue

            u = cx_local / pw + 0.5
            v = cy_local / pd + 0.5
            u_half = (obj_w / 2.0) / pw
            v_half = (obj_d / 2.0) / pd

            obj_type = ""
            ot_raw = other_obj.get("object_type", "")
            if isinstance(ot_raw, str):
                obj_type = ot_raw

            results.append({
                "name": other_name,
                "object_type": obj_type,
                "local_uv": [round(u, 3), round(v, 3)],
                "size_wd": [round(obj_w, 3), round(obj_d, 3)],
                "uv_half_extents": [round(u_half, 3), round(v_half, 3)],
                "bottom_z": round(min_qual_bot_z, 3),
            })
        return results

    def _compute_plane_occupants(self) -> None:
        """Populate ``self.plane_occupants`` for every discovered plane."""
        self.plane_occupants = {}
        for plane in self.planes:
            try:
                cross_obj = self._find_objects_on_plane(plane)
            except Exception as exc:  # noqa: BLE001
                self._log(
                    f"   ! occupant detection failed for {plane.plane_id}: {exc}",
                    "warning",
                )
                cross_obj = []
            try:
                siblings = self._find_sibling_part_occupants(plane)
            except Exception as exc:  # noqa: BLE001
                self._log(
                    f"   ! sibling occupant detection failed for "
                    f"{plane.plane_id}: {exc}",
                    "warning",
                )
                siblings = []
            occupants = cross_obj + siblings
            self.plane_occupants[plane.plane_id] = occupants
            if occupants and self.verbose:
                names = ", ".join(o["name"] for o in occupants)
                self._log(
                    f"  {plane.plane_id} already occupied by: {names}",
                    "warning",
                )

    def _find_sibling_part_occupants(self, plane: Plane) -> List[Dict[str, Any]]:
        """Find sibling parts inside ``plane.parent_name`` that obstruct
        ``plane``.

        The cross-object path (``_find_objects_on_plane``) skips the
        parent itself, so features that are built into the SAME
        furniture piece as the worktop — sink rims and basins, faucets,
        backsplashes, divider panels, grilles, burner housings — are
        never reported as obstacles. That is fine for placement-wise
        irrelevant siblings (the worktop slab itself) but it lets the
        LLM stack a flask right on top of the sink cutout, which looks
        broken in the render.

        This helper walks the parent's own parts, skips the part that
        produced the plane (the worktop is not its own obstacle), and
        reports any remaining box/cylinder part whose top sticks up at
        least ``_OCCUPANT_MIN_STICKUP_M`` above the plane top and whose
        XY footprint overlaps the plane. Each qualifying part becomes
        an individual occupant entry, so adjacent features (sink_rim,
        sink_basin, faucet_*) are listed separately and the LLM keeps
        a clear keep-out region.
        """
        parent_obj = self.detailed_geometry.get(plane.parent_name)
        if not parent_obj:
            return []
        parts = parent_obj.get("parts", []) or []
        if not parts:
            return []
        try:
            center = parent_obj.get("center", parent_obj.get("center_location",
                                                              [0.0, 0.0, 0.0]))
            ocx = float(center[0])
            ocy = float(center[1])
            ocz = float(center[2])
        except (TypeError, ValueError, IndexError):
            return []
        rot = parent_obj.get("rotation", parent_obj.get("base_rotation",
                                                         [0.0, 0.0, 0.0]))
        try:
            obj_yaw = float(rot[2]) if len(rot) >= 3 else 0.0
        except (TypeError, ValueError):
            obj_yaw = 0.0

        pcx, pcy, pcz = plane.world_center
        pw, pd = plane.size_wd
        if pw <= 0 or pd <= 0:
            return []
        plane_yaw = plane.orientation_rad
        cos_inv = math.cos(-plane_yaw)
        sin_inv = math.sin(-plane_yaw)
        half_w = pw / 2.0
        half_d = pd / 2.0

        occupants: List[Dict[str, Any]] = []
        for p in parts:
            pname = str(p.get("name", ""))
            # The worktop itself never counts as its own obstacle.
            if pname and pname == plane.source_part_name:
                continue
            ptype = str(p.get("type", p.get("shape_type", "box"))).lower()
            if ptype not in ("box", "cylinder"):
                continue
            loc = p.get("loc", p.get("relative_location", [0, 0, 0]))
            dim = p.get("dim", p.get("dimensions", [0, 0, 0]))
            try:
                rx, ry, rz = float(loc[0]), float(loc[1]), float(loc[2])
                dx, dy, dz = float(dim[0]), float(dim[1]), float(dim[2])
            except (TypeError, ValueError, IndexError):
                continue
            if dx <= 0 or dy <= 0 or dz <= 0:
                continue
            # Wires / trim are filtered out — same threshold as the
            # cross-object path so behaviour stays consistent.
            if (dx < self._OCCUPANT_PART_MIN_FOOTPRINT_M
                    or dy < self._OCCUPANT_PART_MIN_FOOTPRINT_M):
                continue

            wx_off, wy_off = self._rotate_xy(rx, ry, obj_yaw)
            wcx = ocx + wx_off
            wcy = ocy + wy_off
            wcz = ocz + rz
            part_bot_z = wcz - dz / 2.0
            part_top_z = wcz + dz / 2.0

            # Must rise above the plane top. Sibling parts that sit
            # entirely below the plane (e.g. the base cabinet body
            # under a worktop) are NOT obstacles.
            if part_top_z - pcz <= self._OCCUPANT_MIN_STICKUP_M:
                continue
            # Vertical band: anything more than 1.2 m above the surface
            # is a hood/upper-shelf and is irrelevant to placement.
            if part_bot_z > pcz + self._OCCUPANT_PART_Z_ABOVE_M:
                continue

            local_xs: List[float] = []
            local_ys: List[float] = []
            for sx in (-dx / 2.0, dx / 2.0):
                for sy in (-dy / 2.0, dy / 2.0):
                    rrx, rry = self._rotate_xy(sx, sy, obj_yaw)
                    wx = wcx + rrx
                    wy = wcy + rry
                    rel_x = wx - pcx
                    rel_y = wy - pcy
                    lx = rel_x * cos_inv - rel_y * sin_inv
                    ly = rel_x * sin_inv + rel_y * cos_inv
                    local_xs.append(lx)
                    local_ys.append(ly)
            lx_min, lx_max = min(local_xs), max(local_xs)
            ly_min, ly_max = min(local_ys), max(local_ys)
            if lx_min > half_w or lx_max < -half_w:
                continue
            if ly_min > half_d or ly_max < -half_d:
                continue
            cx_local = (max(lx_min, -half_w) + min(lx_max, half_w)) / 2.0
            cy_local = (max(ly_min, -half_d) + min(ly_max, half_d)) / 2.0
            obj_w = min(lx_max, half_w) - max(lx_min, -half_w)
            obj_d = min(ly_max, half_d) - max(ly_min, -half_d)
            if obj_w <= 0 or obj_d <= 0:
                continue

            u = cx_local / pw + 0.5
            v = cy_local / pd + 0.5
            u_half = (obj_w / 2.0) / pw
            v_half = (obj_d / 2.0) / pd

            occ_name = (
                f"{plane.parent_name}.{pname}" if pname else plane.parent_name
            )
            occupants.append({
                "name": occ_name,
                "object_type": pname or "sibling_part",
                "local_uv": [round(u, 3), round(v, 3)],
                "size_wd": [round(obj_w, 3), round(obj_d, 3)],
                "uv_half_extents": [round(u_half, 3), round(v_half, 3)],
                "bottom_z": round(part_bot_z, 3),
            })
        return occupants

    # ---------------- orientation hints (walls + nearby bboxes) -----------

    # Distance threshold (plane-local meters) for "nearby" object detection
    # in the nearby_bboxes payload. 1.5x the largest plane edge captures
    # stools pulled out from a desk, walls flush against the plane, and
    # adjacent wall-art mounted slightly above the surface.
    _NEARBY_BBOX_DIST_FACTOR = 1.5
    # A wall side is reported only if its plane-local distance to the plane
    # edge is at most this many meters. Beyond this we don't care.
    _WALL_PROXIMITY_M = 1.0
    # Cap on nearby bboxes per plane to keep the LLM prompt small.
    _NEARBY_BBOX_MAX = 12

    def _walls_for_plane(self, plane: Plane) -> List[Dict[str, Any]]:
        """Walls within _WALL_PROXIMITY_M of `plane`, in plane-local frame.

        Each entry reports which plane-local side ("+u" / "-u" / "+v" / "-v")
        the wall sits past, plus the gap distance in meters from the plane
        edge to the nearest face of the wall AABB. The LLM uses this to
        decide which side of the surface is "back-against-wall" (no user
        sits there) vs free.
        """
        if not self.walls:
            return []
        pcx, pcy, _pcz = plane.world_center
        pw, pd = plane.size_wd
        yaw = plane.orientation_rad
        cos_inv = math.cos(-yaw)
        sin_inv = math.sin(-yaw)
        half_w = pw / 2.0
        half_d = pd / 2.0

        per_side: Dict[str, Dict[str, Any]] = {}
        for w in self.walls:
            try:
                wcx, wcy, _ = w["world_center"]
                wdx, wdy, _ = w["world_size"]
            except (KeyError, TypeError, ValueError):
                continue
            local_xs: List[float] = []
            local_ys: List[float] = []
            for sx in (-wdx / 2.0, wdx / 2.0):
                for sy in (-wdy / 2.0, wdy / 2.0):
                    rel_x = (wcx + sx) - pcx
                    rel_y = (wcy + sy) - pcy
                    lx = rel_x * cos_inv - rel_y * sin_inv
                    ly = rel_x * sin_inv + rel_y * cos_inv
                    local_xs.append(lx)
                    local_ys.append(ly)
            lx_min, lx_max = min(local_xs), max(local_xs)
            ly_min, ly_max = min(local_ys), max(local_ys)
            # Side gap: positive iff the wall lies entirely past that plane
            # edge. A wall that overlaps the plane in an axis is skipped
            # for that axis (it doesn't define a clear "back" side).
            sides = (
                ("+u", lx_min - half_w),
                ("-u", -half_w - lx_max),
                ("+v", ly_min - half_d),
                ("-v", -half_d - ly_max),
            )
            best_side, best_dist = None, float("inf")
            for side, dist in sides:
                if dist < 0:
                    continue
                if dist < best_dist:
                    best_side, best_dist = side, dist
            if best_side is None:
                continue
            if best_dist > self._WALL_PROXIMITY_M:
                continue
            existing = per_side.get(best_side)
            if existing is None or best_dist < existing["distance_m"]:
                per_side[best_side] = {
                    "name": w["name"],
                    "side": best_side,
                    "distance_m": round(best_dist, 3),
                }
        return list(per_side.values())

    @staticmethod
    def _classify_nearby_object(obj_type: str) -> str:
        t = (obj_type or "").lower()
        if any(k in t for k in ("stool", "chair", "sofa", "armchair",
                                "ottoman", "bench seat", "bench_seat")):
            return "seat"
        if any(k in t for k in ("clock", "painting", "picture", "frame",
                                "mirror", "sign", "sconce", "wall_shelf",
                                "tv_mounted", "wall_tv", "art", "pegboard",
                                "curtain")):
            return "wall_decor"
        return "furniture"

    def _object_world_aabb_xy(
        self, obj: Dict[str, Any]
    ) -> Optional[Tuple[float, float, float, float]]:
        """Return (cx, cy, full_w, full_d) world AABB for an object.

        Prefer the explicit ``center_location`` + ``bounding_dimensions``
        fields from Stage 7 / Stage 6 metadata; if absent, fall back to
        the union of part bboxes (less accurate when the object has
        rotation, but this is only used for "nearby" hints anyway).
        """
        center = obj.get("center_location") or obj.get("center")
        dims = obj.get("bounding_dimensions") or obj.get("dimensions")
        try:
            if center is not None and dims is not None:
                return (
                    float(center[0]), float(center[1]),
                    float(dims[0]), float(dims[1]),
                )
        except (TypeError, ValueError, IndexError):
            pass
        parts = obj.get("parts") or []
        if not parts:
            return None
        try:
            ocx = float((center or [0, 0, 0])[0])
            ocy = float((center or [0, 0, 0])[1])
        except (TypeError, ValueError, IndexError):
            ocx, ocy = 0.0, 0.0
        xs: List[float] = []
        ys: List[float] = []
        for p in parts:
            loc = p.get("loc", p.get("relative_location", [0, 0, 0]))
            dim = p.get("dim", p.get("dimensions", [0, 0, 0]))
            try:
                rx, ry = float(loc[0]), float(loc[1])
                dx, dy = float(dim[0]), float(dim[1])
            except (TypeError, ValueError, IndexError):
                continue
            xs.extend([ocx + rx - dx / 2.0, ocx + rx + dx / 2.0])
            ys.extend([ocy + ry - dy / 2.0, ocy + ry + dy / 2.0])
        if not xs:
            return None
        cx_full = (min(xs) + max(xs)) / 2.0
        cy_full = (min(ys) + max(ys)) / 2.0
        return cx_full, cy_full, max(xs) - min(xs), max(ys) - min(ys)

    def _nearby_bboxes_for_plane(self, plane: Plane) -> List[Dict[str, Any]]:
        """DETAILED_GEOMETRY objects near `plane` in plane-local frame.

        Capped at ``_NEARBY_BBOX_MAX`` entries, sorted by distance to the
        plane center (closest first). Includes the parent's siblings —
        e.g. for a Desk's top plane, the Office_Chair is reported as
        ``category == "seat"`` so the LLM can infer the user-facing side.
        """
        pcx, pcy, _pcz = plane.world_center
        pw, pd = plane.size_wd
        yaw = plane.orientation_rad
        cos_inv = math.cos(-yaw)
        sin_inv = math.sin(-yaw)
        threshold = max(pw, pd) * self._NEARBY_BBOX_DIST_FACTOR

        candidates: List[Tuple[float, Dict[str, Any]]] = []
        for other_name, other_obj in self.detailed_geometry.items():
            if other_name == plane.parent_name:
                continue
            aabb = self._object_world_aabb_xy(other_obj)
            if aabb is None:
                continue
            wcx, wcy, dx_full, dy_full = aabb
            rel_x = wcx - pcx
            rel_y = wcy - pcy
            lcx = rel_x * cos_inv - rel_y * sin_inv
            lcy = rel_x * sin_inv + rel_y * cos_inv
            # Clip to a reasonable neighbourhood: drop anything well past
            # the plane on BOTH axes simultaneously (corner-of-room items).
            if abs(lcx) > threshold and abs(lcy) > threshold:
                continue
            ot_raw = other_obj.get("object_type", "")
            ot_str = ot_raw if isinstance(ot_raw, str) else ""
            entry = {
                "name": other_name,
                "object_type": ot_str,
                "category": self._classify_nearby_object(ot_str),
                "local_center": [round(lcx, 3), round(lcy, 3)],
                "local_size_wd": [round(dx_full, 3), round(dy_full, 3)],
            }
            dist = abs(lcx) + abs(lcy)
            candidates.append((dist, entry))
        candidates.sort(key=lambda kv: kv[0])
        return [e for _, e in candidates[:self._NEARBY_BBOX_MAX]]

    # ---------------- LLM output -> world-space items ----------------

    @staticmethod
    def _clamp(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    # Names / item_types that the prompt forbids. Kept here as a safety net
    # in case the LLM still emits a "pack" despite the prompt. We downgrade
    # the offender to a warning so the user can see it in the console,
    # rather than silently dropping the item (which would leave a hole in
    # the scene).
    _PACK_KEYWORDS = (
        "book_stack", "book_set", "bookstack", "books_", "books_set",
        "plate_stack", "plates_", "plate_set",
        "bowl_stack", "bowls_",
        "glass_set", "glasses_",
        "candle_set", "candles_",
        "vase_set",
        "frame_set", "photo_set",
        "box_stack", "box_set",
        "set_of", "row_of", "stack_of", "collection_of", "group_of",
        "pair_of", "cluster_of", "bunch_of", "dense_row",
    )

    def _looks_like_pack(self, name: str, item_type: str) -> bool:
        blob = f"{name} {item_type}".lower().replace(" ", "_")
        return any(kw in blob for kw in self._PACK_KEYWORDS)

    def _resolve_items(self, parent_name: str, planes: List[Plane],
                       llm_result: Dict[str, Any]) -> List[SmallObjectItem]:
        """Convert LLM UV output to world-space `SmallObjectItem`s.

        Applies:
          * uv clamping to [0.05, 0.95]
          * bbox-inside-plane enforcement
          * obstacle rejection — items that overlap an `existing occupant`
            (e.g. a Monitor already on the desk top) are dropped, NOT
            shifted, because the LLM already had this info and refusing
            to comply means the request was poorly scoped anyway.
          * orientation rotation around the plane center
          * z = plane.world_z + size_h / 2 so the item rests on the surface
          * vertical stacking: items sharing (plane, local_uv, rounded) with
            increasing `stack_index` have their Z lifted by the cumulative
            height of the lower siblings, so an atomic stack of plates
            (6 individual plate items at the same uv) renders as a pile.
        """
        plane_by_id: Dict[str, Plane] = {pl.plane_id: pl for pl in planes}
        out: List[SmallObjectItem] = []

        for plane_entry in llm_result.get("planes", []) or []:
            pid = plane_entry.get("plane_id")
            plane = plane_by_id.get(pid)
            if plane is None:
                continue
            pw, pd = plane.size_wd
            cx, cy, cz = plane.world_center
            yaw = plane.orientation_rad
            occupants = self.plane_occupants.get(pid, [])

            used_names: set = set()
            plane_items: List[SmallObjectItem] = []
            for raw_item in plane_entry.get("items", []) or []:
                # Safety net against prompt non-compliance: log but keep.
                raw_name = str(raw_item.get("name", ""))
                raw_type = str(raw_item.get("item_type", ""))
                if self._looks_like_pack(raw_name, raw_type):
                    self._log(
                        f"   pack-like item on {pid}: '{raw_name}' "
                        f"(type='{raw_type}') — prompt violation; keeping it "
                        f"but the unit should have been decomposed.",
                        "warning",
                    )
                try:
                    item = self._resolve_single_item(
                        plane, raw_item, parent_name, cx, cy, cz, pw, pd, yaw,
                        used_names, occupants,
                    )
                except Exception as exc:  # noqa: BLE001
                    self._log(f"   skip bad item on {pid}: {exc}", "warning")
                    continue
                if item is not None:
                    plane_items.append(item)
                    used_names.add(item.name)

            # Apply vertical stacking within this plane.
            self._apply_stack_offsets(plane_items, cz)
            out.extend(plane_items)
        return out

    @staticmethod
    def _apply_stack_offsets(items: List[SmallObjectItem], plane_z: float) -> None:
        """Lift stacked items so each one sits on top of its lower siblings.

        Groups by (rounded XY world position) — items the LLM placed at the
        same `local_uv` will land at the same world XY, so rounding to ~2cm
        is a reliable clustering key without needing to re-plumb uv through
        the resolver.
        """
        if not items:
            return
        groups: Dict[Tuple[int, int], List[SmallObjectItem]] = {}
        for it in items:
            key = (round(it.world_location[0] * 50), round(it.world_location[1] * 50))
            groups.setdefault(key, []).append(it)

        for group in groups.values():
            if len(group) <= 1:
                continue
            # Stable sort: explicit stack_index first, then insertion order.
            group.sort(key=lambda it: it.stack_index)
            running = 0.0  # total height of lower siblings
            for idx, it in enumerate(group):
                new_z = plane_z + running + it.size[2] / 2.0
                it.world_location = (it.world_location[0], it.world_location[1], new_z)
                running += it.size[2]

    def _resolve_single_item(
        self,
        plane: Plane,
        raw_item: Dict[str, Any],
        parent_name: str,
        cx: float, cy: float, cz: float,
        pw: float, pd: float,
        yaw: float,
        used_names: set,
        occupants: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[SmallObjectItem]:
        # ---- size ----
        size = raw_item.get("size")
        if not (isinstance(size, (list, tuple)) and len(size) >= 3):
            return None
        try:
            sw, sd, sh = float(size[0]), float(size[1]), float(size[2])
        except (TypeError, ValueError):
            return None
        if sw <= 0 or sd <= 0 or sh <= 0:
            return None

        # Reject items that cannot physically fit on the plane (with a 2 cm
        # margin on each side). We could shrink instead, but that often
        # produces grotesquely thin objects, so we prefer to drop them.
        margin = 0.02
        if sw > max(0.0, pw - 2 * margin) or sd > max(0.0, pd - 2 * margin):
            return None

        # ---- uv ----
        uv = raw_item.get("local_uv") or raw_item.get("uv")
        if not (isinstance(uv, (list, tuple)) and len(uv) >= 2):
            return None
        try:
            u = self._clamp(float(uv[0]), 0.05, 0.95)
            v = self._clamp(float(uv[1]), 0.05, 0.95)
        except (TypeError, ValueError):
            return None

        # Convert uv (plane-local, 0..1) to plane-local offset in meters, with
        # (0.5, 0.5) mapping to the plane center.
        u_m = (u - 0.5) * pw
        v_m = (v - 0.5) * pd

        # Enforce bbox-inside-plane after clamping: shift uv back inward if
        # necessary so the item's own footprint stays inside the plane.
        max_u = (pw / 2.0) - (sw / 2.0) - margin
        max_v = (pd / 2.0) - (sd / 2.0) - margin
        if max_u < 0 or max_v < 0:
            return None
        u_m = self._clamp(u_m, -max_u, max_u)
        v_m = self._clamp(v_m, -max_v, max_v)

        # ---- obstacle (existing-occupant) overlap check ----
        # Done AFTER uv clamping, BEFORE world-space transform, because all
        # occupants are stored in plane-local meters. Reject the item if its
        # footprint overlaps any existing occupant with less than the
        # required clearance — otherwise the new small object would visibly
        # poke through an existing Monitor/Keyboard/Lamp/etc.
        if occupants:
            CLEARANCE_M = 0.02  # 2 cm hard clearance; matches plane edge margin
            item_half_w = sw / 2.0
            item_half_d = sd / 2.0
            for occ in occupants:
                try:
                    occ_u, occ_v = occ["local_uv"]
                    occ_uh, occ_vh = occ["uv_half_extents"]
                except (KeyError, ValueError, TypeError):
                    continue
                occ_u_m = (float(occ_u) - 0.5) * pw
                occ_v_m = (float(occ_v) - 0.5) * pd
                occ_half_w = float(occ_uh) * pw
                occ_half_d = float(occ_vh) * pd
                # AABB overlap with clearance: separation along EITHER axis
                # by at least (item_half + occ_half + clearance) is OK.
                gap_u = abs(u_m - occ_u_m) - (item_half_w + occ_half_w)
                gap_v = abs(v_m - occ_v_m) - (item_half_d + occ_half_d)
                if gap_u < CLEARANCE_M and gap_v < CLEARANCE_M:
                    self._log(
                        f"   skip '{raw_item.get('name', '?')}' on "
                        f"{plane.plane_id}: overlaps existing "
                        f"'{occ.get('name', '?')}' "
                        f"({occ.get('object_type', '?')})",
                        "warning",
                    )
                    return None

        # Apply plane orientation (yaw) to convert plane-local offset to world.
        c, s = math.cos(yaw), math.sin(yaw)
        dx = u_m * c - v_m * s
        dy = u_m * s + v_m * c
        world_x = cx + dx
        world_y = cy + dy
        world_z = cz + sh / 2.0

        # ---- rotation ----
        try:
            rot_local = float(raw_item.get("rotation_z", 0.0))
        except (TypeError, ValueError):
            rot_local = 0.0
        rot_world = yaw + rot_local

        # ---- name ----
        base_name = str(raw_item.get("name", "")).strip() or raw_item.get(
            "item_type", "item"
        )
        base_name = re.sub(r"[^A-Za-z0-9_]+", "_", str(base_name)).strip("_") or "Item"
        final_name = f"{parent_name}__{plane.plane_type}__{base_name}"
        # Ensure uniqueness within this plane.
        if final_name in used_names:
            suffix = 2
            while f"{final_name}_{suffix}" in used_names:
                suffix += 1
            final_name = f"{final_name}_{suffix}"

        shape = str(raw_item.get("shape", "box")).lower()
        if shape not in ("box", "cylinder"):
            shape = "box"

        try:
            stack_index = int(raw_item.get("stack_index", 0) or 0)
        except (TypeError, ValueError):
            stack_index = 0
        if stack_index < 0:
            stack_index = 0

        return SmallObjectItem(
            name=final_name,
            item_type=str(raw_item.get("item_type", "")),
            shape=shape,
            parent_name=parent_name,
            plane_id=plane.plane_id,
            plane_type=plane.plane_type,
            world_location=(world_x, world_y, world_z),
            size=(sw, sd, sh),
            rotation_z=rot_world,
            color_hint=str(raw_item.get("color_hint", "")),
            description=str(raw_item.get("description", "")),
            stack_index=stack_index,
        )

    # ---------------- code generation ----------------

    def _generate_code(self) -> str:
        """Append small-object create_* calls to the Stage 6 base code."""
        if not self.base_code:
            return ""

        compat_lines: List[str] = []
        if "def create_collection" not in self.base_code:
            compat_lines.extend([
                "",
                "def create_collection(name):",
                "    \"\"\"Create or fetch a Blender collection.\"\"\"",
                "    existing = bpy.data.collections.get(name)",
                "    if existing:",
                "        return existing",
                "    coll = bpy.data.collections.new(name)",
                "    bpy.context.scene.collection.children.link(coll)",
                "    return coll",
            ])

        needs_cylinder = any(it.shape == "cylinder" for it in self.items)
        if needs_cylinder and "def create_cylinder" not in self.base_code:
            compat_lines.extend([
                "",
                "def create_cylinder(name, location, dimensions, rotation=(0,0,0), material=None, collection=None):",
                "    \"\"\"Create a cylinder primitive compatible with Stage 6 output.\"\"\"",
                "    bpy.ops.mesh.primitive_cylinder_add(radius=1, depth=1, location=location, rotation=rotation)",
                "    obj = bpy.context.active_object",
                "    obj.name = name",
                "    obj.dimensions = dimensions",
                "    if material:",
                "        obj.data.materials.append(material)",
                "    if collection:",
                "        old_colls = list(obj.users_collection)",
                "        collection.objects.link(obj)",
                "        for c in old_colls:",
                "            c.objects.unlink(obj)",
                "    return obj",
            ])

        lines: List[str] = [
            "",
            "# ==============================================================================",
            "# Small objects appended by stage7_small_objects.py",
            "# Generated: " + datetime.now().isoformat(timespec="seconds"),
            f"# Total items: {len(self.items)}",
            "# ==============================================================================",
            "",
        ]
        lines.extend(compat_lines)
        if compat_lines:
            lines.append("")
        lines.extend([
            "_small_objects_collection = create_collection(\"Small_Objects\")",
            "",
            "_small_object_materials = {}",
            "",
            "def _get_small_material(color_key, color_rgba):",
            "    \"\"\"Fetch-or-create a cached material for small objects.\"\"\"",
            "    if color_key in _small_object_materials:",
            "        return _small_object_materials[color_key]",
            "    mat = create_material(f\"SmallMat_{color_key}\", color_rgba)",
            "    _small_object_materials[color_key] = mat",
            "    return mat",
            "",
        ])

        for it in self.items:
            color_rgba = _color_for_hint(it.color_hint or it.item_type)
            color_key = re.sub(r"[^a-z0-9]+", "_",
                               (it.color_hint or it.item_type or "default").lower()
                               ).strip("_") or "default"
            loc = tuple(round(v, 4) for v in it.world_location)
            dim = tuple(round(v, 4) for v in it.size)
            rot = round(it.rotation_z, 4)
            creator = "create_box" if it.shape == "box" else "create_cylinder"
            lines.append(
                f"{creator}({it.name!r}, {loc}, {dim}, rotation=(0, 0, {rot}), "
                f"material=_get_small_material({color_key!r}, {color_rgba}), "
                f"collection=_small_objects_collection)"
            )

        addition = "\n".join(lines) + "\n"
        return self.base_code.rstrip() + "\n" + addition

    # ---------------- save ----------------

    def _save_results(self, generated_code: str) -> Dict[str, str]:
        os.makedirs(self.output_dir, exist_ok=True)

        # 1) planes.json — always written, useful for debugging.
        planes_path = os.path.join(self.output_dir, "planes.json")
        planes_payload = []
        for p in self.planes:
            entry = p.to_dict()
            entry["existing_occupants"] = self.plane_occupants.get(p.plane_id, [])
            entry["walls"] = self._walls_for_plane(p)
            entry["nearby_bboxes"] = self._nearby_bboxes_for_plane(p)
            planes_payload.append(entry)
        with open(planes_path, "w", encoding="utf-8") as f:
            json.dump({"planes": planes_payload},
                      f, ensure_ascii=False, indent=2)

        # 2) items.json — the structured output of the LLM step.
        items_path = os.path.join(self.output_dir, "small_objects.json")
        with open(items_path, "w", encoding="utf-8") as f:
            json.dump({
                "items": [it.to_dict() for it in self.items],
                "summary": {
                    "total_items": len(self.items),
                    "total_planes": len(self.planes),
                    "by_plane_type": self._plane_type_counts(),
                    "base_code_path": self.base_code_path,
                    "image_path": self.image_path,
                    "generated_at": datetime.now().isoformat(timespec="seconds"),
                },
            }, f, ensure_ascii=False, indent=2)

        code_path = ""
        if generated_code:
            code_path = os.path.join(self.output_dir, "small_objects_output.py")
            with open(code_path, "w", encoding="utf-8") as f:
                f.write(generated_code)
            self._log(f"Blender code: {code_path}", "save")

        # 3) Memory.
        if self.use_memory and generated_code:
            self.memory.add(
                stage="stage7_small_objects",
                type="result",
                content=generated_code,
                metadata={
                    "title": "Stage Small Objects - surface-driven decoration",
                    "summary": f"{len(self.items)} items across {len(self.planes)} planes",
                    "output_file": code_path,
                    "planes_file": planes_path,
                    "items_file": items_path,
                    "image_path": self.image_path,
                    "base_code_path": self.base_code_path,
                },
                tags=["stage7_small_objects", "small_objects", "surface_decor"],
            )

        self._log(f"Planes file: {planes_path}", "save")
        self._log(f"Items file: {items_path}", "save")
        return {
            "planes": planes_path,
            "items": items_path,
            "code": code_path,
        }

    def _plane_type_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for p in self.planes:
            counts[p.plane_type] = counts.get(p.plane_type, 0) + 1
        return counts

    # ---------------- main pipeline ----------------

    def run(self) -> Tuple[bool, Dict[str, Any]]:
        print("\n" + "=" * 60)
        print("🧸 Stage Small Objects - surface-driven decoration")
        print("=" * 60)

        if not self._load_geometry():
            return False, {}
        self._load_describe()
        self._load_stage1_hints()
        # v3: load Stage 4 Phase B placeholders so we can avoid duplicate
        # placements (e.g. don't put a second table-lamp on a parent that
        # already has a `MinorPlace_obj_011__table_lamp` from Stage 4).
        self._load_stage4_minor_placed()
        if not self._load_base_code():
            return False, {}
        has_image = self._load_image()
        # Parse walls (Wall_*/Window_*) out of the base code so the LLM
        # payload can expose nearby walls per plane in plane-local frame.
        # Cheap (AST + eval), so we always run it; failures are non-fatal.
        self._parse_walls_from_basecode()

        # Plane discovery.
        self._log("Discovering placement planes...", "step")
        finder = PlaneFinder(
            detailed_geometry=self.detailed_geometry,
            describe_objects=self.describe_objects,
            verbose=self.verbose,
        )
        self.planes = finder.find_planes()

        # v3 (2026-05-02): also synthesize top planes for Stage 4 wall items
        # (floating shelves, wall cabinets, spice racks, ...). Stage 6 keeps
        # those as simple bbox so PlaneFinder can't see them; we read Stage
        # 4's wall_objects.json + parse base_code to recover their geometry,
        # then emit Plane objects directly. Side-effect: also injects single-
        # part DETAILED_GEOMETRY entries for these items so occupant scan
        # and nearby-bbox payload still see a parent.
        wall_decor_planes = self._synthesize_wall_decor_planes()
        if wall_decor_planes:
            self.planes.extend(wall_decor_planes)
        self._log(
            f"Found {len(self.planes)} planes ({self._plane_type_counts()})",
            "plane",
        )
        for pl in self.planes:
            self._log(
                f"  {pl.plane_id}  type={pl.plane_type}  size={pl.size_wd}  "
                f"center={tuple(round(v, 2) for v in pl.world_center)}",
                "plane",
            )

        # Detect existing occupants per plane BEFORE running the LLM, so the
        # prompt can include them as obstacles. Critical for tables/desks
        # where Stage 3 already placed Monitor/Keyboard/Lamp/etc.
        self._log("Scanning for objects already on each plane...", "step")
        self._compute_plane_occupants()
        occupied = sum(1 for v in self.plane_occupants.values() if v)
        self._log(
            f"{occupied}/{len(self.planes)} planes already have occupants.",
            "info",
        )

        if self.skip_llm or not has_image:
            self._log("Skipping LLM step (dry run / no image).", "warning")
            # Still save planes.json for debugging.
            self._save_results(generated_code="")
            return True, {
                "planes": [p.to_dict() for p in self.planes],
                "items": [],
            }

        # LLM call per parent (parallelized: each parent is an independent
        # LLM call + pure-function resolve, so they fan out cleanly).
        system_prompt = self._load_system_prompt()
        planes_by_parent: Dict[str, List[Plane]] = {}
        for pl in self.planes:
            planes_by_parent.setdefault(pl.parent_name, []).append(pl)

        n_workers = min(self.parallel, len(planes_by_parent)) if planes_by_parent else 1
        self._log(
            f"Calling LLM for {len(planes_by_parent)} parent objects "
            f"({n_workers} parallel worker{'s' if n_workers != 1 else ''})...",
            "step",
        )

        def _process_one(pname: str, pls: List[Plane]):
            """Run one parent end-to-end inside a worker thread."""
            self._log(f" -> {pname}  ({len(pls)} planes)", "item")
            result = self._call_llm_for_parent(pname, pls, system_prompt)
            if not result:
                return pname, []
            new_items = self._resolve_items(pname, pls, result)
            n_llm = sum(
                len(p.get("items", []) or [])
                for p in result.get("planes", []) or []
            )
            self._log(
                f"    [{pname}] + {len(new_items)} items accepted "
                f"(LLM returned {n_llm})",
                "item",
            )
            return pname, new_items

        if n_workers <= 1:
            for parent_name, planes in planes_by_parent.items():
                _, new_items = _process_one(parent_name, planes)
                self.items.extend(new_items)
        else:
            # Pre-warm the LLM client so the first few workers don't race on
            # the lazy-init path (the DC-locking above also handles it, but
            # pre-warming keeps the first log lines cleaner).
            _ = self.llm
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                futures = {
                    ex.submit(_process_one, pname, pls): pname
                    for pname, pls in planes_by_parent.items()
                }
                for fut in as_completed(futures):
                    try:
                        _, new_items = fut.result()
                    except Exception as exc:  # noqa: BLE001
                        pname = futures[fut]
                        self._log(f"   parallel worker for {pname} raised: {exc}",
                                  "error")
                        continue
                    self.items.extend(new_items)

        # Code generation.
        self._log("Generating updated Blender code...", "step")
        generated_code = self._generate_code()

        paths = self._save_results(generated_code)

        print("\n" + "=" * 60)
        print("✅ Stage Small Objects complete")
        print(f"   planes : {len(self.planes)}")
        print(f"   items  : {len(self.items)}")
        print(f"   output : {paths.get('code', '(no code)')}")
        print("=" * 60)
        return True, {
            "planes": [p.to_dict() for p in self.planes],
            "items": [it.to_dict() for it in self.items],
            "paths": paths,
        }

    def _load_system_prompt(self) -> str:
        """Load Stage7_small_objects_task prompt file, with inline fallback."""
        prompt_path = os.path.join(
            os.path.dirname(current_dir), "agent_prompt", "Stage7_small_objects_task"
        )
        if os.path.exists(prompt_path):
            with open(prompt_path, "r", encoding="utf-8") as f:
                return f.read()
        # Minimal fallback so the module still runs if the prompt file is
        # missing. The real prompt lives in agent_prompt/Stage7_small_objects_task.
        return (
            "You are a 3D Interior Decorator. Place small decorative objects "
            "on the given planes, grounded on the reference image. Return JSON "
            "with the schema: {parent_name, planes:[{plane_id, plane_type, "
            "items:[{name,item_type,shape,local_uv,size,rotation_z,color_hint,"
            "description}]}]}. UVs in [0.05,0.95]. Items must fit inside the "
            "plane with 2 cm margin. No prose, no code fences."
        )


# ==============================================================================
# CLI
# ==============================================================================
def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage Small Objects - surface-driven small-object placement",
    )
    parser.add_argument("--image", "-i",
                        help="Reference top-down image path")
    parser.add_argument("--geometry-json",
                        help="Path to geometry_progress.json (Stage 6 output)")
    parser.add_argument("--describe-json",
                        help="Path to describe_output.json (Stage 7 output)")
    parser.add_argument("--base-code",
                        help="Path to geometry_output.py (Stage 6 integrated code)")
    parser.add_argument("--stage1-json",
                        help="Path to stage1_output.json (optional small-object hints)")
    parser.add_argument("--output-dir", "-o",
                        help="Output directory "
                             "(default: pipeline_output/stage7_small_objects)")
    parser.add_argument("--no-memory", action="store_true",
                        help="Do not read/write the Memory store")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only do plane discovery, skip the LLM call")
    parser.add_argument("--model", help="LLM model override",default="gemini-3.1-pro-preview")
    parser.add_argument("--base-url", help="LLM base URL override", default=os.environ.get("SCENEGEN_BASE_URL"))
    parser.add_argument(
        "--api-key",
        help="LLM API key override",
        default=os.environ.get("SCENEGEN_API_KEY") or os.environ.get("OPENAI_API_KEY"),
    )
    parser.add_argument("--parallel", type=int, default=4,
                        help="Number of parent objects to process in parallel "
                             "(each parent = one LLM call). Default 4; use 1 "
                             "for sequential / deterministic logging.")
    parser.add_argument("--quiet", action="store_true", help="Suppress info logs")
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()
    runner = StageSmallObjectsRunner(
        image_path=args.image,
        geometry_json_path=args.geometry_json,
        describe_json_path=args.describe_json,
        base_code_path=args.base_code,
        stage1_json_path=args.stage1_json,
        output_dir=args.output_dir,
        use_memory=not args.no_memory,
        verbose=not args.quiet,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        skip_llm=args.dry_run,
        parallel=args.parallel,
    )
    success, _ = runner.run()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
