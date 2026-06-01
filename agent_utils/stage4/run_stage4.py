"""Stage4 runner script - wall-mounted decorations + coarse placeholders for important minor objects

Scope (v3, 2026-05-01):
  Stage4 now runs in two phases:

  Phase A - Wall-Mounted Decorations (historical responsibility, unchanged):
      Append paintings / mirrors / wall clocks / wall sconces / wall-
      mounted TVs / wall shelves / pegboards / curtain rods ... on top of
      the Stage 3 code, flush against walls.

  Phase B - Important Minor Object Placeholders (new in v3):
      Read Stage 2 minor sidecar, code-side filter to "visually salient
      large minor items" (floor vases / planters / floor lamps / cushions /
      decorative bottles ...), then call the LLM to place rough bbox
      occupancy for them. Naming convention is
      `MinorPlace_<obj_id>_<short_label>` so that:
        - downstream `stage7_small_objects` can detect & avoid duplicates;
        - downstream `stage6_geometry` can recognize them as bbox-only
          stand-ins (todo for Stage 8 if needed).

  Surface-true placement (table-top, shelf-top, seat-top, etc. with
  realistic geometry) is STILL handled by `stage7_small_objects.py` after
  Stage 8 - Phase B here is intentionally coarse (uses parent bbox top
  guessed from Stage 3 code), so the two stages cooperate via the
  `stage4_minor_placed_obj_ids` list (Phase B publishes it; small_objects
  consumes it).
"""
import os
import re
import sys
import json
import math
import base64
import argparse
from typing import Any, Dict, List, Optional, Set, Tuple

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, "stage3"))

from stage3.core import (
    LLMClient,
    PromptManager,
    extract_python_from_response,
    extract_json_from_response,
)
from memory import Memory
from langchain_core.messages import HumanMessage, SystemMessage


class Stage4Runner:
    """Stage4 runner - appends wall-mounted decorations on top of Stage3 (wall-mounted only).

    Non-wall small objects (table-top, shelf, seat, floor) are handled by
    `stage7_small_objects.py` which runs AFTER Stage 8.
    """

    def __init__(
        self,
        image_path: str = None,
        output_dir: str = "./output",
        verbose: bool = True,
        memory_file: str = "agent_memory.jsonl",
        model: str = "gemini-3.1-pro-preview",
        base_url: str = None,
        api_key: str = None,
    ):
        self.image_path = image_path
        self.output_dir = output_dir
        self.verbose = verbose

        # Initialize
        self.memory = Memory(workspace_dir=parent_dir, memory_file=memory_file)
        self.prompts = PromptManager()
        self.llm = LLMClient(
            model=model,
            base_url=base_url or os.environ.get("SCENEGEN_BASE_URL"),
            api_key=api_key or os.environ.get("SCENEGEN_API_KEY") or os.environ.get("OPENAI_API_KEY"),
        )

        # Data
        self.stage1_json = None
        self.stage2_json: Optional[Dict[str, Any]] = None
        self.stage3_code = None
        # v3: minor sidecar (from stage2)
        self.minor_objects: List[Dict[str, Any]] = []
        # v3: minor obj_ids actually placed by stage4 (Phase B output, used downstream to avoid duplicates)
        self.stage4_minor_placed: List[Dict[str, Any]] = []

    def _log(self, msg: str, level: str = "info"):
        if self.verbose:
            prefix = {"info": "[i]", "success": "[OK]", "warning": "[!]", "error": "[X]", "step": "[>]"}.get(level, "")
            print(f"{prefix} {msg}")

    def _encode_image(self, path: str) -> tuple:
        """Encode an image"""
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        ext = os.path.splitext(path)[1].lower()
        mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}.get(ext, "image/png")
        return b64, mime

    def _load_data(self) -> bool:
        """Load data from Memory"""
        self._log("Fetching data from Memory...", "step")

        # Stage 1
        stage1_entry = self.memory.get_latest(stage="stage1", type="result")
        if not stage1_entry:
            self._log("No Stage1 result in Memory!", "error")
            return False
        self.stage1_json = stage1_entry.content
        self._log("Stage1: OK", "success")

        # Stage 2
        stage2_entry = self.memory.get_latest(stage="stage2", type="result")
        if not stage2_entry:
            self._log("No Stage2 result in Memory!", "error")
            return False
        self.stage2_json = stage2_entry.content
        self._log("Stage2: OK", "success")

        # v3: minor sidecar - prefer stage2 result directly; otherwise a separate entry;
        # finally fall back to disk stage2_minor_objects.json (path in metadata).
        self.minor_objects = self._load_minor_sidecar(stage2_entry)
        self._log(f"Minor sidecar: {len(self.minor_objects)} entries", "success"
                  if self.minor_objects else "warning")

        # Stage 3
        stage3_entry = self.memory.get_latest(stage="stage3", type="result")
        if not stage3_entry:
            self._log("No Stage3 result in Memory!", "error")
            return False
        self.stage3_code = stage3_entry.content
        self._log("Stage3: OK", "success")

        # Image
        if not self.image_path:
            self.image_path = stage1_entry.metadata.get("image_path")

        if self.image_path and os.path.exists(self.image_path):
            self._log(f"Image: {self.image_path}", "success")
        else:
            self._log("No image", "warning")
            self.image_path = None

        return True

    # ------------------------------------------------------------------
    # v3: minor sidecar loading + coarse filter
    # ------------------------------------------------------------------
    def _load_minor_sidecar(self, stage2_entry) -> List[Dict[str, Any]]:
        """Load the stage2 minor sidecar.

        Priority:
          1. The minor_objects field at the top of stage2 result content (v3 new)
          2. A separate stage2/minor_objects entry
          3. The disk file pointed to by metadata['minor_objects_json']
        Returns [] when no source is available, letting Phase B silently skip.
        """
        # 1) Pull directly from result content
        try:
            content = stage2_entry.content if stage2_entry else None
            if isinstance(content, dict):
                lst = content.get("minor_objects")
                if isinstance(lst, list) and lst:
                    return lst
        except Exception:
            pass
        # 2) Separate entry
        try:
            sub = self.memory.get_latest(stage="stage2", type="minor_objects")
            if sub and isinstance(sub.content, dict):
                lst = sub.content.get("minor_objects")
                if isinstance(lst, list) and lst:
                    return lst
        except Exception:
            pass
        # 3) Disk fallback
        try:
            meta = stage2_entry.metadata if stage2_entry else {}
            json_path = (meta or {}).get("minor_objects_json")
            if json_path and os.path.exists(json_path):
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                lst = data.get("minor_objects")
                if isinstance(lst, list):
                    return lst
        except Exception:
            pass
        return []

    # Visually salient minor-type keywords (lowercase substring match against label).
    # This is only a coarse filter; which to place and where is still decided by the LLM looking at the image.
    _SALIENT_MINOR_KEYWORDS: Tuple[str, ...] = (
        "vase", "planter", "plant", "pot ", "pot,", "pots", "potted",
        "flower", "bouquet", "orchid", "fern", "palm", "tree",
        "lamp", "lantern", "candle", "candelabra",
        "sculpture", "statue", "bust", "figurine", "ornament",
        "pillow", "cushion", "throw",
        "basket", "trunk", "chest", "globe",
        "stack of", "tray", "tea set", "decanter", "bottle",
    )

    def _filter_important_minors(
        self, minors: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Code-side coarse filter: select "visually salient large minor items" suitable for stage4 placeholders.

        Rules:
          - placement_type == "wall" -> skip (handled in Phase A)
          - parent_category == "minor" -> skip (nested small object, better handled by
            stage7_small_objects on real planes)
          - label must hit one of _SALIENT_MINOR_KEYWORDS (substring, case-insensitive)
        Returns the filtered list (keeps the original sidecar fields for prompt injection).
        """
        out: List[Dict[str, Any]] = []
        for m in minors or []:
            placement = (m.get("placement_type") or "").lower()
            if placement == "wall":
                continue
            parent_cat = (m.get("parent_category") or "").lower()
            if parent_cat == "minor":
                continue
            label = (m.get("label") or "").lower()
            if not label:
                continue
            if not any(k in label for k in self._SALIENT_MINOR_KEYWORDS):
                continue
            out.append(m)
        return out

    # ------------------------------------------------------------------
    # Prompt slim-down: extract only the wall-relevant slice of Stage 1.
    # ------------------------------------------------------------------
    def _build_wall_hints(self, stage1_data) -> dict:
        """Extract the subset of Stage 1 output relevant to wall placement.

        Returns a small dict with:
            scene_scale : brief room-size info (helps the LLM pick Z heights)
            wall_objects: filtered object_hierarchy entries that are wall-
                          mounted (placement_type == 'wall' OR parent_object
                          starts with 'wall:')

        Stage 1 Memory `content` is usually a parsed JSON object, but be
        defensive — if it's a string, decode it; if it's something else,
        return an empty skeleton rather than failing the run.
        """
        if isinstance(stage1_data, str):
            try:
                stage1_data = json.loads(stage1_data)
            except Exception:
                return {"scene_scale": {}, "wall_objects": []}
        if not isinstance(stage1_data, dict):
            return {"scene_scale": {}, "wall_objects": []}

        scene_scale = stage1_data.get("scene_scale_understanding", {}) or {}

        wall_objects = []
        for zone in stage1_data.get("decoupled_zones", []) or []:
            zone_name = zone.get("zone_name", "")
            for obj in zone.get("object_hierarchy", []) or []:
                placement = obj.get("placement_type", "")
                parent = obj.get("parent_object") or ""
                is_wall = (
                    placement == "wall"
                    or (isinstance(parent, str) and parent.startswith("wall:"))
                )
                if not is_wall:
                    continue
                wall_objects.append({
                    "name": obj.get("name"),
                    "category": obj.get("category"),
                    "placement_type": placement,
                    "parent_object": parent,
                    "zone": zone_name,
                })

        return {"scene_scale": scene_scale, "wall_objects": wall_objects}

    def _generate_code(self) -> str:
        """Generate code with ONLY wall-mounted objects appended.

        Scope is enforced by the system prompt (`Stage4_task`). Table-top,
        shelf, seat, and floor items are out of scope for this stage — they
        are handled by `stage7_small_objects.py` downstream.

        Prompt is intentionally minimal: we only need the wall-mounted
        entries of Stage 1's `object_hierarchy` plus a tiny scene-scale
        hint. Stage 2 (scene graph of floor furniture relationships) is
        irrelevant here and is dropped to keep the request small and fast.
        """
        self._log("Generating code (wall-mounted decorations only)...", "step")

        system_prompt = self.prompts.get("Stage4_task")

        # Filter Stage 1 to just wall-relevant hints. This shrinks the
        # request body by ~95% vs dumping the full stage1_json.
        wall_hints = self._build_wall_hints(self.stage1_json)

        # Early exit: if Stage 1 reports no wall-mounted objects, there is
        # nothing for this stage to add. Return the Stage 3 code unchanged
        # instead of burning an LLM call (which, on a very long Stage 3
        # script, may take minutes and occasionally hangs on the gateway).
        if not wall_hints.get("wall_objects"):
            self._log(
                "No wall-mounted objects in Stage 1 — skipping LLM; "
                "passing Stage 3 code through unchanged.",
                "info",
            )
            return self.stage3_code

        user_text = f"""Add ONLY wall-mounted decorations (paintings, mirrors,
wall clocks, wall sconces, wall-mounted TVs, wall shelves, pegboards,
curtain rods, etc.) to the Stage 3 code.

Everything that sits on a surface (table-top / shelf / cabinet-top / seat /
floor) is OUT OF SCOPE for this stage. Do NOT add it here — it is handled
by a later stage (stage7_small_objects). Ignore any non-wall items even if
listed below.

## Scene scale (from Stage 1)
```json
{json.dumps(wall_hints.get("scene_scale", {}), ensure_ascii=False, indent=2)}
```

## Wall-mounted object hints (filtered from Stage 1 object_hierarchy)
Every entry below has `placement_type == "wall"` and, when available, a
`parent_object` of the form `wall:<direction>`. These are the ONLY objects
this stage is allowed to add.
```json
{json.dumps(wall_hints.get("wall_objects", []), ensure_ascii=False, indent=2)}
```

## Stage 3 Code (Base — DO NOT MODIFY)
The walls are already defined in this code (look for `create_box` calls
whose name contains `wall`). Read their center + dimensions to position
wall-mounted items flush against the correct wall surface.
```python
{self.stage3_code}
```

Task:
1. For each entry in "Wall-mounted object hints" above, check Stage 3 code
   (case-insensitive) to see if it is already present. Skip duplicates.
2. Append the missing wall objects to the end of Stage 3's main build
   function using `create_box` / `create_cylinder`, following the rotation
   / Z-height / wall-flush rules in the system prompt.
3. Do not modify any existing Stage 3 code.
4. Output the COMPLETE executable Blender Python script. Code only, no
   commentary and no markdown fences.
"""
        
        # Build messages
        if self.image_path:
            b64, mime = self._encode_image(self.image_path)
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=[
                    {"type": "text", "text": "Reference image:"},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    {"type": "text", "text": user_text}
                ])
            ]
        else:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_text)
            ]

        try:
            response = self.llm.invoke(messages)

            # Debug: save raw response
            raw_path = os.path.join(self.output_dir, "stage4_raw.txt")
            os.makedirs(self.output_dir, exist_ok=True)
            with open(raw_path, "w", encoding="utf-8") as f:
                f.write(response if response else "(empty response)")

            code = extract_python_from_response(response)

            if not code:
                resp_len = len(response) if response else 0
                self._log(
                    f"Failed to extract code (response length: {resp_len}) - LLM possibly returned "
                    f"empty content (gemini-thinking budget exhaustion or "
                    f"upstream gateway hiccup).",
                    "error",
                )
                self._log(f"Raw response saved: {raw_path}", "info")
                if response:
                    preview = response[:500] + "..." if len(response) > 500 else response
                    self._log(f"Response preview: {preview}", "info")
                # Graceful degradation: pass Stage 3 code through unchanged.
                # Wall decorations are nice-to-have; losing them must NOT
                # cascade-fail the whole pipeline. Downstream stages (small
                # objects / material / texture / render) all happily accept
                # the bare Stage 3 scene.
                self._log(
                    "Falling back to Stage 3 code unchanged (no wall objects "
                    "added this run).",
                    "warning",
                )
                return self.stage3_code

            try:
                compile(code, '<string>', 'exec')
                self._log(f"Code generated ({code.count(chr(10)) + 1} lines)", "success")
            except SyntaxError as e:
                self._log(f"Syntax warning (line {e.lineno}): {e.msg}", "warning")

            return code

        except Exception as e:
            self._log(f"Generation failed: {e}", "error")
            import traceback
            traceback.print_exc()
            # Same graceful-degradation policy as the empty-response branch:
            # pipeline keeps moving with the Stage 3 code.
            self._log(
                "Falling back to Stage 3 code unchanged (no wall objects "
                "added this run).",
                "warning",
            )
            return self.stage3_code

    # ------------------------------------------------------------------
    # v3 Phase B: coarse minor-object placeholder generation
    # ------------------------------------------------------------------
    def _generate_minor_placements(
        self, base_code: str, important_minors: List[Dict[str, Any]]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Append minor-object placeholders on top of code (base_code) that already contains wall-mounted items.

        Patch-style implementation (2026-05-03 rewrite). The LLM no longer
        echoes the entire base script; it returns ONLY a JSON list of
        placements:

            {"placements": [{"obj_id", "label", "primitive",
                             "location", "dimensions", "rotation"}, ...]}

        Python then formats each placement into a single
        ``create_box`` / ``create_cylinder`` call and splices the calls
        into the ``run_layout_engine()`` body just before the
        ``if __name__ == "__main__":`` invocation. This:

          - cuts LLM output from ~10 KB ("repeat the whole script") to
            ~0.5 KB (a tiny JSON), giving 5-10x speedup;
          - eliminates the syntax-error class where the LLM mis-copies
            base code;
          - dodges the gemini-thinking visible-output budget exhaustion
            that empties Phase A on the same stage call.

        Naming is enforced by the splicer (not by the LLM):
        ``MinorPlace_<obj_id>__<snake_label>`` so downstream
        ``stage7_small_objects`` can detect & avoid duplicates.

        On any failure (empty LLM, malformed JSON, all entries rejected),
        returns ``(base_code, [])`` so the pipeline keeps moving.
        """
        if not important_minors:
            return base_code, []

        self._log(
            f"Phase B: {len(important_minors)} candidate minor(s); "
            f"calling LLM to select and place (patch-style)...",
            "step",
        )

        scene_scale = (self.stage1_json or {}).get(
            "scene_scale_understanding", {}) or {}

        candidates_payload = []
        for m in important_minors:
            candidates_payload.append({
                "obj_id": m.get("id"),
                "label": m.get("label"),
                "zone_id": m.get("zone_id"),
                "placement_type": m.get("placement_type"),
                "parent_id": m.get("parent_id"),
                "parent_label": m.get("parent_label"),
            })

        # Compact summary of objects already in base_code so the LLM has
        # parent / collision context without having to read the full
        # script. Each entry: {"name", "kind", "location", "dimensions"}.
        existing_summary = self._extract_existing_object_summary(base_code)

        system_prompt = self.prompts.get("Stage4_phase_b_task")
        if not system_prompt:
            self._log(
                "Stage4_phase_b_task prompt does not exist; skipping Phase B.",
                "warning",
            )
            return base_code, []

        user_text = f"""## Scene scale
```json
{json.dumps(scene_scale, ensure_ascii=False, indent=2)}
```

## Existing objects (from Stage 3 + Phase A wall-mounted)
{len(existing_summary)} entries; each: name, kind, location, dimensions.
Use this for parent lookup and collision avoidance.
```json
{json.dumps(existing_summary, ensure_ascii=False)}
```

## Candidate minor objects (you choose which to keep)
```json
{json.dumps(candidates_payload, ensure_ascii=False, indent=2)}
```

Return the JSON object described in the system prompt. Output JSON only.
"""

        if self.image_path:
            b64, mime = self._encode_image(self.image_path)
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=[
                    {"type": "text", "text": "Reference image:"},
                    {"type": "image_url",
                     "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    {"type": "text", "text": user_text},
                ]),
            ]
        else:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_text),
            ]

        try:
            response = self.llm.invoke(messages)
        except Exception as exc:  # noqa: BLE001
            self._log(f"Phase B LLM call failed: {exc}", "error")
            return base_code, []

        # Debug: persist raw response
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            raw_path = os.path.join(self.output_dir, "stage4_minor_raw.txt")
            with open(raw_path, "w", encoding="utf-8") as f:
                f.write(response if response else "(empty response)")
        except OSError:
            pass

        if not response:
            self._log("Phase B returned empty, skipping minor placeholders", "warning")
            return base_code, []

        placements = self._parse_phase_b_response(
            response, candidates_payload
        )
        if not placements:
            self._log(
                "Phase B parsed 0 valid placements; keeping base_code",
                "warning",
            )
            return base_code, []

        candidates_by_id = {c["obj_id"]: c for c in candidates_payload}

        # Backstop: re-clamp z of soft items (pillow / cushion / throw /
        # blanket) sitting on a soft seat parent so the item TOP is flush
        # with the parent TOP, not floating above it. Catches LLM
        # regressions to the hard-surface rule.
        n_clamped = self._enforce_soft_seat_embed(
            placements, candidates_by_id, existing_summary
        )
        if n_clamped:
            self._log(
                f"Phase B post-process: {n_clamped} soft item(s) re-embedded "
                f"into their soft-seat parent (seat-plane).",
                "info",
            )

        # Greedy collision resolver: every minor must not overlap any
        # non-parent furniture bbox or any earlier-accepted minor. We try
        # to slide along the parent's longest horizontal axis first, then
        # fall back to a small 2D grid; if still impossible we drop it.
        placements, dropped_ids = self._resolve_minor_collisions(
            placements, candidates_by_id, existing_summary,
        )
        if dropped_ids:
            self._log(
                f"Phase B post-process: dropped {len(dropped_ids)} minor "
                f"placement(s) that could not be fit collision-free: "
                f"{dropped_ids}",
                "warning",
            )

        # Build create_* call lines + records side-by-side.
        call_lines: List[str] = []
        records: List[Dict[str, Any]] = []
        for p in placements:
            line = self._format_minor_call(p)
            if not line:
                continue
            call_lines.append(line)
            sidecar = candidates_by_id.get(p["obj_id"], {})
            records.append({
                "obj_id": p["obj_id"],
                "block_name": p["block_name"],
                "short_label": p["label"],
                "label": sidecar.get("label"),
                "zone_id": sidecar.get("zone_id"),
                "placement_type": sidecar.get("placement_type"),
                "parent_id": sidecar.get("parent_id"),
                "parent_label": sidecar.get("parent_label"),
            })

        if not call_lines:
            self._log(
                "Phase B: all placements failed validation; keeping base_code",
                "warning",
            )
            return base_code, []

        new_code = self._splice_minor_into_main_func(base_code, call_lines)
        try:
            compile(new_code, "<string>", "exec")
        except SyntaxError as e:
            self._log(
                f"Phase B syntax error after splicing (line {e.lineno}: {e.msg}), "
                f"falling back to base_code",
                "warning",
            )
            return base_code, []

        self._log(
            f"Phase B done: parsed {len(placements)} placement(s) -> "
            f"appended {len(call_lines)} create_* call(s)",
            "success",
        )
        return new_code, records

    # ------------------------------------------------------------------
    # Phase B helpers (patch-style, 2026-05-03)
    # ------------------------------------------------------------------
    _BASE_CALL_RE = re.compile(
        r"create_(box|cylinder)\s*\(\s*"
        r"(?:f|rf|fr)?[\"']([^\"']+)[\"']\s*,\s*"
        r"(\([^()]*\))\s*,\s*"
        r"(\([^()]*\))",
        re.IGNORECASE,
    )

    def _extract_existing_object_summary(
        self, code: str
    ) -> List[Dict[str, Any]]:
        """Compact name/loc/dim/kind summary of every create_box/cylinder
        call in ``code``. Tuples are evaluated in a math+SCENE_W/SCENE_D
        sandbox; calls whose tuples don't reduce to numbers (e.g. mid-loop
        f-string names) are dropped silently — the LLM doesn't need to
        see them for placement reasoning.
        """
        if not code:
            return []
        ns = self._eval_wall_namespace(code)
        out: List[Dict[str, Any]] = []
        for m in self._BASE_CALL_RE.finditer(code):
            kind, name, loc_str, dim_str = m.group(1), m.group(2), m.group(3), m.group(4)
            if "{" in name:
                continue  # skip f-string templated names
            loc = self._eval_tuple(loc_str, ns)
            dim = self._eval_tuple(dim_str, ns)
            if loc is None or dim is None:
                continue
            out.append({
                "name": name,
                "kind": kind.lower(),
                "location": [round(float(v), 4) for v in loc],
                "dimensions": [round(float(v), 4) for v in dim],
            })
        return out

    @staticmethod
    def _to_snake_label(s: Any) -> str:
        s = str(s or "").strip().lower()
        s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
        return s or "item"

    def _parse_phase_b_response(
        self,
        raw: str,
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Parse the LLM's JSON output into a list of validated placements.

        Each returned dict is normalized to:
            {"obj_id", "label", "primitive",
             "location": (x,y,z), "dimensions": (dx,dy,dz),
             "rotation": (rx,ry,rz),
             "block_name": "MinorPlace_<obj_id>__<label>"}
        Invalid / duplicate entries are dropped.
        """
        json_str = extract_json_from_response(raw)
        try:
            data = json.loads(json_str) if json_str else None
        except json.JSONDecodeError:
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                self._log("Phase B JSON parse failed", "warning")
                return []
        if not isinstance(data, dict):
            return []
        items = data.get("placements")
        if not isinstance(items, list):
            return []

        valid_ids = {c["obj_id"] for c in candidates}
        seen: Set[str] = set()
        out: List[Dict[str, Any]] = []
        for raw_item in items:
            if not isinstance(raw_item, dict):
                continue
            obj_id = str(raw_item.get("obj_id") or "").strip()
            if obj_id not in valid_ids or obj_id in seen:
                continue
            try:
                loc = tuple(float(v) for v in raw_item.get("location") or ())
                dim = tuple(float(v) for v in raw_item.get("dimensions") or ())
                rot_raw = raw_item.get("rotation") or [0, 0, 0]
                rot = tuple(float(v) for v in rot_raw)
            except (TypeError, ValueError):
                continue
            if len(loc) != 3 or len(dim) != 3 or len(rot) != 3:
                continue
            if any(d <= 0 for d in dim):
                continue
            primitive = str(raw_item.get("primitive") or "box").lower()
            if primitive not in ("box", "cylinder"):
                primitive = "box"
            label = self._to_snake_label(raw_item.get("label"))[:32]
            seen.add(obj_id)
            out.append({
                "obj_id": obj_id,
                "label": label,
                "primitive": primitive,
                "location": loc,
                "dimensions": dim,
                "rotation": rot,
                "block_name": f"MinorPlace_{obj_id}__{label}",
            })
        return out

    # Soft-seat keyword sets for the post-process "embed cushion" step.
    # Matches against case-insensitive ``parent_label`` (sidecar) and
    # ``label`` (LLM short tag + sidecar full label).
    _SOFT_SEAT_PARENT_KEYWORDS = (
        "sofa", "couch", "loveseat", "armchair", "accent chair",
        "sectional", "chaise", "daybed", "bed", "pouf", "ottoman",
        "bean bag", "beanbag", "floor cushion", "banquette",
    )
    _SOFT_ITEM_KEYWORDS = (
        "pillow", "cushion", "throw", "blanket", "quilt", "duvet",
        "bedding", "sham",
    )

    # Coarse parent bboxes for upholstered seating include the backrest /
    # headboard, so the bbox crest is NOT the seat plane. We model the seat
    # plane as a fraction of bbox height from the floor: sofas/armchairs ~0.55,
    # beds/poufs almost the whole bbox.
    _SEAT_PLANE_FRACTION_BY_KW = {
        "loveseat":       0.55,
        "sectional":      0.55,
        "accent chair":   0.55,
        "armchair":       0.55,
        "banquette":      0.55,
        "chaise":         0.55,
        "daybed":         0.85,
        "ottoman":        0.95,
        "pouf":           0.95,
        "bean bag":       0.55,
        "beanbag":        0.55,
        "floor cushion":  0.95,
        "couch":          0.55,
        "sofa":           0.55,
        "bed":            0.92,
    }
    # Most-specific keyword first so e.g. "daybed" wins over "bed",
    # "loveseat" wins over "sofa" if both happen to be in the label.
    _SEAT_PLANE_FRACTION_KW_ORDER = (
        "loveseat", "sectional", "accent chair", "armchair", "banquette",
        "chaise", "daybed", "ottoman", "pouf", "bean bag", "beanbag",
        "floor cushion", "couch", "sofa", "bed",
    )
    _SEAT_REST_SINK_MIN_M = 0.005
    _SEAT_REST_SINK_MAX_M = 0.020
    _SEAT_REST_SINK_ITEM_RATIO = 0.10

    @classmethod
    def _seat_plane_fraction(cls, parent_label: str) -> float:
        low = (parent_label or "").lower()
        for kw in cls._SEAT_PLANE_FRACTION_KW_ORDER:
            if kw in low:
                return cls._SEAT_PLANE_FRACTION_BY_KW[kw]
        return 0.55

    @classmethod
    def _soft_accessory_rest_z(
        cls,
        parent_loc_z: float,
        parent_dim_z: float,
        item_h: float,
        parent_label: str,
    ) -> float:
        """Pillow / cushion / blanket nominal centre on the parent SEAT PLANE.

        ``parent_loc_z`` / ``parent_dim_z`` are the parent's coarse bbox center
        and height. We find the seat plane via ``_seat_plane_fraction``, place
        the item ON that plane, and then sink a few millimetres so the bottom
        looks "settled" rather than levitating.
        """
        cz = float(parent_loc_z)
        hz = float(parent_dim_z)
        h_item = float(item_h)
        zb = cz - hz / 2.0
        zt = cz + hz / 2.0

        frac = cls._seat_plane_fraction(parent_label)
        seat_plane_z = zb + frac * hz
        center_z = seat_plane_z + h_item / 2.0
        sink = max(
            cls._SEAT_REST_SINK_MIN_M,
            min(cls._SEAT_REST_SINK_MAX_M, h_item * cls._SEAT_REST_SINK_ITEM_RATIO),
        )
        new_z = center_z - sink

        if new_z + h_item / 2.0 > zt:
            new_z = zt - h_item / 2.0
        if new_z - h_item / 2.0 < zb:
            new_z = zb + h_item / 2.0
        return float(new_z)

    @staticmethod
    def _aabb_overlap_3d(
        loc_a: Tuple[float, float, float],
        dim_a: Tuple[float, float, float],
        loc_b: Tuple[float, float, float],
        dim_b: Tuple[float, float, float],
        slack: float = 0.0,
    ) -> bool:
        """3D axis-aligned bbox overlap test (positive ``slack`` = tighter)."""
        for ax in range(3):
            sep = abs(float(loc_a[ax]) - float(loc_b[ax]))
            min_sep = (float(dim_a[ax]) + float(dim_b[ax])) / 2.0 - slack
            if sep >= min_sep:
                return False
        return True

    def _find_parent_entry_for_minor(
        self,
        candidate_sidecar: Dict[str, Any],
        existing_summary: List[Dict[str, Any]],
        require_soft: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Resolve a Phase B candidate's `parent_label` (and optional
        `parent_id`) to one entry in ``existing_summary``.

        - First tries `parent_id` exact match (we store coarse boxes
          with names like ``obj_002_SouthSofa`` that contain the id).
        - Otherwise falls back to keyword + token overlap between
          `parent_label` and each entry name.
        - When ``require_soft`` is True, restricts candidates to entries
          whose name contains a soft-seat keyword.
        """
        if not candidate_sidecar or not existing_summary:
            return None
        parent_label_raw = (candidate_sidecar.get("parent_label") or "").strip()
        parent_label = parent_label_raw.lower()
        parent_id = (candidate_sidecar.get("parent_id") or "").strip()

        name_low_entries = [(e["name"].lower(), e) for e in existing_summary]

        if parent_id:
            pid_low = parent_id.lower()
            for low, entry in name_low_entries:
                if pid_low in low:
                    if not require_soft or any(
                        kw in low for kw in self._SOFT_SEAT_PARENT_KEYWORDS
                    ):
                        return entry

        if not parent_label:
            return None

        soft_parent_hits = [
            kw for kw in self._SOFT_SEAT_PARENT_KEYWORDS if kw in parent_label
        ]
        if require_soft and not soft_parent_hits:
            return None

        if soft_parent_hits:
            pool = [
                (low, entry) for low, entry in name_low_entries
                if any(kw in low for kw in soft_parent_hits)
            ]
        else:
            label_tokens = {
                t for t in re.findall(r"[a-z]+", parent_label) if len(t) >= 3
            }
            pool = [
                (low, entry) for low, entry in name_low_entries
                if any(t in low for t in label_tokens)
            ]
        if not pool:
            return None
        if len(pool) == 1:
            return pool[0][1]

        p_tokens = {
            t for t in re.findall(r"[a-z]+", parent_label) if len(t) >= 3
        }
        best, best_overlap = None, -1
        for low, entry in pool:
            e_tokens = set(re.findall(r"[a-z]+", low))
            overlap = len(p_tokens & e_tokens)
            if overlap > best_overlap:
                best, best_overlap = entry, overlap
        return best

    # Margin (in meters) used by the collision resolver. Small positive value
    # acts as a tightness — items must be at least this far apart on every
    # axis to be considered non-colliding.
    _MINOR_COLLISION_SLACK_M = 0.005

    def _resolve_minor_collisions(
        self,
        placements: List[Dict[str, Any]],
        candidates_by_id: Dict[str, Dict[str, Any]],
        existing_summary: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Greedy spatial resolver for Phase B placements.

        Each minor is checked against (existing_summary minus its own
        parent) plus already-accepted minors. On collision we try to
        slide along the parent's longest horizontal axis to a free
        slot inside the parent footprint; if that fails we drop the
        placement entirely.

        Mutates ``placements[i]["location"]`` (x or y may move).
        Returns ``(accepted, dropped_obj_ids)``.
        """
        if not placements:
            return placements, []

        slack = self._MINOR_COLLISION_SLACK_M
        accepted: List[Dict[str, Any]] = []
        dropped_ids: List[str] = []

        # Preserve the original (LLM-provided) order: the model has
        # already ranked items by importance. Drop late, low-priority
        # items rather than displacing main-feature ones.
        for p in placements:
            cand = candidates_by_id.get(p["obj_id"], {}) or {}
            parent_entry = self._find_parent_entry_for_minor(cand, existing_summary)

            try:
                loc = [float(v) for v in p["location"]]
                dim = [float(v) for v in p["dimensions"]]
            except (TypeError, ValueError, IndexError):
                dropped_ids.append(p["obj_id"])
                continue

            obstacles: List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = []
            for entry in existing_summary:
                if parent_entry is not None and entry.get("name") == parent_entry.get("name"):
                    continue
                eloc = entry.get("location")
                edim = entry.get("dimensions")
                if not eloc or not edim:
                    continue
                obstacles.append((tuple(eloc), tuple(edim)))
            for q in accepted:
                obstacles.append((tuple(q["location"]), tuple(q["dimensions"])))

            def _ok_at(test_loc: List[float]) -> bool:
                for o_loc, o_dim in obstacles:
                    if self._aabb_overlap_3d(
                        tuple(test_loc), tuple(dim), o_loc, o_dim, slack=slack
                    ):
                        return False
                return True

            if _ok_at(loc):
                accepted.append({
                    "obj_id": p["obj_id"],
                    "location": tuple(loc),
                    "dimensions": tuple(dim),
                })
                continue

            moved = False
            if parent_entry is not None:
                p_loc = parent_entry["location"]
                p_dim = parent_entry["dimensions"]
                long_axis = 0 if float(p_dim[0]) >= float(p_dim[1]) else 1
                short_axis = 1 - long_axis
                half_long = float(p_dim[long_axis]) / 2.0 - float(dim[long_axis]) / 2.0
                half_short = float(p_dim[short_axis]) / 2.0 - float(dim[short_axis]) / 2.0
                step_count = 24
                # First sweep the long axis with x stays / fractional positions.
                # Then a 2-axis grid as a fallback.
                long_fractions: List[float] = [0.0]
                for k in range(1, step_count + 1):
                    long_fractions.extend([k / step_count, -k / step_count])
                for f_long in long_fractions:
                    if half_long <= 0:
                        continue
                    test_loc = list(loc)
                    test_loc[long_axis] = float(p_loc[long_axis]) + f_long * half_long
                    if _ok_at(test_loc):
                        loc = test_loc
                        moved = True
                        break

                if not moved and half_short > 0:
                    short_fractions = [0.0]
                    for k in range(1, 9):
                        short_fractions.extend([k / 8.0, -k / 8.0])
                    for f_long in long_fractions:
                        if moved:
                            break
                        if half_long <= 0:
                            continue
                        for f_short in short_fractions:
                            test_loc = list(loc)
                            test_loc[long_axis] = float(p_loc[long_axis]) + f_long * half_long
                            test_loc[short_axis] = float(p_loc[short_axis]) + f_short * half_short
                            if _ok_at(test_loc):
                                loc = test_loc
                                moved = True
                                break

            if moved:
                p["location"] = (
                    round(loc[0], 4), round(loc[1], 4), round(loc[2], 4)
                )
                accepted.append({
                    "obj_id": p["obj_id"],
                    "location": tuple(p["location"]),
                    "dimensions": tuple(dim),
                })
            else:
                dropped_ids.append(p["obj_id"])

        accepted_ids = {a["obj_id"] for a in accepted}
        kept = [p for p in placements if p["obj_id"] in accepted_ids]
        return kept, dropped_ids

    def _enforce_soft_seat_embed(
        self,
        placements: List[Dict[str, Any]],
        candidates_by_id: Dict[str, Dict[str, Any]],
        existing_summary: List[Dict[str, Any]],
    ) -> int:
        """Backstop for pillow / cushion / blanket placements on soft parents.

        The LLM often leaves soft items halfway between "hard tabletop" (+h/2
        rule) and a physically sensible rest pose. Older guidance also misused
        "align TOP with bbox TOP": that places the pillow in the sofa's upper
        voxel band and punches through Stage 8 mesh.

        Correct coarse bbox heuristic: nominal rest sits ON the parent crest
        (same sign as hard-surface rule: ``zt + item_h/2``), then subtract a SHORT
        world-Z penetration so only the lower cap sinks into cushioning.

        Mutates ``placements[i]["location"]`` in place (x/y unchanged).
        Returns the number of items actually re-clamped.
        """
        if not placements or not existing_summary:
            return 0

        n_changed = 0
        for p in placements:
            cand = candidates_by_id.get(p["obj_id"])
            if not cand:
                continue
            ptype = (cand.get("placement_type") or "").strip().lower()
            if ptype != "surface":
                continue
            parent_label_raw = (cand.get("parent_label") or "").strip()
            if not parent_label_raw:
                continue
            parent_label = parent_label_raw.lower()
            item_label = f"{p.get('label', '')} {cand.get('label', '') or ''}".lower()

            if not any(kw in parent_label for kw in self._SOFT_SEAT_PARENT_KEYWORDS):
                continue
            if not any(kw in item_label for kw in self._SOFT_ITEM_KEYWORDS):
                continue

            parent_entry = self._find_parent_entry_for_minor(
                cand, existing_summary, require_soft=True,
            )
            if parent_entry is None:
                continue
            parent_loc = parent_entry.get("location") or [0.0, 0.0, 0.0]
            parent_dim = parent_entry.get("dimensions") or [0.0, 0.0, 0.0]
            try:
                pz = float(parent_loc[2])
                ph = float(parent_dim[2])
            except (TypeError, ValueError, IndexError):
                continue

            item_h = float(p["dimensions"][2])
            try:
                new_z = self._soft_accessory_rest_z(
                    pz, ph, item_h, parent_label_raw,
                )
            except (TypeError, ValueError):
                continue

            old_loc = p["location"]
            if abs(float(old_loc[2]) - new_z) <= 0.01:
                continue
            p["location"] = (
                float(old_loc[0]), float(old_loc[1]), round(new_z, 4)
            )
            n_changed += 1

        return n_changed

    @staticmethod
    def _format_minor_call(p: Dict[str, Any]) -> str:
        """Render one validated placement as a single Python line."""
        creator = "create_box" if p["primitive"] == "box" else "create_cylinder"
        loc = tuple(round(v, 4) for v in p["location"])
        dim = tuple(round(v, 4) for v in p["dimensions"])
        rot = tuple(round(v, 4) for v in p["rotation"])
        rot_part = f", rotation={rot}" if any(abs(v) > 1e-6 for v in rot) else ""
        # No material / collection: those reference local vars inside
        # run_layout_engine that may or may not exist; downstream
        # stage10_material assigns per-object PBR anyway, so leaving these
        # off is safe and keeps the splice scope-agnostic.
        return (
            f'{creator}("{p["block_name"]}", {loc}, {dim}{rot_part}'
            f', show_direction=False)'
        )

    @staticmethod
    def _splice_minor_into_main_func(
        base_code: str, call_lines: List[str]
    ) -> str:
        """Insert ``call_lines`` (already-formatted Python statements) at
        the END of ``run_layout_engine()``'s body, i.e. immediately before
        the first top-level ``if __name__ == "__main__":`` line.

        Indentation is inferred from the most recent non-empty body line
        above that anchor. Falls back to 4 spaces, which is what every
        Stage 3 emission uses.
        """
        if not call_lines:
            return base_code
        lines = base_code.splitlines()
        anchor_re = re.compile(
            r'^\s*if\s+__name__\s*==\s*[\'"]__main__[\'"]\s*:\s*$'
        )
        anchor_idx = None
        for i, ln in enumerate(lines):
            if anchor_re.match(ln):
                anchor_idx = i
                break

        # Infer indent of run_layout_engine() body.
        indent = "    "
        if anchor_idx is not None:
            for j in range(anchor_idx - 1, -1, -1):
                ln = lines[j]
                if not ln.strip():
                    continue
                if ln.lstrip().startswith("def "):
                    continue
                m = re.match(r"^(\s+)", ln)
                if m:
                    indent = m.group(1)
                    break

        block: List[str] = [
            "",
            f"{indent}# --- Minor Objects (Phase B, programmatic) ---",
        ]
        block += [f"{indent}{c}" for c in call_lines]
        block.append("")

        if anchor_idx is None:
            # No __main__ anchor — splice at end of file but inside no
            # function. Best-effort: append at top level, callers should
            # not hit this path for Stage 3 output.
            return base_code.rstrip() + "\n" + "\n".join(block) + "\n"
        return "\n".join(lines[:anchor_idx] + block + lines[anchor_idx:])

    # MinorPlace_<obj_id>__<label>  rule: obj_id is in the form obj_001 or feat_001
    _MINOR_PLACE_NAME_RE = re.compile(
        r"^MinorPlace_((?:obj|feat)_\d+)__(.+)$"
    )

    def _parse_minor_placed_names(
        self,
        added_names: Set[str],
        important_minors: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Extract the list of placed minor obj_ids from the diff'd create_box names."""
        by_id = {m.get("id"): m for m in important_minors if m.get("id")}
        records: List[Dict[str, Any]] = []
        seen_ids: Set[str] = set()
        for name in sorted(added_names):
            m = self._MINOR_PLACE_NAME_RE.match(name)
            if not m:
                # Not following the convention; treat as Phase A wall-mounted output or LLM disobedience.
                # Don't error out here; absorb via _save_results' wall_object_names flow.
                continue
            obj_id = m.group(1)
            short_label = m.group(2)
            if obj_id in seen_ids:
                # Same obj_id appeared multiple times; record only once
                continue
            seen_ids.add(obj_id)
            sidecar = by_id.get(obj_id, {})
            records.append({
                "obj_id": obj_id,
                "block_name": name,
                "short_label": short_label,
                "label": sidecar.get("label"),
                "zone_id": sidecar.get("zone_id"),
                "placement_type": sidecar.get("placement_type"),
                "parent_id": sidecar.get("parent_id"),
                "parent_label": sidecar.get("parent_label"),
            })
        return records

    # ------------------------------------------------------------------
    # Wall-object extraction (diff Stage3 vs Stage4 code)
    # ------------------------------------------------------------------
    # These are the objects we want downstream stages (Stage 7 describe is
    # fine, Stage 8 geometry must SKIP them) to know were introduced by
    # Stage 4. Stage 8 uses this list to avoid replacing their bbox with
    # detailed composite geometry — wall decorations should stay as flat
    # boxes flush against the wall.

    _NAME_CALL_RE = re.compile(
        r"create_(?:box|cylinder)\(\s*(?:f|rf|fr)?[\"']([^\"']+)[\"']",
        re.IGNORECASE,
    )

    def _extract_object_names(self, code: str) -> Set[str]:
        """Extract literal object names from create_box/create_cylinder calls.

        f-string / format names like "Wall_{i}" cannot be reliably matched
        against Stage 7 describe output (which sees the unrolled name), so
        we drop any candidate that still contains a `{` placeholder. This
        leaves us with a conservative, high-precision set.
        """
        if not code:
            return set()
        names: Set[str] = set()
        for m in self._NAME_CALL_RE.finditer(code):
            raw = (m.group(1) or "").strip()
            if not raw or "{" in raw:
                continue
            names.add(raw)
        return names

    # ------------------------------------------------------------------
    # Wall-orientation post-process
    # ------------------------------------------------------------------
    # The Stage 4 prompt suggests rotation literals, but LLMs (especially
    # flash-thinking variants) regularly drop or shuffle them. Worse, the
    # interaction between ``rotation`` and ``obj.dimensions`` is subtle:
    #
    #   ``create_box`` calls ``primitive_cube_add(rotation=R)`` and THEN
    #   ``obj.dimensions = dim``. Blender's dimensions setter rescales
    #   ``obj.scale`` per WORLD axis to make the current world AABB match
    #   ``dim``. It does NOT undo ``R`` when figuring out which local axis
    #   feeds which world axis. As a result:
    #     - dim=(thin, wide, tall), R=(0,0,0)        -> world AABB (thin,wide,tall)  ✅
    #     - dim=(wide, thin, tall), R=(0,0,±π/2)     -> world AABB (thin,wide,tall)  ✅
    #     - dim=(thin, wide, tall), R=(0,0,±π/2)     -> world AABB (wide,thin,tall)  ❌
    #     - dim=(wide, thin, tall), R=(0,0,0)        -> world AABB (wide,thin,tall)  ❌
    #
    # i.e. dim and R must NOT both encode the swap, or they cancel into a
    # broken AABB. The simplest rule that admits zero ambiguity is:
    #
    #     rotation = (0, 0, 0) for ALL wall objects, and the SHORTEST axis
    #     in `dimensions` must align with the wall's normal:
    #         north / south wall (normal = Y)  ->  thin axis = Y  (dim = (W, T, H))
    #         east  / west  wall (normal = X)  ->  thin axis = X  (dim = (T, W, H))
    #
    # That makes the box's local frame coincident with the world frame, so
    # `dim` IS the world AABB by construction. This is what stage11_texture
    # later relies on to spawn wall-art planes correctly.
    #
    # The previous implementation used non-zero canonical rotations AND
    # reordered dim, which fell into row 3 of the table above and silently
    # rotated paintings 90° on east/west walls. Locking to (0, 0, 0)
    # eliminates the entire class of bugs.
    _WALL_ROTATION_LITERAL = {
        "north": "(0, 0, 0)",
        "south": "(0, 0, 0)",
        "east":  "(0, 0, 0)",
        "west":  "(0, 0, 0)",
    }

    @staticmethod
    def _eval_wall_namespace(code: str) -> Dict[str, Any]:
        """Build a sandbox namespace: math + top-level numeric constants
        defined in the base code (SCENE_W, SCENE_D, WALL_T, ...).

        Anything not understood stays unresolved → expressions referring to
        it will fail to eval and we'll skip the corresponding object.
        """
        ns: Dict[str, Any] = {"math": math}
        for cm in re.finditer(
            r"^\s*([A-Z_][A-Z0-9_]*)\s*=\s*([\-\d\.]+)\s*$",
            code,
            re.MULTILINE,
        ):
            try:
                ns[cm.group(1)] = float(cm.group(2))
            except ValueError:
                continue
        return ns

    @staticmethod
    def _eval_tuple(expr: Optional[str], ns: Dict[str, Any]) -> Optional[Tuple[float, float, float]]:
        if not expr:
            return None
        try:
            val = eval(expr, {"__builtins__": {}}, ns)
        except Exception:
            return None
        if not isinstance(val, (tuple, list)) or len(val) < 3:
            return None
        try:
            return (float(val[0]), float(val[1]), float(val[2]))
        except (TypeError, ValueError):
            return None

    def _enforce_wall_orientation(
        self, code: str, wall_names: List[str]
    ) -> str:
        """Rewrite rotation (and, if needed, dimension order) of newly added
        wall-mounted objects so that each is locked to its canonical wall.

        Side effects logged via ``self._log``. Returns the patched code; on
        any error, returns the input unchanged so this stage stays
        non-blocking.
        """
        if not code or not wall_names:
            return code

        ns = self._eval_wall_namespace(code)
        scene_w = float(ns.get("SCENE_W", 8.0))
        scene_d = float(ns.get("SCENE_D", 6.0))

        patched_code = code
        n_rot_fixed = 0
        n_dim_fixed = 0
        n_skipped_far = 0

        for name in wall_names:
            safe = re.escape(name)
            # Match a single-line call: create_box/cylinder("name", (loc),
            # (dim) [, kwargs])
            #
            # Inner tuples must NOT contain nested parens for this to work,
            # which holds for Stage 3/4 emitted code (numbers / SCENE_W/2
            # style expressions) but not for fancy ones like math.cos(...).
            # We accept that limitation: those are rare and we just skip
            # them rather than risk corrupting the call.
            pat = re.compile(
                rf'create_(box|cylinder)\s*\(\s*'
                rf'(["\']){safe}\2\s*,\s*'
                rf'(\([^()]*\))\s*,\s*'              # group 3: loc tuple
                rf'(\([^()]*\))'                     # group 4: dim tuple
                rf'((?:[^()]|\([^()]*\))*)'          # group 5: rest, allows
                                                     # one level of nested
                                                     # parens (e.g. rotation=
                                                     # (math.pi/2, 0, 0))
                rf'\)',
            )
            m = pat.search(patched_code)
            if not m:
                continue

            kind = m.group(1)
            quote = m.group(2)
            loc_str = m.group(3)
            dim_str = m.group(4)
            rest = m.group(5) or ""

            loc = self._eval_tuple(loc_str, ns)
            dim = self._eval_tuple(dim_str, ns)
            if loc is None or dim is None:
                continue

            lx, ly, lz = loc
            dists = {
                "north": abs(scene_d / 2.0 - ly),
                "south": abs(-scene_d / 2.0 - ly),
                "east":  abs(scene_w / 2.0 - lx),
                "west":  abs(-scene_w / 2.0 - lx),
            }
            wall_side = min(dists, key=dists.get)
            # Sanity gate: must be near the wall plane. 0.5 m is generous
            # enough to cover (depth/2 + WALL_T/2) for chunky pegboards
            # while still rejecting mid-room placements.
            if dists[wall_side] > 0.5:
                n_skipped_far += 1
                continue

            normal_axis = 0 if wall_side in ("east", "west") else 1
            abs_dim = (abs(dim[0]), abs(dim[1]), abs(dim[2]))
            thickness_axis = min(range(3), key=lambda i: abs_dim[i])

            new_dim = dim
            if thickness_axis != normal_axis:
                # Move the thinnest axis to the wall-normal slot. Keep the
                # other two slots in their original order so that the LLM's
                # intent for "width" vs "height" is preserved (paintings
                # can be wider than tall or taller than wide; we cannot
                # assume the largest axis is always Z).
                tmp = list(dim)
                tmp[thickness_axis], tmp[normal_axis] = (
                    tmp[normal_axis],
                    tmp[thickness_axis],
                )
                new_dim = tuple(tmp)

            new_dim_str = (
                f"({new_dim[0]:.4f}, {new_dim[1]:.4f}, {new_dim[2]:.4f})"
                if tuple(new_dim) != tuple(dim)
                else dim_str
            )
            new_rot_literal = self._WALL_ROTATION_LITERAL[wall_side]

            # Build a clean rest_str without any pre-existing rotation kwarg.
            rest_no_rot = re.sub(
                r"\s*,\s*rotation\s*=\s*\([^()]*\)",
                "",
                rest,
            ).strip()
            if rest_no_rot and not rest_no_rot.startswith(","):
                rest_no_rot = ", " + rest_no_rot

            new_call = (
                f"create_{kind}({quote}{name}{quote}, "
                f"{loc_str}, {new_dim_str}, "
                f"rotation={new_rot_literal}{rest_no_rot})"
            )

            patched_code = (
                patched_code[: m.start()]
                + new_call
                + patched_code[m.end():]
            )

            # Track what changed for the log line.
            old_rot_match = re.search(
                r"rotation\s*=\s*(\([^()]*\))", rest
            )
            old_rot_str = old_rot_match.group(1) if old_rot_match else "(none)"
            if old_rot_str != new_rot_literal:
                n_rot_fixed += 1
            if tuple(new_dim) != tuple(dim):
                n_dim_fixed += 1

        if n_rot_fixed or n_dim_fixed or n_skipped_far:
            self._log(
                f"Wall orientation post-process: "
                f"{n_rot_fixed} rotation(s) normalized, "
                f"{n_dim_fixed} dim order(s) corrected, "
                f"{n_skipped_far} skipped (too far from any wall).",
                "info",
            )
        return patched_code

    def _extract_wall_object_names(
        self, stage3_code: str, stage4_code: str
    ) -> List[str]:
        """Names present ONLY in Stage4 code = objects added by this stage.

        v3: exclude `MinorPlace_*` names (those are Phase B minor placeholders,
        tracked separately via `stage4_minor_placed`).
        """
        s3 = self._extract_object_names(stage3_code)
        s4 = self._extract_object_names(stage4_code)
        added = s4 - s3
        return sorted(n for n in added if not n.startswith("MinorPlace_"))

    def _save_results(self, code: str):
        """Save results (including Phase A wall + Phase B minor placeholder diffs)."""
        os.makedirs(self.output_dir, exist_ok=True)

        output_path = os.path.join(self.output_dir, "stage4_output.py")
        with open(output_path, "w") as f:
            f.write(code)
        self._log(f"Code saved: {output_path}")

        # Phase A diff: wall-mounted (excludes MinorPlace_*)
        wall_names = self._extract_wall_object_names(self.stage3_code, code)
        wall_json_path = os.path.join(self.output_dir, "wall_objects.json")
        with open(wall_json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "stage": "stage4",
                    "scope": "wall_mounted",
                    "wall_object_names": wall_names,
                    "count": len(wall_names),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        self._log(
            f"Phase A: {len(wall_names)} wall-mounted objects -> {wall_json_path}",
            "success",
        )

        # Phase B: minor placeholders (from self.stage4_minor_placed, populated by run())
        placed = self.stage4_minor_placed or []
        minor_json_path = os.path.join(self.output_dir, "stage4_minor_placed.json")
        placed_obj_ids = [r.get("obj_id") for r in placed if r.get("obj_id")]
        placed_block_names = [
            r.get("block_name") for r in placed if r.get("block_name")
        ]
        with open(minor_json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "stage": "stage4",
                    "scope": "minor_placeholders",
                    "count": len(placed),
                    "placed_obj_ids": placed_obj_ids,
                    "placed_block_names": placed_block_names,
                    "items": placed,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        self._log(
            f"Phase B: {len(placed)} minor placeholders -> {minor_json_path}",
            "success",
        )

        # Write to Memory. Keep stage="stage4" / type="result" compatible with the original;
        # add minor_placed_* fields to metadata for downstream stage7_small_objects to consume.
        self.memory.add(
            stage="stage4",
            type="result",
            content=code,
            metadata={
                "title": (
                    "Stage4 Code (Wall + Minor Placeholders)"
                    if placed
                    else "Stage4 Code (Wall-Mounted Only)"
                ),
                "summary": (
                    f"{code.count(chr(10)) + 1} lines, "
                    f"{len(wall_names)} wall, {len(placed)} minor"
                ),
                "output_file": output_path,
                "image_path": self.image_path,
                "scope": "wall_and_minor_placeholders",
                # Phase A: used by stage6_geometry to skip geometry detailing
                "wall_object_names": wall_names,
                "wall_objects_json": wall_json_path,
                # Phase B: used by stage7_small_objects to avoid duplicates
                "minor_placed_obj_ids": placed_obj_ids,
                "minor_placed_block_names": placed_block_names,
                "minor_placed_count": len(placed),
                "minor_placed_json": minor_json_path,
            },
            tags=["stage4", "blender_code", "wall_mounted",
                  "minor_placeholders" if placed else "wall_only"],
        )
        self._log("Stored in Memory", "success")
    
    def run(self) -> tuple:
        """Run Stage4: Phase A (wall-mounted) -> Phase B (minor placeholders).

        Returns:
            (success, code)
        """
        print("\n" + "=" * 60)
        print("Stage4 - Wall Decorations + Minor Placeholders")
        print("=" * 60)

        # 1. Load data (including minor sidecar)
        if not self._load_data():
            return False, None

        # 2. Phase A: wall-mounted (historical responsibility)
        code = self._generate_code()
        if not code:
            # A Phase A failure doesn't block Phase B (there may simply be no wall objects);
            # _generate_code falls back to stage3_code when there are no walls, not None,
            # so a None here means a real error and we abort
            return False, None

        # 2b. Hard-enforce canonical wall rotations on Phase A's additions.
        # The LLM gets the rotation right most of the time but quietly
        # mis-shuffles it just often enough that downstream stages keep
        # mis-rotating wall art / mirrors. Rather than relying on prompt
        # discipline alone, we re-derive each newly-added wall object's
        # closest wall from its location and overwrite rotation (+ fix dim
        # order if the smallest axis isn't on the wall normal).
        try:
            wall_names_phase_a = self._extract_wall_object_names(
                self.stage3_code, code
            )
            if wall_names_phase_a:
                code = self._enforce_wall_orientation(code, wall_names_phase_a)
        except Exception as exc:  # noqa: BLE001
            self._log(
                f"Wall-orientation post-process skipped: {exc}",
                "warning",
            )

        # 3. Phase B: coarse minor-object placeholders (new in v3)
        important_minors = self._filter_important_minors(self.minor_objects)
        if important_minors:
            self._log(
                f"Phase B candidates: {len(important_minors)} (filtered from {len(self.minor_objects)} "
                f"minor sidecar entries)", "info",
            )
            new_code, placed = self._generate_minor_placements(
                code, important_minors)
            code = new_code
            self.stage4_minor_placed = placed
        else:
            self._log("Phase B skipped: no minor passes coarse filter (sidecar empty or no salient items)", "info")
            self.stage4_minor_placed = []

        # 4. Save results
        self._save_results(code)

        print("\n" + "=" * 60)
        print("Stage4 done!")
        print("=" * 60)

        return True, code


def show_memory_status():
    """Show Memory status"""
    memory = Memory(workspace_dir=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    print("=" * 60)
    print("Memory status")
    print("=" * 60)

    for stage in ["stage1", "stage2", "stage3", "stage4"]:
        entry = memory.get_latest(stage=stage, type="result")
        if entry:
            title = entry.metadata.get("title", "untitled")
            from datetime import datetime
            time_str = datetime.fromtimestamp(entry.timestamp).strftime("%m-%d %H:%M")
            print(f"[OK] {stage}: {title} ({time_str})")
        else:
            print(f"[X] {stage}: no data")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Stage4 - Wall-mounted decorations only "
                    "(paintings / mirrors / clocks / sconces / shelves)"
    )
    parser.add_argument("--image", "-i", help="Reference image path")
    parser.add_argument("--output-dir", "-o", default="./output", help="Output directory")
    parser.add_argument("--status", "-s", action="store_true", help="Show Memory status")
    
    args = parser.parse_args()
    
    if args.status:
        show_memory_status()
        return 0
    
    runner = Stage4Runner(
        image_path=args.image,
        output_dir=args.output_dir
    )
    
    success, code = runner.run()
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
