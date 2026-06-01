"""Stage3 runner script - supports iterative refinement"""
import os
import sys
import json
import base64
import argparse
import subprocess

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)
sys.path.insert(0, parent_dir)

from core import LLMClient, PromptManager, extract_python_from_response, extract_json_from_response
from memory import Memory
from langchain_core.messages import HumanMessage, SystemMessage


# ---------------------------------------------------------------------------
# Architectural name filter
# ---------------------------------------------------------------------------
# Walls / floor / ceiling / windows are NOT emitted into the Object Layout
# Table fed to the LLM (see `_is_furniture` in the render script below). If the
# LLM nevertheless reports an overlap whose one side is a wall / floor / window
# etc., it is a hallucination and must be dropped BEFORE it reaches the fixer.
_ARCH_NAME_TOKENS = (
    "wall", "walls",
    "floor", "floors", "ceiling",
    "window", "windows", "door", "doorway",
    "north wall", "south wall", "east wall", "west wall",
    "n wall", "s wall", "e wall", "w wall",
    "boundary", "perimeter", "room edge", "room bound", "room boundary",
)

def _looks_like_architecture(name) -> bool:
    """Return True if `name` refers to a wall / floor / window / doorway / etc.

    Very defensive: LLMs spell wall objects in many ways ("West Wall",
    "wall_west", "S_Wall", "South-Wall", "Boundary", etc.). We normalise and
    then check token membership.
    """
    if not name or not isinstance(name, str):
        return False
    n = name.strip().lower()
    if not n:
        return False
    # Exact tokens / multi-word forms
    for tok in _ARCH_NAME_TOKENS:
        if tok == n:
            return True
        # "west wall", "west-wall", "west_wall"
        if tok in n.replace("_", " ").replace("-", " "):
            return True
    # Common prefixes used in the scene code itself
    for pref in ("wall_", "s_wall", "n_wall", "e_wall", "w_wall",
                 "s_window", "e_glass"):
        if n.startswith(pref):
            return True
    return False


class Stage3Runner:
    """Stage3 runner - supports generation and iterative refinement"""

    def __init__(
        self,
        image_path: str = None,
        output_dir: str = "./output",
        blender_path: str = "/Applications/Blender.app/Contents/MacOS/Blender",
        max_iterations: int = 3,
        target_score: float = 0.7,
        verbose: bool = True,
        model: str = None,
        base_url: str = None,
        api_key: str = None,
        memory_file: str = "agent_memory.jsonl",
        render_labels: bool = True
    ):
        self.image_path = image_path
        self.output_dir = output_dir
        self.blender_path = blender_path
        self.max_iterations = max_iterations
        self.target_score = target_score
        self.verbose = verbose
        self.render_labels = render_labels

        # Initialize
        self.memory = Memory(workspace_dir=parent_dir, memory_file=memory_file)
        self.prompts = PromptManager()
        self.llm = LLMClient(
            model=model,
            base_url=base_url,
            api_key=api_key
        )

        # Data
        self.stage1_json = None
        self.stage2_json = None
        self.current_code = None

    def _log(self, msg: str, level: str = "info"):
        if self.verbose:
            prefix = {"info": "[i]", "success": "[OK]", "warning": "[!]", "error": "[X]", "step": "[>]"}.get(level, "")
            print(f"{prefix} {msg}")

    def _apply_scene_addendum(self, base_prompt: str) -> str:
        """Append scene-specific addendum (e.g., lab) to the Stage3 system prompt.

        Reads the scene_type record produced by `scene_classifier` (run as
        Stage 0 in unified_pipeline) from Memory. When the scene is a lab
        with confidence >= 0.5, appends `Stage3_lab_addendum`. Falls back
        silently to the unmodified base prompt on any failure so the
        production path is never broken.
        """
        try:
            from scene_classifier import read_scene_type  # type: ignore
            info = read_scene_type(self.memory)
        except Exception as exc:
            self._log(f"Stage3: cannot read scene_type ({exc}); using base prompt", "warning")
            return base_prompt

        scene_type = info.get("scene_type", "other")
        confidence = float(info.get("confidence", 0.0) or 0.0)

        if scene_type == "lab" and confidence >= 0.5:
            try:
                addendum = self.prompts.get("Stage3_lab_addendum")
            except Exception as exc:
                self._log(f"Stage3: failed to load lab addendum ({exc}); using base prompt", "warning")
                return base_prompt
            subtype = info.get("lab_subtype") or "general"
            self._log(
                f"Stage3: routing to lab prompt variant (subtype={subtype}, "
                f"confidence={confidence:.2f})",
                "info",
            )
            return base_prompt.rstrip() + "\n\n" + addendum.lstrip()

        self._log(f"Stage3: using base prompt (scene_type={scene_type})", "info")
        return base_prompt

    def _encode_image(self, path: str) -> tuple:
        """Encode image"""
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        ext = os.path.splitext(path)[1].lower()
        mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}.get(ext, "image/png")
        return b64, mime
    
    def _extract_layout_from_code(self) -> list:
        """Load layout data extracted by the Blender render script.
        
        The render script introspects the live Blender scene after code execution,
        which is far more reliable than regex-parsing source code (handles variables,
        expressions, and evaluated transforms correctly).
        """
        layout_file = os.path.join(self.output_dir, "_layout.json")
        if os.path.exists(layout_file):
            with open(layout_file, "r") as f:
                return json.load(f)
        self._log("Layout JSON not found, falling back to empty list", "warning")
        return []
    
    def _load_data(self) -> bool:
        """Load data from Memory"""
        self._log("Fetching data from Memory...", "step")

        stage1_entry = self.memory.get_latest(stage="stage1", type="result")
        if not stage1_entry:
            self._log("No Stage1 result in Memory!", "error")
            return False
        self.stage1_json = stage1_entry.content
        self._log(f"Stage1: OK", "success")

        stage2_entry = self.memory.get_latest(stage="stage2", type="result")
        if not stage2_entry:
            self._log("No Stage2 result in Memory!", "error")
            return False
        self.stage2_json = stage2_entry.content
        self._log(f"Stage2: OK", "success")

        # Image
        if not self.image_path:
            self.image_path = stage1_entry.metadata.get("image_path")

        if self.image_path and os.path.exists(self.image_path):
            self._log(f"Image: {self.image_path}", "success")
        else:
            self._log(f"No image", "warning")
            self.image_path = None

        return True

    def _generate_code(self) -> bool:
        """Generate the initial code"""
        self._log("Generating code...", "step")

        system_prompt = self.prompts.get("Stage3_task")
        system_prompt = self._apply_scene_addendum(system_prompt)

        user_text = f"""Please generate Blender Python code based on the following data:

## Stage 1 (Architecture)
```json
{json.dumps(self.stage1_json, ensure_ascii=False, indent=2)}
```

## Stage 2 (Scene Graph)
```json
{json.dumps(self.stage2_json, ensure_ascii=False, indent=2)}
```

Please generate complete, executable Blender Python code.
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
            self.current_code = extract_python_from_response(response)

            if not self.current_code:
                # Persist raw response for postmortem ("what did the LLM return that we failed to recognize as code?")
                try:
                    os.makedirs(self.output_dir, exist_ok=True)
                    dump_path = os.path.join(self.output_dir, "_failed_response.txt")
                    with open(dump_path, "w", encoding="utf-8") as f:
                        f.write(response if isinstance(response, str) else str(response))
                    raw_len = len(response) if isinstance(response, str) else len(str(response))
                    preview = (response if isinstance(response, str) else str(response))[:200].replace("\n", " ")
                    self._log(
                        f"Failed to extract code (raw len={raw_len}, preview={preview!r}); "
                        f"raw saved to {dump_path}",
                        "error",
                    )
                except Exception:
                    self._log("Failed to extract code", "error")
                return False

            # Validate syntax
            try:
                compile(self.current_code, '<string>', 'exec')
                self._log(f"Code generated ({self.current_code.count(chr(10)) + 1} lines)", "success")
            except SyntaxError as e:
                self._log(f"Syntax error (line {e.lineno}): {e.msg}", "warning")
                self._fix_code_errors(f"SyntaxError at line {e.lineno}: {e.msg}")

            return True

        except Exception as e:
            self._log(f"Generation failed: {e}", "error")
            return False

    def _render_scene(self) -> str | dict:
        """Render the scene; on success returns the render path, on failure returns {'error': str} or None"""
        self._log("Rendering scene...", "step")

        if not os.path.exists(self.blender_path):
            self._log(f"Blender not found: {self.blender_path}", "error")
            return None

        # Save code
        code_file = os.path.join(self.output_dir, "_temp_code.py")
        with open(code_file, "w") as f:
            f.write(self.current_code)

        # Render script
        # Use forward slashes in paths embedded in the f-string template to avoid
        # Windows backslash escape issues (e.g. \r → carriage-return) when the
        # generated script is executed by Blender's Python interpreter.
        render_image = os.path.join(self.output_dir, "render_topdown.png")
        layout_json = os.path.join(self.output_dir, "_layout.json")
        _code_file_safe = code_file.replace("\\", "/")
        _render_image_safe = render_image.replace("\\", "/")
        _layout_json_safe = layout_json.replace("\\", "/")
        render_script = f'''
import bpy
import sys
import math
import mathutils
import json as _json

try:
    bpy.context.scene.render.engine = 'BLENDER_EEVEE_NEXT'
except:
    try:
        bpy.context.scene.render.engine = 'BLENDER_EEVEE'
    except:
        bpy.context.scene.render.engine = 'CYCLES'

code_text = open("{_code_file_safe}").read()

# Inject missing helper stubs so exec() doesn't fail on undefined functions
if 'def create_collection' not in code_text:
    def create_collection(name):
        coll = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(coll)
        return coll

exec(code_text)

import re
main_func_match = re.search(r'def (run_layout_engine|main|create_scene|build_scene)\\s*\\(', code_text)
if main_func_match:
    func_name = main_func_match.group(1)
    exec(f"{{func_name}}()")

print(f"Objects: {{len(bpy.data.objects)}}")

_ARCH_EXACT = {{'floor'}}
_ARCH_PREFIX = ('wall_', 's_wall', 'e_wall', 's_window', 'e_glass')
_SKIP_NAMES = {{'\\u7acb\\u65b9\\u4f53', '\\u5706\\u9525', '\\u5706\\u67f1', '\\u7403\\u4f53'}}
def _is_furniture(name):
    nl = name.lower()
    if nl in _ARCH_EXACT or name in _SKIP_NAMES:
        return False
    if any(nl.startswith(p) for p in _ARCH_PREFIX):
        return False
    if nl.startswith('cone') or '.' in name:
        return False
    try:
        name.encode('ascii')
    except UnicodeEncodeError:
        return False
    return True

# --- Extract layout from live scene (reliable, handles vars/expressions) ---
_layout = []
for _obj in bpy.data.objects:
    if _obj.type != 'MESH':
        continue
    if not _is_furniture(_obj.name):
        continue
    if max(_obj.dimensions) < 0.15:
        continue
    _layout.append({{
        "name": _obj.name,
        "x": round(_obj.location.x, 2),
        "y": round(_obj.location.y, 2),
        "z": round(_obj.location.z, 2),
        "width": round(_obj.dimensions.x, 2),
        "depth": round(_obj.dimensions.y, 2),
        "height": round(_obj.dimensions.z, 2),
    }})
with open("{_layout_json_safe}", "w") as _f:
    _json.dump(_layout, _f, indent=2)
print(f"Layout JSON: {{len(_layout)}} furniture objects")

# --- Camera ---
for obj in list(bpy.data.objects):
    if obj.type == 'CAMERA':
        bpy.data.objects.remove(obj)

min_x, max_x = float('inf'), float('-inf')
min_y, max_y = float('inf'), float('-inf')
for obj in bpy.data.objects:
    if obj.type == 'MESH':
        for v in obj.bound_box:
            try:
                world_v = obj.matrix_world @ mathutils.Vector(v)
            except:
                world_v = v
            wx = world_v.x if hasattr(world_v, 'x') else world_v[0]
            wy = world_v.y if hasattr(world_v, 'y') else world_v[1]
            min_x, max_x = min(min_x, wx), max(max_x, wx)
            min_y, max_y = min(min_y, wy), max(max_y, wy)

if min_x != float('inf'):
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    scene_width = max_x - min_x
    scene_height = max_y - min_y
    ortho_scale = max(scene_width, scene_height) * 1.2
else:
    center_x, center_y = 0, 0
    ortho_scale = 12

bpy.ops.object.camera_add(location=(center_x, center_y, 15))
cam = bpy.context.active_object
cam.rotation_euler = (0, 0, 0)
cam.data.type = 'ORTHO'
cam.data.ortho_scale = ortho_scale
bpy.context.scene.camera = cam

# --- Lighting ---
for obj in list(bpy.data.objects):
    if obj.type == 'LIGHT':
        bpy.data.objects.remove(obj)

bpy.ops.object.light_add(type='SUN', location=(center_x, center_y, 10))
sun = bpy.context.active_object
sun.data.energy = 2.5
sun.rotation_euler = (0, 0, 0)
sun.data.use_shadow = False

bpy.ops.object.light_add(type='AREA', location=(center_x, center_y, 8))
area = bpy.context.active_object
area.data.energy = 50
area.data.size = ortho_scale
area.data.use_shadow = False

if bpy.context.scene.world is None:
    bpy.context.scene.world = bpy.data.worlds.new("World")
bpy.context.scene.world.use_nodes = True
world_nodes = bpy.context.scene.world.node_tree.nodes
bg_node = world_nodes.get('Background')
if bg_node:
    bg_node.inputs['Color'].default_value = (1.0, 1.0, 1.0, 1.0)
    bg_node.inputs['Strength'].default_value = 0.5

# --- Labels (furniture only, skip architecture) ---
_RENDER_LABELS = {self.render_labels}
_text_size = ortho_scale * 0.022
_label_objs = []
_label_mat = bpy.data.materials.new(name="_LabelMat")
_label_mat.use_nodes = True
_lb = _label_mat.node_tree.nodes.get('Principled BSDF')
if _lb:
    _lb.inputs['Base Color'].default_value = (0.05, 0.02, 0.02, 1)

if _RENDER_LABELS:
    for _obj in list(bpy.data.objects):
        if _obj.type != 'MESH':
            continue
        if not _is_furniture(_obj.name):
            continue
        if max(_obj.dimensions) < 0.15:
            continue
        _tz = _obj.location.z + _obj.dimensions.z / 2 + 0.15
        bpy.ops.object.text_add(location=(_obj.location.x, _obj.location.y, _tz))
        _t = bpy.context.active_object
        _t.data.body = _obj.name
        _t.data.size = _text_size
        _t.data.align_x = 'CENTER'
        _t.data.align_y = 'CENTER'
        _t.name = f"_lbl_{{_obj.name}}"
        _t.data.materials.append(_label_mat)
        _label_objs.append(_t)
print(f"Labels: {{len(_label_objs)}} (furniture only)")

# --- Render settings ---
if hasattr(bpy.context.scene, 'eevee'):
    bpy.context.scene.eevee.taa_render_samples = 64
    if hasattr(bpy.context.scene.eevee, 'use_gtao'):
        bpy.context.scene.eevee.use_gtao = False
    if hasattr(bpy.context.scene.eevee, 'use_soft_shadows'):
        bpy.context.scene.eevee.use_soft_shadows = False
    if hasattr(bpy.context.scene.eevee, 'use_shadows'):
        bpy.context.scene.eevee.use_shadows = False

bpy.context.scene.render.resolution_x = 1024
bpy.context.scene.render.resolution_y = 1024
bpy.context.scene.render.filepath = "{_render_image_safe}"
bpy.context.scene.render.image_settings.file_format = 'PNG'
bpy.context.scene.render.film_transparent = False
bpy.context.scene.view_layers[0].use_pass_combined = True

bpy.ops.render.render(write_still=True)

# --- Cleanup: remove all label objects ---
for _t in _label_objs:
    bpy.data.objects.remove(_t, do_unlink=True)
print("Labels removed, render done!")
'''
        
        render_file = os.path.join(self.output_dir, "_temp_render.py")
        with open(render_file, "w") as f:
            f.write(render_script)
        
        try:
            result = subprocess.run(
                [self.blender_path, "--background", "--factory-startup", "--python", render_file],
                capture_output=True, text=True, timeout=120
            )
            
            if os.path.exists(render_image):
                self._log(f"Render succeeded: {render_image}", "success")
                return render_image
            else:
                self._log(f"Render output not found", "error")
                error_lines = []
                if result.stderr:
                    for line in result.stderr.split('\n'):
                        if 'Error' in line or 'TypeError' in line or 'SyntaxError' in line:
                            self._log(f"  {line.strip()}", "error")
                            error_lines.append(line.strip())
                if result.stdout:
                    for line in result.stdout.split('\n'):
                        if 'Error' in line or 'Traceback' in line:
                            error_lines.append(line.strip())
                error_msg = '\n'.join(error_lines) if error_lines else "Render failed with unknown error"
                return {"error": error_msg}

        except subprocess.TimeoutExpired:
            self._log("Render timeout", "error")
            return {"error": "Render timed out after 120 seconds"}
        except Exception as e:
            self._log(f"Render error: {e}", "error")
            return {"error": str(e)}
    
    def _analyze_and_fix(self, rendered_image: str) -> float:
        """Analyze layout with structured data + visual comparison, then fix."""
        self._log("Analyzing layout (structured + visual)...", "step")
        
        analyze_prompt = self.prompts.get("Stage3_analyze_system")
        
        layout_table = self._extract_layout_from_code()
        layout_str = json.dumps(layout_table, indent=2, ensure_ascii=False)
        
        context_parts = [f"## Object Layout Table (current code)\n```json\n{layout_str}\n```"]
        if self.stage1_json:
            s1_str = json.dumps(self.stage1_json, indent=2, ensure_ascii=False)
            context_parts.append(f"## Stage1 Room Analysis\n```json\n{s1_str}\n```")
        if self.stage2_json:
            s2_str = json.dumps(self.stage2_json, indent=2, ensure_ascii=False)
            context_parts.append(f"## Stage2 Spatial Relationships\n```json\n{s2_str}\n```")
        context_text = "\n\n".join(context_parts)
        
        orig_b64, orig_mime = self._encode_image(self.image_path)
        rend_b64, rend_mime = self._encode_image(rendered_image)
        
        messages = [
            SystemMessage(content=analyze_prompt),
            HumanMessage(content=[
                {"type": "text", "text": "Reference image (target layout):"},
                {"type": "image_url", "image_url": {"url": f"data:{orig_mime};base64,{orig_b64}"}},
                {"type": "text", "text": "Rendered image (current, with name labels):"},
                {"type": "image_url", "image_url": {"url": f"data:{rend_mime};base64,{rend_b64}"}},
                {"type": "text", "text": context_text}
            ])
        ]
        
        try:
            response = self.llm.invoke(messages)
            json_str = extract_json_from_response(response)
            analysis = json.loads(json_str)
            
            score = float(analysis.get("overall_score", 0.5))
            score = max(0.0, min(1.0, score))
            
            # --- Sanitize overlapping_pairs: drop any pair that references a
            # wall / floor / window / doorway (LLM hallucinated, see prompt).
            raw_pairs = analysis.get("overlapping_pairs", []) or []
            clean_pairs = []
            dropped_arch_pairs = []
            for _op in raw_pairs:
                if not isinstance(_op, dict):
                    continue
                _a = _op.get("object_a", "")
                _b = _op.get("object_b", "")
                if _looks_like_architecture(_a) or _looks_like_architecture(_b):
                    dropped_arch_pairs.append((_a, _b))
                    continue
                clean_pairs.append(_op)
            analysis["overlapping_pairs"] = clean_pairs
            if dropped_arch_pairs:
                self._log(
                    f"Dropped {len(dropped_arch_pairs)} bogus furniture-vs-wall overlap(s) "
                    f"reported by the analyzer (walls are not in the layout table):",
                    "warning",
                )
                for _a, _b in dropped_arch_pairs[:8]:
                    self._log(f"    · {_a} vs {_b}  (ignored)", "info")
            
            issues = []
            for obj in analysis.get("missing_objects", []):
                desc = obj.get("description", obj.get("type", "?")) if isinstance(obj, dict) else str(obj)
                issues.append(f"Missing: {desc}")
            for pi in analysis.get("position_issues", []):
                if isinstance(pi, dict):
                    issues.append(f"Position: {pi.get('object_id', '?')} -> {pi.get('expected', '?')}")
            for ri in analysis.get("relationship_issues", []):
                if isinstance(ri, dict):
                    issues.append(f"Relationship: {ri.get('issue', '?')}")
            for op in analysis.get("overlapping_pairs", []):
                if isinstance(op, dict) and op.get("severity", "major") != "minor":
                    issues.append(f"Overlap: {op.get('object_a', '?')} vs {op.get('object_b', '?')}")
            for oob in analysis.get("out_of_bounds", []) or []:
                if isinstance(oob, dict):
                    issues.append(
                        f"OutOfBounds: {oob.get('object_id', '?')} - {oob.get('description', '')}"
                    )
            for db in analysis.get("doorway_blocking", []):
                if isinstance(db, dict):
                    issues.append(f"Blocking: {db.get('object_id', '?')}")
            for si in analysis.get("size_issues", []):
                if isinstance(si, dict):
                    issues.append(f"Size: {si.get('object_id', '?')} - {si.get('issue', '?')}")
            
            self._log(f"Score: {score:.0%}", "success" if score >= self.target_score else "warning")
            for issue in issues[:8]:
                self._log(f"  - {issue}", "info")
            
            if analysis.get("objects_to_fix"):
                self._log("Generating fix...", "step")
                self._fix_code(analysis)
            
            return score
            
        except Exception as e:
            self._log(f"Analysis failed: {e}", "error")
            return 0
    
    def _extract_object_lines(self, object_ids: list) -> str:
        """Extract code lines related to specific objects for targeted fixing."""
        import re
        lines = self.current_code.split('\n')
        relevant = []
        for i, line in enumerate(lines):
            for obj_id in object_ids:
                obj_lower = obj_id.lower().replace(' ', '_').replace('-', '_')
                line_lower = line.lower()
                if obj_lower in line_lower or obj_id.lower() in line_lower:
                    start = max(0, i - 1)
                    end = min(len(lines), i + 2)
                    for j in range(start, end):
                        relevant.append(f"L{j+1}: {lines[j]}")
                    break
        return '\n'.join(relevant) if relevant else "(no matching lines found)"

    def _fix_code(self, analysis: dict):
        """Fix code based on structured analysis results."""
        fix_prompt = self.prompts.get("Stage3_fix_system")
        if not fix_prompt:
            fix_prompt = "You are a Blender code fixer. Fix ONLY the specific issues. Output COMPLETE code."
        
        issues_summary = []
        object_ids_to_find = []
        
        for obj in analysis.get("missing_objects", []):
            if isinstance(obj, dict):
                issues_summary.append(f"ADD missing: {obj.get('type', 'unknown')} - {obj.get('description', '')}")
            else:
                issues_summary.append(f"ADD missing: {obj}")
        
        for pi in analysis.get("position_issues", []):
            if isinstance(pi, dict):
                obj_id = pi.get("object_id", "?")
                sx, sy = pi.get("suggested_x"), pi.get("suggested_y")
                coord = f" -> target location=({sx}, {sy})" if sx is not None and sy is not None else ""
                issues_summary.append(f"MOVE {obj_id}: currently {pi.get('current', '?')}, should be {pi.get('expected', '?')}{coord}")
                object_ids_to_find.append(obj_id)
        
        for ri in analysis.get("relationship_issues", []):
            if isinstance(ri, dict):
                objs = ri.get("objects", [])
                issues_summary.append(f"RELATIONSHIP: {ri.get('issue', '?')} ({', '.join(objs)})")
                object_ids_to_find.extend(objs)
        
        for pair in analysis.get("overlapping_pairs", []):
            if isinstance(pair, dict):
                a, b = pair.get('object_a', '?'), pair.get('object_b', '?')
                # Defensive: even if the upstream sanitizer missed something,
                # never ask the fixer to resolve a furniture-vs-wall overlap.
                if _looks_like_architecture(a) or _looks_like_architecture(b):
                    continue
                issues_summary.append(f"OVERLAP: {a} vs {b} - {pair.get('description', '')}")
                object_ids_to_find.extend([a, b])
        
        for oob in analysis.get("out_of_bounds", []) or []:
            if isinstance(oob, dict):
                obj_id = oob.get("object_id", "?")
                issues_summary.append(
                    f"OUT_OF_BOUNDS: {obj_id} - {oob.get('description', '')} "
                    f"(pull this object back inside the room outline, do NOT invent a wall object)"
                )
                object_ids_to_find.append(obj_id)
        
        for block in analysis.get("doorway_blocking", []):
            if isinstance(block, dict):
                obj_id = block.get('object_id', '?')
                issues_summary.append(f"BLOCKING: {obj_id} - {block.get('description', '')}")
                object_ids_to_find.append(obj_id)
        
        for si in analysis.get("size_issues", []):
            if isinstance(si, dict):
                obj_id = si.get("object_id", "?")
                issues_summary.append(f"RESIZE {obj_id}: {si.get('issue', '?')}")
                object_ids_to_find.append(obj_id)
        
        to_fix = analysis.get("objects_to_fix", [])
        if to_fix:
            issues_summary.append("\n## Fix targets (prioritized):")
            for obj in to_fix:
                if isinstance(obj, dict):
                    obj_id = obj.get("object_id", "?")
                    action = obj.get("action", "move")
                    sx, sy = obj.get("suggested_x"), obj.get("suggested_y")
                    coord = f" -> location=({sx}, {sy})" if sx is not None and sy is not None else ""
                    issues_summary.append(f"  {action.upper()} {obj_id}: {obj.get('reason', '')}{coord}")
                    object_ids_to_find.append(obj_id)
        
        relevant_lines = self._extract_object_lines(object_ids_to_find) if object_ids_to_find else ""
        
        user_text = f"""## Issues to fix:
{chr(10).join(f"- {issue}" for issue in issues_summary)}

## Relevant code lines:
{relevant_lines}

## Full current code:
```python
{self.current_code}
```

Fix ONLY the listed issues. For MOVE: modify location=(x, y, z). For RESIZE: modify dimensions=(w, d, h). Output COMPLETE code."""
        
        messages = [
            SystemMessage(content=fix_prompt),
            HumanMessage(content=user_text)
        ]
        
        try:
            response = self.llm.invoke(messages)
            fixed_code = extract_python_from_response(response)
            
            if fixed_code:
                try:
                    compile(fixed_code, '<string>', 'exec')
                    self.current_code = fixed_code
                    self._log("Code fixed", "success")
                except SyntaxError as e:
                    self._log(f"Fixed code has syntax error, keeping original: {e.msg}", "warning")
        except Exception as e:
            self._log(f"Fix failed: {e}", "error")

    def _fix_code_errors(self, error_msg: str, max_retries: int = 2) -> bool:
        """Use LLM to fix syntax/runtime errors in the generated code.
        
        Returns True if code was successfully fixed.
        """
        self._log("Code Critic: fixing code errors...", "step")
        
        for attempt in range(max_retries):
            syntax_err = None
            try:
                compile(self.current_code, '<string>', 'exec')
            except SyntaxError as e:
                syntax_err = e

            if syntax_err is None and attempt > 0:
                self._log("Code Critic: syntax is now valid", "success")
                return True
            
            if syntax_err:
                error_detail = f"SyntaxError at line {syntax_err.lineno}: {syntax_err.msg}"
                if syntax_err.text:
                    error_detail += f"\n  Line content: {syntax_err.text.strip()}"
            else:
                error_detail = error_msg

            self._log(f"Fix attempt {attempt + 1}/{max_retries}: {error_detail.splitlines()[0]}", "step")

            numbered_code = '\n'.join(
                f"{i+1:4d} | {line}" for i, line in enumerate(self.current_code.split('\n'))
            )

            user_text = f"""The following Blender Python script has errors that prevent it from running.

## Error message:
```
{error_detail}

{error_msg}
```

## Full code (with line numbers):
```python
{numbered_code}
```

Fix ALL errors in the code. Common issues:
- Unclosed parentheses/brackets/strings
- Incorrect indentation
- Missing imports
- Wrong Blender API usage

Output the COMPLETE fixed Python script. Do NOT omit any part of the code."""

            messages = [
                SystemMessage(content="You are a Blender Python code debugger. Fix the errors in the code. Output ONLY the complete fixed Python script."),
                HumanMessage(content=user_text)
            ]

            try:
                response = self.llm.invoke(messages)
                fixed_code = extract_python_from_response(response)

                if not fixed_code:
                    self._log(f"Attempt {attempt + 1}: failed to extract code from response", "warning")
                    continue

                try:
                    compile(fixed_code, '<string>', 'exec')
                    self.current_code = fixed_code
                    self._log(f"Code Critic: fixed successfully (attempt {attempt + 1})", "success")
                    return True
                except SyntaxError as e:
                    error_msg = f"SyntaxError at line {e.lineno}: {e.msg}"
                    self._log(f"Attempt {attempt + 1}: fix still has syntax error: {e.msg}", "warning")

            except Exception as e:
                self._log(f"Attempt {attempt + 1}: LLM call failed: {e}", "error")

        self._log("Code Critic: failed to fix after all attempts", "error")
        return False

    def _save_results(self, score: float):
        """Save the results"""
        os.makedirs(self.output_dir, exist_ok=True)

        output_path = os.path.join(self.output_dir, "stage3_output.py")
        with open(output_path, "w") as f:
            f.write(self.current_code)
        self._log(f"Code saved: {output_path}")

        # Store in Memory
        self.memory.add(
            stage="stage3",
            type="result",
            content=self.current_code,
            metadata={
                "title": f"Stage3 Code (Score: {score:.0%})",
                "summary": f"{self.current_code.count(chr(10)) + 1} lines, score {score:.0%}",
                "score": score,
                "output_file": output_path,
                "image_path": self.image_path
            },
            tags=["stage3", "blender_code", f"score:{int(score*100)}"]
        )
        self._log("Stored in Memory", "success")

    def run(self, iterate: bool = True) -> tuple:
        """
        Run Stage3.

        Args:
            iterate: whether to enable iterative refinement.

        Returns:
            (success, score)
        """
        print("\n" + "=" * 60)
        print(f"Stage3 {'iterative mode' if iterate else 'single-shot generation'}")
        print(f"   target score: {self.target_score:.0%}, max iterations: {self.max_iterations}")
        print("=" * 60)

        # 1. Load data
        if not self._load_data():
            return False, 0

        # 2. Generate initial code
        if not self._generate_code():
            return False, 0

        # Save initial code
        os.makedirs(self.output_dir, exist_ok=True)
        with open(os.path.join(self.output_dir, "stage3_initial.py"), "w") as f:
            f.write(self.current_code)

        score = 0

        # 3. Iterative refinement
        if iterate and self.image_path:
            for i in range(self.max_iterations):
                print(f"\n{'-' * 40}")
                self._log(f"Iteration {i + 1}/{self.max_iterations}", "step")
                print(f"{'-' * 40}")

                # Render
                rendered = self._render_scene()
                if isinstance(rendered, dict) and "error" in rendered:
                    self._log("Render failed, launching Code Critic fix...", "warning")
                    if self._fix_code_errors(rendered["error"]):
                        with open(os.path.join(self.output_dir, f"stage3_iter{i+1}.py"), "w") as f:
                            f.write(self.current_code)
                    continue
                if not rendered:
                    self._log("Render failed (no error info); skipping this iteration", "warning")
                    continue

                # Analyze and fix
                score = self._analyze_and_fix(rendered)

                # Save this iteration
                with open(os.path.join(self.output_dir, f"stage3_iter{i+1}.py"), "w") as f:
                    f.write(self.current_code)

                # Check whether target met
                if score >= self.target_score:
                    self._log(f"Reached target score {self.target_score:.0%}!", "success")
                    break
        else:
            self._log("Skipping iteration (no image or disabled)", "info")

        # 4. Save final result
        self._save_results(score)

        print("\n" + "=" * 60)
        status = "Success" if score >= self.target_score else "Done"
        print(f"{status}! Final score: {score:.0%}")
        print("=" * 60)

        return score >= self.target_score, score


def show_memory_status():
    """Show Memory status"""
    memory = Memory(workspace_dir=parent_dir)

    print("=" * 60)
    print("Memory status")
    print("=" * 60)

    for stage in ["stage1", "stage2", "stage3"]:
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
    parser = argparse.ArgumentParser(description="Stage3 code generation")
    parser.add_argument("--image", "-i", help="Reference image path")
    parser.add_argument("--output-dir", "-o", default="./output", help="Output directory")
    parser.add_argument("--blender", "-b",
                        default="/Applications/Blender.app/Contents/MacOS/Blender",
                        help="Blender path")
    parser.add_argument("--max-iter", "-n", type=int, default=3, help="Max iterations")
    parser.add_argument("--target", "-t", type=float, default=0.8, help="Target score (0-1)")
    parser.add_argument("--no-iterate", action="store_true", help="Disable iteration, generate once")
    parser.add_argument("--status", "-s", action="store_true", help="Show Memory status")

    # Model configuration arguments
    parser.add_argument("--model", "-m", type=str, default=None,
                        help="LLM model name (e.g. gpt-5.1-codex-max)")
    parser.add_argument("--base-url", type=str, default=None,
                        help="API base URL (e.g. https://api.openai.com/v1)")
    parser.add_argument("--api-key", type=str, default=None,
                        help="API Key (or set OPENAI_API_KEY env var)")
    
    args = parser.parse_args()
    
    if args.status:
        show_memory_status()
        return 0
    
    runner = Stage3Runner(
        image_path=args.image,
        output_dir=args.output_dir,
        blender_path=args.blender,
        max_iterations=args.max_iter,
        target_score=args.target,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key
    )
    
    success, score = runner.run(iterate=not args.no_iterate)
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
