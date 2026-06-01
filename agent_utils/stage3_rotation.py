"""
Stage 3 Rotation Correction
============================
After Stage 3 generates the scene layout, this stage iteratively corrects
object rotations by rendering a top-down view, comparing with the reference
image, and asking the LLM to fix only rotation parameters.
"""
import os
import sys
import json
import base64
import subprocess
from typing import Optional

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, "stage3"))

from stage3.core import LLMClient, PromptManager, extract_python_from_response
from memory import Memory
from langchain_core.messages import HumanMessage, SystemMessage


class Stage3RotationRunner:
    """Iteratively corrects object rotations to match the reference image."""

    def __init__(
        self,
        image_path: str,
        output_dir: str,
        blender_path: str = "/Applications/Blender.app/Contents/MacOS/Blender",
        max_iterations: int = 3,
        target_score: float = 0.95,
        verbose: bool = True,
        memory_file: str = "agent_memory.jsonl",
        model: str = None,
        base_url: str = None,
        api_key: str = None,
    ):
        self.image_path = image_path
        self.output_dir = output_dir
        self.blender_path = blender_path
        self.max_iterations = max_iterations
        self.target_score = target_score
        self.verbose = verbose

        self.memory = Memory(workspace_dir=current_dir, memory_file=memory_file)
        self.prompts = PromptManager()
        self.llm = LLMClient(model=model, base_url=base_url, api_key=api_key)

        self.current_code: Optional[str] = None
        # Note (v2, 2026-05-02): the old `_history` anti-oscillation list
        # has been removed; the pending-state-machine in run() now owns
        # per-object retry tracking via the `pending` list.

    # ------------------------------------------------------------------
    def _log(self, msg: str, level: str = "info"):
        if self.verbose:
            prefix = {
                "info": "ℹ️", "success": "✅", "warning": "⚠️",
                "error": "❌", "step": "📋",
            }.get(level, "")
            print(f"{prefix} [RotFix] {msg}")

    def _encode_image(self, path: str):
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        ext = os.path.splitext(path)[1].lower()
        mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}.get(ext, "image/png")
        return b64, mime

    # ------------------------------------------------------------------
    def _load_stage3_code(self) -> bool:
        entry = self.memory.get_latest(stage="stage3", type="result")
        if not entry:
            self._log("No Stage3 result in Memory!", "error")
            return False
        self.current_code = entry.content
        self._log(f"Stage3 code loaded ({self.current_code.count(chr(10))+1} lines)", "success")
        return True

    # ------------------------------------------------------------------
    def _render_topdown(self, iteration: int) -> Optional[str]:
        """Render the current code as a top-down image. Returns image path or None."""
        os.makedirs(self.output_dir, exist_ok=True)
        code_file = os.path.join(self.output_dir, "_rot_temp_code.py")
        with open(code_file, "w") as f:
            f.write(self.current_code)

        render_image = os.path.join(self.output_dir, f"render_rotation_{iteration}.png")

        render_script = self._build_render_script(code_file, render_image)
        render_file = os.path.join(self.output_dir, "_rot_temp_render.py")
        with open(render_file, "w") as f:
            f.write(render_script)

        try:
            result = subprocess.run(
                [self.blender_path, "--background", "--factory-startup", "--python", render_file],
                capture_output=True, text=True, timeout=120,
            )
            if os.path.exists(render_image):
                self._log(f"Render succeeded: render_rotation_{iteration}.png", "success")
                return render_image
            else:
                stderr_errors = [
                    l.strip() for l in (result.stderr or "").split("\n")
                    if any(k in l for k in ("Error", "Traceback", "SyntaxError"))
                ]
                self._log(f"Render failed: {'; '.join(stderr_errors[:3]) or 'unknown'}", "error")
                return None
        except subprocess.TimeoutExpired:
            self._log("Render timeout (120s)", "error")
            return None
        except Exception as e:
            self._log(f"Render exception: {e}", "error")
            return None

    # ------------------------------------------------------------------
    def _extract_object_rotations(self) -> str:
        """Extract object names, type, dimensions, and rotation values from code."""
        import re
        lines = []
        seen = set()
        for m in re.finditer(
            r'create_(box|cylinder)\(\s*"([^"]+)"(.+)',
            self.current_code
        ):
            obj_type = m.group(1)
            name = m.group(2)
            rest_of_line = m.group(3)
            dims_m = re.search(r'\)\s*,\s*\(([^)]+)\)', rest_of_line)
            dims = dims_m.group(1) if dims_m else "?"
            rot_m = re.search(r'rotation\s*=\s*\(([^)]*)\)', rest_of_line)
            rot = rot_m.group(1) if rot_m else "0, 0, 0"
            lines.append(f"  {name} [{obj_type}]: dims=({dims}), rotation=({rot})")
            seen.add(name)
        return "\n".join(lines) if lines else "(no objects found)"

    def _analyze_rotations(self, rendered_path: str, iteration: int = 0) -> dict:
        """Compare rendered vs reference, return analysis JSON."""
        analyze_prompt = self.prompts.get("Stage3_rotation_analyze") or ""

        orig_b64, orig_mime = self._encode_image(self.image_path)
        rend_b64, rend_mime = self._encode_image(rendered_path)

        obj_table = self._extract_object_rotations()

        # v2 (2026-05-02): removed the old anti-oscillation `_history`
        # block — iter 1 is the ONLY full-analyze pass, so no history exists
        # yet. Subsequent iterations go through `_verify_pending` instead.
        messages = [
            SystemMessage(content=analyze_prompt),
            HumanMessage(content=[
                {"type": "text", "text": "Reference image (ground truth):"},
                {"type": "image_url", "image_url": {"url": f"data:{orig_mime};base64,{orig_b64}"}},
                {"type": "text", "text": "Rendered top-down view (current scene):"},
                {"type": "image_url", "image_url": {"url": f"data:{rend_mime};base64,{rend_b64}"}},
                {"type": "text", "text": f"""Analyze rotation differences. Here are the current objects and their rotations in the code:

{obj_table}

Check each object's orientation against the reference image. Pay special attention to:
- Elongated objects (treadmill, bench, barbell, rack) — is the long axis aligned correctly?
- Furniture with a clear front/back — is it facing the right direction?
- Objects that appear as a dot from top-down might be oriented wrong (e.g., a barbell standing vertical instead of horizontal).

For each issue, provide the EXACT target_rotation_z value (one of: 0, math.pi/2, math.pi, -math.pi/2).

Output JSON only."""},
            ]),
        ]

        try:
            raw = self.llm.invoke(messages)
            suffix = f"_iter{iteration}" if iteration > 0 else ""
            with open(os.path.join(self.output_dir, f"_rot_analysis{suffix}.txt"), "w") as f:
                f.write(raw)
            return self._parse_analysis(raw)
        except Exception as e:
            self._log(f"Analysis failed: {e}", "error")
            return {"score": 0, "rotation_issues": [], "summary": str(e)}

    def _verify_pending(
        self,
        rendered_path: str,
        pending: list,
        iteration: int,
    ) -> dict:
        """Re-render and ask the LLM ONLY about objects already in `pending`.

        The LLM must NOT invent new issues — it can only mark each pending
        object as `ok` (rotation now matches reference) or `still_wrong`
        (give a new target_rotation_z to try).

        Args:
            rendered_path: The just-rendered top-down PNG.
            pending: List of dicts {object_name, target_rotation_z, ...}
                — the items that were flagged in iteration 1 and not yet OK.
            iteration: Current iteration index (>= 2).

        Returns:
            {
                "rotation_issues": [<subset of pending that is still_wrong,
                                     each with possibly updated target>],
                "summary": "...",
                "score": float (0..1, based on how many pending items are OK),
            }
        """
        analyze_prompt = self.prompts.get("Stage3_rotation_analyze") or ""
        # Augment with verify-only constraint, layered on top of the same
        # base instructions so JSON schema is unchanged.
        verify_constraint = (
            "\n\n## VERIFY-ONLY MODE — STRICT\n"
            "You are NOT allowed to look for new rotation problems in this "
            "iteration. You may ONLY evaluate the objects listed under "
            "'## Pending objects to verify'. For every other object in the "
            "scene, treat it as CORRECT and ignore it.\n"
            "For each pending object, decide:\n"
            "  - status='ok'           → it now matches the reference; drop it\n"
            "  - status='still_wrong'  → still misaligned; give a new "
            "target_rotation_z (one of: 0, math.pi/2, math.pi, -math.pi/2)\n"
            "Output JSON with `rotation_issues` containing ONLY the "
            "still_wrong items (i.e. an empty array means 'all pending fixed')."
        )

        orig_b64, orig_mime = self._encode_image(self.image_path)
        rend_b64, rend_mime = self._encode_image(rendered_path)
        cur_rotations = self._extract_object_rotations()

        pending_lines = []
        for p in pending:
            nm = p.get("object_name", "?")
            t = p.get("target_rotation_z", "?")
            attempts = p.get("attempts", 1)
            prev = p.get("tried_targets") or []
            extra = (
                f"  (attempts so far: {attempts}; "
                f"tried targets: {prev})" if attempts > 1 else ""
            )
            pending_lines.append(f"  - {nm}: tried target={t}{extra}")
        pending_text = "\n".join(pending_lines) if pending_lines else "  (empty)"

        messages = [
            SystemMessage(content=analyze_prompt + verify_constraint),
            HumanMessage(content=[
                {"type": "text", "text": "Reference image (ground truth):"},
                {"type": "image_url",
                 "image_url": {"url": f"data:{orig_mime};base64,{orig_b64}"}},
                {"type": "text", "text": "Rendered top-down view (current scene after the previous fix):"},
                {"type": "image_url",
                 "image_url": {"url": f"data:{rend_mime};base64,{rend_b64}"}},
                {"type": "text", "text": f"""## Pending objects to verify
Only evaluate these objects. Do NOT add or evaluate any other object.

{pending_text}

## Current rotations in code (full table for context — but only judge the pending ones)

{cur_rotations}

For each pending object, output one entry in `rotation_issues` ONLY IF it
is still wrong. If it now looks correct in the rendered image, OMIT it from
the output. An empty `rotation_issues` array means "all fixed — stop the loop".

Output JSON only."""},
            ]),
        ]

        try:
            raw = self.llm.invoke(messages)
            with open(
                os.path.join(self.output_dir, f"_rot_verify_iter{iteration}.txt"),
                "w",
            ) as f:
                f.write(raw)
            parsed = self._parse_analysis(raw)
            # Filter: silently drop any "issue" whose object_name is not in
            # the pending set (defends against LLM disobeying verify-only).
            allowed = {p.get("object_name") for p in pending}
            kept = [
                iss for iss in parsed.get("rotation_issues", [])
                if iss.get("object_name") in allowed
            ]
            dropped = len(parsed.get("rotation_issues", [])) - len(kept)
            if dropped > 0:
                self._log(
                    f"Verify iter{iteration}: dropped {dropped} non-pending "
                    f"issue(s) from LLM output (verify-only enforcement)",
                    "warning",
                )
            parsed["rotation_issues"] = kept
            # Compute a score from "how many pending items are now OK"
            n_pending = max(1, len(pending))
            n_fixed = n_pending - len(kept)
            parsed["score"] = round(n_fixed / n_pending, 3)
            return parsed
        except Exception as e:
            self._log(f"Validation failed: {e}", "error")
            # Fail-safe: treat all pending as still_wrong with their last target
            return {
                "score": 0.0,
                "rotation_issues": list(pending),
                "summary": f"verify exception: {e}",
            }

    def _parse_analysis(self, raw: str) -> dict:
        """Extract JSON from LLM analysis response."""
        import re
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw, re.DOTALL)
        text = match.group(1) if match else raw
        # Try to find JSON object
        brace_start = text.find("{")
        if brace_start < 0:
            return {"score": 0.5, "rotation_issues": [], "summary": "Could not parse"}
        depth = 0
        for i in range(brace_start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[brace_start:i+1])
                    except json.JSONDecodeError:
                        break
        return {"score": 0.5, "rotation_issues": [], "summary": "JSON parse error"}

    # ------------------------------------------------------------------
    _RZ_MAP = {
        "0": "(0, 0, 0)",
        "math.pi/2": "(0, 0, math.pi/2)",
        "math.pi": "(0, 0, math.pi)",
        "-math.pi/2": "(0, 0, -math.pi/2)",
        "1.5708": "(0, 0, math.pi/2)",
        "3.1416": "(0, 0, math.pi)",
        "-1.5708": "(0, 0, -math.pi/2)",
    }

    def _rule_based_fix(self, analysis: dict) -> Optional[str]:
        """Try to fix rotations by direct regex substitution (no LLM needed)."""
        import re
        code = self.current_code
        issues = analysis.get("rotation_issues", [])
        applied = 0

        for issue in issues:
            name = issue.get("object_name", "")
            target = str(issue.get("target_rotation_z", "")).strip()
            if not name or not target:
                continue

            target_tuple = None
            for key, val in self._RZ_MAP.items():
                if target == key or target.replace(" ", "") == key.replace(" ", ""):
                    target_tuple = val
                    break
            if not target_tuple:
                target_tuple = f"(0, 0, {target})"

            pattern = re.compile(
                r'(create_(?:box|cylinder)\(\s*"' + re.escape(name) + r'"'
                r'.*?rotation\s*=\s*)\([^)]*\)',
                re.DOTALL
            )
            new_code, n = pattern.subn(r'\g<1>' + target_tuple, code)
            if n > 0:
                code = new_code
                applied += 1
                self._log(f"  Rule-fix: {name} → rotation={target_tuple}", "info")

        if applied > 0:
            self._log(f"Rule-based fix applied {applied}/{len(issues)} rotations", "success")
            return code
        return None

    def _fix_rotations(self, analysis: dict) -> Optional[str]:
        """Fix rotations: try rule-based first, fall back to LLM."""
        rule_result = self._rule_based_fix(analysis)
        if rule_result:
            return rule_result

        self._log("Rule-based fix failed, falling back to LLM fix", "warning")
        fix_prompt = self.prompts.get("Stage3_rotation_fix") or ""

        issues_text = json.dumps(analysis.get("rotation_issues", []), indent=2, ensure_ascii=False)

        user_text = f"""## Rotation Issues to Fix

```json
{issues_text}
```

## Current Blender Code

```python
{self.current_code}
```

For each issue, set the rotation= parameter to use the target_rotation_z value. Output the COMPLETE corrected code.
"""
        messages = [
            SystemMessage(content=fix_prompt),
            HumanMessage(content=user_text),
        ]

        try:
            raw = self.llm.invoke(messages)
            code = extract_python_from_response(raw)
            if code:
                try:
                    compile(code, "<rotation_fix>", "exec")
                    return code
                except SyntaxError as e:
                    self._log(f"Fixed code has syntax error: {e}", "warning")
                    return code
            self._log("Failed to extract fixed code", "error")
            return None
        except Exception as e:
            self._log(f"Fix failed: {e}", "error")
            return None

    # ------------------------------------------------------------------
    def _save_result(self, code: str):
        """Save corrected code to memory and file."""
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, "stage3_rotation_output.py")
        with open(path, "w") as f:
            f.write(code)
        self._log(f"Saved: {path}", "success")

        self.memory.add(
            stage="stage3",
            type="result",
            content=code,
            metadata={
                "title": "Stage3 Code (Rotation Corrected)",
                "summary": f"{code.count(chr(10))+1} lines, rotation-fixed",
                "output_file": path,
                "image_path": self.image_path,
            },
            tags=["stage3", "blender_code", "rotation_fixed"],
        )
        self._log("Memory stage3 entry overwritten", "success")

    # ------------------------------------------------------------------
    def run(self) -> tuple[bool, float]:
        """
        Run the rotation correction loop.

        Pending-state-machine design (v2, 2026-05-02):
        - Iteration 1 ONLY: full analyze → builds the pending list (the
          authoritative set of "rotations that need fixing"). Then fix them.
        - Iteration 2..N: verify-only — re-render, ask the LLM whether each
          PENDING item is now correct in the rendered image. Items judged OK
          drop out and are NEVER re-evaluated. Items still wrong get a new
          target_rotation_z and are fixed again.
        - Items NOT in pending are considered correct from iteration 1
          onwards and will never be touched, so the loop cannot regress
          them by mistake.

        Returns:
            (success, final_score)
        """
        print("\n" + "=" * 60)
        print("🔄 Stage3 Rotation Correction (verify-only after iter 1)")
        print(f"   max_iterations={self.max_iterations}, target={self.target_score:.0%}")
        print("=" * 60)

        if not self._load_stage3_code():
            return False, 0.0
        if not self.image_path or not os.path.exists(self.image_path):
            self._log(f"Reference image does not exist: {self.image_path}", "error")
            return False, 0.0

        # ---------------- Iteration 1: render → full analyze → fix ----------------
        print(f"\n{'─'*40}")
        print(f"📋 Rotation iter 1/{self.max_iterations} (full analyze)")
        print(f"{'─'*40}")

        rendered = self._render_topdown(1)
        if not rendered:
            self._log("Initial render failed, aborting rotation fix", "error")
            self._save_result(self.current_code)
            return False, 0.0

        analysis = self._analyze_rotations(rendered, iteration=1)
        initial_issues = analysis.get("rotation_issues", []) or []
        self._log(
            f"Iter1 full-analyze: score={float(analysis.get('score', 0)):.0%}, "
            f"issues={len(initial_issues)} | {analysis.get('summary', '')}",
            "info",
        )

        if not initial_issues:
            self._log("No rotation issues, saving directly", "success")
            self._save_result(self.current_code)
            print("\n" + "=" * 60)
            print(f"✅ Rotation correction done (score=100%, 0 issues)")
            print("=" * 60)
            return True, 1.0

        # Build authoritative pending list. Each item carries metadata so we
        # can detect oscillation and cap retries per object.
        pending: list = []
        for iss in initial_issues:
            nm = iss.get("object_name")
            if not nm:
                continue
            pending.append({
                "object_name": nm,
                "target_rotation_z": iss.get("target_rotation_z"),
                "tried_targets": [iss.get("target_rotation_z")],
                "attempts": 1,
                "issue_text": iss.get("issue", iss.get("summary", "")),
            })
        n_initial = len(pending)
        self._log(f"Pending list locked: {n_initial} object(s) → {[p['object_name'] for p in pending]}", "info")

        # First fix
        fix_pkg = {"rotation_issues": pending}
        fixed_code = self._fix_rotations(fix_pkg)
        if fixed_code:
            self.current_code = fixed_code
            iter_path = os.path.join(self.output_dir, "stage3_rotation_iter1.py")
            with open(iter_path, "w") as f:
                f.write(fixed_code)
            self._log("Intermediate result: stage3_rotation_iter1.py", "info")
        else:
            self._log("Iter1 fix failed, keeping original code and continuing validation", "warning")

        # ---------------- Iterations 2..N: verify-only loop ----------------
        # Per-object retry cap (avoid infinite loops if some object can't be fixed)
        MAX_ATTEMPTS_PER_OBJ = 3
        score = 0.0

        for i in range(2, self.max_iterations + 1):
            if not pending:
                break
            print(f"\n{'─'*40}")
            print(f"📋 Rotation iter {i}/{self.max_iterations} (verify-only, {len(pending)} pending)")
            print(f"{'─'*40}")

            rendered = self._render_topdown(i)
            if not rendered:
                self._log("Render failed, skipping this iteration", "warning")
                continue

            verify = self._verify_pending(rendered, pending, iteration=i)
            still_wrong = verify.get("rotation_issues", []) or []
            score = float(verify.get("score", 0.0))
            self._log(
                f"Iter{i} verify: {len(pending) - len(still_wrong)}/{len(pending)} "
                f"pending now OK (score={score:.0%})",
                "info",
            )

            if not still_wrong:
                self._log("All pending objects fixed, ending early", "success")
                pending = []
                break

            # Update pending: keep only still-wrong items, refresh target,
            # bump attempts; cap by MAX_ATTEMPTS_PER_OBJ.
            still_map = {iss.get("object_name"): iss for iss in still_wrong}
            new_pending = []
            for p in pending:
                nm = p["object_name"]
                if nm not in still_map:
                    self._log(f"  ✓ {nm} → confirmed OK in iter{i}", "success")
                    continue
                iss = still_map[nm]
                new_target = iss.get("target_rotation_z", p["target_rotation_z"])
                if new_target not in p["tried_targets"]:
                    p["tried_targets"].append(new_target)
                p["target_rotation_z"] = new_target
                p["attempts"] += 1
                p["issue_text"] = iss.get("issue", iss.get("summary", p.get("issue_text", "")))
                if p["attempts"] > MAX_ATTEMPTS_PER_OBJ:
                    self._log(
                        f"  ⏭ {nm} → exceeded {MAX_ATTEMPTS_PER_OBJ} attempts, "
                        f"giving up (last target={new_target})",
                        "warning",
                    )
                    continue
                new_pending.append(p)
            pending = new_pending

            if not pending:
                self._log("All remaining pending items have exhausted retries, ending", "warning")
                break

            # Re-fix only the still-pending items (with possibly new targets)
            fix_pkg = {"rotation_issues": pending}
            fixed_code = self._fix_rotations(fix_pkg)
            if fixed_code:
                self.current_code = fixed_code
                iter_path = os.path.join(self.output_dir, f"stage3_rotation_iter{i}.py")
                with open(iter_path, "w") as f:
                    f.write(fixed_code)
                self._log(f"Intermediate result: stage3_rotation_iter{i}.py", "info")
            else:
                self._log(f"Iter{i} fix failed, keeping current code", "warning")

        # Save final result
        self._save_result(self.current_code)

        # Final score = fraction of initial pending objects that ended OK.
        final_score = round((n_initial - len(pending)) / max(1, n_initial), 3)
        ok_icon = "✅" if final_score >= self.target_score else "⚠️"
        print("\n" + "=" * 60)
        print(
            f"{ok_icon} Rotation correction done: "
            f"{n_initial - len(pending)}/{n_initial} fixed "
            f"(score={final_score:.0%}, {len(pending)} unresolved)"
        )
        if pending:
            print(f"   Unresolved: {[p['object_name'] for p in pending]}")
        print("=" * 60)
        return True, final_score

    # ------------------------------------------------------------------
    def _build_render_script(self, code_file: str, render_image: str) -> str:
        """Build the Blender render script (same as Stage 3)."""
        # Use forward slashes to avoid Windows backslash escape issues
        # (e.g. \r → carriage-return) in the generated Python script.
        code_file = code_file.replace("\\", "/")
        render_image = render_image.replace("\\", "/")
        return f'''
import bpy
import sys
import math
import mathutils

try:
    bpy.context.scene.render.engine = 'BLENDER_EEVEE_NEXT'
except:
    try:
        bpy.context.scene.render.engine = 'BLENDER_EEVEE'
    except:
        bpy.context.scene.render.engine = 'CYCLES'

code_text = open("{code_file}").read()

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
_SKIP_NAMES = set()
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

# --- Labels ---
_text_size = ortho_scale * 0.022
_label_objs = []
_label_mat = bpy.data.materials.new(name="_LabelMat")
_label_mat.use_nodes = True
_lb = _label_mat.node_tree.nodes.get('Principled BSDF')
if _lb:
    _lb.inputs['Base Color'].default_value = (0.05, 0.02, 0.02, 1)

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
print(f"Labels: {{len(_label_objs)}}")

# --- Render ---
if hasattr(bpy.context.scene, 'eevee'):
    bpy.context.scene.eevee.taa_render_samples = 64
    if hasattr(bpy.context.scene.eevee, 'use_shadows'):
        bpy.context.scene.eevee.use_shadows = False

bpy.context.scene.render.resolution_x = 1024
bpy.context.scene.render.resolution_y = 1024
bpy.context.scene.render.filepath = "{render_image}"
bpy.context.scene.render.image_settings.file_format = 'PNG'
bpy.context.scene.render.film_transparent = False
bpy.context.scene.view_layers[0].use_pass_combined = True

bpy.ops.render.render(write_still=True)

for _t in _label_objs:
    bpy.data.objects.remove(_t, do_unlink=True)
print("Render done!")
'''
