"""Stage3 Pipeline - orchestrates the generate / render / analyze / fix flow"""
import os
import json
import subprocess
import tempfile
from typing import Optional, Dict, Tuple
from dataclasses import dataclass

from code_gen_agent import CodeGenAgent
from analyze_agent import AnalyzeAgent
from fix_agent import FixAgent
from code_patcher import CodePatcher
from validator import CodeValidator
from core import LLMClient, PromptManager


@dataclass
class PipelineConfig:
    """Pipeline configuration"""
    max_iterations: int = 3
    target_score: float = 0.85
    blender_path: str = "/Applications/Blender.app/Contents/MacOS/Blender"
    output_dir: str = "."
    verbose: bool = True


class Stage3Pipeline:
    """
    Stage3 Pipeline - the full generate -> render -> analyze -> fix flow

    Usage:
        pipeline = Stage3Pipeline(config)
        success, code = pipeline.run(stage1_json, stage2_json, original_image)
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()

        # Shared LLM client
        self.llm = LLMClient()
        self.prompts = PromptManager()
        
        # Agents
        self.code_gen = CodeGenAgent(self.llm, self.prompts, self.config.verbose)
        self.analyzer = AnalyzeAgent(self.llm, self.prompts, self.config.verbose)
        self.fixer = FixAgent(self.llm, self.prompts, self.config.verbose)
        
        # Tools
        self.patcher = CodePatcher(self.config.verbose)
        self.validator = CodeValidator(self.config.verbose)
        
        # State
        self.current_code = None
        self.stage2_json = None
        self.iteration = 0
    
    def _log(self, msg: str, level: str = "info"):
        if self.config.verbose:
            prefix = {"info": "ℹ️", "success": "✅", "warning": "⚠️", "error": "❌", "step": "📋"}.get(level, "")
            print(f"{prefix} {msg}")
    
    def run(
        self,
        stage1_json: Dict,
        stage2_json: Dict,
        original_image: str,
        initial_code: Optional[str] = None
    ) -> Tuple[bool, str, Dict]:
        """
        Run the full pipeline.

        Args:
            stage1_json: Stage 1 output.
            stage2_json: Stage 2 output.
            original_image: Path to the original reference image.
            initial_code: Optional initial code (skips the generation step).

        Returns:
            (success, final_code, metadata)
        """
        print("\n" + "=" * 60)
        print("🚀 Stage3 Pipeline Started")
        print("=" * 60)
        
        self.stage2_json = stage2_json
        
        # Step 1: Generate or use initial code
        if initial_code:
            self._log("Using provided initial code", "step")
            self.current_code = initial_code
        else:
            self._log("Step 1: Code Generation", "step")
            success, code, meta = self.code_gen.run(stage1_json, stage2_json, original_image)
            if not success:
                return False, "", {"error": "Code generation failed"}
            self.current_code = code
        
        # Validate initial code
        validation = self.validator.validate(self.current_code)
        if not validation.is_valid:
            self._log(f"Initial code has errors: {validation.errors}", "error")
            return False, self.current_code, {"error": validation.errors}
        
        # Save initial code
        self._save_code("stage3_initial.py")
        
        # Step 2-N: Iterate (render → analyze → fix)
        metadata = {"iterations": []}
        
        for self.iteration in range(self.config.max_iterations):
            iter_num = self.iteration + 1
            print(f"\n{'─' * 40}")
            self._log(f"Iteration {iter_num}/{self.config.max_iterations}", "step")
            print(f"{'─' * 40}")
            
            # 2.1: Render
            rendered_image = self._render_scene()
            if not rendered_image:
                self._log("Render failed, skipping iteration", "warning")
                continue
            
            # 2.2: Analyze
            self._log("Analyzing differences...", "step")
            success, analysis = self.analyzer.run(
                original_image, rendered_image, self.current_code, self.stage2_json
            )
            
            if not success:
                self._log("Analysis failed, skipping iteration", "warning")
                continue
            
            # Check score
            score = self.analyzer.get_score(analysis)
            corrections = self.analyzer.get_corrections(analysis)
            
            metadata["iterations"].append({
                "iteration": iter_num,
                "score": score,
                "corrections_count": len(corrections)
            })
            
            self._log(f"Match score: {score:.0%}", "info")
            
            # Check convergence
            if score >= self.config.target_score:
                self._log(f"Target score reached! ({score:.0%} >= {self.config.target_score:.0%})", "success")
                break
            
            if not corrections:
                self._log("No corrections suggested, stopping", "info")
                break
            
            # 2.3: Fix
            self._log(f"Applying {len(corrections)} corrections...", "step")
            
            # Try rule-based patching first
            original_code = self.current_code
            patched_code, applied_count = self.patcher.apply_corrections(
                self.current_code, corrections
            )
            
            # Validate patched code
            patch_valid, _ = self.validator.quick_check(patched_code)
            
            if patch_valid and applied_count > 0:
                self.current_code = patched_code
                self._log(f"Applied {applied_count} patches", "success")
            else:
                # Fall back to LLM fixing
                self._log("Patches failed, using LLM fixer...", "warning")
                success, fixed_code = self.fixer.run(self.current_code, analysis)
                
                if success:
                    fix_valid, _ = self.validator.quick_check(fixed_code)
                    if fix_valid:
                        self.current_code = fixed_code
                    else:
                        self._log("LLM fix produced invalid code, keeping original", "error")
                        self.current_code = original_code
            
            # Save iteration result
            self._save_code(f"stage3_iter{iter_num}.py")
        
        # Final save
        self._save_code("stage3_final.py")
        
        # Final analysis
        final_score = metadata["iterations"][-1]["score"] if metadata["iterations"] else 0
        success = final_score >= 0.7
        
        metadata["final_score"] = final_score
        metadata["success"] = success
        metadata["total_iterations"] = len(metadata["iterations"])
        
        print("\n" + "=" * 60)
        self._log(f"Pipeline Complete! Final score: {final_score:.0%}", "success" if success else "warning")
        print("=" * 60)
        
        return success, self.current_code, metadata
    
    def _render_scene(self) -> Optional[str]:
        """Render the scene to an image"""
        self._log("Rendering scene...", "step")

        # Check Blender path
        if not os.path.exists(self.config.blender_path):
            self._log(f"Blender not found: {self.config.blender_path}", "error")
            return None

        # Create temporary render script
        render_script = self._create_render_script()
        # Use absolute path
        output_image = os.path.abspath(os.path.join(self.config.output_dir, "render_topdown.png"))

        try:
            # Save current code to a temp file (using absolute path)
            code_file = os.path.abspath(os.path.join(self.config.output_dir, "_temp_code.py"))
            with open(code_file, "w") as f:
                f.write(self.current_code)

            # Save render script (using absolute path)
            render_file = os.path.abspath(os.path.join(self.config.output_dir, "_temp_render.py"))
            with open(render_file, "w") as f:
                # Use forward slashes to avoid Windows backslash escape issues
                # (e.g. \r → carriage-return) in the generated Python script.
                f.write(render_script.format(
                    code_file=code_file.replace("\\", "/"),
                    output_image=output_image.replace("\\", "/")
                ))

            # Run Blender
            result = subprocess.run(
                [self.config.blender_path, "--background", "--factory-startup", "--python", render_file],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode != 0:
                self._log(f"Blender error: {result.stderr[:500]}", "error")
                return None
            
            if os.path.exists(output_image):
                self._log(f"Rendered: {output_image}", "success")
                return output_image
            else:
                self._log("Render output not found", "error")
                return None
                
        except subprocess.TimeoutExpired:
            self._log("Render timeout", "error")
            return None
        except Exception as e:
            self._log(f"Render error: {e}", "error")
            return None
    
    def _create_render_script(self) -> str:
        """Create the Blender render script"""
        return '''
import bpy
import sys
import os

# Execute the layout code
exec(open("{code_file}").read())

# Configure top-down camera
scene = bpy.context.scene

# Remove existing cameras
for obj in bpy.data.objects:
    if obj.type == 'CAMERA':
        bpy.data.objects.remove(obj)

# Create top-down camera
bpy.ops.object.camera_add(location=(0, 0, 15))
camera = bpy.context.active_object
camera.rotation_euler = (0, 0, 0)
camera.data.type = 'ORTHO'
camera.data.ortho_scale = 15

scene.camera = camera

# Lighting
bpy.ops.object.light_add(type='SUN', location=(5, 5, 10))
sun = bpy.context.active_object
sun.data.energy = 3

# Render settings
scene.render.resolution_x = 1024
scene.render.resolution_y = 1024
scene.render.film_transparent = True
scene.render.filepath = "{output_image}"
scene.render.image_settings.file_format = 'PNG'

# Render
bpy.ops.render.render(write_still=True)

print("Render complete!")
'''

    def _save_code(self, filename: str):
        """Save code to a file"""
        path = os.path.join(self.config.output_dir, filename)
        with open(path, "w") as f:
            f.write(self.current_code)
        self._log(f"Saved: {path}")


def run_pipeline(
    stage1_json: Dict,
    stage2_json: Dict,
    original_image: str,
    output_dir: str = ".",
    blender_path: str = "/Applications/Blender.app/Contents/MacOS/Blender",
    max_iterations: int = 3
) -> Tuple[bool, str]:
    """
    Convenience function - run the Stage3 Pipeline.

    Example:
        success, code = run_pipeline(
            stage1_json=json.load(open("stage1.json")),
            stage2_json=json.load(open("stage2.json")),
            original_image="input.png"
        )
    """
    config = PipelineConfig(
        max_iterations=max_iterations,
        blender_path=blender_path,
        output_dir=output_dir
    )
    
    pipeline = Stage3Pipeline(config)
    success, code, metadata = pipeline.run(stage1_json, stage2_json, original_image)
    
    return success, code

