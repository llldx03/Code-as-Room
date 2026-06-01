"""
==============================================================================
Unified Pipeline - main entry point for all stages
==============================================================================

Each run creates an isolated directory next to the input image by default:
<image_dir>/run_<timestamp>_<image>/
with its own agent_memory.jsonl. Previous runs are never overwritten.

Usage:
    # Run the full pipeline and create an isolated run directory automatically
    python run_pipeline.py --image input.png


    # List all runs
    python run_pipeline.py --list-runs


    # Clear memory
    python run_pipeline.py --clear-memory


    python /Users/yangyixuan/Code-as-Room_github/run_pipeline.py --config /Users/yangyixuan/Code-as-Room_github/example/pipeline_config.example.json

"""

import os
import sys
import json
import math
import re
import argparse
import importlib.util
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

# ============================================================================
# Path setup
# ============================================================================
SCRIPT_DIR = Path(__file__).parent.absolute()
if (SCRIPT_DIR / "agent_utils").is_dir():
    PROJECT_ROOT = SCRIPT_DIR
    AGENT_UTILS_DIR = PROJECT_ROOT / "agent_utils"
else:
    AGENT_UTILS_DIR = SCRIPT_DIR
    PROJECT_ROOT = AGENT_UTILS_DIR.parent

# Backward-compatible name used throughout this file and by Memory. It should
# always point to the package directory that contains stage modules, regardless
# of whether this entry file lives at the repository root or in agent_utils/.
CURRENT_DIR = AGENT_UTILS_DIR
sys.path.insert(0, str(AGENT_UTILS_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================================
# Dynamically import stage runners
# ============================================================================
def _import_runner(stage_path: str, class_name: str):
    """Dynamically import a stage runner class."""
    spec = importlib.util.spec_from_file_location("runner", stage_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)

# Import each stage
try:
    Stage1Runner = _import_runner(CURRENT_DIR / "stage1" / "run_stage1.py", "Stage1Runner")
except Exception as e:
    print(f"⚠️ Failed to import Stage1Runner: {e}")
    Stage1Runner = None

try:
    Stage2Runner = _import_runner(CURRENT_DIR / "stage2" / "run_stage2.py", "Stage2Runner")
except Exception as e:
    print(f"⚠️ Failed to import Stage2Runner: {e}")
    Stage2Runner = None

try:
    Stage3Runner = _import_runner(CURRENT_DIR / "stage3" / "run_stage3.py", "Stage3Runner")
except Exception as e:
    print(f"⚠️ Failed to import Stage3Runner: {e}")
    Stage3Runner = None

try:
    Stage4Runner = _import_runner(CURRENT_DIR / "stage4" / "run_stage4.py", "Stage4Runner")
except Exception as e:
    print(f"⚠️ Failed to import Stage4Runner: {e}")
    Stage4Runner = None

# Import the arrow cleanup utility
try:
    from stage_clean_arrows import ArrowCleaner
    ARROW_CLEANER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Failed to import ArrowCleaner: {e}")
    ARROW_CLEANER_AVAILABLE = False

# Import Memory
from memory import Memory


# ============================================================================
# Stage 1 scene-scale validator (auto-rescale guard)
#
# Vision models systematically under-estimate the dimensions of furniture-rich
# residential / office rooms. Even with the residential prompt addendum, an
# occasional cramped estimate slips through. This module-level helper works
# off the parsed Stage 1 JSON and bumps `estimated_dimensions` up when the
# total floor footprint of MAJOR FLOOR objects is incompatible with the
# claimed area, leaving an auditable record on the Stage 1 result.
# ============================================================================

# Approximate floor footprint in m² for common major floor furniture, keyed by
# substring patterns matched against the object name. First match wins;
# patterns are evaluated in declaration order, so longer/more-specific keys
# go FIRST. Values are deliberately on the LOW side of typical real
# dimensions so we don't aggressively over-bump rooms that are genuinely
# small. Unknown major-floor objects fall back to 1.0 m².
_FURNITURE_FOOTPRINT_M2: List[Tuple[Tuple[str, ...], float]] = [
    (("king bed", "king-size"),                                 3.6),
    (("queen bed", "queen-size"),                               3.0),
    (("double bed", "full bed"),                                2.6),
    (("single bed", "twin bed"),                                1.9),
    (("loft bed", "bunk bed", "murphy bed"),                    2.5),
    (("crib", "cot"),                                           1.4),
    (("bed",),                                                  2.5),
    (("sectional", "l-shaped sofa", "l shaped sofa"),           4.0),
    (("3-seater", "three-seat", "sofa", "couch"),               2.0),
    (("loveseat", "2-seater", "two-seat"),                      1.5),
    (("u-shaped desk", "u shaped desk", "u-desk"),              4.0),
    (("l-shaped desk", "l shaped desk", "corner desk"),         2.5),
    (("executive desk",),                                       1.8),
    (("standing desk", "writing desk", "office desk"),          1.4),
    (("desk",),                                                 1.4),
    (("drafting table", "drawing table", "architect table"),    1.2),
    (("conference table", "meeting table", "boardroom table"),  2.5),
    (("dining table", "kitchen table"),                         1.6),
    (("coffee table", "cocktail table"),                        0.8),
    (("kitchen island",),                                       2.0),
    (("kitchen counter", "kitchen worktop", "kitchenette"),     1.0),
    (("kitchen base cabinet", "base cabinet", "lower cabinet"), 0.6),
    (("wardrobe", "armoire", "closet"),                         1.4),
    (("dresser", "chest of drawers", "tallboy"),                0.7),
    (("bookcase", "bookshelf", "etagere", "library shelf"),     0.7),
    (("filing cabinet", "file cabinet"),                        0.3),
    (("media console", "tv console", "tv stand", "media unit"), 0.7),
    (("console table", "sideboard", "credenza", "buffet"),      0.6),
    (("storage cabinet", "tall cabinet", "display cabinet"),    0.7),
    (("piano",),                                                1.5),
    (("grand piano",),                                          3.0),
    (("pool table", "billiard"),                                4.5),
    (("ping pong", "table tennis"),                             4.0),
    (("bathtub", "tub"),                                        1.4),
    (("shower stall", "shower"),                                0.9),
    (("toilet", "wc"),                                          0.3),
    (("bathroom vanity", "vanity"),                             0.6),
    (("sink", "washbasin", "lavatory"),                         0.3),
    (("refrigerator", "fridge"),                                0.6),
    (("dishwasher",),                                           0.4),
    (("stove", "oven", "range cooker", "cooker"),               0.5),
    (("washing machine", "washer", "dryer"),                    0.5),
    (("armchair", "lounge chair", "recliner", "accent chair"),  1.0),
    (("rocking chair",),                                        0.7),
    (("office chair", "task chair", "swivel chair"),            0.4),
    (("dining chair", "side chair"),                            0.25),
    (("bar stool", "stool"),                                    0.2),
    (("bench", "ottoman", "footstool", "settee"),               0.5),
    (("nightstand", "bedside table", "side table", "end table"), 0.25),
    (("planter", "potted plant", "tree", "plant"),              0.2),
    (("floor lamp", "torchiere"),                               0.1),
    (("rug", "carpet", "doormat"),                              0.0),
    (("curtain", "drape"),                                      0.0),
]


def _major_floor_footprint(name: str) -> float:
    """Best-effort floor footprint in m² for a major-floor object given its name."""
    n = (name or "").lower()
    for keys, m2 in _FURNITURE_FOOTPRINT_M2:
        if any(k in n for k in keys):
            return m2
    return 1.0  # unknown major-floor object → modest default


def _parse_dim_string(s: Any) -> Tuple[Optional[float], Optional[float]]:
    """Parse strings like '6.5m x 4.5m', '6.5 m × 4.5 m', '6.5x4.5m × 2.8m'.

    Returns (W, D) in meters, or (None, None) if unparseable. The first two
    numbers found are taken as W, D; any third number (height) is ignored.
    """
    if not isinstance(s, str) or not s.strip():
        return None, None
    nums = re.findall(r"\d+(?:\.\d+)?", s)
    if len(nums) < 2:
        return None, None
    try:
        return float(nums[0]), float(nums[1])
    except (ValueError, IndexError):
        return None, None


def _format_dim_string(w: float, d: float) -> str:
    return f"{w:.1f}m x {d:.1f}m"


def _compute_scale_audit(
    result: dict,
    packing_factor: float = 0.35,
    min_objects_for_check: int = 3,
    hard_floor_area_m2: float = 6.0,
    tolerance: float = 0.05,
) -> Tuple[bool, Optional[Tuple[float, float]], Dict[str, Any]]:
    """Pure function: decide whether Stage 1 dimensions need rescaling.

    Args:
        result: Stage 1 parsed JSON dict.
        packing_factor: max fraction of floor area covered by furniture
            footprint. 0.35 means a typical residential room is ≤35% covered.
        min_objects_for_check: skip rescale if fewer major-floor objects.
        hard_floor_area_m2: minimum floor area we'll ever bump to.
        tolerance: ignore bumps under this fraction (avoid 1-2% nudges).

    Returns:
        (should_rescale, (new_W, new_D) | None, audit_dict)
        - should_rescale=True ⇒ (new_W, new_D) is the rounded new dim pair.
        - should_rescale=False ⇒ second element may be None or original pair;
          audit_dict still describes what we computed (for logging).
    """
    audit: Dict[str, Any] = {
        "packing_factor": packing_factor,
        "min_objects_for_check": min_objects_for_check,
    }

    if not isinstance(result, dict):
        audit["skipped"] = "result is not a dict"
        return False, None, audit

    scale_block = result.get("scene_scale_understanding") or {}
    dims_str = scale_block.get("estimated_dimensions") if isinstance(scale_block, dict) else None
    audit["original_dimensions"] = dims_str

    major_floor_names: List[str] = []
    for zone in result.get("decoupled_zones", []) or []:
        for obj in zone.get("object_hierarchy", []) or []:
            if (obj.get("category") == "major"
                    and obj.get("placement_type") == "floor"):
                major_floor_names.append(obj.get("name") or "")

    n_objects = len(major_floor_names)
    audit["n_major_floor_objects"] = n_objects
    if n_objects < min_objects_for_check:
        audit["skipped"] = f"only {n_objects} major floor objects (< {min_objects_for_check})"
        return False, None, audit

    total_footprint = sum(_major_floor_footprint(n) for n in major_floor_names)
    min_area = max(total_footprint / packing_factor, hard_floor_area_m2)
    audit["estimated_total_footprint_m2"] = round(total_footprint, 2)
    audit["min_required_area_m2"] = round(min_area, 2)

    W, D = _parse_dim_string(dims_str)

    if W is None or D is None:
        new_W = math.sqrt(min_area * 1.3)
        new_D = min_area / new_W
        audit["reason"] = (
            f"estimated_dimensions {dims_str!r} unparseable; derived from "
            f"{n_objects} major floor objects (footprint {total_footprint:.1f} m², "
            f"min area {min_area:.1f} m²)"
        )
    else:
        cur_area = W * D
        audit["original_area_m2"] = round(cur_area, 2)
        if cur_area >= min_area * (1.0 - tolerance):
            audit["skipped"] = (
                f"current area {cur_area:.1f} m² ≥ min {min_area:.1f} m² "
                f"(within {tolerance*100:.0f}% tolerance)"
            )
            return False, (W, D), audit
        scale = math.sqrt(min_area / cur_area)
        new_W = W * scale
        new_D = D * scale
        audit["scale_factor"] = round(scale, 3)
        audit["reason"] = (
            f"current {W:.1f}m × {D:.1f}m = {cur_area:.1f} m² is below "
            f"furniture-density min {min_area:.1f} m² ({n_objects} major floor "
            f"objects, footprint {total_footprint:.1f} m²); scaled by {scale:.2f}×"
        )

    new_W_rounded = round(new_W * 2) / 2
    new_D_rounded = round(new_D * 2) / 2
    audit["rescaled_dimensions"] = _format_dim_string(new_W_rounded, new_D_rounded)
    return True, (new_W_rounded, new_D_rounded), audit

# Import Stage 3 post-processing: static expansion for composite helpers.
# Load directly via spec_from_file_location to avoid stage3/__init__.py side
# effects; it imports stage3.code_gen_agent -> core, which does not work here.
try:
    _ch_spec = importlib.util.spec_from_file_location(
        "stage3_composite_helpers",
        str(CURRENT_DIR / "stage3" / "composite_helpers.py"),
    )
    _ch_module = importlib.util.module_from_spec(_ch_spec)
    _ch_spec.loader.exec_module(_ch_module)
    expand_composite_helpers = _ch_module.expand_composite_helpers
    COMPOSITE_HELPERS_AVAILABLE = True
except Exception as _e:
    print(f"⚠️ Failed to import composite_helpers: {_e}")
    COMPOSITE_HELPERS_AVAILABLE = False
    expand_composite_helpers = None

# Import image processing utilities
try:
    from image_utils import (
        prepare_image_for_pipeline,
        get_file_size_kb,
        get_image_info,
        compress_to_target_size
    )
    IMAGE_UTILS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Failed to import image_utils: {e}")
    IMAGE_UTILS_AVAILABLE = False


# ============================================================================
# Configuration
# ============================================================================
class PipelineConfig:
    """Pipeline configuration."""
    
    def __init__(
        self,
        image_path: str = None,
        output_dir: str = None,
        run_dir: str = None,
        max_iterations: int = 3,
        
        # Stage 1 configuration
        # Per-stage LLM slots intentionally default to None. Until 2026-05
        # they had hardcoded model defaults (e.g. stage2_model=
        # "gemini-3-flash-preview-thinking") that *appeared* to set per-stage
        # behaviour. In practice argparse forwarded None which overrode
        # them, so Stage 2 silently fell through to the global pro model
        # via `or self.config.model`. When that "bug" was "fixed" by
        # forwarding the constructor defaults, Stage 2 actually started
        # using flash and lost ~80% of its node recall (56 → 13 on the
        # cafe scene, and Stage 3 inherited the loss). The defaults are
        # now None on purpose so the global pro model wins unless the
        # user explicitly opts in via `--stageN-model X`.
        stage1_model: str = None,
        stage1_base_url: str = None,
        stage1_api_key: str = None,

        # Stage 2 configuration
        # None → follow the global --model (so switching the whole pipeline,
        # e.g. to Qwen, only needs the global model/base_url/api_key set).
        # Pin a different model for Stage 2 only via --stage2-model if desired.
        stage2_model: str = None,
        #stage2_model: str = "gpt-5.5",
        stage2_base_url: str = None,
        stage2_api_key: str = None,

        # Stage7–9 & small-object LLM slots (None → fall back to global
        # model / base_url / api_key, same pattern as Stage3).
        stage7_model: str = None,
        stage7_base_url: str = None,
        stage7_api_key: str = None,
        stage8_model: str = None,
        stage8_base_url: str = None,
        stage8_api_key: str = None,
        stage7_small_objects_model: str = None,
        stage7_small_objects_base_url: str = None,
        stage7_small_objects_api_key: str = None,
        stage8_small_describe_model: str = None,
        stage8_small_describe_base_url: str = None,
        stage8_small_describe_api_key: str = None,
        stage9_small_geometry_model: str = None,
        stage9_small_geometry_base_url: str = None,
        stage9_small_geometry_api_key: str = None,
        stage9_model: str = None,
        stage9_base_url: str = None,
        stage9_api_key: str = None,

        # Stage 3 configuration
        stage3_iterate: bool = True,
        stage3_target_score: float = 0.85,
        blender_path: str = "/Applications/Blender.app/Contents/MacOS/Blender",
        stage3_model: str = None,
        stage3_base_url: str = None,
        stage3_api_key: str = None,
        
        # Stage 3 rotation configuration
        stage3_rotation: bool = False,
        stage3_rotation_iterations: int = 3,
        stage3_rotation_target: float = 0.95,
        
        # Stage 8 configuration
        stage8_parallel: int = 8,
        stage8_geometry_max_attempts: int = 3,
        stage8_geometry_retry_delay: float = 2.0,

        # Stage 8 / 9 (detailed small objects) is disabled by default;
        # enable it with --detail-small-objects.
        detail_small_objects: bool = False,
        small_describe_batch_size: int = 8,
        small_describe_parallel: int = 4,
        small_geometry_parallel: int = 8,

        # Stage 10 (material) Option-C batching: scene-palette pre-pass +
        # parallel per-part batches. See stage10_material.py for details.
        material_batch_size: int = 6,
        material_parallel: int = 4,
        material_max_attempts: int = 3,

        # Stage 11 image-texture generation configuration.
        stage11_texture_model: str = None,
        stage11_texture_base_url: str = None,
        stage11_texture_api_key: str = None,
        stage11_texture_image_size: str = "1K",
        stage11_texture_parallel: int = 8,
        stage11_texture_max_wall_arts: int = 20,
        
        # LLM configuration
        # model: str = "gemini-3.1-pro-preview-thinking",
        base_url: str = None,
        api_key: str = None,
        model: str = None,
        stage2_max_tokens: int = 65536,
        
        # Runtime configuration
        verbose: bool = True,
        start_stage: int = 1,
        end_stage: int = 12,
        
        # Image compression configuration
        compress_image: bool = True,
        image_target_kb: int = 800,

        # Scene classification configuration for Stage 1 / Stage 3 prompt routing
        scene_classify: bool = True,
        scene_type_override: Optional[str] = None,
        scene_classify_model: Optional[str] = None,
    ):
        self.image_path = image_path
        
        # Run directory: each run gets its own folder
        if run_dir:
            run_path = Path(run_dir)
            if not run_path.is_absolute():
                run_path = CURRENT_DIR / run_path
            self.output_dir = str(run_path)
        elif output_dir:
            # Treat --output-dir as a parent directory and create a
            # `run_<timestamp>_<image_stem>` subfolder underneath it, matching
            # the default layout. (If the supplied path itself already starts
            # with `run_`, assume the user wants to use it verbatim.)
            out_path = Path(output_dir)
            if out_path.name.startswith("run_"):
                self.output_dir = str(out_path)
            else:
                self.output_dir = self._generate_run_dir(image_path, base_dir=output_dir)
        else:
            self.output_dir = self._generate_run_dir(image_path)
        
        # Memory file lives inside the run directory.
        # When output_dir is under CURRENT_DIR we keep the relative form so
        # logs and memory listings stay short (and tools that join against
        # workspace_dir keep working). When the user supplies an absolute
        # output path *outside* agent_utils/ (e.g. --output-root
        # /Users/.../CAR3D_output/...), fall back to an absolute path —
        # Memory(__init__) handles that correctly because pathlib's `/`
        # operator returns the right operand verbatim when it's absolute.
        out_path = Path(self.output_dir).resolve()
        try:
            self.memory_file = str(
                out_path.relative_to(CURRENT_DIR) / "agent_memory.jsonl"
            )
        except ValueError:
            self.memory_file = str(out_path / "agent_memory.jsonl")
        
        # Image compression
        self.compress_image = compress_image
        self.image_target_kb = image_target_kb
        self.max_iterations = max_iterations
        
        # Stage1
        self.stage1_model = stage1_model
        self.stage1_base_url = stage1_base_url
        self.stage1_api_key = stage1_api_key
        
        # Stage2
        self.stage2_model = stage2_model
        self.stage2_base_url = stage2_base_url
        self.stage2_api_key = stage2_api_key

        # Stage7–9 & small-object dedicated LLM overrides
        # Respect the constructor arg (from --stage5-model); None → fall back
        # to the global model via _resolve_stage_llm. Previously hardcoded to
        # gemini, which silently ignored any global/per-stage override.
        self.stage7_model = stage7_model
        self.stage7_base_url = stage7_base_url
        self.stage7_api_key = stage7_api_key


        # Respect the constructor arg (from --stage6-model); None → global model.
        self.stage8_model = stage8_model
        self.stage8_base_url = stage8_base_url
        self.stage8_api_key = stage8_api_key


        self.stage7_small_objects_model =  None
        self.stage7_small_objects_base_url = stage7_small_objects_base_url
        self.stage7_small_objects_api_key = stage7_small_objects_api_key

        # Respect the constructor arg (from --stage8-small-describe-model); None → global model.
        self.stage8_small_describe_model = stage8_small_describe_model
        self.stage8_small_describe_base_url = stage8_small_describe_base_url
        self.stage8_small_describe_api_key = stage8_small_describe_api_key


        self.stage9_small_geometry_model = None
        self.stage9_small_geometry_base_url = stage9_small_geometry_base_url
        self.stage9_small_geometry_api_key = stage9_small_geometry_api_key


        self.stage9_model =  None
        self.stage9_base_url = stage9_base_url
        self.stage9_api_key = stage9_api_key
        
        # Stage3
        self.stage3_iterate = stage3_iterate
        self.stage3_target_score = stage3_target_score
        self.blender_path = blender_path
        self.stage3_model = stage3_model
        self.stage3_base_url = stage3_base_url
        self.stage3_api_key = stage3_api_key
        
        # Stage3 Rotation
        self.stage3_rotation = stage3_rotation
        self.stage3_rotation_iterations = stage3_rotation_iterations
        self.stage3_rotation_target = stage3_rotation_target
        
        # Stage8
        self.stage8_parallel = stage8_parallel
        self.stage8_geometry_max_attempts = stage8_geometry_max_attempts
        self.stage8_geometry_retry_delay = stage8_geometry_retry_delay

        # LLM
        self.model = model or os.environ.get("SCENEGEN_MODEL") or "gemini-3.1-pro-preview"
        self.base_url = base_url or os.environ.get("SCENEGEN_BASE_URL")
        self.api_key = api_key or os.environ.get("SCENEGEN_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.stage2_max_tokens = stage2_max_tokens
        
        # Runtime
        self.verbose = verbose
        self.start_stage = start_stage
        self.end_stage = end_stage

        # Stage 5-12 configuration
        self.stage7_enabled = True   # stage5_describe
        self.stage8_enabled = True   # stage6_geometry
        self.stage7_small_objects_enabled = True  # surface-driven small objects
        # Parallel LLM workers for Stage 7 (one per parent object). Each
        # parent = one vision-language call, so parallel=4 typically cuts
        # this stage's wallclock by ~3-4x without overloading the server.
        self.stage7_small_objects_parallel = 4

        # Stage 8 / 9 (detailed small objects): off by default. Toggle via
        # `--detail-small-objects` on the CLI. When on, every Stage-7 item
        # is sent through Stage 8 (describe) and Stage 9 (detailed composite
        # geometry) before Stage 10 (material). Strategy = `all`;
        # cost is controlled by `small_describe_parallel` /
        # `small_describe_batch_size`.
        self.detail_small_objects = detail_small_objects
        self.small_describe_batch_size = small_describe_batch_size
        self.small_describe_parallel = small_describe_parallel
        self.small_geometry_parallel = small_geometry_parallel
        # Stage 10 (material) batching knobs
        self.material_batch_size = material_batch_size
        self.material_parallel = material_parallel
        self.material_max_attempts = material_max_attempts
        
        self.stage10_enabled = True  # stage10_material
        self.stage11_enabled = True  # stage11_texture (nanobanana)
        self.stage12_enabled = True  # stage12_render

        # Scene classification
        self.scene_classify = scene_classify
        self.scene_type_override = scene_type_override
        self.scene_classify_model = scene_classify_model

        # Stage 11 (stage11_texture) image-generation configuration.
        # Historical attribute names are kept for downstream compatibility.
        self.stage10_model = (
            stage11_texture_model
            or os.environ.get("SCENEGEN_TEXTURE_MODEL")
            or "gemini-3-pro-image-preview"
        )
        self.stage10_base_url = (
            stage11_texture_base_url
            or os.environ.get("SCENEGEN_TEXTURE_BASE_URL")
        )
        self.stage10_api_key = (
            stage11_texture_api_key
            or os.environ.get("SCENEGEN_TEXTURE_API_KEY")
            or self.api_key
        )
        self.stage10_image_size = stage11_texture_image_size
        self.stage10_parallel = stage11_texture_parallel
        self.stage10_max_wall_arts = stage11_texture_max_wall_arts
        # Wall visual intensity: 'subtle' | 'bold' | 'mural_like' | None
        # None uses stage11_texture's fallback chain
        # (WALL_MAT_DATA -> material_config -> default subtle).
        self.stage10_wall_intensity: Optional[str] = None
    
    @staticmethod
    def _default_output_base_dir(image_path: str = None) -> Path:
        """Return the default parent directory for a new run.

        Normal runs live next to the input image so image/output pairs stay
        together. Calls without an image keep the historical workspace default.
        """
        if image_path:
            return Path(image_path).expanduser().resolve().parent
        return CURRENT_DIR / "pipeline_output"

    @staticmethod
    def _generate_run_dir(image_path: str = None, base_dir: str = None) -> str:
        """Generate a unique run directory name based on timestamp and image name.

        Args:
            image_path: Source image, used to derive a human-readable stem.
            base_dir:   Parent directory for the new `run_*` folder. Defaults to
                        the input image directory; pass an explicit path via
                        ``--output-dir`` to redirect runs elsewhere.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if image_path:
            stem = Path(image_path).stem
            stem = stem[:50].rstrip("_").rstrip("-")
            run_name = f"run_{timestamp}_{stem}"
        else:
            run_name = f"run_{timestamp}"
        base = (
            Path(base_dir).expanduser().resolve()
            if base_dir
            else PipelineConfig._default_output_base_dir(image_path)
        )
        return str(base / run_name)


# ============================================================================
# Main pipeline class
# ============================================================================
class UnifiedPipeline:
    """
    Unified pipeline that orchestrates all stages.
    
    Flow:
        Stage1 (spatial semantic analysis)
          → Stage2 (Scene Graph)
            → Stage3 (Blender code generation)
              → Stage4 (wall-mounted decor only)
                → Stage5 (major object descriptions)
                  → Stage6 (detailed major-object geometry)
                    → Stage7 (surface small objects from plane detection)
                      → Stage8 (detailed small-object descriptions, optional)
                        → Stage9 (detailed small-object geometry, optional)
                          → Stage10 (per-part materials)
                            → Stage11 (nanobanana real textures)
                              → Stage12 (render-ready script)
    """
    
    # Stage metadata
    STAGE_INFO = {
        1: {"name": "Stage1", "desc": "Spatial semantic analysis", "emoji": "🔍"},
        2: {"name": "Stage2", "desc": "Scene Graph construction", "emoji": "🌳"},
        3: {"name": "Stage3", "desc": "Blender code generation", "emoji": "🎨"},
        4: {"name": "Stage4", "desc": "Small-object insertion", "emoji": "🪑"},
        5: {"name": "Stage5_describe", "desc": "Object description generation", "emoji": "📝"},
        6: {"name": "Stage6_geometry", "desc": "Detailed geometry generation", "emoji": "🔷"},
        7: {"name": "Stage7_small_objects", "desc": "Surface small-object placement", "emoji": "🧸"},
        8: {"name": "Stage8_small_describe", "desc": "Detailed small-object descriptions (optional)", "emoji": "🔬"},
        9: {"name": "Stage9_small_geometry", "desc": "Detailed small-object geometry (optional)", "emoji": "🧩"},
        10: {"name": "Stage10_material", "desc": "Material texture generation", "emoji": "🎨"},
        11: {"name": "Stage11_texture", "desc": "Real texture generation (nanobanana)", "emoji": "🖼️"},
        12: {"name": "Stage12_render", "desc": "Render-ready script generation", "emoji": "💡"},
    }
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.memory = Memory(
            workspace_dir=str(CURRENT_DIR),
            memory_file=config.memory_file
        )
        self.results: Dict[str, Any] = {}
        self.start_time = None
    
    def _llm_kwargs(self) -> Dict[str, Any]:
        """Return global LLM kwargs, dropping any that are None.

        Rationale: Stage Runner constructors have their own sensible
        defaults (e.g. Stage4Runner points at a working base_url). Passing
        an explicit None would OVERRIDE those Python-signature defaults and
        force the Runner to fall back to LLMClient's global default — which
        may be an unreachable gateway. By only forwarding keys the user
        actually set, each Runner's own default remains in effect.
        """
        kw: Dict[str, Any] = {}
        if self.config.model is not None:
            kw["model"] = self.config.model
        if self.config.base_url is not None:
            kw["base_url"] = self.config.base_url
        if self.config.api_key is not None:
            kw["api_key"] = self.config.api_key
        return kw

    def _resolve_stage_llm(
        self,
        stage_model: Optional[str],
        stage_base_url: Optional[str],
        stage_api_key: Optional[str],
    ) -> Tuple[str, Optional[str], Optional[str]]:
        """Resolve model/base_url/api_key for a stage (same rule as Stage2/3).

        Unset stage slots fall back to global ``model`` / ``base_url`` /
        ``api_key`` (same ``or`` rule as ``run_stage2``).
        """
        m = stage_model or self.config.model
        u = stage_base_url or self.config.base_url
        k = stage_api_key or self.config.api_key
        return m, u, k

    def _log(self, msg: str, level: str = "info"):
        """Log a message."""
        if not self.config.verbose:
            return
        
        prefix = {
            "info": "ℹ️",
            "success": "✅",
            "warning": "⚠️",
            "error": "❌",
            "stage": "🔷",
            "step": "📋",
        }.get(level, "")
        print(f"{prefix} {msg}")

    def _log_stage_model(
        self,
        stage_label: str,
        model: Optional[str],
        base_url: Optional[str],
        dedicated: bool = False,
        extra: Optional[str] = None,
    ) -> None:
        """Print which LLM a stage is about to use, with provenance.

        ``dedicated=True`` means the value came from a stage-specific config
        slot (either an explicit CLI override OR the per-stage constructor
        default in ``PipelineConfig``). ``dedicated=False`` means it fell
        back to the global ``model`` slot. Either way we always print, so
        future regressions don't go silent.
        """
        if not self.config.verbose:
            return
        source = "dedicated model" if dedicated else "global model"
        suffix = f" [{extra}]" if extra else ""
        url_part = f" @ {base_url}" if base_url else ""
        self._log(
            f"{stage_label} using {source}: {model}{url_part}{suffix}",
            "info",
        )

    def _print_header(self, title: str, width: int = 60):
        """Print a boxed section header."""
        print("\n")
        print("╔" + "═" * (width - 2) + "╗")
        print(f"║  {title:^{width - 6}}  ║")
        print("╚" + "═" * (width - 2) + "╝")
    
    def _print_stage_header(self, stage_num):
        """Print a boxed stage header."""
        info = self.STAGE_INFO.get(stage_num, {})
        name = info.get("name", f"Stage{stage_num}")
        desc = info.get("desc", "")
        emoji = info.get("emoji", "")
        
        self._print_header(f"{emoji} {name}: {desc}")
    
    def _save_run_config(self):
        """Save run configuration to the run directory for reproducibility."""
        config_path = os.path.join(self.config.output_dir, "run_config.json")
        run_config = {
            "image_path": self.config.image_path,
            "start_stage": self.config.start_stage,
            "end_stage": self.config.end_stage,
            "model": self.config.model,
            "stage3_model": self.config.stage3_model,
            "stage5_model": self.config.stage7_model,
            "stage6_model": self.config.stage8_model,
            "stage7_small_objects_model": self.config.stage7_small_objects_model,
            "stage8_small_describe_model": self.config.stage8_small_describe_model,
            "stage9_small_geometry_model": self.config.stage9_small_geometry_model,
            "stage10_material_model": self.config.stage9_model,
            "max_iterations": self.config.max_iterations,
            "stage3_iterate": self.config.stage3_iterate,
            "stage3_target_score": self.config.stage3_target_score,
            "compress_image": self.config.compress_image,
            "image_target_kb": self.config.image_target_kb,
            "timestamp": datetime.now().isoformat(),
            "output_dir": self.config.output_dir,
        }
        try:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(run_config, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
    
    # ========================================================================
    # Image preprocessing
    # ========================================================================
    def _preprocess_image(self) -> bool:
        """
        Preprocess the image by compressing it to the target size.
        
        Returns:
            Whether preprocessing succeeded.
        """
        if not self.config.image_path:
            return True  # No image path; skip.
        
        if not self.config.compress_image:
            self._log("Image compression is disabled", "info")
            return True
        
        if not IMAGE_UTILS_AVAILABLE:
            self._log("image_utils is unavailable; skipping image compression", "warning")
            return True
        
        if not os.path.exists(self.config.image_path):
            self._log(f"Image does not exist: {self.config.image_path}", "error")
            return False
        
        try:
            original_size = get_file_size_kb(self.config.image_path)
            self._log(f"Original image size: {original_size:.1f} KB", "info")
            
            # Skip compression if the image is already small enough.
            if original_size <= self.config.image_target_kb * 1.2:
                self._log("Image is already within the target size range; no compression needed", "success")
                return True
            
            # Compress the image.
            self._log(f"Compressing image to {self.config.image_target_kb} KB...", "step")
            
            # Output directory
            compressed_dir = Path(self.config.output_dir) / "compressed_images"
            compressed_dir.mkdir(parents=True, exist_ok=True)
            
            # Compress
            compressed_path = prepare_image_for_pipeline(
                self.config.image_path,
                output_dir=str(compressed_dir),
                target_kb=self.config.image_target_kb,
                verbose=self.config.verbose
            )
            
            # Update the image path in the config.
            if compressed_path != self.config.image_path:
                new_size = get_file_size_kb(compressed_path)
                reduction = (1 - new_size / original_size) * 100
                self._log(f"Compression complete: {original_size:.1f} KB → {new_size:.1f} KB ({reduction:.1f}% smaller)", "success")
                self._log(f"Using compressed image: {compressed_path}", "info")
                self.config.image_path = compressed_path
            
            return True
            
        except Exception as e:
            self._log(f"Image compression failed: {e}", "error")
            self._log("Continuing with the original image", "warning")
            return True  # Compression failure should not block the pipeline.
    
    # ========================================================================
    # Stage 0: scene classification for downstream prompt routing
    # ========================================================================
    def _classify_scene(self) -> None:
        """Run the lightweight scene classifier and write the result to Memory.

        Fallback behavior writes scene_type='other' and does not block the pipeline.
        """
        try:
            from scene_classifier import classify_scene
        except Exception as exc:
            self._log(f"scene_classifier is unavailable ({exc}); skipping classification", "warning")
            return

        try:
            memory = Memory(
                workspace_dir=str(CURRENT_DIR),
                memory_file=self.config.memory_file,
            )
        except Exception as exc:
            self._log(f"Memory initialization failed ({exc}); skipping classification persistence", "warning")
            memory = None

        clf_model = (
            self.config.scene_classify_model
            or self.config.stage1_model
            or self.config.model
        )
        clf_base_url = self.config.stage1_base_url or self.config.base_url
        clf_api_key = self.config.stage1_api_key or self.config.api_key

        result = classify_scene(
            self.config.image_path,
            memory=memory,
            manual_override=self.config.scene_type_override,
            use_llm=True,
            model=clf_model,
            base_url=clf_base_url,
            api_key=clf_api_key,
            verbose=self.config.verbose,
        )

        scene_type = result.get("scene_type", "other")
        confidence = result.get("confidence", 0.0)
        source = result.get("source", "unknown")
        self.results["scene_classify"] = result
        self._log(
            f"Scene classification complete: scene_type={scene_type} "
            f"(confidence={confidence:.2f}, source={source})",
            "success" if scene_type != "other" else "info",
        )

    # ========================================================================
    # Stage 1: spatial semantic analysis
    # ========================================================================
    def run_stage1(self) -> Tuple[bool, Any, Dict]:
        """Run Stage 1: spatial semantic analysis."""
        if Stage1Runner is None:
            self._log("Stage1Runner was not imported", "error")
            return False, None, {"error": "Stage1Runner not available"}
        
        self._print_stage_header(1)

        stage1_model = self.config.stage1_model or self.config.model
        stage1_base_url = self.config.stage1_base_url or self.config.base_url
        stage1_api_key = self.config.stage1_api_key or self.config.api_key

        self._log_stage_model("Stage1", stage1_model, stage1_base_url,
                              dedicated=bool(self.config.stage1_model))

        runner = Stage1Runner(
            image_path=self.config.image_path,
            output_dir=os.path.join(self.config.output_dir, "stage1"),
            max_iterations=self.config.max_iterations,
            verbose=self.config.verbose,
            model=stage1_model,
            base_url=stage1_base_url,
            api_key=stage1_api_key,
            memory_file=self.config.memory_file
        )
        
        success, result, meta = runner.run()

        # Furniture-density auto-rescale: catch obviously-cramped rooms even
        # when the residential prompt addendum was used. Mutates `result` in
        # place and writes the bump back to Memory + stage1_output.json so
        # downstream stages pick up the corrected dimensions automatically.
        if success and isinstance(result, dict):
            try:
                self._validate_and_rescale_stage1(result)
            except Exception as e:
                self._log(f"Stage 1 scale validator failed ({e}); continuing with original values", "warning")

        self.results["stage1"] = {"success": success, "result": result, "meta": meta}

        return success, result, meta

    # ------------------------------------------------------------------
    # Stage 1 scene-scale validator
    # ------------------------------------------------------------------
    def _validate_and_rescale_stage1(self, result: dict) -> bool:
        """Sanity-check Stage 1's `estimated_dimensions` and bump if too small.

        Uses module-level `_compute_scale_audit` to decide; on bump, mutates
        `result` in place, rewrites the corresponding Memory entry plus
        `stage1_output.json`, and prints an audit warning. Returns True iff
        a rescale happened.
        """
        should_rescale, new_dims, audit = _compute_scale_audit(result)
        if not should_rescale or not new_dims:
            if audit.get("skipped") and self.config.verbose:
                self._log(f"Stage 1 scale check: {audit['skipped']}", "info")
            return False

        new_W, new_D = new_dims
        new_dims_str = _format_dim_string(new_W, new_D)

        scale_block = result.setdefault("scene_scale_understanding", {})
        scale_block["estimated_dimensions"] = new_dims_str

        existing_audit = scale_block.get("scale_audit") or {}
        existing_audit.update({
            "auto_rescaled": True,
            "rescaled_at": datetime.now().isoformat(),
            **audit,
        })
        scale_block["scale_audit"] = existing_audit

        # Persist back to Memory: mutate the latest entry in-place.
        try:
            latest = self.memory.get_latest(stage="stage1", type="result")
            if latest is not None and isinstance(latest.content, dict):
                latest.content = result
                self.memory._save()
        except Exception as e:
            self._log(f"Stage 1 rescale: memory update skipped ({e})", "warning")

        # Persist back to stage1_output.json so any external reader sees the
        # bumped value as well.
        try:
            out_path = os.path.join(
                self.config.output_dir, "stage1", "stage1_output.json"
            )
            if os.path.exists(out_path):
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self._log(f"Stage 1 rescale: file rewrite skipped ({e})", "warning")

        # Upper-bound warning (do not auto-shrink — let user review)
        if new_W > 14 or new_D > 14:
            self._log(
                f"Stage 1 dimension > 14 m after rescale: {new_W:.1f} × "
                f"{new_D:.1f} m — please review",
                "warning",
            )

        self._log(
            f"📏 Stage 1 scale auto-rescaled: "
            f"{audit.get('original_dimensions')!r} → {new_dims_str!r}",
            "warning",
        )
        self._log(f"    reason: {audit.get('reason')}", "info")
        return True

    # ========================================================================
    # Stage 2: Scene Graph construction
    # ========================================================================
    def run_stage2(self) -> Tuple[bool, Any, Dict]:
        """Run Stage 2: Scene Graph construction."""
        if Stage2Runner is None:
            self._log("Stage2Runner was not imported", "error")
            return False, None, {"error": "Stage2Runner not available"}
        
        self._print_stage_header(2)
        
        stage2_model = self.config.stage2_model or self.config.model
        stage2_base_url = self.config.stage2_base_url or self.config.base_url
        stage2_api_key = self.config.stage2_api_key or self.config.api_key

        self._log_stage_model("Stage2", stage2_model, stage2_base_url,
                              dedicated=bool(self.config.stage2_model))

        runner = Stage2Runner(
            image_path=self.config.image_path,
            output_dir=os.path.join(self.config.output_dir, "stage2"),
            max_iterations=self.config.max_iterations,
            verbose=self.config.verbose,
            model=stage2_model,
            base_url=stage2_base_url,
            api_key=stage2_api_key,
            memory_file=self.config.memory_file,
        )
        
        success, result, meta = runner.run()
        self.results["stage2"] = {"success": success, "result": result, "meta": meta}
        
        return success, result, meta
    
    # ========================================================================
    # Stage 3: Blender code generation
    # ========================================================================
    def run_stage3(self) -> Tuple[bool, float, Dict]:
        """Run Stage 3: Blender code generation."""
        if Stage3Runner is None:
            self._log("Stage3Runner was not imported", "error")
            return False, 0, {"error": "Stage3Runner not available"}
        
        self._print_stage_header(3)
        
        # Stage 3 can use a dedicated model, including Codex/OpenAI models.
        stage3_model = self.config.stage3_model or self.config.model
        stage3_base_url = self.config.stage3_base_url or self.config.base_url
        stage3_api_key = self.config.stage3_api_key or self.config.api_key
        
        # If stage3_model is an OpenAI model, use the OpenAI API automatically.
        openai_models = {"gpt-5.1-codex-max", "gpt-5.1-codex", "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "o1", "o1-mini", "o3-mini"}
        is_openai = bool(stage3_model and any(stage3_model.startswith(m) for m in openai_models))
        is_gpt_model = bool(stage3_model and stage3_model.lower().startswith("gpt"))
        if is_openai:
            if self.config.stage3_base_url is None:
                stage3_base_url = None  # Let LLMClient use the OpenAI API automatically.
            if self.config.stage3_api_key is None:
                stage3_api_key = None  # Let LLMClient read the key from the environment.
        if is_gpt_model:
            self._log("Stage3: GPT model detected; preview render labels disabled", "info")

        self._log_stage_model(
            "Stage3", stage3_model, stage3_base_url,
            dedicated=bool(self.config.stage3_model),
            extra=("OpenAI" if is_openai else None),
        )

        runner = Stage3Runner(
            image_path=self.config.image_path,
            output_dir=os.path.join(self.config.output_dir, "stage3"),
            blender_path=self.config.blender_path,
            max_iterations=self.config.max_iterations if self.config.stage3_iterate else 1,
            target_score=self.config.stage3_target_score,
            verbose=self.config.verbose,
            model=stage3_model,
            base_url=stage3_base_url,
            api_key=stage3_api_key,
            memory_file=self.config.memory_file,
            render_labels=not is_gpt_model
        )
        
        success, score = runner.run(iterate=self.config.stage3_iterate)
        self.results["stage3"] = {"success": success, "score": score}
        
        # Auto-run rotation correction after Stage 3
        # Rotation correction is disabled by default because results are weak.
        # Enable it with --enable-rotation when needed.
        if success and self.config.stage3_rotation:
            self._log("Stage 3 complete; running rotation correction...", "info")
            rot_success, rot_score = self._run_stage3_rotation()
            if rot_success:
                self._log(f"Rotation correction complete (score={rot_score:.0%})", "success")
        elif success:
            self._log("Rotation correction is disabled; enable it with --enable-rotation", "info")
        
        # Stage 3 post-process: expand composite helpers such as
        # create_double_deck_bench so downstream stages see the primitive
        # create_box calls directly. The resulting Blender geometry is
        # equivalent to running the helper.
        if success:
            self._expand_stage3_composite_helpers()
        
        return success, score, {"score": score}
    
    # ========================================================================
    # Stage 3 post-process: static composite-helper expansion
    # ========================================================================
    def _expand_stage3_composite_helpers(self) -> None:
        """Run expand_composite_helpers on the latest Stage 3 code.
        
        On success, updates Memory and the script on disk. On failure, logs a
        warning and keeps the pipeline running with the original helper calls.
        """
        if not COMPOSITE_HELPERS_AVAILABLE:
            return
        try:
            entry = self.memory.get_latest(stage="stage3", type="result")
            if not entry or not isinstance(entry.content, str):
                return
            before = entry.content
            after = expand_composite_helpers(before)
            if after == before:
                return
            
            # Write back to Memory as a new latest result entry.
            self.memory.add(stage="stage3", type="result", content=after)
            
            # Sync the disk script: prefer stage3_rotation_output.py when
            # rotation ran, otherwise write stage3_output.py.
            stage3_dir = os.path.join(self.config.output_dir, "stage3")
            written = False
            for fname in ("stage3_rotation_output.py", "stage3_output.py"):
                fpath = os.path.join(stage3_dir, fname)
                if os.path.exists(fpath):
                    with open(fpath, "w") as f:
                        f.write(after)
                    self._log(f"Composite helpers expanded -> {fname}", "success")
                    written = True
                    break
            if not written:
                self._log("Composite helpers expanded in Memory only; no disk script found", "info")
        except Exception as e:
            self._log(f"Composite helper expansion failed; downstream stages will see helper calls: {e}", "warning")
    
    # ========================================================================
    # Stage 3 Rotation: rotation correction
    # ========================================================================
    def _run_stage3_rotation(self) -> Tuple[bool, float]:
        """Run Stage 3 rotation correction."""
        try:
            from stage3_rotation import Stage3RotationRunner
            
            stage3_model = self.config.stage3_model or self.config.model
            stage3_base_url = self.config.stage3_base_url or self.config.base_url
            stage3_api_key = self.config.stage3_api_key or self.config.api_key
            
            runner = Stage3RotationRunner(
                image_path=self.config.image_path,
                output_dir=os.path.join(self.config.output_dir, "stage3"),
                blender_path=self.config.blender_path,
                max_iterations=self.config.stage3_rotation_iterations,
                target_score=self.config.stage3_rotation_target,
                verbose=self.config.verbose,
                memory_file=self.config.memory_file,
                model=stage3_model,
                base_url=stage3_base_url,
                api_key=stage3_api_key,
            )
            
            return runner.run()
        except Exception as e:
            self._log(f"Rotation correction error: {e}", "error")
            import traceback
            traceback.print_exc()
            return False, 0.0
    
    # ========================================================================
    # Stage 4: add small objects
    # ========================================================================
    def run_stage4(self) -> Tuple[bool, Any, Dict]:
        """Run Stage 4: add small objects."""
        if Stage4Runner is None:
            self._log("Stage4Runner was not imported", "error")
            return False, None, {"error": "Stage4Runner not available"}
        
        self._print_stage_header(4)
        
        runner = Stage4Runner(
            image_path=self.config.image_path,
            output_dir=os.path.join(self.config.output_dir, "stage4"),
            verbose=self.config.verbose,
            memory_file=self.config.memory_file,
            **self._llm_kwargs(),
        )
        
        success, code = runner.run()
        self.results["stage4"] = {"success": success, "has_code": code is not None}
        
        # Remove direction arrows.
        if success and code and ARROW_CLEANER_AVAILABLE:
            self._log("Removing direction arrows...", "step")
            cleaner = ArrowCleaner(
                output_dir=os.path.join(self.config.output_dir, "stage4"),
                verbose=self.config.verbose
            )
            clean_success, clean_code = cleaner.run(stage4_code=code)
            if clean_success:
                code = clean_code
                self._log("Arrow cleanup complete", "success")
        
        return success, code, {"has_code": code is not None}
    
    # ========================================================================
    # Stage 5: object description generation (stage5_describe)
    # ========================================================================
    def run_stage5(self) -> Tuple[bool, Any, Dict]:
        """Run Stage 5: object description generation."""
        self._print_stage_header(5)

        try:
            # Dynamically import StageDescribeRunner.
            import sys
            sys.path.insert(0, str(CURRENT_DIR))
            from stage5_describe import StageDescribeRunner

            sm, sbu, sak = self._resolve_stage_llm(
                self.config.stage7_model,
                self.config.stage7_base_url,
                self.config.stage7_api_key,
            )
            self._log_stage_model(
                "Stage5",
                sm,
                sbu,
                dedicated=bool(self.config.stage7_model),
            )

            runner = StageDescribeRunner(
                image_path=self.config.image_path,
                scene_code_path=None,
                output_dir=os.path.join(self.config.output_dir, "stage5_describe"),
                use_memory=True,
                verbose=self.config.verbose,
                memory_file=self.config.memory_file,
                model=sm,
                base_url=sbu,
                api_key=sak,
            )

            success = runner.run()
            self.results["stage5"] = {"success": success}

            return success, None, {"success": success}

        except Exception as e:
            self._log(f"Stage 5 error: {e}", "error")
            import traceback
            traceback.print_exc()
            return False, None, {"error": str(e)}

    # ========================================================================
    # Stage 6: detailed geometry generation (stage6_geometry)
    # ========================================================================
    def run_stage6(self) -> Tuple[bool, Any, Dict]:
        """Run Stage 6: detailed geometry generation."""
        self._print_stage_header(6)

        try:
            # Dynamically import StageGeometryRunner.
            import sys
            sys.path.insert(0, str(CURRENT_DIR))
            from stage6_geometry import StageGeometryRunner

            sm, sbu, sak = self._resolve_stage_llm(
                self.config.stage8_model,
                self.config.stage8_base_url,
                self.config.stage8_api_key,
            )
            self._log_stage_model(
                "Stage6",
                sm,
                sbu,
                dedicated=bool(self.config.stage8_model),
            )

            runner = StageGeometryRunner(
                describe_json_path=None,
                output_dir=os.path.join(self.config.output_dir, "stage6_geometry"),
                use_memory=True,
                verbose=self.config.verbose,
                memory_file=self.config.memory_file,
                geometry_max_attempts=self.config.stage8_geometry_max_attempts,
                geometry_retry_delay_sec=self.config.stage8_geometry_retry_delay,
                model=sm,
                base_url=sbu,
                api_key=sak,
            )

            parallel = self.config.stage8_parallel
            if parallel > 0:
                self._log(f"Stage 6 parallel mode: {parallel} workers", "info")
            success, generated_count = runner.run(resume=True, generate_code=True, parallel=parallel)
            self.results["stage6"] = {"success": success, "generated": generated_count}

            return success, None, {"success": success, "generated": generated_count}

        except Exception as e:
            self._log(f"Stage 6 error: {e}", "error")
            import traceback
            traceback.print_exc()
            return False, None, {"error": str(e)}

    # ========================================================================
    # Stage 7: surface small-object placement (stage7_small_objects)
    # ========================================================================
    def run_stage7(self) -> Tuple[bool, Any, Dict]:
        """Run Stage 7 small-object placement from Stage 6 plane detection.

        Reads `stage6_geometry` output, detects placeable planes (table tops,
        open shelves, sofa/chair seats), calls LLM to decorate each plane with
        small objects grounded in the reference image, and appends the resulting
        `create_box`/`create_cylinder` calls to the Stage 6 Blender script.
        """
        self._print_stage_header(7)

        try:
            import sys
            sys.path.insert(0, str(CURRENT_DIR))
            from stage7_small_objects import StageSmallObjectsRunner

            sm, sbu, sak = self._resolve_stage_llm(
                self.config.stage7_small_objects_model,
                self.config.stage7_small_objects_base_url,
                self.config.stage7_small_objects_api_key,
            )
            self._log_stage_model(
                "Stage7",
                sm,
                sbu,
                dedicated=bool(self.config.stage7_small_objects_model),
            )

            runner = StageSmallObjectsRunner(
                image_path=self.config.image_path,
                geometry_json_path=None,
                describe_json_path=None,
                base_code_path=None,
                stage1_json_path=None,
                output_dir=os.path.join(
                    self.config.output_dir, "stage7_small_objects"
                ),
                use_memory=True,
                verbose=self.config.verbose,
                memory_file=self.config.memory_file,
                parallel=self.config.stage7_small_objects_parallel,
                model=sm,
                base_url=sbu,
                api_key=sak,
            )

            success, summary = runner.run()
            plane_count = len(summary.get("planes", [])) if summary else 0
            item_count = len(summary.get("items", [])) if summary else 0
            self.results["stage7"] = {
                "success": success,
                "planes": plane_count,
                "items": item_count,
            }

            return success, None, {
                "success": success,
                "planes": plane_count,
                "items": item_count,
            }

        except Exception as e:
            self._log(f"Stage 7 error: {e}", "error")
            import traceback
            traceback.print_exc()
            return False, None, {"error": str(e)}

    # ========================================================================
    # Stage 8: small-object descriptions (stage8_small_describe), optional
    # ========================================================================
    def run_stage8(self) -> Tuple[bool, Any, Dict]:
        """Run Stage 8 small-object descriptions for each Stage 7 object.

        Only runs when `config.detail_small_objects=True`. Reads
        Stage 7 `small_objects.json` + reference image, calls a VLM
        per-batch to produce object_type / appearance / material / color /
        part_hierarchy_hint records, and stores them under Memory stage
        `stage8_small_describe` so Stage 9 (small geometry) can consume.
        """
        self._print_stage_header(8)

        try:
            import sys
            sys.path.insert(0, str(CURRENT_DIR))
            from stage8_small_describe import StageSmallDescribeRunner

            sm, sbu, sak = self._resolve_stage_llm(
                self.config.stage8_small_describe_model,
                self.config.stage8_small_describe_base_url,
                self.config.stage8_small_describe_api_key,
            )
            self._log_stage_model(
                "Stage8",
                sm,
                sbu,
                dedicated=bool(self.config.stage8_small_describe_model),
            )

            runner = StageSmallDescribeRunner(
                image_path=self.config.image_path,
                small_objects_json_path=None,
                output_dir=os.path.join(
                    self.config.output_dir, "stage8_small_describe"
                ),
                use_memory=True,
                verbose=self.config.verbose,
                memory_file=self.config.memory_file,
                batch_size=self.config.small_describe_batch_size,
                parallel=self.config.small_describe_parallel,
                model=sm,
                base_url=sbu,
                api_key=sak,
            )

            success, summary = runner.run()
            item_count = (
                summary.get("total_items", 0)
                if isinstance(summary, dict)
                else 0
            )
            self.results["stage8"] = {
                "success": success,
                "items": item_count,
            }
            return success, None, {"success": success, "items": item_count}

        except Exception as e:
            self._log(f"Stage 8 error: {e}", "error")
            import traceback
            traceback.print_exc()
            return False, None, {"error": str(e)}

    # ========================================================================
    # Stage 9: detailed small-object geometry (stage9_small_geometry), optional
    # ========================================================================
    def run_stage9(self) -> Tuple[bool, Any, Dict]:
        """Run Stage 9 small-object geometry to generate <=5 primitives per Stage 8 object.

        Only runs when `config.detail_small_objects=True`. Reads the Stage
        Stage 8 describe JSON, the Stage 7 Blender script, calls the LLM
        per-item (parallel + retry, identical to Stage 6), then rewrites
        the Stage 7 small-object `create_box`/`create_cylinder` calls
        into `create_detailed_object_small` calls backed by an injected
        `DETAILED_GEOMETRY_SMALL` dict.
        """
        self._print_stage_header(9)

        try:
            import sys
            sys.path.insert(0, str(CURRENT_DIR))
            from stage9_small_geometry import StageSmallGeometryRunner

            sm, sbu, sak = self._resolve_stage_llm(
                self.config.stage9_small_geometry_model,
                self.config.stage9_small_geometry_base_url,
                self.config.stage9_small_geometry_api_key,
            )
            self._log_stage_model(
                "Stage9",
                sm,
                sbu,
                dedicated=bool(self.config.stage9_small_geometry_model),
            )

            runner = StageSmallGeometryRunner(
                image_path=self.config.image_path,
                describe_json_path=None,
                base_code_path=None,
                output_dir=os.path.join(
                    self.config.output_dir, "stage9_small_geometry"
                ),
                use_memory=True,
                verbose=self.config.verbose,
                memory_file=self.config.memory_file,
                parallel=self.config.small_geometry_parallel,
                max_attempts=self.config.stage8_geometry_max_attempts,
                retry_delay_sec=self.config.stage8_geometry_retry_delay,
                model=sm,
                base_url=sbu,
                api_key=sak,
            )

            success, summary = runner.run(resume=True, generate_code=True)
            total = (
                summary.get("total_items", 0)
                if isinstance(summary, dict)
                else 0
            )
            generated = (
                summary.get("generated_items", 0)
                if isinstance(summary, dict)
                else 0
            )
            self.results["stage9"] = {
                "success": success,
                "total": total,
                "generated": generated,
            }
            return success, None, {
                "success": success,
                "total": total,
                "generated": generated,
            }

        except Exception as e:
            self._log(f"Stage 9 error: {e}", "error")
            import traceback
            traceback.print_exc()
            return False, None, {"error": str(e)}

    # ========================================================================
    # Stage 10: material texture generation (stage10_material)
    # ========================================================================
    def run_stage10(self) -> Tuple[bool, Any, Dict]:
        """Run Stage 10: material texture generation with per-part PBR materials."""
        self._print_stage_header(10)

        try:
            import sys
            sys.path.insert(0, str(CURRENT_DIR))
            from stage10_material import StageMaterialRunner

            sm, sbu, sak = self._resolve_stage_llm(
                self.config.stage9_model,
                self.config.stage9_base_url,
                self.config.stage9_api_key,
            )
            self._log_stage_model(
                "Stage10",
                sm,
                sbu,
                dedicated=bool(self.config.stage9_model),
            )

            runner = StageMaterialRunner(
                image_path=self.config.image_path,
                geometry_code_path=None,
                output_dir=os.path.join(self.config.output_dir, "stage10_material"),
                use_memory=True,
                verbose=self.config.verbose,
                memory_file=self.config.memory_file,
                batch_size=self.config.material_batch_size,
                parallel=self.config.material_parallel,
                max_attempts=self.config.material_max_attempts,
                model=sm,
                base_url=sbu,
                api_key=sak,
            )

            success, code = runner.run()
            self.results["stage10"] = {"success": success}

            return success, code, {"success": success}

        except Exception as e:
            self._log(f"Stage 10 (material) error: {e}", "error")
            import traceback
            traceback.print_exc()
            return False, None, {"error": str(e)}

    # ========================================================================
    # Stage 11: real texture generation (stage11_texture, nanobanana)
    # ========================================================================
    def run_stage11(self) -> Tuple[bool, Any, Dict]:
        """Run Stage 11: generate wall/floor/art textures and inject them into code."""
        self._print_stage_header(11)

        try:
            import sys
            sys.path.insert(0, str(CURRENT_DIR))
            from stage11_texture import StageTextureRunner

            runner = StageTextureRunner(
                image_path=self.config.image_path,
                material_code_path=None,
                output_dir=os.path.join(self.config.output_dir, "stage11_texture"),
                use_memory=True,
                verbose=self.config.verbose,
                memory_file=self.config.memory_file,
                texture_model=self.config.stage10_model,
                texture_base_url=self.config.stage10_base_url,
                texture_api_key=self.config.stage10_api_key,
                image_size=self.config.stage10_image_size,
                parallel=self.config.stage10_parallel,
                max_wall_arts=self.config.stage10_max_wall_arts,
                wall_intensity=self.config.stage10_wall_intensity,
            )

            success, code_path = runner.run()
            self.results["stage11"] = {"success": success}

            return success, code_path, {"success": success}

        except Exception as e:
            self._log(f"Stage 11 (texture) error: {e}", "error")
            import traceback
            traceback.print_exc()
            return False, None, {"error": str(e)}

    # ========================================================================
    # Stage 12: render-ready script generation (stage12_render)
    # ========================================================================
    def run_stage12(self) -> Tuple[bool, Any, Dict]:
        """Run Stage 12: render-ready script generation with lighting and render settings."""
        self._print_stage_header(12)

        try:
            import sys
            sys.path.insert(0, str(CURRENT_DIR))
            from stage12_render import StageRenderRunner

            runner = StageRenderRunner(
                image_path=self.config.image_path,
                scene_code_path=None,
                output_dir=os.path.join(self.config.output_dir, "stage12_render"),
                use_memory=True,
                verbose=self.config.verbose,
                memory_file=self.config.memory_file,
                model=self.config.model,
                base_url=self.config.base_url,
                api_key=self.config.api_key,
            )

            success, code = runner.run()
            self.results["stage12"] = {"success": success}

            return success, code, {"success": success}

        except Exception as e:
            self._log(f"Stage 12 (render) error: {e}", "error")
            import traceback
            traceback.print_exc()
            return False, None, {"error": str(e)}

    # ========================================================================
    # Main run function
    # ========================================================================
    def run(self) -> Dict[str, Any]:
        """
        Run the full pipeline.
        
        Returns:
            Result dictionary containing:
            - success: whether all stages succeeded
            - stages: per-stage results
            - elapsed_seconds: elapsed wall-clock time
        """
        self.start_time = datetime.now()
        
        # Print pipeline information.
        print("\n")
        print("╔" + "═" * 58 + "╗")
        print("║" + " " * 58 + "║")
        print("║       🚀  UNIFIED PIPELINE - 3D Scene Generation  🚀    ║")
        print("║" + " " * 58 + "║")
        print("╚" + "═" * 58 + "╝")
        
        print(f"\n📁 Image: {self.config.image_path or '(read from Memory)'}")
        print(f"📂 Run directory: {self.config.output_dir}")
        print(f"💾 Memory: {self.config.memory_file}")
        print(f"🔄 Stages: Stage{self.config.start_stage} → Stage{self.config.end_stage}")
        print(f"🗜️ Image compression: {'enabled (target: ' + str(self.config.image_target_kb) + 'KB)' if self.config.compress_image else 'disabled'}")
        print(f"\n💡 Pipeline: Stage 1-4 base scene → Stage 5-12 descriptions/geometry/materials/textures/rendering")
        
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        self._save_run_config()
        
        # ====================================================================
        # Image preprocessing (compression)
        # ====================================================================
        if self.config.image_path and self.config.compress_image:
            self._print_header("🗜️ Image Preprocessing")
            self._preprocess_image()

        # ====================================================================
        # Scene classification (Step 0): choose Stage 1 / Stage 3 prompt routing.
        # Only run when start_stage <= 1 and classification is enabled; other
        # flows read the classification from Memory downstream.
        # ====================================================================
        if (
            self.config.image_path
            and self.config.scene_classify
            and self.config.start_stage <= 1
        ):
            self._print_header("🏷️ Scene Classification (Stage 0)")
            self._classify_scene()

        # Define the stage list.
        stages = [
            (1, self.run_stage1),
            (2, self.run_stage2),
            (3, self.run_stage3),
            (4, self.run_stage4),
            (5, self.run_stage5),
            (6, self.run_stage6),
            (7, self.run_stage7),
            (8, self.run_stage8),
            (9, self.run_stage9),
            (10, self.run_stage10),  # Material parameter generation (procedural)
            (11, self.run_stage11),  # Real texture generation (nanobanana)
            (12, self.run_stage12),  # Render-ready script generation
        ]
        
        success_count = 0
        total_stages = 0
        abort_pipeline = False
        stage3_abort_threshold = 0.05
        
        # Run the main flow.
        for stage_num, run_func in stages:
            if stage_num < self.config.start_stage:
                continue
            if stage_num > self.config.end_stage:
                continue

            # Stage 8 / 9 (detailed small objects) is opt-in via
            # --detail-small-objects. When the flag is off we silently skip
            # so existing CLI usage stays bit-for-bit identical.
            if stage_num in (8, 9) and not self.config.detail_small_objects:
                continue

            # Stage 11 (texture image generation) is opt-out via --skip-texture
            # (e.g. a Qwen-only run with no image-generation backend). Stage 12
            # still runs using Stage 10's procedural materials from Memory.
            if stage_num == 11 and not getattr(self.config, "stage11_enabled", True):
                self._log("Stage 11 (texture) skipped (--skip-texture)", "info")
                continue

            total_stages += 1
            
            try:
                result = run_func()
                success = result[0] if isinstance(result, tuple) else False
                stage_score = result[1] if stage_num == 3 and isinstance(result, tuple) and len(result) > 1 else None
                
                if success:
                    success_count += 1
                else:
                    self._log(f"Stage {stage_num} failed", "error")
                    if stage_num == 3 and (stage_score is None or stage_score < stage3_abort_threshold):
                        self._log(
                            f"Stage 3 score is too low ({stage_score or 0:.0%} < {stage3_abort_threshold:.0%}); "
                            "treating this as a severe code generation/rendering failure and aborting downstream stages",
                            "error",
                        )
                        abort_pipeline = True
                        break
                    if stage_num < self.config.end_stage:
                        self._log("Downstream stages may be affected", "warning")
            except Exception as e:
                self._log(f"Stage {stage_num} exception: {e}", "error")
                if stage_num == 3:
                    self._log("Stage 3 raised an exception; aborting downstream stages", "error")
                    abort_pipeline = True
                    break
        
        # Compute elapsed time.
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        # Print result summary.
        self._print_summary(elapsed, success_count, total_stages)
        
        return {
            "success": success_count == total_stages,
            "stages": self.results,
            "elapsed_seconds": elapsed,
            "success_count": success_count,
            "total_stages": total_stages
        }
    
    def _print_summary(self, elapsed: float, success_count: int, total_stages: int):
        """Print the result summary."""
        print("\n")
        print("╔" + "═" * 58 + "╗")
        print("║                    📊  PIPELINE RESULTS                  ║")
        print("╠" + "═" * 58 + "╣")
        
        for stage_num in sorted(self.STAGE_INFO):
            if stage_num < self.config.start_stage or stage_num > self.config.end_stage:
                continue
            if stage_num in (8, 9) and not self.config.detail_small_objects:
                continue

            info = self.STAGE_INFO.get(stage_num, {})
            result = self.results.get(f"stage{stage_num}", {})
            status = "✅" if result.get("success") else "❌"
            extra = ""

            if stage_num == 3 and "score" in result:
                extra = f" (score: {result['score']:.0%})"
            elif stage_num == 6 and "generated" in result:
                extra = f" (generated: {result['generated']})"
            elif stage_num == 7 and ("planes" in result or "items" in result):
                extra = f" ({result.get('planes') or 0} planes/{result.get('items') or 0} items)"
            elif stage_num == 8 and "items" in result:
                extra = f" ({result['items']} items)"
            elif stage_num == 9 and "total" in result:
                extra = f" ({result.get('generated') or 0}/{result['total']})"

            name = info.get("name", f"Stage{stage_num}")
            desc = info.get("desc", "")
            print(f"║   {status} {name}: {desc}{extra:20}║")

        print("╠" + "═" * 58 + "╣")
        print(f"║   ⏱️  Elapsed: {elapsed:.1f} sec                              ║")
        print(f"║   📈 Success: {success_count}/{total_stages} stages                            ║")
        print("╚" + "═" * 58 + "╝")


# ============================================================================
# Memory status helpers
# ============================================================================
def _resolve_memory_file(run_dir: str = None) -> str:
    """Resolve memory file path from run_dir.

    Returns a path relative to CURRENT_DIR when run_dir lives under
    agent_utils/ (default layout), otherwise returns an absolute path.
    Memory.__init__ accepts both because `Path(workspace) / abs_path`
    yields the absolute path verbatim.
    """
    if run_dir:
        run_path = Path(run_dir)
        if not run_path.is_absolute():
            run_path = CURRENT_DIR / run_path
        run_path = run_path.resolve()
        try:
            return str(run_path.relative_to(CURRENT_DIR) / "agent_memory.jsonl")
        except ValueError:
            return str(run_path / "agent_memory.jsonl")
    return "agent_memory.jsonl"


def show_memory_status(run_dir: str = None):
    """Show Memory status."""
    memory_file = _resolve_memory_file(run_dir)
    memory = Memory(workspace_dir=str(CURRENT_DIR), memory_file=memory_file)

    print("╔" + "═" * 58 + "╗")
    if run_dir:
        print(f"║  📋  MEMORY STATUS ({Path(run_dir).name:^30}) ║")
    else:
        print("║                    📋  MEMORY STATUS                    ║")
    print("╠" + "═" * 58 + "╣")

    all_stages = [
        "stage1", "stage2", "stage3", "stage4",
        "stage5_describe", "stage6_geometry", "stage7_small_objects",
        "stage8_small_describe", "stage9_small_geometry",
        "stage10_material", "stage11_texture", "stage12_render",
    ]

    for stage in all_stages:
        entry = memory.get_latest(stage=stage, type="result")
        if entry:
            title = entry.metadata.get("title", "untitled")[:35]
            time_str = datetime.fromtimestamp(entry.timestamp).strftime("%m-%d %H:%M")
            print(f"║   ✅ {stage:18}: {title:20} ({time_str}) ║")
        else:
            print(f"║   ❌ {stage:18}: {'no data':20}           ║")

    print("╚" + "═" * 58 + "╝")


def list_all_runs():
    """List all run records under pipeline_output/."""
    output_root = CURRENT_DIR / "pipeline_output"
    if not output_root.exists():
        print("📂 pipeline_output/ does not exist; no runs yet")
        return
    
    run_dirs = sorted(
        [d for d in output_root.iterdir() if d.is_dir() and d.name.startswith("run_")],
        key=lambda d: d.name,
        reverse=True
    )
    
    if not run_dirs:
        print("📂 No run_* records found under pipeline_output/")
        return
    
    print("╔" + "═" * 72 + "╗")
    print("║                         📂  ALL RUNS                                ║")
    print("╠" + "═" * 72 + "╣")
    
    for d in run_dirs:
        config_file = d / "run_config.json"
        memory_file_path = d / "agent_memory.jsonl"
        
        stages_done = []
        if memory_file_path.exists():
            mem_rel = str(d.relative_to(CURRENT_DIR) / "agent_memory.jsonl")
            mem = Memory(workspace_dir=str(CURRENT_DIR), memory_file=mem_rel)
            for s in [
                "stage1", "stage2", "stage3", "stage4",
                "stage5_describe", "stage6_geometry", "stage7_small_objects",
                "stage8_small_describe", "stage9_small_geometry",
                "stage10_material", "stage11_texture", "stage12_render",
            ]:
                if mem.get_latest(stage=s, type="result"):
                    short = s.replace("stage_", "S").replace("stage", "S")
                    stages_done.append(short)
        
        image_name = ""
        if config_file.exists():
            try:
                with open(config_file, "r") as f:
                    cfg = json.load(f)
                img = cfg.get("image_path", "")
                if img:
                    image_name = Path(img).name
            except Exception:
                pass
        
        stages_str = ",".join(stages_done) if stages_done else "empty"
        line = f"  {d.name}  [{stages_str}]"
        if image_name:
            line += f"  img={image_name}"
        print(f"║ {line:70} ║")
    
    print("╠" + "═" * 72 + "╣")
    print(f"║  Total: {len(run_dirs)} runs{' ' * 57}║")
    print("╚" + "═" * 72 + "╝")
    print(f"\n💡 Show a run: python run_pipeline.py --status --run-dir pipeline_output/<run_name>")
    print(f"💡 Resume a run: python run_pipeline.py --image img.png --start 3 --run-dir pipeline_output/<run_name>")


def clear_memory(run_dir: str = None):
    """Clear Memory."""
    memory_file = _resolve_memory_file(run_dir)
    memory = Memory(workspace_dir=str(CURRENT_DIR), memory_file=memory_file)
    memory.clear()
    if run_dir:
        print(f"✅ Memory cleared for {run_dir}")
    else:
        print("✅ Memory cleared")


def clear_stage_memory(stage: str, run_dir: str = None):
    """Clear Memory for a specific stage."""
    memory_file = _resolve_memory_file(run_dir)
    memory = Memory(workspace_dir=str(CURRENT_DIR), memory_file=memory_file)
    memory.clear(stage=stage)
    print(f"✅ {stage} Memory cleared")


# ============================================================================
# Convenience run helpers
# ============================================================================
def run_full_pipeline(
    image_path: str,
    output_dir: str = None,
    run_dir: str = None,
    start_stage: int = 1,
    end_stage: int = 12,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience helper for running the full pipeline.
    
    Args:
        image_path: Input image path.
        output_dir: Output directory.
        run_dir: Existing run directory for resuming from an intermediate stage.
        start_stage: Starting stage (1-12).
        end_stage: Ending stage (1-12).
        **kwargs: Additional configuration parameters.
    
    Returns:
        Pipeline run result.
    """
    config = PipelineConfig(
        image_path=image_path,
        output_dir=output_dir,
        run_dir=run_dir,
        start_stage=start_stage,
        end_stage=end_stage,
        **kwargs
    )
    
    pipeline = UnifiedPipeline(config)
    return pipeline.run()


def run_stage1_only(image_path: str, output_dir: str = None, **kwargs) -> Dict:
    """Run Stage 1 only."""
    return run_full_pipeline(image_path, output_dir, start_stage=1, end_stage=1, **kwargs)


def run_stage1_to_4(image_path: str, output_dir: str = None, **kwargs) -> Dict:
    """Run Stage 1 through Stage 4."""
    return run_full_pipeline(image_path, output_dir, start_stage=1, end_stage=4, **kwargs)


# ============================================================================
# CLI config-file helpers
# ============================================================================
_CONFIG_KEY_ALIASES = {
    # Public stage numbers in the CLI differ from a few historical internal
    # config slots. Accept both forms in JSON files.
    "stage5_model": "stage7_model",
    "stage5_base_url": "stage7_base_url",
    "stage5_api_key": "stage7_api_key",
    "stage6_model": "stage8_model",
    "stage6_base_url": "stage8_base_url",
    "stage6_api_key": "stage8_api_key",
    "stage10_material_model": "stage9_model",
    "stage10_material_base_url": "stage9_base_url",
    "stage10_material_api_key": "stage9_api_key",
    "texture_model": "stage11_texture_model",
    "texture_base_url": "stage11_texture_base_url",
    "texture_api_key": "stage11_texture_api_key",
    "texture_image_size": "stage11_texture_image_size",
    "texture_parallel": "stage11_texture_parallel",
    "texture_max_wall_arts": "stage11_texture_max_wall_arts",
}


def _load_cli_config_file(config_path: str, parser: argparse.ArgumentParser) -> Dict[str, Any]:
    """Load argparse defaults from a JSON config file.

    Keys should match CLI option names without leading dashes, using
    underscores instead of hyphens; e.g. ``base_url`` for ``--base-url``.
    """
    path = Path(config_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        parser.error(f"Config file does not exist: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except json.JSONDecodeError as exc:
        parser.error(f"Invalid JSON config file {path}: {exc}")
    except OSError as exc:
        parser.error(f"Failed to read config file {path}: {exc}")

    if not isinstance(raw, dict):
        parser.error(f"Config file must contain a JSON object: {path}")

    valid_dests = {
        action.dest
        for action in parser._actions
        if action.dest and action.dest != argparse.SUPPRESS
    }
    valid_dests.discard("help")
    defaults: Dict[str, Any] = {}
    unknown: List[str] = []

    for key, value in raw.items():
        normalized = str(key).strip().lstrip("-").replace("-", "_")
        dest = _CONFIG_KEY_ALIASES.get(normalized, normalized)
        if dest not in valid_dests:
            unknown.append(str(key))
            continue
        defaults[dest] = value

    if unknown:
        known = ", ".join(sorted(valid_dests | set(_CONFIG_KEY_ALIASES)))
        parser.error(
            "Unknown config key(s): "
            + ", ".join(sorted(unknown))
            + f"\nKnown keys include: {known}"
        )

    return defaults


# ============================================================================
# Command-line entry point
# ============================================================================
def main():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args()

    parser = argparse.ArgumentParser(
        description="Unified Pipeline - run all stages from one command",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with a JSON config file
  python run_pipeline.py --config pipeline_config.json

  # Run the full pipeline (Stage 1-12)
  # Default output: <image_dir>/run_YYYYMMDD_HHMMSS_<image>/
  python run_pipeline.py --image input.png

  # Override output parent directory
  python run_pipeline.py --image input.png --output-dir /path/to/output_root

  # Run only Stage 1-4 (base scene generation)
  python run_pipeline.py --image input.png --end 4

  # Resume from Stage 3 using a previous run directory and Memory
  python run_pipeline.py --image input.png --start 3 \\
      --run-dir pipeline_output/run_20260323_143022_input

  # Run Stage 5-12 based on a previous run
  python run_pipeline.py --image input.png --start 5 \\
      --run-dir pipeline_output/run_20260323_143022_input

  # Run Stage 3 with an OpenAI Codex model
  python run_pipeline.py --image input.png --stage3-model gpt-5.1-codex-max

  # Show Memory status for a run
  python run_pipeline.py --status --run-dir pipeline_output/run_20260323_143022_input

  # List all runs
  python run_pipeline.py --list-runs

  # Clear Memory
  python run_pipeline.py --clear-memory

Stage guide:
  Stage 1-4: Base scene generation (spatial analysis → Scene Graph → Blender code → small objects)
  Stage 5: Object descriptions (type / appearance / material / color for each object)
  Stage 6: Detailed geometry (replace simple bounding boxes with composite geometry)
  Stage 7: Surface small-object placement
  Stage 8: Detailed small-object descriptions (optional)
  Stage 9: Detailed small-object geometry (optional)
  Stage 10: Material texture generation (per-part PBR materials plus floor/wall materials)
  Stage 11: Real texture generation (nanobanana floor/wall/art textures and code injection)
  Stage 12: Render-ready script generation (lighting + render settings)

Recommended flow: Stage 1-4 → Stage 5-12
"""
    )
    
    # Basic arguments
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "JSON config file containing the same options as the CLI. Use "
            "underscore keys, e.g. image, output_dir, base_url, stage5_model. "
            "Explicit CLI arguments override values from this file."
        ),
    )
    parser.add_argument("--image", "-i", help="Input image path")
    parser.add_argument("--output-dir", "-o", default=None,
                        help=(
                            "Custom output directory. If the path name starts with run_, "
                            "it is used directly; otherwise a run_<timestamp>_<image>/ "
                            "subdirectory is created inside it. Default: create the run "
                            "directory next to the input image."
                        ))
    parser.add_argument("--run-dir", "-r", default=None,
                        help="Existing run directory for resuming from an intermediate stage (reads Memory from that directory)")
    parser.add_argument("--start", type=int, default=1, choices=list(range(1, 13)), help="Starting stage (1-12)")
    parser.add_argument("--end", type=int, default=12, choices=list(range(1, 13)), help="Ending stage (1-12)")
    parser.add_argument("--max-iter", "-n", type=int, default=5, help="Maximum iterations per stage")

    # Global LLM configuration shared by all stages unless overridden.
    parser.add_argument("--model", type=str, default=None,
                        help="Global LLM model name (overrides default)")
    parser.add_argument("--base-url", type=str, default=None,
                        help="Global API base URL (overrides default)")
    parser.add_argument("--api-key", type=str, default=None,
                        help="Global API key (overrides default)")
    parser.add_argument(
        "--stage2-max-tokens",
        type=int,
        default=65536,
        help="Stage2: max completion tokens (same role as max_tokens in Chat Completions API)",
    )

    # Stage 3 arguments
    parser.add_argument("--no-iterate", action="store_true", help="Disable Stage 3 iteration")
    parser.add_argument("--skip-texture", action="store_true",
                        help="Skip Stage 11 (texture image generation); Stage 12 still runs using Stage 10 procedural materials")
    parser.add_argument("--target", "-t", type=float, default=0.85, help="Stage 3 target score")
    parser.add_argument("--blender", "-b", 
                        default="/Applications/Blender.app/Contents/MacOS/Blender",
                        help="Blender executable path")
    parser.add_argument("--stage1-model", type=str, default=None,
                        help="Stage 1 dedicated model (default: gemini-3.1-pro-preview)")
    parser.add_argument("--stage1-base-url", type=str, default=None,
                        help="Stage 1 dedicated API base URL")
    parser.add_argument("--stage1-api-key", type=str, default=None,
                        help="Stage 1 dedicated API key")
    parser.add_argument("--stage2-model", type=str, default=None,
                        help="Stage 2 dedicated model")
    parser.add_argument("--stage2-base-url", type=str, default=None,
                        help="Stage 2 dedicated API base URL")
    parser.add_argument("--stage2-api-key", type=str, default=None,
                        help="Stage 2 dedicated API key")
    parser.add_argument("--stage3-model", type=str, default=None,
                        help="Stage 3 dedicated model (e.g. gpt-5.1-codex-max)")
    parser.add_argument("--stage3-base-url", type=str, default=None,
                        help="Stage 3 dedicated API base URL (e.g. https://api.openai.com/v1)")
    parser.add_argument("--stage3-api-key", type=str, default=None,
                        help="Stage 3 dedicated API key (or set OPENAI_API_KEY)")

    parser.add_argument(
        "--stage5-model",
        dest="stage7_model",
        metavar="STAGE5_MODEL",
        type=str,
        default=None,
        help="Stage 5 (describe) dedicated model (default: reuse global --model)",
    )
    parser.add_argument(
        "--stage5-base-url",
        dest="stage7_base_url",
        metavar="STAGE5_BASE_URL",
        type=str,
        default=None,
        help="Stage 5 dedicated API base URL (default: reuse global --base-url)",
    )
    parser.add_argument(
        "--stage5-api-key",
        dest="stage7_api_key",
        metavar="STAGE5_API_KEY",
        type=str,
        default=None,
        help="Stage 5 dedicated API key (default: reuse global --api-key)",
    )
    parser.add_argument(
        "--stage6-model",
        dest="stage8_model",
        metavar="STAGE6_MODEL",
        type=str,
        default=None,
        help="Stage 6 (geometry) dedicated model (default: reuse global --model)",
    )
    parser.add_argument(
        "--stage6-base-url",
        dest="stage8_base_url",
        metavar="STAGE6_BASE_URL",
        type=str,
        default=None,
        help="Stage 6 dedicated API base URL",
    )
    parser.add_argument(
        "--stage6-api-key",
        dest="stage8_api_key",
        metavar="STAGE6_API_KEY",
        type=str,
        default=None,
        help="Stage 6 dedicated API key",
    )
    parser.add_argument(
        "--stage7-small-objects-model",
        dest="stage7_small_objects_model",
        metavar="STAGE7_SMALL_OBJECTS_MODEL",
        type=str,
        default=None,
        help="Stage 7 (small_objects) dedicated model",
    )
    parser.add_argument(
        "--stage7-small-objects-base-url",
        dest="stage7_small_objects_base_url",
        metavar="STAGE7_SMALL_OBJECTS_BASE_URL",
        type=str,
        default=None,
        help="Stage 7 dedicated API base URL",
    )
    parser.add_argument(
        "--stage7-small-objects-api-key",
        dest="stage7_small_objects_api_key",
        metavar="STAGE7_SMALL_OBJECTS_API_KEY",
        type=str,
        default=None,
        help="Stage 7 dedicated API key",
    )
    parser.add_argument(
        "--stage8-small-describe-model",
        dest="stage8_small_describe_model",
        metavar="STAGE8_SMALL_DESCRIBE_MODEL",
        type=str,
        default=None,
        help="Stage 8 (small_describe) dedicated model",
    )
    parser.add_argument(
        "--stage8-small-describe-base-url",
        dest="stage8_small_describe_base_url",
        metavar="STAGE8_SMALL_DESCRIBE_BASE_URL",
        type=str,
        default=None,
        help="Stage 8 dedicated API base URL",
    )
    parser.add_argument(
        "--stage8-small-describe-api-key",
        dest="stage8_small_describe_api_key",
        metavar="STAGE8_SMALL_DESCRIBE_API_KEY",
        type=str,
        default=None,
        help="Stage 8 dedicated API key",
    )
    parser.add_argument(
        "--stage9-small-geometry-model",
        dest="stage9_small_geometry_model",
        metavar="STAGE9_SMALL_GEOMETRY_MODEL",
        type=str,
        default=None,
        help="Stage 9 (small_geometry) dedicated model",
    )
    parser.add_argument(
        "--stage9-small-geometry-base-url",
        dest="stage9_small_geometry_base_url",
        metavar="STAGE9_SMALL_GEOMETRY_BASE_URL",
        type=str,
        default=None,
        help="Stage 9 dedicated API base URL",
    )
    parser.add_argument(
        "--stage9-small-geometry-api-key",
        dest="stage9_small_geometry_api_key",
        metavar="STAGE9_SMALL_GEOMETRY_API_KEY",
        type=str,
        default=None,
        help="Stage 9 dedicated API key",
    )
    parser.add_argument(
        "--stage10-material-model",
        dest="stage9_model",
        metavar="STAGE10_MATERIAL_MODEL",
        type=str,
        default=None,
        help="Stage 10 (material) dedicated model",
    )
    parser.add_argument(
        "--stage10-material-base-url",
        dest="stage9_base_url",
        metavar="STAGE10_MATERIAL_BASE_URL",
        type=str,
        default=None,
        help="Stage 10 dedicated API base URL",
    )
    parser.add_argument(
        "--stage10-material-api-key",
        dest="stage9_api_key",
        metavar="STAGE10_MATERIAL_API_KEY",
        type=str,
        default=None,
        help="Stage 10 dedicated API key",
    )
    parser.add_argument(
        "--stage11-texture-model",
        dest="stage11_texture_model",
        type=str,
        default=None,
        help="Stage 11 image-texture model",
    )
    parser.add_argument(
        "--stage11-texture-base-url",
        dest="stage11_texture_base_url",
        type=str,
        default=None,
        help="Stage 11 image-texture API base URL",
    )
    parser.add_argument(
        "--stage11-texture-api-key",
        dest="stage11_texture_api_key",
        type=str,
        default=None,
        help="Stage 11 image-texture API key",
    )
    parser.add_argument(
        "--stage11-texture-image-size",
        dest="stage11_texture_image_size",
        type=str,
        default="1K",
        help="Stage 11 generated texture image size (default: 1K)",
    )
    parser.add_argument(
        "--stage11-texture-parallel",
        dest="stage11_texture_parallel",
        type=int,
        default=8,
        help="Stage 11 parallel texture workers (default: 8)",
    )
    parser.add_argument(
        "--stage11-texture-max-wall-arts",
        dest="stage11_texture_max_wall_arts",
        type=int,
        default=20,
        help="Stage 11 maximum wall-art textures to generate (default: 20)",
    )
    
    # Stage 3 rotation arguments
    # Rotation validation/correction is disabled by default because results are weak.
    # Use --enable-rotation to turn it on when needed.
    parser.add_argument("--enable-rotation", action="store_true",
                        help="Enable rotation correction after Stage 3 (disabled by default)")
    parser.add_argument("--no-rotation", action="store_true",
                        help="[deprecated] rotation correction is already disabled by default; kept for compatibility")
    parser.add_argument("--rotation-iter", type=int, default=3,
                        help="Maximum rotation-correction iterations (default=3)")
    parser.add_argument("--rotation-target", type=float, default=0.85,
                        help="Rotation-correction target score (default=0.85)")
    
    # Stage 6 geometry arguments
    parser.add_argument("--parallel", type=int, default=8, help="Stage 6 geometry parallel workers (default=8)")
    parser.add_argument(
        "--geometry-max-attempts",
        type=int,
        default=3,
        help="Stage 6: max LLM attempts per object on failure (default=3)",
    )
    parser.add_argument(
        "--geometry-retry-delay",
        type=float,
        default=2.0,
        help="Stage 6: seconds between retries (default=2.0)",
    )

    # Stage 8 / 9 (detailed small objects, optional)
    parser.add_argument(
        "--detail-small-objects",
        action="store_true",
        help=(
            "Enable per-item detailed geometry for Stage 7 small objects: "
            "runs Stage 8 (LLM description) and Stage 9 (composite geometry) "
            "before Stage 10 (material). Strategy = all (every Stage-7 item "
            "is described and detailed). OFF by default."
        ),
    )
    parser.add_argument(
        "--small-describe-batch-size",
        type=int,
        default=8,
        help=(
            "Stage 8: small objects per LLM call (default 8). Smaller "
            "batches → more LLM calls but lower truncation risk."
        ),
    )
    parser.add_argument(
        "--small-describe-parallel",
        type=int,
        default=4,
        help=(
            "Stage 8: parallel batch workers (default 4). Similar to "
            "Stage 6's --parallel, capped to avoid endpoint rate limits."
        ),
    )
    parser.add_argument(
        "--small-geometry-parallel",
        type=int,
        default=8,
        help=(
            "Stage 9: per-item parallel workers (default 8). Mirrors "
            "Stage 6 --parallel. Each item is one LLM call (≤5 primitives)."
        ),
    )

    # Stage 10 (material) Option-C batching
    parser.add_argument(
        "--material-batch-size",
        type=int,
        default=6,
        help=(
            "Stage 10 Pass 2: objects per LLM call (default 6). Smaller "
            "batches reduce truncation risk; larger batches improve "
            "cross-object consistency within a batch."
        ),
    )
    parser.add_argument(
        "--material-parallel",
        type=int,
        default=4,
        help=(
            "Stage 10 Pass 2: parallel batch workers (default 4). "
            "Capped to avoid endpoint rate limits."
        ),
    )
    parser.add_argument(
        "--material-max-attempts",
        type=int,
        default=3,
        help=(
            "Stage 10: max retries per batch when LLM returns incomplete "
            "part coverage (default 3)."
        ),
    )

    # Stage 11 (stage11_texture) arguments
    parser.add_argument(
        "--wall-intensity",
        type=str,
        default="subtle",
        choices=["subtle", "bold", "mural_like"],
        help=(
            "Stage 11 wall visual intensity: subtle=clean wall (default), "
            "bold=decorative patterned wall, mural_like=central emblem mural. "
            "If unspecified, stage11_texture decides via its fallback chain "
            "(WALL_MAT_DATA -> material_config -> subtle)."
        ),
    )

    # Image compression arguments
    parser.add_argument("--no-compress", action="store_true", help="Disable image compression")
    parser.add_argument("--image-target-kb", type=int, default=500, help="Target image size in KB")

    # Scene classification arguments for Stage 1 / Stage 3 prompt routing
    parser.add_argument(
        "--no-scene-classify", action="store_true",
        help="Disable automatic Stage 0 scene classification (Stage 1/3 use the base prompt)",
    )
    parser.add_argument(
        "--scene-type", type=str, default=None,
        choices=["lab", "residential", "office", "industrial", "retail", "other"],
        help="Manually set scene type and skip automatic classification (e.g. lab forces the lab prompt variant)",
    )
    parser.add_argument(
        "--scene-classify-model", type=str, default=None,
        help="Scene classifier dedicated model (default: reuse --stage1-model or global --model)",
    )
    
    # Memory operations
    parser.add_argument("--status", "-s", action="store_true", help="Show Memory status (use with --run-dir for a specific run)")
    parser.add_argument("--clear-memory", action="store_true", help="Clear all Memory")
    parser.add_argument("--clear-stage", type=str, help="Clear Memory for a specific stage")
    parser.add_argument("--list-runs", action="store_true", help="List all run records")
    
    # Other
    parser.add_argument("--quiet", "-q", action="store_true", help="Quiet mode")
    
    if pre_args.config:
        parser.set_defaults(**_load_cli_config_file(pre_args.config, parser))

    args = parser.parse_args()
    
    # List all runs
    if args.list_runs:
        list_all_runs()
        return 0
    
    # Memory operations
    if args.status:
        show_memory_status(run_dir=args.run_dir)
        return 0
    
    if args.clear_memory:
        clear_memory(run_dir=args.run_dir)
        return 0
    
    if args.clear_stage:
        clear_stage_memory(args.clear_stage, run_dir=args.run_dir)
        return 0
    
    # Validate arguments.
    if not args.image and args.start == 1:
        parser.error("--image is required, or use --start 2+ with --run-dir to resume from an existing run")
    
    if args.image and not os.path.exists(args.image):
        print(f"❌ Image does not exist: {args.image}")
        return 1
    
    if args.start > args.end:
        print(f"❌ Starting stage ({args.start}) cannot be greater than ending stage ({args.end})")
        return 1
    
    if args.start > 1 and not args.run_dir:
        print(f"⚠️ Starting from Stage {args.start} without --run-dir; a new run directory will be created with no prior Memory")
    
    # Create configuration.
    #
    # Forwarding rule for *all* CLI-overridable LLM args:
    #   only pass through to PipelineConfig when the user actually set it
    #   on the command line (i.e. argparse value is not None).
    #
    # Why: PipelineConfig.__init__ has meaningful per-stage defaults
    #   (e.g. stage2_model="gemini-3-flash-preview", stage2_base_url
    #   pinned to us.novaiapi.com). argparse defaults each --stageN-xxx
    #   to None, and Python's "default value" only applies when an arg
    #   is OMITTED — passing None EXPLICITLY overrides it. So the naive
    #   `stage2_model=args.stage2_model` was silently nuking the
    #   constructor default to None, which then fell back to the global
    #   --model in run_stageN(). That defeated the whole point of having
    #   per-stage defaults. We now mirror the proven pattern used for
    #   the global --model/--base-url/--api-key below.
    extra_kw = {}
    if args.model:
        extra_kw["model"] = args.model
    if args.base_url:
        extra_kw["base_url"] = args.base_url
    if args.api_key:
        extra_kw["api_key"] = args.api_key

    optional_overrides = {
        "stage1_model":         args.stage1_model,
        "stage1_base_url":      args.stage1_base_url,
        "stage1_api_key":       args.stage1_api_key,
        "stage2_model":         args.stage2_model,
        "stage2_base_url":      args.stage2_base_url,
        "stage2_api_key":       args.stage2_api_key,
        "stage3_model":         args.stage3_model,
        "stage3_base_url":      args.stage3_base_url,
        "stage3_api_key":       args.stage3_api_key,
        "stage7_model":         args.stage7_model,
        "stage7_base_url":      args.stage7_base_url,
        "stage7_api_key":       args.stage7_api_key,
        "stage8_model":         args.stage8_model,
        "stage8_base_url":      args.stage8_base_url,
        "stage8_api_key":       args.stage8_api_key,
        "stage7_small_objects_model":    args.stage7_small_objects_model,
        "stage7_small_objects_base_url": args.stage7_small_objects_base_url,
        "stage7_small_objects_api_key":  args.stage7_small_objects_api_key,
        "stage8_small_describe_model":    args.stage8_small_describe_model,
        "stage8_small_describe_base_url": args.stage8_small_describe_base_url,
        "stage8_small_describe_api_key":  args.stage8_small_describe_api_key,
        "stage9_small_geometry_model":    args.stage9_small_geometry_model,
        "stage9_small_geometry_base_url": args.stage9_small_geometry_base_url,
        "stage9_small_geometry_api_key":  args.stage9_small_geometry_api_key,
        "stage9_model":         args.stage9_model,
        "stage9_base_url":      args.stage9_base_url,
        "stage9_api_key":       args.stage9_api_key,
        "stage11_texture_model":         args.stage11_texture_model,
        "stage11_texture_base_url":      args.stage11_texture_base_url,
        "stage11_texture_api_key":       args.stage11_texture_api_key,
        "scene_classify_model": args.scene_classify_model,
    }
    for _k, _v in optional_overrides.items():
        if _v is not None:
            extra_kw[_k] = _v

    config = PipelineConfig(
        image_path=args.image,
        output_dir=args.output_dir,
        run_dir=args.run_dir,
        max_iterations=args.max_iter,
        stage3_iterate=not args.no_iterate,
        stage3_target_score=args.target,
        blender_path=args.blender,
        stage3_rotation=args.enable_rotation and not args.no_rotation,
        stage3_rotation_iterations=args.rotation_iter,
        stage3_rotation_target=args.rotation_target,
        stage8_parallel=args.parallel,
        stage8_geometry_max_attempts=args.geometry_max_attempts,
        stage8_geometry_retry_delay=args.geometry_retry_delay,
        detail_small_objects=args.detail_small_objects,
        small_describe_batch_size=args.small_describe_batch_size,
        small_describe_parallel=args.small_describe_parallel,
        small_geometry_parallel=args.small_geometry_parallel,
        material_batch_size=args.material_batch_size,
        material_parallel=args.material_parallel,
        material_max_attempts=args.material_max_attempts,
        stage11_texture_image_size=args.stage11_texture_image_size,
        stage11_texture_parallel=args.stage11_texture_parallel,
        stage11_texture_max_wall_arts=args.stage11_texture_max_wall_arts,
        verbose=not args.quiet,
        start_stage=args.start,
        end_stage=args.end,
        compress_image=not args.no_compress,
        image_target_kb=args.image_target_kb,
        scene_classify=not args.no_scene_classify,
        scene_type_override=args.scene_type,
        **extra_kw
    )

    # Stage 11 (stage11_texture) wall-intensity override.
    if args.wall_intensity:
        config.stage10_wall_intensity = args.wall_intensity

    # Skip Stage 11 (texture image generation) when requested, e.g. a Qwen-only
    # run with no image-generation backend. Stage 12 still runs using Stage 10's
    # procedural materials from Memory.
    if args.skip_texture:
        config.stage11_enabled = False

    # Run the pipeline.
    pipeline = UnifiedPipeline(config)
    result = pipeline.run()
    
    return 0 if result["success"] else 1


if __name__ == "__main__":
    # Force UTF-8 mode on Windows. LLM-generated Blender code often contains
    # non-ASCII characters (e.g. the degree sign '°' that Qwen and other models
    # emit in comments like "rotate 90°"). Without UTF-8 mode, open(..., "w")
    # uses the Windows locale codec (GBK/cp936) and writes '°' as byte 0xA1;
    # Blender's embedded Python then reads the file as UTF-8 and raises
    # UnicodeDecodeError at render time, failing every Stage 3 render. Re-exec
    # once under `-X utf8` so all file writes are UTF-8 (Blender reads UTF-8 by
    # default). No-op if already in UTF-8 mode or on non-Windows platforms.
    if sys.platform == "win32" and not sys.flags.utf8_mode:
        os.environ["PYTHONUTF8"] = "1"
        os.environ.setdefault("PYTHONIOENCODING", "utf-8")
        os.execv(sys.executable, [sys.executable, "-X", "utf8", *sys.argv])
    exit(main())
