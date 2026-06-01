"""
Batch prompt generator (and optional image generator) for top-down interior renders.
- Text mode: LLM generates diverse parameter combos -> fills prompt template -> saves JSON
- Image mode: takes generated prompts and calls image generation API to produce images

python image_prompt_gen/topdown_room_image_generator.py all \
  --api-key "$SCENEGEN_TEXTURE_API_KEY" \
  --base-url "$SCENEGEN_TEXTURE_BASE_URL" \
  --image-model gemini-3-pro-image-preview \
  --count 20 \
  --prompt-model gpt-4o \
  --scene-scope non_residential
"""

import os
import re
import json
import time
import base64
import argparse
import math
from datetime import datetime
from typing import Optional
from pathlib import Path
PROMPT_TEMPLATE = (
    "Top-down orthographic view of a richly furnished {room_type}, camera at 90 degrees straight down. "
    "Designed for {user_type}. "
    "Complete room visible, centered, all walls shown, plain white background. "
    "Architectural structure must be explicit and readable from above: "
    "perimeter wall layout clearly defined, "
    "door openings clearly visible as distinct gaps or door slabs on walls, "
    "windows clearly visible as glazed strips or panels on exterior walls, "
    "wall thickness and corners crisp, no fused or ambiguous wall masses. "
    "Main furniture: {key_furniture}. "
    "Decorations and accessories: {decorations}. "
    "Zones: {zones}. "
    "The room should feel lived-in and realistically dense with objects, "
    "every functional zone should have appropriate furniture AND small accessories on surfaces. "
    "{style} style, {palette} palette, {floor_material} flooring. "
    "Room shape: {room_shape}. Prefer a clean rectangular or near-rectangular floor plan footprint with mostly orthogonal walls. "
    "Photorealistic textures, flat overhead view, no perspective, no isometric, no 3D angle. "
    "No text, no labels, no legends, no annotations, no dimensions, no arrows, no icons, no watermarks."
)

PROMPT_TEMPLATE_LAB = (
    "Top-down orthographic view of a {room_type}, camera at 90 degrees straight down. "
    "Designed for {user_type}. "
    "Complete room visible, centered, all walls shown, plain white background. "
    "Architectural structure must be explicit and readable from above: "
    "perimeter wall layout clearly defined, "
    "door openings clearly visible as distinct gaps or door slabs on walls, "
    "windows clearly visible as glazed strips or panels on exterior walls, "
    "wall thickness and corners crisp, no fused or ambiguous wall masses. "
    "Furniture (use ONLY these pieces, do NOT invent extra benches, desks or workstations): {key_furniture}. "
    "Bench-top items, including but not limited to: {decorations}. "
    "Functional zones: {zones}. "
    "Each major bench or workstation has 1 stool or chair tucked nearby. "
    "Bench tops carry a moderate, working amount of glassware, tools, notebooks, "
    "monitors, cables and small instruments that researchers actually use day-to-day; "
    "the lab is lived-in and actively in use. "
    "Avoid BOTH extremes: do NOT show a sterile empty showroom, "
    "and also do NOT pile dozens of unrelated items on every surface. "
    "Keep clear walking aisles between furniture. "
    "Do NOT add homely / residential accessories: no plants, no picture frames, no rugs, no throw pillows, "
    "no decorative book stacks, no candles, no decorative trays, no vases. "
    "{style} style, {palette} palette, {floor_material} flooring. "
    "Room shape: {room_shape}. Clean rectangular floor plan with orthogonal walls. "
    "Photorealistic textures and accurate professional lab finishes. "
    "Flat overhead view, no perspective, no isometric, no 3D angle. "
    "No text, no labels, no legends, no annotations, no dimensions, no arrows, no icons, no watermarks."
)

REQUIRED_FIELDS = [
    "room_type", "user_type", "key_furniture", "decorations",
    "zones", "style", "palette", "floor_material", "room_shape",
]

SYSTEM_PROMPT = """You are an expert interior designer generating parameter sets for top-down room renders.
Focus: generate DENSELY FURNISHED, realistic single-room interiors with MANY objects.

Requirements:
- Each scene is ONE specific room (not a whole apartment), richly furnished and decorated
- Architecturally plausible and internally consistent; layouts should imply clear walls, at least one door, and one or more windows where sensible for the room type
- Diverse room types, styles, user demographics, color palettes
- key_furniture: COMPREHENSIVE list of ALL major furniture pieces (at least 8-12 items). Include multiples where realistic (e.g. "2 nightstands" not just "nightstand"). Names only, NO shape/size/material adjectives. Example: "king bed, 2 nightstands, dresser, wardrobe, armchair, ottoman, vanity desk, vanity chair, floor lamp, tall bookshelf, bench"
- decorations: list of small objects, accessories, and decorative items that sit ON or NEAR the furniture (at least 6-10 items). Example: "table lamp, potted plant, photo frames, books stack, candle set, decorative tray, throw pillows, area rug, wall clock, vase with flowers"
- Zones: at least 3-4 distinct functional zones per room
- Room shapes vary (rectangular, L-shaped, square, etc.) with REALISTIC full-size dimensions, typically 6m-12m per side. Minimum 5m on any side. Studios/apartments can be up to 15m
- Styles vary (modern, Scandinavian, industrial, Japanese, mid-century, bohemian, farmhouse, Art Deco, etc.)

Return ONLY a JSON array. No markdown, no explanation.
Keys: room_type, user_type, key_furniture, decorations, zones, style, palette, floor_material, room_shape"""

SCENE_SCOPE_RULES = {
    "home": (
        "Only generate residential home spaces (bedroom, living room, kitchen, bathroom, home office, etc.)."
    ),
    "mixed": (
        "Generate a broad mix of residential and non-residential spaces. "
        "At least half of the outputs should be non-residential "
        "(e.g. cafe, retail shop, clinic room, classroom, studio, office, workshop, hotel suite)."
    ),
    "non_residential": (
        "Do NOT generate home/residential rooms. Only generate non-residential indoor spaces "
        "(e.g. cafe, bar lounge, retail store, office meeting room, coworking space, clinic, salon, classroom, museum room, workshop)."
    ),
}


# ---------------------------------------------------------------------------
# Prompt generation (text mode)
# ---------------------------------------------------------------------------

def build_system_prompt(scene_scope: str, rectangular_ratio: float) -> str:
    if scene_scope not in SCENE_SCOPE_RULES:
        raise ValueError(f"Unsupported scene_scope: {scene_scope}")
    rectangular_ratio_pct = int(round(rectangular_ratio * 100))
    return (
        SYSTEM_PROMPT
        + f"\n- Scene scope constraint: {SCENE_SCOPE_RULES[scene_scope]}"
        + "\n- Do not output outdoor scenes."
        + f"\n- Prioritize rectangular room plans: at least {rectangular_ratio_pct}% of outputs should be rectangular or near-rectangular."
    )


def build_user_prompt(count: int, scene_scope: str, rectangular_ratio: float) -> str:
    example = {
        "room_type": "master bedroom",
        "user_type": "a young couple",
        "key_furniture": "king bed, 2 nightstands, tall dresser, wardrobe, armchair, ottoman, vanity desk, vanity chair, floor lamp, tall bookshelf, accent bench",
        "decorations": "2 table lamps, potted plant, photo frames, books stack, candle set, decorative tray, throw pillows, area rug, wall clock, vase with flowers, jewelry box",
        "zones": "sleeping area, dressing area, reading nook, vanity corner",
        "style": "Scandinavian minimalist",
        "palette": "soft white, warm oak, muted sage green",
        "floor_material": "light oak hardwood",
        "room_shape": "rectangular 8m x 6m",
    }
    min_rectangular = max(1, math.ceil(count * rectangular_ratio))
    scope_rule = SCENE_SCOPE_RULES[scene_scope]
    return (
        f"Generate exactly {count} diverse interior room parameter sets.\n"
        f"Each room should be DENSELY furnished with at least 8-12 major furniture pieces "
        f"and 6-10 decorative accessories. Think of a real, lived-in room.\n"
        f"Scene scope rule: {scope_rule}\n"
        f"Rectangular room-plan target: at least {min_rectangular} out of {count} outputs should be rectangular or near-rectangular.\n"
        f"For rectangular outputs, use room_shape values like 'rectangular 9m x 7m' or 'near-rectangular 10m x 8m with small recess'.\n"
        f"Do not generate outdoor scenes.\n"
        f"Example of one element:\n{json.dumps(example, indent=2)}\n\n"
        f"Now generate {count} sets. Return ONLY the JSON array."
    )


def extract_json_array(text: str) -> list:
    """Extract a JSON array from LLM response, handling markdown fences and truncation."""
    fence_match = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", text)
    if fence_match:
        text = fence_match.group(1)
    else:
        arr_match = re.search(r"\[[\s\S]*\]", text)
        if arr_match:
            text = arr_match.group(0)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Truncated response: find the last complete JSON object and close the array
    arr_start = text.find("[")
    if arr_start == -1:
        raise json.JSONDecodeError("No array start found", text, 0)
    body = text[arr_start + 1:]
    last_brace = body.rfind("}")
    if last_brace == -1:
        raise json.JSONDecodeError("No complete object found", text, 0)
    repaired = "[" + body[: last_brace + 1] + "]"
    result = json.loads(repaired)
    print(f"[WARN] JSON was truncated; salvaged {len(result)} complete entries")
    return result


def validate_params(params: dict) -> bool:
    return all(
        isinstance(params.get(f), str) and len(params[f].strip()) > 0
        for f in REQUIRED_FIELDS
    )


def _openai_chat_base_url(base_url: Optional[str]) -> Optional[str]:
    """OpenAI-compatible chat uses .../v1/chat/completions. Gemini image uses host root."""
    if not base_url:
        return None
    b = base_url.rstrip("/")
    if b.endswith("/v1"):
        return b
    return f"{b}/v1"


def _message_content_to_text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text" and "text" in block:
                    parts.append(block["text"])
                elif "text" in block:
                    parts.append(str(block["text"]))
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content)


def deduplicate(param_list: list) -> list:
    seen = set()
    unique = []
    for p in param_list:
        key = p.get("room_type", "").lower() + "|" + p.get("style", "").lower()
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


BATCH_SIZE = 5


def _call_llm_for_params(
    client,
    model: str,
    batch_count: int,
    scene_scope: str,
    rectangular_ratio: float,
) -> list:
    """Single LLM call that requests *batch_count* parameter sets."""
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": build_system_prompt(scene_scope, rectangular_ratio)},
            {"role": "user", "content": build_user_prompt(batch_count, scene_scope, rectangular_ratio)},
        ],
        temperature=1.0,
        max_tokens=16384,
    )
    msg = completion.choices[0].message
    raw_text = _message_content_to_text(getattr(msg, "content", None))

    try:
        return extract_json_array(raw_text)
    except json.JSONDecodeError as e:
        print(f"[WARN] Failed to parse batch response: {e}")
        print(f"[DEBUG] Raw response snippet:\n{raw_text[:1500]}")
        return []


def generate_params(
    model: str,
    count: int,
    api_key: str = None,
    base_url: str = None,
    scene_scope: str = "mixed",
    rectangular_ratio: float = 0.85,
) -> list:
    from openai import OpenAI

    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "API key is required. Either pass --api-key or set OPENAI_API_KEY env var."
        )

    chat_base = _openai_chat_base_url(base_url)
    client_kw: dict = {"api_key": api_key}
    if chat_base:
        client_kw["base_url"] = chat_base
    client = OpenAI(**client_kw)

    all_params: list = []
    remaining = count
    batch_idx = 0
    while remaining > 0:
        batch = min(remaining, BATCH_SIZE)
        batch_idx += 1
        print(f"Calling {model} – batch {batch_idx} ({batch} sets, {len(all_params)} collected so far)...")
        try:
            results = _call_llm_for_params(
                client, model, batch, scene_scope=scene_scope, rectangular_ratio=rectangular_ratio
            )
        except Exception as e:
            print(f"[ERROR] Batch {batch_idx} failed: {e}")
            break
        valid = [p for p in results if validate_params(p)]
        if len(valid) < len(results):
            print(f"[WARN] Dropped {len(results) - len(valid)} invalid entries in batch {batch_idx}")
        all_params.extend(valid)
        remaining -= batch

    print(f"Collected {len(all_params)} valid parameter sets in total")

    unique = deduplicate(all_params)
    if len(unique) < len(all_params):
        print(f"[WARN] Removed {len(all_params) - len(unique)} duplicate entries")

    return unique


def build_prompts(params_list: list, template: str = PROMPT_TEMPLATE) -> list:
    prompts = []
    for i, params in enumerate(params_list, start=1):
        prompt_text = template.format(**{f: params[f] for f in REQUIRED_FIELDS})
        prompts.append({
            "id": i,
            "params": params,
            "prompt": prompt_text,
        })
    return prompts


def save_output(prompts: list, model: str, output_path: str, template: str = PROMPT_TEMPLATE):
    output = {
        "metadata": {
            "model": model,
            "count": len(prompts),
            "generated_at": datetime.now().isoformat(),
            "template": template,
        },
        "prompts": prompts,
    }
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(prompts)} prompts to {output_path}")


# ---------------------------------------------------------------------------
# Image generation mode
# ---------------------------------------------------------------------------

def _normalize_gemini_base_url(base_url: str) -> str:
    """Strip trailing slash so we can append /v1beta/... consistently."""
    return base_url.rstrip("/")


def _extract_inline_image_bytes(part: dict) -> Optional[bytes]:
    """Support both camelCase (API) and snake_case (some proxies)."""
    inline = part.get("inlineData") or part.get("inline_data")
    if not inline:
        return None
    b64 = inline.get("data")
    if not b64:
        return None
    return base64.b64decode(b64)


def _filename_slug(value: str, fallback: str = "item") -> str:
    """Convert prompt metadata into a compact filesystem-safe filename part."""
    value = re.sub(r"[\s/\\:;|]+", "_", value.strip())
    value = re.sub(r"[^A-Za-z0-9_.-]", "", value)
    value = value.strip("._-")
    return value or fallback


def _call_gemini_image(
    base_url: str,
    api_key: str,
    model: str,
    prompt_text: str,
    aspect_ratio: str = "1:1",
    image_size: str = "1K",
    reference_images: Optional[list] = None,
    reference_mime: str = "image/png",
    timeout: int = 300,
) -> bytes:
    """
    Call Gemini generateContent (nanobanana-style): POST JSON body as string,
    URL .../v1beta/models/{model}:generateContent/

    *reference_images* accepts a list of raw-bytes images that are sent as
    inline image parts alongside the text prompt for visual guidance.
    """
    import requests

    root = _normalize_gemini_base_url(base_url)
    url = f"{root}/v1beta/models/{model}:generateContent?key={api_key}"

    parts: list = [{"text": prompt_text}]
    for img_bytes in (reference_images or []):
        parts.append({
            "inlineData": {
                "mimeType": reference_mime,
                "data": base64.b64encode(img_bytes).decode("ascii"),
            }
        })

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": parts,
            }
        ],
        "generationConfig": {
            "responseModalities": ["TEXT", "IMAGE"],
            "imageConfig": {
                "aspectRatio": aspect_ratio,
                "imageSize": image_size,
            },
        },
    }
    body = json.dumps(payload, ensure_ascii=False)
    headers = {
        "Content-Type": "application/json",
    }

    resp = requests.request("POST", url, headers=headers, data=body.encode("utf-8"), timeout=timeout)
    if not resp.ok:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:2000]}")

    try:
        result = resp.json()
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON response: {e}\n{resp.text[:2000]}") from e

    for candidate in result.get("candidates", []):
        for part in candidate.get("content", {}).get("parts", []):
            raw = _extract_inline_image_bytes(part)
            if raw:
                return raw

    raise ValueError(
        f"No image in response. Top-level keys: {list(result.keys())}. "
        f"Snippet: {resp.text[:1500]}"
    )


def generate_images(
    prompts_json_path: str,
    image_model: str,
    api_key: str,
    base_url: str,
    output_dir: str,
    delay: float = 2.0,
    aspect_ratio: str = "1:1",
    image_size: str = "1K",
):
    """
    Read prompts from JSON file and call Gemini generateContent API for each.
    """
    with open(prompts_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    prompt_list = data.get("prompts", [])
    if not prompt_list:
        print("[ERROR] No prompts found in JSON file.")
        return

    os.makedirs(output_dir, exist_ok=True)

    total = len(prompt_list)
    succeeded, failed = 0, 0

    for item in prompt_list:
        idx = item["id"]
        prompt_text = item["prompt"]
        room_type = _filename_slug(item["params"].get("room_type", "room"), "room")
        style = _filename_slug(item["params"].get("style", "default"), "default")
        filename = f"{idx:03d}_{room_type}_{style}.png"
        filepath = os.path.join(output_dir, filename)

        if os.path.exists(filepath):
            print(f"[{idx}/{total}] SKIP (exists): {filename}")
            succeeded += 1
            continue

        print(f"[{idx}/{total}] Generating: {filename} ...")
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                image_bytes = _call_gemini_image(
                    base_url, api_key, image_model, prompt_text,
                    aspect_ratio=aspect_ratio, image_size=image_size,
                )
                Path(filepath).write_bytes(image_bytes)
                print(f"  -> Saved ({len(image_bytes) // 1024} KB)")
                succeeded += 1
                break
            except Exception as e:
                if attempt < max_retries:
                    wait = delay * (2 ** (attempt - 1))
                    print(f"  -> Attempt {attempt}/{max_retries} failed: {e}")
                    print(f"     Retrying in {wait:.0f}s ...")
                    time.sleep(wait)
                else:
                    print(f"  -> FAILED after {max_retries} attempts: {e}")
                    failed += 1

        if delay > 0:
            time.sleep(delay)

    print(f"\nImage generation done: {succeeded} succeeded, {failed} failed out of {total}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch generate top-down interior render prompts and images"
    )
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # --- sub-command: prompt ---
    p_prompt = subparsers.add_parser("prompt", help="Generate text prompts via LLM")
    p_prompt.add_argument("--count", type=int, default=20)
    p_prompt.add_argument("--model", type=str, default="gpt-4o")
    p_prompt.add_argument("--api-key", type=str, default=None)
    p_prompt.add_argument("--base-url", type=str, default=None)
    p_prompt.add_argument("--output", type=str,
                          default=os.path.join(os.path.dirname(__file__), "generated_prompts.json"))
    p_prompt.add_argument(
        "--scene-scope",
        type=str,
        default="mixed",
        choices=["home", "mixed", "non_residential"],
        help="Scene category scope for prompt generation",
    )
    p_prompt.add_argument(
        "--rectangular-ratio",
        type=float,
        default=0.85,
        help="Target ratio (0~1) of rectangular/near-rectangular room plans",
    )

    # --- sub-command: image ---
    p_image = subparsers.add_parser("image", help="Generate images from prompt JSON file")
    p_image.add_argument("--prompts", type=str,
                         default=os.path.join(os.path.dirname(__file__), "generated_prompts.json"),
                         help="Path to generated_prompts.json")
    p_image.add_argument("--image-model", type=str, default="gemini-3-pro-image-preview")
    p_image.add_argument("--api-key", type=str, default=None)
    p_image.add_argument("--base-url", type=str, default=None,
                         help="Gemini proxy root URL, no /v1 suffix (e.g. http://host:3888)")
    p_image.add_argument("--output-dir", type=str,
                         default=os.path.join(os.path.dirname(__file__), "generated_images"))
    p_image.add_argument("--delay", type=float, default=2.0,
                         help="Seconds to wait between API calls (default: 2)")
    p_image.add_argument("--aspect-ratio", type=str, default="16:9",
                         help="Image aspect ratio (1:1, 16:9, 4:3, etc.)")
    p_image.add_argument("--image-size", type=str, default="4K",
                         help="Image size: 512, 1K, 2K, 4K")

    # --- sub-command: all (prompt + image in one go) ---
    p_all = subparsers.add_parser("all", help="Generate prompts then images in one go")
    p_all.add_argument("--count", type=int, default=20)
    p_all.add_argument("--prompt-model", type=str, default="gpt-4o",
                       help="LLM model for prompt generation")
    p_all.add_argument("--image-model", type=str, default="gemini-3-pro-image-preview",
                       help="Model for image generation")
    p_all.add_argument("--api-key", type=str, default=None)
    p_all.add_argument("--base-url", type=str, default=None)
    p_all.add_argument("--output", type=str,
                       default=os.path.join(os.path.dirname(__file__), "generated_prompts.json"))
    p_all.add_argument("--output-dir", type=str,
                       default=os.path.join(os.path.dirname(__file__), "generated_images"))
    p_all.add_argument("--delay", type=float, default=2.0)
    p_all.add_argument("--aspect-ratio", type=str, default="16:9")
    p_all.add_argument("--image-size", type=str, default="4K")
    p_all.add_argument(
        "--scene-scope",
        type=str,
        default="mixed",
        choices=["home", "mixed", "non_residential"],
        help="Scene category scope for prompt generation",
    )
    p_all.add_argument(
        "--rectangular-ratio",
        type=float,
        default=0.85,
        help="Target ratio (0~1) of rectangular/near-rectangular room plans",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 2

    api_key = getattr(args, "api_key", None) or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("[ERROR] API key required. Pass --api-key or set OPENAI_API_KEY env var.")
        return 2
    rectangular_ratio = getattr(args, "rectangular_ratio", 0.85)
    if not (0 < rectangular_ratio <= 1):
        print("[ERROR] --rectangular-ratio must be in (0, 1].")
        return 2
    count = getattr(args, "count", 1)
    if count < 1:
        print("[ERROR] --count must be >= 1.")
        return 2

    if args.command == "prompt":
        params_list = generate_params(
            args.model, args.count, api_key, args.base_url,
            scene_scope=args.scene_scope, rectangular_ratio=rectangular_ratio
        )
        if not params_list:
            print("[ERROR] No valid parameters generated.")
            return 1
        prompts = build_prompts(params_list)
        save_output(prompts, args.model, args.output)
        print(f"\nDone! Generated {len(prompts)} prompts.")
        return 0

    elif args.command == "image":
        base_url = args.base_url
        if not base_url:
            print("[ERROR] --base-url is required for image generation.")
            return 2
        generate_images(args.prompts, args.image_model, api_key, base_url,
                        args.output_dir, args.delay,
                        args.aspect_ratio, args.image_size)
        return 0

    elif args.command == "all":
        base_url = args.base_url
        if not base_url:
            print("[ERROR] --base-url is required.")
            return 2
        # Step 1: generate prompts
        params_list = generate_params(
            args.prompt_model, args.count, api_key, base_url,
            scene_scope=args.scene_scope, rectangular_ratio=rectangular_ratio
        )
        if not params_list:
            print("[ERROR] No valid parameters generated.")
            return 1
        prompts = build_prompts(params_list)
        save_output(prompts, args.prompt_model, args.output)
        print(f"\nGenerated {len(prompts)} prompts. Now generating images...\n")
        # Step 2: generate images
        generate_images(args.output, args.image_model, api_key, base_url,
                        args.output_dir, args.delay,
                        args.aspect_ratio, args.image_size)
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
"""
python /Users/yangyixuan/SceneGen_Agent_final/image_prompt_gen/topdown_room_image_generator.py  all \
  --api-key "$SCENEGEN_TEXTURE_API_KEY" \
  --base-url "$SCENEGEN_TEXTURE_BASE_URL" \
  --image-model gemini-3-pro-image-preview \
  --aspect-ratio 16:9 \
  --image-size 1K \
  --count 20 \
  --prompt-model gpt-4o \
  --scene-scope non_residential
"""
