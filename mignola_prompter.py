import json, os, sys, time
from openai import OpenAI

PROMPTER_STYLES = {
    "mignola": {
        "system_prompt": """You are an expert prompt engineer for AI image generation specializing in the art style of Mike Mignola (creator of Hellboy). Transform each scene concept into a vivid, detailed image prompt that captures Mignola's distinctive aesthetic.

MIKE MIGNOLA STYLE SIGNATURE:
- Heavy black shadows and negative space — 40-60% of the image should be pure black
- High-contrast chiaroscuro lighting with sharp transitions between light and shadow
- Simplified, geometric shapes with thick, graphic outlines
- Gothic horror and dark fantasy themes rooted in folklore and mythology
- Limited, muted color palette dominated by blacks, deep reds, ochres, browns, and occasional cold blues
- Strong silhouettes against glowing, textured, or patterned backgrounds
- Atmospheric environments: ruined cathedrals, crypts, castles, misty graveyards, ancient forests, stone circles
- Supernatural creatures: demons, ghosts, vampire-like beings, antlered forest spirits, tentacled entities
- Architectural detail: arched windows, stone masonry, wrought iron, stained glass, carved pillars, heavy doors
- A sense of ancient mystery, impending doom, or quiet dread

RULES:
1. Start with the main subject — be specific about pose, expression, and key features
2. Describe the lighting and where shadows fall (Mignola shadows are decisive, not gradient)
3. Set the environment with architectural or natural framing
4. Include "Mignola style, dark fantasy, gothic comic art" as style anchors
5. Keep prompts 3-5 sentences — vivid but not bloated
6. End with technical quality tags: "high contrast, thick linework, graphic novel aesthetic, cinematic"
7. NO text, letters, or words should appear in the image
8. Focus on a single strong composition

Return ONLY a valid JSON object with keys "id", "title", and "prompt" (your generated prompt). No markdown, no backticks, no explanation.""",
        "user_template": """Scene ID: {id}
Title: {title}
Description: {description}

Generate a Mignola-style image prompt. Use the exact scene ID and title in your JSON response.""",
        "scenes_file": "mignola_scenes.json",
    },
    "cinematic": {
        "system_prompt": """You are an expert prompt engineer for AI image generation specializing in dark cinematic photography. Transform each scene concept into a vivid, detailed image prompt that reads like a master photographer's shot description.

DARK CINEMATIC STYLE SIGNATURE:
- Low-key lighting with a single strong light source creating deep, dramatic shadows
- Chiaroscuro: strong contrast between light and dark, with rich blacks that hold detail
- Moody, atmospheric mood — fog, rain, smoke, mist, or steam as atmospheric elements
- Muted, desaturated color palette: deep blues, cold greys, amber/orange for warmth, black
- Realistic textures: wet pavement, worn wood, rusted metal, fabric, skin with pores
- Cinematic composition: rule of thirds, leading lines, negative space, depth through fog/atmosphere
- Environmental storytelling: a scene that implies a narrative without showing it
- Settings: urban night, derelict interiors, wild landscapes in harsh weather, moody interiors

RULES:
1. Open with the main subject in a specific lighting condition
2. Describe the light source, direction, and quality (hard/soft, warm/cool)
3. Describe the atmosphere (fog, rain, smoke, mist, humidity)
4. Include at least one texture detail (wet, rough, peeling, weathered)
5. End with technical tags: "cinematic, photorealistic, moody, low-key lighting, 8K, shot on 35mm film"
6. Keep prompts 3-5 sentences — vivid but precise
7. NO text, letters, or words should appear in the image
8. Use photographic terminology: depth of field, focal length, color grading, lighting ratio

Return ONLY a valid JSON object with keys "id", "title", and "prompt" (your generated prompt). No markdown, no backticks, no explanation.""",
        "user_template": """Scene ID: {id}
Title: {title}
Description: {description}

Generate a dark cinematic image prompt. Use the exact scene ID and title in your JSON response.""",
        "scenes_file": "dark_cinematic_scenes.json",
    },
}


def load_api_key():
    paths = [
        os.path.expanduser("~/Video_summary/.env"),
        os.path.expanduser("~/Video_summary/.env.prod"),
        os.path.expanduser("~/Video_summary/pipeline/.env"),
    ]
    for p in paths:
        if os.path.exists(p):
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("LLM_API_KEY="):
                        return line.split("=", 1)[1].strip().strip("\"'")
    env_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("LLM_API_KEY")
    if env_key:
        return env_key
    raise RuntimeError(
        "No LLM_API_KEY found in Video_summary .env files or environment"
    )


def load_scenes(path):
    with open(path) as f:
        return json.load(f)["scenes"]


def parse_json_from_text(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        text = text.rsplit("```", 1)[0]
    return json.loads(text.strip())


def generate_prompt(client, model, system_prompt, user_template, scene, retries=3):
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": user_template.format(
                            id=scene["id"],
                            title=scene["title"],
                            description=scene["description"],
                        ),
                    },
                ],
                temperature=0.7,
                max_tokens=1024,
                extra_body={"provider": {"ignore": ["Chutes"]}},
            )
            content = resp.choices[0].message.content
            if not content:
                raise ValueError("Empty response from model")
            result = parse_json_from_text(content)
            if "prompt" not in result:
                raise ValueError(f"No 'prompt' key in response: {result}")
            return result
        except Exception as e:
            if attempt < retries - 1:
                print(f"  Retry {attempt + 1}/{retries}: {e}")
                time.sleep(3)
            else:
                raise


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--style",
        default=os.environ.get("STYLE", "cinematic"),
        choices=list(PROMPTER_STYLES.keys()),
        help="Prompt style to use",
    )
    args = parser.parse_args()

    style_config = PROMPTER_STYLES[args.style]
    model = os.environ.get("PROMPTER_MODEL", "meta-llama/llama-3.3-70b-instruct")
    scenes_path = os.environ.get(
        "SCENES_PATH", os.path.join(SCRIPT_DIR, style_config["scenes_file"])
    )
    output_path = os.environ.get(
        "OUTPUT_PATH", os.path.join(SCRIPT_DIR, "generated_prompts.json")
    )

    api_key = load_api_key()
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1", api_key=api_key, timeout=120
    )

    scenes = load_scenes(scenes_path)
    print(f"Style: {args.style}")
    print(f"Loaded {len(scenes)} scenes from {scenes_path}")
    print(f"Model: {model}")
    print()

    results = []
    for i, scene in enumerate(scenes):
        print(f"[{i + 1}/{len(scenes)}] {scene['title']}...")
        result = generate_prompt(
            client,
            model,
            style_config["system_prompt"],
            style_config["user_template"],
            scene,
        )
        results.append(result)
        print(f"  Prompt: {result.get('prompt', '')[:100]}...")
        print()

    with open(output_path, "w") as f:
        json.dump(
            {"model": model, "style": args.style, "prompts": results}, f, indent=2
        )

    print(f"\nDone. Saved {len(results)} prompts to {output_path}")


if __name__ == "__main__":
    main()
