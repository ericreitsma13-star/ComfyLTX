import json, requests, time, os, sys

WORKFLOW_FILE = "/opt/comfy-workflows/workflow_4step_fp8.json"
API_BASE = "http://127.0.0.1:8188"

PROMPTS = [
    "Cinematic close-up of a futuristic CPU on a high-end motherboard, intense electric sparks flowing between circuits, ultra-detailed microchips, dramatic cyberpunk lighting, high contrast, metallic textures with reflections, shallow depth of field, sci-fi technology photography, sharp focus, no text.",
    "A serene Japanese garden at sunset with cherry blossom petals falling, koi fish in a pond, wooden bridge, soft golden lighting, bokeh effect, ultra-detailed foliage, atmospheric haze, peaceful mood, photorealistic.",
    "Steampunk airship floating above a Victorian-era city at dusk, brass and copper details, giant gears, hot air balloons in the distance, warm amber lighting, volumetric fog, intricate mechanical details, cinematic composition.",
    "Macro photograph of a vibrant blue morpho butterfly on a tropical flower, dew drops on wings, extreme detail, shallow depth of field, soft natural lighting, rainforest bokeh background, 8k quality.",
    "Futuristic neon-lit city street in Tokyo at night, rain-slicked pavement reflecting signs, holographic advertisements, flying cars in the distance, cyberpunk aesthetic, blue and magenta color palette, volumetric lighting, cinematic.",
]

with open(WORKFLOW_FILE) as f:
    base_workflow = json.load(f)

for i, prompt_text in enumerate(PROMPTS):
    print(f"\n=== Prompt {i + 1}/{len(PROMPTS)} ===")
    print(f"Prompt: {prompt_text[:60]}...")

    workflow = json.loads(json.dumps(base_workflow))
    workflow["6"]["inputs"]["text"] = prompt_text
    workflow["3"]["inputs"]["seed"] = int(time.time() * 1000) % (2**32)

    r = requests.post(f"{API_BASE}/prompt", json={"prompt": workflow}, timeout=30)
    r.raise_for_status()
    prompt_id = r.json()["prompt_id"]
    print(f"  Queued: {prompt_id}")

    start = time.time()
    while True:
        try:
            r = requests.get(f"{API_BASE}/history/{prompt_id}", timeout=30)
            if r.status_code == 200 and r.json():
                data = r.json()[prompt_id]
                if data.get("status", {}).get("completed") is True:
                    elapsed = time.time() - start
                    for node_id, node_out in data.get("outputs", {}).items():
                        for img in node_out.get("images", []):
                            print(f"  Done: {img['filename']} ({elapsed:.1f}s)")
                    break
                if data.get("status", {}).get("error"):
                    raise RuntimeError(f"Prompt failed: {data['status']}")
        except requests.RequestException:
            pass
        time.sleep(1)

print(f"\n=== All {len(PROMPTS)} prompts completed ===")
