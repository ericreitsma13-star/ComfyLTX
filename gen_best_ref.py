#!/usr/bin/env python3
"""Generate multiple SDXL medium-shot references, pick best by size."""
import json, urllib.request, time, os, shutil

COMFY = "http://127.0.0.1:8188"
SDXL = "sd_xl_base_1.0.safetensors"

prompt = (
    "young woman with long dark hair wearing a flowing white dress, "
    "medium shot from chest up, facing viewer, standing on misty pine forest path, "
    "golden dawn sunlight streaming through pine trees, morning fog, "
    "photorealistic, 8k, highly detailed face, sharp focus, cinematic lighting, "
    "misty atmosphere, forest background, depth of field"
)
neg = (
    "headshot, close up, extreme close up, portrait, from behind, back view, "
    "profile, looking away, ugly, deformed, blurry, low quality, bad anatomy, "
    "watermark, text, extra limbs, cartoon, painting"
)

best_size, best_file = 0, None

for seed in [500, 501, 502]:
    wf = {
        "10": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": SDXL}},
        "20": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["10",1]}},
        "21": {"class_type": "CLIPTextEncode", "inputs": {"text": neg, "clip": ["10",1]}},
        "32": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        "30": {"class_type": "KSampler", "inputs": {"model": ["10",0], "positive": ["20",0], "negative": ["21",0], "latent_image": ["32",0], "seed": seed, "steps": 30, "cfg": 7.0, "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0}},
        "31": {"class_type": "VAEDecode", "inputs": {"vae": ["10",2], "samples": ["30",0]}},
        "33": {"class_type": "SaveImage", "inputs": {"filename_prefix": f"sdxl_ref_{seed}", "images": ["31",0]}},
    }
    req = urllib.request.Request(f"{COMFY}/prompt", data=json.dumps({"prompt": wf}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        pid = json.loads(r.read())["prompt_id"]
    start = time.time()
    while time.time()-start < 120:
        try:
            h = json.loads(urllib.request.urlopen(f"{COMFY}/history/{pid}").read())
        except: time.sleep(2); continue
        if pid in h:
            print(f"Seed {seed}: {time.time()-start:.0f}s — {h[pid]['status']['status_str']}")
            if h[pid]['status']['status_str'] == 'success':
                for no, out in h[pid].get("outputs",{}).items():
                    for img in out.get("images",[]):
                        fp = os.path.join("/home/ericr/ComfyUI/output", img["filename"])
                        sz = os.path.getsize(fp)
                        print(f"  → {os.path.basename(fp)} ({sz/1024:.0f} KB)")
                        if sz > best_size:
                            best_size = sz
                            best_file = fp
            break
        time.sleep(2)

if best_file:
    shutil.copy(best_file, "/home/ericr/ComfyUI/input/ref_medium_shot.png")
    print(f"\nBest: {os.path.basename(best_file)} ({best_size/1024:.0f} KB)")
    print("→ Copied to input/ref_medium_shot.png")
