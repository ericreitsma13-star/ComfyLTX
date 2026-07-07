#!/usr/bin/env python3
"""Generate full-body walking portraits for LoRA training."""
import json, urllib.request, time

COMFY = "http://127.0.0.1:8188"
CKPT = "sd_xl_base_1.0.safetensors"
W, H = 1024, 1024
steps, cfg = 25, 4.0
NEG = "cartoon, anime, deformed, blurry, low quality, ugly, bad anatomy, watermark, squinty eyes"

BASE = "young woman with long dark brown hair, brown eyes, oval face, walking in misty pine forest"
prompts = [
    (301, f"{BASE}, full body shot, walking on forest path, golden sunlight filtering through trees, cinematic, detailed"),
    (302, f"{BASE}, three quarter body, walking through pine forest, morning mist, warm lighting, photorealistic"),
    (303, f"{BASE}, medium shot walking on forest trail, dappled sunlight, professional photography, high detail"),
    (304, f"{BASE}, full body walking away from camera on forest path, cinematic lighting, depth of field, sharp focus"),
    (305, f"{BASE}, three quarter shot standing in forest clearing, looking up at trees, soft golden light, photorealistic"),
]

for i, (seed, prompt) in enumerate(prompts):
    wf = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": NEG, "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": W, "height": H, "batch_size": 1}},
        "5": {"class_type": "KSampler", "inputs": {
            "seed": seed, "steps": steps, "cfg": float(cfg),
            "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0,
            "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0],
            "latent_image": ["4", 0]}},
        "6": {"class_type": "VAEDecode", "inputs": {"vae": ["1", 2], "samples": ["5", 0]}},
        "7": {"class_type": "SaveImage", "inputs": {"images": ["6", 0], "filename_prefix": f"walk_{i+1:02d}"}},
    }
    print(f"{i+1}/5 seed={seed}: ", end="", flush=True)
    req = urllib.request.Request(f"{COMFY}/prompt",
        data=json.dumps({"prompt": wf}).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            pid = json.loads(r.read())["prompt_id"]
    except urllib.error.HTTPError as e:
        print(f"ERR {e.code}")
        continue
    start = time.time()
    while time.time() - start < 120:
        try:
            h = json.loads(urllib.request.urlopen(f"{COMFY}/history/{pid}").read())
        except:
            time.sleep(1); continue
        if pid in h:
            print(f"{time.time()-start:.0f}s {h[pid]['status']['status_str']}")
            break
        time.sleep(1)
    else:
        print("TIMEOUT")
