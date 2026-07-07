#!/usr/bin/env python3
"""Generate consistent portraits for LoRA - same character."""
import json, urllib.request, time, os

COMFY = "http://127.0.0.1:8188"
CKPT = "sd_xl_base_1.0.safetensors"
W, H = 1024, 1024
steps, cfg = 25, 4.0
NEG = "cartoon, anime, deformed, blurry, low quality, ugly, bad anatomy, watermark, squinty eyes"

# Same character description for consistency
CHAR = "young woman with dark brown hair and brown eyes, oval face, straight nose, medium lips"
PROMPTS = [
    f"{CHAR}, looking at camera, portrait, soft natural lighting, detailed face, professional photography",
    f"{CHAR}, three quarter portrait, warm lighting, professional headshot, detailed facial features",
    f"{CHAR}, slightly smiling, portrait photography, studio lighting, sharp focus, high detail",
    f"{CHAR}, serious expression, fashion editorial, dramatic lighting, cinematic portrait",
    f"{CHAR}, close-up portrait, soft golden hour lighting, detailed eyes, professional photoshoot",
]
SEEDS = [201, 202, 203, 204, 205]  # same family as the best one (201)

for i, (seed, prompt) in enumerate(zip(SEEDS, PROMPTS)):
    wf = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": NEG, "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": W, "height": H, "batch_size": 1}},
        "5": {"class_type": "KSampler", "inputs": {
            "seed": seed, "steps": steps, "cfg": float(cfg),
            "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0,
            "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0],
            "latent_image": ["4", 0]}},
        "6": {"class_type": "VAEDecode", "inputs": {"vae": ["1", 2], "samples": ["5", 0]}},
        "7": {"class_type": "SaveImage", "inputs": {"images": ["6", 0], "filename_prefix": f"char_dark_{i+1:02d}"}},
    }
    print(f"{i+1}/5 seed={seed}: ", end="", flush=True)
    req = urllib.request.Request(f"{COMFY}/prompt",
        data=json.dumps({"prompt": wf, "client_id": f"c{i}"}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        pid = json.loads(r.read())["prompt_id"]
    start = time.time()
    while time.time() - start < 120:
        try:
            h = json.loads(urllib.request.urlopen(f"{COMFY}/history/{pid}").read())
        except:
            time.sleep(1); continue
        if pid in h:
            print(f"{time.time()-start:.0f}s {h[pid].get('status',{}).get('status_str','?')}")
            break
        time.sleep(1)
    else:
        print("TIMEOUT")
