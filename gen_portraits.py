#!/usr/bin/env python3
"""Dark-haired woman portraits via SDXL for LoRA training."""
import json, urllib.request, time, os

COMFY = "http://127.0.0.1:8188"
CKPT = "sd_xl_base_1.0.safetensors"
W, H = 1024, 1024
steps, cfg = 25, 4.0
NEG = "cartoon, anime, deformed, blurry, low quality, ugly, bad anatomy, watermark, squinty eyes, closed eyes"

prompts = [
    (101, "photograph of a young woman with dark brown hair, portrait, cinematic lighting, detailed face, beautiful symmetrical features, open eyes, high quality"),
    (201, "young woman with long dark brown hair, close-up portrait, soft natural lighting, professional photography, detailed eyes looking at camera, flawless skin"),
    (301, "beautiful young woman with dark hair, medium shot portrait, warm golden hour lighting, professional photoshoot, sharp focus, high detail, open eyes"),
    (401, "young woman dark hair looking at camera, portrait photography, studio lighting, detailed facial features, realistic skin texture, symmetrical face, cinematic"),
    (501, "portrait of a young woman with dark brown hair, headshot, professional lighting, high quality photography, detailed eyes and face"),
    (601, "young woman with dark brown hair, smiling portrait, warm lighting, professional headshot, detailed face, open eyes"),
    (701, "young woman with dark hair, serious expression, fashion portrait, editorial lighting, sharp focus, high detail"),
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
        "7": {"class_type": "SaveImage", "inputs": {"images": ["6", 0], "filename_prefix": f"portrait_dark_{i+1:02d}"}},
    }
    print(f"{i+1}/7 seed={seed}: ", end="", flush=True)
    req = urllib.request.Request(f"{COMFY}/prompt",
        data=json.dumps({"prompt": wf, "client_id": f"g{i}"}).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            pid = json.loads(r.read())["prompt_id"]
    except urllib.error.HTTPError as e:
        print(f"ERROR {e.read().decode()[:100]}")
        continue
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
