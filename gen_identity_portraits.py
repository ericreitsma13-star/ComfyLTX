#!/usr/bin/env python3
"""Img2img from best portrait — preserves identity while varying details."""
import json, urllib.request, time, os

COMFY = "http://127.0.0.1:8188"
CKPT = "sd_xl_base_1.0.safetensors"
W, H = 1024, 1024
steps = 20
NEG = "cartoon, anime, deformed, blurry, low quality, ugly, bad anatomy, watermark"

BASE_IMAGE = "char_base.png"
BASE_CHAR = "young woman with long dark brown hair, brown eyes, oval face"

# Different denoise levels create variations while keeping identity
variations = [
    (201, f"{BASE_CHAR}, looking at camera, portrait, soft natural lighting", 0.25),
    (202, f"{BASE_CHAR}, three quarter portrait, warm lighting", 0.30),
    (203, f"{BASE_CHAR}, slightly smiling, portrait photography, studio lighting", 0.25),
    (204, f"{BASE_CHAR}, serious expression, dramatic lighting, cinematic portrait", 0.30),
    (205, f"{BASE_CHAR}, close-up portrait, soft golden hour lighting, detailed eyes", 0.20),
]

for i, (seed, prompt, denoise) in enumerate(variations):
    wf = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
        "2": {"class_type": "LoadImage", "inputs": {"image": BASE_IMAGE}},
        "3": {"class_type": "VAEEncode", "inputs": {"vae": ["1", 2], "pixels": ["2", 0]}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["1", 1]}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": NEG, "clip": ["1", 1]}},
        "6": {"class_type": "KSampler", "inputs": {
            "seed": seed, "steps": steps, "cfg": 4.0,
            "sampler_name": "dpmpp_2m", "scheduler": "karras",
            "denoise": float(denoise),
            "model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0],
            "latent_image": ["3", 0]}},
        "7": {"class_type": "VAEDecode", "inputs": {"vae": ["1", 2], "samples": ["6", 0]}},
        "8": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": f"char_v{i+1:02d}"}},
    }
    print(f"{i+1}/{len(variations)} seed={seed} denoise={denoise}: ", end="", flush=True)
    req = urllib.request.Request(f"{COMFY}/prompt",
        data=json.dumps({"prompt": wf, "client_id": f"v{i}"}).encode(),
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
            s = h[pid].get('status', {}).get('status_str', '?')
            print(f"{time.time()-start:.0f}s {s}")
            break
        time.sleep(1)
    else:
        print("TIMEOUT")
