#!/usr/bin/env python3
"""Walking poses via img2img from clean reference — consistent face."""
import json, urllib.request, time

COMFY = "http://127.0.0.1:8188"
CKPT = "sd_xl_base_1.0.safetensors"
REF = "char_base.png"
steps = 25
NEG = "cartoon, anime, deformed, blurry, low quality, ugly, bad anatomy, squinty eyes"

BASE = "young woman with long dark brown hair, walking in misty pine forest"
variations = [
    (301, f"{BASE}, full body shot, walking on forest path, golden sunlight, cinematic", 0.35),
    (302, f"{BASE}, three quarter body, walking through pine forest, morning mist, warm lighting", 0.35),
    (303, f"{BASE}, medium shot walking on forest trail, dappled sunlight, professional photography", 0.30),
    (304, f"{BASE}, full body walking on forest path, cinematic lighting, depth of field", 0.35),
    (305, f"{BASE}, standing in forest clearing, looking up at trees, soft golden light", 0.30),
]

for i, (seed, prompt, denoise) in enumerate(variations):
    wf = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
        "2": {"class_type": "LoadImage", "inputs": {"image": REF}},
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
        "8": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": f"walk_v{i+1:02d}"}},
    }
    print(f"{i+1}/5 seed={seed} denoise={denoise}: ", end="", flush=True)
    req = urllib.request.Request(f"{COMFY}/prompt", data=json.dumps({"prompt": wf}).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            pid = json.loads(r.read())["prompt_id"]
    except urllib.error.HTTPError as e:
        print(f"ERR {e.code}: {e.read().decode()[:100]}")
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
