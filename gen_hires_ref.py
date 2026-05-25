#!/usr/bin/env python3
"""Generate high-res reference: SDXL base + img2img upscale to 2K."""
import json, urllib.request, time, os, shutil

COMFY = "http://127.0.0.1:8188"
OUT = "/home/ericr/ComfyUI/output"
SDXL = "sd_xl_base_1.0.safetensors"

P = "young woman with long dark hair, white dress, medium shot chest up facing viewer, misty pine forest path at golden dawn, cinematic, photorealistic, sharp, highly detailed face, elegant"
N = "headshot, close up, portrait, from behind, back view, ugly, deformed, blurry, low quality, cartoon, painting"

# Pass 1: Generate base at 1024x1024
print("Pass 1: SDXL base generation...")
wf1 = {
    "10": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": SDXL}},
    "20": {"class_type": "CLIPTextEncode", "inputs": {"text": P, "clip": ["10",1]}},
    "21": {"class_type": "CLIPTextEncode", "inputs": {"text": N, "clip": ["10",1]}},
    "22": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
    "23": {"class_type": "KSampler", "inputs": {"model": ["10",0], "positive": ["20",0], "negative": ["21",0], "latent_image": ["22",0], "seed": 800, "steps": 30, "cfg": 7.0, "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0}},
    "24": {"class_type": "VAEDecode", "inputs": {"vae": ["10",2], "samples": ["23",0]}},
    "25": {"class_type": "SaveImage", "inputs": {"images": ["24",0], "filename_prefix": "hq_base"}},
}
req = urllib.request.Request(f"{COMFY}/prompt", data=json.dumps({"prompt": wf1}).encode(),
    headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=30) as r:
    base_pid = json.loads(r.read())["prompt_id"]

start = time.time()
base_file = None
while time.time()-start < 120:
    try:
        h = json.loads(urllib.request.urlopen(f"{COMFY}/history/{base_pid}").read())
    except: time.sleep(2); continue
    if base_pid in h:
        print(f"  {h[base_pid]['status']['status_str']} ({time.time()-start:.0f}s)")
        if h[base_pid]['status']['status_str'] == 'success':
            for no, out in h[base_pid].get("outputs",{}).items():
                for img in out.get("images",[]):
                    base_file = os.path.join(OUT, img["filename"])
                    print(f"  → {img['filename']} ({os.path.getsize(base_file)/1024:.0f} KB)")
                    shutil.copy(base_file, "/home/ericr/ComfyUI/input/_hq_base.png")
                    print(f"  → Copied to input/_hq_base.png")
        break
    time.sleep(2)

if not base_file or not os.path.exists(base_file):
    print("Failed to generate base image")
    exit(1)

# Pass 2: Upscale to 2048x2048 via img2img
print("\nPass 2: Upscaling to 2K...")
wf2 = {
    "10": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": SDXL}},
    "20": {"class_type": "LoadImage", "inputs": {"image": "_hq_base.png"}},
    "21": {"class_type": "ImageScale", "inputs": {"upscale_method": "lanczos", "image": ["20",0], "width": 2048, "height": 2048, "crop": "disabled"}},
    "22": {"class_type": "VAEEncode", "inputs": {"vae": ["10",2], "pixels": ["21",0]}},
    "30": {"class_type": "CLIPTextEncode", "inputs": {"text": P + ", highly detailed, sharp focus, 8k", "clip": ["10",1]}},
    "31": {"class_type": "CLIPTextEncode", "inputs": {"text": N, "clip": ["10",1]}},
    "32": {"class_type": "KSampler", "inputs": {"model": ["10",0], "positive": ["30",0], "negative": ["31",0], "latent_image": ["22",0], "seed": 801, "steps": 20, "cfg": 5.0, "sampler_name": "euler", "scheduler": "normal", "denoise": 0.35}},
    "33": {"class_type": "VAEDecode", "inputs": {"vae": ["10",2], "samples": ["32",0]}},
    "34": {"class_type": "SaveImage", "inputs": {"images": ["33",0], "filename_prefix": "hq_2k"}},
}
req = urllib.request.Request(f"{COMFY}/prompt", data=json.dumps({"prompt": wf2}).encode(),
    headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=30) as r:
    up_pid = json.loads(r.read())["prompt_id"]

start = time.time()
while time.time()-start < 180:
    try:
        h = json.loads(urllib.request.urlopen(f"{COMFY}/history/{up_pid}").read())
    except: time.sleep(2); continue
    if up_pid in h:
        print(f"  {h[up_pid]['status']['status_str']} ({time.time()-start:.0f}s)")
        if h[up_pid]['status']['status_str'] == 'success':
            for no, out in h[up_pid].get("outputs",{}).items():
                for img in out.get("images",[]):
                    fp = os.path.join(OUT, img["filename"])
                    sz = os.path.getsize(fp)
                    print(f"  → {img['filename']} ({sz/1024:.0f} KB, {sz/1024/1024:.1f} MB)")
                    shutil.copy(fp, "/home/ericr/ComfyUI/input/ref_2k.png")
                    print(f"  → Copied to input/ref_2k.png")
        break
    time.sleep(2)

print("\nDone. Load ref_2k.png (2K) in the ComfyUI UI workflow.")
