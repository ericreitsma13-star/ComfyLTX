#!/usr/bin/env python3
"""Generate high quality SDXL reference: woman singing in forest, facing viewer."""
import json, urllib.request, time, os, shutil

COMFY = "http://127.0.0.1:8188"
SDXL = "sd_xl_base_1.0.safetensors"

prompt = (
    "young woman with long dark hair, medium shot from chest up facing camera, "
    "singing on a misty pine forest path, golden dawn sunlight streaming through trees, "
    "wearing casual outdoor jacket, looking directly at viewer, lips parted singing, "
    "photorealistic, 8k, highly detailed face, cinematic lighting, "
    "sharp focus, detailed forest background, depth of field"
)
neg = (
    "headshot, close up, extreme close up, from behind, back view, "
    "looking away, profile, turned around, facing away, "
    "ugly, deformed, blurry, low quality, bad anatomy, "
    "watermark, text, extra limbs, distorted face"
)

gen_wf = {
    "10": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": SDXL}},
    "20": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["10",1]}},
    "21": {"class_type": "CLIPTextEncode", "inputs": {"text": neg, "clip": ["10",1]}},
    "32": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
    "30": {"class_type": "KSampler", "inputs": {"model": ["10",0], "positive": ["20",0], "negative": ["21",0], "latent_image": ["32",0], "seed": 400, "steps": 30, "cfg": 7.0, "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0}},
    "31": {"class_type": "VAEDecode", "inputs": {"vae": ["10",2], "samples": ["30",0]}},
    "33": {"class_type": "SaveImage", "inputs": {"filename_prefix": "sdxl_front_v3", "images": ["31",0]}},
}

req = urllib.request.Request(f"{COMFY}/prompt", data=json.dumps({"prompt": gen_wf}).encode(),
    headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=30) as r:
    pid = json.loads(r.read())["prompt_id"]
print(f"Queued: {pid}")

start = time.time()
while time.time()-start < 600:
    try:
        h = json.loads(urllib.request.urlopen(f"{COMFY}/history/{pid}").read())
    except: time.sleep(2); continue
    if pid in h:
        print(f"{time.time()-start:.0f}s — {h[pid]['status']['status_str']}")
        if h[pid]['status']['status_str'] == 'success':
            for no, out in h[pid].get("outputs",{}).items():
                for img in out.get("images",[]):
                    fp = os.path.join("/home/ericr/ComfyUI/output", img["filename"])
                    sz = os.path.getsize(fp)
                    print(f"→ {fp} ({sz/1024:.0f} KB)")
                    shutil.copy(fp, os.path.join("/home/ericr/ComfyUI/input", "ref_front_v3.png"))
                    print("→ Copied to input/ref_front_v3.png")
        break
    time.sleep(2)
else: print("TIMEOUT")
