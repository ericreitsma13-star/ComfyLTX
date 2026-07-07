#!/usr/bin/env python3
"""Step A (Z-Image): Per-scene references using Z-Image GGUF via API."""
import json, urllib.request, time, os, shutil, subprocess

COMFY = "http://127.0.0.1:8188"
OUT = "/home/ericr/ComfyUI/output"
IN = "/home/ericr/ComfyUI/input"
full_audio = "/home/ericr/ComfyUI/input/pines_full.mp3"
NUM_SCENES = 15
DURATION = 4.0

def free_vram():
    try:
        req = urllib.request.Request(f"{COMFY}/free", data=json.dumps({"free_memory": True}).encode(), headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=30)
    except: pass

scenes = [
    {"seed": 100, "prompt": "young woman with long dark hair, white dress, medium shot chest up facing viewer, misty pine forest path at golden dawn, cinematic, photorealistic, sharp"},
    {"seed": 101, "prompt": "young woman with long dark hair, white dress, medium shot facing viewer, forest clearing with golden rays piercing mist, cinematic"},
    {"seed": 102, "prompt": "young woman with long dark hair, white dress, medium shot, walking on pine forest path at dawn, warm golden light, cinematic"},
    {"seed": 103, "prompt": "young woman with long dark hair, white dress, facing viewer, pine forest with golden sunrise through branches, mist, cinematic"},
    {"seed": 104, "prompt": "young woman with long dark hair, white dress, medium shot, deep forest path with morning light, cinematic"},
    {"seed": 105, "prompt": "young woman with long dark hair, white dress, facing camera, forest edge with light through mist, cinematic"},
    {"seed": 106, "prompt": "young woman with long dark hair, white dress, medium shot, misty forest trail with dappled light, cinematic"},
    {"seed": 107, "prompt": "young woman with long dark hair, white dress, facing viewer, pine forest at dawn with golden rays, fog, cinematic"},
    {"seed": 108, "prompt": "young woman with long dark hair, white dress, medium shot, forest path in morning mist, warm light, cinematic"},
    {"seed": 109, "prompt": "young woman with long dark hair, white dress, facing viewer, pine forest clearing with golden sunrise, mist, cinematic"},
    {"seed": 110, "prompt": "young woman with long dark hair, white dress, medium shot, walking through pine forest at dawn, sunbeams, cinematic"},
    {"seed": 111, "prompt": "young woman with long dark hair, white dress, facing camera, thick morning fog with golden light, cinematic"},
    {"seed": 112, "prompt": "young woman with long dark hair, white dress, medium shot, forest trail at dawn with dappled light, mist, cinematic"},
    {"seed": 113, "prompt": "young woman with long dark hair, white dress, facing viewer, pine forest clearing with golden rays, mist, cinematic"},
    {"seed": 114, "prompt": "young woman with long dark hair, white dress, medium shot, misty pine forest at sunrise, warm golden light, cinematic"},
]

def queue(wf):
    req = urllib.request.Request(f"{COMFY}/prompt", data=json.dumps({"prompt": wf}).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())["prompt_id"]

def wait(pid, timeout=300):
    start = time.time()
    while time.time()-start < timeout:
        try:
            h = json.loads(urllib.request.urlopen(f"{COMFY}/history/{pid}").read())
        except: time.sleep(3); continue
        if pid in h: return h[pid]
        time.sleep(3)
    return None

# Ensure audio segments exist
print("Preparing audio segments...")
for i in range(NUM_SCENES):
    seg = os.path.join(IN, f"seg_{i:03d}.wav")
    if not os.path.exists(seg):
        subprocess.run(["ffmpeg","-y","-i",full_audio,"-ss",str(i*DURATION),"-t",str(DURATION),"-c","copy",seg], capture_output=True)

print(f"\n=== Step A (Z-Image): {NUM_SCENES} references ===")
start_all = time.time()
refs = []

for i, s in enumerate(scenes):
    # Z-Image workflow using GGUF UNet (Q6_K, 5.9 GB vs 12 GB BF16)
    wf = {
        "1": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_3_4b.safetensors", "type": "qwen_image"}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": s["prompt"], "clip": ["1",0]}},
        "3": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "z-image-turbo-Q6_K.gguf"}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": 960, "height": 544, "batch_size": 1}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": "ae.safetensors"}},
        "6": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "7": {"class_type": "FlowMatchEulerDiscreteScheduler (Custom)", "inputs": {
            "steps": 4, "denoise": 0.0, "sigmas_pt": 10, "s_churn": 256, "s_tmin": 0.5,
            "s_tmax_opt": "disable", "s_noise": 8192, "sigma_max": 1.15, "sigma_min": 1000,
            "rho": 3, "sigma_bias_switch": 0, "flip_sigmas_opt": "disable",
            "scheduler_type": "exponential", "use_clipped_sigmas": "disable",
            "use_timestamped_noise": "disable", "quantize_sigmas": "disable",
            "use_rescaling": "disable", "device_opt": "cuda"
        }},
        "8": {"class_type": "SamplerCustom", "inputs": {
            "model": ["3",0], "add_noise": True, "noise_seed": s["seed"], "cfg": 1.5,
            "positive": ["2",0], "negative": ["2",0], "sampler": ["6",0],
            "sigmas": ["7",0], "latent_image": ["4",0]
        }},
        "9": {"class_type": "VAEDecode", "inputs": {"vae": ["5",0], "samples": ["8",0]}},
        "10": {"class_type": "SaveImage", "inputs": {"images": ["9",0], "filename_prefix": f"z_ref_{i:03d}"}},
    }
    
    pid = queue(wf)
    print(f"  Ref {i+1}/{NUM_SCENES}: queued")
    res = wait(pid, 180)
    free_vram()
    if res and res['status']['status_str'] == 'success':
        for no, out in res.get("outputs",{}).items():
            for img in out.get("images",[]):
                fp = os.path.join(OUT, img["filename"])
                if os.path.exists(fp):
                    dst = os.path.join(IN, f"ref_{i:03d}.png")
                    shutil.copy(fp, dst)
                    refs.append(dst)
                    eta = (time.time()-start_all)/(i+1)*(NUM_SCENES-i-1)/60
                    print(f"  → ref_{i:03d}.png ({os.path.getsize(fp)/1024:.0f} KB) ETA: {eta:.0f}m")
    else:
        if res:
            for m in res.get('status',{}).get('messages',[]):
                if m[0]=='execution_error':
                    print(f"  ERROR: {m[1]['exception_message'][:200]}")
                    shutil.copy(f"{IN}/ref_2k.png", f"{IN}/ref_{i:03d}.png")
                    refs.append(f"{IN}/ref_{i:03d}.png")
                    print(f"  → (fallback) ref_{i:03d}.png from ref_2k.png")

elapsed = (time.time()-start_all)/60
print(f"\n✅ Step A done: {len(refs)}/{NUM_SCENES} references in {elapsed:.0f}m")
print(f"Run: python3 gen_mv_step_b.py")
