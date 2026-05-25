#!/usr/bin/env python3
"""
Full pipeline: Z-Image generates reference → LTX I2V + audio → stitch.
Waits for Z-Image downloads if needed.
"""
import json, urllib.request, time, subprocess, os, sys

COMFY = "http://127.0.0.1:8188"
OUT_DIR = "/home/ericr/ComfyUI/output"

# LTX models - use Q3_K_M GGUF for UNet, keep checkpoint for text proj + audio VAE
CKPT = "ltx-2.3-22b-distilled-1.1.safetensors"
UNET_GGUF = "LTX-2.3-22B-distilled-1.1-Q3_K_M.gguf"
VIDEO_VAE = "ltx-2.3-22b-distilled_video_vae.safetensors"
TEXT_ENCODER = "gemma_3_12B_it_fp4_mixed.safetensors"
TEXT_ENCODER_DEVICE = "cpu"  # saves ~4 GB VRAM, text encoding is fast enough on CPU
LTX_LORA = "ltx-2.3-22b-distilled-lora-384-1.1.safetensors"

# Z-Image models
Z_UNET = "z_image_turbo_bf16.safetensors"
Z_CLIP = "qwen_3_4b.safetensors"
Z_VAE = "ae.safetensors"

W, H, FPS = 832, 480, 24
steps = 15
PER_SCENE = 4.0

NEG = "headshot, close up, portrait, from behind, back view, ugly, deformed, blurry, low quality, cartoon"
CHAR_DESC = "young woman with long dark hair wearing a white dress"

scenes = [
    {"env": "walking on misty pine forest path at golden dawn, dappled sunlight through trees, morning fog, cinematic, sharp", "seed": 80},
    {"env": "singing in forest clearing, golden rays piercing through pine trees, mist rising, cinematic, sharp", "seed": 81},
    {"env": "standing on forest path surrounded by towering pine trees, dawn light, fog, cinematic", "seed": 82},
]

def queue(prompt_wf):
    req = urllib.request.Request(f"{COMFY}/prompt", data=json.dumps({"prompt": prompt_wf}).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())["prompt_id"]
    except urllib.error.HTTPError as e:
        err = e.read().decode()[:500]
        print(f"  API Error: {err}")
        return None

def wait_for(pid, timeout=300):
    start = time.time()
    while time.time()-start < timeout:
        try:
            h = json.loads(urllib.request.urlopen(f"{COMFY}/history/{pid}").read())
        except: time.sleep(3); continue
        if pid in h:
            return h[pid]
        time.sleep(3)
    return None

def free_vram():
    """Unload ALL models from VRAM. Call between stages to prevent OOM."""
    req = urllib.request.Request(
        f"{COMFY}/free",
        data=json.dumps({"free_memory": True}).encode(),
        headers={"Content-Type": "application/json"}
    )
    try:
        urllib.request.urlopen(req, timeout=30)
        return True
    except Exception as e:
        print(f"  free_vram error: {e}")
        return False

def wait_for_and_free(pid, timeout=300):
    """Wait for a prompt to finish, then free VRAM."""
    result = wait_for(pid, timeout)
    free_vram()
    return result

# Verify models exist
missing = []
for f, p in [
    (Z_UNET, "/home/ericr/ComfyUI/models/diffusion_models/"),
    (Z_CLIP, "/home/ericr/ComfyUI/models/text_encoders/"),
    (Z_VAE, "/home/ericr/ComfyUI/models/vae/"),
]:
    if not os.path.exists(os.path.join(p, f)):
        missing.append(f)

if missing:
    print(f"Waiting for Z-Image downloads: {missing}")
    print("Check download progress in /tmp/dl_*.log")
    sys.exit(1)

print(f"All Z-Image models found. Generating {len(scenes)} scenes...\n")

# Step 1: Z-Image generates reference images for each scene
refs = []
for i, s in enumerate(scenes):
    prompt = f"{CHAR_DESC}, {s['env']}"
    z_wf = {
        "57": {"class_type": "f2fdebf6-dfaf-43b6-9eb2-7f70613cfdc1", "inputs": {
            "text": prompt, "width": 960, "height": 544,
            "seed": s["seed"], "steps": 4,
            "unet_name": Z_UNET, "clip_name": Z_CLIP, "vae_name": Z_VAE,
        }},
        "58": {"class_type": "SaveImage", "inputs": {"images": ["57",0], "filename_prefix": f"z_ref_{i}"}},
    }
    print(f"Z-Image scene {i+1}: {prompt[:60]}...")
    pid = queue(z_wf)
    if not pid: continue
    result = wait_for_and_free(pid)  # ← frees VRAM for next scene
    if result and result['status']['status_str'] == 'success':
        for no, out in result.get("outputs",{}).items():
            for img in out.get("images",[]):
                fp = os.path.join(OUT_DIR, img["filename"])
                if os.path.exists(fp):
                    # Copy to input for LTX
                    import shutil
                    shutil.copy(fp, f"/home/ericr/ComfyUI/input/z_ref_{i}.png")
                    refs.append(f"z_ref_{i}.png")
                    print(f"  → {img['filename']} ({os.path.getsize(fp)/1024:.0f} KB)")
    else:
        print(f"  FAILED")

if len(refs) != len(scenes):
    print(f"Only got {len(refs)}/{len(scenes)} references. Aborting.")
    sys.exit(1)

# Step 2: LTX I2V + audio for each scene
clips = []
for i, s in enumerate(scenes):
    # Crop audio segment
    seg = f"/home/ericr/ComfyUI/input/z_audio_{i}.wav"
    start_t = i * PER_SCENE
    subprocess.run(["ffmpeg","-y","-i","/home/ericr/ComfyUI/input/pines_vocals.wav",
        "-ss",str(start_t),"-t",str(PER_SCENE),"-c","copy",seg],
        check=True, capture_output=True)

    fc = max(9, ((int(round(PER_SCENE*FPS))-1)//8)*8+1)
    ltx_wf = {
        # UNet: use Q3_K_M GGUF (14 GB) instead of 43 GB checkpoint
        "10": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": UNET_GGUF}},
        # Video VAE
        "11": {"class_type": "VAELoader", "inputs": {"vae_name": VIDEO_VAE}},
        # Audio VAE from checkpoint (loads only ~348 MB of audio VAE weights)
        "12": {"class_type": "LTXVAudioVAELoader", "inputs": {"ckpt_name": CKPT}},
        # Text encoder forced to CPU to save ~4 GB VRAM
        "13": {"class_type": "LTXAVTextEncoderLoader", "inputs": {"text_encoder": TEXT_ENCODER, "ckpt_name": CKPT, "device": TEXT_ENCODER_DEVICE}},
        "20": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["10",0], "lora_name": LTX_LORA, "strength_model": 0.8}},
        "30": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["13",0]}},
        "31": {"class_type": "CLIPTextEncode", "inputs": {"text": NEG, "clip": ["13",0]}},
        "32": {"class_type": "LTXVConditioning", "inputs": {"positive": ["30",0], "negative": ["31",0], "frame_rate": float(FPS)}},
        "40": {"class_type": "LoadImage", "inputs": {"image": refs[i]}},
        "41": {"class_type": "LoadAudio", "inputs": {"audio": f"z_audio_{i}.wav"}},
        "42": {"class_type": "LTXVAudioVAEEncode", "inputs": {"audio": ["41",0], "audio_vae": ["12",0]}},
        "43": {"class_type": "EmptyLTXVLatentVideo", "inputs": {"width": W, "height": H, "length": fc, "batch_size": 1}},
        "44": {"class_type": "LTXVImgToVideoInplace", "inputs": {"vae": ["11",0], "image": ["40",0], "latent": ["43",0], "strength": 0.7, "bypass": False}},
        "46": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": ["44",0], "audio_latent": ["42",0]}},
        "47": {"class_type": "LTXVSetAudioVideoMaskByTime", "inputs": {"av_latent": ["46",0], "positive": ["32",0], "negative": ["32",1], "model": ["20",0], "vae": ["11",0], "audio_vae": ["12",0], "start_time": 0.0, "end_time": float(PER_SCENE), "video_fps": float(FPS), "mask_video": True, "mask_audio": False, "mask_init_value_video": 0.0, "mask_init_value_audio": 0.0, "slope_len": 3}},
        "50": {"class_type": "RandomNoise", "inputs": {"noise_seed": s["seed"]}},
        "51": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "52": {"class_type": "BasicScheduler", "inputs": {"model": ["20",0], "scheduler": "linear_quadratic", "steps": steps, "denoise": 1.0}},
        "53": {"class_type": "CFGGuider", "inputs": {"model": ["20",0], "positive": ["47",0], "negative": ["47",1], "cfg": 3.0}},
        "54": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["50",0], "guider": ["53",0], "sampler": ["51",0], "sigmas": ["52",0], "latent_image": ["47",2]}},
        "60": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["54",0]}},
        "61": {"class_type": "LTXVSpatioTemporalTiledVAEDecode", "inputs": {"vae": ["11",0], "latents": ["60",0], "spatial_tiles": 2, "spatial_overlap": 4, "temporal_tile_length": 16, "temporal_overlap": 4, "last_frame_fix": False, "working_device": "auto", "working_dtype": "auto"}},
        "70": {"class_type": "CreateVideo", "inputs": {"images": ["61",0], "fps": float(FPS)}},
        "71": {"class_type": "SaveVideo", "inputs": {"video": ["70",0], "filename_prefix": f"z_clip_{i}", "format": "mp4", "codec": "h264"}},
    }
    print(f"\nLTX scene {i+1}: {refs[i]}")
    pid = queue(ltx_wf)
    if not pid: continue
    result = wait_for_and_free(pid, timeout=600)  # ← frees VRAM before next scene
    if result and result['status']['status_str'] == 'success':
        for no, out in result.get("outputs",{}).items():
            for v in out.get("images", out.get("media",[])):
                fp = os.path.join(OUT_DIR, v.get("filename"))
                if fp and os.path.exists(fp):
                    clips.append(fp)
                    print(f"  → {os.path.basename(fp)} ({os.path.getsize(fp)/1024:.0f} KB)")

# Step 3: Stitch
print(f"\n=== Stitching {len(clips)} clips ===")
if len(clips) >= 1:
    if len(clips) == 1:
        final = clips[0]
    else:
        with open("/tmp/z_clips.txt", "w") as f:
            for c in clips: f.write(f"file '{c}'\n")
        raw = os.path.join(OUT_DIR, "z_raw.mp4")
        subprocess.run(["ffmpeg","-y","-f","concat","-safe","0","-i","/tmp/z_clips.txt",
            "-c","copy",raw], check=True, capture_output=True)
        final = os.path.join(OUT_DIR, "z_final.mp4")
        full_audio = "/home/ericr/ComfyUI/input/pines_vocals.wav"
        subprocess.run(["ffmpeg","-y","-i",raw,"-i",full_audio,
            "-c:v","copy","-c:a","aac","-b:a","192k","-shortest",final],
            check=True, capture_output=True)
    r = subprocess.run(["ffprobe","-v","error","-show_entries","format=duration,size","-of","csv=p=0",final],
        capture_output=True, text=True)
    print(f"\n✅ {os.path.basename(final)} ({r.stdout.strip()})")
else:
    print("No clips generated")
