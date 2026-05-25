#!/usr/bin/env python3
"""1-minute music video: 15 scenes × 4s, stitched with crossfade."""
import json, urllib.request, time, subprocess, os

COMFY = "http://127.0.0.1:8188"
OUT = "/home/ericr/ComfyUI/output"
CKPT = "ltx-2.3-22b-distilled-1.1.safetensors"
VIDEO_VAE = "ltx-2.3-22b-distilled_video_vae.safetensors"
TEXT_ENC = "gemma_3_12B_it_fp4_mixed.safetensors"
LTX_LORA = "ltx-2.3-22b-distilled-lora-384-1.1.safetensors"
W, H, FPS = 832, 480, 24
STEPS = 15
PER_SCENE = 4.0
I2V_STR = 0.8
CFG = 3.0

CHAR = "young woman with long dark hair, white dress"
NEG = "headshot, close up, portrait, from behind, back view, ugly, deformed, blurry, low quality"

scenes = [
    {"seed": 100, "env": "misty pine forest path at golden dawn, dappled sunlight through trees, morning fog, cinematic"},
    {"seed": 101, "env": "forest clearing surrounded by tall pine trees, golden rays piercing mist, cinematic"},
    {"seed": 102, "env": "walking on forest path at dawn, warm golden light, foggy atmospheric morning, cinematic"},
    {"seed": 103, "env": "pine forest with golden sunrise light streaming through branches, mist rising, cinematic"},
    {"seed": 104, "env": "deep forest path winding through pines, early morning light, atmospheric haze, cinematic"},
    {"seed": 105, "env": "forest edge where light breaks through mist, golden hour, tall pine silhouettes, cinematic"},
    {"seed": 106, "env": "misty forest trail with dappled morning light, pine needles on ground, cinematic"},
    {"seed": 107, "env": "pine forest at dawn with golden rays, fog swirling between trees, cinematic"},
    {"seed": 108, "env": "forest path covered in morning mist, warm light filtering through, cinematic"},
    {"seed": 109, "env": "clearing in pine forest with golden sunrise, mist rising from ground, cinematic"},
    {"seed": 110, "env": "walking through pine forest at golden dawn, sunbeams through trees, misty, cinematic"},
    {"seed": 111, "env": "pine forest with thick morning fog, golden light breaking through, cinematic"},
    {"seed": 112, "env": "forest trail at dawn, dappled light on path, mist between pine trees, cinematic"},
    {"seed": 113, "env": "pine forest clearing with golden rays and rising mist, early morning, cinematic"},
    {"seed": 114, "env": "misty pine forest path at sunrise, warm golden light, atmospheric depth, cinematic"},
]

full_audio = "/home/ericr/ComfyUI/input/pines_full.mp3"

def queue(wf):
    req = urllib.request.Request(f"{COMFY}/prompt", data=json.dumps({"prompt": wf}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())["prompt_id"]

def wait(pid, timeout=600):
    start = time.time()
    while time.time()-start < timeout:
        try:
            h = json.loads(urllib.request.urlopen(f"{COMFY}/history/{pid}").read())
        except: time.sleep(3); continue
        if pid in h:
            return h[pid]
        time.sleep(3)
    return None

# Generate audio segments (60 seconds = 15 × 4s)
print("Preparing audio segments...")
for i, s in enumerate(scenes):
    seg = f"/home/ericr/ComfyUI/input/seg_{i}.wav"
    if not os.path.exists(seg):
        start_t = i * PER_SCENE
        subprocess.run(["ffmpeg","-y","-i",full_audio,"-ss",str(start_t),"-t",str(PER_SCENE),"-c","copy",seg],
            check=True, capture_output=True)

# Generate ref image if needed
ref = "/home/ericr/ComfyUI/input/ref_2k.png"
if not os.path.exists(ref):
    print("Generating 2K reference...")
    # Use SDXL
    sdxl_wf = {
        "10": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "sd_xl_base_1.0.safetensors"}},
        "20": {"class_type": "CLIPTextEncode", "inputs": {"text": f"{CHAR}, medium shot chest up facing viewer, misty pine forest path at golden dawn, cinematic, photorealistic, sharp", "clip": ["10",1]}},
        "21": {"class_type": "CLIPTextEncode", "inputs": {"text": NEG, "clip": ["10",1]}},
        "22": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        "23": {"class_type": "KSampler", "inputs": {"model": ["10",0], "positive": ["20",0], "negative": ["21",0], "latent_image": ["22",0], "seed": 800, "steps": 30, "cfg": 7.0, "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0}},
        "24": {"class_type": "VAEDecode", "inputs": {"vae": ["10",2], "samples": ["23",0]}},
        "25": {"class_type": "SaveImage", "inputs": {"images": ["24",0], "filename_prefix": "ref_base"}},
    }
    pid = queue(sdxl_wf)
    res = wait(pid, 120)
    if res and res['status']['status_str'] == 'success':
        for no, out in res.get("outputs",{}).items():
            for img in out.get("images",[]):
                fp = os.path.join(OUT, img["filename"])
                import shutil
                shutil.copy(fp, "/home/ericr/ComfyUI/input/ref_2k.png")
                print(f"  ref_2k.png ({os.path.getsize(fp)/1024:.0f} KB)")

# Generate all scenes
clips = []
print(f"\n=== Generating {len(scenes)} scenes (~{len(scenes)*PER_SCENE}s total) ===")
start_all = time.time()
for i, s in enumerate(scenes):
    fc = max(9, ((int(round(PER_SCENE*FPS))-1)//8)*8+1)
    prompt = f"{CHAR}, medium shot facing viewer chest up, {s['env']}, singing softly, cinematic, sharp"
    wf = {
        "10": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
        "11": {"class_type": "VAELoader", "inputs": {"vae_name": VIDEO_VAE}},
        "12": {"class_type": "LTXVAudioVAELoader", "inputs": {"ckpt_name": CKPT}},
        "13": {"class_type": "LTXAVTextEncoderLoader", "inputs": {"text_encoder": TEXT_ENC, "ckpt_name": CKPT, "device": "default"}},
        "20": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["10",0], "lora_name": LTX_LORA, "strength_model": 0.8}},
        "30": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["13",0]}},
        "31": {"class_type": "CLIPTextEncode", "inputs": {"text": NEG, "clip": ["13",0]}},
        "32": {"class_type": "LTXVConditioning", "inputs": {"positive": ["30",0], "negative": ["31",0], "frame_rate": float(FPS)}},
        "40": {"class_type": "LoadImage", "inputs": {"image": "ref_2k.png"}},
        "41": {"class_type": "LoadAudio", "inputs": {"audio": f"seg_{i}.wav"}},
        "42": {"class_type": "LTXVAudioVAEEncode", "inputs": {"audio": ["41",0], "audio_vae": ["12",0]}},
        "43": {"class_type": "EmptyLTXVLatentVideo", "inputs": {"width": W, "height": H, "length": fc, "batch_size": 1}},
        "44": {"class_type": "LTXVImgToVideoInplace", "inputs": {"vae": ["11",0], "image": ["40",0], "latent": ["43",0], "strength": I2V_STR, "bypass": False}},
        "46": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": ["44",0], "audio_latent": ["42",0]}},
        "47": {"class_type": "LTXVSetAudioVideoMaskByTime", "inputs": {"av_latent": ["46",0], "positive": ["32",0], "negative": ["32",1], "model": ["20",0], "vae": ["11",0], "audio_vae": ["12",0], "start_time": 0.0, "end_time": float(PER_SCENE), "video_fps": float(FPS), "mask_video": True, "mask_audio": False, "mask_init_value_video": 0.0, "mask_init_value_audio": 0.0, "slope_len": 3}},
        "50": {"class_type": "RandomNoise", "inputs": {"noise_seed": s["seed"]}},
        "51": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "52": {"class_type": "BasicScheduler", "inputs": {"model": ["20",0], "scheduler": "linear_quadratic", "steps": STEPS, "denoise": 1.0}},
        "53": {"class_type": "CFGGuider", "inputs": {"model": ["20",0], "positive": ["47",0], "negative": ["47",1], "cfg": CFG}},
        "54": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["50",0], "guider": ["53",0], "sampler": ["51",0], "sigmas": ["52",0], "latent_image": ["47",2]}},
        "60": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["54",0]}},
        "61": {"class_type": "LTXVSpatioTemporalTiledVAEDecode", "inputs": {"vae": ["11",0], "latents": ["60",0], "spatial_tiles": 2, "spatial_overlap": 4, "temporal_tile_length": 16, "temporal_overlap": 4, "last_frame_fix": False, "working_device": "auto", "working_dtype": "auto"}},
        "70": {"class_type": "CreateVideo", "inputs": {"images": ["61",0], "fps": float(FPS)}},
        "71": {"class_type": "SaveVideo", "inputs": {"video": ["70",0], "filename_prefix": f"mv_scene_{i}", "format": "mp4", "codec": "h264"}},
    }
    pid = queue(wf)
    print(f"\nScene {i+1}/{len(scenes)}: queued ({s['seed']})")
    res = wait(pid, 600)
    if res and res['status']['status_str'] == 'success':
        for no, out in res.get("outputs",{}).items():
            for v in out.get("images", out.get("media",[])):
                fp = os.path.join(OUT, v.get("filename"))
                if fp and os.path.exists(fp) and fp.endswith('.mp4'):
                    clips.append(fp)
                    elapsed = time.time()-start_all
                    eta = (elapsed/(i+1))*(len(scenes)-i-1)/60
                    print(f"  → {os.path.basename(fp)} ({os.path.getsize(fp)/1024:.0f} KB) | elapsed: {elapsed/60:.0f}m, ETA: {eta:.0f}m")

# Stitch with crossfade
print(f"\n=== Stitching {len(clips)} clips with crossfade ===")
if len(clips) >= 2:
    filter_parts = []
    for i in range(len(clips) - 1):
        offset = (i + 1) * (PER_SCENE - 0.5)
        if i == 0:
            filter_parts.append(f"[0][1]xfade=transition=fade:duration=0.5:offset={offset}[v{i}]")
        else:
            filter_parts.append(f"[v{i-1}][{i+1}]xfade=transition=fade:duration=0.5:offset={offset}[v{i}]")
    
    filter_complex = ";".join(filter_parts)
    final = os.path.join(OUT, "mv_1min_final.mp4")
    
    cmd = ["ffmpeg","-y"]
    for c in clips:
        cmd += ["-i", c]
    
    last_idx = f"v{len(clips)-2}" if len(clips) > 2 else "v0"
    
    if len(clips) == 2:
        cmd += ["-filter_complex", f"[0][1]xfade=transition=fade:duration=0.5:offset={PER_SCENE-0.5}[v]",
                "-map", "[v]", "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p"]
    else:
        cmd += ["-filter_complex", filter_complex, "-map", f"[{last_idx[1:] if 'v' in last_idx else last_idx}]",
                "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p"]
    
    cmd += ["-c:a", "aac", "-b:a", "192k"]
    for i in range(len(clips)):
        cmd += ["-map", f"{i}:a?"]
    
    cmd.append(final)
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError:
        # Fallback to simple concat
        with open("/tmp/mv_concat.txt","w") as f:
            for c in clips: f.write(f"file '{c}'\n")
        raw = os.path.join(OUT, "mv_raw.mp4")
        subprocess.run(["ffmpeg","-y","-f","concat","-safe","0","-i","/tmp/mv_concat.txt","-c","copy",raw],
            check=True, capture_output=True)
        final2 = os.path.join(OUT, "mv_1min_final.mp4")
        subprocess.run(["ffmpeg","-y","-i",raw,"-i",full_audio,
            "-c:v","copy","-c:a","aac","-b:a","192k","-shortest",final2],
            check=True, capture_output=True)
        final = final2
    
    r = subprocess.run(["ffprobe","-v","error","-show_entries","format=duration,size","-of","csv=p=0",final],
        capture_output=True, text=True)
    print(f"\n✅ {os.path.basename(final)} ({r.stdout.strip()})")
elif len(clips):
    print(f"Only {len(clips)} clip(s) generated. No stitch possible.")
else:
    print("No clips generated")
