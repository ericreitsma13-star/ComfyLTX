#!/usr/bin/env python3
"""
Pro Music Video Pipeline: Generate music → scenes → Z-Image refs → LTX video → stitch
Single script, end-to-end. Requires only: transformers, scipy, requests.
"""
import json, urllib.request, time, subprocess, os, sys, math, gc, torch
import scipy.io.wavfile as wavfile

# ── Config ──────────────────────────────────────────────────────────────
COMFY = "http://127.0.0.1:8188"
OUT = "/home/ericr/ComfyUI/output"
INP = "/home/ericr/ComfyUI/input"
RENDER_DIR = os.path.join(OUT, "pro_mv")
os.makedirs(RENDER_DIR, exist_ok=True)

# ── Music Generation Config ─────────────────────────────────────────────
MUSIC_PROMPT = "upbeat electronic pop with driving bass, synth pads, female vocals, 120 BPM"
MUSIC_DURATION = 30  # seconds
MUSIC_MODEL = "facebook/musicgen-medium"  # small/medium/large

# ── Video Config ────────────────────────────────────────────────────────
NUM_SCENES = 8
SCENE_DURATION = 4.0
FPS = 24
W, H = 832, 480
STEPS = 15
CFG = 3.0
I2V_STRENGTH = 0.7

# ── Models ──────────────────────────────────────────────────────────────
UNET = "LTX-2.3-22B-distilled-1.1-Q4_K_M.gguf"
VIDEO_VAE = "ltx-2.3-22b-distilled_video_vae.safetensors"
TEXT_ENC = "gemma_3_12B_it_fp4_mixed.safetensors"
CKPT = "ltx-2.3-22b-distilled-1.1.safetensors"
LORA = "ltx-2.3-22b-distilled-lora-384-1.1.safetensors"
IC_LORA = "ltx-2.3-22b-ic-lora-lipdub.safetensors"
Z_UNET = "z-image-turbo-Q6_K.gguf"
Z_CLIP = "qwen_3_4b.safetensors"
Z_VAE = "ae.safetensors"

NEG = "headshot, close up, portrait, from behind, back view, ugly, deformed, blurry, low quality, cartoon"
CHAR = "young woman with long dark hair"

# Scene prompts (generated from lyrics in production)
scenes = [
    {"prompt": f"{CHAR} walking through neon-lit city streets at night, rain on pavement, reflections, cyberpunk", "seed": 100},
    {"prompt": f"{CHAR} standing on rooftop overlooking futuristic city, wind in hair, neon glow", "seed": 101},
    {"prompt": f"{CHAR} in crowded nightclub, laser lights, smoke machine, dancing", "seed": 102},
    {"prompt": f"{CHAR} sitting alone at rainy window, city lights outside, melancholy mood", "seed": 103},
    {"prompt": f"{CHAR} running through alleyway with neon signs, steam rising from vents, cinematic", "seed": 104},
    {"prompt": f"{CHAR} on stage performing, spotlight, crowd silhouettes, dramatic lighting", "seed": 105},
    {"prompt": f"{CHAR} walking away from explosion, slow motion, debris flying, epic", "seed": 106},
    {"prompt": f"{CHAR} close up face, tears of joy, sunrise over city behind, hopeful", "seed": 107},
]

# ══════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════

def free_vram():
    try:
        req = urllib.request.Request(f"{COMFY}/free", data=json.dumps({"free_memory": True}).encode(), headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=30)
    except: pass

def queue(prompt_wf):
    req = urllib.request.Request(f"{COMFY}/prompt", data=json.dumps({"prompt": prompt_wf}).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())["prompt_id"]

def wait(pid, timeout=600):
    start = time.time()
    while time.time() - start < timeout:
        try:
            h = json.loads(urllib.request.urlopen(f"{COMFY}/history/{pid}").read())
        except: time.sleep(3); continue
        if pid in h: return h[pid]
        time.sleep(3)
    return None

# ══════════════════════════════════════════════════════════════════════════
# STAGE 0: GENERATE MUSIC
# ══════════════════════════════════════════════════════════════════════════

def generate_music(prompt, duration=30, output_path="generated_music.wav"):
    """Generate music from text using MusicGen via transformers."""
    from transformers import MusicgenForConditionalGeneration, AutoProcessor
    print(f"🎵 Generating music: '{prompt}' ({duration}s) ...")
    
    device = "cpu"  # CPU is fine for one-time generation, saves VRAM for video
    processor = AutoProcessor.from_pretrained(MUSIC_MODEL)
    model = MusicgenForConditionalGeneration.from_pretrained(MUSIC_MODEL).to(device)
    
    inputs = processor(
        text=[prompt],
        padding=True,
        return_tensors="pt",
    ).to(device)
    
    # Calculate tokens needed for duration (MusicGen generates at 320 tokens/sec at 32kHz)
    max_new_tokens = int(duration * 50)  # 50 tokens/sec
    
    with torch.no_grad():
        audio_values = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            guidance_scale=3.0,
        )
    
    audio = audio_values[0, 0].cpu().numpy()
    sample_rate = model.config.audio_encoder.sampling_rate
    wavfile.write(output_path, sample_rate, audio)
    
    del model; torch.cuda.empty_cache(); gc.collect()
    print(f"  → {output_path} ({duration}s @ {sample_rate}Hz)")
    return output_path

# ══════════════════════════════════════════════════════════════════════════
# STAGE 1: SPLIT AUDIO
# ══════════════════════════════════════════════════════════════════════════

def split_audio(audio_path, num_scenes, duration):
    """Split audio into scene-length segments + create per-scene audio files."""
    print(f"🔊 Splitting audio into {num_scenes} segments...")
    segs = []
    for i in range(num_scenes):
        seg = os.path.join(INP, f"pro_scene_{i:03d}.wav")
        start = i * duration
        subprocess.run(["ffmpeg", "-y", "-i", audio_path, "-ss", str(start),
            "-t", str(duration), seg], capture_output=True)
        segs.append(seg)
    return segs

# ══════════════════════════════════════════════════════════════════════════
# STAGE 2: GENERATE Z-IMAGE REFERENCES
# ══════════════════════════════════════════════════════════════════════════

def generate_refs():
    """Generate Z-Image reference images for each scene via ComfyUI API."""
    print(f"\n🖼️ Generating Z-Image references...")
    refs = []
    for i, s in enumerate(scenes):
        prompt = s["prompt"]
        wf = {
            "1": {"class_type": "CLIPLoader", "inputs": {"clip_name": Z_CLIP, "type": "qwen_image"}},
            "2": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["1",0]}},
            "3": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": Z_UNET}},
            "4": {"class_type": "EmptyLatentImage", "inputs": {"width": 960, "height": 544, "batch_size": 1}},
            "5": {"class_type": "VAELoader", "inputs": {"vae_name": Z_VAE}},
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
            "10": {"class_type": "SaveImage", "inputs": {"images": ["9",0], "filename_prefix": f"pro_ref_{i:03d}"}},
        }
        pid = queue(wf)
        print(f"  Scene {i+1}/{NUM_SCENES}: queued")
        res = wait(pid, 180)
        free_vram()
        if res and res['status']['status_str'] == 'success':
            for no, out in res.get("outputs",{}).items():
                for img in out.get("images",[]):
                    fp = os.path.join(OUT, img["filename"])
                    if os.path.exists(fp):
                        dst = os.path.join(INP, f"pro_ref_{i:03d}.png")
                        import shutil; shutil.copy(fp, dst)
                        refs.append(dst)
                        print(f"    → ref_{i:03d}.png")
        else:
            # Fallback
            if os.path.exists(f"{INP}/ref_2k.png"):
                shutil.copy(f"{INP}/ref_2k.png", f"{INP}/pro_ref_{i:03d}.png")
                refs.append(f"{INP}/pro_ref_{i:03d}.png")
    return refs

# ══════════════════════════════════════════════════════════════════════════
# STAGE 3: GENERATE LTX VIDEOS
# ══════════════════════════════════════════════════════════════════════════

def generate_videos(refs, audio_segs, use_lip_sync=False):
    """Generate LTX video clips with optional IC-LoRA lip sync."""
    print(f"\n🎬 Generating LTX videos..." if not use_lip_sync else f"\n🎬 Generating LTX videos with IC-LoRA lip sync...")
    clips = []
    for i, (s, ref, seg) in enumerate(zip(scenes, refs, audio_segs)):
        prompt = f"{s['prompt']}, singing passionately, cinematic, sharp"
        fc = max(9, ((int(round(SCENE_DURATION * FPS)) - 1) // 8) * 8 + 1)
        
        wf = {
            "10": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": UNET}},
            "11": {"class_type": "VAELoader", "inputs": {"vae_name": VIDEO_VAE}},
            "12": {"class_type": "LTXVAudioVAELoader", "inputs": {"ckpt_name": CKPT}},
            "13": {"class_type": "LTXAVTextEncoderLoader", "inputs": {"text_encoder": TEXT_ENC, "ckpt_name": CKPT, "device": "cpu"}},
            "20": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["10",0], "lora_name": LORA, "strength_model": 0.8}},
            "30": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["13",0]}},
            "31": {"class_type": "CLIPTextEncode", "inputs": {"text": NEG, "clip": ["13",0]}},
            "32": {"class_type": "LTXVConditioning", "inputs": {"positive": ["30",0], "negative": ["31",0], "frame_rate": float(FPS)}},
            "40": {"class_type": "LoadImage", "inputs": {"image": os.path.basename(ref)}},
            "41": {"class_type": "LoadAudio", "inputs": {"audio": os.path.basename(seg)}},
            "42": {"class_type": "LTXVAudioVAEEncode", "inputs": {"audio": ["41",0], "audio_vae": ["12",0]}},
            "43": {"class_type": "EmptyLTXVLatentVideo", "inputs": {"width": W, "height": H, "length": fc, "batch_size": 1}},
            "44": {"class_type": "LTXVImgToVideoInplace", "inputs": {"vae": ["11",0], "image": ["40",0], "latent": ["43",0], "strength": I2V_STRENGTH, "bypass": False}},
            "46": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": ["44",0], "audio_latent": ["42",0]}},
            "47": {"class_type": "LTXVSetAudioVideoMaskByTime", "inputs": {"av_latent": ["46",0], "positive": ["32",0], "negative": ["32",1], "model": ["20",0], "vae": ["11",0], "audio_vae": ["12",0], "start_time": 0.0, "end_time": float(SCENE_DURATION), "video_fps": float(FPS), "mask_video": True, "mask_audio": False, "mask_init_value_video": 0.0, "mask_init_value_audio": 0.0, "slope_len": 3}},
            "50": {"class_type": "RandomNoise", "inputs": {"noise_seed": s["seed"]}},
            "51": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
            "52": {"class_type": "BasicScheduler", "inputs": {"model": ["20",0], "scheduler": "linear_quadratic", "steps": STEPS, "denoise": 1.0}},
            "53": {"class_type": "CFGGuider", "inputs": {"model": ["20",0], "positive": ["47",0], "negative": ["47",1], "cfg": CFG}},
            "54": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["50",0], "guider": ["53",0], "sampler": ["51",0], "sigmas": ["52",0], "latent_image": ["47",2]}},
            "60": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["54",0]}},
            "61": {"class_type": "LTXVSpatioTemporalTiledVAEDecode", "inputs": {"vae": ["11",0], "latents": ["60",0], "spatial_tiles": 2, "spatial_overlap": 4, "temporal_tile_length": 16, "temporal_overlap": 4, "last_frame_fix": False, "working_device": "auto", "working_dtype": "auto"}},
        }
        
        if use_lip_sync:
            # IC-LoRA for lip sync
            wf["20"] = {"class_type": "LTXICLoRALoaderModelOnly", "inputs": {"model": ["10",0], "lora_name": IC_LORA, "strength_model": 0.5}}
            wf["45"] = {"class_type": "LTXAddVideoICLoRAGuide", "inputs": {"positive": ["32",0], "negative": ["32",1], "vae": ["11",0], "latent": ["44",0], "image": ["40",0], "frame_idx": 0, "strength": 0.3, "latent_downscale_factor": 2, "crop": "center", "use_tiled_encode": False, "tile_size": 64, "tile_overlap": 16}}
            wf["47"] = {"class_type": "LTXVSetAudioVideoMaskByTime", "inputs": {"av_latent": ["46",0], "positive": ["45",0], "negative": ["45",1], "model": ["20",0], "vae": ["11",0], "audio_vae": ["12",0], "start_time": 0.0, "end_time": float(SCENE_DURATION), "video_fps": float(FPS), "mask_video": True, "mask_audio": False, "mask_init_value_video": 0.0, "mask_init_value_audio": 0.0, "slope_len": 3}}
        
        wf["70"] = {"class_type": "CreateVideo", "inputs": {"images": ["61",0], "fps": float(FPS)}}
        wf["71"] = {"class_type": "SaveVideo", "inputs": {"video": ["70",0], "filename_prefix": f"pro_clip_{i:03d}", "format": "mp4", "codec": "h264"}}
        
        pid = queue(wf)
        print(f"  Scene {i+1}/{NUM_SCENES}: queued")
        res = wait(pid, 600)
        free_vram()
        if res and res['status']['status_str'] == 'success':
            for no, out in res.get("outputs",{}).items():
                for v in out.get("images", out.get("media",[])):
                    fp = os.path.join(OUT, v.get("filename"))
                    if fp and os.path.exists(fp):
                        clips.append(fp)
                        print(f"    → clip_{i:03d}.mp4")
    return clips

# ══════════════════════════════════════════════════════════════════════════
# STAGE 4: STITCH
# ══════════════════════════════════════════════════════════════════════════

def stitch(clips, audio_path, output="pro_mv_final.mp4"):
    """Stitch clips together with crossfade and original audio."""
    print(f"\n🎞️ Stitching {len(clips)} clips...")
    if len(clips) >= 2:
        with open("/tmp/pro_stitch.txt", "w") as f:
            for c in clips: f.write(f"file '{c}'\n")
        raw = os.path.join(RENDER_DIR, "pro_raw.mp4")
        subprocess.run(["ffmpeg","-y","-f","concat","-safe","0","-i","/tmp/pro_stitch.txt",
            "-c","copy",raw], check=True, capture_output=True)
        final = os.path.join(RENDER_DIR, output)
        subprocess.run(["ffmpeg","-y","-i",raw,"-i",audio_path,
            "-c:v","copy","-c:a","aac","-b:a","192k","-shortest",final],
            check=False, capture_output=True)
    elif len(clips) == 1:
        final = clips[0]
    else:
        print("No clips to stitch")
        return None
    
    r = subprocess.run(["ffprobe","-v","error","-show_entries","format=duration,size","-of","csv=p=0",final],
        capture_output=True, text=True)
    print(f"✅ {output} ({r.stdout.strip()})")
    return final

# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    start_all = time.time()
    
    # Generate music
    audio_path = os.path.join(RENDER_DIR, "pro_music.wav")
    generate_music(MUSIC_PROMPT, MUSIC_DURATION, audio_path)
    
    # Split audio into scene segments
    segs = split_audio(audio_path, NUM_SCENES, SCENE_DURATION)
    
    # Generate Z-Image reference images
    refs = generate_refs()
    if len(refs) < NUM_SCENES:
        print(f"⚠️ Only got {len(refs)}/{NUM_SCENES} references, continuing anyway")
    
    # Generate video clips (choose lip_sync=True for IC-LoRA lip sync)
    clips = generate_videos(refs, segs, use_lip_sync=False)
    
    # Stitch final video
    if clips:
        stitch(clips, audio_path)
    
    elapsed = (time.time() - start_all) / 60
    print(f"\n⏱️ Total: {elapsed:.0f}m")
    print(f"📁 Output: {RENDER_DIR}")
