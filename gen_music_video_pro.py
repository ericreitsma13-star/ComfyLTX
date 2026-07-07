#!/usr/bin/env python3
"""
Pro Music Video Pipeline: Generate music → scenes → Z-Image refs → LTX video → stitch
Single script, end-to-end. Requires only: transformers, scipy, requests.
"""
import json, urllib.request, time, subprocess, os, sys, math, gc, torch
import scipy.io.wavfile as wavfile

COMFY = "http://127.0.0.1:8188"
OUT = "/home/ericr/ComfyUI/output"
INP = "/home/ericr/ComfyUI/input"
RENDER_DIR = os.path.join(OUT, "pro_mv")
os.makedirs(RENDER_DIR, exist_ok=True)

# ── Music Generation Config ─────────────────────────────────────────────
MUSIC_PROMPT = "aggressive rap beat with heavy 808 bass, sharp hi-hats, synth stabs, dark urban atmosphere, 90 BPM"
MUSIC_DURATION = 40  # seconds
MUSIC_MODEL = "facebook/musicgen-medium"
HEARTMULA_LYRICS = """(Verse 1)
Neon lights on my fur, city reflects in my eyes
Every rooftop's my stage under these midnight skies
Microphone in my paw, spitting fire through the mist
Every shadow in this city knows I exist

(Chorus)
I'm the king of the concrete jungle, the mouse that roared
Running these neon streets, I can't be ignored
Heartbeats match the 808, let the bass explode
Every corner of this cyberpunk city is my episode"""
# ── Video Config ────────────────────────────────────────────────────────
NUM_SCENES = 4
SCENE_DURATION = 10.0
FPS = 24
W, H = 832, 480
# ── Load the proven workflow template ───────────────────────────────────
WORKFLOW_TEMPLATE = "/home/ericr/ComfyUI/workflow_local_gguf_dual.json"

Z_UNET = "z-image-turbo-Q6_K.gguf"
Z_CLIP = "qwen_3_4b.safetensors"
Z_VAE = "ae.safetensors"

NEG = "headshot, close up, portrait, from behind, back view, ugly, deformed, blurry, low quality, cartoon"
CHAR = "anthropomorphic mouse rat, standing upright, wearing streetwear, large round ears, whiskers, snout, furry face"

NEG = "headshot, close up, portrait, from behind, back view, ugly, deformed, blurry, low quality, cartoon, disney"

# Scene prompts for a rap song about anthropomorphic mice
scenes = [
    {"prompt": f"{CHAR} standing on a rooftop overlooking a neon-lit cyberpunk city at night, holding a microphone, rapping with attitude, steam rising from vents, cinematic lighting", "seed": 200},
    {"prompt": f"{CHAR} in a dark alley with dripping water and flickering neon signs, leaning against brick wall, wearing oversized hoodie and chains, shadows, moody", "seed": 201},
    {"prompt": f"{CHAR} performing on stage under spotlights, crowd silhouetted, smoke machine, holding mic, energetic pose, dramatic backlighting", "seed": 202},
    {"prompt": f"{CHAR} sitting on a fire escape at golden hour, looking down at the city, thoughtful expression, warm light, depth of field", "seed": 203},
    {"prompt": f"{CHAR} walking down rain-slicked street at night with posse of other anthropomorphic animals, street lamps reflecting on wet pavement, cinematic wide shot", "seed": 204},
    {"prompt": f"{CHAR} in a graffiti-covered subway station, rapping aggressively, train arriving in background, motion blur, gritty urban atmosphere", "seed": 205},
    {"prompt": f"{CHAR} close up face, intense expression, neon light casting purple and blue across face, whiskers and fur detailed, cinematic portrait", "seed": 206},
    {"prompt": f"{CHAR} standing victorious on rooftop at dawn, arms raised, city skyline behind, golden sunrise, epic hero shot, triumphant mood", "seed": 207},
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
    """Generate LTX video clips using the proven workflow template."""
    with open(WORKFLOW_TEMPLATE) as f:
        base_wf = json.load(f)
    
    clips = []
    for i, (s, ref, seg) in enumerate(zip(scenes, refs, audio_segs)):
        wf = json.loads(json.dumps(base_wf))  # deep copy
        prompt = f"{s['prompt']}, singing passionately, cinematic, sharp"
        
        # Modify nodes per scene
        wf["30"]["inputs"]["text"] = prompt  # CLIPTextEncode positive
        wf["40"]["inputs"]["image"] = os.path.basename(ref)  # LoadImage
        wf["41"]["inputs"]["audio"] = os.path.basename(seg)  # LoadAudio
        wf["47"]["inputs"]["end_time"] = float(SCENE_DURATION)
        
        fc = max(9, ((int(round(SCENE_DURATION * FPS)) - 1) // 8) * 8 + 1)
        wf["43"]["inputs"]["length"] = fc
        wf["70"]["inputs"]["filename_prefix"] = f"pro_clip_{i:03d}"
        
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
        else:
            print(f"    ❌ failed or timeout")
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
