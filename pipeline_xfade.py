#!/usr/bin/env python3
"""Pipeline with crossfade transitions and proper audio sync."""
import json, urllib.request, time, subprocess, os, shutil

COMFY = "http://127.0.0.1:8188"
OUT = "/home/ericr/ComfyUI/output"
CKPT = "ltx-2.3-22b-distilled-1.1.safetensors"
VIDEO_VAE = "ltx-2.3-22b-distilled_video_vae.safetensors"
TEXT_ENC = "gemma_3_12B_it_fp4_mixed.safetensors"
LTX_LORA = "ltx-2.3-22b-distilled-lora-384-1.1.safetensors"

W, H, FPS = 832, 480, 24
STEPS = 15
DURATION = 4.0
I2V_STR = 0.8
CFG = 3.0

CHAR = "young woman with long dark hair, white dress"
NEG = "headshot, close up, portrait, from behind, back view, ugly, deformed, blurry, low quality"

scenes = [
    {"seed": 100, "env": "misty pine forest path at golden dawn, dappled sunlight through trees, morning fog, cinematic"},
    {"seed": 101, "env": "forest clearing surrounded by tall pine trees, golden rays piercing mist, cinematic"},
    {"seed": 102, "env": "walking on forest path at dawn, warm golden light, foggy atmospheric morning, cinematic"},
]

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

full_audio = "/home/ericr/ComfyUI/input/pines_vocals.wav"
full_dur = float(subprocess.run(["ffprobe","-v","error","-show_entries","format=duration","-of","csv=p=0",full_audio],
    capture_output=True,text=True).stdout.strip())
print(f"Audio: {full_dur:.1f}s")

# Generate audio segments
for i in range(len(scenes)):
    seg = f"/home/ericr/ComfyUI/input/seg_{i}.wav"
    subprocess.run(["ffmpeg","-y","-i",full_audio,"-ss",str(i*DURATION),"-t",str(DURATION),"-c","copy",seg],
        check=True,capture_output=True)

# Generate all scenes with model-decoded audio
clips = []
audio_files = []
print(f"\n=== Generating {len(scenes)} scenes ===")
for i, s in enumerate(scenes):
    fc = max(9, ((int(round(DURATION*FPS))-1)//8)*8+1)
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
        "40": {"class_type": "LoadImage", "inputs": {"image": f"ref_s{i}.png"}},
        "41": {"class_type": "LoadAudio", "inputs": {"audio": f"seg_{i}.wav"}},
        "42": {"class_type": "LTXVAudioVAEEncode", "inputs": {"audio": ["41",0], "audio_vae": ["12",0]}},
        "43": {"class_type": "EmptyLTXVLatentVideo", "inputs": {"width": W, "height": H, "length": fc, "batch_size": 1}},
        "44": {"class_type": "LTXVImgToVideoInplace", "inputs": {"vae": ["11",0], "image": ["40",0], "latent": ["43",0], "strength": I2V_STR, "bypass": False}},
        "46": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": ["44",0], "audio_latent": ["42",0]}},
        "47": {"class_type": "LTXVSetAudioVideoMaskByTime", "inputs": {"av_latent": ["46",0], "positive": ["32",0], "negative": ["32",1], "model": ["20",0], "vae": ["11",0], "audio_vae": ["12",0], "start_time": 0.0, "end_time": float(DURATION), "video_fps": float(FPS), "mask_video": True, "mask_audio": False, "mask_init_value_video": 0.0, "mask_init_value_audio": 0.0, "slope_len": 3}},
        "50": {"class_type": "RandomNoise", "inputs": {"noise_seed": s["seed"]}},
        "51": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "52": {"class_type": "BasicScheduler", "inputs": {"model": ["20",0], "scheduler": "linear_quadratic", "steps": STEPS, "denoise": 1.0}},
        "53": {"class_type": "CFGGuider", "inputs": {"model": ["20",0], "positive": ["47",0], "negative": ["47",1], "cfg": CFG}},
        "54": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["50",0], "guider": ["53",0], "sampler": ["51",0], "sigmas": ["52",0], "latent_image": ["47",2]}},
        "60": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["54",0]}},
        "61": {"class_type": "LTXVSpatioTemporalTiledVAEDecode", "inputs": {"vae": ["11",0], "latents": ["60",0], "spatial_tiles": 2, "spatial_overlap": 4, "temporal_tile_length": 16, "temporal_overlap": 4, "last_frame_fix": False, "working_device": "auto", "working_dtype": "auto"}},
        "70": {"class_type": "CreateVideo", "inputs": {"images": ["61",0], "fps": float(FPS)}},
        "71": {"class_type": "SaveVideo", "inputs": {"video": ["70",0], "filename_prefix": f"sc_{i}", "format": "mp4", "codec": "h264"}},
        "80": {"class_type": "LTXVAudioVAEDecode", "inputs": {"samples": ["60",1], "audio_vae": ["12",0]}},
        "81": {"class_type": "SaveAudio", "inputs": {"audio": ["80",0], "filename_prefix": f"aud_{i}"}},
    }
    pid = queue(wf)
    print(f"  Scene {i+1}: queued")
    res = wait(pid, 600)
    if res and res['status']['status_str'] == 'success':
        for no, out in res.get("outputs",{}).items():
            for v in out.get("images", out.get("media",[])):
                fp = os.path.join(OUT, v.get("filename"))
                if fp and os.path.exists(fp) and fp.endswith('.mp4'):
                    clips.append(fp)
                    print(f"  → video: {os.path.basename(fp)} ({os.path.getsize(fp)/1024:.0f} KB)")
            for v in out.get("audio",[]):
                af = os.path.join(OUT, v.get("filename"))
                if af and os.path.exists(af):
                    audio_files.append(af)
                    print(f"  → audio: {os.path.basename(af)} ({os.path.getsize(af)/1024:.0f} KB)")

# Stitch with crossfade transitions
print(f"\n=== Stitching {len(clips)} clips with crossfade ===")
if len(clips) >= 2:
    # Build FFmpeg xfade filter
    filter_parts = []
    inputs = ""
    for i in range(len(clips)):
        inputs += f"[{i}:v]"
    
    for i in range(len(clips) - 1):
        offset = (i + 1) * (DURATION - 0.5)
        transition = f"xfade=transition=fade:duration=0.5:offset={offset}"
        if i == 0:
            filter_parts.append(f"[0][1]{transition}[v{i}]")
        else:
            filter_parts.append(f"[v{i-1}][{i+1}]{transition}[v{i}]")
    
    last_label = f"v{len(clips)-2}" if len(clips) > 2 else "v0"
    if len(clips) == 2:
        last_label = "v0"
    
    filter_complex = ";".join(filter_parts)
    
    final = os.path.join(OUT, "mv_crossfade.mp4")
    cmd = ["ffmpeg","-y"]
    for c in clips:
        cmd += ["-i", c]
    
    if len(clips) == 2:
        filter_str = f"[0][1]xfade=transition=fade:duration=0.5:offset={DURATION-0.5}[v]"
        cmd += ["-filter_complex", filter_str, "-map", "[v]"]
    else:
        cmd += ["-filter_complex", filter_complex, "-map", f"[{last_label.split('v')[1] if 'v' in last_label else last_label}]"]
    
    cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p"]
    
    # Try to use model-decoded audio if available
    if len(audio_files) == len(clips):
        print("Using model-decoded audio for sync...")
        # Mix all audio files
        amix = ",".join([f"[{i}:a]" for i in range(len(clips))])
        amix_inputs = "".join([f"[{i}:a]" for i in range(len(clips))])
        if len(clips) == 3:
            cmd += ["-filter_complex", f"[0][1]xfade=transition=fade:duration=0.5:offset={DURATION-0.5}[v];{amix_inputs}amix=inputs={len(clips)}:duration=first[a]",
                    "-map", "[v]", "-map", "[a]"]
        elif len(clips) == 2:
            cmd += ["-filter_complex", f"[0][1]xfade=transition=fade:duration=0.5:offset={DURATION-0.5}[v];{amix_inputs}amix=inputs={len(clips)}:duration=first[a]",
                    "-map", "[v]", "-map", "[a]"]
    else:
        cmd += ["-c:a", "aac", "-b:a", "192k"]
        for i in range(len(clips)):
            cmd += ["-map", f"{i}:a"]
    
    cmd.append(final)
    
    try:
        r = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"→ {os.path.basename(final)} ({r.stdout.strip()})")
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e.stderr[-200:] if e.stderr else 'no stderr'}")
        # Fallback: simple concat
        with open("/tmp/mv_concat.txt","w") as f:
            for c in clips: f.write(f"file '{c}'\n")
        raw = os.path.join(OUT, "mv_raw.mp4")
        subprocess.run(["ffmpeg","-y","-f","concat","-safe","0","-i","/tmp/mv_concat.txt","-c","copy",raw],
            check=True, capture_output=True)
        final2 = os.path.join(OUT, "mv_crossfade_fb.mp4")
        subprocess.run(["ffmpeg","-y","-i",raw,"-i",full_audio,
            "-c:v","copy","-c:a","aac","-b:a","192k","-shortest",final2],
            check=True, capture_output=True)
        final = final2
        r2 = subprocess.run(["ffprobe","-v","error","-show_entries","format=duration,size","-of","csv=p=0",final],
            capture_output=True, text=True)
        print(f"→ {os.path.basename(final)} (fallback: {r2.stdout.strip()})")
    
    r = subprocess.run(["ffprobe","-v","error","-show_entries","format=duration,size","-of","csv=p=0",final],
        capture_output=True, text=True)
    print(f"\n✅ {os.path.basename(final)} ({r.stdout.strip()})")
elif len(clips) == 1:
    final = clips[0]
    print(f"Only 1 clip: {final}")
else:
    print("No clips generated")
