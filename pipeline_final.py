#!/usr/bin/env python3
"""
Full pipeline: SDXL refs → LTX I2V + audio → VRGDG stitch.
Uses our high-res 2K reference image.
"""
import json, urllib.request, time, subprocess, os, shutil

COMFY = "http://127.0.0.1:8188"
OUT = "/home/ericr/ComfyUI/output"
SDXL = "sd_xl_base_1.0.safetensors"
CKPT = "ltx-2.3-22b-distilled-1.1.safetensors"
VIDEO_VAE = "ltx-2.3-22b-distilled_video_vae.safetensors"
TEXT_ENC = "gemma_3_12B_it_fp4_mixed.safetensors"
LTX_LORA = "ltx-2.3-22b-distilled-lora-384-1.1.safetensors"

W, H, FPS = 832, 480, 24
STEPS = 15
DURATION = 4.0
I2V_STR = 0.8

CHAR = "young woman with long dark hair, white dress"
NEG = "headshot, close up, portrait, from behind, back view, ugly, deformed, blurry, low quality"

scenes = [
    {"seed": 100, "env": "misty pine forest path at golden dawn, dappled sunlight through trees, morning fog"},
    {"seed": 101, "env": "forest clearing surrounded by tall pines, golden rays piercing mist, cinematic"},
    {"seed": 102, "env": "walking on forest path, warm golden dawn light, foggy atmospheric morning"},
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

# Generate per-scene audio crops using VRGDG_AudioCrop
audio_segs = []
for i, s in enumerate(scenes):
    seg_path = f"/home/ericr/ComfyUI/input/seg_{i}.wav"
    start_t = i * DURATION
    subprocess.run(["ffmpeg","-y","-i",full_audio,"-ss",str(start_t),"-t",str(DURATION),"-c","copy",seg_path],
        check=True, capture_output=True)
    audio_segs.append(f"seg_{i}.wav")

# Step 1: SDXL generates per-scene references (img2img from 2K ref for consistency)
refs = []
print("\n=== Generating per-scene references ===")
for i, s in enumerate(scenes):
    prompt = f"{CHAR}, medium shot chest up facing viewer, {s['env']}, cinematic, photorealistic, sharp, highly detailed"
    i2i_wf = {
        "10": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": SDXL}},
        "20": {"class_type": "LoadImage", "inputs": {"image": "ref_2k.png"}},
        "21": {"class_type": "ImageScale", "inputs": {"upscale_method": "lanczos", "image": ["20",0], "width": 1024, "height": 1024, "crop": "disabled"}},
        "22": {"class_type": "VAEEncode", "inputs": {"vae": ["10",2], "pixels": ["21",0]}},
        "30": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["10",1]}},
        "31": {"class_type": "CLIPTextEncode", "inputs": {"text": NEG, "clip": ["10",1]}},
        "32": {"class_type": "KSampler", "inputs": {"model": ["10",0], "positive": ["30",0], "negative": ["31",0], "latent_image": ["22",0], "seed": s["seed"], "steps": 25, "cfg": 6.0, "sampler_name": "euler", "scheduler": "normal", "denoise": 0.45}},
        "33": {"class_type": "VAEDecode", "inputs": {"vae": ["10",2], "samples": ["32",0]}},
        "34": {"class_type": "SaveImage", "inputs": {"images": ["33",0], "filename_prefix": f"ref_scene_{i}"}},
    }
    pid = queue(i2i_wf)
    res = wait(pid, 120)
    if res and res['status']['status_str'] == 'success':
        for no, out in res.get("outputs",{}).items():
            for img in out.get("images",[]):
                fp = os.path.join(OUT, img["filename"])
                shutil.copy(fp, f"/home/ericr/ComfyUI/input/ref_s{i}.png")
                refs.append(f"ref_s{i}.png")
                print(f"  Scene {i+1}: {img['filename']} ({os.path.getsize(fp)/1024:.0f} KB)")

if len(refs) != len(scenes):
    print(f"Got {len(refs)}/{len(scenes)} refs, aborting")
    exit(1)

# Step 2: LTX I2V + audio for each scene
clips = []
print("\n=== Generating video scenes ===")
for i, s in enumerate(scenes):
    fc = max(9, ((int(round(DURATION*FPS))-1)//8)*8+1)
    prompt = f"{CHAR}, medium shot facing viewer chest up, {s['env']}, singing, cinematic"
    ltx_wf = {
        "10": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
        "11": {"class_type": "VAELoader", "inputs": {"vae_name": VIDEO_VAE}},
        "12": {"class_type": "LTXVAudioVAELoader", "inputs": {"ckpt_name": CKPT}},
        "13": {"class_type": "LTXAVTextEncoderLoader", "inputs": {"text_encoder": TEXT_ENC, "ckpt_name": CKPT, "device": "default"}},
        "20": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["10",0], "lora_name": LTX_LORA, "strength_model": 0.8}},
        "30": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["13",0]}},
        "31": {"class_type": "CLIPTextEncode", "inputs": {"text": NEG, "clip": ["13",0]}},
        "32": {"class_type": "LTXVConditioning", "inputs": {"positive": ["30",0], "negative": ["31",0], "frame_rate": float(FPS)}},
        "40": {"class_type": "LoadImage", "inputs": {"image": refs[i]}},
        "41": {"class_type": "LoadAudio", "inputs": {"audio": audio_segs[i]}},
        "42": {"class_type": "LTXVAudioVAEEncode", "inputs": {"audio": ["41",0], "audio_vae": ["12",0]}},
        "43": {"class_type": "EmptyLTXVLatentVideo", "inputs": {"width": W, "height": H, "length": fc, "batch_size": 1}},
        "44": {"class_type": "LTXVImgToVideoInplace", "inputs": {"vae": ["11",0], "image": ["40",0], "latent": ["43",0], "strength": I2V_STR, "bypass": False}},
        "46": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": ["44",0], "audio_latent": ["42",0]}},
        "47": {"class_type": "LTXVSetAudioVideoMaskByTime", "inputs": {"av_latent": ["46",0], "positive": ["32",0], "negative": ["32",1], "model": ["20",0], "vae": ["11",0], "audio_vae": ["12",0], "start_time": 0.0, "end_time": float(DURATION), "video_fps": float(FPS), "mask_video": True, "mask_audio": False, "mask_init_value_video": 0.0, "mask_init_value_audio": 0.0, "slope_len": 3}},
        "50": {"class_type": "RandomNoise", "inputs": {"noise_seed": s["seed"]}},
        "51": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "52": {"class_type": "BasicScheduler", "inputs": {"model": ["20",0], "scheduler": "linear_quadratic", "steps": STEPS, "denoise": 1.0}},
        "53": {"class_type": "CFGGuider", "inputs": {"model": ["20",0], "positive": ["47",0], "negative": ["47",1], "cfg": 3.0}},
        "54": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["50",0], "guider": ["53",0], "sampler": ["51",0], "sigmas": ["52",0], "latent_image": ["47",2]}},
        "60": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["54",0]}},
        "61": {"class_type": "LTXVSpatioTemporalTiledVAEDecode", "inputs": {"vae": ["11",0], "latents": ["60",0], "spatial_tiles": 2, "spatial_overlap": 4, "temporal_tile_length": 16, "temporal_overlap": 4, "last_frame_fix": False, "working_device": "auto", "working_dtype": "auto"}},
        "70": {"class_type": "CreateVideo", "inputs": {"images": ["61",0], "fps": float(FPS)}},
        "71": {"class_type": "SaveVideo", "inputs": {"video": ["70",0], "filename_prefix": f"mv_scene_{i}", "format": "mp4", "codec": "h264"}},
    }
    pid = queue(ltx_wf)
    print(f"  Scene {i+1}: queued {pid}")
    res = wait(pid, 600)
    if res and res['status']['status_str'] == 'success':
        for no, out in res.get("outputs",{}).items():
            for v in out.get("images", out.get("media",[])):
                fp = os.path.join(OUT, v.get("filename"))
                if fp and os.path.exists(fp):
                    clips.append(fp)
                    print(f"  → {os.path.basename(fp)} ({os.path.getsize(fp)/1024:.0f} KB)")

# Step 3: Stitch with FFmpeg
print(f"\n=== Stitching {len(clips)} clips ===")
if len(clips) >= 1:
    if len(clips) == 1:
        raw = clips[0]
    else:
        with open("/tmp/mv_stitch.txt","w") as f:
            for c in clips: f.write(f"file '{c}'\n")
        raw = os.path.join(OUT, "mv_stitched_raw.mp4")
        subprocess.run(["ffmpeg","-y","-f","concat","-safe","0","-i","/tmp/mv_stitch.txt",
            "-c","copy",raw],check=True,capture_output=True)
    final = os.path.join(OUT, "mv_final_2k_ref.mp4")
    subprocess.run(["ffmpeg","-y","-i",raw,"-i",full_audio,
        "-c:v","copy","-c:a","aac","-b:a","192k","-shortest",final],
        check=True,capture_output=True)
    r = subprocess.run(["ffprobe","-v","error","-show_entries","format=duration,size","-of","csv=p=0",final],
        capture_output=True,text=True)
    print(f"\n✅ {os.path.basename(final)} ({r.stdout.strip()})")
