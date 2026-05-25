#!/usr/bin/env python3
"""Multi-scene music video: 4 scenes stitched with transitions."""
import json, urllib.request, time, subprocess, os, shutil

COMFY = "http://127.0.0.1:8188"
OUTPUT_DIR = "/home/ericr/ComfyUI/output"
CKPT = "ltx-2.3-22b-distilled-1.1.safetensors"
VIDEO_VAE = "ltx-2.3-22b-distilled_video_vae.safetensors"
TEXT_ENCODER = "gemma_3_12B_it_fp4_mixed.safetensors"
LORA = "ltx-2.3-22b-distilled-lora-384-1.1.safetensors"
LORA_STRENGTH = 0.8
W, H, FPS = 832, 480, 24
steps = 15
PER_SCENE = 3.0  # seconds per scene (total 12s = 4 scenes)
NEG = "headshot, close up, portrait, from behind, back view, ugly, deformed, blurry, low quality, cartoon"

scenes = [
    {"prompt": "young woman with long dark hair wearing white dress, medium shot facing viewer, walking on misty pine forest path, golden dawn light, cinematic, sharp", "seed": 60, "start": 0},
    {"prompt": "young woman with long dark hair white dress, medium shot waist up, singing in misty pine forest clearing, dappled sunlight through trees, cinematic, sharp", "seed": 61, "start": 3},
    {"prompt": "young woman with long dark hair white dress, medium shot facing camera, walking through pine trees with morning fog, golden rays, cinematic, sharp detail", "seed": 62, "start": 6},
    {"prompt": "young woman with long dark hair white dress, medium shot front view, singing on forest path at dawn, mist rising, warm golden light, cinematic, sharp", "seed": 63, "start": 9},
]

# Crop full audio into segments
full_audio = "/home/ericr/ComfyUI/input/pines_vocals.wav"
for i, s in enumerate(scenes):
    seg = f"/home/ericr/ComfyUI/input/seg_{i}.wav"
    if not os.path.exists(seg):
        subprocess.run(["ffmpeg","-y","-i",full_audio, "-ss",str(s["start"]),
            "-t",str(PER_SCENE),"-c","copy",seg],check=True,capture_output=True)
    s["audio_file"] = f"seg_{i}.wav"

def queue_gen(scene_idx, prompt, seed, audio_file, ref_image):
    fc = max(9, ((int(round(PER_SCENE*FPS))-1)//8)*8+1)
    wf = {
        "10": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
        "11": {"class_type": "VAELoader", "inputs": {"vae_name": VIDEO_VAE}},
        "12": {"class_type": "LTXVAudioVAELoader", "inputs": {"ckpt_name": CKPT}},
        "13": {"class_type": "LTXAVTextEncoderLoader", "inputs": {"text_encoder": TEXT_ENCODER, "ckpt_name": CKPT, "device": "default"}},
        "20": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["10",0], "lora_name": LORA, "strength_model": LORA_STRENGTH}},
        "30": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["13",0]}},
        "31": {"class_type": "CLIPTextEncode", "inputs": {"text": NEG, "clip": ["13",0]}},
        "32": {"class_type": "LTXVConditioning", "inputs": {"positive": ["30",0], "negative": ["31",0], "frame_rate": float(FPS)}},
        "40": {"class_type": "LoadImage", "inputs": {"image": ref_image}},
        "41": {"class_type": "LoadAudio", "inputs": {"audio": audio_file}},
        "42": {"class_type": "LTXVAudioVAEEncode", "inputs": {"audio": ["41",0], "audio_vae": ["12",0]}},
        "43": {"class_type": "EmptyLTXVLatentVideo", "inputs": {"width": W, "height": H, "length": fc, "batch_size": 1}},
        "44": {"class_type": "LTXVImgToVideoInplace", "inputs": {"vae": ["11",0], "image": ["40",0], "latent": ["43",0], "strength": 0.3, "bypass": False}},
        "46": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": ["44",0], "audio_latent": ["42",0]}},
        "47": {"class_type": "LTXVSetAudioVideoMaskByTime", "inputs": {"av_latent": ["46",0], "positive": ["32",0], "negative": ["32",1], "model": ["20",0], "vae": ["11",0], "audio_vae": ["12",0], "start_time": 0.0, "end_time": float(PER_SCENE), "video_fps": float(FPS), "mask_video": True, "mask_audio": False, "mask_init_value_video": 0.0, "mask_init_value_audio": 0.0, "slope_len": 3}},
        "50": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "51": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "52": {"class_type": "BasicScheduler", "inputs": {"model": ["20",0], "scheduler": "linear_quadratic", "steps": steps, "denoise": 1.0}},
        "53": {"class_type": "CFGGuider", "inputs": {"model": ["20",0], "positive": ["47",0], "negative": ["47",1], "cfg": 3.0}},
        "54": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["50",0], "guider": ["53",0], "sampler": ["51",0], "sigmas": ["52",0], "latent_image": ["47",2]}},
        "60": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["54",0]}},
        "61": {"class_type": "LTXVSpatioTemporalTiledVAEDecode", "inputs": {"vae": ["11",0], "latents": ["60",0], "spatial_tiles": 2, "spatial_overlap": 4, "temporal_tile_length": 16, "temporal_overlap": 4, "last_frame_fix": False, "working_device": "auto", "working_dtype": "auto"}},
        "62": {"class_type": "LTXVAudioVAEDecode", "inputs": {"samples": ["60",1], "audio_vae": ["12",0]}},
        "70": {"class_type": "CreateVideo", "inputs": {"images": ["61",0], "fps": float(FPS)}},
        "71": {"class_type": "SaveVideo", "inputs": {"video": ["70",0], "filename_prefix": f"mv_clip_{scene_idx}", "format": "mp4", "codec": "h264"}},
    }
    req = urllib.request.Request(f"{COMFY}/prompt", data=json.dumps({"prompt": wf}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())["prompt_id"]

# Generate all scenes
clips = []
for i, s in enumerate(scenes):
    print(f"\n=== Scene {i+1}/4 ===")
    pid = queue_gen(i, s["prompt"], s["seed"], s["audio_file"], "char_clean.png")
    start = time.time()
    rv = None
    while time.time()-start < 600:
        try:
            h = json.loads(urllib.request.urlopen(f"{COMFY}/history/{pid}").read())
        except: time.sleep(5); continue
        if pid in h:
            print(f"  {time.time()-start:.0f}s — {h[pid]['status']['status_str']}")
            if h[pid]['status']['status_str'] == 'success':
                for no, out in h[pid].get("outputs",{}).items():
                    for v in out.get("images", out.get("media",[])):
                        rv = os.path.join(OUTPUT_DIR, v.get("filename"))
                        if rv and os.path.exists(rv):
                            clips.append(rv)
                            sz = os.path.getsize(rv)
                            print(f"  → {os.path.basename(rv)} ({sz/1024:.0f} KB)")
            break
        time.sleep(5)
    else:
        print(f"  TIMEOUT scene {i+1}")

# Stitch all clips with crossfade
print(f"\n=== Stitching {len(clips)} clips ===")
if len(clips) >= 2:
    clip_list = "/tmp/mv_clips.txt"
    with open(clip_list, "w") as f:
        for c in clips:
            f.write(f"file '{c}'\n")
    
    raw_concat = os.path.join(OUTPUT_DIR, "mv_raw.mp4")
    subprocess.run(["ffmpeg","-y","-f","concat","-safe","0","-i",clip_list,
        "-c","copy",raw_concat], check=True, capture_output=True)
    print(f"→ Raw concat: {os.path.basename(raw_concat)} ({os.path.getsize(raw_concat)/1024:.0f} KB)")
    
    # Mux full audio
    final = os.path.join(OUTPUT_DIR, "mv_shadows_final.mp4")
    subprocess.run(["ffmpeg","-y","-i",raw_concat,"-i",full_audio,
        "-c:v","copy","-c:a","aac","-b:a","192k","-shortest",final],
        check=True, capture_output=True)
    r = subprocess.run(["ffprobe","-v","error","-show_entries","format=duration,size","-of","csv=p=0",final],
        capture_output=True, text=True)
    print(f"\n✅ {os.path.basename(final)} ({r.stdout.strip()})")
elif len(clips) == 1:
    final = clips[0]
    print(f"Only 1 clip: {final}")
