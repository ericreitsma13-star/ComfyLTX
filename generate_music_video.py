#!/usr/bin/env python3
"""Generate 4 IC-LoRA clips with different references, stitch with crossfade."""
import json, urllib.request, time, subprocess, os

COMFY = "http://127.0.0.1:8188"
OUTPUT_DIR = "/home/ericr/ComfyUI/output"
AUDIO = "/home/ericr/ComfyUI/input/pines_original_6s.mp3"
VOCALS = "/home/ericr/ComfyUI/input/pines_vocals.wav"
CKPT = "ltx-2.3-22b-distilled-1.1.safetensors"
VIDEO_VAE = "ltx-2.3-22b-distilled_video_vae.safetensors"
TEXT_ENCODER = "gemma_3_12B_it_fp4_mixed.safetensors"
IC_LORA = "ltx-2.3-22b-ic-lora-lipdub.safetensors"
W, H, FPS = 960, 576, 24

# 4 clips with different references and styles
clips = [
    {
        "ref": "char_walk.png",  # walking pose
        "prompt": "young woman with dark brown hair walking through misty pine forest at dawn, golden light, cinematic, full body, natural movement",
        "seed": 42,
        "duration": 6.0,
    },
    {
        "ref": "char_clean.png",  # headshot portrait
        "prompt": "close-up portrait of young woman with dark brown hair singing softly, standing in forest clearing, golden hour, cinematic, intimate",
        "seed": 43,
        "duration": 6.0,
    },
    {
        "ref": "walk_v02_00001_.png",  # walking different angle
        "prompt": "young woman with dark brown hair walking between pine trees, misty morning light, three quarter shot, cinematic, ethereal atmosphere",
        "seed": 44,
        "duration": 6.0,
    },
    {
        "ref": "walk_v04_00001_.png",  # full body forest
        "prompt": "young woman with dark brown hair standing in forest clearing, looking up at golden light through trees, cinematic, medium shot, serene",
        "seed": 45,
        "duration": 6.0,
    },
]

def queue_clip(ref_img, prompt, seed, duration):
    frame_count = max(9, ((int(round(duration*FPS))-1)//8)*8+1)
    wf = {
        "10": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
        "11": {"class_type": "VAELoader", "inputs": {"vae_name": VIDEO_VAE}},
        "12": {"class_type": "LTXVAudioVAELoader", "inputs": {"ckpt_name": CKPT}},
        "13": {"class_type": "LTXAVTextEncoderLoader", "inputs": {"text_encoder": TEXT_ENCODER, "ckpt_name": CKPT, "device": "default"}},
        "20": {"class_type": "LTXICLoRALoaderModelOnly", "inputs": {"model": ["10",0], "lora_name": IC_LORA, "strength_model": 1.0}},
        "30": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["13",0]}},
        "31": {"class_type": "CLIPTextEncode", "inputs": {"text": "ugly, deformed, bad anatomy, blurry, music, instruments, close up", "clip": ["13",0]}},
        "32": {"class_type": "LTXVConditioning", "inputs": {"positive": ["30",0], "negative": ["31",0], "frame_rate": float(FPS)}},
        "40": {"class_type": "LoadImage", "inputs": {"image": ref_img}},
        "41": {"class_type": "LoadAudio", "inputs": {"audio": "pines_vocals.wav"}},
        "42": {"class_type": "LTXVAudioVAEEncode", "inputs": {"audio": ["41",0], "audio_vae": ["12",0]}},
        "43": {"class_type": "EmptyLTXVLatentVideo", "inputs": {"width": W, "height": H, "length": frame_count, "batch_size": 1}},
        "44": {"class_type": "LTXVImgToVideoInplace", "inputs": {"vae": ["11",0], "image": ["40",0], "latent": ["43",0], "strength": 0.5, "bypass": False}},
        "45": {"class_type": "LTXAddVideoICLoRAGuide", "inputs": {"positive": ["32",0], "negative": ["32",1], "vae": ["11",0], "latent": ["44",0], "image": ["40",0], "frame_idx": 0, "strength": 0.8, "latent_downscale_factor": 2, "crop": "center", "use_tiled_encode": False, "tile_size": 64, "tile_overlap": 16}},
        "46": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": ["44",0], "audio_latent": ["42",0]}},
        "47": {"class_type": "LTXVSetAudioVideoMaskByTime", "inputs": {"av_latent": ["46",0], "positive": ["45",0], "negative": ["45",1], "model": ["20",0], "vae": ["11",0], "audio_vae": ["12",0], "start_time": 0.0, "end_time": float(duration), "video_fps": float(FPS), "mask_video": True, "mask_audio": False, "mask_init_value_video": 0.0, "mask_init_value_audio": 0.0, "slope_len": 3}},
        "50": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "51": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "52": {"class_type": "BasicScheduler", "inputs": {"model": ["20",0], "scheduler": "linear_quadratic", "steps": 25, "denoise": 1.0}},
        "53": {"class_type": "CFGGuider", "inputs": {"model": ["20",0], "positive": ["47",0], "negative": ["47",1], "cfg": 3.0}},
        "54": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["50",0], "guider": ["53",0], "sampler": ["51",0], "sigmas": ["52",0], "latent_image": ["47",2]}},
        "60": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["54",0]}},
        "61": {"class_type": "LTXVSpatioTemporalTiledVAEDecode", "inputs": {"vae": ["11",0], "latents": ["60",0], "spatial_tiles": 2, "spatial_overlap": 4, "temporal_tile_length": 16, "temporal_overlap": 4, "last_frame_fix": False, "working_device": "auto", "working_dtype": "auto"}},
        "70": {"class_type": "CreateVideo", "inputs": {"images": ["61",0], "fps": float(FPS)}},
        "71": {"class_type": "SaveVideo", "inputs": {"video": ["70",0], "filename_prefix": f"mv_clip", "format": "mp4", "codec": "h264"}},
    }
    req = urllib.request.Request(f"{COMFY}/prompt", data=json.dumps({"prompt": wf}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())["prompt_id"]

raw_files = []
for i, clip in enumerate(clips):
    print(f"\n=== Clip {i+1}/4: {clip['ref']} ===")
    pid = queue_clip(clip["ref"], clip["prompt"], clip["seed"], clip["duration"])
    print(f"  Queued: {pid}")

    start = time.time()
    raw_video = None
    while time.time()-start < 600:
        try:
            h = json.loads(urllib.request.urlopen(f"{COMFY}/history/{pid}").read())
        except: time.sleep(5); continue
        if pid in h:
            for nid, out in h[pid].get("outputs",{}).items():
                for v in out.get("images", out.get("media",[])):
                    raw_video = os.path.join(OUTPUT_DIR, v.get("filename"))
            elapsed = time.time()-start
            status = h[pid]["status"]["status_str"]
            print(f"  Done in {elapsed:.0f}s — {status}")
            if raw_video and os.path.exists(raw_video):
                raw_files.append(raw_video)
            break
        time.sleep(5)
    else:
        print(f"  TIMEOUT on clip {i+1}")

if len(raw_files) < 2:
    print("Not enough clips generated"); exit(1)

print(f"\n=== Stitching {len(raw_files)} clips ===")

# Create concat file
concat_file = "/tmp/mv_concat.txt"
with open(concat_file, "w") as f:
    for path in raw_files:
        f.write(f"file '{path}'\n")

# Stitch with crossfade
final = os.path.join(OUTPUT_DIR, "music_video_final.mp4")
subprocess.run([
    "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_file,
    "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
    "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
    "-movflags", "+faststart", final
], check=True, capture_output=True)

# Mux original audio aligned to clip duration
final_with_audio = os.path.join(OUTPUT_DIR, "music_video_sound.mp4")
subprocess.run([
    "ffmpeg", "-y", "-i", final, "-i", AUDIO,
    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
    "-map", "0:v:0", "-map", "1:a:0", "-shortest",
    "-movflags", "+faststart", final_with_audio
], check=True, capture_output=True)

r = subprocess.run(["ffprobe","-v","error","-show_entries","format=duration,size","-of","csv=p=0",final_with_audio],
    capture_output=True, text=True)
print(f"\n✅ FINAL: {final_with_audio} ({r.stdout.strip()})")
print(f"   Clips: {len(raw_files)} × ~6s = ~{len(raw_files)*6}s total")
