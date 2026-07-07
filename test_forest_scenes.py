#!/usr/bin/env python3
"""Forest walking + singing — medium shots for visible lip sync."""
import json, urllib.request, time, subprocess, os

COMFY = "http://127.0.0.1:8188"
OUTPUT_DIR = "/home/ericr/ComfyUI/output"
CKPT = "ltx-2.3-22b-distilled-1.1.safetensors"
VIDEO_VAE = "ltx-2.3-22b-distilled_video_vae.safetensors"
TEXT_ENCODER = "gemma_3_12B_it_fp4_mixed.safetensors"
W, H, FPS = 832, 480, 24
steps = 15

scenes = [
    {
        "ref": "walk_v04_00001_.png",
        "prompt": "young woman with long dark hair walking through misty pine forest at dawn, medium shot showing forest background and her face, golden light, cinematic, singing softly, mouth moving",
        "neg": "close up, extreme wide shot, ugly, deformed, blurry, music, instruments",
        "seed": 50, "duration": 8.0,
    },
    {
        "ref": "walk_v02_00001_.png",
        "prompt": "young woman with long dark hair on forest path between pine trees, medium shot, morning mist, warm golden light, cinematic, walking and singing, visible mouth movement",
        "neg": "close up, extreme wide shot, ugly, deformed, blurry, music",
        "seed": 51, "duration": 8.0,
    },
    {
        "ref": "walk_v03_00001_.png",
        "prompt": "young woman with long dark hair in pine forest clearing, dappled sunlight, medium shot, misty atmosphere, cinematic, singing while walking, expressive face",
        "neg": "close up, extreme wide shot, ugly, deformed, blurry, music",
        "seed": 52, "duration": 8.0,
    },
    {
        "ref": "walk_v05_00001_.png",
        "prompt": "young woman with long dark hair in misty pine forest at dawn, medium shot, golden light rays through trees, cinematic, singing softly, visible facial expressions",
        "neg": "close up, extreme wide shot, ugly, deformed, blurry, music",
        "seed": 53, "duration": 8.0,
    },
]

# Create 8s vocals segment
if not os.path.exists("/home/ericr/ComfyUI/input/vocals_8s.wav"):
    subprocess.run(["ffmpeg","-y","-i","/home/ericr/ComfyUI/input/pines_vocals.wav",
        "-t","8","-c","copy","/home/ericr/ComfyUI/input/vocals_8s.wav"], check=True, capture_output=True)

raw_files = []
for i, s in enumerate(scenes):
    print(f"\n=== Scene {i+1}/4 ===")
    frame_count = max(9, ((int(round(s['duration']*FPS))-1)//8)*8+1)
    wf = {
        "10": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
        "11": {"class_type": "VAELoader", "inputs": {"vae_name": VIDEO_VAE}},
        "12": {"class_type": "LTXVAudioVAELoader", "inputs": {"ckpt_name": CKPT}},
        "13": {"class_type": "LTXAVTextEncoderLoader", "inputs": {"text_encoder": TEXT_ENCODER, "ckpt_name": CKPT, "device": "default"}},
        "30": {"class_type": "CLIPTextEncode", "inputs": {"text": s["prompt"], "clip": ["13",0]}},
        "31": {"class_type": "CLIPTextEncode", "inputs": {"text": s["neg"], "clip": ["13",0]}},
        "32": {"class_type": "LTXVConditioning", "inputs": {"positive": ["30",0], "negative": ["31",0], "frame_rate": float(FPS)}},
        "40": {"class_type": "LoadImage", "inputs": {"image": s["ref"]}},
        "41": {"class_type": "LoadAudio", "inputs": {"audio": "vocals_8s.wav"}},
        "42": {"class_type": "LTXVAudioVAEEncode", "inputs": {"audio": ["41",0], "audio_vae": ["12",0]}},
        "43": {"class_type": "EmptyLTXVLatentVideo", "inputs": {"width": W, "height": H, "length": frame_count, "batch_size": 1}},
        "44": {"class_type": "LTXVImgToVideoInplace", "inputs": {"vae": ["11",0], "image": ["40",0], "latent": ["43",0], "strength": 0.5, "bypass": False}},
        "46": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": ["44",0], "audio_latent": ["42",0]}},
        "47": {"class_type": "LTXVSetAudioVideoMaskByTime", "inputs": {"av_latent": ["46",0], "positive": ["32",0], "negative": ["32",1], "model": ["10",0], "vae": ["11",0], "audio_vae": ["12",0], "start_time": 0.0, "end_time": float(s['duration']), "video_fps": float(FPS), "mask_video": True, "mask_audio": False, "mask_init_value_video": 0.0, "mask_init_value_audio": 0.0, "slope_len": 3}},
        "50": {"class_type": "RandomNoise", "inputs": {"noise_seed": s["seed"]}},
        "51": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "52": {"class_type": "BasicScheduler", "inputs": {"model": ["10",0], "scheduler": "linear_quadratic", "steps": steps, "denoise": 1.0}},
        "53": {"class_type": "CFGGuider", "inputs": {"model": ["10",0], "positive": ["47",0], "negative": ["47",1], "cfg": 3.0}},
        "54": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["50",0], "guider": ["53",0], "sampler": ["51",0], "sigmas": ["52",0], "latent_image": ["47",2]}},
        "60": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["54",0]}},
        "61": {"class_type": "LTXVSpatioTemporalTiledVAEDecode", "inputs": {"vae": ["11",0], "latents": ["60",0], "spatial_tiles": 2, "spatial_overlap": 4, "temporal_tile_length": 16, "temporal_overlap": 4, "last_frame_fix": False, "working_device": "auto", "working_dtype": "auto"}},
        "80": {"class_type": "LTXVAudioVAEDecode", "inputs": {"samples": ["60",1], "audio_vae": ["12",0]}},
        "70": {"class_type": "CreateVideo", "inputs": {"images": ["61",0], "fps": float(FPS)}},
        "71": {"class_type": "SaveVideo", "inputs": {"video": ["70",0], "filename_prefix": f"scene_{i+1}", "format": "mp4", "codec": "h264"}},
    }
    req = urllib.request.Request(f"{COMFY}/prompt", data=json.dumps({"prompt": wf}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        pid = json.loads(r.read())["prompt_id"]
    print(f"  Queued: {pid}")
    start = time.time(); rv = None
    while time.time()-start < 600:
        try:
            h = json.loads(urllib.request.urlopen(f"{COMFY}/history/{pid}").read())
        except: time.sleep(5); continue
        if pid in h:
            for no, out in h[pid].get("outputs",{}).items():
                for v in out.get("images", out.get("media",[])):
                    rv = os.path.join(OUTPUT_DIR, v.get("filename"))
            t = time.time()-start
            print(f"  {t:.0f}s — {h[pid]['status']['status_str']}")
            if rv and os.path.exists(rv):
                raw_files.append(rv)
                r = subprocess.run(["ffprobe","-v","error","-show_entries","format=duration,size","-of","csv=p=0",rv],
                    capture_output=True, text=True)
                print(f"  → {rv} ({r.stdout.strip()})")
            break
        time.sleep(5)
    else: print(f"  TIMEOUT")

if len(raw_files) >= 2:
    print(f"\n=== Stitching {len(raw_files)} scenes ===")
    with open("/tmp/scenes.txt","w") as f:
        for p in raw_files: f.write(f"file '{p}'\n")
    final = os.path.join(OUTPUT_DIR, "forest_music_video.mp4")
    subprocess.run(["ffmpeg","-y","-f","concat","-safe","0","-i","/tmp/scenes.txt",
        "-c:v","libx264","-preset","fast","-crf","18","-pix_fmt","yuv420p",
        "-c:a","aac","-b:a","192k","-movflags","+faststart",final], check=True, capture_output=True)
    r = subprocess.run(["ffprobe","-v","error","-show_entries","format=duration,size","-of","csv=p=0",final],
        capture_output=True, text=True)
    print(f"\n✅ {final} ({r.stdout.strip()})")
