#!/usr/bin/env python3
"""SOTAI native test: decode model audio to verify lip sync. Uses full song segment."""
import json, urllib.request, time, subprocess, os

COMFY = "http://127.0.0.1:8188"
OUTPUT_DIR = "/home/ericr/ComfyUI/output"
CKPT = "ltx-2.3-22b-distilled-1.1.safetensors"
VIDEO_VAE = "ltx-2.3-22b-distilled_video_vae.safetensors"
TEXT_ENCODER = "gemma_3_12B_it_fp4_mixed.safetensors"
IMAGE = "walk_v04_00001_.png"
AUDIO = "pines_vocals.wav"
PROMPT = "young woman with long dark hair walking through misty pine forest, full body visible, wide shot, distant view, golden dawn light, cinematic, natural walking, singing softly to herself"
NEGATIVE = "close up, headshot, portrait, face closeup, ugly, deformed, bad anatomy, blurry, music, instruments"
W, H, FPS = 832, 480, 24  # SOTAI native resolution
DURATION = 6.0
SEED = 42
steps = 15  # SOTAI's recommended steps for lip sync

frame_count = max(9, ((int(round(DURATION*FPS))-1)//8)*8+1)

# Full workflow: encode audio → concat AV → sample → separate → decode both
wf = {
    "10": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
    "11": {"class_type": "VAELoader", "inputs": {"vae_name": VIDEO_VAE}},
    "12": {"class_type": "LTXVAudioVAELoader", "inputs": {"ckpt_name": CKPT}},
    "13": {"class_type": "LTXAVTextEncoderLoader", "inputs": {"text_encoder": TEXT_ENCODER, "ckpt_name": CKPT, "device": "default"}},
    "30": {"class_type": "CLIPTextEncode", "inputs": {"text": PROMPT, "clip": ["13", 0]}},
    "31": {"class_type": "CLIPTextEncode", "inputs": {"text": NEGATIVE, "clip": ["13", 0]}},
    "32": {"class_type": "LTXVConditioning", "inputs": {"positive": ["30", 0], "negative": ["31", 0], "frame_rate": float(FPS)}},
    "40": {"class_type": "LoadImage", "inputs": {"image": IMAGE}},
    "41": {"class_type": "LoadAudio", "inputs": {"audio": AUDIO}},
    "42": {"class_type": "LTXVAudioVAEEncode", "inputs": {"audio": ["41", 0], "audio_vae": ["12", 0]}},
    "43": {"class_type": "EmptyLTXVLatentVideo", "inputs": {"width": W, "height": H, "length": frame_count, "batch_size": 1}},
    "44": {"class_type": "LTXVImgToVideoInplace", "inputs": {"vae": ["11", 0], "image": ["40", 0], "latent": ["43", 0], "strength": 0.5, "bypass": False}},
    "46": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": ["44", 0], "audio_latent": ["42", 0]}},
    "47": {"class_type": "LTXVSetAudioVideoMaskByTime", "inputs": {"av_latent": ["46", 0], "positive": ["32", 0], "negative": ["32", 1], "model": ["10", 0], "vae": ["11", 0], "audio_vae": ["12", 0], "start_time": 0.0, "end_time": float(DURATION), "video_fps": float(FPS), "mask_video": True, "mask_audio": False, "mask_init_value_video": 0.0, "mask_init_value_audio": 0.0, "slope_len": 3}},
    "50": {"class_type": "RandomNoise", "inputs": {"noise_seed": SEED}},
    "51": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
    "52": {"class_type": "BasicScheduler", "inputs": {"model": ["10", 0], "scheduler": "linear_quadratic", "steps": steps, "denoise": 1.0}},
    "53": {"class_type": "CFGGuider", "inputs": {"model": ["10", 0], "positive": ["47", 0], "negative": ["47", 1], "cfg": 3.0}},
    "54": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["50", 0], "guider": ["53", 0], "sampler": ["51", 0], "sigmas": ["52", 0], "latent_image": ["47", 2]}},
    "60": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["54", 0]}},
    "61": {"class_type": "LTXVSpatioTemporalTiledVAEDecode", "inputs": {"vae": ["11", 0], "latents": ["60", 0], "spatial_tiles": 2, "spatial_overlap": 4, "temporal_tile_length": 16, "temporal_overlap": 4, "last_frame_fix": False, "working_device": "auto", "working_dtype": "auto"}},
    # KEY: Decode the model's OWN audio (never did this before!)
    "80": {"class_type": "LTXVAudioVAEDecode", "inputs": {"samples": ["60", 1], "audio_vae": ["12", 0]}},
    # Save video with model's native audio
    "70": {"class_type": "CreateVideo", "inputs": {"images": ["61", 0], "fps": float(FPS)}},
    "71": {"class_type": "SaveVideo", "inputs": {"video": ["70", 0], "filename_prefix": "sotai_lipsync", "format": "mp4", "codec": "h264"}},
}

# Queue it
print(f"SOTAI test: {DURATION}s ({frame_count}fr) @ {W}x{H}, steps={steps}")
req = urllib.request.Request(f"{COMFY}/prompt", data=json.dumps({"prompt": wf}).encode(),
    headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=30) as r:
    pid = json.loads(r.read())["prompt_id"]
print(f"Prompt: {pid}")

start = time.time()
raw_video = None
while time.time()-start < 600:
    try:
        h = json.loads(urllib.request.urlopen(f"{COMFY}/history/{pid}").read())
    except: time.sleep(2); continue
    if pid in h:
        for nid, out in h[pid].get("outputs",{}).items():
            for v in out.get("images", out.get("media",[])):
                raw_video = os.path.join(OUTPUT_DIR, v.get("filename"))
        print(f"Done in {time.time()-start:.0f}s — {h[pid]['status']['status_str']}")
        break
    time.sleep(2)
else: print("TIMEOUT"); exit(1)

if raw_video and os.path.exists(raw_video):
    r = subprocess.run(["ffprobe","-v","error","-show_entries","format=duration,size","-of","csv=p=0",raw_video],
        capture_output=True, text=True)
    print(f"Raw output: {raw_video} ({r.stdout.strip()})")
    print(f"\n✅ Check {raw_video} — this has the MODEL'S OWN AUDIO, not muxed!")
    print(f"   If you hear singing, lip sync IS working and we were overwriting it.")
    print(f"   If silent/static, the model isn't producing audio from conditioning.")
