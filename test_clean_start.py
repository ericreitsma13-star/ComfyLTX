#!/usr/bin/env python3
"""Clean keyframe + audio conditioning — no IC-LoRA distortion."""
import json, urllib.request, time, sys, subprocess, os
COMFY = "http://127.0.0.1:8188"
CKPT = "ltx-2.3-22b-distilled-1.1.safetensors"
VIDEO_VAE = "ltx-2.3-22b-distilled_video_vae.safetensors"
TEXT_ENCODER = "gemma_3_12B_it_fp4_mixed.safetensors"
IMAGE = "char_clean.png"
AUDIO = "pines_vocals.wav"
PROMPT = "close-up portrait of a young woman with dark brown hair, softly singing, standing still, cinematic lighting, subtle head movement"
NEGATIVE = "walking, running, ugly, deformed, bad anatomy, blurry, music, instruments"
W, H, FPS = 832, 480, 24
DURATION = 6.0
SEED = 42
steps, cfg = 25, 3.0
frame_count = max(9, ((int(round(DURATION*FPS))-1)//8)*8+1)
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
    "44": {"class_type": "LTXVImgToVideoInplace", "inputs": {"vae": ["11", 0], "image": ["40", 0], "latent": ["43", 0], "strength": 0.7, "bypass": False}},
    "46": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": ["44", 0], "audio_latent": ["42", 0]}},
    "47": {"class_type": "LTXVSetAudioVideoMaskByTime", "inputs": {"av_latent": ["46", 0], "positive": ["32", 0], "negative": ["32", 1], "model": ["10", 0], "vae": ["11", 0], "audio_vae": ["12", 0], "start_time": 0.0, "end_time": float(DURATION), "video_fps": float(FPS), "mask_video": True, "mask_audio": False, "mask_init_value_video": 0.0, "mask_init_value_audio": 0.0, "slope_len": 5}},
    "50": {"class_type": "RandomNoise", "inputs": {"noise_seed": SEED}},
    "51": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
    "52": {"class_type": "BasicScheduler", "inputs": {"model": ["10", 0], "scheduler": "linear_quadratic", "steps": steps, "denoise": 1.0}},
    "53": {"class_type": "CFGGuider", "inputs": {"model": ["10", 0], "positive": ["47", 0], "negative": ["47", 1], "cfg": float(cfg)}},
    "54": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["50", 0], "guider": ["53", 0], "sampler": ["51", 0], "sigmas": ["52", 0], "latent_image": ["47", 2]}},
    "60": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["54", 0]}},
    "61": {"class_type": "LTXVSpatioTemporalTiledVAEDecode", "inputs": {"vae": ["11", 0], "latents": ["60", 0], "spatial_tiles": 2, "spatial_overlap": 4, "temporal_tile_length": 16, "temporal_overlap": 4}},
    "70": {"class_type": "CreateVideo", "inputs": {"images": ["61", 0], "fps": float(FPS)}},
    "71": {"class_type": "SaveVideo", "inputs": {"video": ["70", 0], "filename_prefix": "clean_start", "format": "mp4", "codec": "h264"}},
}
print(f"Clean start: {DURATION}s ({frame_count}fr) @ {W}x{H}")
req = urllib.request.Request(f"{COMFY}/prompt", data=json.dumps({"prompt": wf}).encode(), headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=30) as r:
    pid = json.loads(r.read())["prompt_id"]
print(f"Prompt: {pid}")
start = time.time()
while time.time()-start < 600:
    try:
        h = json.loads(urllib.request.urlopen(f"{COMFY}/history/{pid}").read())
    except: time.sleep(2); continue
    if pid in h:
        print(f"Done in {time.time()-start:.0f}s — {h[pid]["status"]["status_str"]}")
        break
print("Check output/clean_start_* for result")
