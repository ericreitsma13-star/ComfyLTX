#!/usr/bin/env python3
"""IC-LoRA + vocals-only + zero-value audio mask for proper lip sync."""
import json, urllib.request, time, sys, subprocess, os

COMFY = "http://127.0.0.1:8188"
CKPT = "ltx-2.3-22b-distilled-1.1.safetensors"
VIDEO_VAE = "ltx-2.3-22b-distilled_video_vae.safetensors"
TEXT_ENCODER = "gemma_3_12B_it_fp4_mixed.safetensors"
IC_LORA = "ltx-2.3-22b-ic-lora-lipdub.safetensors"
IMAGE = "char_keyframe.png"
AUDIO = "pines_vocals.wav"  # vocals-only!
PROMPT = "static portrait of a young woman softly singing, standing still, misty pine forest at dawn, cinematic, golden light filtering through trees, intimate close-up, subtle head movement only"
NEGATIVE = "walking, running, moving, motion blur, ugly, deformed, bad anatomy, extra limbs, blurry, low quality, watermark, music"
W, H, FPS = 960, 576, 24  # 576/32=18 (even) — needed for IC-LoRA latent_downscale_factor=2
DURATION = 4.0
SEED = 42

target_frames = int(round(DURATION * FPS))
frame_count = max(9, ((target_frames - 1) // 8) * 8 + 1)
steps, cfg = 20, 3.0

OUTPUT_DIR = "/home/ericr/ComfyUI/output"
AUDIO_PATH = "/home/ericr/ComfyUI/input/pines_vocals.wav"

wf = {
    # Models
    "10": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
    "11": {"class_type": "VAELoader", "inputs": {"vae_name": VIDEO_VAE}},
    "12": {"class_type": "LTXVAudioVAELoader", "inputs": {"ckpt_name": CKPT}},
    "13": {"class_type": "LTXAVTextEncoderLoader", "inputs": {
        "text_encoder": TEXT_ENCODER, "ckpt_name": CKPT, "device": "default"}},

    # IC-LoRA for character consistency
    "20": {"class_type": "LTXICLoRALoaderModelOnly", "inputs": {
        "model": ["10", 0], "lora_name": IC_LORA, "strength_model": 1.0}},

    # Prompt
    "30": {"class_type": "CLIPTextEncode", "inputs": {"text": PROMPT, "clip": ["13", 0]}},
    "31": {"class_type": "CLIPTextEncode", "inputs": {"text": NEGATIVE, "clip": ["13", 0]}},
    "32": {"class_type": "LTXVConditioning", "inputs": {
        "positive": ["30", 0], "negative": ["31", 0], "frame_rate": float(FPS)}},

    # Image + Audio
    "40": {"class_type": "LoadImage", "inputs": {"image": IMAGE}},
    "41": {"class_type": "LoadAudio", "inputs": {"audio": AUDIO}},

    # Audio encode
    "42": {"class_type": "LTXVAudioVAEEncode", "inputs": {
        "audio": ["41", 0], "audio_vae": ["12", 0]}},

    # Video latent
    "43": {"class_type": "EmptyLTXVLatentVideo", "inputs": {
        "width": W, "height": H, "length": frame_count, "batch_size": 1}},
    "44": {"class_type": "LTXVImgToVideoInplace", "inputs": {
        "vae": ["11", 0], "image": ["40", 0], "latent": ["43", 0],
        "strength": 0.5, "bypass": False}},  # lower I2V strength to suppress reference motion bias

    # IC-LoRA Guide: inject character reference into conditioning
    "45": {"class_type": "LTXAddVideoICLoRAGuide", "inputs": {
        "positive": ["32", 0], "negative": ["32", 1],
        "vae": ["11", 0], "latent": ["44", 0],
        "image": ["40", 0], "frame_idx": 0,
        "strength": 1.0,
        "latent_downscale_factor": 2,
        "crop": "center",
        "use_tiled_encode": False,
        "tile_size": 64, "tile_overlap": 16}},

    # Concat AV
    "46": {"class_type": "LTXVConcatAVLatent", "inputs": {
        "video_latent": ["44", 0], "audio_latent": ["42", 0]}},

    # Zero-value audio mask (SOTAI's trick — preserve audio signal during sampling)
    "47": {"class_type": "LTXVSetAudioVideoMaskByTime", "inputs": {
        "av_latent": ["46", 0],
        "positive": ["45", 0], "negative": ["45", 1],
        "model": ["20", 0],
        "vae": ["11", 0], "audio_vae": ["12", 0],
        "start_time": 0.0, "end_time": float(DURATION),
        "video_fps": float(FPS),
        "mask_video": True, "mask_audio": False,
        "mask_init_value_video": 0.0, "mask_init_value_audio": 0.0,
        "slope_len": 3}},

    # Sampling
    "50": {"class_type": "RandomNoise", "inputs": {"noise_seed": SEED}},
    "51": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
    "52": {"class_type": "BasicScheduler", "inputs": {
        "model": ["20", 0], "scheduler": "linear_quadratic",
        "steps": steps, "denoise": 1.0}},
    "53": {"class_type": "CFGGuider", "inputs": {
        "model": ["20", 0], "positive": ["47", 0], "negative": ["47", 1], "cfg": float(cfg)}},
    "54": {"class_type": "SamplerCustomAdvanced", "inputs": {
        "noise": ["50", 0], "guider": ["53", 0], "sampler": ["51", 0],
        "sigmas": ["52", 0], "latent_image": ["47", 2]}},

    # Separate AV and decode video only
    "60": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["54", 0]}},
    "61": {"class_type": "LTXVSpatioTemporalTiledVAEDecode", "inputs": {
        "vae": ["11", 0], "latents": ["60", 0],
        "spatial_tiles": 2, "spatial_overlap": 4,
        "temporal_tile_length": 16, "temporal_overlap": 4,
        "last_frame_fix": False, "working_device": "auto", "working_dtype": "auto"}},
    "70": {"class_type": "CreateVideo", "inputs": {"images": ["61", 0], "fps": float(FPS)}},
    "71": {"class_type": "SaveVideo", "inputs": {
        "video": ["70", 0], "filename_prefix": "iclora_lipsync",
        "format": "mp4", "codec": "h264"}},
}

print(f"IC-LoRA + vocals-only: {DURATION}s ({frame_count}fr) @ {W}x{H}")
print(f"IC-LoRA: {IC_LORA}, Audio: {AUDIO}, Seed: {SEED}")
print(f"Zero-value audio mask: ON (mask_audio=False mask_init=0.0)")

client_id = f"test_{int(time.time())}"
req = urllib.request.Request(f"{COMFY}/prompt",
    data=json.dumps({"prompt": wf, "client_id": client_id}).encode(),
    headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=30) as r:
    res = json.loads(r.read())
pid = res.get("prompt_id")
print(f"Prompt ID: {pid}")

start = time.time()
raw_video = None
while time.time() - start < 600:
    try:
        h = json.loads(urllib.request.urlopen(f"{COMFY}/history/{pid}").read())
    except:
        time.sleep(2)
        continue
    if pid in h:
        data = h[pid]
        outputs = data.get("outputs", {})
        for nid, out in outputs.items():
            for v in out.get("images", out.get("media", [])):
                raw_video = os.path.join(OUTPUT_DIR, v.get("filename"))
        elapsed = time.time() - start
        status = data.get('status', {}).get('status_str', '?')
        print(f"Generated in {elapsed:.1f}s — {status}")
        break
    time.sleep(2)
else:
    print("TIMEOUT"); sys.exit(1)

# Mux vocals-only audio into video
if raw_video and os.path.exists(raw_video):
    final = os.path.join(OUTPUT_DIR, "iclora_final.mp4")
    subprocess.run(["ffmpeg", "-y", "-i", raw_video, "-i", AUDIO_PATH,
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-map", "0:v:0", "-map", "1:a:0", "-shortest",
        "-movflags", "+faststart", final], check=True, capture_output=True)
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration,size",
        "-of", "default=noprint_wrappers=1", final], capture_output=True, text=True)
    print(f"Final: {final}")
    print(r.stdout)
else:
    print("No raw video found")
