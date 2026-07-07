#!/usr/bin/env python3
"""6s lip sync test — longer clip so audio conditioning has time to kick in."""
import json, urllib.request, time, sys, subprocess, os

COMFY = "http://127.0.0.1:8188"
CKPT = "ltx-2.3-22b-distilled-1.1.safetensors"
VIDEO_VAE = "ltx-2.3-22b-distilled_video_vae.safetensors"
TEXT_ENCODER = "gemma_3_12B_it_fp4_mixed.safetensors"
IMAGE = "char_keyframe.png"
AUDIO = "pines_vocals.wav"
PROMPT = "medium close-up of a young woman singing softly, standing still in misty pine forest, golden dawn light, cinematic, subtle head movement, lip movement matching voice"
NEGATIVE = "ugly, deformed, bad anatomy, blurry, music, melody, instruments"
W, H, FPS = 832, 480, 24
DURATION = 6.0  # longer — give audio time to condition
SEED = 42
steps, cfg = 25, 3.0  # more steps for stronger lip sync

target_frames = int(round(DURATION * FPS))
frame_count = max(9, ((target_frames - 1) // 8) * 8 + 1)
OUTPUT_DIR = "/home/ericr/ComfyUI/output"

# Use vocals clip extended — extract 6s segment
AUDIO_SEGMENT = "pines_vocals_6s.wav"
if not os.path.exists(f"/home/ericr/ComfyUI/input/{AUDIO_SEGMENT}"):
    subprocess.run(["ffmpeg", "-y", "-i", "/home/ericr/ComfyUI/input/pines_vocals.wav",
        "-t", "6", "-c", "copy",
        f"/home/ericr/ComfyUI/input/{AUDIO_SEGMENT}"], check=True, capture_output=True)

wf = {
    "10": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
    "11": {"class_type": "VAELoader", "inputs": {"vae_name": VIDEO_VAE}},
    "12": {"class_type": "LTXVAudioVAELoader", "inputs": {"ckpt_name": CKPT}},
    "13": {"class_type": "LTXAVTextEncoderLoader", "inputs": {
        "text_encoder": TEXT_ENCODER, "ckpt_name": CKPT, "device": "default"}},
    "30": {"class_type": "CLIPTextEncode", "inputs": {"text": PROMPT, "clip": ["13", 0]}},
    "31": {"class_type": "CLIPTextEncode", "inputs": {"text": NEGATIVE, "clip": ["13", 0]}},
    "32": {"class_type": "LTXVConditioning", "inputs": {
        "positive": ["30", 0], "negative": ["31", 0], "frame_rate": float(FPS)}},
    "40": {"class_type": "LoadImage", "inputs": {"image": IMAGE}},
    "41": {"class_type": "LoadAudio", "inputs": {"audio": AUDIO_SEGMENT}},
    "42": {"class_type": "LTXVAudioVAEEncode", "inputs": {
        "audio": ["41", 0], "audio_vae": ["12", 0]}},
    "43": {"class_type": "EmptyLTXVLatentVideo", "inputs": {
        "width": W, "height": H, "length": frame_count, "batch_size": 1}},
    "44": {"class_type": "LTXVImgToVideoInplace", "inputs": {
        "vae": ["11", 0], "image": ["40", 0], "latent": ["43", 0],
        "strength": 0.7, "bypass": False}},
    "46": {"class_type": "LTXVConcatAVLatent", "inputs": {
        "video_latent": ["44", 0], "audio_latent": ["42", 0]}},
    "47": {"class_type": "LTXVSetAudioVideoMaskByTime", "inputs": {
        "av_latent": ["46", 0],
        "positive": ["32", 0], "negative": ["32", 1],
        "model": ["10", 0],
        "vae": ["11", 0], "audio_vae": ["12", 0],
        "start_time": 0.0, "end_time": float(DURATION),
        "video_fps": float(FPS),
        "mask_video": True, "mask_audio": False,
        "mask_init_value_video": 0.0, "mask_init_value_audio": 0.0,
        "slope_len": 5}},  # longer slope for smoother transition
    "50": {"class_type": "RandomNoise", "inputs": {"noise_seed": SEED}},
    "51": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
    "52": {"class_type": "BasicScheduler", "inputs": {
        "model": ["10", 0], "scheduler": "linear_quadratic",
        "steps": steps, "denoise": 1.0}},
    "53": {"class_type": "CFGGuider", "inputs": {
        "model": ["10", 0], "positive": ["47", 0], "negative": ["47", 1], "cfg": float(cfg)}},
    "54": {"class_type": "SamplerCustomAdvanced", "inputs": {
        "noise": ["50", 0], "guider": ["53", 0], "sampler": ["51", 0],
        "sigmas": ["52", 0], "latent_image": ["47", 2]}},
    "60": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["54", 0]}},
    "61": {"class_type": "LTXVSpatioTemporalTiledVAEDecode", "inputs": {
        "vae": ["11", 0], "latents": ["60", 0],
        "spatial_tiles": 2, "spatial_overlap": 4,
        "temporal_tile_length": 16, "temporal_overlap": 4,
        "last_frame_fix": False, "working_device": "auto", "working_dtype": "auto"}},
    "70": {"class_type": "CreateVideo", "inputs": {"images": ["61", 0], "fps": float(FPS)}},
    "71": {"class_type": "SaveVideo", "inputs": {
        "video": ["70", 0], "filename_prefix": "lipsync_6s",
        "format": "mp4", "codec": "h264"}},
}

print(f"6s lip sync test: {DURATION}s ({frame_count}fr) @ {W}x{H}, steps={steps}, seed={SEED}")
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

if raw_video and os.path.exists(raw_video):
    final = os.path.join(OUTPUT_DIR, "lipsync_6s_final.mp4")
    subprocess.run(["ffmpeg", "-y", "-i", raw_video,
        "-i", f"/home/ericr/ComfyUI/input/{AUDIO_SEGMENT}",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-map", "0:v:0", "-map", "1:a:0", "-shortest",
        "-movflags", "+faststart", final], check=True, capture_output=True)
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration,size",
        "-of", "default=noprint_wrappers=1", final], capture_output=True, text=True)
    print(f"Final: {final}")
    print(r.stdout)
