#!/usr/bin/env python3
"""LTX 2.3 audio-conditioned I2V — mux original audio into output for lip sync verification."""
import json, urllib.request, time, sys, subprocess, os

COMFY = "http://127.0.0.1:8188"
CKPT = "ltx-2.3-22b-distilled-1.1.safetensors"
VIDEO_VAE = "ltx-2.3-22b-distilled_video_vae.safetensors"
TEXT_ENCODER = "gemma_3_12B_it_fp4_mixed.safetensors"
LORA = "ltx-2.3-22b-distilled-lora-1.1_fro90_ceil72_condsafe.safetensors"
IMAGE = "char_keyframe.png"
AUDIO = "pines_clip.mp3"
PROMPT = "A young woman singing in a misty pine forest at dawn, cinematic, golden light filtering through trees, intimate close-up portrait, lips moving with the music"
NEGATIVE = "ugly, deformed, bad anatomy, extra limbs, blurry, low quality, watermark, music"
W, H, FPS = 960, 544, 24
DURATION = 4.0
SEED = 42

target_frames = int(round(DURATION * FPS))
frame_count = max(9, ((target_frames - 1) // 8) * 8 + 1)
steps, cfg = 20, 3.0

OUTPUT_DIR = "/home/ericr/ComfyUI/output"
AUDIO_PATH = "/home/ericr/ComfyUI/input/pines_clip.mp3"

wf = {
    "10": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
    "11": {"class_type": "VAELoader", "inputs": {"vae_name": VIDEO_VAE}},
    "12": {"class_type": "LTXVAudioVAELoader", "inputs": {"ckpt_name": CKPT}},
    "13": {"class_type": "LTXAVTextEncoderLoader", "inputs": {
        "text_encoder": TEXT_ENCODER, "ckpt_name": CKPT, "device": "default"}},
    "20": {"class_type": "LoraLoaderModelOnly", "inputs": {
        "model": ["10", 0], "lora_name": LORA, "strength_model": 1.0}},
    "30": {"class_type": "CLIPTextEncode", "inputs": {"text": PROMPT, "clip": ["13", 0]}},
    "31": {"class_type": "CLIPTextEncode", "inputs": {"text": NEGATIVE, "clip": ["13", 0]}},
    "32": {"class_type": "LTXVConditioning", "inputs": {
        "positive": ["30", 0], "negative": ["31", 0], "frame_rate": float(FPS)}},
    "40": {"class_type": "LoadImage", "inputs": {"image": IMAGE}},
    "41": {"class_type": "LoadAudio", "inputs": {"audio": AUDIO}},
    "42": {"class_type": "LTXVAudioVAEEncode", "inputs": {
        "audio": ["41", 0], "audio_vae": ["12", 0]}},
    "43": {"class_type": "EmptyLTXVLatentVideo", "inputs": {
        "width": W, "height": H, "length": frame_count, "batch_size": 1}},
    "44": {"class_type": "LTXVImgToVideoInplace", "inputs": {
        "vae": ["11", 0], "image": ["40", 0], "latent": ["43", 0],
        "strength": 1.0, "bypass": False}},
    "45": {"class_type": "LTXVConcatAVLatent", "inputs": {
        "video_latent": ["44", 0], "audio_latent": ["42", 0]}},
    "50": {"class_type": "RandomNoise", "inputs": {"noise_seed": SEED}},
    "51": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
    "52": {"class_type": "BasicScheduler", "inputs": {
        "model": ["20", 0], "scheduler": "linear_quadratic",
        "steps": steps, "denoise": 1.0}},
    "53": {"class_type": "CFGGuider", "inputs": {
        "model": ["20", 0], "positive": ["32", 0], "negative": ["32", 1], "cfg": float(cfg)}},
    "54": {"class_type": "SamplerCustomAdvanced", "inputs": {
        "noise": ["50", 0], "guider": ["53", 0], "sampler": ["51", 0],
        "sigmas": ["52", 0], "latent_image": ["45", 0]}},
    "60": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["54", 0]}},
    "61": {"class_type": "LTXVSpatioTemporalTiledVAEDecode", "inputs": {
        "vae": ["11", 0], "latents": ["60", 0],
        "spatial_tiles": 2, "spatial_overlap": 4,
        "temporal_tile_length": 16, "temporal_overlap": 4,
        "last_frame_fix": False, "working_device": "auto", "working_dtype": "auto"}},
    "70": {"class_type": "CreateVideo", "inputs": {"images": ["61", 0], "fps": float(FPS)}},
    "71": {"class_type": "SaveVideo", "inputs": {
        "video": ["70", 0], "filename_prefix": "lipsync_raw",
        "format": "mp4", "codec": "h264"}},
}

print(f"Audio-conditioned I2V: {DURATION}s ({frame_count}fr) @ {W}x{H}")
print(f"Image: {IMAGE}, Audio: {AUDIO}, Seed: {SEED}")

# Submit
client_id = f"test_{int(time.time())}"
req = urllib.request.Request(f"{COMFY}/prompt",
    data=json.dumps({"prompt": wf, "client_id": client_id}).encode(),
    headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=30) as r:
    res = json.loads(r.read())
pid = res.get("prompt_id")
print(f"Prompt ID: {pid}")

# Wait
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
            for v in out.get("videos", out.get("media", [])):
                raw_video = os.path.join(OUTPUT_DIR, v.get("filename"))
                print(f"  Raw video: {v.get('filename')}")
        elapsed = time.time() - start
        print(f"Generated in {elapsed:.1f}s — {data.get('status',{}).get('status_str','?')}")
        break
    time.sleep(2)
else:
    print("TIMEOUT")
    sys.exit(1)

# Mux original audio into the video
if raw_video and os.path.exists(raw_video):
    final_path = os.path.join(OUTPUT_DIR, "lipsync_final.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-i", raw_video,
        "-i", AUDIO_PATH,
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        "-movflags", "+faststart",
        final_path
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration,size",
         "-of", "default=noprint_wrappers=1", final_path],
        capture_output=True, text=True)
    print(f"\nFinal output: {final_path}")
    print(result.stdout)
    print("Done — play lipsync_final.mp4 to check lip sync against original audio")
else:
    print("No raw video found")
