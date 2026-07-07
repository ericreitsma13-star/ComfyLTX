#!/usr/bin/env python3
"""Compare silent I2V vs audio-conditioned I2V to verify lip sync works."""
import json, urllib.request, time, sys

COMFY = "http://127.0.0.1:8188"
CKPT = "ltx-2.3-22b-distilled-1.1.safetensors"
VIDEO_VAE = "ltx-2.3-22b-distilled_video_vae.safetensors"
TEXT_ENCODER = "gemma_3_12B_it_fp4_mixed.safetensors"
LORA = "ltx-2.3-22b-distilled-lora-1.1_fro90_ceil72_condsafe.safetensors"
IMAGE = "example.png"
AUDIO = "pines_clip.mp3"
PROMPT = "A woman singing in a misty pine forest at dawn, cinematic, golden light, intimate close-up"
NEGATIVE = "ugly, deformed, bad anatomy, extra limbs, blurry, low quality, watermark, music"
W, H, FPS = 832, 480, 24
DURATION = 4.0
SEED = 42  # Same seed for fair comparison

target_frames = int(round(DURATION * FPS))
frame_count = max(9, ((target_frames - 1) // 8) * 8 + 1)
steps, cfg = 20, 3.0

def build_workflow(audio_cond, prefix):
    """Build workflow. If audio_cond=True, inject real audio; else use empty audio latent."""
    # Shared nodes
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
        "43": {"class_type": "EmptyLTXVLatentVideo", "inputs": {
            "width": W, "height": H, "length": frame_count, "batch_size": 1}},
        "44": {"class_type": "LTXVImgToVideoInplace", "inputs": {
            "vae": ["11", 0], "image": ["40", 0], "latent": ["43", 0],
            "strength": 1.0, "bypass": False}},
    }

    if audio_cond:
        # Audio-conditioned path: encode real audio
        wf["41"] = {"class_type": "LoadAudio", "inputs": {"audio": AUDIO}}
        wf["42"] = {"class_type": "LTXVAudioVAEEncode", "inputs": {
            "audio": ["41", 0], "audio_vae": ["12", 0]}}
        wf["45"] = {"class_type": "LTXVConcatAVLatent", "inputs": {
            "video_latent": ["44", 0], "audio_latent": ["42", 0]}}
    else:
        # Silent path: empty audio latent
        wf["42"] = {"class_type": "LTXVEmptyLatentAudio", "inputs": {
            "frames_number": frame_count, "frame_rate": float(FPS),
            "batch_size": 1, "audio_vae": ["12", 0]}}
        wf["45"] = {"class_type": "LTXVConcatAVLatent", "inputs": {
            "video_latent": ["44", 0], "audio_latent": ["42", 0]}}

    # Sampling
    wf["50"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": SEED}}
    wf["51"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}}
    wf["52"] = {"class_type": "BasicScheduler", "inputs": {
        "model": ["20", 0], "scheduler": "linear_quadratic",
        "steps": steps, "denoise": 1.0}}
    wf["53"] = {"class_type": "CFGGuider", "inputs": {
        "model": ["20", 0], "positive": ["32", 0], "negative": ["32", 1], "cfg": float(cfg)}}
    wf["54"] = {"class_type": "SamplerCustomAdvanced", "inputs": {
        "noise": ["50", 0], "guider": ["53", 0], "sampler": ["51", 0],
        "sigmas": ["52", 0], "latent_image": ["45", 0]}}

    # Separate AV and decode
    wf["60"] = {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["54", 0]}}
    wf["61"] = {"class_type": "LTXVSpatioTemporalTiledVAEDecode", "inputs": {
        "vae": ["11", 0], "latents": ["60", 0],
        "spatial_tiles": 2, "spatial_overlap": 4,
        "temporal_tile_length": 16, "temporal_overlap": 4,
        "last_frame_fix": False, "working_device": "auto", "working_dtype": "auto"}}
    wf["70"] = {"class_type": "CreateVideo", "inputs": {"images": ["61", 0], "fps": float(FPS)}}
    wf["71"] = {"class_type": "SaveVideo", "inputs": {
        "video": ["70", 0], "filename_prefix": prefix, "format": "mp4", "codec": "h264"}}
    return wf

def submit_and_wait(wf, label):
    client_id = f"test_{int(time.time())}"
    req = urllib.request.Request(f"{COMFY}/prompt",
        data=json.dumps({"prompt": wf, "client_id": client_id}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        res = json.loads(r.read())
    pid = res.get("prompt_id")
    print(f"[{label}] Prompt ID: {pid}")

    start = time.time()
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
                    print(f"[{label}] Output: {v.get('filename')}")
            elapsed = time.time() - start
            status = data.get('status', {}).get('status_str', '?')
            print(f"[{label}] Done in {elapsed:.1f}s — {status}")
            return elapsed, status
        time.sleep(2)
    print(f"[{label}] TIMEOUT")
    return None, "timeout"

# Run silent first, then audio-conditioned
print("=" * 60)
print("TEST 1: Silent I2V (empty audio latent)")
print("=" * 60)
wf_silent = build_workflow(audio_cond=False, prefix="test_silent")
t1, s1 = submit_and_wait(wf_silent, "SILENT")

print()
print("=" * 60)
print("TEST 2: Audio-conditioned I2V (real audio injected)")
print("=" * 60)
wf_audio = build_workflow(audio_cond=True, prefix="test_audiocond")
t2, s2 = submit_and_wait(wf_audio, "AUDIO_COND")

print()
print("=" * 60)
print("RESULTS")
print("=" * 60)
print(f"Silent I2V:        {t1:.1f}s — {s1}" if t1 else f"Silent I2V:        FAILED")
print(f"Audio-conditioned: {t2:.1f}s — {s2}" if t2 else f"Audio-conditioned: FAILED")

# Check if files differ
import subprocess, os
output_dir = "/home/ericr/ComfyUI/output"
for f in sorted(os.listdir(output_dir)):
    if f.startswith("test_silent") or f.startswith("test_audiocond"):
        fp = os.path.join(output_dir, f)
        size = os.path.getsize(fp)
        print(f"  {f}: {size} bytes")
