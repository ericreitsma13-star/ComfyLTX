#!/usr/bin/env python3
"""Quick I2V test with CondSafe LoRA."""
import json, urllib.request, time, sys

COMFY = "http://127.0.0.1:8188"
CKPT = "ltx-2.3-22b-distilled-1.1.safetensors"
VAE = "ltx-2.3-22b-distilled_video_vae.safetensors"
TEXT_ENCODER = "gemma_3_12B_it_fp4_mixed.safetensors"
LORA = "ltx-2.3-22b-distilled-lora-1.1_fro90_ceil72_condsafe.safetensors"
IMAGE = "example.png"
PROMPT = "A woman walking through a misty pine forest at dawn, cinematic, golden light filtering through trees"
NEGATIVE = "ugly, deformed, bad anatomy, extra limbs, blurry, low quality, watermark"
W, H, FPS = 832, 480, 24
DURATION = 4.0
SEED = 42

target_frames = int(round(DURATION * FPS))
frame_count = max(9, ((target_frames - 1) // 8) * 8 + 1)
steps, cfg = 20, 3.0

wf = {
    "10": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
    "11": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
    "12": {"class_type": "LTXAVTextEncoderLoader", "inputs": {
        "text_encoder": TEXT_ENCODER, "ckpt_name": CKPT, "device": "default"}},
    "20": {"class_type": "LoraLoaderModelOnly", "inputs": {
        "model": ["10", 0], "lora_name": LORA, "strength_model": 1.0}},
    "30": {"class_type": "CLIPTextEncode", "inputs": {"text": PROMPT, "clip": ["12", 0]}},
    "31": {"class_type": "CLIPTextEncode", "inputs": {"text": NEGATIVE, "clip": ["12", 0]}},
    "32": {"class_type": "LTXVConditioning", "inputs": {
        "positive": ["30", 0], "negative": ["31", 0], "frame_rate": float(FPS)}},
    "40": {"class_type": "LoadImage", "inputs": {"image": IMAGE}},
    "41": {"class_type": "EmptyLTXVLatentVideo", "inputs": {
        "width": W, "height": H, "length": frame_count, "batch_size": 1}},
    "42": {"class_type": "LTXVImgToVideoInplace", "inputs": {
        "vae": ["11", 0], "image": ["40", 0], "latent": ["41", 0],
        "strength": 1.0, "bypass": False}},
    "50": {"class_type": "RandomNoise", "inputs": {"noise_seed": SEED}},
    "51": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
    "52": {"class_type": "BasicScheduler", "inputs": {
        "model": ["20", 0], "scheduler": "linear_quadratic",
        "steps": steps, "denoise": 1.0}},
    "53": {"class_type": "CFGGuider", "inputs": {
        "model": ["20", 0], "positive": ["32", 0], "negative": ["32", 1], "cfg": float(cfg)}},
    "54": {"class_type": "SamplerCustomAdvanced", "inputs": {
        "noise": ["50", 0], "guider": ["53", 0], "sampler": ["51", 0],
        "sigmas": ["52", 0], "latent_image": ["42", 0]}},
    "61": {"class_type": "LTXVSpatioTemporalTiledVAEDecode", "inputs": {
        "vae": ["11", 0], "latents": ["54", 0],
        "spatial_tiles": 2, "spatial_overlap": 4,
        "temporal_tile_length": 16, "temporal_overlap": 4,
        "last_frame_fix": False, "working_device": "auto", "working_dtype": "auto"}},
    "70": {"class_type": "CreateVideo", "inputs": {"images": ["61", 0], "fps": float(FPS)}},
    "71": {"class_type": "SaveVideo", "inputs": {
        "video": ["70", 0], "filename_prefix": "test_condsafe", "format": "mp4", "codec": "h264"}},
}

print(f"Submitting I2V: {DURATION}s ({frame_count}frames) @ {W}x{H}")
print(f"CondSafe LoRA strength: 1.0, steps: {steps}, CFG: {cfg}, seed: {SEED}")

client_id = f"test_{int(time.time())}"
req = urllib.request.Request(f"{COMFY}/prompt",
    data=json.dumps({"prompt": wf, "client_id": client_id}).encode(),
    headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=30) as r:
    res = json.loads(r.read())
pid = res.get("prompt_id")
print(f"Prompt ID: {pid}")

start = time.time()
while time.time() - start < 300:
    h = json.loads(urllib.request.urlopen(f"{COMFY}/history/{pid}").read())
    if pid in h:
        data = h[pid]
        outputs = data.get("outputs", {})
        for nid, out in outputs.items():
            videos = out.get("videos", out.get("media", []))
            if videos:
                for v in videos:
                    print(f"  Output: {v.get('filename')} ({v.get('type','')})")
        elapsed = time.time() - start
        print(f"Done in {elapsed:.1f}s — {data.get('status',{}).get('status_str','?')}")
        break
    time.sleep(2)
else:
    print("TIMEOUT — check ComfyUI")
