#!/usr/bin/env python3
"""Wan2.2 T2V workflow test via ComfyUI API"""
import json, sys, time, os, requests, uuid, struct, io as sio
from PIL import Image

API_HOST = "http://127.0.0.1:8188"
CLIENT_ID = str(uuid.uuid4())

def queue_prompt(workflow):
    r = requests.post(f"{API_HOST}/prompt", json={"prompt": workflow, "client_id": CLIENT_ID})
    r.raise_for_status()
    return r.json()

def get_history(prompt_id):
    r = requests.get(f"{API_HOST}/history/{prompt_id}")
    r.raise_for_status()
    return r.json()

def get_node_outputs(history, prompt_id):
    return history.get(prompt_id, {}).get("outputs", {})

def load_checkpoint(ckpt_name):
    """Load Wan2.2 for dual-model: high noise + low noise"""
    # Node 1: VAELoader
    vae = node("VAELoader", {"vae_name": "wan_2.1_vae.safetensors"})
    # Node 2: CLIPLoader (wan type)
    clip = node("CLIPLoader", {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan"})
    # Node 3: UNETLoader (high noise)
    unet_high = node("UNETLoader", {"unet_name": "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors", "weight_dtype": "default"})
    # Node 4: UNETLoader (low noise)
    unet_low = node("UNETLoader", {"unet_name": "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors", "weight_dtype": "default"})
    return vae, clip, unet_high, unet_low

def node(class_type, inputs=None):
    return {"class_type": class_type, "inputs": inputs or {}}

def link(nd, slot=0):
    return [nd, slot]

# =========== BUILD WORKFLOW ===========
w = {}

# 1. Loaders
w["1"] = node("VAELoader", {"vae_name": "wan_2.1_vae.safetensors"})
w["2"] = node("CLIPLoader", {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan"})

# High noise model
w["3"] = node("UNETLoader", {
    "unet_name": "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors",
    "weight_dtype": "default"
})
# Low noise model  
w["4"] = node("UNETLoader", {
    "unet_name": "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors",
    "weight_dtype": "default"
})

# 2. CLIP Text Encode (positive)
w["5"] = node("CLIPTextEncode", {
    "text": "a cute fluffy cat walking through a sunlit garden with flowers, cinematic quality, sharp focus, 4K",
    "clip": link("2")
})

# 3. CLIP Text Encode (negative)
w["6"] = node("CLIPTextEncode", {
    "text": "blurry, low quality, distorted, ugly, bad anatomy, watermark, text",
    "clip": link("2")
})

# 4. Wan22 ImageToVideo Latent (empty latent for T2V)
w["7"] = node("Wan22ImageToVideoLatent", {
    "vae": link("1"),
    "width": 832,
    "height": 480,
    "length": 49,
    "batch_size": 1
})

# 5. First KSampler (high noise model, denoise 0.6 = first 60% of steps)
w["8"] = node("KSampler", {
    "seed": 42,
    "steps": 30,
    "cfg": 5.0,
    "sampler_name": "euler",
    "scheduler": "sgm_uniform",
    "denoise": 0.6,
    "model": link("3"),
    "positive": link("5"),
    "negative": link("6"),
    "latent_image": link("7")
})

# 6. Second KSampler (low noise model, denoise 1.0 of remaining 40%)
w["9"] = node("KSampler", {
    "seed": 42,
    "steps": 30,
    "cfg": 5.0,
    "sampler_name": "euler",
    "scheduler": "sgm_uniform",
    "denoise": 1.0,
    "model": link("4"),
    "positive": link("5"),
    "negative": link("6"),
    "latent_image": link("8")
})

# 7. VAE Decode
w["10"] = node("VAEDecode", {
    "samples": link("9"),
    "vae": link("1")
})

# 8. Save as video frames (PNG sequence)
w["11"] = node("SaveAnimatedPNG", {
    "images": link("10"),
    "filename_prefix": "wan22_test",
    "fps": 8,
    "lossless_webp": True,
    "quality": 95
})

# 9. Also save a preview image
w["12"] = node("PreviewImage", {
    "images": link("10")
})

if __name__ == "__main__":
    print(f"Connecting to ComfyUI at {API_HOST}...")
    try:
        r = requests.get(f"{API_HOST}/object_info", timeout=5)
        print("ComfyUI is running!")
    except requests.exceptions.ConnectionError:
        print("ComfyUI not running. Launch it first:")
        print("  cd /home/ericr/ComfyUI && source venv/bin/activate && python main.py --listen")
        sys.exit(1)

    print("Queueing Wan2.2 T2V workflow...")
    result = queue_prompt(w)
    pid = result["prompt_id"]
    print(f"Prompt queued: {pid}")

    # Poll for completion
    while True:
        time.sleep(5)
        history = get_history(pid)
        if pid in history:
            outputs = get_node_outputs(history, pid)
            print(f"\nDone! Outputs: {json.dumps(outputs, indent=2)}")
            break
        status = requests.get(f"{API_HOST}/queue").json()
        remaining = status.get("queue_running", 0) + len(status.get("queue_pending", []))
        print(f"  Running... queue remaining: {remaining}", end="\r")
