#!/usr/bin/env python3
"""Test with spatial upscaler on latent for better quality."""
import json, urllib.request, time, subprocess, os

COMFY = "http://127.0.0.1:8188"
OUT = "/home/ericr/ComfyUI/output"
CKPT = "ltx-2.3-22b-distilled-1.1.safetensors"
VIDEO_VAE = "ltx-2.3-22b-distilled_video_vae.safetensors"
TEXT_ENC = "gemma_3_12B_it_fp4_mixed.safetensors"
LTX_LORA = "ltx-2.3-22b-distilled-lora-384-1.1.safetensors"
SDXL = "sd_xl_base_1.0.safetensors"

W, H, FPS = 768, 512, 24  # Both divisible by 32, good for upscaling
steps = 20
duration = 4.0

P = "young woman with long dark hair white dress, medium shot chest up facing viewer, misty pine forest path golden dawn, cinematic, sharp, singing"
N = "headshot, close up, portrait, from behind, back view, ugly, deformed, blurry, low quality"

# Ensure reference image exists
ref = "/home/ericr/ComfyUI/input/ref_best.png"
if not os.path.exists(ref):
    print("Generating SDXL reference...")
    sdxl_wf = {
        "10": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": SDXL}},
        "20": {"class_type": "CLIPTextEncode", "inputs": {"text": P, "clip": ["10",1]}},
        "21": {"class_type": "CLIPTextEncode", "inputs": {"text": N, "clip": ["10",1]}},
        "22": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        "23": {"class_type": "KSampler", "inputs": {"model": ["10",0], "positive": ["20",0], "negative": ["21",0], "latent_image": ["22",0], "seed": 700, "steps": 30, "cfg": 7.0, "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0}},
        "24": {"class_type": "VAEDecode", "inputs": {"vae": ["10",2], "samples": ["23",0]}},
        "25": {"class_type": "SaveImage", "inputs": {"images": ["24",0], "filename_prefix": "ref_best"}},
    }
    req = urllib.request.Request(f"{COMFY}/prompt", data=json.dumps({"prompt": sdxl_wf}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        pid = json.loads(r.read())["prompt_id"]
    start = time.time()
    while time.time()-start < 120:
        try:
            h = json.loads(urllib.request.urlopen(f"{COMFY}/history/{pid}").read())
        except: time.sleep(2); continue
        if pid in h:
            if h[pid]['status']['status_str'] == 'success':
                for no, out in h[pid].get("outputs",{}).items():
                    for img in out.get("images",[]):
                        fp = os.path.join(OUT, img["filename"])
                        import shutil
                        shutil.copy(fp, ref)
                        print(f"  → ref_best.png ({os.path.getsize(fp)/1024:.0f} KB)")
            break
        time.sleep(2)

# Generate video at 768x512, then upscale latent to 1536x1024
vocals = f"/home/ericr/ComfyUI/input/vocals_4s.wav"
if not os.path.exists(vocals):
    subprocess.run(["ffmpeg","-y","-i","/home/ericr/ComfyUI/input/pines_vocals.wav","-t","4","-c","copy",vocals],check=True,capture_output=True)

fc = max(9, ((int(round(duration*FPS))-1)//8)*8+1)
print(f"Generating at {W}x{H}, {fc} frames...")

wf = {
    "10": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
    "11": {"class_type": "VAELoader", "inputs": {"vae_name": VIDEO_VAE}},
    "12": {"class_type": "LTXVAudioVAELoader", "inputs": {"ckpt_name": CKPT}},
    "13": {"class_type": "LTXAVTextEncoderLoader", "inputs": {"text_encoder": TEXT_ENC, "ckpt_name": CKPT, "device": "default"}},
    "20": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["10",0], "lora_name": LTX_LORA, "strength_model": 0.8}},
    "30": {"class_type": "CLIPTextEncode", "inputs": {"text": P, "clip": ["13",0]}},
    "31": {"class_type": "CLIPTextEncode", "inputs": {"text": N, "clip": ["13",0]}},
    "32": {"class_type": "LTXVConditioning", "inputs": {"positive": ["30",0], "negative": ["31",0], "frame_rate": float(FPS)}},
    "40": {"class_type": "LoadImage", "inputs": {"image": "ref_best.png"}},
    "41": {"class_type": "LoadAudio", "inputs": {"audio": "vocals_4s.wav"}},
    "42": {"class_type": "LTXVAudioVAEEncode", "inputs": {"audio": ["41",0], "audio_vae": ["12",0]}},
    "43": {"class_type": "EmptyLTXVLatentVideo", "inputs": {"width": W, "height": H, "length": fc, "batch_size": 1}},
    "44": {"class_type": "LTXVImgToVideoInplace", "inputs": {"vae": ["11",0], "image": ["40",0], "latent": ["43",0], "strength": 0.8, "bypass": False}},
    "46": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": ["44",0], "audio_latent": ["42",0]}},
    "47": {"class_type": "LTXVSetAudioVideoMaskByTime", "inputs": {"av_latent": ["46",0], "positive": ["32",0], "negative": ["32",1], "model": ["20",0], "vae": ["11",0], "audio_vae": ["12",0], "start_time": 0.0, "end_time": float(duration), "video_fps": float(FPS), "mask_video": True, "mask_audio": False, "mask_init_value_video": 0.0, "mask_init_value_audio": 0.0, "slope_len": 3}},
    "50": {"class_type": "RandomNoise", "inputs": {"noise_seed": 700}},
    "51": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
    "52": {"class_type": "BasicScheduler", "inputs": {"model": ["20",0], "scheduler": "linear_quadratic", "steps": steps, "denoise": 1.0}},
    "53": {"class_type": "CFGGuider", "inputs": {"model": ["20",0], "positive": ["47",0], "negative": ["47",1], "cfg": 3.0}},
    "54": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["50",0], "guider": ["53",0], "sampler": ["51",0], "sigmas": ["52",0], "latent_image": ["47",2]}},
    "60": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["54",0]}},
    "70": {"class_type": "UpscaleModelLoader", "inputs": {"model_name": "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"}},
    "71": {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["70",0], "image": ["60",0]}},
}

# Test just the upscaler connection
print("Testing upscaler workflow...")
req = urllib.request.Request(f"{COMFY}/prompt", data=json.dumps({"prompt": wf}).encode(),
    headers={"Content-Type": "application/json"})
try:
    with urllib.request.urlopen(req, timeout=30) as r:
        pid = json.loads(r.read())["prompt_id"]
        print(f"Queued: {pid}")
except urllib.error.HTTPError as e:
    print(f"Error: {e.read().decode()[:600]}")
PYEOF