#!/usr/bin/env python3
"""I2V test with medium-shot reference, high I2V strength, quality LoRA."""
import json, urllib.request, time, subprocess, os

COMFY = "http://127.0.0.1:8188"
OUTPUT_DIR = "/home/ericr/ComfyUI/output"
CKPT = "ltx-2.3-22b-distilled-1.1.safetensors"
VIDEO_VAE = "ltx-2.3-22b-distilled_video_vae.safetensors"
TEXT_ENCODER = "gemma_3_12B_it_fp4_mixed.safetensors"
LORA = "ltx-2.3-22b-distilled-lora-384-1.1.safetensors"
W, H, FPS = 832, 480, 24
steps = 15
duration = 4.0

REF_IMAGE = "ref_medium_shot.png"
SEED = 70

PROMPT = "young woman with long dark hair white dress, medium shot facing viewer chest up, singing on misty pine forest path at golden dawn, sharp, cinematic, photorealistic, detailed"
NEG = "headshot, close up, portrait, from behind, back view, ugly, deformed, blurry, low quality, cartoon"

vocals_4s = "/home/ericr/ComfyUI/input/vocals_4s.wav"
if not os.path.exists(vocals_4s):
    subprocess.run(["ffmpeg","-y","-i","/home/ericr/ComfyUI/input/pines_vocals.wav",
        "-t","4","-c","copy",vocals_4s],check=True,capture_output=True)

fc = max(9, ((int(round(duration*FPS))-1)//8)*8+1)
print(f"Medium ref test: {W}x{H} {fc}frames, I2V strength=0.7, LORA={LORA}")

wf = {
    "10": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
    "11": {"class_type": "VAELoader", "inputs": {"vae_name": VIDEO_VAE}},
    "12": {"class_type": "LTXVAudioVAELoader", "inputs": {"ckpt_name": CKPT}},
    "13": {"class_type": "LTXAVTextEncoderLoader", "inputs": {"text_encoder": TEXT_ENCODER, "ckpt_name": CKPT, "device": "default"}},
    "20": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["10",0], "lora_name": LORA, "strength_model": 0.8}},
    "30": {"class_type": "CLIPTextEncode", "inputs": {"text": PROMPT, "clip": ["13",0]}},
    "31": {"class_type": "CLIPTextEncode", "inputs": {"text": NEG, "clip": ["13",0]}},
    "32": {"class_type": "LTXVConditioning", "inputs": {"positive": ["30",0], "negative": ["31",0], "frame_rate": float(FPS)}},
    "40": {"class_type": "LoadImage", "inputs": {"image": REF_IMAGE}},
    "41": {"class_type": "LoadAudio", "inputs": {"audio": "vocals_4s.wav"}},
    "42": {"class_type": "LTXVAudioVAEEncode", "inputs": {"audio": ["41",0], "audio_vae": ["12",0]}},
    "43": {"class_type": "EmptyLTXVLatentVideo", "inputs": {"width": W, "height": H, "length": fc, "batch_size": 1}},
    "44": {"class_type": "LTXVImgToVideoInplace", "inputs": {"vae": ["11",0], "image": ["40",0], "latent": ["43",0], "strength": 0.7, "bypass": False}},
    "46": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": ["44",0], "audio_latent": ["42",0]}},
    "47": {"class_type": "LTXVSetAudioVideoMaskByTime", "inputs": {"av_latent": ["46",0], "positive": ["32",0], "negative": ["32",1], "model": ["20",0], "vae": ["11",0], "audio_vae": ["12",0], "start_time": 0.0, "end_time": float(duration), "video_fps": float(FPS), "mask_video": True, "mask_audio": False, "mask_init_value_video": 0.0, "mask_init_value_audio": 0.0, "slope_len": 3}},
    "50": {"class_type": "RandomNoise", "inputs": {"noise_seed": SEED}},
    "51": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
    "52": {"class_type": "BasicScheduler", "inputs": {"model": ["20",0], "scheduler": "linear_quadratic", "steps": steps, "denoise": 1.0}},
    "53": {"class_type": "CFGGuider", "inputs": {"model": ["20",0], "positive": ["47",0], "negative": ["47",1], "cfg": 3.0}},
    "54": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["50",0], "guider": ["53",0], "sampler": ["51",0], "sigmas": ["52",0], "latent_image": ["47",2]}},
    "60": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["54",0]}},
    "61": {"class_type": "LTXVSpatioTemporalTiledVAEDecode", "inputs": {"vae": ["11",0], "latents": ["60",0], "spatial_tiles": 2, "spatial_overlap": 4, "temporal_tile_length": 16, "temporal_overlap": 4, "last_frame_fix": False, "working_device": "auto", "working_dtype": "auto"}},
    "70": {"class_type": "CreateVideo", "inputs": {"images": ["61",0], "fps": float(FPS)}},
    "71": {"class_type": "SaveVideo", "inputs": {"video": ["70",0], "filename_prefix": "test_fixed", "format": "mp4", "codec": "h264"}},
}

req = urllib.request.Request(f"{COMFY}/prompt", data=json.dumps({"prompt": wf}).encode(),
    headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=30) as r:
    pid = json.loads(r.read())["prompt_id"]
print(f"Queued: {pid}")

start = time.time()
rv = None
while time.time()-start < 600:
    try:
        h = json.loads(urllib.request.urlopen(f"{COMFY}/history/{pid}").read())
    except: time.sleep(5); continue
    if pid in h:
        elapsed = time.time()-start
        print(f"{elapsed:.0f}s — {h[pid]['status']['status_str']}")
        if h[pid]['status']['status_str'] == 'success':
            for no, out in h[pid].get("outputs",{}).items():
                for v in out.get("images", out.get("media",[])):
                    rv = os.path.join(OUTPUT_DIR, v.get("filename"))
                    if rv and os.path.exists(rv) and rv.endswith('.mp4'):
                        final = rv.replace(".mp4", "_wav.mp4")
                        subprocess.run(["ffmpeg","-y","-i",rv,"-i",vocals_4s,
                            "-c:v","copy","-c:a","aac","-b:a","192k","-shortest",final],
                            check=True, capture_output=True)
                        r = subprocess.run(["ffprobe","-v","error","-show_entries",
                            "format=duration,size","-of","csv=p=0",final],
                            capture_output=True, text=True)
                        print(f"→ {os.path.basename(final)} ({r.stdout.strip()})")
                        rv = final
        break
    time.sleep(5)
else: print("TIMEOUT")
