#!/usr/bin/env python3
"""Extract the working workflow as API JSON."""
import json, subprocess

# This is the exact workflow from test_fixed.py that ran successfully
ckpt = "ltx-2.3-22b-distilled-1.1.safetensors"
video_vae = "ltx-2.3-22b-distilled_video_vae.safetensors"
text_enc = "gemma_3_12B_it_fp4_mixed.safetensors"
lora = "ltx-2.3-22b-distilled-lora-384-1.1.safetensors"
ref = "ref_medium_shot.png"
audio = "vocals_4s.wav"
prompt = "young woman with long dark hair white dress, medium shot facing viewer chest up, singing on misty pine forest path at golden dawn, sharp, cinematic, photorealistic, detailed"
neg = "headshot, close up, portrait, from behind, back view, ugly, deformed, blurry, low quality, cartoon"

wf = {
    "10": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
    "11": {"class_type": "VAELoader", "inputs": {"vae_name": video_vae}},
    "12": {"class_type": "LTXVAudioVAELoader", "inputs": {"ckpt_name": ckpt}},
    "13": {"class_type": "LTXAVTextEncoderLoader", "inputs": {"text_encoder": text_enc, "ckpt_name": ckpt, "device": "default"}},
    "20": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["10",0], "lora_name": lora, "strength_model": 0.8}},
    "30": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["13",0]}},
    "31": {"class_type": "CLIPTextEncode", "inputs": {"text": neg, "clip": ["13",0]}},
    "32": {"class_type": "LTXVConditioning", "inputs": {"positive": ["30",0], "negative": ["31",0], "frame_rate": 24.0}},
    "40": {"class_type": "LoadImage", "inputs": {"image": ref}},
    "41": {"class_type": "LoadAudio", "inputs": {"audio": audio}},
    "42": {"class_type": "LTXVAudioVAEEncode", "inputs": {"audio": ["41",0], "audio_vae": ["12",0]}},
    "43": {"class_type": "EmptyLTXVLatentVideo", "inputs": {"width": 832, "height": 480, "length": 89, "batch_size": 1}},
    "44": {"class_type": "LTXVImgToVideoInplace", "inputs": {"vae": ["11",0], "image": ["40",0], "latent": ["43",0], "strength": 0.7, "bypass": False}},
    "46": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": ["44",0], "audio_latent": ["42",0]}},
    "47": {"class_type": "LTXVSetAudioVideoMaskByTime", "inputs": {"av_latent": ["46",0], "positive": ["32",0], "negative": ["32",1], "model": ["20",0], "vae": ["11",0], "audio_vae": ["12",0], "start_time": 0.0, "end_time": 4.0, "video_fps": 24.0, "mask_video": True, "mask_audio": False, "mask_init_value_video": 0.0, "mask_init_value_audio": 0.0, "slope_len": 3}},
    "50": {"class_type": "RandomNoise", "inputs": {"noise_seed": 70}},
    "51": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
    "52": {"class_type": "BasicScheduler", "inputs": {"model": ["20",0], "scheduler": "linear_quadratic", "steps": 15, "denoise": 1.0}},
    "53": {"class_type": "CFGGuider", "inputs": {"model": ["20",0], "positive": ["47",0], "negative": ["47",1], "cfg": 3.0}},
    "54": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["50",0], "guider": ["53",0], "sampler": ["51",0], "sigmas": ["52",0], "latent_image": ["47",2]}},
    "60": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["54",0]}},
    "61": {"class_type": "LTXVSpatioTemporalTiledVAEDecode", "inputs": {"vae": ["11",0], "latents": ["60",0], "spatial_tiles": 2, "spatial_overlap": 4, "temporal_tile_length": 16, "temporal_overlap": 4, "last_frame_fix": False, "working_device": "auto", "working_dtype": "auto"}},
    "62": {"class_type": "LTXVAudioVAEDecode", "inputs": {"samples": ["60",1], "audio_vae": ["12",0]}},
    "70": {"class_type": "CreateVideo", "inputs": {"images": ["61",0], "fps": 24.0}},
    "71": {"class_type": "SaveVideo", "inputs": {"video": ["70",0], "filename_prefix": "ltx_mv", "format": "mp4", "codec": "h264"}},
}

path = "/home/ericr/ComfyUI/workflow_i2v_audio_api.json"
with open(path, "w") as f:
    json.dump(wf, f, indent=2)

# Validate by re-loading
with open(path) as f:
    loaded = json.load(f)
print(f"✅ {path} — {len(loaded)} nodes, valid JSON")

# Also output a simpler config summary
print(f"\nConfiguration:")
print(f"  Model: {ckpt}")
print(f"  VAE: {video_vae}")
print(f"  Text Encoder: {text_enc}")
print(f"  LoRA: {lora} @ 0.8")
print(f"  Reference: {ref}")
print(f"  Audio: {audio}")
print(f"  Resolution: 832x480 @ 24fps, {89} frames ({89/24:.1f}s)")
print(f"  Steps: 15, CFG: 3.0, Sampler: euler")
print(f"  I2V Strength: 0.7")
print(f"  Audio mask: start=0, end=4.0, init_video=0.0, init_audio=0.0")
