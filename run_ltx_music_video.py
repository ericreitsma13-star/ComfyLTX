import json, requests, random, sys, subprocess, os

API = "http://127.0.0.1:8188/prompt"

prompt_text = sys.argv[1] if len(sys.argv) > 1 else "mysterious young woman singing in a misty forest at dawn, ethereal, cinematic, soft light through pine trees, flowing dark hair, wearing a flowing white dress, atmospheric fog, dappled sunlight, intimate close-up, emotional, hauntingly beautiful"
CKPT = "ltx-2.3-22b-distilled-1.1.safetensors"
AUDIO_FILE = "input/pines_clip.mp3"
WIDTH, HEIGHT, FPS = 960, 544, 24
NUM_FRAMES = 288

def node(class_type, **inputs):
    return {"class_type": class_type, "inputs": inputs}

# Step 1: Generate video with model audio
workflow = {
    "1": node("UNETLoader", unet_name=CKPT, weight_dtype="default"),
    "2": node("LoraLoaderModelOnly", model=("1", 0), lora_name="ltx-2.3-22b-distilled-lora-384-1.1.safetensors", strength_model=0.5),
    "3": node("LoraLoaderModelOnly", model=("2", 0), lora_name="sulphur_final.safetensors", strength_model=1.0),

    "4": node("LTXAPITextEncode", api_key="ltxv_GJNim3DP2LwmLgF2VdTV-YV5hX9pxoIZOnlKNKTVkU2MRlesiTcBs-881s5gQ5RYi06kmcV-oYZMIUJxa82loKetMpHEV9JieH-g0r6UFe7_kJzxpfJ8wouidUA_doQoaYGPOvyrw2Kl7LjNiEZ1BGtRzYRT7GAtIB8VaHKOmHb3", prompt=prompt_text, ckpt_name=CKPT, enhance_prompt=True),
    "5": node("LTXAPITextEncode", api_key="ltxv_GJNim3DP2LwmLgF2VdTV-YV5hX9pxoIZOnlKNKTVkU2MRlesiTcBs-881s5gQ5RYi06kmcV-oYZMIUJxa82loKetMpHEV9JieH-g0r6UFe7_kJzxpfJ8wouidUA_doQoaYGPOvyrw2Kl7LjNiEZ1BGtRzYRT7GAtIB8VaHKOmHb3", prompt="blurry, low quality, distorted, deformed, ugly", ckpt_name=CKPT, enhance_prompt=False),

    "6": node("EmptyLTXVLatentVideo", width=WIDTH, height=HEIGHT, length=NUM_FRAMES, batch_size=1),

    "7": node("LTXVAudioVAELoader", ckpt_name=CKPT),
    "8": node("VHS_LoadAudio", audio_file=AUDIO_FILE),
    "9": node("VAEEncodeAudio", audio=("8", 0), vae=("7", 0)),

    "10": node("LTXVConcatAVLatent", video_latent=("6", 0), audio_latent=("9", 0)),

    "11": node("ModelSamplingLTXV", model=("3", 0), max_shift=2.05, base_shift=0.95),

    "12": node("KSampler",
        model=("11", 0), positive=("4", 0), negative=("5", 0),
        latent_image=("10", 0),
        seed=random.randint(0, 2**63), steps=4, cfg=1.0,
        sampler_name="euler", scheduler="beta", denoise=1.0),

    "13": node("LTXVSeparateAVLatent", av_latent=("12", 0)),

    "14": node("VAELoader", vae_name="ltx-2.3-22b-distilled_video_vae.safetensors"),
    "15": node("VAEDecode", samples=("13", 0), vae=("14", 0)),

    "17": node("VHS_VideoCombine",
        images=("15", 0),
        frame_rate=FPS, loop_count=1, filename_prefix="ltx_mv_temp",
        format="video/h264-mp4", pingpong=False, save_output=True),
}

r = requests.post(API, json={"prompt": workflow})
print("Gen status:", r.status_code)
if r.status_code != 200:
    print(r.text[:500])
