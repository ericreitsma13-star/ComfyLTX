import json, requests, random, sys

API = "http://127.0.0.1:8188/prompt"
API_KEY = "ltxv_GJNim3DP2LwmLgF2VdTV-YV5hX9pxoIZOnlKNKTVkU2MRlesiTcBs-881s5gQ5RYi06kmcV-oYZMIUJxa82loKetMpHEV9JieH-g0r6UFe7_kJzxpfJ8wouidUA_doQoaYGPOvyrw2Kl7LjNiEZ1BGtRzYRT7GAtIB8VaHKOmHb3"

prompt_text = sys.argv[1] if len(sys.argv) > 1 else "police car chasing a red sports car on a wet city highway at night, high speed pursuit, dynamic camera, cinematic motion blur, realistic, Hollywood action scene"
CKPT = "ltx-2.3-22b-distilled-1.1.safetensors"
WIDTH, HEIGHT, FPS = 960, 544, 24
NUM_FRAMES = 241

def node(class_type, **inputs):
    return {"class_type": class_type, "inputs": inputs}

workflow = {
    "1": node("UNETLoader", unet_name=CKPT, weight_dtype="default"),
    "2": node("LoraLoaderModelOnly", model=("1", 0), lora_name="ltx-2.3-22b-distilled-lora-384-1.1.safetensors", strength_model=0.5),
    "3": node("LoraLoaderModelOnly", model=("2", 0), lora_name="sulphur_final.safetensors", strength_model=1.0),

    "4": node("LTXAPITextEncode", api_key=API_KEY, prompt=prompt_text, ckpt_name=CKPT, enhance_prompt=True),
    "5": node("LTXAPITextEncode", api_key=API_KEY, prompt="blurry, low quality, distorted, deformed, ugly", ckpt_name=CKPT, enhance_prompt=False),

    "6": node("EmptyLTXVLatentVideo", width=WIDTH, height=HEIGHT, length=NUM_FRAMES, batch_size=1),

    "13": node("LTXVAudioVAELoader", ckpt_name=CKPT),
    "7": node("LTXVEmptyLatentAudio", frames_number=NUM_FRAMES, frame_rate=FPS, batch_size=1, audio_vae=("13", 0)),

    "8": node("LTXVConcatAVLatent", video_latent=("6", 0), audio_latent=("7", 0)),

    "9": node("ModelSamplingLTXV", model=("3", 0), max_shift=2.05, base_shift=0.95),

    "10": node("KSampler",
        model=("9", 0), positive=("4", 0), negative=("5", 0),
        latent_image=("8", 0),
        seed=random.randint(0, 2**63), steps=4, cfg=1.0,
        sampler_name="euler", scheduler="beta", denoise=1.0),

    "11": node("LTXVSeparateAVLatent", av_latent=("10", 0)),

    "12": node("VAELoader", vae_name="ltx-2.3-22b-distilled_video_vae.safetensors"),
    "14": node("VAEDecode", samples=("11", 0), vae=("12", 0)),

    "15": node("VAEDecodeAudio", samples=("11", 1), vae=("13", 0)),

    "16": node("VHS_VideoCombine",
        images=("14", 0), audio=("15", 0),
        frame_rate=FPS, loop_count=1, filename_prefix="ltx_av",
        format="video/h264-mp4", pingpong=False, save_output=True),
}

r = requests.post(API, json={"prompt": workflow})
print(r.status_code, r.text[:500])
