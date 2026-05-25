import json, requests, random, sys

API = "http://127.0.0.1:8188/prompt"

import requests as http_requests

API_KEY = "ltxv_GJNim3DP2LwmLgF2VdTV-YV5hX9pxoIZOnlKNKTVkU2MRlesiTcBs-881s5gQ5RYi06kmcV-oYZMIUJxa82loKetMpHEV9JieH-g0r6UFe7_kJzxpfJ8wouidUA_doQoaYGPOvyrw2Kl7LjNiEZ1BGtRzYRT7GAtIB8VaHKOmHb3"

prompt_text = sys.argv[1] if len(sys.argv) > 1 else "a flower bud slowly blooming into a beautiful rose, petals unfurling in time-lapse, dewdrops sparkling, soft morning sunlight, cinematic macro shot, 8K"

WIDTH = 512
HEIGHT = 512
NUM_FRAMES = 81  # ~3.4s at 24fps
STEPS = 20
HIGH_MODEL = "Wan2_2-T2V-A14B_HIGH_fp8_e4m3fn_scaled_KJ.safetensors"
LOW_MODEL = "Wan2_2-T2V-A14B-LOW_fp8_e4m3fn_scaled_KJ.safetensors"
VAE_NAME = "wan_2.1_vae.safetensors"
T5_MODEL = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"

def node(class_type, **inputs):
    return {"class_type": class_type, "inputs": inputs}

workflow = {
    "1": node("LoadWanVideoT5TextEncoder",
        model_name=T5_MODEL,
        precision="bf16",
        load_device="offload_device",
        quantization="disabled"),

    "2": node("WanVideoTextEncode",
        positive_prompt=prompt_text,
        negative_prompt="",
        t5=("1", 0),
        force_offload=True,
        device="gpu"),

    "3": node("WanVideoVAELoader",
        model_name=VAE_NAME,
        precision="bf16"),

    "4": node("WanVideoEmptyEmbeds",
        width=WIDTH,
        height=HEIGHT,
        num_frames=NUM_FRAMES),

    "10": node("WanVideoBlockSwap",
        blocks_to_swap=20,
        offload_img_emb=True,
        offload_txt_emb=True,
        use_non_blocking=True,
        block_swap_debug=False),

    "5": node("WanVideoModelLoader",
        model=HIGH_MODEL,
        base_precision="bf16",
        quantization="disabled",
        load_device="offload_device",
        block_swap_args=("10", 0)),

    "6": node("WanVideoSampler",
        model=("5", 0),
        image_embeds=("4", 0),
        steps=STEPS,
        cfg=5.0,
        shift=5.0,
        seed=random.randint(0, 2**63),
        force_offload=True,
        scheduler="unipc",
        riflex_freq_index=0,
        text_embeds=("2", 0)),

    "7": node("WanVideoDecode",
        vae=("3", 0),
        samples=("6", 0),
        enable_vae_tiling=True,
        tile_x=WIDTH,
        tile_y=HEIGHT,
        tile_stride_x=WIDTH // 2,
        tile_stride_y=HEIGHT // 2),

    "8": node("CreateVideo",
        images=("7", 0),
        fps=24),

    "9": node("SaveVideo",
        video=("8", 0),
        filename_prefix="wan22_kijai",
        format="auto",
        codec="auto"),
}

r = requests.post(API, json={"prompt": workflow})
print(r.status_code, r.text[:500])
