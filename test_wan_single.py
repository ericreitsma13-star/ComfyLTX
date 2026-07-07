import json, requests, random

API = "http://127.0.0.1:8188/prompt"

def node(class_type, **inputs):
    return {"class_type": class_type, "inputs": inputs}

# Test 1: Single high-noise model, 4 steps (turbov style)
prompt = {
    "1": node("WanVideoModelLoader",
        model="wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors",
        base_precision="fp16_fast",
        quantization="fp8_e4m3fn_scaled",
        load_device="offload_device",
        attention_mode="sdpa"),

    "2": node("WanVideoVAELoader",
        model_name="wan_2.1_vae.safetensors",
        precision="bf16"),

    "3": node("CLIPLoader",
        clip_name="umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        type="wan",
        device="default"),

    "4": node("CLIPTextEncode",
        clip=("3", 0),
        text="Cinematic aerial shot of a misty mountain landscape at sunrise, golden light piercing through clouds, majestic peaks, smooth camera movement, photorealistic, ultra detailed, 8K"),

    "5": node("CLIPTextEncode",
        clip=("3", 0),
        text="blurry, low quality, distorted"),

    "6": node("WanVideoTextEmbedBridge",
        positive=("4", 0),
        negative=("5", 0)),

    "7": node("WanVideoEmptyEmbeds",
        width=256,
        height=256,
        num_frames=16),

    # Single sampler (no two-stage split)
    "8": node("WanVideoSampler",
        model=("1", 0),
        image_embeds=("7", 0),
        text_embeds=("6", 0),
        steps=4,
        cfg=5.0,
        shift=5.0,
        seed=random.randint(0, 2**63),
        force_offload=True,
        scheduler="euler",
        riflex_freq_index=0),

    "9": node("WanVideoDecode",
        vae=("2", 0),
        samples=("8", 0),
        enable_vae_tiling=False,
        tile_x=272,
        tile_y=272,
        tile_stride_x=144,
        tile_stride_y=128),

    "10": node("CreateVideo",
        images=("9", 0),
        fps=16),

    "11": node("SaveVideo",
        video=("10", 0),
        filename_prefix="wan_test_single",
        format="auto",
        codec="auto"),
}

r = requests.post(API, json={"prompt": prompt})
print(r.status_code, r.text[:500])
