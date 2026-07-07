import json, requests, random
import torch

API = "http://127.0.0.1:8188/prompt"

def node(class_type, **inputs):
    return {"class_type": class_type, "inputs": inputs}

# Use KSampler with noise on random latents (not zero)
prompt = {
    "75": node("UNETLoader",
        unet_name="Wan2_2-T2V-A14B_HIGH_fp8_e4m3fn_scaled_KJ.safetensors",
        weight_dtype="default"),

    "71": node("CLIPLoader",
        clip_name="umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        type="wan"),

    "73": node("VAELoader",
        vae_name="wan_2.1_vae.safetensors"),

    "89": node("CLIPTextEncode",
        clip=("71", 0),
        text="Cinematic aerial shot of a misty mountain landscape at sunrise"),

    "72": node("CLIPTextEncode",
        clip=("71", 0),
        text=""),

    # Empty latent with batch_size
    "74": node("EmptyHunyuanLatentVideo",
        width=512,
        height=512,
        length=33,
        batch_size=1),

    "82": node("ModelSamplingSD3",
        model=("75", 0),
        shift=5.0),

    # Single KSampler with noise enabled
    "81": node("KSamplerAdvanced",
        model=("82", 0),
        positive=("89", 0),
        negative=("72", 0),
        latent_image=("74", 0),
        add_noise="enable",
        noise_seed=random.randint(0, 2**63),
        steps=20,
        cfg=5.0,
        sampler_name="euler",
        scheduler="simple",
        start_at_step=0,
        end_at_step=20,
        return_with_leftover_noise="disable"),

    "87": node("VAEDecode",
        samples=("81", 0),
        vae=("73", 0)),

    "88": node("CreateVideo",
        images=("87", 0),
        fps=16),

    "80": node("SaveVideo",
        video=("88", 0),
        filename_prefix="wan22_minimal",
        format="auto",
        codec="auto"),
}

r = requests.post(API, json={"prompt": prompt})
print(r.status_code, r.text[:500])
