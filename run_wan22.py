import json, requests, random

API = "http://127.0.0.1:8188/prompt"

def node(class_type, **inputs):
    return {"class_type": class_type, "inputs": inputs}

prompt = {
    # 1. Load high-noise model
    "75": node("UNETLoader",
        unet_name="Wan2_2-T2V-A14B_HIGH_fp8_e4m3fn_scaled_KJ.safetensors",
        weight_dtype="default"),

    # 2. Load low-noise model
    "76": node("UNETLoader",
        unet_name="Wan2_2-T2V-A14B-LOW_fp8_e4m3fn_scaled_KJ.safetensors",
        weight_dtype="default"),

    # 3. Load CLIP (umT5)
    "71": node("CLIPLoader",
        clip_name="umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        type="wan"),

    # 4. Load VAE
    "73": node("VAELoader",
        vae_name="wan_2.1_vae.safetensors"),

    # 5. Positive prompt encode
    "89": node("CLIPTextEncode",
        clip=("71", 0),
        text="Cinematic aerial shot of a misty mountain landscape at sunrise, golden light piercing through clouds, majestic peaks, smooth camera movement, photorealistic, ultra detailed, 8K"),

    # 6. Negative prompt encode
    "72": node("CLIPTextEncode",
        clip=("71", 0),
        text="blurry, low quality, distorted, deformed, ugly, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, watermark, text, logo, signature"),

    # 7. Empty latent video
    "74": node("EmptyHunyuanLatentVideo",
        width=512,
        height=512,
        length=81,
        batch_size=1),

    # 8. ModelSamplingSD3 for high-noise model (shift)
    "82": node("ModelSamplingSD3",
        model=("75", 0),
        shift=5.0),

    # 9. ModelSamplingSD3 for low-noise model (shift)
    "86": node("ModelSamplingSD3",
        model=("76", 0),
        shift=5.0),

    # 10. First KSampler (high-noise, steps 0→10)
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
        end_at_step=10,
        return_with_leftover_noise="enable"),

    # 11. Second KSampler (low-noise, steps 10→20)
    "78": node("KSamplerAdvanced",
        model=("86", 0),
        positive=("89", 0),
        negative=("72", 0),
        latent_image=("81", 0),
        add_noise="disable",
        noise_seed=0,
        steps=20,
        cfg=5.0,
        sampler_name="euler",
        scheduler="simple",
        start_at_step=10,
        end_at_step=20,
        return_with_leftover_noise="disable"),




    # 12. VAE Decode
    "87": node("VAEDecode",
        samples=("78", 0),
        vae=("73", 0)),

    # 13. Create video
    "88": node("CreateVideo",
        images=("87", 0),
        fps=16),

    # 14. Save video
    "80": node("SaveVideo",
        video=("88", 0),
        filename_prefix="wan22_t2v",
        format="auto",
        codec="auto"),
}

r = requests.post(API, json={"prompt": prompt})
print(r.status_code, r.text[:500])
