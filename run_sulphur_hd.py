import json, requests, random, sys

API = "http://127.0.0.1:8188/prompt"
API_KEY = "ltxv_GJNim3DP2LwmLgF2VdTV-YV5hX9pxoIZOnlKNKTVkU2MRlesiTcBs-881s5gQ5RYi06kmcV-oYZMIUJxa82loKetMpHEV9JieH-g0r6UFe7_kJzxpfJ8wouidUA_doQoaYGPOvyrw2Kl7LjNiEZ1BGtRzYRT7GAtIB8VaHKOmHb3"

prompt_text = sys.argv[1] if len(sys.argv) > 1 else "police car chasing a red sports car on a wet city highway at night, high speed pursuit, dynamic camera, cinematic motion blur, realistic, Hollywood action scene"

def node(class_type, **inputs):
    return {"class_type": class_type, "inputs": inputs}

prompt = {
    "10": node("UNETLoader",
        unet_name="ltx-2.3-22b-distilled-1.1.safetensors",
        weight_dtype="default"),

    "11": node("LoraLoaderModelOnly",
        model=("10", 0),
        lora_name="ltx-2.3-22b-distilled-lora-384-1.1.safetensors",
        strength_model=0.5),

    "12": node("LoraLoaderModelOnly",
        model=("11", 0),
        lora_name="sulphur_final.safetensors",
        strength_model=1.0),

    "40": node("LTXAPITextEncode",
        api_key=API_KEY,
        prompt=prompt_text,
        ckpt_name="ltx-2.3-22b-distilled-1.1.safetensors",
        enhance_prompt=True),

    "41": node("LTXAPITextEncode",
        api_key=API_KEY,
        prompt="blurry, low quality, distorted, deformed, ugly, bad anatomy, watermark, text, logo",
        ckpt_name="ltx-2.3-22b-distilled-1.1.safetensors",
        enhance_prompt=False),

    "42": node("EmptyLTXVLatentVideo",
        width=960,
        height=544,
        length=241,
        batch_size=1),

    "48": node("VAELoader",
        vae_name="ltx-2.3-22b-distilled_video_vae.safetensors"),

    "43": node("ModelSamplingLTXV",
        model=("12", 0),
        max_shift=2.05,
        base_shift=0.95),

    "44": node("KSampler",
        model=("43", 0),
        positive=("40", 0),
        negative=("41", 0),
        latent_image=("42", 0),
        seed=random.randint(0, 2**63),
        steps=6,
        cfg=1.0,
        sampler_name="euler",
        scheduler="beta",
        denoise=1.0),

    "50": node("ModelSamplingLTXV",
        model=("12", 0),
        max_shift=2.05,
        base_shift=0.95),

    "51": node("LatentUpscale",
        samples=("44", 0),
        upscale_method="bilinear",
        width=1440,
        height=816,
        crop="disabled"),

    "52": node("KSampler",
        model=("50", 0),
        positive=("40", 0),
        negative=("41", 0),
        latent_image=("51", 0),
        seed=random.randint(0, 2**63),
        steps=6,
        cfg=1.0,
        sampler_name="euler",
        scheduler="beta",
        denoise=0.25),

    "45": node("VAEDecode",
        samples=("52", 0),
        vae=("48", 0)),

    "46": node("CreateVideo",
        images=("45", 0),
        fps=24),

    "47": node("SaveVideo",
        video=("46", 0),
        filename_prefix="sulphur_hd",
        format="auto",
        codec="auto"),
}

r = requests.post(API, json={"prompt": prompt})
print(r.status_code, r.text[:500])
