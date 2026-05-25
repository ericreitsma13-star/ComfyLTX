import json, requests, random, sys

API = "http://127.0.0.1:8188/prompt"

prompt_text = sys.argv[1] if len(sys.argv) > 1 else "sports car speeding through a neon-lit cyberpunk city at night, dynamic camera following alongside, fast motion, cinematic motion blur, rain on windshield, city lights reflecting, high energy action scene, Blade Runner 2049 aesthetic"

HIGH_MODEL = "Wan2_2-T2V-A14B_HIGH_fp8_e4m3fn_scaled_KJ.safetensors"
LOW_MODEL = "Wan2_2-T2V-A14B-LOW_fp8_e4m3fn_scaled_KJ.safetensors"
T5_GGUF = "umt5-xxl-encoder-Q3_K_M.gguf"
VAE_NAME = "wan_2.1_vae.safetensors"

WIDTH = 512
HEIGHT = 512
NUM_FRAMES = 25

def node(class_type, **inputs):
    return {"class_type": class_type, "inputs": inputs}

workflow = {
    "1": node("UNETLoader",
        unet_name=HIGH_MODEL,
        weight_dtype="default"),

    "2": node("CLIPLoaderGGUF",
        clip_name=T5_GGUF,
        type="wan"),

    "3": node("CLIPTextEncode",
        text=prompt_text,
        clip=("2", 0)),

    "4": node("CLIPTextEncode",
        text="slow motion, static camera, blurry, low quality, jittery",
        clip=("2", 0)),

    "5": node("ModelSamplingSD3",
        model=("1", 0),
        shift=5.0),

    "6": node("EmptyHunyuanLatentVideo",
        width=WIDTH,
        height=HEIGHT,
        length=NUM_FRAMES,
        batch_size=1),

    "7": node("VAELoader",
        vae_name=VAE_NAME),

    "8": node("KSamplerAdvanced",
        model=("5", 0),
        add_noise="enable",
        noise_seed=random.randint(0, 2**63),
        steps=20,
        cfg=5.0,
        sampler_name="euler",
        scheduler="simple",
        start_at_step=0,
        end_at_step=10,
        return_with_leftover_noise="enable",
        positive=("3", 0),
        negative=("4", 0),
        latent_image=("6", 0)),

    "9": node("UNETLoader",
        unet_name=LOW_MODEL,
        weight_dtype="default"),

    "10": node("ModelSamplingSD3",
        model=("9", 0),
        shift=5.0),

    "11": node("KSamplerAdvanced",
        model=("10", 0),
        add_noise="disable",
        noise_seed=random.randint(0, 2**63),
        steps=20,
        cfg=5.0,
        sampler_name="euler",
        scheduler="simple",
        start_at_step=10,
        end_at_step=20,
        return_with_leftover_noise="disable",
        positive=("3", 0),
        negative=("4", 0),
        latent_image=("8", 0)),

    "12": node("VAEDecode",
        samples=("11", 0),
        vae=("7", 0)),

    "13": node("CreateVideo",
        images=("12", 0),
        fps=24),

    "14": node("SaveVideo",
        video=("13", 0),
        filename_prefix="wan22_native",
        format="auto",
        codec="auto"),
}

r = requests.post(API, json={"prompt": workflow})
print(r.status_code, r.text[:500])
