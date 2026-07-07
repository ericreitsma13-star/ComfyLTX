import json, requests, random, sys

API = "http://127.0.0.1:8188/prompt"

prompt_text = sys.argv[1] if len(sys.argv) > 1 else "police car chasing a red sports car on a highway at night, speeding, dynamic camera, motion blur, cinematic lighting, fast action, realistic"

WIDTH = 512
HEIGHT = 512
NUM_FRAMES = 81

def node(class_type, **inputs):
    return {"class_type": class_type, "inputs": inputs}

workflow = {
    "1": node("UnetLoaderGGUF",
        unet_name=GGUF_MODEL),

    "2": node("CLIPLoaderGGUF",
        clip_name=T5_GGUF,
        type="wan"),

    "3": node("CLIPTextEncode",
        text=prompt_text,
        clip=("2", 0)),

    "4": node("CLIPTextEncode",
        text="slow motion, static camera, blurry, low quality, video game graphics, CGI, unnatural movement",
        clip=("2", 0)),

    "5": node("ModelSamplingSD3",
        model=("1", 0),
        shift=8.0),

    "6": node("EmptyHunyuanLatentVideo",
        width=WIDTH,
        height=HEIGHT,
        length=NUM_FRAMES,
        batch_size=1),

    "7": node("VAELoader",
        vae_name=VAE_NAME),

    "8": node("KSampler",
        model=("5", 0),
        positive=("3", 0),
        negative=("4", 0),
        latent_image=("6", 0),
        seed=random.randint(0, 2**63),
        steps=4,
        cfg=1.5,
        sampler_name="euler",
        scheduler="beta",
        denoise=1.0),

    "9": node("VAEDecode",
        samples=("8", 0),
        vae=("7", 0)),

    "10": node("CreateVideo",
        images=("9", 0),
        fps=24),

    "11": node("SaveVideo",
        video=("10", 0),
        filename_prefix="wan22_gguf",
        format="auto",
        codec="auto"),
}

r = requests.post(API, json={"prompt": workflow})
print(r.status_code, r.text[:500])
