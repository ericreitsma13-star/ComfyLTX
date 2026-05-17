import sys, os, time, secrets, gc, argparse, json, traceback
from PIL import Image

os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"
import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
LORA_DIR = "/root/comfy-models/loras"
VAE_PATH = "/root/comfy-models/vae/sdxl_vae_fp16.safetensors"
OUTPUT_DIR = "/opt/ComfyUI/output/sdxl"
PROMPTS_DIR = "/opt/comfy-toolbox/content_packs"

parser = argparse.ArgumentParser()
parser.add_argument("--pack", help="Content pack name (subdir in content_packs/)")
parser.add_argument("--prompt", help="Single prompt override")
parser.add_argument("--steps", type=int, default=25)
parser.add_argument("--cfg", type=float, default=7.0)
parser.add_argument("--width", type=int, default=1024)
parser.add_argument("--height", type=int, default=1024)
args = parser.parse_args()

device = "cuda"
torch_dtype = torch.bfloat16

print(f"Loading SDXL from HF hub ({MODEL_ID})...")
sys.stdout.flush()
vae = None
if os.path.exists(VAE_PATH):
    print(f"Loading fp16 VAE: {VAE_PATH}")
    vae = AutoencoderKL.from_single_file(VAE_PATH, torch_dtype=torch_dtype)
pipe = StableDiffusionXLPipeline.from_pretrained(
    MODEL_ID,
    vae=vae,
    torch_dtype=torch_dtype,
    variant="fp16",
    use_safetensors=True,
)
pipe.to(device=device)
pipe.enable_attention_slicing()
pipe.vae.enable_tiling()

MEM_GB = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU memory: {MEM_GB:.0f} GB")
sys.stdout.flush()

prompts = []
gen_width, gen_height = args.width, args.height
steps, cfg = args.steps, args.cfg

if args.pack:
    pack_path = os.path.join(PROMPTS_DIR, args.pack, "pack.json")
    if not os.path.exists(pack_path):
        print(f"Pack not found: {pack_path}")
        sys.exit(1)
    with open(pack_path) as f:
        pack = json.load(f)
    print(f"Loaded pack: {pack['name']} v{pack['version']}")
    sys.stdout.flush()
    for lora_ref in pack.get("loras", []):
        lora_file = os.path.join(LORA_DIR, lora_ref["file"])
        if os.path.exists(lora_file):
            print(
                f"Loading LoRA: {lora_ref['file']} (weight={lora_ref.get('weight', 1.0)})"
            )
            pipe.load_lora_weights(lora_file, adapter_name=lora_ref.get("name", "pack"))
            sys.stdout.flush()
    prompts = pack.get("prompts", [])
    settings = pack.get("settings", {})
    gen_width = settings.get("width", gen_width)
    gen_height = settings.get("height", gen_height)
    steps = settings.get("steps", steps)
    cfg = settings.get("cfg", cfg)

if args.prompt:
    prompts.append(args.prompt)

if not prompts:
    prompts = [
        "A serene mountain landscape at sunset, cinematic lighting, highly detailed, 8k"
    ]

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(
    f"Generating {len(prompts)} image(s) at {gen_width}x{gen_height}, {steps} steps, CFG={cfg}"
)
sys.stdout.flush()

for idx, prompt_text in enumerate(prompts):
    print(f"\n[{idx + 1}/{len(prompts)}] {prompt_text[:80]}...")
    sys.stdout.flush()
    seed = secrets.randbits(63)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    try:
        t0 = time.time()
        result = pipe(
            prompt=prompt_text,
            negative_prompt="low quality, blurry, distorted, ugly, bad anatomy",
            width=gen_width,
            height=gen_height,
            num_inference_steps=steps,
            guidance_scale=cfg,
            generator=generator,
        )
        image = result.images[0]
        elapsed = time.time() - t0
        filename = f"sdxl_{idx:04d}.png"
        path = os.path.join(OUTPUT_DIR, filename)
        image.save(path)
        print(f"  Saved: {filename} ({elapsed:.1f}s)")
    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
    sys.stdout.flush()
    gc.collect()
    torch.cuda.empty_cache()

print(f"\nDone. {len(prompts)} image(s) in {OUTPUT_DIR}")
sys.stdout.flush()
