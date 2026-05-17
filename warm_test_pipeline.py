import sys, os, time, secrets, gc, argparse, json
from PIL import Image

sys.path.insert(0, "/opt/qwen-image-studio/src")

os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"
import torch
from diffusers import DiffusionPipeline
from diffusers.pipelines.qwenimage import pipeline_qwenimage as _qimg
from qwen_image_mps import merge_lora_from_safetensors

LORA_PATH = (
    "/root/comfy-models/loras/Qwen-Image-2512-Lightning-4steps-V1.0-bf16.safetensors"
)
FP8_UNET_CANDIDATES = [
    "/root/comfy-models/diffusion_models/qwen-image-2512-fp8.safetensors",
    "/root/comfy-models/diffusion_models/qwen_image_2512_fp8_e4m3fn.safetensors",
]
MODEL_ID = "Qwen/Qwen-Image-2512"
OUTPUT_DIR = "/opt/ComfyUI/output"

GEN_WIDTH = 1024
GEN_HEIGHT = 1024
UPSCALE_TARGET = (1328, 1328)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--prompts-file", help="JSON file with prompts (overrides built-in defaults)"
)
parser.add_argument(
    "--steps", type=int, default=4, help="Inference steps (default 4, quality 20-30)"
)
parser.add_argument(
    "--cfg", type=float, default=1.0, help="CFG scale (default 1.0, quality 3.0-5.0)"
)
parser.add_argument(
    "--width",
    type=int,
    default=GEN_WIDTH,
    help=f"Generation width (default {GEN_WIDTH})",
)
parser.add_argument(
    "--height",
    type=int,
    default=GEN_HEIGHT,
    help=f"Generation height (default {GEN_HEIGHT})",
)
parser.add_argument(
    "--no-lora", action="store_true", help="Skip Lightning LoRA merge for quality"
)
args = parser.parse_args()

if args.prompts_file:
    with open(args.prompts_file) as f:
        data = json.load(f)
    PROMPTS = [(p["id"], p["prompt"]) for p in data["prompts"]]
    print(f"Loaded {len(PROMPTS)} prompts from {args.prompts_file}")
else:
    PROMPTS = [
        (
            "cold",
            "Cinematic close-up of a futuristic CPU on a high-end motherboard, intense electric sparks flowing between circuits, ultra-detailed microchips, dramatic cyberpunk lighting, high contrast, metallic textures with reflections, shallow depth of field, sci-fi technology photography, sharp focus, no text.",
        ),
        (
            "text1",
            'A clean, minimalist tech conference slide with a dark navy gradient background. At the top, the text "Qwen-Image-2512" in bold white sans-serif, with a subtle cyan glow. Below it, a timeline graphic with three numbered nodes connected by a horizontal glowing cyan line. Each node has a label: "Training", "Quantization", "Deployment". At the bottom right, small white text reads "unsloth.ai". No people, no noise, ultra-clean render.',
        ),
        (
            "text2",
            'A product label on a kraft paper background for a fictional craft soda. The brand name "FIZZ" is centered in bold retro serif font, dark brown, with a slight drop shadow. Below it in smaller caps: "ARTISAN SODA • EST. 2026". A simple illustration of a lemon half and mint leaves sits above the text. The label has a thin decorative border. Warm lighting, slightly desaturated, shallow depth of field, photorealistic.',
        ),
    ]

os.makedirs(OUTPUT_DIR, exist_ok=True)


def _rt_no_sigmas(scheduler, num_inference_steps=None, device=None, **kwargs):
    scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
    return scheduler.timesteps, len(scheduler.timesteps)


_qimg.retrieve_timesteps = _rt_no_sigmas

device = "cuda"
torch_dtype = torch.bfloat16

print(f"Loading {MODEL_ID}...")
torch.set_default_device(device)
pipe = DiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch_dtype,
    use_safetensors=True,
    device_map=None,
    low_cpu_mem_usage=True,
)
pipe.to(device=device, dtype=torch_dtype)
print("Pipeline loaded.")

fp8_path = next((p for p in FP8_UNET_CANDIDATES if os.path.exists(p)), None)
if fp8_path:
    print(f"Loading fp8 UNet weights from {fp8_path} (casting to bf16)...")
    from safetensors import safe_open

    with safe_open(fp8_path, framework="pt", device="cpu") as f:
        for name, param in pipe.transformer.named_parameters():
            key = f"model.diffusion_model.{name}"
            if key in f.keys():
                param.data = f.get_tensor(key).to(
                    device=param.device, dtype=torch.bfloat16
                )
    torch.cuda.empty_cache()
    print("fp8 UNet weights loaded.")
else:
    print("No fp8 UNet found, using bf16 model.")

try:
    if hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()
    print("VAE tiling enabled")
except Exception as e:
    print(f"VAE: {e}")

try:
    pipe.enable_attention_slicing()
    print("Attention slicing enabled")
except Exception:
    pass

MEM_GB = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU memory: {MEM_GB:.0f} GB")

if not args.no_lora:
    print(f"Loading Lightning LoRA from {LORA_PATH}...")
    pipe = merge_lora_from_safetensors(pipe, LORA_PATH)
    print("LoRA merged.")
else:
    print("Skipping LoRA (quality mode).")

gen_width = args.width
gen_height = args.height
steps = args.steps
cfg = args.cfg

print(
    f"Generating at {gen_width}x{gen_height}, {steps} steps, CFG={cfg}, LoRA={'no' if args.no_lora else 'yes'}"
)
print()

for label, prompt_text in PROMPTS:
    print(f"\n=== [{label}] {prompt_text[:60]}... ===")
    seed = secrets.randbits(63)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    t0 = time.time()
    output = pipe(
        prompt=prompt_text,
        negative_prompt="low quality, blurry, oversaturated, noise, artifacts, distorted",
        width=gen_width,
        height=gen_height,
        num_inference_steps=steps,
        true_cfg_scale=cfg,
        generator=generator,
    )
    image = output.images[0]
    del output
    gc.collect()
    torch.cuda.empty_cache()
    elapsed = time.time() - t0
    image = image.resize(UPSCALE_TARGET, Image.Resampling.LANCZOS)
    filename = f"{label}.png"
    path = os.path.join(OUTPUT_DIR, filename)
    image.save(path)
    print(
        f"  Done: {filename} ({elapsed:.1f}s), {gen_width}x{gen_height} -> {UPSCALE_TARGET[0]}x{UPSCALE_TARGET[1]}, {steps} steps, CFG={cfg}"
    )

print(f"\n=== All {len(PROMPTS)} prompts completed ===")
