import sys, os, time, secrets, gc, argparse
from PIL import Image

os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"
import torch
from diffusers import LTX2Pipeline
from diffusers.utils import export_to_video
from transformers import Gemma3ForConditionalGeneration

DISTILLED_FP8_PATH = (
    "/root/comfy-models/diffusion_models/ltx-2-19b-distilled-fp8.safetensors"
)
OUTPUT_DIR = "/opt/ComfyUI/output"
MODEL_ID = "Lightricks/LTX-2"

def parse_args():
    parser = argparse.ArgumentParser(description="LTX-2 low-pressure Strix Halo test")
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt to run")
    parser.add_argument("--width", type=int, default=576, help="Output width (multiple of 64)")
    parser.add_argument("--height", type=int, default=320, help="Output height (multiple of 64)")
    parser.add_argument("--frames", type=int, default=17, help="Frame count (8k+1)")
    parser.add_argument("--fps", type=int, default=12, help="Output FPS")
    parser.add_argument("--steps", type=int, default=8, help="Inference steps")
    parser.add_argument("--cfg", type=float, default=1.0, help="Guidance scale")
    parser.add_argument("--count", type=int, default=1, help="Number of clips to generate")
    return parser.parse_args()


def validate_args(args):
    if args.width % 64 != 0 or args.height % 64 != 0:
        raise ValueError(f"Width/height must be multiples of 64. Got {args.width}x{args.height}")
    if args.frames < 1 or (args.frames - 1) % 8 != 0:
        raise ValueError(f"Frames must follow 8k+1. Got {args.frames}")
    if args.count < 1:
        raise ValueError("--count must be >= 1")


args = parse_args()
validate_args(args)
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = "cuda"
torch_dtype = torch.bfloat16
try:
    # Some launch environments set the global default device to CUDA, which
    # makes huge module construction allocate directly on VRAM and OOM early.
    torch.set_default_device("cpu")
except Exception:
    pass

MEM_GB = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU memory: {MEM_GB:.0f} GB")
if MEM_GB < 40:
    print("WARNING: <40GB available, will likely OOM on 19B model")

print("Loading LTX-2 pipeline configs from HF...")
# Do not set default device to CUDA here.
# On Strix Halo this causes huge model init allocations on GPU before
# low-memory controls (tiling/slicing/offload) can take effect.

text_encoder = None
q4_path = "google/gemma-3-12b-it-qat-q4_0-unquantized"
try:
    print(f"Loading Q4 Gemma3 text encoder ({q4_path})...")
    text_encoder = Gemma3ForConditionalGeneration.from_pretrained(
        q4_path,
        torch_dtype=torch_dtype,
    )
    print("Q4 Gemma3 text encoder loaded.")
except Exception as e:
    print(f"Q4 Gemma3 not available: {e}")

transformer = None
if os.path.exists(DISTILLED_FP8_PATH):
    print(f"Loading distilled fp8 transformer from {DISTILLED_FP8_PATH}...")
    from safetensors import safe_open
    from diffusers import LTX2VideoTransformer3DModel

    config = LTX2VideoTransformer3DModel.load_config(MODEL_ID, subfolder="transformer")
    transformer = LTX2VideoTransformer3DModel.from_config(config)
    transformer.to(dtype=torch_dtype, device="cpu")

    with safe_open(DISTILLED_FP8_PATH, framework="pt", device="cpu") as f:
        for name, param in transformer.named_parameters():
            if name in f.keys():
                param.data = f.get_tensor(name).to(device="cpu", dtype=torch_dtype)
    del f
    torch.cuda.empty_cache()
    print("Distilled fp8 transformer loaded.")

# LTX2Pipeline currently expects torch_dtype (dtype is ignored on this class).
pipe_kwargs = {"torch_dtype": torch_dtype}
if transformer is not None:
    pipe_kwargs["transformer"] = transformer
    del transformer
if text_encoder is not None:
    pipe_kwargs["text_encoder"] = text_encoder

pipe = LTX2Pipeline.from_pretrained(MODEL_ID, **pipe_kwargs)
# Ensure connector/linear layers match prompt embed dtype before offload.
pipe.to(dtype=torch_dtype)
if hasattr(pipe.vae, "enable_tiling"):
    pipe.vae.enable_tiling()
    print("VAE tiling enabled")

try:
    pipe.enable_attention_slicing()
    print("Attention slicing enabled")
except Exception:
    pass

offload_enabled = False
try:
    pipe.enable_model_cpu_offload()
    offload_enabled = True
    print("Model CPU offload enabled")
except Exception:
    pass

if not offload_enabled:
    pipe.to(device=device)
    print(f"Pipeline moved to {device} (CPU offload unavailable)")

print("Pipeline ready.")

base_prompt = (
    args.prompt
    or "A serene Japanese garden at sunset with cherry blossom petals falling, koi fish in a pond, wooden bridge, soft golden lighting, cinematic quality."
)
prompts = [(f"test{i+1}", base_prompt) for i in range(args.count)]

for label, prompt in prompts:
    print(f"\n=== [{label}] {prompt[:60]}... ===")
    seed = secrets.randbits(63)
    generator = torch.Generator(device="cpu").manual_seed(seed)

    t0 = time.time()
    output = pipe(
        prompt=prompt,
        negative_prompt="low quality, blurry, distorted",
        width=args.width,
        height=args.height,
        num_frames=args.frames,
        frame_rate=args.fps,
        num_inference_steps=args.steps,
        guidance_scale=args.cfg,
        generator=generator,
    )
    elapsed = time.time() - t0

    video_path = os.path.join(OUTPUT_DIR, f"{label}.mp4")
    export_to_video(output.frames[0], video_path, fps=args.fps)
    print(f"  Done: {video_path} ({elapsed:.1f}s)")

    del output
    gc.collect()
    torch.cuda.empty_cache()

print("\nDone.")
