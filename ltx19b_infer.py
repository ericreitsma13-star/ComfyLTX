import sys, os, time, gc, argparse, secrets
from datetime import datetime

os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"
os.environ["HIP_VISIBLE_DEVICES"] = "0"

import torch
from diffusers import LTXPipeline
from diffusers.utils import export_to_video

DISTILLED_FP8_PATH = (
    "/root/comfy-models/diffusion_models/ltx-2-19b-distilled-fp8.safetensors"
)
OUTPUT_DIR = "/opt/ComfyUI/output"
MODEL_ID = "Lightricks/LTX-2"


def parse_args():
    parser = argparse.ArgumentParser(description="LTX-2 19B inference")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--frames", type=int, default=49)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()

    prompt = (
        args.prompt
        or "A serene Japanese garden at sunset with cherry blossom petals falling, koi fish swimming in a pond, a wooden bridge, soft golden lighting, cinematic quality."
    )
    negative_prompt = args.negative_prompt or "low quality, blurry, distorted, ugly"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = "cuda"
    torch_dtype = torch.bfloat16

    MEM_GB = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU memory: {MEM_GB:.0f} GB")

    print("Loading LTX-2 pipeline configs from HF...")
    torch.set_default_device(device)

    transformer = None
    if os.path.exists(DISTILLED_FP8_PATH):
        print(f"Loading distilled fp8 transformer from {DISTILLED_FP8_PATH}...")
        from safetensors import safe_open
        from diffusers import LTX2VideoTransformer3DModel

        config = LTX2VideoTransformer3DModel.load_config(
            MODEL_ID, subfolder="transformer"
        )
        transformer = LTX2VideoTransformer3DModel.from_config(config)
        transformer.to(dtype=torch_dtype)

        with safe_open(DISTILLED_FP8_PATH, framework="pt", device="cpu") as f:
            for name, param in transformer.named_parameters():
                if name in f.keys():
                    param.data = f.get_tensor(name).to(device=param.device)
        del f
        torch.cuda.empty_cache()
        print("Distilled fp8 transformer loaded.")

    pipe_kwargs = {"torch_dtype": torch_dtype}
    if transformer is not None:
        pipe_kwargs["transformer"] = transformer

    pipe = LTXPipeline.from_pretrained(MODEL_ID, **pipe_kwargs)
    pipe.to(device=device)
    if hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass
    print("Pipeline ready.")

    seed = args.seed
    generator = torch.Generator(device="cpu").manual_seed(seed)

    print(f"Generating {args.frames} frames at {args.width}x{args.height}...")
    t0 = time.time()

    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=args.width,
        height=args.height,
        num_frames=args.frames,
        frame_rate=args.fps,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
    )

    elapsed = time.time() - t0
    print(f"Generation done in {elapsed:.1f}s")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = args.output or f"ltx19b_{ts}.mp4"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    export_to_video(output.frames[0], output_path, fps=args.fps)
    print(f"Output: {output_path}")

    del output
    gc.collect()
    torch.cuda.empty_cache()
    print("Done.")


if __name__ == "__main__":
    main()
