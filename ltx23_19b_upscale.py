"""
Stage 2: Spatial upscale.
Loads the video encoder + spatial upscaler (~1 GB), upscales the half-res video
latent 2x, saves to disk, then exits. Audio latent passes through unchanged.
"""
import gc, json, logging, os, sys, time

os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"
os.environ["HIP_VISIBLE_DEVICES"] = "0"
sys.path.insert(0, "/opt/LTX-2/packages/ltx-core/src")
sys.path.insert(0, "/opt/LTX-2/packages/ltx-pipelines/src")

import torch
from ltx_pipelines.utils.blocks import VideoUpsampler
from safetensors.torch import load_file, save_file


def load_tensor(path: str) -> torch.Tensor:
    """Load a single tensor from a safetensors file."""
    return load_file(path)["data"]


def main():
    import argparse

    parser = argparse.ArgumentParser(description="LTX-2.3 19B Upscale (spatial x2)")
    parser.add_argument("--checkpoint", default="/root/comfy-models/diffusion_models/ltx-2-19b-distilled-fp8.safetensors")
    parser.add_argument("--upsampler-path", default="/root/comfy-models/diffusion_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors")
    parser.add_argument("--temp-dir", default="/tmp/ltx23_19b_split")
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Load half-res latent from disk
    print("[upscale] Loading half-res video latent...")
    video_latent = load_tensor(os.path.join(args.temp_dir, "video_latent_stage1.safetensors"))
    video_latent = video_latent.to(device=device, dtype=dtype)
    print(f"[upscale] video_latent shape: {video_latent.shape}")

    # Build upscaler and run
    print("[upscale] Building VideoUpsampler...")
    upsampler = VideoUpsampler(
        checkpoint_path=args.checkpoint,
        upsampler_path=args.upsampler_path,
        dtype=dtype,
        device=device,
    )

    print("[upscale] Upscaling video latent (x2)...")
    t0 = time.time()
    upscaled_video_latent = upsampler(video_latent[:1])
    elapsed = time.time() - t0
    print(f"[upscale] Upscale done in {elapsed:.1f}s")
    print(f"[upscale] upscaled shape: {upscaled_video_latent.shape}")

    # Clean up upscaler
    del upsampler, video_latent
    torch.cuda.empty_cache()
    gc.collect()

    # Save upscaled latent
    save_file({"data": upscaled_video_latent.cpu()}, os.path.join(args.temp_dir, "video_latent_upscaled.safetensors"))
    print(f"[upscale] Saved upscaled latent to {args.temp_dir}/video_latent_upscaled.safetensors")
    print("[upscale] Done. Exiting.")


if __name__ == "__main__":
    main()
