"""
Stage 3: Full-resolution refinement.
Loads the transformer again, loads upscaled latents + context from disk, runs 4
distilled denoising steps at full resolution, saves refined latents, then exits.

Memory: ~19 GB model weights + ~22 GB KV cache (at full resolution).
"""
import gc, json, logging, os, sys, time

os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"
os.environ["HIP_VISIBLE_DEVICES"] = "0"
sys.path.insert(0, "/opt/LTX-2/packages/ltx-core/src")
sys.path.insert(0, "/opt/LTX-2/packages/ltx-pipelines/src")

import torch

# Import order matters: blocks triggers loader chain before quantization
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.utils.blocks import DiffusionStage, ImageConditioner
from ltx_pipelines.utils.constants import STAGE_2_DISTILLED_SIGMAS
from ltx_pipelines.utils.denoisers import SimpleDenoiser
from ltx_pipelines.utils.helpers import combined_image_conditionings
from ltx_pipelines.utils.types import ModalitySpec, OffloadMode

from ltx_core.components.noisers import GaussianNoiser
from ltx_core.quantization import QuantizationPolicy
from safetensors.torch import load_file, save_file


def load_tensor(path: str) -> torch.Tensor:
    return load_file(path)["data"]


def main():
    import argparse

    parser = argparse.ArgumentParser(description="LTX-2.3 19B Stage 2 (full-res refinement)")
    parser.add_argument("--distilled-checkpoint", default="/root/comfy-models/diffusion_models/ltx-2-19b-distilled-fp8.safetensors")
    parser.add_argument("--gemma-root", default="/root/.cache/huggingface/hub/models--google--gemma-3-12b-it-qat-q4_0-unquantized/snapshots/68f7ee4fbd59087436ada77ed2d62f373fdd4482")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--frames", type=int, default=97)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--image", default=None)
    parser.add_argument("--image-strength", type=float, default=0.3)
    parser.add_argument("--temp-dir", default="/tmp/ltx23_19b_split")
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Load metadata
    with open(os.path.join(args.temp_dir, "metadata.json")) as f:
        metadata = json.load(f)

    # Load contexts from disk (saved by stage 1)
    print("[stage2] Loading contexts from disk...")
    video_context = load_tensor(os.path.join(args.temp_dir, "video_context.safetensors")).to(device=device, dtype=dtype)
    audio_context = load_tensor(os.path.join(args.temp_dir, "audio_context.safetensors")).to(device=device, dtype=dtype)

    # Load upscaled video latent
    print("[stage2] Loading upscaled video latent...")
    upscaled_video_latent = load_tensor(os.path.join(args.temp_dir, "video_latent_upscaled.safetensors")).to(device=device, dtype=dtype)
    print(f"[stage2] upscaled_video_latent shape: {upscaled_video_latent.shape}")

    # Load audio latent from stage 1
    print("[stage2] Loading audio latent from stage 1...")
    audio_latent_s1 = load_tensor(os.path.join(args.temp_dir, "audio_latent_stage1.safetensors")).to(device=device, dtype=dtype)
    print(f"[stage2] audio_latent shape: {audio_latent_s1.shape}")

    # Build image conditioner for full-res conditionings
    images = []
    if args.image:
        images.append(ImageConditioningInput(args.image, 0, args.image_strength))

    image_conditioner = ImageConditioner(
        checkpoint_path=args.distilled_checkpoint,
        dtype=dtype,
        device=device,
    )
    print(f"[stage2] Stage 2 resolution: {args.width}x{args.height}")
    stage_2_conditionings = image_conditioner(
        lambda enc: combined_image_conditionings(
            images=images,
            height=args.height,
            width=args.width,
            video_encoder=enc,
            dtype=dtype,
            device=device,
        )
    )
    del image_conditioner
    torch.cuda.empty_cache()
    gc.collect()

    # Build diffusion stage for refinement
    diffusion_stage = DiffusionStage(
        checkpoint_path=args.distilled_checkpoint,
        dtype=dtype,
        device=device,
        loras=(),
        quantization=QuantizationPolicy.fp8_cast(),
        offload_mode=OffloadMode.CPU,
    )

    stage_2_sigmas = STAGE_2_DISTILLED_SIGMAS.to(dtype=torch.float32, device=device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    noiser = GaussianNoiser(generator=generator)

    # Stage 2: refine at full resolution
    print("[stage2] Running stage 2 refinement (4 steps)...")
    t0 = time.time()
    video_state, audio_state = diffusion_stage(
        denoiser=SimpleDenoiser(video_context, audio_context),
        sigmas=stage_2_sigmas,
        noiser=noiser,
        width=args.width,
        height=args.height,
        frames=args.frames,
        fps=args.fps,
        video=ModalitySpec(
            context=video_context,
            conditionings=stage_2_conditionings,
            noise_scale=stage_2_sigmas[0].item(),
            initial_latent=upscaled_video_latent,
        ),
        audio=ModalitySpec(
            context=audio_context,
            noise_scale=stage_2_sigmas[0].item(),
            initial_latent=audio_latent_s1,
        ),
    )
    elapsed = time.time() - t0
    print(f"[stage2] Stage 2 done in {elapsed:.1f}s")

    print(f"[stage2] video_state.latent shape: {video_state.latent.shape}")
    print(f"[stage2] audio_state.latent shape: {audio_state.latent.shape}")

    # Save refined latents
    save_file({"data": video_state.latent.cpu()}, os.path.join(args.temp_dir, "video_latent_refined.safetensors"))
    save_file({"data": audio_state.latent.cpu()}, os.path.join(args.temp_dir, "audio_latent_refined.safetensors"))

    # Update metadata with refined latent shape info
    metadata["video_latent_refined_shape"] = list(video_state.latent.shape)
    metadata["audio_latent_refined_shape"] = list(audio_state.latent.shape)
    with open(os.path.join(args.temp_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("[stage2] Done. Exiting.")


if __name__ == "__main__":
    main()
