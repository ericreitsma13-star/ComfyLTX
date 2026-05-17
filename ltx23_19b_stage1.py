"""
Stage 1: prompt encoding + half-res denoising.
Loads Gemma (text encoder) and the transformer, runs 8 denoising steps at half
resolution, saves intermediate latents + context to disk, then exits.

This is the heaviest stage (~19 GB model weights + ~5 GB KV cache).
"""
import gc, json, logging, os, sys, time
from pathlib import Path

os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"
os.environ["HIP_VISIBLE_DEVICES"] = "0"
sys.path.insert(0, "/opt/LTX-2/packages/ltx-core/src")
sys.path.insert(0, "/opt/LTX-2/packages/ltx-pipelines/src")

import torch

# Import order matters: ltx_pipelines.utils.blocks triggers the loader chain
# in the correct order. Importing ltx_core.quantization first causes a
# circular import (loader → quantization → loader) and crashes.
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.utils.blocks import DiffusionStage, ImageConditioner, PromptEncoder
from ltx_pipelines.utils.constants import DISTILLED_SIGMAS
from ltx_pipelines.utils.denoisers import SimpleDenoiser
from ltx_pipelines.utils.helpers import combined_image_conditionings
from ltx_pipelines.utils.types import ModalitySpec, OffloadMode

from ltx_core.components.noisers import GaussianNoiser
from safetensors.torch import save_file


def save_tensor(t: torch.Tensor, path: str):
    """Save a single tensor as safetensors with a 'data' key."""
    save_file({"data": t.cpu()}, path)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="LTX-2.3 19B Stage 1 (half-res denoising)")
    parser.add_argument("--distilled-checkpoint", default="/root/comfy-models/diffusion_models/ltx-2-19b-distilled-fp8.safetensors")
    parser.add_argument("--gemma-root", default="/root/.cache/huggingface/hub/models--google--gemma-3-12b-it-qat-q4_0-unquantized/snapshots/68f7ee4fbd59087436ada77ed2d62f373fdd4482")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--frames", type=int, default=97)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--image", default=None)
    parser.add_argument("--image-strength", type=float, default=0.3)
    parser.add_argument("--temp-dir", default="/tmp/ltx23_19b_split")
    parser.add_argument("--enhance-prompt", action="store_true", default=False)
    args = parser.parse_args()

    os.makedirs(args.temp_dir, exist_ok=True)
    logging.getLogger().setLevel(logging.INFO)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    prompt = args.prompt or (
        "A serene Japanese garden at sunset with cherry blossom petals falling, "
        "koi fish swimming in a pond, a wooden bridge, soft golden lighting, "
        "gentle water sounds, birds chirping in the distance, cinematic quality."
    )

    # --- Build prompt encoder, encode prompt ---
    print("[stage1] Building PromptEncoder...")
    prompt_encoder = PromptEncoder(
        checkpoint_path=args.distilled_checkpoint,
        gemma_root=args.gemma_root,
        dtype=dtype,
        device=device,
        offload_mode=OffloadMode.CPU,
    )
    images = []
    if args.image:
        images.append(ImageConditioningInput(args.image, 0, args.image_strength))

    print("[stage1] Encoding prompt...")
    (ctx_p,) = prompt_encoder(
        [prompt],
        enhance_first_prompt=args.enhance_prompt,
        enhance_prompt_image=images[0][0] if len(images) > 0 else None,
    )
    del prompt_encoder
    torch.cuda.empty_cache()
    gc.collect()

    video_context = ctx_p.video_encoding
    audio_context = ctx_p.audio_encoding
    print(f"[stage1] video_context shape: {video_context.shape}, audio_context shape: {audio_context.shape}")

    # --- Save context tensors ---
    save_tensor(video_context, os.path.join(args.temp_dir, "video_context.safetensors"))
    save_tensor(audio_context, os.path.join(args.temp_dir, "audio_context.safetensors"))

    # --- Stage 1: half-res denoising ---
    stage_1_sigmas = DISTILLED_SIGMAS.to(dtype=torch.float32, device=device)
    stage_1_w, stage_1_h = args.width // 2, args.height // 2

    # Build image conditioner and get half-res conditionings
    image_conditioner = ImageConditioner(
        checkpoint_path=args.distilled_checkpoint,
        dtype=dtype,
        device=device,
    )
    print(f"[stage1] Stage 1 resolution: {stage_1_w}x{stage_1_h}, {args.frames} frames")
    stage_1_conditionings = image_conditioner(
        lambda enc: combined_image_conditionings(
            images=images,
            height=stage_1_h,
            width=stage_1_w,
            video_encoder=enc,
            dtype=dtype,
            device=device,
        )
    )
    del image_conditioner
    torch.cuda.empty_cache()
    gc.collect()

    # Build diffusion stage and run
    from ltx_core.quantization import QuantizationPolicy
    diffusion_stage = DiffusionStage(
        checkpoint_path=args.distilled_checkpoint,
        dtype=dtype,
        device=device,
        loras=(),
        quantization=QuantizationPolicy.fp8_cast(),
        offload_mode=OffloadMode.CPU,
    )

    generator = torch.Generator(device=device).manual_seed(args.seed)
    noiser = GaussianNoiser(generator=generator)

    print("[stage1] Running stage 1 denoising (8 steps)...")
    t0 = time.time()
    video_state, audio_state = diffusion_stage(
        denoiser=SimpleDenoiser(video_context, audio_context),
        sigmas=stage_1_sigmas,
        noiser=noiser,
        width=stage_1_w,
        height=stage_1_h,
        frames=args.frames,
        fps=args.fps,
        video=ModalitySpec(context=video_context, conditionings=stage_1_conditionings),
        audio=ModalitySpec(context=audio_context),
    )
    elapsed = time.time() - t0
    print(f"[stage1] Stage 1 done in {elapsed:.1f}s")

    # Save video and audio latents
    print(f"[stage1] video_state.latent shape: {video_state.latent.shape}")
    save_tensor(video_state.latent, os.path.join(args.temp_dir, "video_latent_stage1.safetensors"))
    print(f"[stage1] audio_state.latent shape: {audio_state.latent.shape}")
    save_tensor(audio_state.latent, os.path.join(args.temp_dir, "audio_latent_stage1.safetensors"))

    # Save metadata for downstream stages
    metadata = {
        "width": args.width,
        "height": args.height,
        "frames": args.frames,
        "fps": args.fps,
        "seed": args.seed,
        "stage_1_width": stage_1_w,
        "stage_1_height": stage_1_h,
        "video_latent_channels": video_state.latent.shape[1],
        "video_latent_frames": video_state.latent.shape[2],
        "video_latent_height": video_state.latent.shape[3],
        "video_latent_width": video_state.latent.shape[4],
        "audio_latent_channels": audio_state.latent.shape[1],
        "audio_latent_frames": audio_state.latent.shape[2],
        "audio_latent_mel_bins": audio_state.latent.shape[3],
    }
    with open(os.path.join(args.temp_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[stage1] Temp files saved to {args.temp_dir}/")
    print("[stage1] Done. Exiting — memory will be reclaimed.")


if __name__ == "__main__":
    main()
