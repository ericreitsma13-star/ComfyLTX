"""
LTX-2.3 19B Batch Pipeline — one-shot, no streaming builder.

Design:
- Builds the transformer ONCE on GPU as FP8 (~19 GB) and keeps it alive
  for all prompts via DiffusionStage.model_context().  
- USES OffloadMode.NONE — no pinned memory waste. OffloadMode.CPU pins
  ~19 GB in the unified pool, which is actively harmful on Strix Halo.
- Gemma and VAE are loaded/unloaded per prompt (their blocks handle this).
- Supports batch processing: pass multiple prompts with --prompts-file or
  call the script once per prompt (transformer stays hot).

Memory:
- Steady-state (transformer on GPU): ~19 GB
- Peak per prompt (transformer + Gemma + KV cache + activations): ~60 GB
  This fits in 93 GB unified with room to spare.
"""
import gc, json, logging, os, sys, time
from datetime import datetime
from pathlib import Path

os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"
os.environ["HIP_VISIBLE_DEVICES"] = "0"
sys.path.insert(0, "/opt/LTX-2/packages/ltx-core/src")
sys.path.insert(0, "/opt/LTX-2/packages/ltx-pipelines/src")

import torch
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.model.video_vae.tiling import TemporalTilingConfig
from ltx_core.quantization import QuantizationPolicy
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.utils.blocks import (
    AudioDecoder,
    DiffusionStage,
    ImageConditioner,
    PromptEncoder,
    VideoDecoder,
    VideoUpsampler,
)
from ltx_pipelines.utils.constants import DISTILLED_SIGMAS, STAGE_2_DISTILLED_SIGMAS
from ltx_pipelines.utils.denoisers import SimpleDenoiser
from ltx_pipelines.utils.helpers import combined_image_conditionings
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.types import ModalitySpec, OffloadMode


# ---------------------------------------------------------------------------
# Shared defaults
# ---------------------------------------------------------------------------
CHECKPOINT = "/root/comfy-models/diffusion_models/ltx-2-19b-distilled-fp8.safetensors"
SPATIAL_UPSCALER = (
    "/root/comfy-models/diffusion_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
)
GEMMA_ROOT = (
    "/root/.cache/huggingface/hub/"
    "models--google--gemma-3-12b-it-qat-q4_0-unquantized/"
    "snapshots/68f7ee4fbd59087436ada77ed2d62f373fdd4482"
)
OUTPUT_DIR = "/opt/ComfyUI/output"


def log_mem(phase):
    import psutil
    proc = psutil.Process()
    rss = proc.memory_info().rss / 1e9
    total = psutil.virtual_memory().total / 1e9
    used = psutil.virtual_memory().used / 1e9
    print(
        f"  [mem] {phase}: RSS={rss:.1f}GB, sys_used={used:.1f}GB/"
        f"{total:.0f}GB ({used / total * 100:.0f}%)"
    )


def run_prompt(
    prompt: str,
    seed: int,
    width: int,
    height: int,
    num_frames: int,
    fps: int,
    transformer: object,
    diffusion_stage: DiffusionStage,
    prompt_encoder: PromptEncoder,
    image_conditioner: ImageConditioner,
    upsampler: VideoUpsampler,
    video_decoder: VideoDecoder,
    audio_decoder: AudioDecoder,
    tiling_config: TilingConfig,
    images: list,
    enhance_prompt: bool,
    output_path: str,
    stage_1_sigmas: torch.Tensor,
    stage_2_sigmas: torch.Tensor,
    generator: torch.Generator,
):
    """Run a single prompt through the pipeline, reusing the hot transformer."""

    # --- Encode prompt (Gemma loaded/ freed per call) ---
    (ctx_p,) = prompt_encoder(
        [prompt],
        enhance_first_prompt=enhance_prompt,
        enhance_prompt_image=images[0][0] if len(images) > 0 else None,
    )
    video_context, audio_context = ctx_p.video_encoding, ctx_p.audio_encoding

    # --- Stage 1: half-res denoising ---
    stage_1_w, stage_1_h = width // 2, height // 2
    stage_1_conditionings = image_conditioner(
        lambda enc: combined_image_conditionings(
            images=images,
            height=stage_1_h,
            width=stage_1_w,
            video_encoder=enc,
            dtype=torch.bfloat16,
            device=transformer.device,
        )
    )

    video_state, audio_state = diffusion_stage.run(
        transformer=transformer,
        denoiser=SimpleDenoiser(video_context, audio_context),
        sigmas=stage_1_sigmas,
        noiser=GaussianNoiser(generator),  # each prompt gets independent noise
        width=stage_1_w,
        height=stage_1_h,
        frames=num_frames,
        fps=fps,
        video=ModalitySpec(context=video_context, conditionings=stage_1_conditionings),
        audio=ModalitySpec(context=audio_context),
    )

    # --- Upscale ---
    upscaled_video_latent = upsampler(video_state.latent[:1])

    # --- Stage 2: full-res refinement ---
    stage_2_conditionings = image_conditioner(
        lambda enc: combined_image_conditionings(
            images=images,
            height=height,
            width=width,
            video_encoder=enc,
            dtype=torch.bfloat16,
            device=transformer.device,
        )
    )

    video_state, audio_state = diffusion_stage.run(
        transformer=transformer,
        denoiser=SimpleDenoiser(video_context, audio_context),
        sigmas=stage_2_sigmas,
        noiser=GaussianNoiser(generator),
        width=width,
        height=height,
        frames=num_frames,
        fps=fps,
        video=ModalitySpec(
            context=video_context,
            conditionings=stage_2_conditionings,
            noise_scale=stage_2_sigmas[0].item(),
            initial_latent=upscaled_video_latent,
        ),
        audio=ModalitySpec(
            context=audio_context,
            noise_scale=stage_2_sigmas[0].item(),
            initial_latent=audio_state.latent,
        ),
    )

    # --- Decode ---
    video_chunks_number = get_video_chunks_number(num_frames, tiling_config)
    decoded_video = video_decoder(video_state.latent, tiling_config, generator)
    decoded_audio = audio_decoder(audio_state.latent)

    # --- Save ---
    encode_video(
        video=decoded_video,
        fps=fps,
        audio=decoded_audio,
        output_path=output_path,
        video_chunks_number=video_chunks_number,
    )

    # Clean up working memory (but NOT the transformer — it's held externally)
    del video_state, audio_state, ctx_p, video_context, audio_context
    del decoded_video, decoded_audio
    del stage_1_conditionings, stage_2_conditionings
    del upscaled_video_latent
    torch.cuda.empty_cache()
    gc.collect()

    print(f"  Output: {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="LTX-2.3 19B Batch — transformer stays hot on GPU"
    )
    parser.add_argument("--prompt", default=None, help="Single prompt")
    parser.add_argument(
        "--prompts-file", default=None, help="File with one prompt per line"
    )
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--image", default=None)
    parser.add_argument("--image-strength", type=float, default=0.3)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--frames", type=int, default=97)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None)
    parser.add_argument("--enhance-prompt", action="store_true", default=False)
    args = parser.parse_args()

    # Gather prompts
    prompts = []
    if args.prompts_file:
        with open(args.prompts_file) as f:
            prompts = [line.strip() for line in f if line.strip()]
    if args.prompt:
        prompts.append(args.prompt)
    if not prompts:
        prompts.append(
            "A serene Japanese garden at sunset with cherry blossom petals "
            "falling, koi fish swimming in a pond, cinematic quality."
        )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logging.getLogger().setLevel(logging.INFO)
    device = torch.device("cuda")
    MEM_GB = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU memory: {MEM_GB:.0f} GB")
    log_mem("start")
    print(f"Processing {len(prompts)} prompt(s) at {args.width}x{args.height}, "
          f"{args.frames} frames")

    images = []
    if args.image:
        if not os.path.exists(args.image):
            print(f"ERROR: image not found: {args.image}")
            sys.exit(1)
        images.append(ImageConditioningInput(args.image, 0, args.image_strength))

    # --- Build sub-components ONCE ---

    print("Building PromptEncoder (Gemma)...")
    prompt_encoder = PromptEncoder(
        checkpoint_path=CHECKPOINT,
        gemma_root=GEMMA_ROOT,
        dtype=torch.bfloat16,
        device=device,
        offload_mode=OffloadMode.NONE,
    )
    log_mem("after_prompt_encoder_init")  # Gemma and EmbeddingsProcessor builders

    print("Building ImageConditioner...")
    image_conditioner = ImageConditioner(
        checkpoint_path=CHECKPOINT,
        dtype=torch.bfloat16,
        device=device,
    )

    print("Building DiffusionStage...")
    diffusion_stage = DiffusionStage(
        checkpoint_path=CHECKPOINT,
        dtype=torch.bfloat16,
        device=device,
        loras=(),
        quantization=QuantizationPolicy.fp8_cast(),
        offload_mode=OffloadMode.NONE,  # KEY: no streaming, no pinned memory
    )

    print("Building VideoUpsampler...")
    upsampler = VideoUpsampler(
        checkpoint_path=CHECKPOINT,
        upsampler_path=SPATIAL_UPSCALER,
        dtype=torch.bfloat16,
        device=device,
    )

    print("Building VideoDecoder...")
    video_decoder = VideoDecoder(
        checkpoint_path=CHECKPOINT,
        dtype=torch.bfloat16,
        device=device,
    )

    print("Building AudioDecoder...")
    audio_decoder = AudioDecoder(
        checkpoint_path=CHECKPOINT,
        dtype=torch.bfloat16,
        device=device,
    )
    log_mem("after_blocks_init")

    # --- Tiling config ---
    tiling_config = TilingConfig(
        spatial_config=TilingConfig.default().spatial_config,
        temporal_config=TemporalTilingConfig(
            tile_size_in_frames=48,
            tile_overlap_in_frames=24,
        ),
    )

    # --- Sigmas (pre-computed) ---
    stage_1_sigmas = DISTILLED_SIGMAS.to(dtype=torch.float32, device=device)
    stage_2_sigmas = STAGE_2_DISTILLED_SIGMAS.to(dtype=torch.float32, device=device)

    # --- Build transformer ONCE, keep it alive for all prompts ---
    print("Building transformer (this loads ~19 GB onto GPU)...")
    with diffusion_stage.model_context() as transformer:
        log_mem("after_transformer_build")

        for i, prompt in enumerate(prompts):
            print(f"\n--- Prompt {i+1}/{len(prompts)} ---")
            print(f"  \"{prompt[:80]}...\"")
            t0 = time.time()

            generator = torch.Generator(device=device).manual_seed(args.seed + i)

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = args.output or f"ltx23_19b_{ts}_p{i+1}.mp4"
            output_path = os.path.join(OUTPUT_DIR, output_filename)

            run_prompt(
                prompt=prompt,
                seed=args.seed + i,
                width=args.width,
                height=args.height,
                num_frames=args.frames,
                fps=args.fps,
                transformer=transformer,
                diffusion_stage=diffusion_stage,
                prompt_encoder=prompt_encoder,
                image_conditioner=image_conditioner,
                upsampler=upsampler,
                video_decoder=video_decoder,
                audio_decoder=audio_decoder,
                tiling_config=tiling_config,
                images=images,
                enhance_prompt=args.enhance_prompt,
                output_path=output_path,
                stage_1_sigmas=stage_1_sigmas,
                stage_2_sigmas=stage_2_sigmas,
                generator=generator,
            )

            elapsed = time.time() - t0
            print(f"  Done in {elapsed:.1f}s")

    # Transformer is freed here (model_context exits)

    print("\nAll prompts complete.")
    log_mem("end")


if __name__ == "__main__":
    main()
