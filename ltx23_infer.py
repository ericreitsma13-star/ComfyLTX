import sys, os, logging, time, gc, psutil, argparse
from datetime import datetime
from PIL import Image

os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"
os.environ["HIP_VISIBLE_DEVICES"] = "0"

sys.path.insert(0, "/opt/LTX-2/packages/ltx-core/src")
sys.path.insert(0, "/opt/LTX-2/packages/ltx-pipelines/src")

import torch
from ltx_pipelines.distilled import DistilledPipeline
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.types import OffloadMode
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.model.video_vae.tiling import TemporalTilingConfig
from ltx_core.quantization import QuantizationPolicy
from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps

CHECKPOINT = "/root/comfy-models/diffusion_models/ltx-2.3-22b-distilled-1.1.safetensors"
SPATIAL_UPSCALER = (
    "/root/comfy-models/diffusion_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
)
TEMPORAL_UPSCALER = (
    "/root/comfy-models/diffusion_models/ltx-2.3-temporal-upscaler-x2-1.0.safetensors"
)
LORA_22B = (
    "/root/comfy-models/diffusion_models/ltx-2.3-22b-distilled-lora-384-1.1.safetensors"
)
GEMMA_ROOT = "/root/.cache/huggingface/hub/models--google--gemma-3-12b-it-qat-q4_0-unquantized/snapshots/68f7ee4fbd59087436ada77ed2d62f373fdd4482"
OUTPUT_DIR = "/opt/ComfyUI/output"


def parse_args():
    parser = argparse.ArgumentParser(description="LTX-2.3 inference")
    parser.add_argument(
        "--prompt", default=None, help="Text prompt (overrides default)"
    )
    parser.add_argument("--negative-prompt", default=None, help="Negative prompt")
    parser.add_argument("--image", default=None, help="Path to input image for I2V")
    parser.add_argument(
        "--image-strength", type=float, default=0.3, help="Image conditioning strength"
    )
    parser.add_argument(
        "--lora", default=None, help="Path to Distill LoRA .safetensors"
    )
    parser.add_argument("--lora-strength", type=float, default=0.6, help="LoRA weight")
    parser.add_argument(
        "--temporal-upscaler",
        action="store_true",
        help="Enable temporal upscaler (24->48fps)",
    )
    parser.add_argument("--width", type=int, default=1280, help="Output width")
    parser.add_argument("--height", type=int, default=768, help="Output height")
    parser.add_argument("--frames", type=int, default=121, help="Number of frames")
    parser.add_argument("--fps", type=int, default=24, help="Output framerate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", default=None, help="Output filename")
    return parser.parse_args()


def log_mem(phase):
    proc = psutil.Process()
    rss = proc.memory_info().rss / 1e9
    total = psutil.virtual_memory().total / 1e9
    used = psutil.virtual_memory().used / 1e9
    print(
        f"[mem] {phase}: RSS={rss:.1f}GB, sys_used={used:.1f}GB/{total:.0f}GB ({used / total * 100:.0f}%)"
    )


def main():
    args = parse_args()

    prompt = (
        args.prompt
        or "A serene Japanese garden at sunset with cherry blossom petals falling, koi fish swimming in a pond, a wooden bridge, soft golden lighting, gentle water sounds, birds chirping in the distance, cinematic quality."
    )
    negative_prompt = (
        args.negative_prompt
        or "blurry, out of focus, low quality, artifacts, distorted, silent or muted audio, robotic voice, background noise"
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logging.getLogger().setLevel(logging.INFO)

    device = torch.device("cuda")
    MEM_GB = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU memory: {MEM_GB:.0f} GB")
    log_mem("start")

    loras = []
    lora_path = args.lora
    if lora_path is None and os.path.exists(LORA_22B):
        lora_path = LORA_22B
    if lora_path and os.path.exists(lora_path):
        print(f"Loading LoRA: {lora_path} (strength={args.lora_strength})")
        loras.append(
            LoraPathStrengthAndSDOps(
                lora_path, args.lora_strength, LTXV_LORA_COMFY_RENAMING_MAP
            )
        )

    upscaler_path = SPATIAL_UPSCALER
    if args.temporal_upscaler and os.path.exists(TEMPORAL_UPSCALER):
        print("Enabling temporal upscaler (x2 frame rate)")
        upscaler_path = os.path.dirname(SPATIAL_UPSCALER)

    print("Building DistilledPipeline with FP8 cast...")
    pipeline = DistilledPipeline(
        distilled_checkpoint_path=CHECKPOINT,
        gemma_root=GEMMA_ROOT,
        spatial_upsampler_path=upscaler_path,
        loras=loras,
        device=device,
        offload_mode=OffloadMode.CPU,
        quantization=QuantizationPolicy.fp8_cast(),
    )
    log_mem("after_pipeline_build")

    images = []
    if args.image:
        if not os.path.exists(args.image):
            print(f"ERROR: image not found: {args.image}")
            sys.exit(1)
        print(f"Using I2V with image: {args.image} (strength={args.image_strength})")
        images.append(ImageConditioningInput(args.image, 0, args.image_strength))

    tiling_config = TilingConfig(
        spatial_config=TilingConfig.default().spatial_config,
        temporal_config=TemporalTilingConfig(
            tile_size_in_frames=48, tile_overlap_in_frames=24
        ),
    )
    video_chunks_number = get_video_chunks_number(args.frames, tiling_config)

    torch.cuda.empty_cache()
    gc.collect()
    print(f"Generating video+audio from prompt...")
    t0 = time.time()

    video, audio = pipeline(
        prompt=prompt,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_frames=args.frames,
        frame_rate=args.fps,
        images=images,
        tiling_config=tiling_config,
        enhance_prompt=False,
    )

    elapsed = time.time() - t0
    log_mem("after_generation")
    print(f"Generation done in {elapsed:.1f}s")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = args.output or f"ltx23_{ts}.mp4"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    encode_video(
        video=video,
        fps=args.fps,
        audio=audio,
        output_path=output_path,
        video_chunks_number=video_chunks_number,
    )

    print(f"Output: {output_path}")
    print(f"Audio channels: {audio.array.shape if hasattr(audio, 'array') else 'N/A'}")
    print(
        f"Audio sample rate: {audio.sample_rate if hasattr(audio, 'sample_rate') else 'N/A'}"
    )


if __name__ == "__main__":
    main()
