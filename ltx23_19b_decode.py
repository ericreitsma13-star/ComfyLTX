"""
Stage 4: VAE decode + audio decode + MP4 export.
Loads the video VAE decoder and audio VAE decoder + vocoder (~3 GB total),
decodes the refined latents to pixels/audio, writes MP4, then exits.
"""
import gc, json, logging, os, sys, time
from datetime import datetime

os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"
os.environ["HIP_VISIBLE_DEVICES"] = "0"
sys.path.insert(0, "/opt/LTX-2/packages/ltx-core/src")
sys.path.insert(0, "/opt/LTX-2/packages/ltx-pipelines/src")

import torch
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.model.video_vae.tiling import TemporalTilingConfig
from ltx_pipelines.utils.blocks import AudioDecoder, VideoDecoder
from ltx_pipelines.utils.media_io import encode_video
from safetensors.torch import load_file


def load_tensor(path: str) -> torch.Tensor:
    return load_file(path)["data"]


def main():
    import argparse

    parser = argparse.ArgumentParser(description="LTX-2.3 19B VAE Decode + MP4 Export")
    parser.add_argument("--checkpoint", default="/root/comfy-models/diffusion_models/ltx-2-19b-distilled-fp8.safetensors")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--tile-frames", type=int, default=48)
    parser.add_argument("--tile-overlap", type=int, default=24)
    parser.add_argument("--temp-dir", default="/tmp/ltx23_19b_split")
    parser.add_argument("--output-dir", default="/opt/ComfyUI/output")
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logging.getLogger().setLevel(logging.INFO)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Load metadata
    with open(os.path.join(args.temp_dir, "metadata.json")) as f:
        metadata = json.load(f)

    # Load refined latents
    print("[decode] Loading refined latents...")
    video_latent = load_tensor(os.path.join(args.temp_dir, "video_latent_refined.safetensors")).to(device=device, dtype=dtype)
    audio_latent = load_tensor(os.path.join(args.temp_dir, "audio_latent_refined.safetensors")).to(device=device, dtype=dtype)
    print(f"[decode] video_latent shape: {video_latent.shape}")
    print(f"[decode] audio_latent shape: {audio_latent.shape}")

    frames = metadata.get("frames", 97)
    fps = args.fps or metadata.get("fps", 24)

    # Build tiling config
    print(f"[decode] VAE tiling: {args.tile_frames} frames, {args.tile_overlap} overlap")
    tiling_config = TilingConfig(
        spatial_config=TilingConfig.default().spatial_config,
        temporal_config=TemporalTilingConfig(
            tile_size_in_frames=args.tile_frames,
            tile_overlap_in_frames=args.tile_overlap,
        ),
    )
    video_chunks_number = get_video_chunks_number(frames, tiling_config)

    # Build video decoder
    print("[decode] Building VideoDecoder...")
    video_decoder = VideoDecoder(
        checkpoint_path=args.checkpoint,
        dtype=dtype,
        device=device,
    )
    generator = torch.Generator(device=device).manual_seed(args.seed)

    print("[decode] Decoding video...")
    t0 = time.time()
    decoded_video = video_decoder(video_latent, tiling_config, generator)
    elapsed = time.time() - t0
    print(f"[decode] Video decode done in {elapsed:.1f}s")

    # Build audio decoder
    print("[decode] Building AudioDecoder...")
    audio_decoder = AudioDecoder(
        checkpoint_path=args.checkpoint,
        dtype=dtype,
        device=device,
    )

    print("[decode] Decoding audio...")
    t0 = time.time()
    decoded_audio = audio_decoder(audio_latent)
    elapsed = time.time() - t0
    print(f"[decode] Audio decode done in {elapsed:.1f}s")

    # Save MP4
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = args.output or f"ltx23_19b_{ts}.mp4"
    output_path = os.path.join(args.output_dir, output_filename)

    print(f"[decode] Encoding MP4 to {output_path}...")
    encode_video(
        video=decoded_video,
        fps=fps,
        audio=decoded_audio,
        output_path=output_path,
        video_chunks_number=video_chunks_number,
    )
    print(f"[decode] Output: {output_path}")
    print(f"[decode] Audio channels: {decoded_audio.array.shape if hasattr(decoded_audio, 'array') else 'N/A'}")
    print(f"[decode] Audio sample rate: {decoded_audio.sample_rate if hasattr(decoded_audio, 'sample_rate') else 'N/A'}")
    print("[decode] Done. Exiting.")


if __name__ == "__main__":
    main()
