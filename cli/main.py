#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FlashVSR CLI Tool
A command-line interface for running FlashVSR video super-resolution inference.
"""

import os
import re
import sys
import argparse
import numpy as np
from PIL import Image
import imageio
from tqdm import tqdm
import torch
from einops import rearrange

# Add examples/WanVSR to path for imports
# Get the directory where this script is located
_script_dir = os.path.dirname(os.path.abspath(__file__))

try:
    from diffsynth import ModelManager, save_video
    from flashvsr.pipelines.flashvsr_full import FlashVSRFullPipeline
    from flashvsr.pipelines.flashvsr_tiny import FlashVSRTinyPipeline
    from flashvsr.pipelines.flashvsr_tiny_long import FlashVSRTinyLongPipeline
    from flashvsr.models.flashvsr_utils import Causal_LQ4x_Proj, Buffer_LQ4x_Proj
    from flashvsr.models.flashvsr_tcdecoder import build_tcdecoder
    from flashvsr.registry import register_wan_models

    # Register Wan models with diffsynth
    register_wan_models()
except ImportError as e:
    print(f"Error importing dependencies: {e}")
    print("Please ensure you have installed the package correctly.")
    sys.exit(1)


# ==================== Utility Functions ====================

def tensor2video(frames: torch.Tensor):
    """Convert tensor to video frames."""
    frames = rearrange(frames, "C T H W -> T H W C")
    frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
    frames = [Image.fromarray(frame) for frame in frames]
    return frames


def natural_key(name: str):
    """Natural sort key for filenames."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'([0-9]+)', os.path.basename(name))]


def list_images_natural(folder: str):
    """List images in folder with natural sorting."""
    exts = ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')
    fs = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(exts)]
    fs.sort(key=natural_key)
    return fs


def largest_8n1_leq(n):  # 8n+1
    """Find largest number <= n that is 8n+1."""
    return 0 if n < 1 else ((n - 1) // 8) * 8 + 1


def is_video(path):
    """Check if path is a video file."""
    return os.path.isfile(path) and path.lower().endswith(('.mp4', '.mov', '.avi', '.mkv'))


def pil_to_tensor_neg1_1(img: Image.Image, dtype=torch.bfloat16, device='cuda'):
    """Convert PIL image to tensor in [-1, 1] range."""
    t = torch.from_numpy(np.asarray(img, np.uint8)).to(device=device, dtype=torch.float32)  # HWC
    t = t.permute(2, 0, 1) / 255.0 * 2.0 - 1.0  # CHW in [-1,1]
    return t.to(dtype)


def save_video(frames, save_path, fps=30, quality=5):
    """Save video frames to file."""
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    w = imageio.get_writer(save_path, fps=fps, quality=quality)
    for f in tqdm(frames, desc=f"Saving {os.path.basename(save_path)}"):
        w.append_data(np.array(f))
    w.close()


def compute_scaled_and_target_dims(w0: int, h0: int, scale: float = 4.0, multiple: int = 128):
    """Compute scaled and target dimensions."""
    if w0 <= 0 or h0 <= 0:
        raise ValueError("Invalid original size")
    if scale <= 0:
        raise ValueError("scale must be > 0")

    sW = int(round(w0 * scale))
    sH = int(round(h0 * scale))

    tW = (sW // multiple) * multiple
    tH = (sH // multiple) * multiple

    if tW == 0 or tH == 0:
        raise ValueError(
            f"Scaled size too small ({sW}x{sH}) for multiple={multiple}. "
            f"Increase scale (got {scale})."
        )

    return sW, sH, tW, tH


def upscale_then_center_crop(img: Image.Image, scale: float, tW: int, tH: int) -> Image.Image:
    """Upscale image and center crop to target dimensions."""
    w0, h0 = img.size
    sW = int(round(w0 * scale))
    sH = int(round(h0 * scale))

    if tW > sW or tH > sH:
        raise ValueError(
            f"Target crop ({tW}x{tH}) exceeds scaled size ({sW}x{sH}). "
            f"Increase scale."
        )

    up = img.resize((sW, sH), Image.BICUBIC)
    l = (sW - tW) // 2
    t = (sH - tH) // 2
    return up.crop((l, t, l + tW, t + tH))


def prepare_input_tensor(path: str, scale: float = 4, dtype=torch.bfloat16, device='cuda'):
    """Prepare input tensor from video file or image directory."""
    if os.path.isdir(path):
        paths0 = list_images_natural(path)
        if not paths0:
            raise FileNotFoundError(f"No images in {path}")

        with Image.open(paths0[0]) as _img0:
            w0, h0 = _img0.size
        N0 = len(paths0)
        print(f"[{os.path.basename(path)}] Original Resolution: {w0}x{h0} | Original Frames: {N0}")

        sW, sH, tW, tH = compute_scaled_and_target_dims(w0, h0, scale=scale, multiple=128)
        print(f"[{os.path.basename(path)}] Scaled (x{scale:.2f}): {sW}x{sH} -> Target (128-multiple): {tW}x{tH}")

        paths = paths0 + [paths0[-1]] * 4
        F = largest_8n1_leq(len(paths))
        if F == 0:
            raise RuntimeError(f"Not enough frames after padding in {path}. Got {len(paths)}.")
        paths = paths[:F]
        print(f"[{os.path.basename(path)}] Target Frames (8n-3): {F-4}")

        frames = []
        for p in tqdm(paths, desc="Loading frames"):
            with Image.open(p).convert('RGB') as img:
                img_out = upscale_then_center_crop(img, scale=scale, tW=tW, tH=tH)
            frames.append(pil_to_tensor_neg1_1(img_out, dtype, device))
        vid = torch.stack(frames, 0).permute(1, 0, 2, 3).unsqueeze(0)  # 1 C F H W
        fps = 30
        return vid, tH, tW, F, fps

    if is_video(path):
        rdr = imageio.get_reader(path)
        first = Image.fromarray(rdr.get_data(0)).convert('RGB')
        w0, h0 = first.size

        meta = {}
        try:
            meta = rdr.get_meta_data()
        except Exception:
            pass
        fps_val = meta.get('fps', 30)
        fps = int(round(fps_val)) if isinstance(fps_val, (int, float)) else 30

        def count_frames(r):
            try:
                nf = meta.get('nframes', None)
                if isinstance(nf, int) and nf > 0:
                    return nf
            except Exception:
                pass
            try:
                return r.count_frames()
            except Exception:
                n = 0
                try:
                    while True:
                        r.get_data(n)
                        n += 1
                except Exception:
                    return n

        total = count_frames(rdr)
        if total <= 0:
            rdr.close()
            raise RuntimeError(f"Cannot read frames from {path}")

        print(f"[{os.path.basename(path)}] Original Resolution: {w0}x{h0} | Original Frames: {total} | FPS: {fps}")

        sW, sH, tW, tH = compute_scaled_and_target_dims(w0, h0, scale=scale, multiple=128)
        print(f"[{os.path.basename(path)}] Scaled (x{scale:.2f}): {sW}x{sH} -> Target (128-multiple): {tW}x{tH}")

        idx = list(range(total)) + [total - 1] * 4
        F = largest_8n1_leq(len(idx))
        if F == 0:
            rdr.close()
            raise RuntimeError(f"Not enough frames after padding in {path}. Got {len(idx)}.")
        idx = idx[:F]
        print(f"[{os.path.basename(path)}] Target Frames (8n-3): {F-4}")

        frames = []
        try:
            for i in tqdm(idx, desc="Loading frames"):
                img = Image.fromarray(rdr.get_data(i)).convert('RGB')
                img_out = upscale_then_center_crop(img, scale=scale, tW=tW, tH=tH)
                frames.append(pil_to_tensor_neg1_1(img_out, dtype, device))
        finally:
            try:
                rdr.close()
            except Exception:
                pass

        vid = torch.stack(frames, 0).permute(1, 0, 2, 3).unsqueeze(0)  # 1 C F H W
        return vid, tH, tW, F, fps

    raise ValueError(f"Unsupported input: {path}")


# ==================== Pipeline Initialization ====================

def init_pipeline(pipeline_type: str, model_version: str, model_dir: str = None, dtype=torch.bfloat16, device='cuda'):
    """Initialize the appropriate FlashVSR pipeline."""
    
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Determine model directory
    if model_dir is None:
        # Try relative to script directory first
        if model_version == "v1.1":
            model_dir = os.path.join(script_dir, "examples", "WanVSR", "FlashVSR-v1.1")
        else:  # v1
            model_dir = os.path.join(script_dir, "examples", "WanVSR", "FlashVSR")
        
        # If not found, try relative to current working directory
        if not os.path.exists(model_dir):
            if model_version == "v1.1":
                model_dir = os.path.join(os.getcwd(), "examples", "WanVSR", "FlashVSR-v1.1")
            else:  # v1
                model_dir = os.path.join(os.getcwd(), "examples", "WanVSR", "FlashVSR")
    else:
        model_dir = os.path.expanduser(model_dir)
        if not os.path.isabs(model_dir):
            # If relative path, try relative to script dir first, then cwd
            abs_path = os.path.join(script_dir, model_dir)
            if os.path.exists(abs_path):
                model_dir = abs_path
            else:
                model_dir = os.path.abspath(model_dir)
    
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    print(f"Using model directory: {model_dir}")
    print(f"GPU: {torch.cuda.current_device()} - {torch.cuda.get_device_name(torch.cuda.current_device())}")
    
    mm = ModelManager(torch_dtype=dtype, device="cpu")
    
    # Load base model
    diffusion_model_path = os.path.join(model_dir, "diffusion_pytorch_model_streaming_dmd.safetensors")
    if not os.path.exists(diffusion_model_path):
        raise FileNotFoundError(f"Diffusion model not found: {diffusion_model_path}")
    
    mm.load_models([diffusion_model_path])
    
    # Initialize pipeline based on type
    if pipeline_type == "full":
        pipe = FlashVSRFullPipeline.from_model_manager(mm, device=device)
        
        # Load VAE for full pipeline
        vae_path = os.path.join(model_dir, "Wan2.1_VAE.pth")
        if not os.path.exists(vae_path):
            raise FileNotFoundError(f"VAE model not found: {vae_path}")
        mm.load_models([vae_path])
        
        # Setup LQ projection
        if model_version == "v1.1":
            pipe.denoising_model().LQ_proj_in = Causal_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1).to(device, dtype=dtype)
        else:  # v1
            pipe.denoising_model().LQ_proj_in = Buffer_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1).to(device, dtype=dtype)
        
        LQ_proj_in_path = os.path.join(model_dir, "LQ_proj_in.ckpt")
        if os.path.exists(LQ_proj_in_path):
            pipe.denoising_model().LQ_proj_in.load_state_dict(
                torch.load(LQ_proj_in_path, map_location="cpu"), strict=True
            )
        
        pipe.denoising_model().LQ_proj_in.to(device)
        pipe.vae.model.encoder = None
        pipe.vae.model.conv1 = None
        
    elif pipeline_type == "tiny":
        pipe = FlashVSRTinyPipeline.from_model_manager(mm, device=device)
        
        # Setup LQ projection
        pipe.denoising_model().LQ_proj_in = Causal_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1).to(device, dtype=dtype)
        
        LQ_proj_in_path = os.path.join(model_dir, "LQ_proj_in.ckpt")
        if os.path.exists(LQ_proj_in_path):
            pipe.denoising_model().LQ_proj_in.load_state_dict(
                torch.load(LQ_proj_in_path, map_location="cpu"), strict=True
            )
        
        pipe.denoising_model().LQ_proj_in.to(device)
        
        # Setup TCDecoder
        multi_scale_channels = [512, 256, 128, 128]
        pipe.TCDecoder = build_tcdecoder(new_channels=multi_scale_channels, new_latent_channels=16+768)
        tcdecoder_path = os.path.join(model_dir, "TCDecoder.ckpt")
        if os.path.exists(tcdecoder_path):
            mis = pipe.TCDecoder.load_state_dict(torch.load(tcdecoder_path), strict=False)
            print(f"TCDecoder loading info: {mis}")
        
    elif pipeline_type == "tiny-long":
        pipe = FlashVSRTinyLongPipeline.from_model_manager(mm, device=device)
        
        # Setup LQ projection
        pipe.denoising_model().LQ_proj_in = Causal_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1).to(device, dtype=dtype)
        
        LQ_proj_in_path = os.path.join(model_dir, "LQ_proj_in.ckpt")
        if os.path.exists(LQ_proj_in_path):
            pipe.denoising_model().LQ_proj_in.load_state_dict(
                torch.load(LQ_proj_in_path, map_location="cpu"), strict=True
            )
        
        pipe.denoising_model().LQ_proj_in.to(device)
        
        # Setup TCDecoder
        multi_scale_channels = [512, 256, 128, 128]
        pipe.TCDecoder = build_tcdecoder(new_channels=multi_scale_channels, new_latent_channels=16+768)
        tcdecoder_path = os.path.join(model_dir, "TCDecoder.ckpt")
        if os.path.exists(tcdecoder_path):
            mis = pipe.TCDecoder.load_state_dict(torch.load(tcdecoder_path), strict=False)
            print(f"TCDecoder loading info: {mis}")
    else:
        raise ValueError(f"Unknown pipeline type: {pipeline_type}")
    
    # Finalize pipeline setup
    pipe.to(device)
    pipe.enable_vram_management(num_persistent_param_in_dit=None)
    pipe.init_cross_kv()
    pipe.load_models_to_device(["dit", "vae"])
    
    return pipe


# ==================== Main CLI ====================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="FlashVSR CLI - Video Super-Resolution Inference Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default settings
  python flashvsr_cli.py input.mp4

  # Use Tiny pipeline with v1.1 model
  python flashvsr_cli.py input.mp4 --pipeline tiny --version v1.1

  # Custom output path and scale
  python flashvsr_cli.py input.mp4 -o output.mp4 --scale 4.0

  # Process image directory
  python flashvsr_cli.py ./images/ -o output.mp4

  # Customize inference parameters
  python flashvsr_cli.py input.mp4 --sparse-ratio 1.5 --local-range 9 --tiled

  # Use CPU (slower, for testing)
  python flashvsr_cli.py input.mp4 --device cpu
        """
    )
    
    # Required arguments
    parser.add_argument(
        "input",
        type=str,
        help="Input video file or directory containing images"
    )
    
    # Output
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output video path (default: auto-generated in results/ directory)"
    )
    
    # Pipeline selection
    parser.add_argument(
        "--pipeline",
        type=str,
        choices=["full", "tiny", "tiny-long"],
        default="full",
        help="Pipeline type: 'full' (best quality), 'tiny' (faster), 'tiny-long' (for long videos) (default: full)"
    )
    
    parser.add_argument(
        "--version",
        type=str,
        choices=["v1", "v1.1"],
        default="v1.1",
        help="Model version: 'v1' or 'v1.1' (default: v1.1)"
    )
    
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Custom model directory path (default: auto-detect based on version)"
    )
    
    # Inference parameters
    parser.add_argument(
        "--scale",
        type=float,
        default=4.0,
        help="Super-resolution scale factor (default: 4.0, recommended: 4.0)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility (default: 0)"
    )
    
    parser.add_argument(
        "--sparse-ratio",
        type=float,
        default=2.0,
        help="Sparse attention ratio. Recommended: 1.5 (faster) or 2.0 (more stable) (default: 2.0)"
    )
    
    parser.add_argument(
        "--local-range",
        type=int,
        default=11,
        choices=[9, 11],
        help="Local attention range. 9 → sharper details, 11 → more stable (default: 11)"
    )
    
    parser.add_argument(
        "--tiled",
        action="store_true",
        help="Enable tiling for lower VRAM usage (slower but uses less memory)"
    )
    
    parser.add_argument(
        "--color-fix",
        action="store_true",
        default=True,
        help="Enable color fix (default: True)"
    )
    
    parser.add_argument(
        "--no-color-fix",
        dest="color_fix",
        action="store_false",
        help="Disable color fix"
    )
    
    # Output quality
    parser.add_argument(
        "--quality",
        type=int,
        default=6,
        choices=range(1, 11),
        metavar="[1-10]",
        help="Output video quality (1=lowest, 10=highest) (default: 6)"
    )
    
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Output FPS (default: use input FPS for videos, 30 for image sequences)"
    )
    
    # Device and dtype
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
        choices=["cuda", "cpu", "mps"],
        help="Device to use: 'cuda', 'mps' (Mac), or 'cpu' (default: auto-detect)"
    )
    
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Data type: 'bfloat16', 'float16', or 'float32' (default: bfloat16)"
    )
    
    return parser.parse_args()


def get_dtype(dtype_str: str):
    """Convert dtype string to torch dtype."""
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return dtype_map.get(dtype_str, torch.bfloat16)


def main():
    """Main CLI entry point."""
    args = parse_args()
    
    # Validate input
    if not os.path.exists(args.input):
        print(f"Error: Input path does not exist: {args.input}")
        sys.exit(1)
    
    # Determine output path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.output is None:
        input_basename = os.path.basename(args.input.rstrip('/'))
        input_name = os.path.splitext(input_basename)[0]
        if input_name.startswith('.'):
            input_name = "output"
        
        # Try to use examples/WanVSR/results if it exists, otherwise use ./results
        output_dir = os.path.join(script_dir, "examples", "WanVSR", "results")
        if not os.path.exists(output_dir):
            output_dir = os.path.join(script_dir, "results")
        if not os.path.exists(output_dir):
            output_dir = "./results"
        os.makedirs(output_dir, exist_ok=True)
        
        pipeline_suffix = args.pipeline.replace("-", "_")
        version_suffix = args.version.replace(".", "_")
        args.output = os.path.join(
            output_dir,
            f"FlashVSR_{version_suffix}_{pipeline_suffix}_{input_name}_seed{args.seed}.mp4"
        )
    
    # Convert dtype
    dtype = get_dtype(args.dtype)
    
    # Check device availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        args.device = "cpu"
    elif args.device == "mps" and not torch.backends.mps.is_available():
        print("Warning: MPS not available, falling back to CPU")
        args.device = "cpu"
    
    print(f"\n{'='*60}")
    print(f"FlashVSR Inference")
    print(f"{'='*60}")
    print(f"Input:        {args.input}")
    print(f"Output:       {args.output}")
    print(f"Pipeline:     {args.pipeline}")
    print(f"Version:      {args.version}")
    print(f"Scale:        {args.scale}x")
    print(f"Seed:         {args.seed}")
    print(f"Sparse Ratio: {args.sparse_ratio}")
    print(f"Local Range:  {args.local_range}")
    print(f"Tiled:        {args.tiled}")
    print(f"Color Fix:    {args.color_fix}")
    print(f"Device:       {args.device}")
    print(f"DTYPE:        {args.dtype}")
    print(f"{'='*60}\n")
    
    try:
        # Initialize pipeline
        print("Initializing pipeline...")
        pipe = init_pipeline(
            pipeline_type=args.pipeline,
            model_version=args.version,
            model_dir=args.model_dir,
            dtype=dtype,
            device=args.device
        )
        print("Pipeline initialized successfully.\n")
        
        # Prepare input
        print("Preparing input...")
        LQ, th, tw, F, fps = prepare_input_tensor(
            args.input,
            scale=args.scale,
            dtype=dtype,
            device=args.device
        )
        
        # Use custom FPS if provided
        if args.fps is not None:
            fps = args.fps
        
        print(f"\nRunning inference...")
        
        # Clear cache before inference
        # Clear cache before inference
        if args.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        elif args.device == "mps":
            torch.mps.empty_cache()
        
        # Run inference
        video = pipe(
            prompt="",
            negative_prompt="",
            cfg_scale=1.0,
            num_inference_steps=1,
            seed=args.seed,
            tiled=args.tiled,
            LQ_video=LQ,
            num_frames=F,
            height=th,
            width=tw,
            is_full_block=False,
            if_buffer=True,
            topk_ratio=args.sparse_ratio * 768 * 1280 / (th * tw),
            kv_ratio=3.0,
            local_range=args.local_range,
            color_fix=args.color_fix,
        )
        
        # Convert and save
        print(f"\nSaving output video...")
        video_frames = tensor2video(video)
        save_video(video_frames, args.output, fps=fps, quality=args.quality)
        
        print(f"\n{'='*60}")
        print(f"✓ Success! Output saved to: {args.output}")
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"✗ Error: {str(e)}")
        print(f"{'='*60}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

