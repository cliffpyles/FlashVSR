#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLI Command Handlers
Command execution logic for FlashVSR CLI commands.
"""

import os
import sys
import torch
import traceback
from flashvsr.registry import register_wan_models
from flashvsr.model_downloader import download_model
from .pipeline_utils import init_pipeline
from .utils import (
    prepare_input_tensor,
    tensor2video,
    save_video,
    get_dtype
)

# Register Wan models with diffsynth
register_wan_models()


def setup_command(args):
    """Handle the setup command for downloading models."""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up from cli/ to project root
        project_root = os.path.dirname(script_dir)
        base_dir = os.path.join(project_root, "models")
        
        print(f"\n{'='*60}")
        print(f"FlashVSR Setup - Downloading Model Weights")
        print(f"{'='*60}\n")
        
        model_dir = download_model(
            version=args.version,
            pipeline_type=args.pipeline,
            base_dir=base_dir,
            resume_download=True,
            quiet=args.quiet
        )
        
        print(f"\n{'='*60}")
        print(f"✓ Setup complete! Models available at: {model_dir}")
        print(f"{'='*60}\n")
        
        return 0
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"✗ Setup failed: {str(e)}")
        print(f"{'='*60}\n")
        traceback.print_exc()
        return 1


def inference_command(args):
    """Handle the inference command for video super-resolution."""
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
        
        # Use results directory in project root
        project_root = os.path.dirname(script_dir)  # Go up from cli/ to project root
        output_dir = os.path.join(project_root, "results")
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
        # Initialize pipeline (with auto-download enabled)
        print("Initializing pipeline...")
        pipe = init_pipeline(
            pipeline_type=args.pipeline,
            model_version=args.version,
            model_dir=args.model_dir,
            dtype=dtype,
            device=args.device,
            auto_download=True
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
        traceback.print_exc()
        sys.exit(1)

