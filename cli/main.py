#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FlashVSR CLI Tool
A command-line interface for running FlashVSR video super-resolution inference.
"""

import sys
import argparse
from cli.commands import setup_command, inference_command


def parse_args():
    """Parse command-line arguments."""
    # Check if first arg is "setup" command
    if len(sys.argv) > 1 and sys.argv[1] == "setup":
        return parse_setup_args()
    else:
        return parse_inference_args()


def parse_setup_args():
    """Parse arguments for setup command."""
    parser = argparse.ArgumentParser(
        prog="flashvsr setup",
        description="Download FlashVSR model weights from Hugging Face",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download v1.1 model (recommended)
  flashvsr setup

  # Download v1 model
  flashvsr setup --version v1

  # Download models for specific pipeline
  flashvsr setup --pipeline full
  flashvsr setup --pipeline tiny
        """
    )
    parser.add_argument(
        "--version",
        type=str,
        choices=["v1", "v1.1"],
        default="v1.1",
        help="Model version to download (default: v1.1)"
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        choices=["base", "full", "tiny", "tiny-long"],
        default="base",
        help="Pipeline type to determine which files to verify (default: base)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )
    args = parser.parse_args()
    args.command = "setup"
    return args


def parse_inference_args():
    """Parse arguments for inference command."""
    parser = argparse.ArgumentParser(
        description="FlashVSR CLI - Video Super-Resolution Inference Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default settings
  flashvsr input.mp4

  # Use Tiny pipeline with v1.1 model
  flashvsr input.mp4 --pipeline tiny --version v1.1

  # Custom output path and scale
  flashvsr input.mp4 -o output.mp4 --scale 4.0

  # Process image directory
  flashvsr ./images/ -o output.mp4

  # Customize inference parameters
  flashvsr input.mp4 --sparse-ratio 1.5 --local-range 9 --tiled

  # Use CPU (slower, for testing)
  flashvsr input.mp4 --device cpu
        """
    )
    
    # Required arguments for inference
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
    
    args = parser.parse_args()
    args.command = "infer"
    return args


def main():
    """Main CLI entry point."""
    args = parse_args()
    
    # Handle setup command
    if args.command == "setup":
        sys.exit(setup_command(args))
    
    # Handle inference command (default)
    if args.command == "infer":
        inference_command(args)


if __name__ == "__main__":
    main()
