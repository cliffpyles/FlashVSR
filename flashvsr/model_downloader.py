#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FlashVSR Model Downloader
Automatically downloads model weights from Hugging Face.
"""

import os
import sys
from pathlib import Path
from typing import Optional, List, Tuple
from huggingface_hub import snapshot_download, hf_hub_download
from tqdm import tqdm


# Model repository mappings
MODEL_REPOS = {
    "v1": "JunhaoZhuang/FlashVSR",
    "v1.1": "JunhaoZhuang/FlashVSR-v1.1",
}

# Required files for each pipeline type
REQUIRED_FILES = {
    "base": [
        "diffusion_pytorch_model_streaming_dmd.safetensors",
        "LQ_proj_in.ckpt",
    ],
    "full": [
        "diffusion_pytorch_model_streaming_dmd.safetensors",
        "LQ_proj_in.ckpt",
        "Wan2.1_VAE.pth",
    ],
    "tiny": [
        "diffusion_pytorch_model_streaming_dmd.safetensors",
        "LQ_proj_in.ckpt",
        "TCDecoder.ckpt",
    ],
    "tiny-long": [
        "diffusion_pytorch_model_streaming_dmd.safetensors",
        "LQ_proj_in.ckpt",
        "TCDecoder.ckpt",
    ],
}


def get_model_dir(version: str, base_dir: Optional[str] = None) -> Path:
    """
    Get the model directory path for a given version.
    
    Args:
        version: Model version ("v1" or "v1.1")
        base_dir: Base directory (defaults to models/ in project root)
    
    Returns:
        Path to the model directory
    """
    if base_dir is None:
        # Get the directory where this module is located (flashvsr/)
        # Go up to project root, then to models/
        project_root = Path(__file__).parent.parent
        base_dir = project_root / "models"
    else:
        base_dir = Path(base_dir)
    
    if version == "v1.1":
        model_dir = base_dir / "FlashVSR-v1.1"
    else:  # v1
        model_dir = base_dir / "FlashVSR"
    
    return model_dir


def check_model_files(model_dir: Path, pipeline_type: str = "base") -> Tuple[bool, List[str]]:
    """
    Check if required model files exist.
    
    Args:
        model_dir: Path to model directory
        pipeline_type: Pipeline type ("base", "full", "tiny", "tiny-long")
    
    Returns:
        Tuple of (all_exist, missing_files)
    """
    if not model_dir.exists():
        return False, REQUIRED_FILES.get(pipeline_type, REQUIRED_FILES["base"])
    
    required = REQUIRED_FILES.get(pipeline_type, REQUIRED_FILES["base"])
    missing = []
    
    for filename in required:
        filepath = model_dir / filename
        if not filepath.exists():
            missing.append(filename)
    
    return len(missing) == 0, missing


def download_model(
    version: str,
    pipeline_type: str = "base",
    base_dir: Optional[str] = None,
    resume_download: bool = True,
    quiet: bool = False
) -> Path:
    """
    Download model weights from Hugging Face.
    
    Args:
        version: Model version ("v1" or "v1.1")
        pipeline_type: Pipeline type to determine which files to check ("base", "full", "tiny", "tiny-long")
        base_dir: Base directory for model storage (defaults to examples/WanVSR)
        resume_download: Whether to resume interrupted downloads
        quiet: If True, suppress progress output
    
    Returns:
        Path to the downloaded model directory
    
    Raises:
        ValueError: If version is invalid
        RuntimeError: If download fails
    """
    if version not in MODEL_REPOS:
        raise ValueError(f"Invalid version: {version}. Must be one of {list(MODEL_REPOS.keys())}")
    
    repo_id = MODEL_REPOS[version]
    model_dir = get_model_dir(version, base_dir)
    
    # Check if models already exist
    all_exist, missing = check_model_files(model_dir, pipeline_type)
    if all_exist:
        if not quiet:
            print(f"✓ Model files already exist: {model_dir}")
        return model_dir
    
    if not quiet:
        print(f"Downloading FlashVSR {version} model from Hugging Face...")
        print(f"Repository: {repo_id}")
        print(f"Target directory: {model_dir}")
        if missing:
            print(f"Missing files: {', '.join(missing)}")
        print()
    
    # Create model directory
    model_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Download all files from the repository
        # snapshot_download downloads the entire repo, which is what we want
        if not quiet:
            print("Downloading model files (this may take a while)...")
        
        downloaded_path = snapshot_download(
            repo_id=repo_id,
            local_dir=str(model_dir),
            local_dir_use_symlinks=False,
            resume_download=resume_download,
        )
        
        # Verify required files exist
        all_exist, still_missing = check_model_files(model_dir, pipeline_type)
        if not all_exist:
            raise RuntimeError(
                f"Download completed but some required files are missing: {', '.join(still_missing)}\n"
                f"Please check the Hugging Face repository: https://huggingface.co/{repo_id}"
            )
        
        if not quiet:
            print(f"\n✓ Successfully downloaded FlashVSR {version} model to: {model_dir}")
            print(f"  Files: {', '.join(REQUIRED_FILES.get(pipeline_type, REQUIRED_FILES['base']))}")
        
        return Path(downloaded_path)
    
    except Exception as e:
        # Clean up partial download on error
        if model_dir.exists() and not any(model_dir.iterdir()):
            try:
                model_dir.rmdir()
            except Exception:
                pass
        
        raise RuntimeError(
            f"Failed to download model from Hugging Face: {str(e)}\n"
            f"Repository: {repo_id}\n"
            f"Please check your internet connection and try again."
        ) from e


def download_models_for_pipeline(
    version: str,
    pipeline_type: str,
    base_dir: Optional[str] = None,
    auto_download: bool = True,
    quiet: bool = False
) -> Optional[Path]:
    """
    Download models if missing, with auto-download option.
    
    Args:
        version: Model version ("v1" or "v1.1")
        pipeline_type: Pipeline type ("full", "tiny", "tiny-long")
        base_dir: Base directory for model storage
        auto_download: If True, automatically download if missing. If False, raise error.
        quiet: If True, suppress progress output
    
    Returns:
        Path to model directory, or None if auto_download is False and models are missing
    
    Raises:
        FileNotFoundError: If models are missing and auto_download is False
    """
    model_dir = get_model_dir(version, base_dir)
    all_exist, missing = check_model_files(model_dir, pipeline_type)
    
    if all_exist:
        return model_dir
    
    if not auto_download:
        raise FileNotFoundError(
            f"Model directory not found: {model_dir}\n"
            f"Missing files: {', '.join(missing)}\n"
            f"Run 'flashvsr setup --version {version}' to download models, "
            f"or set auto_download=True to download automatically."
        )
    
    # Auto-download
    if not quiet:
        print(f"Models not found. Auto-downloading FlashVSR {version}...")
    
    return download_model(version, pipeline_type, base_dir, resume_download=True, quiet=quiet)


if __name__ == "__main__":
    # CLI for standalone usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Download FlashVSR model weights from Hugging Face")
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
        "--base-dir",
        type=str,
        default=None,
        help="Base directory for model storage (default: examples/WanVSR)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )
    
    args = parser.parse_args()
    
    try:
        model_dir = download_model(
            version=args.version,
            pipeline_type=args.pipeline,
            base_dir=args.base_dir,
            resume_download=True,
            quiet=args.quiet
        )
        print(f"\n✓ Setup complete! Models available at: {model_dir}")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        sys.exit(1)

