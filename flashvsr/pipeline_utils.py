#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline setup helpers shared by the library and CLI.
"""

import os
from typing import Optional
import torch
from diffsynth import ModelManager
from flashvsr.pipelines.flashvsr_full import FlashVSRFullPipeline
from flashvsr.pipelines.flashvsr_tiny import FlashVSRTinyPipeline
from flashvsr.pipelines.flashvsr_tiny_long import FlashVSRTinyLongPipeline
from flashvsr.models.flashvsr_utils import Causal_LQ4x_Proj, Buffer_LQ4x_Proj
from flashvsr.models.flashvsr_tcdecoder import build_tcdecoder
from flashvsr.model_downloader import download_models_for_pipeline


def _resolve_model_dir(pipeline_type: str, model_version: str, model_dir: Optional[str]) -> str:
    """Resolve the model directory, falling back to common defaults."""
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if model_dir is None:
        project_root = os.path.dirname(script_dir)
        if model_version == "v1.1":
            model_dir = os.path.join(project_root, "models", "FlashVSR-v1.1")
        else:
            model_dir = os.path.join(project_root, "models", "FlashVSR")

        if not os.path.exists(model_dir):
            old_dir = os.path.join(os.getcwd(), "examples", "WanVSR", "FlashVSR-v1.1" if model_version == "v1.1" else "FlashVSR")
            if os.path.exists(old_dir):
                model_dir = old_dir
            else:
                old_dir = os.path.join(project_root, "examples", "WanVSR", "FlashVSR-v1.1" if model_version == "v1.1" else "FlashVSR")
                if os.path.exists(old_dir):
                    model_dir = old_dir
    else:
        model_dir = os.path.expanduser(model_dir)
        if not os.path.isabs(model_dir):
            abs_path = os.path.join(script_dir, model_dir)
            if os.path.exists(abs_path):
                model_dir = abs_path
            else:
                model_dir = os.path.abspath(model_dir)

    if not os.path.exists(model_dir):
        raise FileNotFoundError(model_dir)

    return model_dir


def init_pipeline(
    pipeline_type: str,
    model_version: str,
    model_dir: Optional[str] = None,
    dtype=torch.bfloat16,
    device="cuda",
    auto_download: bool = True,
):
    """Initialize the appropriate FlashVSR pipeline."""
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if model_dir is None or not os.path.exists(os.path.expanduser(model_dir)):
        try:
            model_dir = _resolve_model_dir(pipeline_type, model_version, model_dir)
        except FileNotFoundError as e:
            missing_dir = e.args[0] if e.args else model_dir
            if auto_download:
                print(f"Model directory not found: {missing_dir}")
                print(f"Auto-downloading FlashVSR {model_version} model...")
                project_root = os.path.dirname(script_dir)
                base_dir = os.path.join(project_root, "models")
                downloaded_dir = download_models_for_pipeline(
                    version=model_version,
                    pipeline_type=pipeline_type,
                    base_dir=base_dir,
                    auto_download=True,
                    quiet=False,
                )
                if downloaded_dir:
                    model_dir = str(downloaded_dir)
                else:
                    model_dir = missing_dir
            else:
                raise FileNotFoundError(
                    f"Model directory not found: {missing_dir}\n"
                    f"Run 'flashvsr setup --version {model_version}' to download models."
                )

    if model_dir is None or not os.path.exists(os.path.expanduser(model_dir)):
        raise FileNotFoundError(
            f"Model directory not found: {model_dir}\n"
            f"Run 'flashvsr setup --version {model_version}' to download models."
        )

    print(f"Using model directory: {model_dir}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.current_device()} - {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print("GPU: Not available (using CPU)")

    mm = ModelManager(torch_dtype=dtype, device="cpu")

    diffusion_model_path = os.path.join(model_dir, "diffusion_pytorch_model_streaming_dmd.safetensors")
    if not os.path.exists(diffusion_model_path):
        raise FileNotFoundError(f"Diffusion model not found: {diffusion_model_path}")
    mm.load_models([diffusion_model_path])

    if pipeline_type == "full":
        pipe = FlashVSRFullPipeline.from_model_manager(mm, device=device)

        vae_path = os.path.join(model_dir, "Wan2.1_VAE.pth")
        if not os.path.exists(vae_path):
            raise FileNotFoundError(f"VAE model not found: {vae_path}")
        mm.load_models([vae_path])

        if model_version == "v1.1":
            pipe.denoising_model().LQ_proj_in = Causal_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1).to(device, dtype=dtype)
        else:
            pipe.denoising_model().LQ_proj_in = Buffer_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1).to(device, dtype=dtype)

        LQ_proj_in_path = os.path.join(model_dir, "LQ_proj_in.ckpt")
        if os.path.exists(LQ_proj_in_path):
            pipe.denoising_model().LQ_proj_in.load_state_dict(torch.load(LQ_proj_in_path, map_location="cpu"), strict=True)

        pipe.denoising_model().LQ_proj_in.to(device)
        pipe.vae.model.encoder = None
        pipe.vae.model.conv1 = None

    elif pipeline_type == "tiny":
        pipe = FlashVSRTinyPipeline.from_model_manager(mm, device=device)

        pipe.denoising_model().LQ_proj_in = Causal_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1).to(device, dtype=dtype)

        LQ_proj_in_path = os.path.join(model_dir, "LQ_proj_in.ckpt")
        if os.path.exists(LQ_proj_in_path):
            pipe.denoising_model().LQ_proj_in.load_state_dict(torch.load(LQ_proj_in_path, map_location="cpu"), strict=True)

        pipe.denoising_model().LQ_proj_in.to(device)

        multi_scale_channels = [512, 256, 128, 128]
        pipe.TCDecoder = build_tcdecoder(new_channels=multi_scale_channels, new_latent_channels=16 + 768)
        tcdecoder_path = os.path.join(model_dir, "TCDecoder.ckpt")
        if os.path.exists(tcdecoder_path):
            mis = pipe.TCDecoder.load_state_dict(torch.load(tcdecoder_path), strict=False)
            print(f"TCDecoder loading info: {mis}")

    elif pipeline_type == "tiny-long":
        pipe = FlashVSRTinyLongPipeline.from_model_manager(mm, device=device)

        pipe.denoising_model().LQ_proj_in = Causal_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1).to(device, dtype=dtype)

        LQ_proj_in_path = os.path.join(model_dir, "LQ_proj_in.ckpt")
        if os.path.exists(LQ_proj_in_path):
            pipe.denoising_model().LQ_proj_in.load_state_dict(torch.load(LQ_proj_in_path, map_location="cpu"), strict=True)

        pipe.denoising_model().LQ_proj_in.to(device)

        multi_scale_channels = [512, 256, 128, 128]
        pipe.TCDecoder = build_tcdecoder(new_channels=multi_scale_channels, new_latent_channels=16 + 768)
        tcdecoder_path = os.path.join(model_dir, "TCDecoder.ckpt")
        if os.path.exists(tcdecoder_path):
            mis = pipe.TCDecoder.load_state_dict(torch.load(tcdecoder_path), strict=False)
            print(f"TCDecoder loading info: {mis}")
    else:
        raise ValueError(f"Unknown pipeline type: {pipeline_type}")

    pipe.to(device)
    pipe.enable_vram_management(num_persistent_param_in_dit=None)
    pipe.init_cross_kv()
    pipe.load_models_to_device(["dit", "vae"])

    return pipe


__all__ = ["init_pipeline"]
