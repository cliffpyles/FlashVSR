#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLI Utility Functions
General utility functions for the FlashVSR CLI.
"""

import os
import re
import numpy as np
from PIL import Image
import imageio
from tqdm import tqdm
import torch
from einops import rearrange


def tensor2video(frames: torch.Tensor):
    """Convert tensor to video frames."""
    frames = rearrange(frames, "C T H W -> T H W C")
    frames = ((frames.float() + 1) * 127.5).clip(0, 255)
    frames = frames.detach().cpu()
    try:
        frames_np = frames.numpy().astype(np.uint8)
    except RuntimeError:
        # Fallback for NumPy compatibility issues (e.g., NumPy 2.x with PyTorch compiled for 1.x)
        # Convert via list intermediate
        frames_np = np.array(frames.tolist(), dtype=np.uint8)
    frames = [Image.fromarray(frame) for frame in frames_np]
    return frames


def natural_key(name: str):
    """Natural sort key for filenames.
    
    Splits filename into alternating non-digit and digit parts.
    Non-digit parts are compared as strings, digit parts as integers.
    This ensures natural sorting: "test.jpg" < "test2.jpg" < "test10.jpg"
    
    Uses tuples (type, value) where type 0=string, 1=int to ensure
    strings compare before integers when prefixes match.
    """
    basename = os.path.basename(name).lower()
    # Split filename into name and extension
    name_part, ext = os.path.splitext(basename)
    
    # Split name part into digits and non-digits
    name_parts = re.split(r'([0-9]+)', name_part)
    
    # Convert to comparable format: (type, value) where type 0=string, 1=int
    key = []
    for t in name_parts:
        if t:  # Skip empty strings
            if t.isdigit():
                key.append((1, int(t)))  # Type 1 for integers
            else:
                key.append((0, t))  # Type 0 for strings
    
    # Add extension as a string part
    if ext:
        key.append((0, ext))
    
    return key


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
    img_array = np.asarray(img, dtype=np.uint8)
    try:
        t = torch.from_numpy(img_array).to(device=device, dtype=torch.float32)  # HWC
    except RuntimeError:
        # Fallback for NumPy compatibility issues
        t = torch.tensor(img_array, device=device, dtype=torch.float32)  # HWC
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


def get_dtype(dtype_str: str):
    """Convert dtype string to torch dtype."""
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return dtype_map.get(dtype_str, torch.bfloat16)

