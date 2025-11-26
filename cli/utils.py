#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLI compatibility wrappers for shared FlashVSR utilities.

All logic lives in flashvsr.utils; this module simply re-exports
the helpers for existing CLI imports.
"""

from flashvsr.utils import (
    compute_scaled_and_target_dims,
    get_dtype,
    is_video,
    largest_8n1_leq,
    list_images_natural,
    natural_key,
    pil_to_tensor_neg1_1,
    prepare_input_tensor,
    save_video,
    tensor2video,
    upscale_then_center_crop,
)

__all__ = [
    "tensor2video",
    "natural_key",
    "list_images_natural",
    "largest_8n1_leq",
    "is_video",
    "pil_to_tensor_neg1_1",
    "save_video",
    "compute_scaled_and_target_dims",
    "upscale_then_center_crop",
    "prepare_input_tensor",
    "get_dtype",
]
