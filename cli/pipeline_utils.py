#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLI compatibility wrapper for pipeline initialization.

The implementation now lives in flashvsr.pipeline_utils to keep
shared logic within the library package.
"""

from flashvsr.pipeline_utils import init_pipeline

__all__ = ["init_pipeline"]
