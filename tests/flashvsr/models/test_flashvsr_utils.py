"""Tests for flashvsr/models/flashvsr_utils.py"""

import pytest
import torch
import torch.nn as nn

from flashvsr.models.flashvsr_utils import (
    RMS_norm,
    CausalConv3d,
    PixelShuffle3d,
    Buffer_LQ4x_Proj,
    Causal_LQ4x_Proj,
    CACHE_T,
)


@pytest.mark.unit
class TestRMSNorm:
    """Tests for RMS_norm class."""

    def test_rms_norm_forward_channel_first(self):
        """Test RMS_norm forward pass with channel_first=True."""
        # Arrange
        dim = 64
        norm = RMS_norm(dim, channel_first=True, images=True)
        x = torch.randn(2, dim, 32, 32)  # (B, C, H, W)
        
        # Act
        result = norm(x)
        
        # Assert
        assert result.shape == x.shape
        assert result.dtype == x.dtype

    def test_rms_norm_forward_not_channel_first(self):
        """Test RMS_norm forward pass with channel_first=False."""
        # Arrange
        dim = 64
        norm = RMS_norm(dim, channel_first=False, images=True)
        x = torch.randn(2, 32, 32, dim)  # (B, H, W, C)
        
        # Act
        result = norm(x)
        
        # Assert
        assert result.shape == x.shape

    def test_rms_norm_forward_not_images(self):
        """Test RMS_norm forward pass with images=False."""
        # Arrange
        dim = 64
        norm = RMS_norm(dim, channel_first=True, images=False)
        x = torch.randn(2, dim, 8, 8, 8)  # (B, C, F, H, W)
        
        # Act
        result = norm(x)
        
        # Assert
        assert result.shape == x.shape

    def test_rms_norm_with_bias(self):
        """Test RMS_norm with bias enabled."""
        # Arrange
        dim = 64
        norm = RMS_norm(dim, channel_first=True, images=True, bias=True)
        x = torch.randn(2, dim, 32, 32)
        
        # Act
        result = norm(x)
        
        # Assert
        assert result.shape == x.shape
        assert norm.bias is not None


@pytest.mark.unit
class TestCausalConv3d:
    """Tests for CausalConv3d class."""

    def test_causal_conv3d_forward(self):
        """Test CausalConv3d forward pass."""
        # Arrange
        conv = CausalConv3d(3, 16, kernel_size=3, padding=1)
        x = torch.randn(2, 3, 8, 32, 32)  # (B, C, F, H, W)
        
        # Act
        result = conv(x)
        
        # Assert
        assert result.shape[0] == x.shape[0]  # Batch size preserved
        assert result.shape[1] == 16  # Output channels
        assert result.shape[2] == x.shape[2]  # Temporal dimension preserved

    def test_causal_conv3d_with_cache(self):
        """Test CausalConv3d forward pass with cache."""
        # Arrange
        conv = CausalConv3d(3, 16, kernel_size=3, padding=1)
        x = torch.randn(2, 3, 4, 32, 32)
        cache_x = torch.randn(2, 3, CACHE_T, 32, 32)
        
        # Act
        result = conv(x, cache_x)
        
        # Assert
        assert result.shape[0] == x.shape[0]
        assert result.shape[1] == 16


@pytest.mark.unit
class TestPixelShuffle3d:
    """Tests for PixelShuffle3d class."""

    def test_pixel_shuffle3d_forward(self):
        """Test PixelShuffle3d forward pass."""
        # Arrange
        ff, hh, ww = 1, 2, 2
        shuffle = PixelShuffle3d(ff, hh, ww)
        # Input: (B, C, F*ff, H*hh, W*ww)
        x = torch.randn(2, 16, 4, 64, 64)  # After shuffle: (B, C*ff*hh*ww, F, H, W)
        
        # Act
        result = shuffle(x)
        
        # Assert
        assert result.shape[0] == x.shape[0]  # Batch preserved
        assert result.shape[2] == x.shape[2] // ff  # Temporal dimension
        assert result.shape[3] == x.shape[3] // hh  # Height
        assert result.shape[4] == x.shape[4] // ww  # Width
        assert result.shape[1] == x.shape[1] * ff * hh * ww  # Channels expanded


@pytest.mark.unit
class TestBufferLQ4xProj:
    """Tests for Buffer_LQ4x_Proj class."""

    def test_buffer_lq4x_proj_init(self):
        """Test Buffer_LQ4x_Proj initialization."""
        # Arrange & Act
        proj = Buffer_LQ4x_Proj(in_dim=16, out_dim=512, layer_num=30)
        
        # Assert
        assert proj.ff == 1
        assert proj.hh == 16
        assert proj.ww == 16
        assert proj.layer_num == 30
        assert len(proj.linear_layers) == 30

    def test_buffer_lq4x_proj_clear_cache(self):
        """Test Buffer_LQ4x_Proj clear_cache method."""
        # Arrange
        proj = Buffer_LQ4x_Proj(in_dim=16, out_dim=512, layer_num=30)
        proj.cache = {'conv1': torch.randn(1, 1, 1, 1, 1), 'conv2': torch.randn(1, 1, 1, 1, 1)}
        proj.clip_idx = 5
        
        # Act
        proj.clear_cache()
        
        # Assert
        assert proj.cache['conv1'] is None
        assert proj.cache['conv2'] is None
        assert proj.clip_idx == 0

    def test_buffer_lq4x_proj_forward(self):
        """Test Buffer_LQ4x_Proj forward pass."""
        # Arrange
        proj = Buffer_LQ4x_Proj(in_dim=16, out_dim=512, layer_num=30)
        # Input needs enough frames: iter_ = 1 + (t - 1) // 4, and we need iter_ >= 2
        # because first iteration is skipped. So we need t >= 5 frames
        # Pixel shuffle expects (B, C, F*ff, H*hh, W*ww) = (B, 16, 5, 256, 256)
        video = torch.randn(1, 16, 5, 256, 256)
        
        # Act
        outputs = proj(video)
        
        # Assert
        assert isinstance(outputs, list)
        assert len(outputs) == 30
        assert all(isinstance(out, torch.Tensor) for out in outputs)

    def test_buffer_lq4x_proj_stream_forward_first_clip(self):
        """Test Buffer_LQ4x_Proj stream_forward on first clip."""
        # Arrange
        proj = Buffer_LQ4x_Proj(in_dim=16, out_dim=512, layer_num=30)
        proj.clear_cache()  # Initialize cache
        video_clip = torch.randn(1, 16, 4, 256, 256)
        
        # Act
        result = proj.stream_forward(video_clip)
        
        # Assert
        assert result is None
        assert proj.clip_idx == 1

    def test_buffer_lq4x_proj_stream_forward_subsequent_clips(self):
        """Test Buffer_LQ4x_Proj stream_forward on subsequent clips."""
        # Arrange
        proj = Buffer_LQ4x_Proj(in_dim=16, out_dim=512, layer_num=30)
        proj.clear_cache()  # Initialize cache
        video_clip = torch.randn(1, 16, 4, 256, 256)
        
        # First clip (initializes cache)
        proj.stream_forward(video_clip)
        
        # Act - second clip
        result = proj.stream_forward(video_clip)
        
        # Assert
        assert result is not None
        assert isinstance(result, list)
        assert len(result) == 30
        assert proj.clip_idx == 2


@pytest.mark.unit
class TestCausalLQ4xProj:
    """Tests for Causal_LQ4x_Proj class."""

    def test_causal_lq4x_proj_init(self):
        """Test Causal_LQ4x_Proj initialization."""
        # Arrange & Act
        proj = Causal_LQ4x_Proj(in_dim=16, out_dim=512, layer_num=30)
        
        # Assert
        assert proj.ff == 1
        assert proj.hh == 16
        assert proj.ww == 16
        assert proj.layer_num == 30
        assert len(proj.linear_layers) == 30

    def test_causal_lq4x_proj_clear_cache(self):
        """Test Causal_LQ4x_Proj clear_cache method."""
        # Arrange
        proj = Causal_LQ4x_Proj(in_dim=16, out_dim=512, layer_num=30)
        proj.cache = {'conv1': torch.randn(1, 1, 1, 1, 1), 'conv2': torch.randn(1, 1, 1, 1, 1)}
        proj.clip_idx = 5
        
        # Act
        proj.clear_cache()
        
        # Assert
        assert proj.cache['conv1'] is None
        assert proj.cache['conv2'] is None
        assert proj.clip_idx == 0

    def test_causal_lq4x_proj_forward(self):
        """Test Causal_LQ4x_Proj forward pass."""
        # Arrange
        proj = Causal_LQ4x_Proj(in_dim=16, out_dim=512, layer_num=30)
        # Input needs enough frames: iter_ = 1 + (t - 1) // 4, and we need iter_ >= 2
        # because first iteration is skipped. So we need t >= 5 frames
        video = torch.randn(1, 16, 5, 256, 256)
        
        # Act
        outputs = proj(video)
        
        # Assert
        assert isinstance(outputs, list)
        assert len(outputs) == 30
        assert all(isinstance(out, torch.Tensor) for out in outputs)

    def test_causal_lq4x_proj_stream_forward_first_clip(self):
        """Test Causal_LQ4x_Proj stream_forward on first clip."""
        # Arrange
        proj = Causal_LQ4x_Proj(in_dim=16, out_dim=512, layer_num=30)
        proj.clear_cache()  # Initialize cache
        video_clip = torch.randn(1, 16, 4, 256, 256)
        
        # Act
        result = proj.stream_forward(video_clip)
        
        # Assert
        assert result is None
        assert proj.clip_idx == 1

    def test_causal_lq4x_proj_stream_forward_subsequent_clips(self):
        """Test Causal_LQ4x_Proj stream_forward on subsequent clips."""
        # Arrange
        proj = Causal_LQ4x_Proj(in_dim=16, out_dim=512, layer_num=30)
        proj.clear_cache()  # Initialize cache
        video_clip = torch.randn(1, 16, 4, 256, 256)
        
        # First clip (initializes cache)
        proj.stream_forward(video_clip)
        
        # Act - second clip
        result = proj.stream_forward(video_clip)
        
        # Assert
        assert result is not None
        assert isinstance(result, list)
        assert len(result) == 30
        assert proj.clip_idx == 2

