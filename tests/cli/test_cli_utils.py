"""Tests for cli/utils.py"""

import os
import tempfile
import pytest
import torch
import numpy as np
from PIL import Image
from cli.utils import (
    tensor2video,
    natural_key,
    list_images_natural,
    largest_8n1_leq,
    is_video,
    pil_to_tensor_neg1_1,
    compute_scaled_and_target_dims,
    upscale_then_center_crop,
    get_dtype,
)


@pytest.mark.unit
class TestTensor2Video:
    """Tests for tensor2video function."""
    
    def test_tensor2video_shape(self):
        """Test tensor2video converts tensor to list of PIL Images."""
        # Create a dummy tensor: C T H W = 3 5 64 64
        frames = torch.randn(3, 5, 64, 64)
        result = tensor2video(frames)
        
        assert len(result) == 5
        assert all(isinstance(img, Image.Image) for img in result)
        assert result[0].size == (64, 64)
    
    def test_tensor2video_values(self):
        """Test tensor2video converts values correctly."""
        # Create tensor in [-1, 1] range
        frames = torch.ones(3, 2, 32, 32)  # All ones should map to 255
        result = tensor2video(frames)
        
        # Check that values are in [0, 255] range
        img_array = np.array(result[0])
        assert img_array.min() >= 0
        assert img_array.max() <= 255


@pytest.mark.unit
class TestNaturalKey:
    """Tests for natural_key function."""
    
    def test_natural_key_simple(self):
        """Test natural key for simple strings."""
        assert natural_key("test.jpg") < natural_key("test2.jpg")
        assert natural_key("img1.jpg") < natural_key("img10.jpg")
    
    def test_natural_key_numeric(self):
        """Test natural key handles numbers correctly."""
        files = ["file10.jpg", "file2.jpg", "file1.jpg"]
        sorted_files = sorted(files, key=natural_key)
        assert sorted_files == ["file1.jpg", "file2.jpg", "file10.jpg"]


@pytest.mark.unit
class TestListImagesNatural:
    """Tests for list_images_natural function."""
    
    def test_list_images_natural(self):
        """Test listing images with natural sorting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test images
            for i in [10, 2, 1]:
                img = Image.new('RGB', (100, 100), color='red')
                img.save(os.path.join(tmpdir, f"img{i}.png"))
            
            result = list_images_natural(tmpdir)
            filenames = [os.path.basename(f) for f in result]
            assert filenames == ["img1.png", "img2.png", "img10.png"]
    
    def test_list_images_natural_case_insensitive(self):
        """Test that list_images_natural handles different cases."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img = Image.new('RGB', (100, 100), color='red')
            img.save(os.path.join(tmpdir, "test.PNG"))
            img.save(os.path.join(tmpdir, "test.jpg"))
            img.save(os.path.join(tmpdir, "test.JPEG"))
            
            result = list_images_natural(tmpdir)
            assert len(result) == 3


@pytest.mark.unit
class TestLargest8n1Leq:
    """Tests for largest_8n1_leq function."""
    
    def test_largest_8n1_leq_valid(self):
        """Test largest_8n1_leq with valid inputs."""
        assert largest_8n1_leq(9) == 9  # 8*1 + 1
        assert largest_8n1_leq(17) == 17  # 8*2 + 1
        assert largest_8n1_leq(10) == 9  # Should round down
        assert largest_8n1_leq(25) == 25  # 8*3 + 1
    
    def test_largest_8n1_leq_edge_cases(self):
        """Test largest_8n1_leq with edge cases."""
        assert largest_8n1_leq(0) == 0
        assert largest_8n1_leq(1) == 1
        assert largest_8n1_leq(8) == 1  # Should round down to 1


@pytest.mark.unit
class TestIsVideo:
    """Tests for is_video function."""
    
    def test_is_video_valid_extensions(self):
        """Test is_video recognizes valid video extensions."""
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            assert is_video(f.name)
            os.unlink(f.name)
        
        with tempfile.NamedTemporaryFile(suffix='.MOV', delete=False) as f:
            assert is_video(f.name)
            os.unlink(f.name)
    
    def test_is_video_invalid_extensions(self):
        """Test is_video rejects non-video files."""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            assert not is_video(f.name)
            os.unlink(f.name)
    
    def test_is_video_nonexistent(self):
        """Test is_video returns False for nonexistent files."""
        assert not is_video("/nonexistent/path/video.mp4")


@pytest.mark.unit
class TestPilToTensorNeg1_1:
    """Tests for pil_to_tensor_neg1_1 function."""
    
    def test_pil_to_tensor_neg1_1_shape(self):
        """Test pil_to_tensor_neg1_1 converts PIL image correctly."""
        img = Image.new('RGB', (64, 64), color='red')
        tensor = pil_to_tensor_neg1_1(img, dtype=torch.float32, device='cpu')
        
        assert tensor.shape == (3, 64, 64)  # CHW format
        assert tensor.dtype == torch.float32
    
    def test_pil_to_tensor_neg1_1_range(self):
        """Test pil_to_tensor_neg1_1 values are in [-1, 1] range."""
        # White image should be close to 1
        img = Image.new('RGB', (32, 32), color='white')
        tensor = pil_to_tensor_neg1_1(img, dtype=torch.float32, device='cpu')
        assert tensor.max() <= 1.0
        assert tensor.min() >= -1.0
        
        # Black image should be close to -1
        img = Image.new('RGB', (32, 32), color='black')
        tensor = pil_to_tensor_neg1_1(img, dtype=torch.float32, device='cpu')
        assert tensor.min() >= -1.0


@pytest.mark.unit
class TestComputeScaledAndTargetDims:
    """Tests for compute_scaled_and_target_dims function."""
    
    def test_compute_scaled_and_target_dims_valid(self):
        """Test compute_scaled_and_target_dims with valid inputs."""
        sW, sH, tW, tH = compute_scaled_and_target_dims(100, 100, scale=4.0, multiple=128)
        
        assert sW == 400
        assert sH == 400
        assert tW == 384  # 400 // 128 * 128 = 3 * 128 = 384
        assert tH == 384
    
    def test_compute_scaled_and_target_dims_invalid_size(self):
        """Test compute_scaled_and_target_dims raises error for invalid size."""
        with pytest.raises(ValueError, match="Invalid original size"):
            compute_scaled_and_target_dims(0, 100, scale=4.0)
        
        with pytest.raises(ValueError, match="Invalid original size"):
            compute_scaled_and_target_dims(100, -1, scale=4.0)
    
    def test_compute_scaled_and_target_dims_invalid_scale(self):
        """Test compute_scaled_and_target_dims raises error for invalid scale."""
        with pytest.raises(ValueError, match="scale must be > 0"):
            compute_scaled_and_target_dims(100, 100, scale=0)
        
        with pytest.raises(ValueError, match="scale must be > 0"):
            compute_scaled_and_target_dims(100, 100, scale=-1)
    
    def test_compute_scaled_and_target_dims_too_small(self):
        """Test compute_scaled_and_target_dims raises error when scaled size is too small."""
        with pytest.raises(ValueError, match="Scaled size too small"):
            compute_scaled_and_target_dims(10, 10, scale=1.0, multiple=128)


@pytest.mark.unit
class TestUpscaleThenCenterCrop:
    """Tests for upscale_then_center_crop function."""
    
    def test_upscale_then_center_crop(self):
        """Test upscale_then_center_crop resizes and crops correctly."""
        img = Image.new('RGB', (100, 100), color='blue')
        result = upscale_then_center_crop(img, scale=4.0, tW=384, tH=384)
        
        assert result.size == (384, 384)
    
    def test_upscale_then_center_crop_invalid_target(self):
        """Test upscale_then_center_crop raises error when target exceeds scaled size."""
        img = Image.new('RGB', (100, 100), color='blue')
        
        with pytest.raises(ValueError, match="Target crop.*exceeds scaled size"):
            upscale_then_center_crop(img, scale=2.0, tW=500, tH=500)


@pytest.mark.unit
class TestGetDtype:
    """Tests for get_dtype function."""
    
    def test_get_dtype_valid(self):
        """Test get_dtype with valid dtype strings."""
        assert get_dtype("bfloat16") == torch.bfloat16
        assert get_dtype("float16") == torch.float16
        assert get_dtype("float32") == torch.float32
    
    def test_get_dtype_invalid_defaults(self):
        """Test get_dtype defaults to bfloat16 for invalid inputs."""
        assert get_dtype("invalid") == torch.bfloat16
        assert get_dtype("") == torch.bfloat16

