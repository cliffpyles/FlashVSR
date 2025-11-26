"""Tests for cli/commands.py"""

import os
import sys
import tempfile
import pytest
import torch
from unittest.mock import Mock, patch, MagicMock, mock_open
from argparse import Namespace
from cli.commands import setup_command, inference_command


@pytest.mark.unit
class TestSetupCommand:
    """Tests for setup_command function."""
    
    @patch('cli.commands.download_model')
    @patch('cli.commands.os.path.dirname')
    @patch('cli.commands.os.path.abspath')
    @patch('cli.commands.os.path.join')
    def test_setup_command_success(self, mock_join, mock_abspath, mock_dirname, mock_download):
        """Test setup_command succeeds when download works."""
        mock_abspath.return_value = "/path/to/cli/commands.py"
        mock_dirname.return_value = "/path/to/cli"
        mock_join.return_value = "/path/to/models"
        mock_download.return_value = "/path/to/models/FlashVSR-v1.1"
        
        args = Namespace(
            version="v1.1",
            pipeline="full",
            quiet=False
        )
        
        result = setup_command(args)
        
        assert result == 0
        mock_download.assert_called_once_with(
            version="v1.1",
            pipeline_type="full",
            base_dir="/path/to/models",
            resume_download=True,
            quiet=False
        )
    
    @patch('cli.commands.download_model')
    @patch('cli.commands.os.path.dirname')
    @patch('cli.commands.os.path.abspath')
    @patch('cli.commands.os.path.join')
    def test_setup_command_failure(self, mock_join, mock_abspath, mock_dirname, mock_download):
        """Test setup_command returns 1 when download fails."""
        mock_abspath.return_value = "/path/to/cli/commands.py"
        mock_dirname.return_value = "/path/to/cli"
        mock_join.return_value = "/path/to/models"
        mock_download.side_effect = Exception("Download failed")
        
        args = Namespace(
            version="v1.1",
            pipeline="full",
            quiet=False
        )
        
        result = setup_command(args)
        
        assert result == 1


@pytest.mark.unit
class TestInferenceCommand:
    """Tests for inference_command function."""
    
    @patch('cli.commands.save_video')
    @patch('cli.commands.tensor2video')
    @patch('cli.commands.prepare_input_tensor')
    @patch('cli.commands.init_pipeline')
    @patch('cli.commands.get_dtype')
    @patch('cli.commands.os.path.exists')
    @patch('cli.commands.os.path.dirname')
    @patch('cli.commands.os.path.abspath')
    @patch('cli.commands.os.path.join')
    @patch('cli.commands.os.makedirs')
    def test_inference_command_success(
        self, mock_makedirs, mock_join, mock_abspath, mock_dirname, 
        mock_exists, mock_get_dtype, mock_init_pipeline, 
        mock_prepare_input, mock_tensor2video, mock_save_video
    ):
        """Test inference_command succeeds with valid input."""
        # Setup mocks
        mock_abspath.return_value = "/path/to/cli/commands.py"
        mock_dirname.return_value = "/path/to/cli"
        mock_join.return_value = "/path/to/results"
        mock_exists.return_value = True
        mock_get_dtype.return_value = torch.float32
        
        # Mock pipeline
        mock_pipe = MagicMock()
        mock_video = torch.randn(3, 5, 64, 64)
        mock_pipe.return_value = mock_video
        mock_init_pipeline.return_value = mock_pipe
        
        # Mock input preparation
        mock_prepare_input.return_value = (
            torch.randn(1, 3, 5, 64, 64),  # LQ
            64,  # th
            64,  # tw
            5,   # F
            30   # fps
        )
        
        # Mock video conversion
        mock_frames = [MagicMock() for _ in range(5)]
        mock_tensor2video.return_value = mock_frames
        
        args = Namespace(
            input="/path/to/input.mp4",
            output=None,
            pipeline="full",
            version="v1.1",
            model_dir=None,
            scale=4.0,
            seed=0,
            sparse_ratio=2.0,
            local_range=11,
            tiled=False,
            color_fix=True,
            quality=6,
            fps=None,
            device="cpu",
            dtype="float32"
        )
        
        # Should not raise exception
        inference_command(args)
        
        # Verify calls
        mock_init_pipeline.assert_called_once()
        mock_prepare_input.assert_called_once()
        mock_pipe.assert_called_once()
        mock_tensor2video.assert_called_once()
        mock_save_video.assert_called_once()
    
    @patch('cli.commands.os.path.exists')
    def test_inference_command_invalid_input(self, mock_exists):
        """Test inference_command exits when input doesn't exist."""
        mock_exists.return_value = False
        
        args = Namespace(
            input="/nonexistent/path.mp4",
            output=None,
            pipeline="full",
            version="v1.1",
            model_dir=None,
            scale=4.0,
            seed=0,
            sparse_ratio=2.0,
            local_range=11,
            tiled=False,
            color_fix=True,
            quality=6,
            fps=None,
            device="cpu",
            dtype="float32"
        )
        
        with pytest.raises(SystemExit):
            inference_command(args)
    
    @patch('cli.commands.save_video')
    @patch('cli.commands.tensor2video')
    @patch('cli.commands.prepare_input_tensor')
    @patch('cli.commands.init_pipeline')
    @patch('cli.commands.get_dtype')
    @patch('cli.commands.os.path.exists')
    @patch('cli.commands.os.path.dirname')
    @patch('cli.commands.os.path.abspath')
    @patch('cli.commands.os.path.join')
    @patch('cli.commands.os.makedirs')
    @patch('cli.commands.torch.cuda.is_available')
    def test_inference_command_cuda_fallback(
        self, mock_cuda_available, mock_makedirs, mock_join, 
        mock_abspath, mock_dirname, mock_exists, mock_get_dtype,
        mock_init_pipeline, mock_prepare_input, mock_tensor2video, mock_save_video
    ):
        """Test inference_command falls back to CPU when CUDA unavailable."""
        mock_abspath.return_value = "/path/to/cli/commands.py"
        mock_dirname.return_value = "/path/to/cli"
        mock_join.return_value = "/path/to/results"
        mock_exists.return_value = True
        mock_get_dtype.return_value = torch.float32
        mock_cuda_available.return_value = False
        
        mock_pipe = MagicMock()
        mock_pipe.return_value = torch.randn(3, 5, 64, 64)
        mock_init_pipeline.return_value = mock_pipe
        
        mock_prepare_input.return_value = (
            torch.randn(1, 3, 5, 64, 64),
            64, 64, 5, 30
        )
        mock_tensor2video.return_value = [MagicMock() for _ in range(5)]
        
        args = Namespace(
            input="/path/to/input.mp4",
            output=None,
            pipeline="full",
            version="v1.1",
            model_dir=None,
            scale=4.0,
            seed=0,
            sparse_ratio=2.0,
            local_range=11,
            tiled=False,
            color_fix=True,
            quality=6,
            fps=None,
            device="cuda",
            dtype="float32"
        )
        
        inference_command(args)
        
        # Verify device was changed to cpu
        assert args.device == "cpu"
    
    @patch('cli.commands.save_video')
    @patch('cli.commands.tensor2video')
    @patch('cli.commands.prepare_input_tensor')
    @patch('cli.commands.init_pipeline')
    @patch('cli.commands.get_dtype')
    @patch('cli.commands.os.path.exists')
    @patch('cli.commands.os.path.dirname')
    @patch('cli.commands.os.path.abspath')
    @patch('cli.commands.os.path.join')
    @patch('cli.commands.os.makedirs')
    def test_inference_command_custom_output(
        self, mock_makedirs, mock_join, mock_abspath, mock_dirname,
        mock_exists, mock_get_dtype, mock_init_pipeline,
        mock_prepare_input, mock_tensor2video, mock_save_video
    ):
        """Test inference_command uses custom output path when provided."""
        mock_abspath.return_value = "/path/to/cli/commands.py"
        mock_dirname.return_value = "/path/to/cli"
        mock_exists.return_value = True
        mock_get_dtype.return_value = torch.float32
        
        mock_pipe = MagicMock()
        mock_pipe.return_value = torch.randn(3, 5, 64, 64)
        mock_init_pipeline.return_value = mock_pipe
        
        mock_prepare_input.return_value = (
            torch.randn(1, 3, 5, 64, 64),
            64, 64, 5, 30
        )
        mock_tensor2video.return_value = [MagicMock() for _ in range(5)]
        
        custom_output = "/custom/path/output.mp4"
        args = Namespace(
            input="/path/to/input.mp4",
            output=custom_output,
            pipeline="full",
            version="v1.1",
            model_dir=None,
            scale=4.0,
            seed=0,
            sparse_ratio=2.0,
            local_range=11,
            tiled=False,
            color_fix=True,
            quality=6,
            fps=None,
            device="cpu",
            dtype="float32"
        )
        
        inference_command(args)
        
        # Verify custom output was used
        assert args.output == custom_output
        mock_save_video.assert_called_once()
        # Check that the output path in save_video call matches
        call_args = mock_save_video.call_args
        assert call_args[0][1] == custom_output
    
    @patch('cli.commands.init_pipeline')
    @patch('cli.commands.get_dtype')
    @patch('cli.commands.os.makedirs')
    @patch('cli.commands.os.path.exists')
    @patch('cli.commands.os.path.dirname')
    @patch('cli.commands.os.path.abspath')
    def test_inference_command_pipeline_error(
        self, mock_abspath, mock_dirname, mock_exists,
        mock_makedirs, mock_get_dtype, mock_init_pipeline
    ):
        """Test inference_command handles pipeline initialization errors."""
        mock_abspath.return_value = "/path/to/cli/commands.py"
        mock_dirname.return_value = "/path/to/cli"
        mock_exists.return_value = True
        mock_get_dtype.return_value = torch.float32
        mock_init_pipeline.side_effect = Exception("Pipeline init failed")
        
        args = Namespace(
            input="/path/to/input.mp4",
            output=None,
            pipeline="full",
            version="v1.1",
            model_dir=None,
            scale=4.0,
            seed=0,
            sparse_ratio=2.0,
            local_range=11,
            tiled=False,
            color_fix=True,
            quality=6,
            fps=None,
            device="cpu",
            dtype="float32"
        )
        
        with pytest.raises(SystemExit):
            inference_command(args)

