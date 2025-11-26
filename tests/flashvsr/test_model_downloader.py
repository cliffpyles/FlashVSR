"""Tests for flashvsr/model_downloader.py"""

import os
import tempfile
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock, call

from flashvsr.model_downloader import (
    get_model_dir,
    check_model_files,
    download_model,
    download_models_for_pipeline,
    MODEL_REPOS,
    REQUIRED_FILES,
)


@pytest.mark.unit
class TestGetModelDir:
    """Tests for get_model_dir function."""

    def test_get_model_dir_v1_default(self):
        """Test get_model_dir returns correct path for v1 with default base_dir."""
        # Arrange
        version = "v1"
        
        # Act
        result = get_model_dir(version)
        
        # Assert
        assert isinstance(result, Path)
        assert result.name == "FlashVSR"
        assert result.parent.name == "models"

    def test_get_model_dir_v1_1_default(self):
        """Test get_model_dir returns correct path for v1.1 with default base_dir."""
        # Arrange
        version = "v1.1"
        
        # Act
        result = get_model_dir(version)
        
        # Assert
        assert isinstance(result, Path)
        assert result.name == "FlashVSR-v1.1"
        assert result.parent.name == "models"

    def test_get_model_dir_custom_base_dir(self):
        """Test get_model_dir uses custom base_dir when provided."""
        # Arrange
        version = "v1"
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = tmpdir
            
            # Act
            result = get_model_dir(version, base_dir)
            
            # Assert
            assert result == Path(tmpdir) / "FlashVSR"
            assert result.parent == Path(tmpdir)


@pytest.mark.unit
class TestCheckModelFiles:
    """Tests for check_model_files function."""

    def test_check_model_files_directory_not_exists(self):
        """Test check_model_files returns False when directory doesn't exist."""
        # Arrange
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "nonexistent"
            pipeline_type = "base"
            
            # Act
            all_exist, missing = check_model_files(model_dir, pipeline_type)
            
            # Assert
            assert all_exist is False
            assert len(missing) > 0
            assert all(f in missing for f in REQUIRED_FILES["base"])

    def test_check_model_files_all_exist(self):
        """Test check_model_files returns True when all files exist."""
        # Arrange
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            pipeline_type = "base"
            
            # Create required files
            for filename in REQUIRED_FILES["base"]:
                (model_dir / filename).touch()
            
            # Act
            all_exist, missing = check_model_files(model_dir, pipeline_type)
            
            # Assert
            assert all_exist is True
            assert len(missing) == 0

    def test_check_model_files_some_missing(self):
        """Test check_model_files returns False when some files are missing."""
        # Arrange
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            pipeline_type = "base"
            
            # Create only first required file
            (model_dir / REQUIRED_FILES["base"][0]).touch()
            
            # Act
            all_exist, missing = check_model_files(model_dir, pipeline_type)
            
            # Assert
            assert all_exist is False
            assert len(missing) == len(REQUIRED_FILES["base"]) - 1
            assert REQUIRED_FILES["base"][0] not in missing

    def test_check_model_files_different_pipeline_types(self):
        """Test check_model_files checks correct files for different pipeline types."""
        # Arrange
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            
            # Create files for "full" pipeline
            for filename in REQUIRED_FILES["full"]:
                (model_dir / filename).touch()
            
            # Act - check for "full"
            all_exist_full, _ = check_model_files(model_dir, "full")
            
            # Act - check for "tiny" (should fail)
            all_exist_tiny, missing_tiny = check_model_files(model_dir, "tiny")
            
            # Assert
            assert all_exist_full is True
            assert all_exist_tiny is False
            assert "TCDecoder.ckpt" in missing_tiny


@pytest.mark.unit
class TestDownloadModel:
    """Tests for download_model function."""

    def test_download_model_invalid_version(self):
        """Test download_model raises ValueError for invalid version."""
        # Arrange
        version = "invalid"
        
        # Act & Assert
        with pytest.raises(ValueError, match="Invalid version"):
            download_model(version)

    @patch('flashvsr.model_downloader.check_model_files')
    @patch('flashvsr.model_downloader.snapshot_download')
    def test_download_model_already_exists(self, mock_snapshot, mock_check):
        """Test download_model returns early when models already exist."""
        # Arrange
        version = "v1"
        mock_check.return_value = (True, [])
        
        # Act
        result = download_model(version, quiet=True)
        
        # Assert
        assert isinstance(result, Path)
        mock_snapshot.assert_not_called()

    @patch('flashvsr.model_downloader.check_model_files')
    @patch('flashvsr.model_downloader.snapshot_download')
    def test_download_model_downloads_when_missing(self, mock_snapshot, mock_check):
        """Test download_model downloads when files are missing."""
        # Arrange
        version = "v1"
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "FlashVSR"
            model_dir.mkdir(parents=True)
            
            mock_check.side_effect = [
                (False, REQUIRED_FILES["base"]),  # First check: missing
                (True, []),  # Second check: all exist after download
            ]
            mock_snapshot.return_value = str(model_dir)
            
            # Act
            result = download_model(version, base_dir=tmpdir, quiet=True)
            
            # Assert
            assert isinstance(result, Path)
            mock_snapshot.assert_called_once()
            call_args = mock_snapshot.call_args
            assert call_args.kwargs['repo_id'] == MODEL_REPOS[version]
            assert call_args.kwargs['local_dir'] == str(model_dir)
            assert call_args.kwargs['resume_download'] is True

    @patch('flashvsr.model_downloader.check_model_files')
    @patch('flashvsr.model_downloader.snapshot_download')
    def test_download_model_raises_when_files_still_missing(self, mock_snapshot, mock_check):
        """Test download_model raises RuntimeError when files still missing after download."""
        # Arrange
        version = "v1"
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "FlashVSR"
            model_dir.mkdir(parents=True)
            
            missing_files = ["file1.ckpt"]
            mock_check.side_effect = [
                (False, missing_files),  # First check: missing
                (False, missing_files),  # Second check: still missing
            ]
            mock_snapshot.return_value = str(model_dir)
            
            # Act & Assert
            with pytest.raises(RuntimeError, match="some required files are missing"):
                download_model(version, base_dir=tmpdir, quiet=True)

    @patch('flashvsr.model_downloader.check_model_files')
    @patch('flashvsr.model_downloader.snapshot_download')
    def test_download_model_handles_download_exception(self, mock_snapshot, mock_check):
        """Test download_model handles download exceptions gracefully."""
        # Arrange
        version = "v1"
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "FlashVSR"
            
            mock_check.return_value = (False, REQUIRED_FILES["base"])
            mock_snapshot.side_effect = Exception("Network error")
            
            # Act & Assert
            with pytest.raises(RuntimeError, match="Failed to download model"):
                download_model(version, base_dir=tmpdir, quiet=True)


@pytest.mark.unit
class TestDownloadModelsForPipeline:
    """Tests for download_models_for_pipeline function."""

    @patch('flashvsr.model_downloader.check_model_files')
    def test_download_models_for_pipeline_already_exists(self, mock_check):
        """Test download_models_for_pipeline returns model_dir when files exist."""
        # Arrange
        version = "v1"
        pipeline_type = "full"
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "FlashVSR"
            model_dir.mkdir(parents=True)
            
            mock_check.return_value = (True, [])
            
            # Act
            result = download_models_for_pipeline(
                version, pipeline_type, base_dir=tmpdir, auto_download=False, quiet=True
            )
            
            # Assert
            assert result == model_dir

    @patch('flashvsr.model_downloader.check_model_files')
    def test_download_models_for_pipeline_raises_when_auto_download_false(self, mock_check):
        """Test download_models_for_pipeline raises FileNotFoundError when auto_download is False."""
        # Arrange
        version = "v1"
        pipeline_type = "full"
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "FlashVSR"
            missing = ["file1.ckpt"]
            
            mock_check.return_value = (False, missing)
            
            # Act & Assert
            with pytest.raises(FileNotFoundError, match="Model directory not found"):
                download_models_for_pipeline(
                    version, pipeline_type, base_dir=tmpdir, auto_download=False, quiet=True
                )

    @patch('flashvsr.model_downloader.download_model')
    @patch('flashvsr.model_downloader.check_model_files')
    def test_download_models_for_pipeline_auto_downloads(self, mock_check, mock_download):
        """Test download_models_for_pipeline auto-downloads when enabled."""
        # Arrange
        version = "v1"
        pipeline_type = "full"
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "FlashVSR"
            missing = ["file1.ckpt"]
            
            mock_check.return_value = (False, missing)
            mock_download.return_value = model_dir
            
            # Act
            result = download_models_for_pipeline(
                version, pipeline_type, base_dir=tmpdir, auto_download=True, quiet=True
            )
            
            # Assert
            assert result == model_dir
            mock_download.assert_called_once_with(
                version, pipeline_type, tmpdir, resume_download=True, quiet=True
            )

