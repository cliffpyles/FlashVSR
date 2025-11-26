"""Tests for flashvsr/models/utils.py"""

import os
import tempfile
import hashlib
import pytest
import torch
from pathlib import Path

from flashvsr.models.utils import (
    load_state_dict,
    load_state_dict_from_folder,
    load_state_dict_from_safetensors,
    load_state_dict_from_bin,
    search_for_embeddings,
    search_parameter,
    search_for_files,
    convert_state_dict_keys_to_single_str,
    split_state_dict_with_prefix,
    hash_state_dict_keys,
)


@pytest.mark.unit
class TestLoadStateDictFromBin:
    """Tests for load_state_dict_from_bin function."""

    def test_load_state_dict_from_bin(self):
        """Test loading state dict from .pth file."""
        # Arrange
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            state_dict = {
                'layer1.weight': torch.randn(3, 3),
                'layer1.bias': torch.randn(3),
            }
            torch.save(state_dict, f.name)
            file_path = f.name
        
        try:
            # Act
            result = load_state_dict_from_bin(file_path)
            
            # Assert
            assert isinstance(result, dict)
            assert 'layer1.weight' in result
            assert 'layer1.bias' in result
            assert torch.equal(result['layer1.weight'], state_dict['layer1.weight'])
        finally:
            os.unlink(file_path)

    def test_load_state_dict_from_bin_with_dtype(self):
        """Test loading state dict with dtype conversion."""
        # Arrange
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            state_dict = {
                'layer1.weight': torch.randn(3, 3, dtype=torch.float32),
            }
            torch.save(state_dict, f.name)
            file_path = f.name
        
        try:
            # Act
            result = load_state_dict_from_bin(file_path, torch_dtype=torch.float16)
            
            # Assert
            assert result['layer1.weight'].dtype == torch.float16
        finally:
            os.unlink(file_path)


@pytest.mark.unit
class TestLoadStateDict:
    """Tests for load_state_dict function."""

    def test_load_state_dict_from_pth(self):
        """Test load_state_dict loads .pth files."""
        # Arrange
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            state_dict = {'weight': torch.randn(3, 3)}
            torch.save(state_dict, f.name)
            file_path = f.name
        
        try:
            # Act
            result = load_state_dict(file_path)
            
            # Assert
            assert 'weight' in result
        finally:
            os.unlink(file_path)

    def test_load_state_dict_from_ckpt(self):
        """Test load_state_dict loads .ckpt files."""
        # Arrange
        with tempfile.NamedTemporaryFile(suffix='.ckpt', delete=False) as f:
            state_dict = {'weight': torch.randn(3, 3)}
            torch.save(state_dict, f.name)
            file_path = f.name
        
        try:
            # Act
            result = load_state_dict(file_path)
            
            # Assert
            assert 'weight' in result
        finally:
            os.unlink(file_path)


@pytest.mark.unit
class TestLoadStateDictFromFolder:
    """Tests for load_state_dict_from_folder function."""

    def test_load_state_dict_from_folder(self):
        """Test loading state dicts from folder with multiple files."""
        # Arrange
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multiple .pth files
            state_dict1 = {'layer1.weight': torch.randn(3, 3)}
            state_dict2 = {'layer2.weight': torch.randn(2, 2)}
            
            torch.save(state_dict1, os.path.join(tmpdir, 'model1.pth'))
            torch.save(state_dict2, os.path.join(tmpdir, 'model2.pth'))
            
            # Act
            result = load_state_dict_from_folder(tmpdir)
            
            # Assert
            assert 'layer1.weight' in result
            assert 'layer2.weight' in result

    def test_load_state_dict_from_folder_ignores_non_model_files(self):
        """Test load_state_dict_from_folder ignores non-model files."""
        # Arrange
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a .pth file and a .txt file
            state_dict = {'weight': torch.randn(3, 3)}
            torch.save(state_dict, os.path.join(tmpdir, 'model.pth'))
            
            with open(os.path.join(tmpdir, 'readme.txt'), 'w') as f:
                f.write('readme')
            
            # Act
            result = load_state_dict_from_folder(tmpdir)
            
            # Assert
            assert 'weight' in result
            assert len(result) == 1


@pytest.mark.unit
class TestSearchForEmbeddings:
    """Tests for search_for_embeddings function."""

    def test_search_for_embeddings_simple(self):
        """Test search_for_embeddings finds tensors in state dict."""
        # Arrange
        state_dict = {
            'embedding1': torch.randn(10, 5),
            'embedding2': torch.randn(20, 3),
            'weight': torch.randn(3, 3),
        }
        
        # Act
        embeddings = search_for_embeddings(state_dict)
        
        # Assert
        assert len(embeddings) == 3
        assert all(isinstance(e, torch.Tensor) for e in embeddings)

    def test_search_for_embeddings_nested(self):
        """Test search_for_embeddings finds tensors in nested dicts."""
        # Arrange
        state_dict = {
            'layer1': {
                'weight': torch.randn(3, 3),
                'bias': torch.randn(3),
            },
            'layer2.weight': torch.randn(2, 2),
        }
        
        # Act
        embeddings = search_for_embeddings(state_dict)
        
        # Assert
        assert len(embeddings) == 3


@pytest.mark.unit
class TestSearchParameter:
    """Tests for search_parameter function."""

    def test_search_parameter_finds_exact_match(self):
        """Test search_parameter finds exact parameter match."""
        # Arrange
        param = torch.randn(3, 3)
        state_dict = {
            'layer1.weight': param.clone(),
            'layer2.weight': torch.randn(2, 2),
        }
        
        # Act
        result = search_parameter(param, state_dict)
        
        # Assert
        assert result == 'layer1.weight'

    def test_search_parameter_finds_reshaped_match(self):
        """Test search_parameter finds match with different shape."""
        # Arrange
        param = torch.randn(9)  # Flattened
        state_dict = {
            'layer1.weight': torch.randn(3, 3),  # Same elements, different shape
        }
        # Make them match by setting same values
        state_dict['layer1.weight'] = param.reshape(3, 3)
        
        # Act
        result = search_parameter(param, state_dict)
        
        # Assert
        assert result == 'layer1.weight'

    def test_search_parameter_returns_none_when_no_match(self):
        """Test search_parameter returns None when no match found."""
        # Arrange
        param = torch.randn(3, 3)
        state_dict = {
            'layer1.weight': torch.randn(2, 2),  # Different size
        }
        
        # Act
        result = search_parameter(param, state_dict)
        
        # Assert
        assert result is None


@pytest.mark.unit
class TestSearchForFiles:
    """Tests for search_for_files function."""

    def test_search_for_files_in_directory(self):
        """Test search_for_files finds files with matching extensions."""
        # Arrange
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create files with different extensions
            Path(tmpdir, 'file1.pth').touch()
            Path(tmpdir, 'file2.ckpt').touch()
            Path(tmpdir, 'file3.txt').touch()
            
            # Act
            result = search_for_files(tmpdir, ['.pth', '.ckpt'])
            
            # Assert
            assert len(result) == 2
            assert any('file1.pth' in f for f in result)
            assert any('file2.ckpt' in f for f in result)

    def test_search_for_files_single_file(self):
        """Test search_for_files returns file if it matches extension."""
        # Arrange
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            file_path = f.name
        
        try:
            # Act
            result = search_for_files(file_path, ['.pth'])
            
            # Assert
            assert len(result) == 1
            assert file_path in result
        finally:
            os.unlink(file_path)


@pytest.mark.unit
class TestConvertStateDictKeysToSingleStr:
    """Tests for convert_state_dict_keys_to_single_str function."""

    def test_convert_state_dict_keys_to_single_str(self):
        """Test convert_state_dict_keys_to_single_str converts keys to string."""
        # Arrange
        state_dict = {
            'layer1.weight': torch.randn(3, 3),
            'layer2.bias': torch.randn(3),
        }
        
        # Act
        result = convert_state_dict_keys_to_single_str(state_dict, with_shape=False)
        
        # Assert
        assert isinstance(result, str)
        assert 'layer1.weight' in result
        assert 'layer2.bias' in result

    def test_convert_state_dict_keys_to_single_str_with_shape(self):
        """Test convert_state_dict_keys_to_single_str includes shapes when requested."""
        # Arrange
        state_dict = {
            'layer1.weight': torch.randn(3, 3),
        }
        
        # Act
        result = convert_state_dict_keys_to_single_str(state_dict, with_shape=True)
        
        # Assert
        assert '3_3' in result  # Shape should be included


@pytest.mark.unit
class TestSplitStateDictWithPrefix:
    """Tests for split_state_dict_with_prefix function."""

    def test_split_state_dict_with_prefix(self):
        """Test split_state_dict_with_prefix splits by prefix."""
        # Arrange
        state_dict = {
            'layer1.weight': torch.randn(3, 3),
            'layer1.bias': torch.randn(3),
            'layer2.weight': torch.randn(2, 2),
        }
        
        # Act
        result = split_state_dict_with_prefix(state_dict)
        
        # Assert
        assert len(result) == 2
        # Check that keys are grouped by prefix
        prefixes = [list(sub_dict.keys())[0].split('.')[0] for sub_dict in result]
        assert 'layer1' in prefixes
        assert 'layer2' in prefixes


@pytest.mark.unit
class TestHashStateDictKeys:
    """Tests for hash_state_dict_keys function."""

    def test_hash_state_dict_keys_consistent(self):
        """Test hash_state_dict_keys produces consistent hashes."""
        # Arrange
        state_dict = {
            'layer1.weight': torch.randn(3, 3),
            'layer2.bias': torch.randn(3),
        }
        
        # Act
        hash1 = hash_state_dict_keys(state_dict)
        hash2 = hash_state_dict_keys(state_dict)
        
        # Assert
        assert hash1 == hash2
        assert len(hash1) == 32  # MD5 hash length

    def test_hash_state_dict_keys_different_for_different_keys(self):
        """Test hash_state_dict_keys produces different hashes for different keys."""
        # Arrange
        state_dict1 = {'layer1.weight': torch.randn(3, 3)}
        state_dict2 = {'layer2.weight': torch.randn(3, 3)}
        
        # Act
        hash1 = hash_state_dict_keys(state_dict1)
        hash2 = hash_state_dict_keys(state_dict2)
        
        # Assert
        assert hash1 != hash2

