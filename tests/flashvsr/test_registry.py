"""Tests for flashvsr/registry.py"""

import pytest
from unittest.mock import patch, MagicMock

from flashvsr.registry import register_wan_models


@pytest.mark.unit
class TestRegisterWanModels:
    """Tests for register_wan_models function."""

    @patch('diffsynth.configs.model_config')
    def test_register_wan_models_adds_configs(self, mock_model_config):
        """Test register_wan_models extends model_loader_configs."""
        # Arrange
        mock_list = MagicMock()
        mock_list.__len__ = lambda x: 0
        mock_model_config.model_loader_configs = mock_list
        
        # Act
        register_wan_models()
        
        # Assert
        # Check that extend was called
        assert mock_list.extend.called
        # Check that extend was called with a list of configs
        call_args = mock_list.extend.call_args
        assert len(call_args[0][0]) > 0

    @patch('diffsynth.configs.model_config')
    def test_register_wan_models_preserves_existing_configs(self, mock_model_config):
        """Test register_wan_models preserves existing configs."""
        # Arrange
        existing_config = ("existing", "hash", ["model"], [MagicMock()], "source")
        mock_list = MagicMock()
        mock_list.__iter__ = lambda x: iter([existing_config])
        mock_list.__len__ = lambda x: 1
        mock_model_config.model_loader_configs = mock_list
        
        # Act
        register_wan_models()
        
        # Assert
        # extend should have been called to add new configs
        assert mock_list.extend.called
        # Check that extend was called with configs
        call_args = mock_list.extend.call_args
        assert len(call_args[0][0]) > 0

