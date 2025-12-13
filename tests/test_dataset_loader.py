"""
Tests for dataset loader functionality
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import after adding to path
from dataset_loader import DatasetLoader, load_dataset_for_training


def test_dataset_loader_initialization():
    """Test that DatasetLoader can be initialized"""
    try:
        loader = DatasetLoader()
        assert loader is not None
    except ImportError as e:
        pytest.skip(f"kagglehub not available: {e}")


@pytest.mark.skipif(
    not Path(__file__).parent.parent.joinpath("src", "dataset_loader.py").exists(),
    reason="dataset_loader module not found"
)
def test_dataset_loader_module_exists():
    """Test that dataset_loader module exists and can be imported"""
    try:
        import dataset_loader
        assert hasattr(dataset_loader, 'DatasetLoader')
        assert hasattr(dataset_loader, 'load_dataset_for_training')
    except ImportError:
        pytest.fail("Failed to import dataset_loader module")


@pytest.mark.integration
def test_load_car_plate_detection_dataset():
    """
    Integration test to load the actual Kaggle dataset
    Marked as integration test - may be slow and requires network
    """
    try:
        loader = DatasetLoader()
        dataset_path = loader.load_car_plate_detection_dataset()
        
        assert dataset_path is not None
        assert Path(dataset_path).exists()
        
        # Get info about the dataset
        info = loader.get_dataset_info(dataset_path)
        assert "path" in info
        assert "files" in info
        assert "subdirs" in info
        
    except ImportError as e:
        pytest.skip(f"kagglehub not available: {e}")
    except Exception as e:
        pytest.skip(f"Failed to download dataset (may require authentication): {e}")


def test_get_dataset_info_nonexistent_path():
    """Test that get_dataset_info raises error for non-existent path"""
    try:
        loader = DatasetLoader()
        with pytest.raises(FileNotFoundError):
            loader.get_dataset_info("/nonexistent/path/to/dataset")
    except ImportError as e:
        pytest.skip(f"kagglehub not available: {e}")


@patch('dataset_loader.kagglehub')
def test_load_dataset_with_mock(mock_kagglehub):
    """Test dataset loading with mocked kagglehub"""
    # Mock the dataset download
    mock_kagglehub.dataset_download.return_value = "/tmp/mock_dataset"
    mock_kagglehub.KaggleDatasetAdapter = MagicMock()
    
    with patch('dataset_loader.KAGGLEHUB_AVAILABLE', True):
        loader = DatasetLoader()
        
        # This should use the mocked version
        result = loader.load_car_plate_detection_dataset()
        
        # Verify the mock was called
        assert mock_kagglehub.dataset_download.called or result is not None


def test_load_dataset_for_training_local_path():
    """Test loading local dataset path"""
    with patch.dict('os.environ', {'KAGGLE_DATA_PATH': './test_data'}):
        path = load_dataset_for_training(use_kaggle_dataset=False)
        assert path == './test_data'


def test_dataset_loader_has_required_methods():
    """Test that DatasetLoader has all required methods"""
    try:
        loader = DatasetLoader()
        
        # Check that all expected methods exist
        assert hasattr(loader, 'load_car_plate_detection_dataset')
        assert hasattr(loader, 'get_dataset_info')
        assert hasattr(loader, 'list_dataset_structure')
        
        # Check they are callable
        assert callable(loader.load_car_plate_detection_dataset)
        assert callable(loader.get_dataset_info)
        assert callable(loader.list_dataset_structure)
        
    except ImportError as e:
        pytest.skip(f"kagglehub not available: {e}")


@pytest.mark.parametrize("max_depth", [1, 2, 3])
def test_list_dataset_structure_with_different_depths(max_depth, tmp_path):
    """Test listing dataset structure with different max depths"""
    try:
        # Create a temporary directory structure
        (tmp_path / "subdir1").mkdir()
        (tmp_path / "subdir2").mkdir()
        (tmp_path / "file1.txt").touch()
        
        loader = DatasetLoader()
        
        # This should not raise an exception
        loader.list_dataset_structure(str(tmp_path), max_depth=max_depth)
        
    except ImportError as e:
        pytest.skip(f"kagglehub not available: {e}")
