"""
Dataset Loader for License Plate Recognition
Supports loading from Kaggle datasets using kagglehub
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

try:
    import kagglehub
    from kagglehub import KaggleDatasetAdapter
    KAGGLEHUB_AVAILABLE = True
except ImportError:
    KAGGLEHUB_AVAILABLE = False
    logger.warning("kagglehub not available. Install with: pip install kagglehub")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    logger.warning("pandas not available. Install with: pip install pandas")


class DatasetLoader:
    """
    Load datasets from Kaggle using kagglehub
    """
    
    def __init__(self):
        if not KAGGLEHUB_AVAILABLE:
            raise ImportError("kagglehub is required. Install with: pip install kagglehub")
    
    @staticmethod
    def load_car_plate_detection_dataset(file_path: str = "") -> Optional[Any]:
        """
        Load the andrewmvd/car-plate-detection dataset from Kaggle
        
        Args:
            file_path: Path to specific file within the dataset (optional)
            
        Returns:
            DataFrame if pandas is available, or dataset path
        """
        if not KAGGLEHUB_AVAILABLE:
            raise ImportError("kagglehub is required")
        
        logger.info("Loading andrewmvd/car-plate-detection dataset from Kaggle...")
        
        try:
            if PANDAS_AVAILABLE and file_path:
                # Load as pandas DataFrame if file_path is provided
                df = kagglehub.load_dataset(
                    KaggleDatasetAdapter.PANDAS,
                    "andrewmvd/car-plate-detection",
                    file_path
                )
                logger.info(f"✅ Dataset loaded successfully. Shape: {df.shape}")
                logger.info(f"First 5 records:\n{df.head()}")
                return df
            else:
                # Download the dataset and return path
                path = kagglehub.dataset_download("andrewmvd/car-plate-detection")
                logger.info(f"✅ Dataset downloaded to: {path}")
                return path
                
        except Exception as e:
            logger.error(f"❌ Failed to load dataset: {e}")
            raise
    
    @staticmethod
    def get_dataset_info(dataset_path: str) -> Dict[str, Any]:
        """
        Get information about downloaded dataset
        
        Args:
            dataset_path: Path to the dataset directory
            
        Returns:
            Dictionary with dataset information
        """
        path = Path(dataset_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
        
        info = {
            "path": str(path),
            "exists": path.exists(),
            "is_dir": path.is_dir(),
            "files": [],
            "subdirs": []
        }
        
        if path.is_dir():
            for item in path.iterdir():
                if item.is_file():
                    info["files"].append({
                        "name": item.name,
                        "size": item.stat().st_size,
                        "path": str(item)
                    })
                elif item.is_dir():
                    info["subdirs"].append(item.name)
        
        logger.info(f"Dataset info: {len(info['files'])} files, {len(info['subdirs'])} subdirectories")
        
        return info
    
    @staticmethod
    def list_dataset_structure(dataset_path: str, max_depth: int = 3):
        """
        Print dataset directory structure
        
        Args:
            dataset_path: Path to the dataset
            max_depth: Maximum depth to traverse
        """
        path = Path(dataset_path)
        
        if not path.exists():
            logger.error(f"Path does not exist: {dataset_path}")
            return
        
        logger.info(f"Dataset structure for: {dataset_path}")
        
        def print_tree(dir_path: Path, prefix: str = "", depth: int = 0):
            if depth > max_depth:
                return
            
            try:
                items = sorted(dir_path.iterdir(), key=lambda x: (not x.is_dir(), x.name))
                for i, item in enumerate(items[:10]):  # Limit to first 10 items per level
                    is_last = i == len(items) - 1
                    current_prefix = "└── " if is_last else "├── "
                    logger.info(f"{prefix}{current_prefix}{item.name}")
                    
                    if item.is_dir() and depth < max_depth:
                        extension = "    " if is_last else "│   "
                        print_tree(item, prefix + extension, depth + 1)
                
                if len(items) > 10:
                    logger.info(f"{prefix}... and {len(items) - 10} more items")
                    
            except PermissionError:
                logger.warning(f"{prefix}[Permission Denied]")
        
        print_tree(path)


def load_dataset_for_training(use_kaggle_dataset: bool = True) -> str:
    """
    Load dataset for training - convenience function
    
    Args:
        use_kaggle_dataset: If True, download from Kaggle. Otherwise use local path.
        
    Returns:
        Path to dataset directory
    """
    if use_kaggle_dataset:
        loader = DatasetLoader()
        dataset_path = loader.load_car_plate_detection_dataset()
        
        # Show structure
        loader.list_dataset_structure(dataset_path)
        
        return dataset_path
    else:
        # Use local dataset path from environment or default
        local_path = os.environ.get("KAGGLE_DATA_PATH", "./data")
        logger.info(f"Using local dataset path: {local_path}")
        return local_path


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("DATASET LOADER - Testing")
    print("=" * 60)
    
    try:
        loader = DatasetLoader()
        
        # Load the dataset
        dataset_path = loader.load_car_plate_detection_dataset()
        
        # Get info
        info = loader.get_dataset_info(dataset_path)
        print(f"\nDataset loaded to: {info['path']}")
        print(f"Files: {len(info['files'])}")
        print(f"Subdirectories: {info['subdirs']}")
        
        # Show structure
        loader.list_dataset_structure(dataset_path)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
