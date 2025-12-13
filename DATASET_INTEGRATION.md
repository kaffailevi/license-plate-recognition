# Dataset Integration Guide

## Overview

This project now supports automatic integration with the `andrewmvd/car-plate-detection` dataset from Kaggle using `kagglehub`.

## Quick Start

### Installation

```bash
pip install kagglehub pandas
```

### Using the Dataset Loader

```python
from src.dataset_loader import DatasetLoader

# Initialize
loader = DatasetLoader()

# Download the dataset
dataset_path = loader.load_car_plate_detection_dataset()

# Explore the dataset
info = loader.get_dataset_info(dataset_path)
print(f"Files: {len(info['files'])}")
print(f"Subdirectories: {info['subdirs']}")

# Show structure
loader.list_dataset_structure(dataset_path)
```

### Training with Kaggle Dataset

Set the environment variable and run training:

```bash
export USE_KAGGLE_DATASET=true
python src/train.py
```

Or directly in Python:

```python
import os
os.environ['USE_KAGGLE_DATASET'] = 'true'

from src.train import train_detector, train_ocr, save_training_metadata

# Training will automatically use the Kaggle dataset
detector_path = train_detector()
ocr_path = train_ocr()
save_training_metadata()
```

## Dataset Information

### andrewmvd/car-plate-detection

- **Source**: Kaggle dataset by andrewmvd
- **Type**: License plate detection and recognition
- **Format**: Images with XML annotations (Pascal VOC format)
- **Use Case**: Training object detection and OCR models for license plates

## Features

### DatasetLoader Class

The `DatasetLoader` class provides:

1. **Automatic Download**: Downloads the dataset from Kaggle using `kagglehub`
2. **Caching**: Uses kagglehub's built-in caching to avoid re-downloading
3. **Dataset Info**: Extract metadata about files and structure
4. **Structure Visualization**: Display directory tree
5. **Pandas Integration**: Load CSV/tabular data as DataFrames

### Methods

#### `load_car_plate_detection_dataset(file_path="")`

Download and return path to the dataset.

**Parameters:**
- `file_path` (str, optional): Path to specific file for pandas loading

**Returns:**
- Dataset path (str) or DataFrame if file_path is provided

**Example:**
```python
# Download entire dataset
path = loader.load_car_plate_detection_dataset()

# Load specific CSV file as DataFrame
df = loader.load_car_plate_detection_dataset("annotations.csv")
```

#### `get_dataset_info(dataset_path)`

Get detailed information about downloaded dataset.

**Parameters:**
- `dataset_path` (str): Path to dataset directory

**Returns:**
- Dictionary with files, subdirs, and metadata

**Example:**
```python
info = loader.get_dataset_info("/path/to/dataset")
for file in info['files']:
    print(f"{file['name']}: {file['size']} bytes")
```

#### `list_dataset_structure(dataset_path, max_depth=3)`

Print directory tree structure.

**Parameters:**
- `dataset_path` (str): Path to dataset
- `max_depth` (int): Maximum depth to traverse

**Example:**
```python
loader.list_dataset_structure(dataset_path, max_depth=2)
```

## Testing

### Unit Tests

Run non-integration tests:

```bash
pytest tests/test_dataset_loader.py -v -m "not integration"
```

### Integration Tests

Run tests that download the actual dataset (requires network):

```bash
pytest tests/test_dataset_loader.py -v -m "integration"
```

### Manual Testing

Test the loader directly:

```bash
python src/dataset_loader.py
```

## CI/CD Integration

The dataset loader is integrated into the CI/CD pipeline:

### GitHub Actions

The workflow in `.github/workflows/ci-test.yml` includes:

1. Install `kagglehub` and `pandas`
2. Run dataset loader tests
3. Skip integration tests by default (use markers)

### Running in CI

```yaml
- name: Run dataset loader tests
  run: |
    pytest tests/test_dataset_loader.py -v -m "not integration"
  continue-on-error: true
```

## Training Integration

### Environment Variables

- `USE_KAGGLE_DATASET`: Set to `"true"` to enable Kaggle dataset loading
- `KAGGLE_DATA_PATH`: Fallback path if Kaggle download fails

### Metadata

Training metadata now includes:

```json
{
  "dataset_source": "kaggle:andrewmvd/car-plate-detection",
  "dataset_path": "/path/to/downloaded/dataset"
}
```

## Architecture

```
┌─────────────────────┐
│   Training Script   │
│    (src/train.py)   │
└──────────┬──────────┘
           │
           ↓ USE_KAGGLE_DATASET=true
┌─────────────────────┐
│  DatasetLoader      │
│ (dataset_loader.py) │
└──────────┬──────────┘
           │
           ↓ kagglehub
┌─────────────────────┐
│  Kaggle Dataset     │
│ andrewmvd/car-plate │
└─────────────────────┘
```

## Troubleshooting

### kagglehub not installed

**Error:**
```
ImportError: kagglehub is required
```

**Solution:**
```bash
pip install kagglehub
```

### Network issues

**Error:**
```
Failed to load dataset: Connection error
```

**Solution:**
- Check internet connection
- Verify Kaggle API access
- Try again or use local dataset with `USE_KAGGLE_DATASET=false`

### Authentication required

Some datasets may require Kaggle authentication. Set up your Kaggle credentials:

```bash
mkdir -p ~/.kaggle
# Download kaggle.json from Kaggle website
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

## Examples

### Example 1: Quick Download

```python
from src.dataset_loader import load_dataset_for_training

# Download and get path
path = load_dataset_for_training(use_kaggle_dataset=True)
print(f"Dataset at: {path}")
```

### Example 2: Explore Dataset

```python
from src.dataset_loader import DatasetLoader
import os

loader = DatasetLoader()
path = loader.load_car_plate_detection_dataset()

# List all files
for root, dirs, files in os.walk(path):
    print(f"Directory: {root}")
    for file in files:
        print(f"  - {file}")
```

### Example 3: Load Annotations

```python
from src.dataset_loader import DatasetLoader
import pandas as pd
import os

loader = DatasetLoader()
path = loader.load_car_plate_detection_dataset()

# Find annotation files
annotation_files = []
for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith('.csv'):
            annotation_files.append(os.path.join(root, file))

# Load first annotation file
if annotation_files:
    df = pd.read_csv(annotation_files[0])
    print(df.head())
```

## Future Enhancements

- Support for additional Kaggle datasets
- Custom dataset registration
- Dataset versioning support
- Automatic format detection
- Data augmentation pipeline integration

## References

- [kagglehub Documentation](https://github.com/Kaggle/kagglehub)
- [andrewmvd/car-plate-detection Dataset](https://www.kaggle.com/datasets/andrewmvd/car-plate-detection)
- [Kaggle API Documentation](https://github.com/Kaggle/kaggle-api)

---

**Version**: 1.0  
**Last Updated**: December 13, 2025  
**Maintainer**: kaffailevi
