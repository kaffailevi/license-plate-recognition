"""
Modell minőség regressziós tesztek
"""

import pytest
import torch
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from inference import LicensePlateRecognizer

QUALITY_THRESHOLDS = {
    "min_accuracy": 0.85,
    "max_inference_time": 2.0,  # másodperc
    "min_model_size": 100000,  # bytes
}

@pytest.fixture
def recognizer():
    try:
        return LicensePlateRecognizer()
    except Exception as e:
        pytest.skip(f"Models not available: {e}")

def test_model_file_exists():
    """Modellfájlok létezésének ellenőrzése"""
    detector_path = Path("models/detector.pt")
    ocr_path = Path("models/ocr.pt")
    
    assert detector_path.exists(), f"Detector model not found: {detector_path}"
    assert ocr_path.exists(), f"OCR model not found: {ocr_path}"

def test_model_file_size():
    """Modell fájlok méretének ellenőrzése"""
    detector_path = Path("models/detector.pt")
    ocr_path = Path("models/ocr.pt")
    
    min_size = QUALITY_THRESHOLDS["min_model_size"]
    
    detector_size = detector_path.stat().st_size
    ocr_size = ocr_path.stat().st_size
    
    assert detector_size > min_size, f"Detector too small: {detector_size} bytes"
    assert ocr_size > min_size, f"OCR too small: {ocr_size} bytes"

def test_inference_speed(recognizer):
    """Inferencia sebességének tesztje"""
    import time
    
    batch = torch.randn(10, 3, 224, 224)
    
    start_time = time.time()
    recognizer.detect(batch)
    elapsed = time.time() - start_time
    
    max_time = QUALITY_THRESHOLDS["max_inference_time"]
    assert elapsed < max_time, f"Inference too slow: {elapsed:.2f}s > {max_time}s"

def test_training_metadata():
    """Tréning metaadatok ellenőrzése"""
    metadata_path = Path("models/training_metadata.json")
    
    assert metadata_path.exists(), "Training metadata not found"
    
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    required_keys = ["training_date", "pytorch_version", "device"]
    for key in required_keys:
        assert key in metadata, f"Missing metadata key: {key}"