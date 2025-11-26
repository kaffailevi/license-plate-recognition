"""
Unit tesztek az inference modulhoz
pytest futtatás: pytest tests/test_inference.py -v
"""

import pytest
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from inference import LicensePlateRecognizer

@pytest.fixture
def recognizer():
    """Shared recognizer instance"""
    try:
        return LicensePlateRecognizer()
    except Exception as e:
        pytest.skip(f"Models not available: {e}")

def test_model_initialization(recognizer):
    """Modell inicializáció teszt"""
    assert recognizer is not None
    assert recognizer.device is not None
    assert recognizer.detector is not None
    assert recognizer.ocr is not None

def test_detect_output_shape(recognizer):
    """Detekció kimenet alakja"""
    test_batch = torch.randn(4, 3, 224, 224)
    output = recognizer.detect(test_batch)
    
    assert output is not None
    assert output.shape[0] == 4
    assert output.shape[1] == 5  # [x, y, w, h, conf]

def test_ocr_output_shape(recognizer):
    """OCR kimenet alakja"""
    test_batch = torch.randn(4, 3, 32, 128)
    output = recognizer.recognize_text(test_batch)
    
    assert output is not None
    assert output.shape[0] == 4

def test_decode_plate(recognizer):
    """Plate dekódolás teszt"""
    char_indices = [15, 25, 10, 0, 1, 2]  # Random indices
    plate_text = recognizer.decode_plate(char_indices)
    
    assert isinstance(plate_text, str)
    assert len(plate_text) == 6
    assert all(c.isalnum() for c in plate_text)

def test_inference_single_image(recognizer):
    """Egyszeri képfeldolgozás"""
    image = torch.randn(1, 3, 224, 224)
    detection = recognizer.detect(image)
    
    assert detection.shape[0] == 1
    # Konfidencia érték [0, 1] között
    assert 0 <= detection[0, 4] <= 1

@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
def test_batch_inference(recognizer, batch_size):
    """Batch feldolgozás különböző méretekben"""
    batch = torch.randn(batch_size, 3, 224, 224)
    output = recognizer.detect(batch)
    
    assert output.shape[0] == batch_size