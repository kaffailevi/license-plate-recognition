"""
Probe Test - Gyors validáció (CI/CD pipeline alatt)
Elég gyors, nem igényel GPU
"""

import torch
import logging
import sys
from pathlib import Path
from inference import LicensePlateRecognizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model_loading():
    """Modell betöltés tesztje"""
    logger.info("Testing model loading...")
    try:
        recognizer = LicensePlateRecognizer()
        logger.info("✅ Models loaded successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        return False

def test_inference():
    """Inference futtatás tesztje"""
    logger.info("Testing inference...")
    try:
        recognizer = LicensePlateRecognizer()
        
        # Dummy batch
        test_batch = torch.randn(2, 3, 224, 224)
        detection = recognizer.detect(test_batch)
        
        assert detection is not None, "Detection output is None"
        assert detection.shape[0] == 2, f"Expected batch size 2, got {detection.shape[0]}"
        
        logger.info(f"✅ Inference successful, output shape: {detection.shape}")
        return True
    except Exception as e:
        logger.error(f"❌ Inference test failed: {e}")
        return False

def test_ocr():
    """OCR tesztje"""
    logger.info("Testing OCR...")
    try:
        recognizer = LicensePlateRecognizer()
        
        # Dummy plate image
        plate_image = torch.randn(1, 3, 32, 128)
        chars = recognizer.recognize_text(plate_image)
        plate_text = recognizer.decode_plate(chars[0])
        
        assert plate_text, "OCR output is empty"
        logger.info(f"✅ OCR successful, decoded: {plate_text}")
        return True
    except Exception as e:
        logger.error(f"❌ OCR test failed: {e}")
        return False

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("PROBE TEST - SANITY CHECKS")
    logger.info("=" * 60)
    
    results = {
        "model_loading": test_model_loading(),
        "inference": test_inference(),
        "ocr": test_ocr(),
    }
    
    logger.info("=" * 60)
    logger.info("RESULTS:")
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    logger.info("=" * 60)
    
    if all_passed:
        logger.info("✅ ALL PROBE TESTS PASSED")
        sys.exit(0)
    else:
        logger.error("❌ SOME TESTS FAILED")
        sys.exit(1)