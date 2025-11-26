"""
License Plate Recognition - Inference Module
Egyszerű és gyors inference pipeline
"""

import torch
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class LicensePlateRecognizer:
    def __init__(self, detector_path: str = "models/detector.pt", 
                 ocr_path: str = "models/ocr.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            self.detector = torch.load(detector_path, map_location=self.device)
            self.ocr = torch.load(ocr_path, map_location=self.device)
            self.detector.eval()
            self.ocr.eval()
            logger.info(f"✅ Models loaded on {self.device}")
        except FileNotFoundError as e:
            logger.error(f"❌ Model file not found: {e}")
            raise
    
    def detect(self, image_tensor):
        """
        Rendszámtábla detektálása
        Input: torch.Tensor (1, 3, 224, 224)
        Output: bbox [x, y, w, h, conf]
        """
        with torch.no_grad():
            output = self.detector(image_tensor.to(self.device))
        return output.cpu().numpy()
    
    def recognize_text(self, plate_image):
        """
        Szöveg felismerése a rendszámtáblán
        Input: torch.Tensor (1, 3, 32, 128)
        Output: character indices
        """
        with torch.no_grad():
            output = self.ocr(plate_image.to(self.device))
        
        # Legvalószínűbb karakter
        predicted_chars = torch.argmax(output, dim=1)
        return predicted_chars.cpu().numpy()
    
    def decode_plate(self, char_indices):
        """Karakterindexek dekódolása szövegre"""
        chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        return "".join(chars[i] for i in char_indices if i < len(chars))

if __name__ == "__main__":
    # Teszt futtatás
    recognizer = LicensePlateRecognizer()
    
    # Dummy input
    test_image = torch.randn(1, 3, 224, 224)
    detection = recognizer.detect(test_image)
    print(f"Detection output: {detection}")
    
    test_plate = torch.randn(1, 3, 32, 128)
    chars = recognizer.recognize_text(test_plate)
    plate_text = recognizer.decode_plate(chars[0])
    print(f"Recognized plate: {plate_text}")