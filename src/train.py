"""
YOLO Detector + OCR Model Training Script
Kaggle GPU-n fut, output: models/*.pt
"""

import os
import torch
import torchvision
from pathlib import Path
import logging
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import dataset loader if available
try:
    from dataset_loader import load_dataset_for_training, DatasetLoader
    DATASET_LOADER_AVAILABLE = True
except ImportError:
    DATASET_LOADER_AVAILABLE = False
    logger.warning("dataset_loader not available. Using default paths.")

# Kaggle futtatáskor:
# KAGGLE_DATA_PATH = "/kaggle/input/license-plate-dataset"
# KAGGLE_OUTPUT_PATH = "/kaggle/working"

# Check if we should use Kaggle dataset
USE_KAGGLE_DATASET = os.environ.get("USE_KAGGLE_DATASET", "false").lower() == "true"

if USE_KAGGLE_DATASET and DATASET_LOADER_AVAILABLE:
    try:
        logger.info("Loading dataset from Kaggle using kagglehub...")
        KAGGLE_DATA_PATH = load_dataset_for_training(use_kaggle_dataset=True)
    except (ImportError, ConnectionError) as e:
        logger.warning(f"Failed to load Kaggle dataset (import/connection error): {e}. Using default path.")
        KAGGLE_DATA_PATH = os.environ.get("KAGGLE_DATA_PATH", "./data")
    except Exception as e:
        logger.warning(f"Unexpected error loading Kaggle dataset: {e}. Using default path.")
        KAGGLE_DATA_PATH = os.environ.get("KAGGLE_DATA_PATH", "./data")
else:
    KAGGLE_DATA_PATH = os.environ.get("KAGGLE_DATA_PATH", "./data")

KAGGLE_OUTPUT_PATH = os.environ.get("KAGGLE_OUTPUT_PATH", "./models")

class LicensePlateDetector(torch.nn.Module):
    """YOLO-style detector modell (simplified)"""
    def __init__(self, num_classes=1):
        super().__init__()
        self.backbone = torchvision.models.resnet50(pretrained=True)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(2048, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, num_classes * 5)  # x, y, w, h, conf
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

class OCRModel(torch.nn.Module):
    """OCR modell szöveg felismeréshez"""
    def __init__(self, num_classes=36):  # 0-9, A-Z
        super().__init__()
        self.backbone = torchvision.models.resnet18(pretrained=True)
        self.head = torch.nn.Linear(512, num_classes)
    
    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

def train_detector():
    logger.info("🚀 Training Detector Model...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    model = LicensePlateDetector().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.MSELoss()
    
    # Dummy training loop (valós projekt: valós adat loader)
    num_epochs = 3
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        
        # TODO: Load real dataset
        dummy_batch = torch.randn(4, 3, 224, 224).to(device)
        dummy_target = torch.randn(4, 5).to(device)
        
        output = model(dummy_batch)
        loss = criterion(output, dummy_target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        logger.info(f"Loss: {loss.item():.4f}")
    
    # Save model
    os.makedirs(KAGGLE_OUTPUT_PATH, exist_ok=True)
    model_path = os.path.join(KAGGLE_OUTPUT_PATH, "detector.pt")
    torch.save(model.state_dict(), model_path)
    logger.info(f"✅ Detector model saved: {model_path}")
    
    return model_path

def train_ocr():
    logger.info("🚀 Training OCR Model...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = OCRModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    num_epochs = 3
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        
        # TODO: Load real dataset
        dummy_batch = torch.randn(16, 3, 32, 128).to(device)
        dummy_target = torch.randint(0, 36, (16,)).to(device)
        
        output = model(dummy_batch)
        loss = criterion(output, dummy_target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        logger.info(f"Loss: {loss.item():.4f}")
    
    # Save model
    os.makedirs(KAGGLE_OUTPUT_PATH, exist_ok=True)
    model_path = os.path.join(KAGGLE_OUTPUT_PATH, "ocr.pt")
    torch.save(model.state_dict(), model_path)
    logger.info(f"✅ OCR model saved: {model_path}")
    
    return model_path

def save_training_metadata():
    """Metaadatok mentése verziózás céljából"""
    metadata = {
        "training_date": datetime.now().isoformat(),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "detector_model": "LicensePlateDetector-v1",
        "ocr_model": "OCRModel-v1",
        "dataset_source": "kaggle:andrewmvd/car-plate-detection" if USE_KAGGLE_DATASET else "local",
        "dataset_path": KAGGLE_DATA_PATH
    }
    
    metadata_path = os.path.join(KAGGLE_OUTPUT_PATH, "training_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"✅ Metadata saved: {metadata_path}")

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("LICENSE PLATE RECOGNITION - TRAINING PIPELINE")
    logger.info("=" * 60)
    
    detector_path = train_detector()
    ocr_path = train_ocr()
    save_training_metadata()
    
    logger.info("=" * 60)
    logger.info("✅ TRAINING COMPLETE!")
    logger.info(f"Detector: {detector_path}")
    logger.info(f"OCR: {ocr_path}")
    logger.info("=" * 60)