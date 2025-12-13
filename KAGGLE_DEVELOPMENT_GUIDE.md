# 🚀 Kaggle Model Development Guide

Complete walkthrough for training and deploying License Plate Recognition models using Kaggle GPU infrastructure.

## 📑 Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Initial Setup](#initial-setup)
4. [Kaggle Environment Setup](#kaggle-environment-setup)
5. [Dataset Preparation](#dataset-preparation)
6. [Model Training Workflow](#model-training-workflow)
7. [Model Export & Versioning](#model-export--versioning)
8. [GitHub Integration](#github-integration)
9. [Testing & Quality Gates](#testing--quality-gates)
10. [Troubleshooting](#troubleshooting)
11. [Best Practices](#best-practices)

---

## Overview

This project uses a **Kaggle-first training pipeline** to leverage free GPU resources for model training. The workflow is:

```
┌─────────────────┐
│  Kaggle GPU     │  ← Train models with GPU acceleration
│  Training       │
└────────┬────────┘
         │
         ↓ Export & Version
┌─────────────────┐
│ Kaggle Dataset  │  ← Store versioned model artifacts
│ (Model Storage) │
└────────┬────────┘
         │
         ↓ Download via API
┌─────────────────┐
│ GitHub Actions  │  ← Automated testing & validation
│ CI/CD Pipeline  │
└────────┬────────┘
         │
         ↓ Quality Gate ✓
┌─────────────────┐
│   Production    │  ← Deploy validated models
└─────────────────┘
```

### Key Components

- **Detector Model**: YOLO-style detector for license plate bounding boxes
- **OCR Model**: Text recognition for extracting characters from plates
- **Training Script**: `src/train.py` orchestrates both model training
- **Inference Module**: `src/inference.py` loads and runs predictions

---

## Prerequisites

### 1. Accounts & Credentials

- **Kaggle Account**: Sign up at [kaggle.com](https://www.kaggle.com)
- **GitHub Account**: Repository access to `kaffailevi/license-plate-recognition`

### 2. Local Environment

- **Python 3.10+**
- **Git** installed
- **Text editor** (VS Code, PyCharm, etc.)

### 3. Required Knowledge

- Basic Python and PyTorch
- Familiarity with Jupyter notebooks
- Understanding of Git workflows

---

## Initial Setup

### 1. Clone Repository

```bash
git clone https://github.com/kaffailevi/license-plate-recognition.git
cd license-plate-recognition
```

### 2. Create Virtual Environment

```bash
# Create venv
python -m venv venv

# Activate
source venv/bin/activate  # Linux/macOS
# OR
venv\Scripts\activate  # Windows
```

### 3. Install Dependencies

```bash
# From the repository root directory
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -c "import torch; print(f'PyTorch {torch.__version__} installed')"
```

---

## Kaggle Environment Setup

### 1. Get Kaggle API Credentials

1. Go to [Kaggle Account Settings](https://www.kaggle.com/settings/account)
2. Scroll to **API** section
3. Click **Create New API Token**
4. Download `kaggle.json` file

### 2. Configure Kaggle API Locally

```bash
# Create Kaggle directory
mkdir -p ~/.kaggle

# Move credentials (replace path with your download location)
cp ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### 3. Verify API Access

```bash
# Replace YOUR_USERNAME with your actual Kaggle username
kaggle datasets list --user YOUR_USERNAME
```

If successful, you'll see your datasets listed.

### 4. Add GitHub Secrets (One-time Setup)

To enable CI/CD integration:

1. Go to GitHub repository: **Settings** → **Secrets and variables** → **Actions**
2. Click **New repository secret**
3. Add two secrets:
   - `KAGGLE_USERNAME`: Your Kaggle username
   - `KAGGLE_KEY`: The `key` value from `kaggle.json`

**Example `kaggle.json` structure:**
```json
{
  "username": "YOUR_USERNAME",
  "key": "your_api_key_here"
}
```

---

## Dataset Preparation

### 1. Dataset Structure

Organize your license plate dataset as follows:

```
license-plate-dataset/
├── images/
│   ├── train/
│   │   ├── img001.jpg
│   │   ├── img002.jpg
│   │   └── ...
│   └── val/
│       ├── img101.jpg
│       └── ...
└── annotations/
    ├── train.json  # Bounding boxes + text labels
    └── val.json
```

### 2. Annotation Format

**For Detector (Bounding Boxes):**
```json
{
  "img001.jpg": {
    "bbox": [x, y, width, height],
    "confidence": 1.0
  }
}
```

**For OCR (Text Labels):**
```json
{
  "img001.jpg": {
    "text": "ABC1234"
  }
}
```

### 3. Upload Dataset to Kaggle

#### Option A: Via Web Interface

1. Go to [kaggle.com/datasets](https://www.kaggle.com/datasets)
2. Click **New Dataset**
3. Upload your `license-plate-dataset/` folder
4. Set visibility (Private/Public)
5. Click **Create**

#### Option B: Via CLI

```bash
# Create dataset metadata
cat > dataset-metadata.json << EOF
{
  "title": "License Plate Dataset",
  "id": "YOUR_USERNAME/license-plate-dataset",
  "licenses": [{"name": "CC0-1.0"}]
}
EOF

# Upload dataset (replace with your actual dataset path)
kaggle datasets create -p /path/to/license-plate-dataset
```

### 4. Note Dataset Path

After upload, note your dataset path:
```
YOUR_USERNAME/license-plate-dataset
```

You'll use this in your Kaggle notebook.

---

## Model Training Workflow

### 1. Create Kaggle Notebook

1. Go to [kaggle.com/code](https://www.kaggle.com/code)
2. Click **New Notebook**
3. Enable **GPU** accelerator:
   - Settings → Accelerator → GPU T4 x2

### 2. Setup Notebook Environment

In the first cell:

```python
# Install additional dependencies if needed
!pip install -q torch torchvision

# Verify GPU
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

### 3. Add Dataset as Input

1. In notebook, click **Add Data** (right sidebar)
2. Search for your dataset: `YOUR_USERNAME/license-plate-dataset`
3. Click **Add**

The dataset will be available at:
```python
KAGGLE_DATA_PATH = "/kaggle/input/license-plate-dataset"
```

### 4. Upload Training Script

**Option A: Add as Dataset**

1. Create a dataset with `src/train.py`
2. Add it to notebook inputs
3. Copy to working directory:

```python
!cp /kaggle/input/training-scripts/train.py /kaggle/working/
```

**Option B: Copy-Paste Code**

Copy the contents of `src/train.py` directly into notebook cells.

### 5. Modify Training Script

Update `src/train.py` to load your actual dataset:

```python
# Replace dummy data loaders with real ones
import os
import json
import torchvision.transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class LicensePlateDataset(Dataset):
    def __init__(self, image_dir, annotation_file):
        self.image_dir = image_dir
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
        self.image_files = list(self.annotations.keys())
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        # Convert to tensor (add your transforms)
        image = torchvision.transforms.ToTensor()(image)
        
        annotation = self.annotations[img_name]
        return image, annotation

# Create data loaders
train_dataset = LicensePlateDataset(
    f"{KAGGLE_DATA_PATH}/images/train",
    f"{KAGGLE_DATA_PATH}/annotations/train.json"
)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
```

### 6. Run Training

Execute training in notebook:

```python
# Set environment variables
import os
os.environ['KAGGLE_DATA_PATH'] = '/kaggle/input/license-plate-dataset'
os.environ['KAGGLE_OUTPUT_PATH'] = '/kaggle/working/models'

# Run training
!python train.py
```

**Expected output:**
```
============================================================
LICENSE PLATE RECOGNITION - TRAINING PIPELINE
============================================================
🚀 Training Detector Model...
Using device: cuda
Epoch 1/10
Loss: 0.5432
...
✅ Detector model saved: /kaggle/working/models/detector.pt
🚀 Training OCR Model...
...
✅ OCR model saved: /kaggle/working/models/ocr.pt
✅ Metadata saved: /kaggle/working/models/training_metadata.json
============================================================
✅ TRAINING COMPLETE!
============================================================
```

### 7. Verify Training Outputs

```python
import os

# Check output directory
!ls -lh /kaggle/working/models/

# Should show:
# detector.pt
# ocr.pt
# training_metadata.json
```

### 8. Quick Inference Test

```python
import torch

# Load models
detector = torch.load('/kaggle/working/models/detector.pt')
ocr = torch.load('/kaggle/working/models/ocr.pt')

# Test inference
test_image = torch.randn(1, 3, 224, 224).cuda()
detector.eval()
with torch.no_grad():
    output = detector(test_image)
print(f"Detector output shape: {output.shape}")
print("✅ Models working!")
```

---

## Model Export & Versioning

### 1. Create Model Storage Dataset

**First time only:**

1. Go to [kaggle.com/datasets](https://www.kaggle.com/datasets)
2. Click **New Dataset**
3. Name it: `license-plate-models`
4. Upload initial model files from `/kaggle/working/models/`
5. Set to **Private** (recommended for production models)

Note your dataset path:
```
YOUR_USERNAME/license-plate-models
```

### 2. Update Model Version (Automated)

In your Kaggle notebook, add a final cell:

```python
# Create dataset metadata
import json

metadata = {
    "title": "License Plate Recognition Models",
    "id": "YOUR_USERNAME/license-plate-models",  # Replace YOUR_USERNAME
    "licenses": [{"name": "Apache 2.0"}]
}

with open('/kaggle/working/dataset-metadata.json', 'w') as f:
    json.dump(metadata, f)

# Version the dataset with Kaggle CLI
# Set your version message here
version_message = "v1.2 - Improved accuracy to 95%"

!kaggle datasets version -p /kaggle/working/models -m "{version_message}"
```

### 3. Manual Version Update

Alternatively, download models and update via CLI:

```bash
# From notebook output
# Click: File → Download → models/

# Upload new version locally
kaggle datasets version -p ./models -m "v1.2 - Accuracy 95%"
```

### 4. Version Naming Convention

Use semantic versioning with descriptive messages:

```
v1.0 - Initial release
v1.1 - Fixed OCR character set
v1.2 - Improved accuracy to 95%
v2.0 - Complete architecture redesign
```

### 5. Verify Version

```bash
# Replace YOUR_USERNAME with your Kaggle username
kaggle datasets list-files YOUR_USERNAME/license-plate-models -v 2
```

---

## GitHub Integration

### 1. Update Workflow Configuration

If your Kaggle dataset path differs from default, update `.github/workflows/ci-test.yml`:

```yaml
- name: Download latest model from Kaggle
  run: |
    # Replace YOUR_USERNAME with your Kaggle username (e.g., kaffailevi)
    kaggle datasets download YOUR_USERNAME/license-plate-models -p models/ --unzip
```

### 2. Trigger CI Pipeline

CI runs automatically on:
- **Push** to `main` or `develop` branches
- **Pull requests** to `main`
- **Daily schedule** (2 AM UTC)

To manually trigger:

```bash
# Make a small change and push
git checkout -b test-new-model
echo "Testing model v1.2" >> README.md
git add README.md
git commit -m "Test new model version"
git push origin test-new-model
```

### 3. Monitor CI Progress

1. Go to repository → **Actions** tab
2. Click on your workflow run
3. Expand each step to see logs

**Expected steps:**
```
✅ Checkout repository
✅ Set up Python 3.10
✅ Install dependencies
✅ Download latest model from Kaggle
✅ Run unit tests
✅ Run model quality regression test
✅ Run inference probe
✅ Upload coverage reports
```

### 4. Review Test Results

CI validates:
- **Model loading**: Both `detector.pt` and `ocr.pt` load successfully
- **Inference speed**: <2 seconds for batch of 10 images
- **Model size**: Both models >100KB
- **Metadata**: `training_metadata.json` contains required fields

---

## Testing & Quality Gates

### 1. Local Testing Before Upload

Before versioning your model on Kaggle, test locally:

```bash
# Download your model version (replace YOUR_USERNAME)
kaggle datasets download YOUR_USERNAME/license-plate-models -p ./models --unzip

# Run full test suite
pytest tests/ -v

# Run probe test
python src/probe_test.py
```

### 2. Unit Tests

**Test inference pipeline:**

```bash
pytest tests/test_inference.py -v
```

Tests validate:
- Model loading
- Detection output shape `(batch, 5)` for `[x, y, w, h, conf]`
- OCR output decoding
- Character mapping `0-9A-Z`

### 3. Quality Tests

**Test model performance:**

```bash
pytest tests/test_model_quality.py -v
```

Quality gates:
- **Inference time**: Max 2 seconds for 10 images (224x224)
- **Model size**: Min 100KB for both models
- **Metadata**: Valid JSON with required fields

### 4. Probe Test (Quick Sanity Check)

```bash
python src/probe_test.py
```

Fast validation that models:
- Load without errors
- Accept correct input shapes
- Produce expected output shapes
- OCR decoding works

### 5. Interpreting Test Results

**✅ All tests pass:**
```
tests/test_inference.py::test_model_loading PASSED
tests/test_inference.py::test_detector_output_shape PASSED
tests/test_model_quality.py::test_inference_speed PASSED
======================== 3 passed in 5.23s ========================
```

**❌ Test failure example:**
```
FAILED tests/test_model_quality.py::test_inference_speed
AssertionError: Inference took 3.2s, exceeds max 2.0s
```

**Action**: Optimize model or increase threshold in `tests/test_model_quality.py`.

---

## Troubleshooting

### Issue: Kaggle API Not Working

**Symptoms:**
```
OSError: Could not find kaggle.json
```

**Solution:**
```bash
# Check file exists
ls -l ~/.kaggle/kaggle.json

# Verify permissions
chmod 600 ~/.kaggle/kaggle.json

# Test API (replace YOUR_USERNAME)
kaggle datasets list -u YOUR_USERNAME
```

---

### Issue: Model Size Too Small

**Symptoms:**
```
AssertionError: Model size 45KB, expected >100KB
```

**Solution:**

Models might have too few parameters. In `src/train.py`, increase model capacity:

```python
# Before
self.head = torch.nn.Linear(512, 36)

# After
self.head = torch.nn.Sequential(
    torch.nn.Linear(512, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 36)
)
```

---

### Issue: CUDA Out of Memory on Kaggle

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solution:**

Reduce batch size in training:

```python
# Before
train_loader = DataLoader(train_dataset, batch_size=64)

# After
train_loader = DataLoader(train_dataset, batch_size=16)
```

Or use gradient accumulation:

```python
accumulation_steps = 4
for i, (images, labels) in enumerate(train_loader):
    outputs = model(images)
    loss = criterion(outputs, labels) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

Or use mixed precision training with `torch.cuda.amp`:

```python
from torch.cuda.amp import autocast, GradScaler

# Setup (assumes model and optimizer already initialized)
device = torch.device("cuda")
model = model.cuda()
model.train()
scaler = GradScaler()

for images, labels in train_loader:
    images, labels = images.to(device), labels.to(device)
    optimizer.zero_grad()
    
    # Automatic mixed precision
    with autocast():
        outputs = model(images)
        loss = criterion(outputs, labels)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

---

### Issue: GitHub Secrets Not Working

**Symptoms:**
```
401 Unauthorized - Invalid Kaggle credentials
```

**Solution:**

1. Verify secrets are set correctly:
   - GitHub Repo → Settings → Secrets → Actions
   - Check `KAGGLE_USERNAME` and `KAGGLE_KEY` exist

2. Regenerate Kaggle API token:
   - Kaggle → Settings → API → Create New API Token
   - Update GitHub secret `KAGGLE_KEY` with new key

3. Check workflow YAML syntax:
   ```yaml
   # Correct
   ${{ secrets.KAGGLE_USERNAME }}
   
   # Wrong
   ${{ secrets.KAGGLE_USER }}  # Typo!
   ```

---

### Issue: Model Loading Fails in CI

**Symptoms:**
```
FileNotFoundError: models/detector.pt not found
```

**Solution:**

1. Verify Kaggle dataset is **Public** or secrets are configured
2. Check dataset path in workflow:
   ```yaml
   kaggle datasets download YOUR_USERNAME/license-plate-models
   ```
3. Ensure models were uploaded to Kaggle dataset:
   ```bash
   kaggle datasets list-files YOUR_USERNAME/license-plate-models
   ```

---

### Issue: Incompatible PyTorch Versions

**Symptoms:**
```
RuntimeError: version mismatch
```

**Solution:**

Match PyTorch versions between Kaggle and CI:

**In Kaggle notebook:**
```python
print(f"PyTorch version: {torch.__version__}")
# Output: 2.2.2
```

**In `requirements.txt`:**
```
torch==2.2.2
torchvision==0.17.2
```

Or save models with `torch.save(..., _use_new_zipfile_serialization=True)` for better compatibility.

---

### Issue: Inference Too Slow

**Symptoms:**
```
FAILED test_inference_speed - Took 3.5s, max 2.0s
```

**Solution:**

1. **Use model.eval()** to disable dropout/batchnorm:
   ```python
   model.eval()
   with torch.no_grad():
       output = model(input)
   ```

2. **Reduce model complexity** or use smaller backbone:
   ```python
   # Before
   self.backbone = torchvision.models.resnet50(pretrained=True)
   
   # After
   self.backbone = torchvision.models.resnet18(pretrained=True)
   ```

3. **Batch processing** instead of sequential:
   ```python
   # Process all 10 images at once
   batch = torch.stack(images)  # Shape: (10, 3, 224, 224)
   outputs = model(batch)
   ```

---

## Best Practices

### 1. Training

✅ **Do:**
- Use GPU accelerator in Kaggle (free P100/T4)
- Save checkpoints every N epochs
- Log training metrics (loss, accuracy)
- Use validation set to prevent overfitting
- Version control your training scripts

❌ **Don't:**
- Train on CPU (very slow for deep learning)
- Overwrite models without versioning
- Forget to save `training_metadata.json`
- Use hardcoded paths (use environment variables)

### 2. Model Versioning

✅ **Do:**
- Use semantic versioning (v1.0, v1.1, v2.0)
- Include metrics in version message ("v1.2 - 95% accuracy")
- Keep previous versions for rollback
- Document changes in version notes

❌ **Don't:**
- Delete old versions immediately
- Use vague messages ("updated model")
- Skip version metadata

### 3. CI/CD Integration

✅ **Do:**
- Test models locally before pushing
- Monitor CI logs for failures
- Set appropriate quality thresholds
- Use continue-on-error for non-critical steps

❌ **Don't:**
- Skip testing phase
- Ignore CI failures
- Set unrealistic thresholds
- Commit secrets to repository

### 4. Collaboration

✅ **Do:**
- Document architecture changes
- Share Kaggle notebooks (Private → Public when ready)
- Use branches for experimental models
- Review PRs before merging

❌ **Don't:**
- Push directly to main
- Change APIs without documentation
- Work on same model version simultaneously

### 5. Performance Optimization

✅ **Do:**
- Profile inference time regularly
- Use mixed precision training with `torch.cuda.amp` for memory efficiency and speed (note: affects training stability via gradient scaling, typically without impacting final model accuracy)
- Optimize model architecture iteratively
- Cache models in production

❌ **Don't:**
- Optimize prematurely
- Sacrifice accuracy for minimal speed gains
- Skip profiling

---

## Additional Resources

### Documentation

- [Kaggle API Documentation](https://github.com/Kaggle/kaggle-api)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)

### Project Files

- `README.md` - Project overview
- `src/train.py` - Training script
- `src/inference.py` - Inference module
- `tests/` - Test suite
- `.github/workflows/` - CI/CD pipelines

### External Tools

- [Kaggle Datasets CLI](https://github.com/Kaggle/kaggle-api#datasets)
- [PyTorch Hub](https://pytorch.org/hub/) - Pre-trained models
- [TorchVision Models](https://pytorch.org/vision/stable/models.html)

---

## Quick Reference

### Common Commands

```bash
# Kaggle
kaggle datasets list                          # List your datasets
kaggle datasets download USER/DATASET         # Download dataset
kaggle datasets version -p PATH -m "message"  # Create new version

# Testing
pytest tests/ -v                             # Run all tests
pytest tests/test_inference.py -v           # Unit tests only
python src/probe_test.py                    # Quick sanity check

# Git
git checkout -b feature/new-model           # Create branch
git add models/                             # Stage changes
git commit -m "Update model to v1.2"        # Commit
git push origin feature/new-model           # Push branch
```

### File Paths

| Component | Local Path | Kaggle Path |
|-----------|-----------|-------------|
| Dataset | `./data/` | `/kaggle/input/license-plate-dataset/` |
| Models | `./models/` | `/kaggle/working/models/` |
| Training script | `./src/train.py` | `/kaggle/working/train.py` |

### Model Specifications

| Model | Input Shape | Output Shape |
|-------|-------------|--------------|
| Detector | `(B, 3, 224, 224)` | `(B, 5)` - `[x, y, w, h, conf]` |
| OCR | `(B, 3, 32, 128)` | `(B, seq_len, 36)` - Character logits |

### Quality Thresholds

| Metric | Threshold |
|--------|-----------|
| Inference time | < 2.0 seconds (batch of 10) |
| Model size | > 100 KB |
| Min accuracy | 85% (adjust in tests) |

---

## Support

For issues or questions:

1. Check this guide's [Troubleshooting](#troubleshooting) section
2. Review project [README.md](README.md)
3. Open an issue on GitHub
4. Contact: **kaffailevi** (repository owner)

---

**Last Updated**: December 13, 2025  
**Version**: 1.0  
**Maintainer**: kaffailevi
