# 🅿️ License Plate Recognition

Machine Learning pipeline a rendszámtábla felismeréshez Kaggle + GitHub integrációval.

## 🏗️ Architektúra

```
Kaggle (GPU Training)
    ↓ [Export Model]
Kaggle Datasets (Model versioning)
    ↓ [Kaggle API]
GitHub Actions (CI/CD)
    ↓ [Download & Test]
Quality Gate ✓
    ↓ [Deploy]
Production
```

## 🚀 Gyorskezdés

### Prerequisites
- Python 3.10+
- Kaggle API key
- GitHub secrets: `KAGGLE_USERNAME`, `KAGGLE_KEY`

### Setup

```bash
git clone https://github.com/kaffailevi/license-plate-recognition.git
cd license-plate-recognition

# Create venv
python -m venv venv
source venv/bin/activate  # Linux/macOS
# vagy: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Local Testing

```bash
# Unit tesztek
pytest tests/test_inference.py -v

# Modell minőség tesztek
pytest tests/test_model_quality.py -v

# Probe test (sanity check)
python src/probe_test.py
```

## 🧠 Workflow

### 1️⃣ Kaggle-en Tréning

**Kaggle Notebook futtatása:**

```bash
# Notebook végén:
!mkdir -p /kaggle/working/models
!python /kaggle/input/training-script/train.py
```

**Model exportálás a Kaggle Datasetbe:**

```python
!kaggle datasets version -p /kaggle/working -m "Model v1.0 - Accuracy: 92%"
```

### 2️⃣ GitHub Actions Tesztelés

Automatikus:
1. Kaggle modellek letöltése
2. Unit tesztek futtatása
3. Modell minőség ellenőrzése
4. Inference probe

### 3️⃣ Deployment

✅ Ha mindent átmegy → Artifact mentés → Produktív deployment

## 📊 CI/CD Status

| Workflow | Status |
|----------|--------|
| CI Testing | [![CI](https://github.com/kaffailevi/license-plate-recognition/actions/workflows/ci-test.yml/badge.svg)](https://github.com/kaffailevi/license-plate-recognition/actions) |
| Model Validation | [![Model QA](https://github.com/kaffailevi/license-plate-recognition/actions/workflows/model-validation.yml/badge.svg)](https://github.com/kaffailevi/license-plate-recognition/actions) |

## 📁 Mappa struktúra

```
license-plate-recognition/
├── .github/workflows/       # CI/CD pipelines
├── src/                     # Tréning + inferencia
├── tests/                   # Unit & regression tesztek
├── models/                  # (Kaggle-ből húzott modellek)
├── kaggle/                  # Kaggle notebook + config
├── requirements.txt         # Python dependencies
└── README.md               # Ezt az fájlt
```

## 🔐 GitHub Secrets Setup

GitHub Settings → Secrets and variables → Actions

```
KAGGLE_USERNAME = "your_kaggle_username"
KAGGLE_KEY = "your_kaggle_api_key"
```

**Kaggle API key lekérése:**

```bash
# Kaggle Settings → API → Download kaggle.json
cat ~/.kaggle/kaggle.json
```

## 🎯 Modell Minőség Küszöbök

| Metrika | Küszöb |
|---------|--------|
| Min. pontosság | 85% |
| Max. inference time | 2s |
| Min. modell méret | 100 KB |

## 📚 Dokumentáció

- [Kaggle Development Guide](KAGGLE_DEVELOPMENT_GUIDE.md) - Complete walkthrough for training on Kaggle
- [Dataset Integration Guide](DATASET_INTEGRATION.md) - Using andrewmvd/car-plate-detection dataset
- [Kaggle Notebook Guide](kaggle/README.md)
- [Training Documentation](src/README.md)
- [CI/CD Workflows](.github/workflows/README.md)

## 🤝 Contributing

1. Fork repo
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m "Add amazing feature"`
4. Push branch: `git push origin feature/amazing-feature`
5. Open Pull Request

## 📝 Licencia

MIT License - lásd [LICENSE](LICENSE) fájl

## 👨‍💻 Szerző

**kaffailevi**

---

**Status:** 🚀 Development in progress