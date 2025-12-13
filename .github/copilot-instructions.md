# AI Agent Guide: License Plate Recognition

These instructions make AI coding agents productive in this repo.

## Big Picture
- Pipeline: Kaggle GPU training → publish model artifacts → GitHub Actions download + tests → Quality gate → deploy.
- Components:
  - `src/train.py`: trains two PyTorch models (detector, OCR) and saves artifacts + `training_metadata.json`.
  - `src/inference.py`: loads models, runs detection and OCR, provides decode utilities.
  - `tests/`: unit + quality checks that gate CI.
  - `kaggle/`: notebook + config used for training/publishing models.

## Model Contract (Critical)
- Expected artifacts under `models/`:
  - `models/detector.pt` and `models/ocr.pt`
  - `models/training_metadata.json` with keys: `training_date`, `pytorch_version`, `device`.
- Inference currently calls `torch.load(<.pt>)` and then `.eval()` and forward. This requires saving FULL modules, not just `state_dict`s.
  - Preferred: `torch.save(model, path)` in training; load with `torch.load(path)`.
  - Alternative: reconstruct modules in inference and `load_state_dict`; if you do this, update `LicensePlateRecognizer.__init__` accordingly.

## Shapes & Decoding
- Detector input: `torch.Tensor[B, 3, 224, 224]` → output: `numpy[B, 5]` (`[x,y,w,h,conf]`).
- OCR input: `torch.Tensor[B, 3, 32, 128]` → output logits → `argmax` indices per item.
- `decode_plate(indices)` maps `0-9A-Z` to text; pass a flat array for one plate.

## Tests & Quality Gates
- Unit tests: `pytest tests/test_inference.py -v`
- Quality tests: `pytest tests/test_model_quality.py -v`
- Thresholds (see `tests/test_model_quality.py`):
  - `min_accuracy` placeholder (enforced indirectly)
  - `max_inference_time`: `2.0s` on a batch of `10 x 224x224`
  - `min_model_size`: `100000` bytes for both detector and ocr
- Probe: `python src/probe_test.py` runs fast sanity checks (load, inference, OCR).

## Training Workflow (Kaggle)
- Environment variables used: `KAGGLE_DATA_PATH`, `KAGGLE_OUTPUT_PATH` (default `./data`, `./models`).
- `src/train.py` trains both models with dummy loops; replace with real dataloaders.
- Publish from Kaggle: version a dataset and include `models/*.pt` + `training_metadata.json`.

## CI/CD Notes
- GitHub Actions expect models to exist or tests will skip/fail.
- Editing files under `.github/workflows/` requires a token with `workflow` scope; use a PAT or `gh auth login`.
- Keep CI fast: prefer probe + small unit tests for PRs; heavy training belongs to Kaggle.

## Project Conventions
- Logging via `logging` across scripts; surface actionable errors (e.g., missing model file).
- Default model paths are relative (`models/detector.pt`, `models/ocr.pt`).
- Batch-first tensors; return numpy from inference for tests.
- Use `pytest` with fixtures; skip tests gracefully if models are unavailable.

## Typical Local Flow
1. Create `venv` and `pip install -r requirements.txt`.
2. Ensure `models/` contains valid `.pt` files + metadata.
3. Run `python src/probe_test.py`.
4. Run `pytest -q` or specific test files.

## Examples
- Save full model in training:
  ```python
  torch.save(model, os.path.join(KAGGLE_OUTPUT_PATH, "detector.pt"))
  ```
- Update inference to load `state_dict` (if you keep `state_dict` saving):
  ```python
  self.detector = LicensePlateDetector(); self.detector.load_state_dict(torch.load(detector_path))
  self.detector.eval()
  ```

## Where to Look
- Architecture overview: `README.md`
- Training details: `src/train.py`
- Inference API: `src/inference.py`
- CI gates and thresholds: `tests/`
- Kaggle notebook & config: `kaggle/`

If any part is unclear (e.g., exact CI expectations, dataset schema), tell me what you need and I’ll refine this guide.