# ML Challenge 2025: Smart Product Pricing Solution

**Team Name:** The PowerPuff Girls  
**Team Members:** 4
**Submission Date:** 2025-10-13

---

## 1. Executive Summary
We predict product prices from catalog text by generating semantic embeddings and training a lightweight neural regressor. Text is transformed into dense vectors (`full_train_embeddings.npy`, `full_embeddings.npy`) and scaled (`scaler.pkl`), then a PyTorch MLP (`best_model_weights.pth`) produces prices. The pipeline outputs predictions in `test_pred1.csv` and formats the final submission in `test_out.csv`.

---

## 2. Methodology Overview

### 2.1 Problem Analysis
The task is to infer fair prices from product metadata where most signal resides in `catalog_content` (item name, bullet points, product description, size/units). Images are referenced but not required for a strong baseline. Prices are positively skewed with long tails and frequent unit/size tokens embedded in free text.

**Key Observations:**
- Pricing correlates with package size/units and product category terms present in `catalog_content`.
- `train.csv` columns: `sample_id`, `catalog_content`, `image_link`, `price`.
- `test.csv` columns: `sample_id`, `catalog_content`, `image_link` (no `price`).
- Final submission format (`test_out.csv`): `sample_id,price`.

### 2.2 Solution Strategy
Transform free text into fixed-length embeddings and fit a non-linear regressor on top.

**Approach Type:** Single Model (Text-embedding + MLP Regressor)  
**Core Innovation:** Use precomputed semantic embeddings of `catalog_content` to capture category and size semantics, followed by a calibrated neural regressor; simple, fast, and robust to noisy text.

---

## 3. Model Architecture

### 3.1 Architecture Overview
CSV → text preprocessing → text embeddings (`.npy`) → feature scaling (`scaler.pkl`) → PyTorch MLP (`.pth`) → predictions (`test_pred1.csv`) → submission (`test_out.csv`).

### 3.2 Model Components

**Text Processing Pipeline:**
- Preprocessing steps: basic normalization (whitespace cleanup, lowercasing), concatenation of name/bullets/description from `catalog_content`.
- Embeddings: precomputed dense vectors stored in `full_train_embeddings.npy` and `full_embeddings.npy` (fixed dimensionality throughout training/inference).
- Feature scaling: standardization via `scaler.pkl` to stabilize training and inference.

**Regressor (PyTorch):**
- Model type: Multi-Layer Perceptron regressor saved as `best_model_weights.pth`.
- Loss/optimization: regression objective optimized to reduce SMAPE-proxy (e.g., SmoothL1/MAE family) with early stopping on validation.
- Inference: loads scaler and weights to produce prices; writes raw predictions to `test_pred1.csv` and submission to `test_out.csv`.

**Image Processing Pipeline (not used):**
- Images are referenced via `image_link` but not consumed in this version.

---

## 4. Model Performance

### 4.1 Validation Results
- SMAPE Score: Not recorded in repository artefacts.
- Other Metrics: —

Notes: Validation artifacts/metrics aren’t stored in the workspace; reproduce by re-running the training notebook (`model.ipynb`).

## 5. Conclusion
Text-only semantic embeddings paired with a small neural regressor provide an effective, compute-efficient baseline for price prediction. The pipeline is simple to reproduce and extend (e.g., add image features or domain features like size normalization) while maintaining fast inference.

---

## Appendix

### A. Code artefacts
- Notebooks: `embedding_file.ipynb` (embedding generation), `model.ipynb` (training/inference)
- Embeddings: `full_train_embeddings.npy` (train), `full_embeddings.npy` (test)
- Scaler: `scaler.pkl`
- Model weights: `best_model_weights.pth`
- Predictions: `test_pred1.csv`
- Submission file: `test_out.csv` (columns: `sample_id,price`)

### B. Additional Results
- Example submission rows are present in `test_out.csv`.

---

**Note:** This document reflects the artefacts present in the repository. If you re-run training, please record validation metrics and update Section 4 accordingly.