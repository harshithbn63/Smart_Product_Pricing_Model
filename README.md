## Smart Product Pricing

Turn messy product data into clean, confident prices. This pipeline embeds multimodal product information (text + images), learns from historical signals, and predicts prices you can trust.

---

### ✨ Highlights
- Multimodal embeddings: JinaCLIP for robust text/image understanding
- Two-stage modeling: baseline ensemble (3) → main ensemble (5)
- Balanced metrics: SMAPE, R², MAE
- Reproducible: notebooks + src utilities + pinned requirements

---

### 📦 Project Structure
```
Smart_Product_Pricing/
├── notebooks/
│   ├── 01_embedding_file.ipynb        # JinaCLIP embeddings
│   ├── 02_model1_ensemble3.ipynb      # Baseline ensemble (3)
│   ├── 03_main_model_ensemble5.ipynb  # Final ensemble (5)
│   └── Documentation_template.md      # Design notes
├── data/
│   ├── train.csv                      # Training data
│   ├── test.csv                       # Test data
│   └── test_out.csv                   # Predictions
├── src/
│   ├── embedding_utils.py             # Embedding helpers
│   ├── model_training.py              # MLP + ensembling
│   └── evaluation.py                  # SMAPE, R², MAE
├── models/
│   └── final_model_weights.pth        # Trained model
├── requirements.txt
└── README.md
```

---

### 🏗️ System Architecture

Bring your data in at the left; collect confident prices at the right. The flow stitches together multimodal embeddings, compact feature engineering, and a resilient MLP ensemble.

```
Raw CSVs (train.csv / test.csv)
        │
        ├──► Notebook: 01_embedding_file.ipynb
        │       ├─ Pull images async (cache) + handle fallbacks
        │       ├─ Encode text (JinaCLIP) + encode images (JinaCLIP)
        │       └─ L2-normalize + fuse (concat or sum) → embeddings.npy
        │
        ├──► Notebook: 02_model1_ensemble3.ipynb
        │       ├─ Standardize (StandardScaler)
        │       ├─ Reduce dims (PCA)
        │       ├─ Train Residual MLP (x3 seeds)
        │       └─ Average predictions
        │
        ├──► Notebook: 03_main_model_ensemble5.ipynb
        │       ├─ Same scaler/PCA
        │       ├─ Train Residual MLP (x5 seeds)
        │       └─ Save artifacts (models/, scaler.pkl, pca.pkl)
        │
        └──► Inference
                ├─ Scale + PCA (loaded)
                ├─ Ensemble forward pass
                └─ Write data/test_out.csv
```

- Core modules:
  - `src/embedding_utils.py`: async image fetch, JinaCLIP loading, text/image encoding, fusion (L2-normalized).
  - `src/model_training.py`: Residual MLP, scaler+PCA prep, early stopping, ensemble save/load, end-to-end predict.
  - `src/evaluation.py`: SMAPE, R², MAE.

Visual slots (add your images by placing files and updating paths):

![Architecture Overview Placeholder](notebooks/images/architecture_overview.png)

![Model Pipeline Placeholder](notebooks/images/model_pipeline.png)

---

### 🚀 Quickstart
1) Create an environment and install deps:
   - `python -m venv .venv && .venv\\Scripts\\activate`
   - `pip install -r Smart_Product_Pricing/requirements.txt`

2) Open notebooks:
   - `jupyter lab Smart_Product_Pricing/notebooks`

3) Run in order:
   - `01_embedding_file.ipynb` → generate embeddings
   - `02_model1_ensemble3.ipynb` → baseline ensemble + metrics
   - `03_main_model_ensemble5.ipynb` → final ensemble + save weights

---

### 🧠 Modeling Notes
- Embeddings: JinaCLIP captures cross-modal semantics for products
- Architecture: MLP heads on embeddings, aggregated via ensembling
- Why SMAPE? Better fairness on low-price items than plain MAPE

---

### 📊 Metrics Reference
- SMAPE, R², MAE implemented in `src/evaluation.py`

---

### 🗂 Data Contracts
- `data/train.csv`: includes target price and features
- `data/test.csv`: same schema minus target
- `data/test_out.csv`: predictions output

---

### 🔧 Extending
- Swap/add embedding models in `src/embedding_utils.py`
- Experiment with heads/optimizers in `src/model_training.py`
- Add business-rule post-processing before writing `test_out.csv`

---

### ✅ Repro Tips
- Pin random seeds
- Keep requirements frozen
- Log library versions per run

---

### 📜 License
MIT (or your preferred)

Happy pricing! 💸

