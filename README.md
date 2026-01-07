# Satellite Imagery-Based Property Valuation: Multimodal Regression Pipeline

<div align="center">

![Status](https://img.shields.io/badge/Status-Complete%20%26%20Production%20Ready-brightgreen)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

**A Deep Learning System for Real Estate Price Prediction Using Satellite Imagery + Tabular Data**

[Quick Start](#-quick-start) • [Features](#-features) • [Results](#-key-results) • [Documentation](#-documentation)

</div>

---

## 📋 Project Overview

This repository contains a complete **multimodal deep learning pipeline** that predicts residential property prices by combining:

1. **Satellite imagery** (600×600px from Mapbox API)
2. **Tabular features** (46 engineered attributes: square footage, grade, condition, etc.)

### Why This Matters

Real estate valuation traditionally relies on tabular data alone (bedrooms, bathrooms, sqft, etc.). This project demonstrates that **satellite imagery adds 23-33% predictive accuracy improvement** by capturing:

- Visual "curb appeal" (maintenance, architecture)
- Neighborhood characteristics (density, green space, road infrastructure)
- Environmental context (proximity to water, parks, highways)

### Key Results

| Metric | Tabular Only | Multimodal | Improvement |
|--------|-------------|-----------|-------------|
| **R² Score** | 0.7053 | **0.8676** | +23.0% |
| **RMSE** | $207.8K | **$139.3K** | -33.0% |
| **Convergence** | Epoch 20 | **Epoch 4** | 4-5x faster |

**Practical Impact:** Reduces valuation error from ±$104K to ±$70K on a $500K property.

---

## 🚀 Quick Start

### Prerequisites

- **Python:** 3.8 or higher
- **NVIDIA GPU:** Optional but recommended (8+ GB VRAM for 32 batch size)
- **Disk Space:** 20+ GB (for satellite images + datasets)
- **Internet Connection:** Required for Mapbox API

### 1️⃣ Clone Repository & Install Dependencies

```bash
# Clone (or download) the project
git clone https://github.com/yourusername/satellite-property-valuation.git
cd satellite-property-valuation

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2️⃣ Download Satellite Images (Optional - Already Included)

Images are already downloaded and included. If you need to update or re-download:

```bash
# Set Mapbox API key in data_fetcher.py
python data_fetcher.py
```

### 3️⃣ Run Preprocessing & Feature Engineering

```bash
jupyter notebook merged_preprocessing.ipynb
```

- **Runtime:** 5-10 minutes
- **Output:** Scaled features saved to `processed_data/`

### 4️⃣ Train & Compare Models

```bash
jupyter notebook training4.ipynb
```

- **Runtime:** 30-40 minutes (GPU), 3-4 hours (CPU)
- **Output:** 
  - `Tabular_Baseline_best.pth` (baseline model weights)
  - `Multimodal_Gated_best.pth` (champion model weights)
  - `submission.csv` (predictions)

### 5️⃣ View Results

Models automatically generate comparison plots:
- Validation MSE over epochs
- Validation R² over epochs
- Performance metrics summary

---

## 📁 Project Structure

```
satellite-property-valuation/
│
├── processed_data/
│   ├── X_train.csv
│   ├── y_train.csv
│   ├── X_test.csv
│   ├── meta_train.csv
│   ├── meta_test.csv
│   └── scaler.pkl
│
├── visualizations/
│   ├── eda_price_dist.png
│   ├── eda_log_price_dist.png
│   ├── eda_correlation.png
│   ├── eda_grade_vs_price.png
│   ├── eda_sqft_vs_price.png
│   └── eda_geo_price.png
│
├── 23118052_final.csv
├── 23118052_report.pdf
├── README.md
├── data_fetcher.py
├── data_fetcher_test.py
├── merged_preprocessing.ipynb
├── requirements.txt
├── test.csv
├── train.csv
└── training.ipynb

```

---

## 🛠️ Installation & Setup

### Step 1: System Requirements

**Minimum:**
- CPU: 8-core modern processor
- RAM: 16 GB
- Storage: 20 GB SSD
- Python: 3.8+

**Recommended:**
- GPU: NVIDIA RTX 3060+ (12GB VRAM)
- RAM: 32 GB
- Storage: 50 GB SSD
- Python: 3.10+

### Step 2: Virtual Environment Setup

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Verify activation (should show (venv) prefix)
python --version
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip setuptools wheel

# Install from requirements.txt
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
# Test PyTorch installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# Test GPU availability
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"

# Test key packages
python -c "import pandas; import numpy; print('✓ All core packages installed')"
```

---

## 📚 Detailed Usage Guide

### Phase 1: Data Preparation (Optional)

If satellite images are missing, download them:

```bash
# Edit data_fetcher.py and set your Mapbox API key:
# API_KEY = "your_mapbox_key_here"

python data_fetcher.py
```

**Expected output:**
```
Reading train_original.csv...
Dataset loaded. Processing ALL 16209 rows.
Starting download...
100%|████████| 16209/16209 [12:34<00:00, 21.5it/s]
Done! Downloaded images saved to 'satellite_images'
```

### Phase 2: Feature Engineering

Open Jupyter and run the preprocessing notebook:

```bash
jupyter notebook merged_preprocessing.ipynb
```

**This notebook does:**
1. ✅ Loads CSV data
2. ✅ Verifies image synchronization
3. ✅ Cleans missing values (median imputation)
4. ✅ Engineers 46 features from 17 raw attributes
5. ✅ Scales features to zero-mean, unit-variance
6. ✅ Generates 6 EDA visualizations
7. ✅ Saves processed data to `processed_data/`

**Output files created:**
- `processed_data/X_train.csv` (16,209 samples × 46 features)
- `processed_data/y_train.csv` (target prices)
- `processed_data/X_test.csv` (test features)
- `processed_data/scaler.pkl` (for inference)
- `visualizations/*.png` (6 EDA plots)

**Runtime:** ~5-10 minutes

### Phase 3: Model Training

```bash
jupyter notebook training4.ipynb
```

**This notebook:**
1. ✅ Defines two models:
   - TabularNet: MLP baseline (46 features only)
   - MultimodalNet: CNN + MLP with gated fusion
2. ✅ Trains both for 20 epochs
3. ✅ Compares performance side-by-side
4. ✅ Saves best model weights
5. ✅ Generates predictions on test set

**Output files created:**
- `Tabular_Baseline_best.pth`
- `Multimodal_Gated_best.pth` ⭐
- `submission.csv` (predictions)
- Training curves (MSE & R² plots)

**Runtime:**
- GPU (RTX 3060+): 30-40 minutes
- GPU (T4): 50-60 minutes
- CPU: 3-4 hours

**Expected Results (Epoch 20):**
```
Tabular Baseline:
  Best Val MSE: $43.18B
  Best Val R²: 0.7053

Multimodal Gated:
  Best Val MSE: $19.41B
  Best Val R²: 0.8556  ✓ BETTER!
```

---

## 📊 Model Architecture Details

### Tabular Baseline (TabularNet)

```
Input: 46 tabular features
  ↓
Dense(128) + ReLU + BatchNorm + Dropout(0.3)
  ↓
Dense(64) + ReLU
  ↓
Output: Price prediction (1D)

Total parameters: ~7,000
```

### Multimodal Model (MultimodalNet) ⭐

```
IMAGE BRANCH:                    TABULAR BRANCH:
  600×600 satellite image          46 scaled features
  ↓                                ↓
  ResNet-18 backbone              Dense(128) + ReLU
  ├─ Freeze layers 1-6            ├─ BatchNorm(128)
  └─ Fine-tune layers 7-8         ├─ Dropout(0.3)
  ↓                                └─ Dense(64) + ReLU
  512-D image embedding            64-D tabular embedding
  ↓                                ↓
  ┌─────────────────────────────────┐
  │   GATED FUSION MECHANISM         │
  │  ┌──────────────────────────┐   │
  │  │ Concatenate [512, 64]    │   │
  │  │ ↓                        │   │
  │  │ Attention Gate α ∈ [0,1] │   │
  │  │ ↓                        │   │
  │  │ Weighted img = img × α   │   │
  │  └──────────────────────────┘   │
  └─────────────────────────────────┘
  ↓
  Concatenate [weighted_img, tabular]
  ↓
  Dense(256) + ReLU + Dropout(0.4)
  Dense(64) + ReLU
  ↓
  Output: Price prediction (1D)

Total parameters: ~11.7M (ResNet backbone pretrained)
```

### Why Gated Fusion?

The **learned gate α** allows adaptive per-sample importance:
- **α → 1.0:** Images highly informative (unique neighborhoods, waterfront)
- **α → 0.5:** Balanced multimodal contribution (standard suburbs)
- **α → 0.0:** Images less relevant (homogeneous areas)

This prevents images from overly influencing predictions where tabular data is already highly predictive.

---

## 🎯 Making Predictions on New Data

### Load Trained Model

```python
import torch
import pandas as pd
import pickle
from PIL import Image
from torchvision import transforms

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultimodalNet(tabular_input_dim=46)
model.load_state_dict(torch.load("Multimodal_Gated_best.pth"))
model.to(device)
model.eval()

# Load scaler
with open("processed_data/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
```

### Prepare Input Data

```python
# Load your test data
X_test = pd.read_csv("processed_data/X_test.csv").values
X_test_scaled = scaler.transform(X_test)

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                        [0.229, 0.224, 0.225])
])

# Load satellite image
image = Image.open("satellite_images/12345.jpg").convert('RGB')
image_tensor = transform(image).unsqueeze(0).to(device)
```

### Get Price Prediction

```python
tabular_tensor = torch.tensor(X_test_scaled[0:1], dtype=torch.float32).to(device)

with torch.no_grad():
    price_prediction = model(image_tensor, tabular_tensor).item()

print(f"Predicted Price: ${price_prediction:,.2f}")
```

---

## 🔧 Configuration & Customization

### Modify Training Hyperparameters

Edit the configuration in `training4.ipynb` (Cell 1):

```python
# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "processed_data"

BATCH_SIZE = 32           # Reduce to 16 if OOM errors
LEARNING_RATE = 1e-3      # Adjust if loss diverges
EPOCHS = 20               # Can increase to 30-50
DROPOUT_RATE = 0.3        # Regularization (0.2-0.5 typical)
```

### Add Custom Features

In `merged_preprocessing.ipynb`, modify the `engineer_features()` function:

```python
def engineer_features(df):
    # ... existing features ...
    
    # Add your custom feature:
    df['custom_feature'] = df['sqft_living'] / (df['age'] + 1)
    
    return df
```

### Adjust Model Architecture

In `training4.ipynb`, modify the `MultimodalNet` class:

```python
class MultimodalNet(nn.Module):
    def __init__(self, tabular_input_dim):
        super().__init__()
        
        # Increase depth
        self.image_extractor = nn.Sequential(
            # ... more layers ...
        )
        
        # Adjust fusion mechanism
        self.gate = nn.Sequential(
            nn.Linear(512 + 64, 64),  # Increased hidden size
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
```

---

## 🐛 Troubleshooting

### Issue: `CUDA Out of Memory (OOM)`

**Symptoms:** Error like `RuntimeError: CUDA out of memory`

**Solution:**
```python
BATCH_SIZE = 16  # or 8
```

---

### Issue: `ModuleNotFoundError: No module named 'torch'`

**Symptoms:** `ImportError` when running notebooks

**Solution:**
```bash
pip install --upgrade torch torchvision torchaudio
```

---

### Issue: Images not found / Preprocessing fails

**Check:**
```bash
# Verify satellite_images directory exists
ls satellite_images/ | wc -l  # Should show ~16,000

# Check paths in meta_train.csv
head processed_data/meta_train.csv
```

---

### Issue: Model isn't improving / Loss diverges

**Diagnostic steps:**
1. Check data scaling: `processed_data/X_train.csv` should have mean ≈ 0, std ≈ 1
2. Reduce learning rate: `LEARNING_RATE = 1e-4`
3. Verify target variable: `processed_data/y_train.csv` for outliers

---

### Issue: Data loading is slow

**Optimization:**
```python
# Use subset for testing
df = df.head(100)

# Increase num_workers in DataLoader
DataLoader(..., num_workers=4)
```

---

## 📈 Performance Monitoring

### Training Metrics

During training, the notebook prints per-epoch metrics:

```
Epoch 1 | Train MSE: 255.67B | Val MSE: 59.85B | R²: 0.5916
Epoch 2 | Train MSE: 46.92B  | Val MSE: 33.70B | R²: 0.7700
...
Epoch 20| Train MSE: 16.01B  | Val MSE: 21.16B | R²: 0.8556
```

**What to look for:**
- ✅ Train MSE decreases epoch-to-epoch
- ✅ Val MSE plateaus (best model selected automatically)
- ✅ R² increases monotonically
- ⚠️ Val loss increasing while train decreases = overfitting (reduce dropout)

---

## 📖 Feature Engineering Details

### 46 Total Features Breakdown

**Original Features (17):**
bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront, view, condition, grade, sqft_above, sqft_basement, yr_built, yr_renovated, lat, long, sqft_living15, sqft_lot15

**Engineered Features (29):**

| Category | Features | Count |
|----------|----------|-------|
| **Temporal** | year, month, quarter, season | 5 |
| **Spatial** | distance_from_center, lat², long², lat×long | 5 |
| **Area Ratios** | log(sqft_living), sqft_per_lot, ratios | 4 |
| **Neighborhood** | living_vs_neighborhood, lot_vs_neighborhood | 2 |
| **Condition/Grade** | is_excellent, is_poor, is_high_grade, grade² | 6 |
| **Amenities** | has_view, is_waterfront, has_basement, multi_level | 4 |
| **Rooms** | total_rooms, bed_bath_ratio | 2 |
| **Age** | age, age², was_renovated, years_since_renovation | 4 |

---

## 📊 EDA & Visualization

The preprocessing notebook generates 6 visualizations saved to `visualizations/`:

1. **eda_price_dist.png** - Raw price distribution (right-skewed)
2. **eda_log_price_dist.png** - Log-transformed prices (approximately normal)
3. **eda_correlation.png** - Feature correlation heatmap
4. **eda_grade_vs_price.png** - Grade quality premium analysis
5. **eda_sqft_vs_price.png** - Living area vs. price scatter
6. **eda_geo_price.png** - Geospatial price distribution map

---

## 📚 References & Documentation

### Papers

- He, K., Zhang, X., Ren, S., & Sun, J. (2015). **Deep Residual Learning for Image Recognition.** arXiv:1512.03385
- Selvaraju et al. (2017). **Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.** ICCV
- Kingma & Ba (2014). **Adam: A Method for Stochastic Optimization.** arXiv:1412.6980

### External Resources

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [ResNet Paper](https://arxiv.org/pdf/1512.03385.pdf)
- [Mapbox API Documentation](https://docs.mapbox.com/api/maps/static/)
- [scikit-learn Preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html)

---

## 💡 Tips & Best Practices

### For GPU Users

```bash
# Monitor GPU usage during training
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader -l 1
```

### For CPU Users

```python
# Reduce batch size and epochs for faster experimentation
BATCH_SIZE = 8
EPOCHS = 5
```

### For Production Deployment

```python
# Convert to ONNX for cross-platform compatibility
import torch.onnx

dummy_img = torch.randn(1, 3, 224, 224)
dummy_tab = torch.randn(1, 46)

torch.onnx.export(model, (dummy_img, dummy_tab), 
                  "multimodal_model.onnx")
```
