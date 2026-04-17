# Fake Currency Detection System — LeJEPA Backend Implementation Plan

## 1. Background & JEPA Variant Analysis

> [!NOTE]
> **Hardware**: CPU-only training (no CUDA GPU). Using **ViT-Tiny** encoder for efficiency.
> **Port**: Backend runs on **port 8001**.
> **Training Endpoint**: Open, no password protection.

### JEPA Variants Comparison

| Variant | Year | Core Innovation | Best For | Suitability for Currency Detection |
|---------|------|----------------|----------|----------------------------------|
| **I-JEPA** | 2023 | Masked image patch prediction in latent space | General image SSL | ⭐⭐⭐ Good foundation, but lacks theoretical stability guarantees |
| **LeJEPA** | 2025 | SIGReg (isotropic Gaussian regularization), heuristic-free, ~50 LOC regularizer | Image SSL with provable stability | ⭐⭐⭐⭐⭐ **Best choice** — stable training, efficient, domain-agnostic, works on small datasets |
| **V-JEPA 2.1** | 2026 | Dense predictive loss, deep self-supervision across layers, multi-modal tokenizers (2D image + 3D video), up to 2B params (ViT-G) | Large-scale video/image representation, robotics, depth estimation, action anticipation | ⭐⭐ Too heavy — smallest model is ViT-B (~80M params), requires GPU, designed for massive datasets (163M images/videos) |
| **Causal-JEPA** | 2026 | Object-level masking, causal inductive bias | Object-centric world models, robotics | ⭐⭐ Overkill — designed for multi-object interaction reasoning |
| **LeWorldModel** | 2026 | Next-embedding prediction + SIGReg for world modeling | RL planning, physical simulations | ⭐ Wrong paradigm — sequential state prediction, not classification |
| **ThinkJEPA** | 2026 | Dual-temporal VLM + JEPA fusion for long-horizon video | Video forecasting, embodied AI | ⭐ Wrong modality — video-centric, requires VLM backbone |
| **Attention-JEPA** | N/A | Not a distinct published architecture — refers to JEPA models using attention-based (ViT) encoders/predictors | General JEPA implementations | ⭐⭐⭐ Generic term, not a specific method |

### V-JEPA 2.1 — Why It Was Considered But Rejected

V-JEPA 2.1 (Meta FAIR, March 2026) is the most powerful JEPA variant available, achieving SOTA on robotics, depth estimation, and action anticipation. However, it is **not suitable** for this project:

| Factor | V-JEPA 2.1 | LeJEPA | Winner for Our Case |
|--------|-----------|--------|-------------------|
| **Smallest model** | ViT-B (~80M params) | ViT-Tiny (~5M params) | ✅ LeJEPA |
| **Training hardware** | Requires high-end GPU (16GB+ VRAM) | Runs on **CPU** | ✅ LeJEPA |
| **Training data scale** | Designed for 163M images/videos | Excels on small datasets (~7K images) | ✅ LeJEPA |
| **Complexity** | Multi-modal tokenizers, deep self-supervision, complex recipe | ~50 lines SIGReg, no heuristics | ✅ LeJEPA |
| **Primary domain** | Video + images (temporal dynamics focus) | Images (static classification focus) | ✅ LeJEPA |
| **Collapse prevention** | Standard SSL tricks (EMA, masking) | Provable SIGReg guarantee | ✅ LeJEPA |
| **Frozen-backbone eval** | Excellent (but needs pre-trained checkpoint download ~GB) | Train from scratch on domain data | ✅ LeJEPA |
| **Domain-specific pretraining** | Overkill for currency images | Ideal — in-domain pretraining outperforms transfer | ✅ LeJEPA |

> [!CAUTION]
> V-JEPA 2.1's smallest variant (ViT-B, ~80M params) would take **days to train on CPU** and requires downloading multi-GB pre-trained checkpoints. Even with frozen-backbone fine-tuning, its representations are optimized for generic video/scene understanding — not the fine-grained texture/pattern discrimination needed for counterfeit detection. LeJEPA trained from scratch on our domain-specific currency dataset will outperform a generic V-JEPA 2.1 transfer for this task.

### Why LeJEPA is the Best Choice

> [!IMPORTANT]
> **LeJEPA** is the optimal architecture for this task for the following reasons:

1. **Heuristic-Free Training**: No stop-gradients, no EMA teacher, no complex schedules — just prediction loss + SIGReg. This means **stable training on our relatively small dataset** (~7,445 images).
2. **Theoretical Guarantees**: Provably prevents representation collapse, which is critical when fine-tuning on domain-specific data (currency images) that differs from ImageNet.
3. **Architecture Agnostic**: Works with ViT-Small/Base (we'll use **ViT-Tiny** for CPU-only training efficiency).
4. **Excellent Low-Shot Performance**: I-JEPA/LeJEPA excel in low-data regimes. Our dataset of ~7,445 images is small by deep learning standards, making this crucial.
5. **Simple Implementation**: SIGReg adds ~50 lines of PyTorch code. The entire training pipeline is clean and maintainable.
6. **High Correlation Between SSL Loss and Downstream Accuracy**: We can monitor training quality without a labeled validation set during pre-training phase.
7. **CPU-Friendly**: ViT-Tiny with LeJEPA trains in ~4-8 hours on CPU. V-JEPA 2.1 would take days.

### Training Strategy: Two-Phase Approach

```
Phase 1: Self-Supervised Pre-training (LeJEPA)
   ┌─────────────┐     ┌──────────┐     ┌─────────────┐
   │ Context      │────▶│ Predictor│────▶│ Predicted   │
   │ Encoder (ViT)│     │ (Narrow  │     │ Target Emb  │
   │              │     │  ViT)    │     │             │
   └─────────────┘     └──────────┘     └──────┬──────┘
                                                │  L2 Loss
   ┌─────────────┐                       ┌──────▼──────┐
   │ Target       │──────────────────────▶│ Target Emb  │
   │ Encoder (ViT)│  (shared weights,    │ (ground     │
   │              │   no stop-gradient)   │  truth)     │
   └─────────────┘                       └─────────────┘
         │
         └──── + SIGReg (Isotropic Gaussian Regularization)

Phase 2: Supervised Fine-tuning (Classification Head)
   ┌─────────────┐     ┌──────────────┐     ┌────────┐
   │ Pre-trained  │────▶│ Global Avg   │────▶│ Linear │──▶ REAL/FAKE
   │ ViT Encoder  │     │ Pool         │     │ Head   │
   └─────────────┘     └──────────────┘     └────────┘
```

---

## 2. Existing Frontend API Contract

Based on the frontend code analysis, the backend **must** provide these exact endpoints:

| Method | Endpoint | Request Body | Response |
|--------|----------|-------------|----------|
| `POST` | `/api/v1/analyze` | `{ image: string (base64), source: "upload" \| "camera" }` | `AnalysisResult` |
| `GET` | `/api/v1/analyze/history` | Query: `page`, `limit`, `filter` | `HistoryResponse` |
| `GET` | `/api/v1/analyze/history/{id}` | — | `AnalysisResult` |
| `DELETE` | `/api/v1/analyze/history/{id}` | — | `204 No Content` |

### `AnalysisResult` Response Shape (must match exactly):

```json
{
  "id": 1,
  "result": "REAL" | "FAKE",
  "confidence": 0.95,
  "currency_denomination": "500",
  "denomination_confidence": 0.92,
  "analysis": {
    "cnn_classification": {
      "result": "REAL",
      "confidence": 0.96,
      "model": "LeJEPA-ViT-T",
      "processing_time_ms": 45
    },
    "watermark": {
      "status": "present" | "absent" | "unknown",
      "confidence": 0.85,
      "location": { "x": 0, "y": 0, "width": 0, "height": 0 } | null,
      "ssim_score": 0.82 | null
    },
    "security_thread": {
      "status": "present" | "absent" | "unknown",
      "confidence": 0.78,
      "position": "left-third" | null,
      "coordinates": { "x_start": 0, "x_end": 0 } | null
    },
    "color_analysis": {
      "status": "match" | "mismatch" | "unknown",
      "confidence": 0.88,
      "bhattacharyya_distance": 0.12 | null,
      "dominant_colors": ["#hex1", "#hex2"] | null
    },
    "texture_analysis": {
      "status": "normal" | "abnormal" | "unknown",
      "confidence": 0.91,
      "glcm_contrast": 0.5 | null,
      "glcm_energy": 0.3 | null,
      "sharpness_score": 0.75 | null
    },
    "serial_number": {
      "status": "valid" | "invalid" | "unknown",
      "confidence": 0.65,
      "extracted_text": "ABC1234567" | null,
      "format_valid": true | null
    },
    "dimensions": {
      "status": "correct" | "incorrect" | "unknown",
      "confidence": 0.72,
      "aspect_ratio": 2.17 | null,
      "expected_aspect_ratio": 2.18 | null,
      "deviation_percent": 0.5 | null
    }
  },
  "ensemble_score": 0.89,
  "annotated_image": "data:image/jpeg;base64,...",
  "processing_time_ms": 320,
  "timestamp": "2026-04-18T01:00:00Z"
}
```

---

## 3. Dataset Analysis

The dataset at `d:\Kajol ME\jepa\dataset\archive\data\data\` contains:

| Denomination | Fake Images | Real Images | Total |
|-------------|-------------|-------------|-------|
| ₹10 | 300 | 950 | 1,250 |
| ₹20 | 299 | 948 | 1,247 |
| ₹50 | 299 | 844 | 1,143 |
| ₹100 | 380 | 726 | 1,106 |
| ₹200 | 278 | 563 | 841 |
| ₹500 | 807 | 649 | 1,456 |
| ₹2000 | 145 | 257 | 402 |
| **Total** | **2,508** | **4,937** | **7,445** |

> [!WARNING]
> The dataset is imbalanced (2:1 real:fake). We'll use **weighted sampling** during training and **class-weighted loss** to handle this.

There are also pre-extracted features at `dataset/archive/Features/Features/{denomination}_Features/` which appear to be screenshot images of features from each denomination.

---

## 4. Proposed Changes

### Project Structure

```
d:\Kajol ME\jepa\
├── frontend/                    # Existing React frontend (untouched)
├── dataset/                     # Existing dataset (untouched)
└── backend/                     # [NEW] Python backend
    ├── pyproject.toml           # [NEW] uv project config
    ├── .env                     # [NEW] Environment variables
    ├── .python-version          # [NEW] Python version pin
    ├── alembic.ini              # [NEW] (not used — auto-migration)
    │
    ├── app/                     # [NEW] Main application package
    │   ├── __init__.py
    │   ├── main.py              # FastAPI app entry point
    │   ├── config.py            # Settings & env config
    │   ├── database.py          # MySQL connection & session
    │   │
    │   ├── models/              # SQLAlchemy ORM models
    │   │   ├── __init__.py
    │   │   ├── analysis.py      # AnalysisRecord model
    │   │   └── training.py      # TrainingRun, ModelVersion models
    │   │
    │   ├── schemas/             # Pydantic request/response schemas
    │   │   ├── __init__.py
    │   │   ├── analysis.py      # AnalysisRequest, AnalysisResult, etc.
    │   │   └── training.py      # TrainingConfig, TrainingStatus
    │   │
    │   ├── api/                 # API route handlers
    │   │   ├── __init__.py
    │   │   ├── router.py        # Main API router
    │   │   ├── analyze.py       # POST /analyze, GET /analyze/history
    │   │   └── training.py      # POST /training/start, GET /training/status
    │   │
    │   ├── services/            # Business logic layer
    │   │   ├── __init__.py
    │   │   ├── analyzer.py      # Orchestrates full analysis pipeline
    │   │   ├── image_processor.py # Base64 decode, resize, preprocess
    │   │   ├── annotator.py     # Draws analysis results on image
    │   │   └── denomination.py  # Currency denomination detection
    │   │
    │   ├── ml/                  # Machine learning module
    │   │   ├── __init__.py
    │   │   ├── lejepa/          # LeJEPA implementation
    │   │   │   ├── __init__.py
    │   │   │   ├── encoder.py   # ViT encoder (context + target)
    │   │   │   ├── predictor.py # Narrow ViT predictor
    │   │   │   ├── sigreg.py    # SIGReg regularization
    │   │   │   ├── masking.py   # Multi-block masking strategy
    │   │   │   └── model.py     # Full LeJEPA model assembly
    │   │   │
    │   │   ├── classifier.py    # Classification head + inference
    │   │   ├── dataset.py       # PyTorch Dataset for currency images
    │   │   ├── trainer.py       # Training orchestrator (pretrain + finetune)
    │   │   ├── inference.py     # Production inference engine
    │   │   └── model_registry.py # Model versioning & loading
    │   │
    │   └── features/            # Traditional CV feature analyzers
    │       ├── __init__.py
    │       ├── watermark.py     # Watermark detection (template matching + SSIM)
    │       ├── security_thread.py # Security thread detection
    │       ├── color_analysis.py  # Color histogram analysis
    │       ├── texture_analysis.py # GLCM texture features
    │       ├── serial_number.py   # OCR-based serial number extraction
    │       └── dimensions.py      # Aspect ratio & dimension check
    │
    ├── models/                  # [NEW] Saved model weights directory
    │   └── .gitkeep
    │
    └── scripts/                 # [NEW] Utility scripts
        ├── train.py             # CLI training script
        └── evaluate.py          # Model evaluation script
```

---

### Component Details

---

#### Database Layer

##### [NEW] [.env](file:///d:/Kajol%20ME/jepa/backend/.env)
Environment variables for MySQL connection and app settings.

```env
DATABASE_URL=mysql+aiomysql://root:root@localhost:3306/fake_currency_db
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=root
DB_NAME=fake_currency_db
DATASET_PATH=../dataset/archive/data/data
FEATURES_PATH=../dataset/archive/Features/Features
MODEL_DIR=./models
SECRET_KEY=fake-currency-detection-secret-key-2026
CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
SERVER_PORT=8001
```

##### [NEW] [config.py](file:///d:/Kajol%20ME/jepa/backend/app/config.py)
Pydantic `BaseSettings` class loading from `.env`. Holds all config: DB credentials, model paths, training hyperparameters.

##### [NEW] [database.py](file:///d:/Kajol%20ME/jepa/backend/app/database.py)
- Uses `sqlalchemy[asyncio]` + `aiomysql` for async MySQL
- `create_all_tables()` function called at startup — creates database and all tables if they don't exist
- Uses raw MySQL connector to create the database itself before SQLAlchemy connects
- Provides `get_db()` async dependency for FastAPI

##### MySQL Schema:

```sql
CREATE DATABASE IF NOT EXISTS fake_currency_db
  CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- Analysis history table
CREATE TABLE IF NOT EXISTS analysis_records (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    result          ENUM('REAL', 'FAKE') NOT NULL,
    confidence      FLOAT NOT NULL,
    currency_denomination VARCHAR(10) NULL,
    denomination_confidence FLOAT NULL,
    
    -- CNN classification
    cnn_result      VARCHAR(10) NOT NULL,
    cnn_confidence  FLOAT NOT NULL,
    cnn_model       VARCHAR(50) NOT NULL,
    cnn_time_ms     FLOAT NOT NULL,
    
    -- Watermark
    watermark_status     VARCHAR(20) NOT NULL DEFAULT 'unknown',
    watermark_confidence FLOAT NOT NULL DEFAULT 0.0,
    watermark_location   JSON NULL,
    watermark_ssim       FLOAT NULL,
    
    -- Security Thread
    thread_status        VARCHAR(20) NOT NULL DEFAULT 'unknown',
    thread_confidence    FLOAT NOT NULL DEFAULT 0.0,
    thread_position      VARCHAR(50) NULL,
    thread_coordinates   JSON NULL,
    
    -- Color Analysis
    color_status         VARCHAR(20) NOT NULL DEFAULT 'unknown',
    color_confidence     FLOAT NOT NULL DEFAULT 0.0,
    color_bhattacharyya  FLOAT NULL,
    color_dominant       JSON NULL,
    
    -- Texture Analysis
    texture_status       VARCHAR(20) NOT NULL DEFAULT 'unknown',
    texture_confidence   FLOAT NOT NULL DEFAULT 0.0,
    texture_contrast     FLOAT NULL,
    texture_energy       FLOAT NULL,
    texture_sharpness    FLOAT NULL,
    
    -- Serial Number
    serial_status        VARCHAR(20) NOT NULL DEFAULT 'unknown',
    serial_confidence    FLOAT NOT NULL DEFAULT 0.0,
    serial_text          VARCHAR(50) NULL,
    serial_format_valid  TINYINT(1) NULL,
    
    -- Dimensions
    dim_status           VARCHAR(20) NOT NULL DEFAULT 'unknown',
    dim_confidence       FLOAT NOT NULL DEFAULT 0.0,
    dim_aspect_ratio     FLOAT NULL,
    dim_expected_ratio   FLOAT NULL,
    dim_deviation        FLOAT NULL,
    
    -- Ensemble & Meta
    ensemble_score       FLOAT NOT NULL,
    source               ENUM('upload', 'camera') NOT NULL DEFAULT 'upload',
    original_image       LONGTEXT NOT NULL,     -- Full base64-encoded original image (no quality loss)
    annotated_image      LONGTEXT NOT NULL,     -- Full base64-encoded annotated result image
    thumbnail            MEDIUMTEXT NOT NULL,   -- Base64-encoded 96x96 JPEG thumbnail for history list
    processing_time_ms   INT NOT NULL,
    analyzed_at          DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_result (result),
    INDEX idx_analyzed_at (analyzed_at DESC),
    INDEX idx_denomination (currency_denomination)
) ENGINE=InnoDB;

-- Training runs tracking table
CREATE TABLE IF NOT EXISTS training_runs (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    run_type        ENUM('pretrain', 'finetune', 'full') NOT NULL,
    status          ENUM('pending', 'running', 'completed', 'failed') NOT NULL DEFAULT 'pending',
    config          JSON NOT NULL,
    
    -- Metrics
    best_loss       FLOAT NULL,
    best_accuracy   FLOAT NULL,
    current_epoch   INT NOT NULL DEFAULT 0,
    total_epochs    INT NOT NULL,
    
    -- Timestamps
    started_at      DATETIME NULL,
    completed_at    DATETIME NULL,
    created_at      DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    error_message   TEXT NULL,
    
    INDEX idx_status (status)
) ENGINE=InnoDB;

-- Model versions table
CREATE TABLE IF NOT EXISTS model_versions (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    version         VARCHAR(20) NOT NULL UNIQUE,
    model_path      VARCHAR(255) NOT NULL,
    training_run_id INT NULL,
    
    -- Performance metrics
    accuracy        FLOAT NULL,
    precision_score FLOAT NULL,
    recall_score    FLOAT NULL,
    f1_score        FLOAT NULL,
    
    is_active       TINYINT(1) NOT NULL DEFAULT 0,
    created_at      DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (training_run_id) REFERENCES training_runs(id),
    INDEX idx_active (is_active),
    INDEX idx_version (version)
) ENGINE=InnoDB;
```

---

#### LeJEPA ML Module

##### [NEW] [encoder.py](file:///d:/Kajol%20ME/jepa/backend/app/ml/lejepa/encoder.py)
Vision Transformer (**ViT-Tiny** — optimized for CPU-only training):
- Patch size: 16×16, image size: 224×224
- Embed dim: **192**, depth: **12**, heads: **3**
- Learnable position embeddings with interpolation support
- Outputs per-patch embeddings (no CLS token for JEPA)

##### [NEW] [predictor.py](file:///d:/Kajol%20ME/jepa/backend/app/ml/lejepa/predictor.py)
Narrow ViT predictor:
- Embed dim: **96** (half of encoder), depth: **6**, heads: **3**
- Takes context patch embeddings + target position tokens
- Predicts target patch embeddings

##### [NEW] [sigreg.py](file:///d:/Kajol%20ME/jepa/backend/app/ml/lejepa/sigreg.py)
The core SIGReg (Sketched Isotropic Gaussian Regularization):
- Random projection matrix for sketch-based distribution testing
- Anderson-Darling goodness-of-fit test on projected embeddings
- Enforces isotropic Gaussian distribution on latent space
- ~50 lines of PyTorch code, no external dependencies

```python
# Pseudocode for SIGReg
def sigreg_loss(embeddings, num_projections=128):
    """Enforce isotropic Gaussian on batch embeddings."""
    B, D = embeddings.shape
    # Random projection directions
    proj = torch.randn(D, num_projections, device=embeddings.device)
    proj = F.normalize(proj, dim=0)
    # Project embeddings to 1D
    projected = embeddings @ proj  # (B, num_projections)
    # Standardize
    projected = (projected - projected.mean(0)) / (projected.std(0) + 1e-8)
    # Sort for goodness-of-fit
    projected_sorted, _ = projected.sort(dim=0)
    # Compare to standard Gaussian quantiles
    expected = torch.erfinv(torch.linspace(0.01, 0.99, B, device=embeddings.device)) * math.sqrt(2)
    # L2 distance between sorted projections and expected Gaussian quantiles
    loss = ((projected_sorted - expected.unsqueeze(1)) ** 2).mean()
    return loss
```

##### [NEW] [masking.py](file:///d:/Kajol%20ME/jepa/backend/app/ml/lejepa/masking.py)
Multi-block masking strategy:
- Generates context mask (large, ~85% of patches)
- Generates target masks (4 non-overlapping blocks, ~15-20% each)
- Aspect ratio between 0.75 and 1.5 for target blocks

##### [NEW] [model.py](file:///d:/Kajol%20ME/jepa/backend/app/ml/lejepa/model.py)
Full LeJEPA model assembly:
- `LeJEPAPretrainModel`: encoder + predictor + SIGReg
- `forward()`: masked prediction + regularization loss
- `LeJEPAClassifier`: pre-trained encoder + linear classification head
- Checkpoint save/load utilities

##### [NEW] [classifier.py](file:///d:/Kajol%20ME/jepa/backend/app/ml/classifier.py)
Classification wrapper:
- Attaches a 2-layer MLP head (**192 → 96 → 2**) to frozen/unfrozen encoder
- Global average pooling of patch embeddings → classification
- Denomination classifier: same encoder → separate head (**192 → 96 → 7**)

##### [NEW] [dataset.py](file:///d:/Kajol%20ME/jepa/backend/app/ml/dataset.py)
PyTorch Dataset:
- Loads images from `dataset/archive/data/data/{fake,real}/{denomination}/`
- Augmentations: RandomResizedCrop, HorizontalFlip, ColorJitter, RandomRotation(±10°)
- Normalization: ImageNet mean/std
- Returns `(image_tensor, label, denomination)` tuples
- Weighted sampler for class imbalance

##### [NEW] [trainer.py](file:///d:/Kajol%20ME/jepa/backend/app/ml/trainer.py)
Training orchestrator:
- **Phase 1 — Pre-training**: LeJEPA self-supervised, **~30 epochs** (CPU-optimized), lr=1.5e-4, AdamW, cosine schedule
- **Phase 2 — Fine-tuning**: Classification head + encoder, **~20 epochs** (CPU-optimized), lr=1e-4
- Logs to `training_runs` table in MySQL
- Saves best model checkpoint to `models/` directory
- Creates entry in `model_versions` table
- Runs in background thread via FastAPI `BackgroundTasks`

##### [NEW] [inference.py](file:///d:/Kajol%20ME/jepa/backend/app/ml/inference.py)
Production inference:
- Loads active model version from `model_versions` table
- Caches loaded model in memory (singleton pattern)
- `predict(image_tensor) → (label, confidence)`
- `predict_denomination(image_tensor) → (denomination, confidence)`
- Handles GPU/CPU device selection automatically

##### [NEW] [model_registry.py](file:///d:/Kajol%20ME/jepa/backend/app/ml/model_registry.py)
- `get_active_model()` — returns currently active model path
- `register_model(version, path, metrics)` — registers new model
- `activate_model(version)` — sets a model as active (deactivates others)
- `list_models()` — list all registered model versions

---

#### Traditional CV Feature Analyzers

##### [NEW] [watermark.py](file:///d:/Kajol%20ME/jepa/backend/app/features/watermark.py)
- Template matching using reference watermark images per denomination
- SSIM (Structural Similarity Index) computation against reference
- Returns bounding box location of detected watermark
- Uses OpenCV `matchTemplate` + `minMaxLoc`

##### [NEW] [security_thread.py](file:///d:/Kajol%20ME/jepa/backend/app/features/security_thread.py)
- Edge detection (Canny) in the expected thread region
- Vertical line detection using Hough Transform
- Returns thread position and x-coordinates

##### [NEW] [color_analysis.py](file:///d:/Kajol%20ME/jepa/backend/app/features/color_analysis.py)
- HSV histogram computation and comparison with reference histograms per denomination
- Bhattacharyya distance metric
- K-means dominant color extraction (top 5 colors → hex)

##### [NEW] [texture_analysis.py](file:///d:/Kajol%20ME/jepa/backend/app/features/texture_analysis.py)
- GLCM (Gray-Level Co-occurrence Matrix) feature extraction
- Contrast, energy, homogeneity, correlation
- Laplacian variance for sharpness score

##### [NEW] [serial_number.py](file:///d:/Kajol%20ME/jepa/backend/app/features/serial_number.py)
- ROI extraction from expected serial number region
- EasyOCR for text extraction
- Regex validation for Indian currency serial number format (e.g., `^[0-9][A-Z]{2}[0-9]{6}$`)

##### [NEW] [dimensions.py](file:///d:/Kajol%20ME/jepa/backend/app/features/dimensions.py)
- Contour detection to find note boundaries
- Aspect ratio computation
- Comparison against known denomination aspect ratios (all ~2.18:1 for Indian notes)

---

#### Service Layer

##### [NEW] [analyzer.py](file:///d:/Kajol%20ME/jepa/backend/app/services/analyzer.py)
Main analysis orchestrator:
1. Decode base64 image → OpenCV array
2. Run LeJEPA CNN classification (async)
3. Run all 6 traditional feature analyzers (parallel via `asyncio.gather`)
4. Compute denomination detection
5. Calculate ensemble score: `0.50 * cnn_score + 0.10 * watermark + 0.08 * thread + 0.10 * color + 0.08 * texture + 0.07 * serial + 0.07 * dimensions`
6. Generate annotated image with bounding boxes and labels
7. Store results in MySQL
8. Return `AnalysisResult` response

##### [NEW] [annotator.py](file:///d:/Kajol%20ME/jepa/backend/app/services/annotator.py)
- Draws colored bounding boxes on the currency image for each detected feature
- Uses OpenCV `rectangle`, `putText`
- Green for pass, red for fail, yellow for warning
- Returns base64-encoded annotated image

##### [NEW] [denomination.py](file:///d:/Kajol%20ME/jepa/backend/app/services/denomination.py)
- Uses the encoder's denomination classification head
- Maps prediction to denomination string ("10", "20", "50", "100", "200", "500", "2000")

---

#### API Layer

##### [NEW] [main.py](file:///d:/Kajol%20ME/jepa/backend/app/main.py)
```python
# FastAPI application
# - CORS middleware for frontend (http://localhost:5173)
# - Startup event: create database/tables, load model
# - Include routers under /api/v1
# - Health check endpoint
```

##### [NEW] [router.py](file:///d:/Kajol%20ME/jepa/backend/app/api/router.py)
Groups all route modules under `/api/v1`.

##### [NEW] [analyze.py](file:///d:/Kajol%20ME/jepa/backend/app/api/analyze.py)
- `POST /api/v1/analyze` → `analyze_currency(request: AnalyzeRequest)` (port **8001**)
- `GET /api/v1/analyze/history` → `get_history(page, limit, filter)`
- `GET /api/v1/analyze/history/{id}` → `get_analysis_by_id(id)`
- `DELETE /api/v1/analyze/history/{id}` → `delete_analysis(id)`

##### [NEW] [training.py](file:///d:/Kajol%20ME/jepa/backend/app/api/training.py)
- `POST /api/v1/training/start` → Start training in background (**no password protection**)
- `GET /api/v1/training/status/{run_id}` → Get training progress
- `GET /api/v1/training/models` → List all model versions
- `POST /api/v1/training/models/{version}/activate` → Activate a model version

---

### Dependencies (`pyproject.toml`)

```toml
[project]
name = "fake-currency-backend"
version = "1.0.0"
description = "LeJEPA-based fake Indian currency detection backend"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.34.0",
    "sqlalchemy[asyncio]>=2.0.0",
    "aiomysql>=0.2.0",
    "pymysql>=1.1.0",
    "pydantic>=2.0.0",
    "pydantic-settings>=2.0.0",
    "python-dotenv>=1.0.0",
    "python-multipart>=0.0.9",
    
    # ML / Deep Learning
    "torch>=2.2.0",
    "torchvision>=0.17.0",
    "timm>=1.0.0",
    
    # Computer Vision
    "opencv-python-headless>=4.9.0",
    "Pillow>=10.0.0",
    "scikit-image>=0.22.0",
    "scikit-learn>=1.4.0",
    
    # OCR
    "easyocr>=1.7.0",
    
    # Utilities
    "numpy>=1.26.0",
    "aiofiles>=24.0.0",
]
```

---

## 5. Configuration Decisions (Resolved)

> [!NOTE]
> **GPU**: No CUDA GPU available. Using **ViT-Tiny** (192 embed dim, 12 layers, 3 heads) instead of ViT-Small. Training epochs reduced (30 pretrain + 20 finetune). Estimated CPU training time: ~4-8 hours total.

> [!NOTE]
> **Training Endpoint**: **Open, no password protection** — suitable for development/local use.

> [!NOTE]
> **Port**: Backend runs on **port 8001**. Frontend `api.ts` updated to `http://127.0.0.1:8001/api/v1`.

> [!WARNING]
> **Database**: The backend will auto-create the MySQL database `fake_currency_db` and all tables on startup. Make sure MySQL is running with `root/root` credentials on `localhost:3306`.

> [!IMPORTANT]
> **Model Bootstrap**: On first startup, if no trained model exists, the CNN classifier will return low-confidence dummy results while traditional CV features still work. You'll need to trigger training via the `/api/v1/training/start` endpoint or the CLI script.

---

## 7. Verification Plan

### Automated Tests
1. **Database**: Startup test — verify database and all three tables are created automatically
2. **API Contract**: Send test requests to all 4 endpoints and verify response schemas match frontend types exactly
3. **Model Training**: Run a quick 2-epoch training cycle to verify the full pipeline works
4. **Inference**: Upload a test image and verify full analysis pipeline produces valid results
5. **CORS**: Verify frontend at `localhost:5173` can make requests without CORS errors

### Manual Verification
1. Start backend with `uv run uvicorn app.main:app --host 0.0.0.0 --port 8001`
2. Start frontend with `npm run dev` in the `frontend/` directory
3. Upload a currency image from the dataset and verify the analysis results display correctly
4. Check history page shows the analysis record
5. Trigger model training and monitor progress
