# Fake Currency Detection — LeJEPA Backend

This repository contains the backend for the Fake Indian Currency Detection system. It is powered by a custom implementation of **LeJEPA** (Joint-Embedding Predictive Architecture with Sketched Isotropic Gaussian Regularization) combined with traditional computer vision feature analyzers.

## Architecture

*   **FastAPI**: High-performance asynchronous API server.
*   **MySQL & SQLAlchemy**: Persistent storage for analysis records and training logs.
*   **PyTorch**: Core machine learning framework.
*   **LeJEPA (ViT-Tiny)**: State-of-the-art self-supervised vision transformer architecture.
*   **OpenCV & EasyOCR**: Extract highly detailed security features (watermarks, security threads, UV colors, texture GLCM features, and precise serial number scanning).

## Which JEPA is Implemented?

This system implements **LeJEPA**. 
LeJEPA is an architectural enhancement over standard I-JEPA that utilizes **SIGReg** (Sketched Isotropic Gaussian Regularization). Unlike older methods that require complex, unstable heuristical tricks (like EMA teacher networks and stop-gradients), our implementation natively enforces an isotropic Gaussian structure natively within the PyTorch representation space. 

This ensures that the encoder won’t suffer from "representation collapse" when fine-tuned rapidly on a limited dataset of currency constraints.

## Project Structure

```text
backend/
├── app/
│   ├── api/          # API route definitions (/analyze, /training)
│   ├── features/     # OpenCV feature detection engines (OpenCV, EasyOCR)
│   ├── ml/           # Core Machine Learning Subsystem
│   │   ├── lejepa/   # LeJEPA logic (ViT Encoder, Predictor, SIGReg)
│   │   ├── classifier.py
│   │   ├── script.py
│   │   └── ...
│   ├── models/       # Database SQLAlchemy models
│   ├── schemas/      # Pydantic JSON contracts (matches frontend)
│   ├── services/     # Orchestrators and Image mapping
│   ├── config.py     # Environment configurations
│   ├── database.py   # Database connection logic
│   └── main.py       # FastAPI application entry
├── models/           # Folder where .pth files will get saved
├── scripts/
│   ├── train.py      # Core CLI training script
│   └── evaluate.py   # CLI evaluation module
├── pyproject.toml    # `uv` dependencies manifest
└── .env              # Configuration variables
```

## How to Train

You can natively invoke the model training sequence without the frontend by hooking into the built-in python CLI. Ensure you are in the `backend/` directory.

### 1. Fully Automated Training (Pretrain + Finetune)
```bash
uv run python scripts/train.py --mode full --pretrain-epochs 30 --finetune-epochs 20
```

### 2. Manual Phased Training
```bash
# Phase 1: SSL Pretraining only
uv run python scripts/train.py --mode pretrain --pretrain-epochs 30

# Phase 2: Supervised fine-tuning using pretrained weights
uv run python scripts/train.py --mode finetune --finetune-epochs 20
```

*Note: Once training completes, the resulting `.pth` checkpoint is automatically saved to your `models/` directory, and if the API server is currently running, it will automatically load the new model into active memory.*
