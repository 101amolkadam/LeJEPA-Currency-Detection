# LeJEPA Fake Currency Detection System

An enterprise-grade, full-stack Fake Indian Currency Detection platform powered by a mathematically optimized **LeJEPA (Joint-Embedding Predictive Architecture with Sketched Isotropic Gaussian Regularization)** PyTorch backend seamlessly fused with a modern React.js frontend.

## 🌟 Project Overview

This repository provides a complete classification ecosystem capable of analyzing high-resolution scans of Indian Currency constraints natively utilizing a Vision Transformer. It fundamentally avoids reliance strictly upon conventional pixel methodologies, moving to sophisticated semantic embeddings.

### Core Architecture
- **Backend (API Component)**: Built via `FastAPI` managed natively by `uv`. Connects natively to an auto-migrating asynchronous `MySQL` database containing rigorous persistence logic over image classifications.
- **Machine Learning**: Utilizes `ViT-Tiny` (192-dim) scaling natively against **SIGReg** regularization structures to bypass the notorious 'Representation Collapse' found in basic SSL neural designs.
- **Computer Vision Extraction**: Leverages `OpenCV` logic utilizing Structural Similarity Mapping (SSIM) and Grey Level Co-occurrence Matrices (GLCM) isolating precise texture variables alongside `EasyOCR` parameterizing serialized note vectors instantly.
- **Frontend (Web Interface)**: Constructed upon `React.js` utilizing `Vite`. Interfaces securely over `http://localhost:5173/` pushing Base64 representations directly into the mathematical API constraints.

## 📚 Essential Documentation

The system explicitly maps heavily isolated feature components that are best understood sequentially:

1. **[Setup Guide](setup_guide.md)**: A step-by-step mapping of booting the system, firing up dependencies (`uv sync`, `npm install`), configuring databases, running fallback logic, and interacting manually with the architecture.
2. **[Backend Technical Specs](backend/README.md)**: An explicit breakdown of the structural engineering applied towards the `app/` Python source directory, detailing SIGReg mathematical bounds, deployment layers, and automated PyTorch dataset execution methodologies.
3. **[Implementation Plan Log](implementation_plan.md)**: A transparent record detailing architectural constraints, the reasoning behind explicitly avoiding older video-modelling methodologies like *V-JEPA 2.1* in favor of *LeJEPA*, and the complete feature payload API contract structures.

## 🚀 Quick Start Guide

### 1. Database & Inference Container Initialization
1. Ensure `MySQL` server (`root/root`) is running natively.
2. Enter the backend environment and install all isolated dependencies:
```bash
cd backend
uv sync
```
3. Boot up the asynchronous API loop:
```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8001
```

### 2. Neural Weights Configuration
On immediate boot, the system routes queries through mathematical OpenCV Fallbacks until checkpoint weights are generated natively. Trigger the CPU-bound `ViT` alignment by issuing a bash command:
```bash
uv run python scripts/train.py --mode full --pretrain-epochs 30 --finetune-epochs 20
```
*Note: Due to CPU constraints natively mapping extreme `ColorJitter` visual augmentation across ~7,000 baseline items, lowering parameters to test-run epochs (`--pretrain-epochs 1`) is highly recommended initially.*

### 3. Frontend Invocation
1. Open a strictly detached terminal module.
2. Formulate Node.js distributions and start Vite:
```bash
cd frontend
npm install
npm run dev
```
3. Upload tests utilizing the `test_images/` directory specifically constructed mappings natively at `http://localhost:5173`. 

## 📊 Evaluation & Benchmarking
The application allows native evaluation testing across testing variables isolating F1 metrics sequentially from your fine-tuned ViT checkpoints:
```bash
# In the backend directory
uv run python scripts/evaluate.py
```
Early epoch runs mathematically dictate aggressive Precision scaling, verifying the explicit isolation mapping of robust security anomaly variances derived off the LeJEPA architecture.
