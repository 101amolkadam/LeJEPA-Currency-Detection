# Full-Stack JEPA Setup Guide

This comprehensive step-by-step guide covers how to set up the MySQL Database, build the PyTorch Backend, configure the React Frontend, train the LeJEPA model, and utilize it in production mode against your custom datasets.

---

## Step 1: Database Setup

The backend leverages a MySQL database.

1. Ensure **MySQL Server** is running locally on port `3306`.
2. Ensure you have a root user configured with the following credentials:
   - **Username**: `root`
   - **Password**: `root`
3. *Note: You do NOT need to manually create the database. The FastAPI backend will automatically formulate the `fake_currency_db` schemas and tables perfectly upon startup.*

---

## Step 2: Backend Configuration

The backend utilizes `uv` as its lightning-fast dependency manager.

1. Open a terminal and navigate into the `backend/` folder.
   ```bash
   cd "backend"
   ```
2. Verify that the `.env` settings map exactly to your project structure. Specifically, verify that `DATASET_PATH` corresponds to your raw image paths.
3. Install all deep learning dependencies accurately:
   ```bash
   uv sync
   ```
4. Start the local server to verify functionality:
   ```bash
   uv run uvicorn app.main:app --port 8001
   ```
   *You should see logs indicating "Database ensured" and the server listening.*

---

## Step 3: Frontend Configuration

1. Open a new terminal and navigate to the `frontend/` folder.
   ```bash
   cd "frontend"
   ```
2. Install NodeJS modules:
   ```bash
   npm install
   ```
3. Boot up the Vite developer server:
   ```bash
   npm run dev
   ```
4. Access the web application interface natively at `http://localhost:5173`.

---

## Step 4: Training the LeJEPA Model (Manual Override)

As a zero-shot environment, the backend will initially rely solely on standard OpenCV calculations to determine fake currency (Fallback Mode). To activate the machine learning sequence, you need to initiate training.

The system features **automatic hardware detection**. It will default to GPU (CUDA) if available, or seamlessly fallback to CPU execution if no supported GPU is found.

> [!TIP]
> **GPU Acceleration (CUDA)**
> If you have an NVIDIA GPU, make sure to install PyTorch with CUDA support to drastically speed up training and inference times: `uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`

### Launching Training From the Terminal

1. In the `backend/` directory, run the full pipeline script:
   ```bash
   uv run python scripts/train.py --mode full --pretrain-epochs 30 --finetune-epochs 20
   ```

> [!CAUTION]  
> **Training Time Expectations on CPU:**  
> With 7,445 images in testing distributions, expect the full 50 epochs to take **4 to 8 hours** entirely contingent on CPU cores.  
> You can lower the epochs temporarily if you wish to run a quick test (e.g., `--pretrain-epochs 2 --finetune-epochs 2`).
> Alternatively, if you wish to modify learning rates, batch sizes, or model sizing, simply edit `backend/app/config.py`!

2. The PyTorch script will:
   - Construct robust visual representations of the notes via native SIGReg pre-training.
   - Construct the classification engine leveraging those representations.
   - Package and automatically save a `lejepa_classifier_XXXXXX.pth` artifact into your native `backend/models` directory.

Once trained, the FastAPI Inference Engine auto-detects this system weight file and integrates it natively into API interactions!

---

## Step 5: Testing on Images

There are two primary directives for testing explicit image datasets:

### Method A: Testing via the Web Interface (Recommended)
1. Open up `http://localhost:5173`.
2. Click **Upload** and select any currency picture from your `test_images` folder mapping.
3. The interface will display your classification score derived instantly from the PyTorch model running inside the backend container alongside dynamic bounding box mappings over the security threads and detected watermarks!

### Method B: Testing via API Script (For batch processing)
If you just want to run an entire folder through the LeJEPA detector from the command line without utilizing a browser, you can query the API directly via python:

Create a file `test_batch.py` inside your root directory mapping:
```python
import os
import base64
import requests

test_dir = "./test_images"

for file in os.listdir(test_dir):
    if not file.endswith((".jpg", ".png")): continue
    path = os.path.join(test_dir, file)
    
    with open(path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        
    res = requests.post("http://127.0.0.1:8001/api/v1/analyze", json={
        "image": f"data:image/jpeg;base64,{encoded_string}",
        "source": "upload"
    }).json()
    
    print(f"{file} -> {res['result']} (Confidence: {res['confidence']*100:.1f}%)")
```
Run `python test_batch.py` to see the exact neural network deductions immediately!
